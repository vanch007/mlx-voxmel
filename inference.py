"""User-facing zero-shot voice cloning inference API in pure MLX."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf
import yaml
from omegaconf import OmegaConf

import mlx_lm.models.qwen2 as qwen2
from audio import extract_mel, load_audio
from models import VoxMelModel
from tokenizer import QwenTokenizer
from vocos import Vocos


class VoxMel:
    def __init__(self, model: VoxMelModel, tokenizer: QwenTokenizer, vocos: Vocos, sample_rate: int = 24000):
        self.model = model
        self.tokenizer = tokenizer
        self.vocos = vocos
        self.sample_rate = sample_rate

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        vocos_path: Optional[str | Path] = None,
    ) -> VoxMel:
        root = Path(model_path).expanduser().resolve()
        if not (root / "weights.safetensors").is_file():
            raise FileNotFoundError(f"Missing weights.safetensors in {root}")

        manifest_file = root / "model_manifest.json"
        manifest = {}
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

        config_path = root / "config.yaml"
        qwen_cfg_path = root / "qwen_config.json"
        cfg = OmegaConf.load(str(config_path))
        mc = cfg.model_conf

        with open(qwen_cfg_path, "r", encoding="utf-8") as f:
            qwen_json = json.load(f)
        llm_args = qwen2.ModelArgs.from_dict(qwen_json)

        tier_attn_window = getattr(mc, "tier_attn_window", None)
        if tier_attn_window is not None:
            tier_attn_window = OmegaConf.to_container(tier_attn_window)

        model = VoxMelModel(
            n_mels=int(mc.n_mels),
            llm_input_size=int(cfg.llm_input_size),
            llm_output_size=int(cfg.llm_output_size),
            llm_args=llm_args,
            dim=int(mc.dim),
            num_heads=int(mc.num_heads),
            n_layers=tuple(int(x) for x in mc.n_layers),
            split_axes=tuple(str(x) for x in mc.split_axes),
            flow_hidden=int(mc.flow_hidden),
            flow_depth=int(mc.flow_depth),
            num_sampling_steps=int(mc.num_sampling_steps),
            sample_rate=int(cfg.sample_rate),
            hop_length=int(mc.hop_length),
            cfg_text_strength=float(mc.cfg_text_strength),
            cfg_spk_strength=float(mc.cfg_spk_strength),
            detail_norm=bool(mc.detail_norm),
            tier_attn_window=tier_attn_window,
        )

        quant_info = manifest.get("quantization")
        if isinstance(quant_info, dict) and quant_info.get("format") == "mlx-nn-quantize":
            bits = quant_info["bits"]
            group_size = quant_info["group_size"]
            nn.quantize(
                model,
                group_size=group_size,
                bits=bits,
                class_predicate=lambda path, m: isinstance(m, nn.Linear) and not path.endswith("stop_token_head") and m.weight.shape[-1] % group_size == 0,
            )

        model.load_weights(str(root / "weights.safetensors"), strict=True)
        mx.eval(model.parameters())
        model.eval()

        tok_dir = root / "tokenizer"
        tokenizer = QwenTokenizer(tok_dir, skip_special_tokens=True)

        if vocos_path is None:
            # Check default locations
            default_vocos = root.parent / "vocos-mel-24khz"
            if default_vocos.exists():
                vocos_path = default_vocos
            else:
                vocos_path = Path(__file__).resolve().parent / "models" / "vocos-mel-24khz"

        vocos_dir = Path(vocos_path).expanduser().resolve()
        with open(vocos_dir / "config.yaml", "r", encoding="utf-8") as f:
            vocos_cfg = yaml.safe_load(f)
        vocos = Vocos.from_hparams(vocos_cfg)
        vocos.load_weights(str(vocos_dir / "weights.safetensors"), strict=False)
        mx.eval(vocos.parameters())
        vocos.eval()

        return cls(model=model, tokenizer=tokenizer, vocos=vocos, sample_rate=int(cfg.sample_rate))

    def generate(
        self,
        text: str,
        *,
        prompt_wav_path: str | Path,
        prompt_text: str,
        seed: Optional[int] = None,
        inference_timesteps: Optional[int] = None,
        up_num_steps: Optional[int] = None,
        cfg_text_strength: Optional[float] = None,
        cfg_spk_strength: Optional[float] = None,
        min_seconds: float = 0.3,
        max_seconds: float = 30.0,
    ) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Target text must be non-empty")
        if not prompt_wav_path or not prompt_text or not prompt_text.strip():
            raise ValueError("prompt_wav_path and prompt_text must be provided together and non-empty")

        if seed is not None:
            mx.random.seed(seed)
            np.random.seed(seed)

        wav = load_audio(prompt_wav_path, target_sr=self.sample_rate)
        prompt_mel = extract_mel(wav, sample_rate=self.sample_rate, hop_length=self.model.hop_length, n_mels=self.model.n_mels)

        prompt_tokens = self.tokenizer.encode(prompt_text)

        target_text = text.lstrip()
        if target_text and not ("一" <= target_text[0] <= "鿿"):
            text = " " + text
        target_tokens = self.tokenizer.encode(text)

        fps = self.sample_rate / self.model.hop_length / (2 ** self.model.num_time_splits)
        min_frames = max(1, math.ceil(min_seconds * fps))
        max_frames = max(1, math.ceil(max_seconds * fps))

        mel = self.model.inference_with_prompt(
            prompt_mel=prompt_mel,
            prompt_text_tokens=prompt_tokens,
            target_text_tokens=target_tokens,
            max_frames=max_frames,
            min_frames=min_frames,
            num_steps=inference_timesteps,
            up_num_steps=up_num_steps,
            cfg_text_strength=cfg_text_strength,
            cfg_spk_strength=cfg_spk_strength,
        )

        # Decode via Vocos: mel is (1, T, 100) -> transpose to (1, 100, T)
        mel_t = mx.transpose(mel, (0, 2, 1))
        audio = self.vocos.decode(mel_t)
        mx.eval(audio)

        return np.asarray(audio, dtype=np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MLX VoxMel TTS Inference CLI")
    parser.add_argument("--model", required=True, help="Path to converted MLX VoxMel model dir")
    parser.add_argument("--prompt-audio", required=True, help="Path to prompt audio wav")
    parser.add_argument("--prompt-text", required=True, help="Transcript of prompt audio")
    parser.add_argument("--text", required=True, help="Target text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output wav path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=32, help="Coarse solver sampling steps")
    parser.add_argument("--up-steps", type=int, default=32, help="Residual solver sampling steps")
    parser.add_argument("--cfg-text", type=float, default=3.5, help="CFG text guidance strength")
    parser.add_argument("--cfg-spk", type=float, default=3.0, help="CFG speaker guidance strength")
    args = parser.parse_args()

    print(f"Loading MLX VoxMel from {args.model}...")
    voxmel = VoxMel.from_pretrained(args.model)

    print(f"Synthesizing: '{args.text}'...")
    audio = voxmel.generate(
        args.text,
        prompt_wav_path=args.prompt_audio,
        prompt_text=args.prompt_text,
        seed=args.seed,
        inference_timesteps=args.steps,
        up_num_steps=args.up_steps,
        cfg_text_strength=args.cfg_text,
        cfg_spk_strength=args.cfg_spk,
    )

    sf.write(args.output, audio, voxmel.sample_rate)
    duration = len(audio) / voxmel.sample_rate
    print(f"Audio saved to {args.output} ({duration:.2f}s, {voxmel.sample_rate}Hz)")


if __name__ == "__main__":
    main()
