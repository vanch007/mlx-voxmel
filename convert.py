"""Convert PyTorch VoxMel-0.6B / VoxMel-0.3B & Vocos checkpoints to pure MLX format (bf16, 8bit, 4bit)."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

import mlx_lm.models.qwen2 as qwen2
from models import VoxMelModel
from vocos import Vocos


def pt_to_mlx_key(k: str) -> str:
    if k == "hist_mask_emb":
        return None

    # LLM weights: llm.model.model.* -> llm.model.*
    if k.startswith("llm.model.model."):
        return "llm.model." + k[len("llm.model.model.") :]

    # Attention in/out proj:
    if ".attn.in_proj_weight" in k:
        k = k.replace(".attn.in_proj_weight", ".in_proj.weight")
    elif ".attn.in_proj_bias" in k:
        k = k.replace(".attn.in_proj_bias", ".in_proj.bias")
    elif ".attn.out_proj.weight" in k:
        k = k.replace(".attn.out_proj.weight", ".out_proj.weight")
    elif ".attn.out_proj.bias" in k:
        k = k.replace(".attn.out_proj.bias", ".out_proj.bias")

    # Flow time_mlp: .time_mlp.{0,2} -> .time_mlp.layers.{0,2}
    k = re.sub(r'(\.time_mlp)\.(\d+)', r'\1.layers.\2', k)

    # Flow out: .out.{0,1} -> .out.layers.{0,1}
    k = re.sub(r'(\.flow\.out)\.(\d+)', r'\1.layers.\2', k)
    if k.startswith("coarse_flow.out."):
        k = re.sub(r'^coarse_flow\.out\.(\d+)', r'coarse_flow.out.layers.\1', k)

    # Flow blocks: .blocks.{i}.{j} -> .blocks.{i}.layers.{j}
    k = re.sub(r'(\.flow\.blocks\.\d+)\.(\d+)', r'\1.layers.\2', k)
    if k.startswith("coarse_flow.blocks."):
        k = re.sub(r'^coarse_flow\.blocks\.(\d+)\.(\d+)', r'coarse_flow.blocks.\1.layers.\2', k)

    # MLP in BidirAttention: up_tiers.{i}.blocks.{j}.mlp.{0,2} -> up_tiers.{i}.blocks.{j}.mlp.layers.{0,2}
    k = re.sub(r'(\.mlp)\.(\d+)', r'\1.layers.\2', k)

    return k


def convert_weights(pt_sd: dict) -> dict[str, mx.array]:
    """Map PyTorch weight keys to MLX VoxMelModel weight keys."""
    if "ema" in pt_sd:
        sd = pt_sd["ema"]
    elif "state_dict" in pt_sd:
        sd = pt_sd["state_dict"]
    else:
        sd = pt_sd

    mlx_dict: dict[str, mx.array] = {}

    for k, v in sd.items():
        mapped_k = pt_to_mlx_key(k)
        if mapped_k is None:
            continue

        arr = v.cpu().float().numpy()
        mlx_dict[mapped_k] = mx.array(arr)

    return mlx_dict


def convert_vocos_weights(pt_sd: dict) -> dict[str, mx.array]:
    """Map PyTorch Vocos weights to MLX Vocos weights."""
    mlx_weights = {}
    for k, v in pt_sd.items():
        if "feature_extractor.mel_spec.spectrogram.window" in k or "head.istft.window" in k:
            continue
        basename, pname = k.rsplit(".", 1)
        arr = mx.array(v.cpu().float().numpy())
        if "backbone.embed" in basename and pname == "weight":
            arr = arr.moveaxis(1, 2)
        elif "dwconv" in basename and pname == "weight":
            arr = arr.moveaxis(1, 2)
        mlx_weights[k] = arr
    return mlx_weights


def build_model_from_config(config_yaml_path: str | Path, qwen_cfg_path: str | Path) -> VoxMelModel:
    cfg = OmegaConf.load(str(config_yaml_path))
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
    return model


def main():
    parser = argparse.ArgumentParser(description="Convert VoxMel PyTorch weights to MLX format.")
    parser.add_argument("--repo-root", default="/Users/vanch/VoxMel-repo")
    parser.add_argument("--output-root", default="/Users/vanch/mlx-voxmel/models")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Convert Vocos
    print("=== Converting Vocos ===")
    vocos_cache = Path("/Users/vanch/.cache/huggingface/hub/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21")
    vocos_out = output_root / "vocos-mel-24khz"
    vocos_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vocos_cache / "config.yaml", vocos_out / "config.yaml")

    vocos_pt = torch.load(vocos_cache / "pytorch_model.bin", map_location="cpu", weights_only=True)
    vocos_mlx = convert_vocos_weights(vocos_pt)
    mx.save_safetensors(str(vocos_out / "weights.safetensors"), vocos_mlx)
    print(f"Saved MLX Vocos weights to {vocos_out / 'weights.safetensors'}")

    # Models definition
    models_info = [
        {
            "name": "VoxMel-0.3B",
            "ckpt": "/Users/vanch/.cache/huggingface/hub/models--VoxDance-Lab--VoxMel-0.3B/snapshots/d2aa83a5b1a8f2a9134ada3853e1093fd7b546fc/model_M_final.pt",
            "config": "/Users/vanch/.cache/huggingface/hub/models--VoxDance-Lab--VoxMel-0.3B/snapshots/d2aa83a5b1a8f2a9134ada3853e1093fd7b546fc/config.yaml",
            "qwen_config": repo_root / "examples/voxmel/qwen_configs/qwen2_medium_300m/config.json",
        },
        {
            "name": "VoxMel-0.6B",
            "ckpt": "/Users/vanch/.cache/huggingface/hub/models--VoxDance-Lab--VoxMel-0.6B/snapshots/0c09bf5331a921dae83221c9e4a0cf4b555d65f6/model_L_final.pt",
            "config": "/Users/vanch/.cache/huggingface/hub/models--VoxDance-Lab--VoxMel-0.6B/snapshots/0c09bf5331a921dae83221c9e4a0cf4b555d65f6/config.yaml",
            "qwen_config": repo_root / "examples/voxmel/qwen_tokenizer/config.json",
        },
    ]

    for minfo in models_info:
        mname = minfo["name"]
        print("==========================================")
        print(f"Processing {mname}...")
        print("==========================================")

        pt_sd = torch.load(minfo["ckpt"], map_location="cpu", weights_only=False)
        mlx_weights = convert_weights(pt_sd)

        # Build base model and verify keys
        model = build_model_from_config(minfo["config"], minfo["qwen_config"])
        model.load_weights(list(mlx_weights.items()), strict=True)
        print(f"Loaded {len(mlx_weights)} weights with strict=True successfully!")

        # Copy tokenizer assets
        tokenizer_src = repo_root / "examples/voxmel/qwen_tokenizer"

        # Generate bf16, 8bit, 4bit variants
        for variant in ["bf16", "8bit", "4bit"]:
            out_dir = output_root / f"{mname}-{variant}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Creating {mname}-{variant} at {out_dir}...")

            # Copy configs and tokenizer
            shutil.copy2(minfo["config"], out_dir / "config.yaml")
            shutil.copy2(minfo["qwen_config"], out_dir / "qwen_config.json")
            shutil.copy2(Path(__file__).parent / "mel_filters_100.npy", out_dir / "mel_filters_100.npy")

            tok_dir = out_dir / "tokenizer"
            tok_dir.mkdir(parents=True, exist_ok=True)
            for f in tokenizer_src.glob("*"):
                if f.is_file():
                    shutil.copy2(f, tok_dir / f.name)

            # Fresh model instance
            variant_model = build_model_from_config(minfo["config"], minfo["qwen_config"])
            variant_model.load_weights(list(mlx_weights.items()), strict=True)

            manifest = {
                "model_name": mname,
                "variant": variant,
                "sample_rate": 24000,
                "architecture": "VoxMel",
            }

            if variant == "bf16":
                bf16_weights = {}
                for k, v in mlx_weights.items():
                    if any(buf in k for buf in ["mean", "std", "var", "special_embedding"]):
                        bf16_weights[k] = v
                    else:
                        bf16_weights[k] = v.astype(mx.bfloat16)
                mx.save_safetensors(str(out_dir / "weights.safetensors"), bf16_weights)
                manifest["quantization"] = "bf16"
            else:
                bits = 8 if variant == "8bit" else 4
                group_size = 64
                print(f"  Quantizing linear layers to {bits}-bit (group_size={group_size})...")
                nn.quantize(
                    variant_model,
                    group_size=group_size,
                    bits=bits,
                    class_predicate=lambda path, m: isinstance(m, nn.Linear) and not path.endswith("stop_token_head") and m.weight.shape[-1] % group_size == 0,
                )
                variant_model.save_weights(str(out_dir / "weights.safetensors"))
                manifest["quantization"] = {
                    "bits": bits,
                    "group_size": group_size,
                    "format": "mlx-nn-quantize",
                }

            with open(out_dir / "model_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            print(f"  Done {mname}-{variant}!")

    print("All weights and variants converted successfully!")


if __name__ == "__main__":
    main()

