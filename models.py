
"""Pure MLX VoxMel architecture:
- Decoder-only Qwen2 LLM backbone
- Continuous FlowMatchingHead for Coarse-Mel
- UpsampleTier bidirectional attention + FlowMatchingHead
- Haar decomposition & reconstruction
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from haar import even_odd, split, interleave, pyramid, tier_dims


def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = mx.expand_dims(t.astype(mx.float32), 1) * mx.expand_dims(freqs, 0)
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
    return emb


def sinusoidal_pos_emb(T: int, dim: int) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
    t = mx.arange(T, dtype=mx.float32)
    args = mx.expand_dims(t, 1) * mx.expand_dims(freqs, 0)
    emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((T, 1), dtype=mx.float32)], axis=-1)
    return mx.expand_dims(emb, 0)


def apply_cfg(v_cond: mx.array, v_uncond: mx.array, cfg_strength: float) -> mx.array:
    return v_cond + cfg_strength * (v_cond - v_uncond)


def apply_cfg_two_term(v_full: mx.array, v_dropT: mx.array, v_dropTS: mx.array, s_text: float, s_spk: float) -> mx.array:
    return v_dropTS + s_text * (v_full - v_dropT) + s_spk * (v_dropT - v_dropTS)


class FlowMatchingHead(nn.Module):
    def __init__(self, data_dim: int, cond_dim: int, hidden: int = 512, depth: int = 3, time_embed_dim: int = 128):
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.in_proj = nn.Linear(data_dim, hidden)

        self.blocks = [
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(depth)
        ]
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, data_dim),
        )

    def velocity(
        self,
        phi: mx.array,
        t: mx.array,
        cond: mx.array,
        cond_hidden: Optional[mx.array] = None,
        time_hidden: Optional[mx.array] = None,
    ) -> mx.array:
        if cond_hidden is None:
            cond_hidden = self.cond_proj(cond)
        if time_hidden is None:
            time_hidden = self.time_mlp(timestep_embedding(t, self.time_embed_dim))
        h = self.in_proj(phi) + cond_hidden + time_hidden
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)

    def sample(
        self,
        cond: mx.array,
        num_steps: int = 16,
        cond_uncond: Optional[mx.array] = None,
        cfg_strength: float = 0.0,
        cond_dropT: Optional[mx.array] = None,
        cond_dropTS: Optional[mx.array] = None,
        s_text: float = 0.0,
        s_spk: float = 0.0,
    ) -> mx.array:
        N = cond.shape[0]
        x = mx.random.normal((N, self.data_dim))
        dt = 1.0 / num_steps

        legacy = cond_uncond is not None and abs(cfg_strength) > 1e-6
        three = (
            cond_dropT is not None
            and cond_dropTS is not None
            and (abs(s_text) > 1e-6 or abs(s_spk) > 1e-6)
        )
        batch_size = N * (3 if three else 2 if legacy else 1)
        if three:
            cond = mx.concatenate([cond, cond_dropT, cond_dropTS], axis=0)
        elif legacy:
            cond = mx.concatenate([cond, cond_uncond], axis=0)

        cond_hidden = self.cond_proj(cond)

        time_hiddens = [
            self.time_mlp(
                timestep_embedding(
                    mx.full((batch_size,), i * dt, dtype=mx.float32),
                    self.time_embed_dim,
                )
            )
            for i in range(num_steps)
        ]

        for i in range(num_steps):
            ti = i * dt
            t = mx.full((N,), ti, dtype=mx.float32)
            time_hidden = time_hiddens[i]

            if three:
                x3 = mx.concatenate([x, x, x], axis=0)
                v3 = self.velocity(x3, t, cond, cond_hidden, time_hidden)
                v_full, v_dropT, v_dropTS = mx.split(v3, 3, axis=0)
                v = apply_cfg_two_term(v_full, v_dropT, v_dropTS, s_text, s_spk)
            elif legacy:
                x2 = mx.concatenate([x, x], axis=0)
                v2 = self.velocity(x2, t, cond, cond_hidden, time_hidden)
                v_cond, v_uncond = mx.split(v2, 2, axis=0)
                v = apply_cfg(v_cond, v_uncond, cfg_strength)
            else:
                v = self.velocity(x, t, cond, cond_hidden, time_hidden)
            x = x + dt * v
        return x


class BidirAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def __call__(self, x: mx.array, attn_mask: Optional[mx.array] = None) -> mx.array:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.in_proj(h)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = mx.transpose(mx.reshape(q, (B, T, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = mx.transpose(mx.reshape(k, (B, T, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = mx.transpose(mx.reshape(v, (B, T, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        if attn_mask is not None:
            scores = scores + mx.where(attn_mask[None, None, ...], -1e9, 0.0)

        attn = mx.softmax(scores, axis=-1)
        context = attn @ v
        context = mx.reshape(mx.transpose(context, (0, 2, 1, 3)), (B, T, C))
        a = self.out_proj(context)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class UpsampleTier(nn.Module):
    def __init__(
        self,
        n_mels: int,
        dim: int,
        num_heads: int,
        n_layers: int,
        flow_hidden: int = 512,
        flow_depth: int = 3,
        detail_norm: bool = False,
        attn_window: int = -1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.dim = dim
        self.attn_window = int(attn_window)

        self.in_proj = nn.Linear(n_mels, dim)
        self.blocks = [BidirAttention(dim, num_heads) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(dim)
        self.flow = FlowMatchingHead(n_mels, dim, hidden=flow_hidden, depth=flow_depth)

        self.detail_norm = detail_norm
        self.detail_mean = mx.zeros((n_mels,))
        self.detail_var = mx.ones((n_mels,))

    def _denorm_detail(self, d: mx.array) -> mx.array:
        if not self.detail_norm:
            return d
        var = mx.clip(self.detail_var, 1e-8, None)
        return d * mx.sqrt(var) + self.detail_mean

    def _window_attn_mask(self, T: int) -> Optional[mx.array]:
        if self.attn_window < 0:
            return None
        idx = mx.arange(T)
        return mx.abs(mx.expand_dims(idx, 0) - mx.expand_dims(idx, 1)) > self.attn_window

    def _context(self, x: mx.array) -> mx.array:
        B, T, M = x.shape
        h = self.in_proj(x) + sinusoidal_pos_emb(T, self.dim)
        mask = self._window_attn_mask(T)
        for blk in self.blocks:
            h = blk(h, attn_mask=mask)
        return self.norm(h)

    def sample(self, x_known: mx.array, num_steps: int = 16) -> mx.array:
        B, T, M = x_known.shape
        cond = mx.reshape(self._context(x_known), (B * T, self.dim))
        y = self.flow.sample(cond, num_steps=num_steps)
        y = self._denorm_detail(y)
        return mx.reshape(y, (B, T, M))


class VoxMelModel(nn.Module):
    def __init__(
        self,
        n_mels: int,
        llm_input_size: int,
        llm_output_size: int,
        llm_args,
        dim: int = 512,
        num_heads: int = 8,
        n_layers: tuple = (6, 4, 4),
        split_axes: tuple = ("t", "f"),
        flow_hidden: int = 512,
        flow_depth: int = 3,
        num_sampling_steps: int = 16,
        sample_rate: int = 24000,
        hop_length: int = 256,
        cfg_text_strength: float = 3.5,
        cfg_spk_strength: float = 3.0,
        detail_norm: bool = True,
        tier_attn_window: Optional[dict] = None,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.dim = dim
        self.n_tiers = len(n_layers)
        self.num_sampling_steps = num_sampling_steps
        self.cfg_text_strength = float(cfg_text_strength)
        self.cfg_spk_strength = float(cfg_spk_strength)

        self.sos_eos = 0
        self.task_id = 1
        self.special_embedding = nn.Embedding(50, llm_input_size)
        self.null_text = mx.zeros((1, llm_input_size))

        self.split_axes = tuple(split_axes)
        self.num_time_splits = sum(a == "t" for a in split_axes)
        self.num_freq_splits = sum(a == "f" for a in split_axes)

        coarse_M, up_dims = tier_dims(n_mels, split_axes)
        self.coarse_M = coarse_M

        self.coarse_frame_proj = nn.Linear(coarse_M, llm_input_size)
        
        from mlx_lm.models import qwen2
        self.llm = qwen2.Model(llm_args)

        self.coarse_flow = FlowMatchingHead(
            coarse_M, cond_dim=llm_output_size, hidden=flow_hidden, depth=flow_depth
        )
        self.stop_token_head = nn.Linear(llm_output_size, 1, bias=False)

        self.detail_norm = bool(detail_norm)
        if tier_attn_window is None:
            _tier_windows = [-1] * len(self.split_axes)
        elif isinstance(tier_attn_window, (int, float)):
            _tier_windows = [int(tier_attn_window)] * len(self.split_axes)
        else:
            _tier_windows = [int(tier_attn_window.get(a, -1)) for a in self.split_axes]
        self.tier_attn_window = _tier_windows

        self.up_tiers = [
            UpsampleTier(
                d, dim, num_heads, nl, flow_hidden, flow_depth,
                detail_norm=self.detail_norm, attn_window=aw
            )
            for nl, d, aw in zip(n_layers[1:], up_dims, _tier_windows)
        ]

        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.mel_mean = mx.zeros((n_mels,))
        self.mel_std = mx.ones((n_mels,))

    def normalize_mel(self, mel: mx.array) -> mx.array:
        return (mel - self.mel_mean) / self.mel_std

    def denormalize_mel(self, mel: mx.array) -> mx.array:
        return mel * self.mel_std + self.mel_mean

    def prompt_tier_grids(self, prompt_mel: mx.array):
        p_coarse, details = pyramid(prompt_mel, self.split_axes)
        p_known_list = [known for (_axis, known, _target) in details]
        return p_coarse, p_known_list

    def inference_with_prompt(
        self,
        prompt_mel: mx.array,
        prompt_text_tokens: list[int] | mx.array,
        target_text_tokens: list[int] | mx.array,
        max_frames: int = 1000,
        min_frames: int = 1,
        num_steps: Optional[int] = None,
        up_num_steps: Optional[int] = None,
        cfg_text_strength: Optional[float] = None,
        cfg_spk_strength: Optional[float] = None,
    ) -> mx.array:
        from mlx_lm.models.cache import make_prompt_cache

        steps = num_steps or self.num_sampling_steps
        up_steps = up_num_steps or steps

        s_text = self.cfg_text_strength if cfg_text_strength is None else float(cfg_text_strength)
        s_spk = self.cfg_spk_strength if cfg_spk_strength is None else float(cfg_spk_strength)
        use_text_cfg = abs(s_text) > 1e-6
        use_spk_cfg = abs(s_spk) > 1e-6

        factor = 2 ** self.num_time_splits
        rem = prompt_mel.shape[1] % factor
        if factor > 1 and rem:
            prompt_mel = prompt_mel[:, :-rem, :]

        prompt_mel = self.normalize_mel(prompt_mel)
        p_coarse, p_known_list = self.prompt_tier_grids(prompt_mel)
        T0_p = p_coarse.shape[1]
        assert T0_p >= 1, "prompt too short: < 1 coarse frame"

        if not isinstance(prompt_text_tokens, mx.array):
            prompt_text_tokens = mx.array(prompt_text_tokens, dtype=mx.int32)[None, :]
        if not isinstance(target_text_tokens, mx.array):
            target_text_tokens = mx.array(target_text_tokens, dtype=mx.int32)[None, :]

        text_token = mx.concatenate([prompt_text_tokens, target_text_tokens], axis=1)
        text_emb = self.llm.model.embed_tokens(text_token)
        L = text_emb.shape[1]

        sos = mx.reshape(self.special_embedding.weight[self.sos_eos], (1, 1, -1))
        task = mx.reshape(self.special_embedding.weight[self.task_id], (1, 1, -1))
        p_frame_emb = self.coarse_frame_proj(p_coarse)

        prime_full = mx.concatenate([sos, text_emb, task, p_frame_emb], axis=1)

        null_txt = mx.broadcast_to(mx.reshape(self.null_text, (1, 1, -1)), (1, L, self.llm_input_size))
        zero_prompt = mx.zeros_like(p_frame_emb)

        prime_p = mx.concatenate([sos, null_txt, task, p_frame_emb], axis=1)
        prime_b = mx.concatenate([sos, null_txt, task, zero_prompt], axis=1)
        use_p = use_text_cfg or use_spk_cfg
        use_b = use_spk_cfg

        cache_f = make_prompt_cache(self.llm.model)
        cache_p = make_prompt_cache(self.llm.model) if use_p else None
        cache_b = make_prompt_cache(self.llm.model) if use_b else None

        cur_f, cur_p, cur_b = prime_full, prime_p, prime_b
        frames = []

        def _coarse_frame(last_f, last_p, last_b):
            cond_dropT = last_p if use_p else None
            cond_dropTS = last_b if use_b else (last_p if use_p else None)
            return self.coarse_flow.sample(
                last_f,
                num_steps=steps,
                cond_dropT=cond_dropT,
                cond_dropTS=cond_dropTS,
                s_text=s_text,
                s_spk=s_spk,
            )

        for tstep in range(max_frames):
            h_f = self.llm.model(inputs=None, cache=cache_f, input_embeddings=cur_f)
            if use_p:
                h_p = self.llm.model(inputs=None, cache=cache_p, input_embeddings=cur_p)
            if use_b:
                h_b = self.llm.model(inputs=None, cache=cache_b, input_embeddings=cur_b)

            last_f = h_f[:, -1]
            last_p = h_p[:, -1] if use_p else None
            last_b = h_b[:, -1] if use_b else None

            stop = mx.sigmoid(self.stop_token_head(last_f)) > 0.5
            if bool(stop.item()) and tstep > min_frames:
                break

            frame = _coarse_frame(last_f, last_p, last_b)
            frames.append(frame)

            nxt = mx.expand_dims(self.coarse_frame_proj(frame), 1)
            cur_f = cur_p = cur_b = nxt

        if len(frames) == 0:
            frames.append(_coarse_frame(last_f, last_p, last_b))

        x = mx.stack(frames, axis=1) # (1, num_gen_frames, coarse_M)

        for k, axis in enumerate(self.split_axes):
            p_known = p_known_list[k]
            x_cat = mx.concatenate([p_known, x], axis=1)
            o_cat = self.up_tiers[k].sample(x_cat, num_steps=up_steps)
            o = o_cat[:, p_known.shape[1] :]
            x = interleave(x, o, axis)

        final = self.denormalize_mel(x)
        return final

