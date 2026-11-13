"""Pure MLX implementation of Vocos neural vocoder with ISTFT synthesis."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import yaml


def hanning(size: int, periodic: bool = False) -> mx.array:
    denom = size if periodic else size - 1
    return mx.array([0.5 * (1.0 - math.cos(2.0 * math.pi * n / denom)) for n in range(size)])


def istft(
    x: mx.array,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[mx.array] = None,
    center: bool = True,
    length: Optional[int] = None,
    normalized: bool = False,
) -> mx.array:
    n_fft = (x.shape[0] - 1) * 2
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4

    if window is None:
        w = hanning(win_length)
    else:
        w = window

    if w.shape[0] < n_fft:
        padding = n_fft - w.shape[0]
        w = mx.pad(w, [(padding // 2, padding - padding // 2)])

    num_frames = x.shape[1]
    t = (num_frames - 1) * hop_length + n_fft

    reconstructed = mx.zeros((t,))
    window_sum = mx.zeros((t,))

    frames_time = mx.fft.irfft(x, axis=0).transpose(1, 0)
    frame_offsets = mx.arange(num_frames) * hop_length
    indices = frame_offsets[:, None] + mx.arange(n_fft)
    indices_flat = indices.flatten()

    updates_reconstructed = (frames_time * w).flatten()
    window_norm = (w * w) if normalized else w
    updates_window = mx.tile(window_norm, (num_frames,)).flatten()

    reconstructed = reconstructed.at[indices_flat].add(updates_reconstructed)
    window_sum = window_sum.at[indices_flat].add(updates_window)

    denominator = mx.where(window_sum > 1e-10, window_sum, 1.0)
    reconstructed = reconstructed / denominator

    start = n_fft // 2 if center else 0
    end = start + length if length is not None else t - start
    reconstructed = reconstructed[start:end]
    if length is not None and reconstructed.shape[0] < length:
        reconstructed = mx.pad(reconstructed, [(0, length - reconstructed.shape[0])])

    return reconstructed


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "center"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.out(x).swapaxes(1, 2)
        mag, p = x.split(2, axis=1)
        mag = mx.exp(mag)
        mag = mx.clip(mag, None, 1e2)
        x_cos = mx.cos(p)
        y_sin = mx.sin(p)
        S = mag * (x_cos + 1j * y_sin)
        audio = istft(
            S.squeeze(0),
            window=hanning(self.n_fft),
            hop_length=self.hop_length,
            win_length=self.n_fft,
        )
        return audio


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        dw_kernel_size: int = 7,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=dw_kernel_size,
            padding=dw_kernel_size // 2,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            layer_scale_init_value * mx.ones(dim)
            if layer_scale_init_value > 0
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return residual + x


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        bias: bool = True,
        input_kernel_size: int = 7,
        dw_kernel_size: int = 7,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(
            input_channels,
            dim,
            kernel_size=input_kernel_size,
            padding=input_kernel_size // 2,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1.0 / num_layers
        self.convnext = [
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
                dw_kernel_size=dw_kernel_size,
            )
            for _ in range(num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self.input_channels:
            x = x.transpose(0, 2, 1)

        x = self.embed(x)
        x = self.norm(x)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x)
        return x


class Vocos(nn.Module):
    def __init__(self, backbone: VocosBackbone, head: ISTFTHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_hparams(cls, config: dict) -> Vocos:
        cfg = SimpleNamespace(**config)
        backbone = VocosBackbone(**cfg.backbone["init_args"])
        head = ISTFTHead(**cfg.head["init_args"])
        return cls(backbone=backbone, head=head)

    def decode(self, features_input: mx.array) -> mx.array:
        x = self.backbone(features_input)
        audio_output = self.head(x)
        return audio_output
