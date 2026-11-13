"""Mel-Spectrogram extraction and audio loading utilities in pure NumPy / MLX."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
import mlx.core as mx


# Load precomputed torchaudio-exact mel filterbank (513, 100)
_FB_PATH = Path(__file__).resolve().parent / "mel_filters_100.npy"
_MEL_FILTERS = None

def get_mel_filters() -> np.ndarray:
    global _MEL_FILTERS
    if _MEL_FILTERS is None:
        if not _FB_PATH.exists():
            fb_asset = Path("/Users/vanch/VoxMel-repo/assets/mel_filters_100.npy")
            if fb_asset.exists():
                _MEL_FILTERS = np.load(str(fb_asset))
            else:
                raise FileNotFoundError(f"Missing mel filters at {_FB_PATH} or {fb_asset}")
        else:
            _MEL_FILTERS = np.load(str(_FB_PATH))
    return _MEL_FILTERS


def load_audio(path: str | Path, target_sr: int = 24000) -> np.ndarray:
    wav, sr = sf.read(str(path))
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    if sr != target_sr:
        import scipy.signal
        num_samples = int(round(len(wav) * target_sr / sr))
        wav = scipy.signal.resample(wav, num_samples).astype(np.float32)
    return wav


def extract_mel(
    wav: np.ndarray,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 100,
) -> mx.array:
    """Exact torchaudio-compatible mel spectrogram extraction.
    Returns: mx.array of shape (1, num_frames, n_mels)
    """
    assert n_fft == 1024 and hop_length == 256 and n_mels == 100 and sample_rate == 24000
    fb = get_mel_filters() # (513, 100)
    
    # Hann window with periodic=True: 1024 points
    denom = n_fft
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / denom)).astype(np.float32)
    
    pad_len = n_fft // 2
    wav_pad = np.pad(wav, (pad_len, pad_len), mode="reflect")
    
    frames = [
        wav_pad[i : i + n_fft] * window
        for i in range(0, len(wav_pad) - n_fft + 1, hop_length)
    ]
    if len(frames) == 0:
        raise ValueError("Audio is too short for mel extraction")
    frames = np.stack(frames, axis=0) # (num_frames, 1024)
    
    spec = np.abs(np.fft.rfft(frames, axis=-1)) # (num_frames, 513)
    mel = np.matmul(spec, fb) # (num_frames, 100)
    mel = np.log(np.maximum(mel, 1e-5))
    
    # Return as MLX array with batch dim: (1, num_frames, 100)
    return mx.array(mel[None, ...], dtype=mx.float32)
