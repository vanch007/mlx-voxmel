# mlx-voxmel

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![MLX](https://img.shields.io/badge/MLX-0.31+-brightgreen.svg)](https://github.com/ml-explore/mlx)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-silver.svg)](https://www.apple.com)

Native Apple Silicon [MLX](https://github.com/ml-explore/mlx) implementation of [VoxMel](https://github.com/VoxDance-Lab/VoxMel) (Wavelet-Factorized Continuous Autoregressive Mel-Spectrogram Modeling for Zero-Shot Speech Synthesis).

Supports both **VoxMel-0.6B** and **VoxMel-0.3B** across **bf16**, **8-bit**, and **4-bit** quantization variants, achieving up to **0.67 RTF** on Apple Silicon with 100% text fidelity.

---

## 🌟 Highlights

- **Pure MLX & NumPy Execution**: Completely free of PyTorch, torchaudio, and torchcodec at runtime.
- **Wavelet Mel-Spectrogram Decomposition**: Exact two-axis Haar transform splits (temporal and frequency) implemented in native MLX.
- **Continuous Flow Matching & Dual-Guidance**: Full classifier-free guidance support (both text guidance and speaker guidance).
- **Pure MLX Vocos 24kHz Vocoder**: Native ConvNeXt backbone and ISTFT audio synthesis running directly on Metal GPU unified memory.
- **Multi-Quantization Matrix**: Full support for official 0.6B and 0.3B checkpoints in **bf16**, **8-bit** (affine group-size 64), and **4-bit**.
- **Verified by Local ASR**: Fully benchmarked against official demo samples using local `Confucius4-R2T2-8bit` speech recognition.

---

## 📊 Benchmark & Evaluation Results

Tested on Apple Silicon using official zero-shot demo samples from [VoxMel Demo](https://voxdance.github.io/voxmel-demo/):

| Model Variant | Format | Model Size | Real-Time Factor (RTF) | Chinese ASR Accuracy | English ASR Accuracy | Status |
|---|---|---|---|---|---|---|
| **Vocos-Mel-24kHz** | fp32 | 52 MB | — | — | — | ✅ Pure MLX Vocoder |
| **VoxMel-0.6B-bf16** | bf16 | 1.3 GB | 2.55 | 100% Match | 100% Match | ✅ Full Precision |
| **VoxMel-0.6B-8bit** | int8 | 1.1 GB | **0.80 (Real-Time)** | 100% Match | 100% Match | 🌟 **Production Choice** |
| **VoxMel-0.6B-4bit** | int4 | 863 MB | **0.72 (Fast)** | Minor initial swallow | High | ⚡ Ultra Fast |
| **VoxMel-0.3B-bf16** | bf16 | 605 MB | 1.15 | 100% Match | 100% Match | ✅ Baseline |
| **VoxMel-0.3B-8bit** | int8 | 666 MB | **0.67 (Real-Time)** | 100% Match | 100% Match | 🌟 **Lightweight Choice** |
| **VoxMel-0.3B-4bit** | int4 | 574 MB | **0.68 (Real-Time)** | 100% Match | 100% Match | ⚡ Minimal Memory |

*Note: In Chinese zero-shot cloning, the official demo audio suffers from ending text hallucination ("否则岂不吓坏了你们"), and in English it generates extra unprompted sentences. MLX VoxMel strictly respects the target text and stop token boundaries.*

A complete interactive single-page HTML evaluation report with audio players and ASR transcripts is located at [`full_evaluation/index.html`](full_evaluation/index.html).

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/vanch007/mlx-voxmel.git
cd mlx-voxmel
pip install mlx mlx-lm soundfile pyyaml omegaconf transformers
```

### 2. Download Weights

Pre-converted MLX weights are hosted on Hugging Face:
- [vanch007/VoxMel-0.6B-MLX](https://huggingface.co/vanch007/VoxMel-0.6B-MLX) (bf16, 8bit, 4bit)
- [vanch007/VoxMel-0.3B-MLX](https://huggingface.co/vanch007/VoxMel-0.3B-MLX) (bf16, 8bit, 4bit)
- [vanch007/vocos-mel-24khz-mlx](https://huggingface.co/vanch007/vocos-mel-24khz-mlx)

Or clone directly into `models/`:
```bash
# Download 8-bit model and vocos
huggingface-cli download vanch007/VoxMel-0.6B-MLX --include "VoxMel-0.6B-8bit/*" --local-dir models/
huggingface-cli download vanch007/vocos-mel-24khz-mlx --local-dir models/vocos-mel-24khz/
```

### 3. CLI Voice Cloning

```bash
python inference.py \
  --model models/VoxMel-0.6B-8bit \
  --prompt-audio demo_samples/zh_test_01/prompt.wav \
  --prompt-text "对,这就是我万人敬仰的太乙真人" \
  --text "我看着他们，意气风发地挺直了胸膛，轻笑道：我身上的肉，是为了掩饰我爆棚的魅力，懂不懂？" \
  --output clone_output.wav \
  --steps 32 --up-steps 32
```

### 4. Python API

```python
from inference import VoxMel
import soundfile as sf

# Load model
model = VoxMel.from_pretrained("models/VoxMel-0.6B-8bit")

# Zero-shot voice cloning
audio = model.generate(
    "这是基于 Apple Silicon MLX 原生运行的语音合成演示。",
    prompt_wav_path="reference.wav",
    prompt_text="参考音频中实际说出的内容。",
    inference_timesteps=32,
    up_num_steps=32,
    cfg_text_strength=3.5,
    cfg_spk_strength=3.0,
    seed=42,
)

sf.write("output.wav", audio, model.sample_rate)
```

---

## 🛠️ Weight Conversion

To convert official PyTorch checkpoints from Hugging Face:

```bash
python convert.py
```

This will convert official `VoxMel-0.6B` and `VoxMel-0.3B` checkpoints along with `charactr/vocos-mel-24khz` into pure MLX safetensors and produce all bf16, 8bit, and 4bit quantized variants.

---

## 📄 License & Attribution

- VoxMel model architecture and pre-trained weights by [VoxDance-Lab](https://github.com/VoxDance-Lab/VoxMel) (Apache License 2.0).
- Vocos vocoder by [charactr-platform](https://github.com/gemfilter/vocos).
- MLX implementation developed for Apple Silicon by [vanch007](https://github.com/vanch007).

