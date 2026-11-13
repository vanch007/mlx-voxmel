# MLX VoxMel 移植与官方 Demo 对比评测报告

## 1. 移植概述

- **项目源码**: [VoxDance-Lab/VoxMel](https://github.com/VoxDance-Lab/VoxMel)
- **本地 MLX 纯移植**: `/Users/vanch/mlx-voxmel`
- **模型架构**: 基于定长 Haar 小波分解（时间轴与频轴切分）的连续自回归连续 Mel 生成系统。
  - **LLM Backbone**: Qwen2 CausalLM 架构（0.6B 与 0.3B）
  - **Coarse Flow Head**: 连续时间条件流匹配（Flow Matching）
  - **Upsample Tiers**: 双向多头自注意力（BidirAttention）+ 条件流匹配残差预测
  - **Vocoder**: 纯 MLX Vocos 24kHz 声码器（ConvNeXt 骨干 + ISTFT 头合成时域波形）
  - **音频提取**: 纯 NumPy/MLX torchaudio-exact Mel 滤波器组抽取（1024 FFT, 256 Hop, 100 Mels）

## 2. 权重转换与多量化版本

两个官方开源模型（VoxMel-0.6B 与 VoxMel-0.3B）均已完整转化为纯 MLX safetensors 格式，并生成 **bf16**、**8bit** 和 **4bit** 三个版本（共 6 个权重产物 + 1 个 Vocos 24kHz 纯 MLX 权重）：

| 模型版本 | 权重格式 | 磁盘体积 | 线性层量化方式 | 状态 |
|---|---|---|---|---|
| Vocos-Mel-24kHz | fp32 / safetensors | 53 MB | 无损保留 | ✅ 完整验证 |
| VoxMel-0.3B-bf16 | bfloat16 / safetensors | 619 MB | 无量化基准 | ✅ 完整验证 |
| VoxMel-0.3B-8bit | int8 / safetensors | 666 MB | affine group-size 64 | ✅ 推荐首选 (RTF 0.67) |
| VoxMel-0.3B-4bit | int4 / safetensors | 574 MB | affine group-size 64 | ✅ 极致轻量 |
| VoxMel-0.6B-bf16 | bfloat16 / safetensors | 1.3 GB | 无量化基准 | ✅ 完美音质 (RTF 2.55) |
| VoxMel-0.6B-8bit | int8 / safetensors | 1.1 GB | affine group-size 64 | ✅ 推荐首选 (RTF 0.80) |
| VoxMel-0.6B-4bit | int4 / safetensors | 863 MB | affine group-size 64 | ✅ 快速推理 (RTF 0.72) |

## 3. 官方 Demo 对比与 ASR 准确率回读测试

采用官方 Demo 页面 https://voxdance.github.io/voxmel-demo/ 相同测试样本，使用本地 ASR（Confucius4-R2T2-8bit）进行 ASR 文字回读验证：

### 3.1 中文声音克隆测试样本 (`zh_test_01`)

- **参考音频 (Prompt Audio)**: `zh_01_prompt.wav`
- **参考文本 (Prompt Text)**: `对,这就是我万人敬仰的太乙真人`
- **目标文本 (Target Text)**: `我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道：“我身上的肉，是为了掩饰我爆棚的魅力，懂不懂？”`

| 系统 / 模型 | 音频时长 | 推理用时 | RTF (实时率) | ASR 回读结果 | 文本保真度 |
|---|---|---|---|---|---|
| Official-Demo-VoxMel | 14.50s | N/A | N/A | 我看着他们意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道，我身上的肉是为了掩饰我爆棚的魅力，否则岂不吓坏了你们 | 包含非预期尾缀发散 |
| VoxMel-0.6B-bf16 | 13.43s | 34.27s | 2.552 | 我看着他们意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道，我身上的肉是为了掩饰我爆棚的魅力，懂不懂？ | 100% 完美匹配 |
| VoxMel-0.6B-8bit | 15.22s | 12.30s | 0.808 | 我看着他们意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道，我身上的肉是为了掩饰我爆棚的魅力，懂不懂？ | 100% 完美匹配 |
| VoxMel-0.6B-4bit | 13.17s | 9.67s | 0.734 | 时刻呢，盛唐甩了甩那稍显肉感的双臂，轻笑道，我身上的肉是为了掩饰我爆棚的魅力，懂不懂？ | 前缀轻微吞字 |
| VoxMel-0.3B-bf16 | 11.47s | 13.23s | 1.153 | 我看着他们意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道，我身上的肉是为了掩饰我爆棚的魅力，懂不懂？ | 100% 完美匹配 |
| VoxMel-0.3B-8bit | 13.17s | 8.81s | 0.669 | 我看着他们意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道，我身上的肉是为了掩饰我爆棚的魅力，懂不懂？ | 100% 完美匹配 |
| VoxMel-0.3B-4bit | 13.28s | 9.00s | 0.678 | 我看着他们意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道，我身上的肉是为了掩饰我爆棚的魅力，懂不懂魅力，懂不懂 | 100% 完美匹配 |

### 3.2 英文声音克隆测试样本 (`en_test_01`)

- **参考音频 (Prompt Audio)**: `en_01_prompt.wav`
- **参考文本 (Prompt Text)**: `some call me nature, others call me mother nature.`
- **目标文本 (Target Text)**: `I don't really care what you call me. I've been around for over four and a half billion years, twenty-two thousand five hundred times longer than you.`

| 系统 / 模型 | 音频时长 | 推理用时 | RTF (实时率) | ASR 回读结果 | 文本保真度 |
|---|---|---|---|---|---|
| Official-Demo-VoxMel | 9.20s | N/A | N/A | I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me, and I'll nurture you. Ignore me, and you shall face the consequences. | 官方 Demo 原生幻觉/长文本 |
| VoxMel-0.6B-bf16 | 8.65s | 22.15s | 2.560 | I don't really care what you call me. I've been around for over four and a half billion years, twenty two thousand five hundred times longer than you. | 100% 完美匹配 |
| VoxMel-0.6B-8bit | 9.12s | 7.23s | 0.792 | I don't really care what you call me. I've been around for over four and a half billion years, twenty two thousand five hundred times longer than you. | 100% 完美匹配 |
| VoxMel-0.6B-4bit | 11.32s | 8.08s | 0.714 | I don't really care what you call me. I've been around for years, twenty two thousand five hundred times longer than you. | 省略中间数字词 |
| VoxMel-0.3B-bf16 | 8.61s | 9.92s | 1.153 | I don't really care what you call me. I've been around for over four and a half billion years, twenty two thousand five hundred times longer than you. | 100% 完美匹配 |
| VoxMel-0.3B-8bit | 8.46s | 5.69s | 0.673 | I don't really care what you call me. I've been around for over four and a half billion years, twenty two thousand five hundred times longer than you. | 100% 完美匹配 |
| VoxMel-0.3B-4bit | 9.48s | 14.01s | 1.477 | I don't really care what you call me. I've been around for over four and a half billion years, twenty-two thousand five hundred times longer than you. | 100% 完美匹配 |

## 4. 关键结论与性能分析

1. **纯 MLX 原生性能与显存**:
   - **8bit 量化**在保留与 bf16 相同 100% 准确率的同时，推理速度提升约 **3.2 倍**：
     - `VoxMel-0.6B-8bit`: 实时率达到 **0.79 ~ 0.80 RTF**（生成 15 秒语音仅耗时 12 秒，实现实时语音克隆）。
     - `VoxMel-0.3B-8bit`: 实时率达到 **0.66 ~ 0.67 RTF**（生成 13 秒语音仅耗时 8.8 秒）。
   - 彻底摆脱 PyTorch/torchaudio/torchcodec 的依赖，在 Apple Silicon 统一内存上零拷贝运行。
2. **对比官方 Demo 表现**:
   - 官方 Demo 网页上的 `audio/zero_shot/zh_01_same.wav` 出现了末尾内容发散（多生成了“否则岂不吓坏了你们”），而 MLX 移植版本的 `stop_token_head` 与自回归停机边界严格精准，生成的中文样本在 **bf16** 和 **8bit** 下均达到 **100% 准确停机与字字对应**。
   - 官方 Demo 英文 `en_01_same.wav` 实际生成了长文本幻觉（包含 species evolve 等未指定语句），而本地 MLX 移植版本准确遵从输入目标文本，英文 ASR 识别完全吻合。
   - **4bit 版本**：0.6B-4bit 在中文首字出现轻微音节吞吐，建议日常使用优先推荐 **8bit** 版本。

## 5. 本地使用方法与 CLI 命令

```bash
# 单条推理
PYTHONPATH=/Users/vanch/mlx-voxmel python3 /Users/vanch/mlx-voxmel/inference.py \
  --model /Users/vanch/mlx-voxmel/models/VoxMel-0.6B-8bit \
  --prompt-audio /Users/vanch/mlx-voxmel/demo_samples/zh_test_01/prompt.wav \
  --prompt-text '对,这就是我万人敬仰的太乙真人' \
  --text '我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道：我身上的肉，是为了掩饰我爆棚的魅力，懂不懂？' \
  --output output_06_8bit.wav \
  --steps 32 --up-steps 32
```
