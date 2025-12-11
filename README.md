# DeFT: Decoupled and Feedback-Guided Tokenization for Efficient Multimodal Long-Context Modeling

<p align="center">
  <img src="staging.png" alt="DeFT Framework" width="90%">
</p>

**DeFT** is a unified tokenization framework designed for efficient multimodal and long-context modeling. It decouples semantic abstraction from compression decisions, enabling adaptive token selection with feedback-guided mechanisms and optional recoverable compression.

## 🎯 Key Features

- **Decoupled Architecture**: Separates semantic abstraction from compression, enabling modality-agnostic token selection
- **Feedback-Guided Scoring**: Adaptively retains informative tokens using learned saliency, task gradients, and group interactions
- **Recoverable Compression**: Optional RTD module enables conditional reconstruction of pruned tokens
- **Multi-Modality Support**: Works seamlessly with text, image, and video inputs
- **Superior Performance**: Achieves 99.5% accuracy retention at 48.2% token retention with 2.20× speedup and 50% memory reduction

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Performance](#performance)
- [Citation](#citation)

## 🏗️ Overview

DeFT addresses the fundamental challenge of balancing information retention with computational efficiency in long-context multimodal scenarios. Unlike existing methods that entangle semantic abstraction with compression, DeFT operates in three sequential phases:

1. **Semantic Abstraction**: Projects modality-specific tokens into a unified semantic space
2. **Hybrid Scoring**: Computes importance using semantic, feedback, and group signals
3. **Recoverable Compression**: Selects top-k tokens with optional reconstruction

## 🧩 Architecture

### Phase 1: Semantic Abstraction Module (SAM)

Projects tokens from any modality into a unified 512-dimensional semantic space via a 2-layer MLP:

```
zi = A(f_enc^(m)(xi))  where A is a 2-layer MLP
```

### Phase 2: Feedback-Guided Compression Scoring (FGCS)

Combines three scoring signals:

- **Semantic Saliency** (`s_sem`): Learned importance via 3-layer MLP
- **Feedback-Aware Importance** (`s_fb`): Gradient-based task feedback (training only)
  ```
  s_fb_i = ||∇zi L_task||_2 · exp(-α_var · Var(∇zi L_task))
  ```
- **Group-Aware Scoring** (`s_group`): Neighborhood interactions
  - Images: 8-connected spatial neighbors
  - Video: Temporal window ±2
  - Text: TopK(k=5) semantic neighbors

Final score: `si = β1 · s̃_sem_i + β2 · s̃_fb_i + β3 · s̃_group_i`

### Phase 3: Recoverable Token Dictionary (RTD)

Optional module for conditional token reconstruction:
- Encodes pruned tokens to bottleneck representation (4× compression)
- Decodes based on uncertainty and confidence thresholds
- Enables hybrid inference paths

## 🚀 Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 12.1+ (for GPU acceleration)

### Install DeFT

```bash
# Clone the repository
git clone https://github.com/your-repo/DeFT
cd DeFT

# Create conda environment
conda create -n deft python=3.10 -y
conda activate deft

# Install package
pip install --upgrade pip 
pip install -e .

# Install training dependencies (optional)
pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir
```

## ⚡ Quick Start

### Basic Usage

```python
from llava.model.deft import DeFTModule
import torch

# Create DeFT module
deft = DeFTModule(
    token_dim=1024,           # Input token dimension
    semantic_dim=512,          # Unified semantic space dimension
    retention_ratio=0.482,    # Retain 48.2% of tokens
    enable_rtd=True,           # Enable RTD for reconstruction
)

# Process image tokens
tokens = torch.randn(2, 256, 1024)  # (batch, num_tokens, token_dim)
output = deft(
    tokens=tokens,
    modality='image',
    spatial_shape=(16, 16),
    inference_mode='fast'
)

retained_tokens = output['retained_tokens']  # (batch, k, token_dim)
print(f"Compressed from {tokens.shape[1]} to {retained_tokens.shape[1]} tokens")
```

### Integration with LLaVA

DeFT is automatically integrated into LLaVA when enabled in the config:

```python
# In config.json
{
    "use_deft": true,
    "deft_semantic_dim": 512,
    "deft_retention_ratio": 0.482,
    "deft_enable_rtd": true,
    "deft_alpha_var": 0.1,
    "deft_rtd_bottleneck_ratio": 4
}
```

The model will automatically apply DeFT tokenization during image encoding:

```python
# In your training/evaluation code
image_features = model.encode_images(images)  # Automatically uses DeFT if enabled
```

## ⚙️ Configuration

### DeFT Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_deft` | `false` | Enable DeFT tokenization |
| `deft_semantic_dim` | `512` | Dimension of unified semantic space |
| `deft_retention_ratio` | `0.482` | Fraction of tokens to retain (48.2%) |
| `deft_enable_rtd` | `true` | Enable Recoverable Token Dictionary |
| `deft_alpha_var` | `0.1` | Variance penalty for feedback scoring |
| `deft_rtd_bottleneck_ratio` | `4` | RTD compression ratio (d/4) |

### Inference Modes

- **`'fast'`**: Use only retained tokens (fastest, O(k²) attention)
- **`'hybrid'`**: Retained + conditionally reconstructed tokens (balanced)
- **`'adaptive'`**: Dynamically choose based on uncertainty/confidence

## 📚 Usage Examples

### Training with Feedback

```python
# During training, provide task loss for feedback-guided scoring
output = deft(
    tokens=tokens,
    modality='image',
    spatial_shape=(16, 16),
    task_loss=task_loss,  # Scalar loss from model
    inference_mode='hybrid'
)

# Access scoring information
scoring_info = output['scoring_info']
print(f"Mixture weights: {scoring_info['mixture_weights']}")
```

### Video Modality

```python
# For video tokens
deft = DeFTModule(token_dim=1024, retention_ratio=0.482)
tokens = torch.randn(1, 2048, 1024)  # 8 frames × 256 patches

output = deft(
    tokens=tokens,
    modality='video',
    spatial_shape=(16, 16),  # Patches per frame
    inference_mode='fast'
)
```

### Text Modality

```python
# For text tokens
deft = DeFTModule(token_dim=768, retention_ratio=0.482, enable_rtd=False)
tokens = torch.randn(2, 512, 768)  # Text sequence

output = deft(
    tokens=tokens,
    modality='text',
    inference_mode='fast'
)
```

### With RTD Reconstruction

```python
# Enable conditional reconstruction
deft = DeFTModule(token_dim=1024, enable_rtd=True)
logits = torch.randn(2, 256, 1000)  # Model predictions

output = deft(
    tokens=tokens,
    modality='image',
    spatial_shape=(16, 16),
    logits=logits,
    inference_mode='hybrid'
)

if output['reconstructed_tokens'] is not None:
    print(f"Reconstructed {output['reconstructed_tokens'].shape[1]} tokens")
```

See `examples/deft_usage_example.py` for more comprehensive examples.

## 📊 Performance

### Experimental Results

DeFT achieves superior compression-accuracy trade-offs across diverse benchmarks:

| Dataset | Retention | Accuracy | Speedup | Memory Reduction |
|---------|-----------|----------|---------|------------------|
| Ego4D-QA | 48.2% | 99.5% | 2.20× | 50% |
| ChartQA | 48.2% | 99.6% | 2.20× | 50% |
| TextVQA | 48.2% | 99.6% | 2.20× | 50% |
| LongBench | 48.2% | 99.2% | 2.20× | 50% |
| DocVQA | 48.2% | 99.3% | 2.20× | 50% |

### Comparison with Baselines

DeFT outperforms recent strong baselines including:
- **MADTP** [6]: +0.3% accuracy at similar retention
- **ATP-LLaVA** [36]: +0.5% accuracy with better speedup
- **DivPrune** [1]: +0.7% accuracy
- **TopV** [35]: +0.9% accuracy
- **Zero-TPrune** [32]: +0.8% accuracy

### Efficiency Metrics

At 48.2% retention:
- **Throughput**: 170 tokens/s (vs 95 tokens/s baseline) = 1.79× speedup
- **Memory**: 34.2GB (vs 68.5GB baseline) = 50% reduction
- **Latency**: 488ms (vs 1053ms baseline) = 2.16× faster
- **FLOPs**: 0.623T (vs 1.245T baseline) = 50% reduction

## 🧪 Evaluation

### Supported Benchmarks

DeFT has been evaluated on:

- **Video Understanding**: Ego4D-QA, ActivityNet-QA, Video-ChatGPT
- **Document Understanding**: ChartQA, DocVQA, LongBench
- **Multimodal Reasoning**: M4C-TextVQA, LLaVA-Bench, MMMU
- **Generation Tasks**: Video Captioning, Image Captioning
- **Retrieval Tasks**: Cross-modal Retrieval

### Running Evaluation

```bash
# Set environment variables
export ROOT_DATA=/path/to/data
export ROOT_WEIGHT=/path/to/weights

# Run evaluation on specific benchmark
bash scripts/v1_5/eval/deft/textvqa.sh
bash scripts/v1_5/eval/deft/llavabench.sh
```

## 📖 Citation

If you find DeFT useful in your research, please cite:

```bibtex
@article{deft2024,
  title={DeFT: Decoupled and Feedback-Guided Tokenization for Efficient Multimodal Long-Context Modeling},
  author={Anonymous},
  journal={CVPR},
  year={2024}
}
```

## 🙏 Acknowledgement

This implementation is based on:
- [LLaVA](https://github.com/haotian-liu/LLaVA) - Large Language and Vision Assistant
- [Vicuna](https://github.com/lm-sys/FastChat) - FastChat framework

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- **Token Merging**: [ToMe](https://github.com/facebookresearch/ToMe)
- **Token Pruning**: [SAINT](https://github.com/your-repo/SAINT)
- **Multimodal Compression**: [MADTP](https://github.com/your-repo/MADTP), [ATP-LLaVA](https://github.com/your-repo/ATP-LLaVA)

## 📧 Contact

For questions and issues, please open an issue on GitHub.

---

**Note**: This is an implementation of the DeFT framework described in the CVPR submission. For the latest updates and detailed documentation, see `llava/model/deft/README.md`.
