# DeFT: Decoupled and Feedback-Guided Tokenization

Implementation of **DeFT (Decoupled and Feedback-Guided Tokenization)** for efficient multimodal long-context modeling.

## Overview

DeFT is a unified tokenization framework that:
1. **Decouples semantic abstraction from compression** - Enables modality-agnostic token selection
2. **Uses feedback-guided scoring** - Adaptively retains informative tokens based on task objectives
3. **Supports recoverable compression** - Optional RTD module for conditional token reconstruction

## Architecture

DeFT consists of three main components:

### 1. Semantic Abstraction Module (SAM)
Projects modality-specific tokens into a unified semantic space (default: 512-dim) via a 2-layer MLP.

### 2. Feedback-Guided Compression Scoring (FGCS)
Computes token importance using three signals:
- **Semantic saliency**: Learned importance via 3-layer MLP
- **Feedback-aware importance**: Gradient-based task feedback (training only)
- **Group-aware scoring**: Neighborhood interaction based on modality

### 3. Recoverable Token Dictionary (RTD)
Enables conditional reconstruction of pruned tokens via information bottleneck (4× compression).

## Usage

### Basic Usage

```python
from llava.model.deft import DeFTModule
import torch

# Create DeFT module
deft = DeFTModule(
    token_dim=1024,           # Input token dimension
    semantic_dim=512,          # Unified semantic space dimension
    retention_ratio=0.482,     # Retain 48.2% of tokens
    enable_rtd=True,           # Enable RTD for reconstruction
)

# Process tokens
tokens = torch.randn(2, 256, 1024)  # (batch, num_tokens, token_dim)
output = deft(
    tokens=tokens,
    modality='image',
    spatial_shape=(16, 16),
    inference_mode='fast'
)

retained_tokens = output['retained_tokens']  # (batch, k, token_dim)
```

### Integration with LLaVA

DeFT is automatically integrated into LLaVA when enabled in the config:

```python
# In config.json or model config
{
    "use_deft": true,
    "deft_semantic_dim": 512,
    "deft_retention_ratio": 0.482,
    "deft_enable_rtd": true,
    "deft_alpha_var": 0.1,
    "deft_rtd_bottleneck_ratio": 4
}
```

### Training with Feedback

During training, provide task loss for feedback-guided scoring:

```python
# Forward pass with task loss
output = deft(
    tokens=tokens,
    modality='image',
    spatial_shape=(16, 16),
    task_loss=task_loss,  # Scalar loss from model
    inference_mode='hybrid'
)
```

### Inference Modes

- **`'fast'`**: Use only retained tokens (fastest)
- **`'hybrid'`**: Retained + conditionally reconstructed tokens (balanced)
- **`'adaptive'`**: Dynamically choose based on uncertainty/confidence

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `token_dim` | - | Dimension of input tokens (required) |
| `semantic_dim` | 512 | Dimension of unified semantic space |
| `retention_ratio` | 0.482 | Fraction of tokens to retain (48.2%) |
| `enable_rtd` | True | Enable Recoverable Token Dictionary |
| `alpha_var` | 0.1 | Variance penalty for feedback scoring |
| `rtd_bottleneck_ratio` | 4 | RTD compression ratio (d/4) |

## Modality Support

### Image
- Uses 8-connected spatial neighbors for group scoring
- Requires `spatial_shape=(height, width)` of patch layout

### Video
- Uses temporal window ±2 for group scoring
- Requires `spatial_shape=(height, width)` of patches per frame

### Text
- Uses TopK(k=5) semantic neighbors for group scoring
- No spatial shape required

## Performance

According to the paper, DeFT achieves:
- **99.5% accuracy retention** at 48.2% token retention
- **2.20× speedup** and **50% memory reduction**
- Superior compression-accuracy trade-offs compared to baselines

## Files

- `deft_module.py`: Main DeFT module integrating all components
- `semantic_abstraction.py`: Semantic Abstraction Module (SAM)
- `scoring.py`: Feedback-Guided Compression Scoring (FGCS)
- `rtd.py`: Recoverable Token Dictionary (RTD)
- `builder.py`: Builder function for easy integration

## References

Based on the paper:
> **DeFT: Decoupled and Feedback-Guided Tokenization for Efficient Multimodal Long-Context Modeling**
> Anonymous CVPR submission

## Example

See `examples/deft_usage_example.py` for comprehensive usage examples.

