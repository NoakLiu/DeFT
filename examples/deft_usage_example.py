"""
Example usage of DeFT module
"""
import torch
import torch.nn as nn
from llava.model.deft import DeFTModule


def example_basic_usage():
    """Basic example of using DeFT for token compression"""
    # Setup
    batch_size = 2
    num_tokens = 256  # e.g., 16x16 image patches
    token_dim = 1024  # CLIP vision encoder output dimension
    
    # Create DeFT module
    deft = DeFTModule(
        token_dim=token_dim,
        semantic_dim=512,
        retention_ratio=0.482,  # Retain 48.2% of tokens
        enable_rtd=True,
    )
    
    # Create dummy tokens (e.g., from vision encoder)
    tokens = torch.randn(batch_size, num_tokens, token_dim)
    
    # Forward pass
    output = deft(
        tokens=tokens,
        modality='image',
        spatial_shape=(16, 16),  # 16x16 spatial layout
        inference_mode='fast'
    )
    
    print(f"Input tokens: {tokens.shape}")
    print(f"Retained tokens: {output['retained_tokens'].shape}")
    print(f"Retention ratio: {output['retention_ratio']:.2%}")
    print(f"Number retained: {output['num_retained']}")
    
    return output


def example_with_feedback():
    """Example with feedback-guided scoring during training"""
    batch_size = 2
    num_tokens = 256
    token_dim = 1024
    
    deft = DeFTModule(
        token_dim=token_dim,
        semantic_dim=512,
        retention_ratio=0.482,
        enable_rtd=True,
    )
    
    tokens = torch.randn(batch_size, num_tokens, token_dim, requires_grad=True)
    
    # Simulate a task loss (in practice, this comes from the model)
    # For demonstration, we'll create a dummy loss
    dummy_output = tokens.mean()
    task_loss = dummy_output ** 2  # Dummy loss
    
    # Forward with feedback
    output = deft(
        tokens=tokens,
        modality='image',
        spatial_shape=(16, 16),
        task_loss=task_loss,
        inference_mode='hybrid'
    )
    
    print(f"Feedback-guided scoring enabled")
    print(f"Scoring info: {output['scoring_info'].keys()}")
    print(f"Mixture weights: {output['scoring_info']['mixture_weights']}")
    
    return output


def example_with_rtd():
    """Example with Recoverable Token Dictionary"""
    batch_size = 2
    num_tokens = 256
    token_dim = 1024
    num_classes = 1000
    
    deft = DeFTModule(
        token_dim=token_dim,
        semantic_dim=512,
        retention_ratio=0.40,  # More aggressive compression
        enable_rtd=True,
    )
    
    tokens = torch.randn(batch_size, num_tokens, token_dim)
    logits = torch.randn(batch_size, num_tokens, num_classes)  # Dummy logits
    
    # Forward with RTD reconstruction
    output = deft(
        tokens=tokens,
        modality='image',
        spatial_shape=(16, 16),
        logits=logits,
        inference_mode='hybrid'
    )
    
    print(f"RTD enabled")
    if output['reconstructed_tokens'] is not None:
        print(f"Reconstructed tokens: {output['reconstructed_tokens'].shape}")
        print(f"Reconstructed indices: {output['reconstructed_indices'].shape}")
    
    return output


def example_video_modality():
    """Example for video tokens"""
    batch_size = 1
    num_frames = 8
    patches_per_frame = 256  # 16x16 patches
    num_tokens = num_frames * patches_per_frame
    token_dim = 1024
    
    deft = DeFTModule(
        token_dim=token_dim,
        semantic_dim=512,
        retention_ratio=0.482,
        enable_rtd=True,
    )
    
    tokens = torch.randn(batch_size, num_tokens, token_dim)
    
    # For video, spatial_shape is (height, width) of patches per frame
    output = deft(
        tokens=tokens,
        modality='video',
        spatial_shape=(16, 16),
        inference_mode='fast'
    )
    
    print(f"Video tokens: {tokens.shape}")
    print(f"Retained tokens: {output['retained_tokens'].shape}")
    
    return output


def example_text_modality():
    """Example for text tokens"""
    batch_size = 2
    num_tokens = 512  # Text sequence length
    token_dim = 768  # Typical text encoder dimension
    
    deft = DeFTModule(
        token_dim=token_dim,
        semantic_dim=512,
        retention_ratio=0.482,
        enable_rtd=False,  # RTD less common for text
    )
    
    tokens = torch.randn(batch_size, num_tokens, token_dim)
    
    output = deft(
        tokens=tokens,
        modality='text',
        inference_mode='fast'
    )
    
    print(f"Text tokens: {tokens.shape}")
    print(f"Retained tokens: {output['retained_tokens'].shape}")
    
    return output


if __name__ == '__main__':
    print("=" * 60)
    print("DeFT Usage Examples")
    print("=" * 60)
    
    print("\n1. Basic Usage:")
    print("-" * 60)
    example_basic_usage()
    
    print("\n2. With Feedback Scoring:")
    print("-" * 60)
    example_with_feedback()
    
    print("\n3. With RTD Reconstruction:")
    print("-" * 60)
    example_with_rtd()
    
    print("\n4. Video Modality:")
    print("-" * 60)
    example_video_modality()
    
    print("\n5. Text Modality:")
    print("-" * 60)
    example_text_modality()
    
    print("\n" + "=" * 60)
    print("All examples completed!")

