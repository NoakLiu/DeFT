"""
Builder for DeFT module
"""
from .deft_module import DeFTModule


def build_deft_module(config, token_dim=None):
    """
    Build DeFT module from config.
    
    Args:
        config: Model config with DeFT parameters
        token_dim: Dimension of input tokens (if not in config)
    
    Returns:
        DeFTModule instance or None if not enabled
    """
    # Check if DeFT is enabled
    if not getattr(config, 'use_deft', False):
        return None
    
    # Get token dimension
    if token_dim is None:
        token_dim = getattr(config, 'mm_hidden_size', 1024)
    
    # Get DeFT parameters from config
    semantic_dim = getattr(config, 'deft_semantic_dim', 512)
    retention_ratio = getattr(config, 'deft_retention_ratio', 0.482)
    enable_rtd = getattr(config, 'deft_enable_rtd', True)
    alpha_var = getattr(config, 'deft_alpha_var', 0.1)
    rtd_bottleneck_ratio = getattr(config, 'deft_rtd_bottleneck_ratio', 4)
    
    # Create DeFT module
    deft_module = DeFTModule(
        token_dim=token_dim,
        semantic_dim=semantic_dim,
        retention_ratio=retention_ratio,
        enable_rtd=enable_rtd,
        alpha_var=alpha_var,
        rtd_bottleneck_ratio=rtd_bottleneck_ratio,
    )
    
    return deft_module

