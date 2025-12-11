"""
Semantic Abstraction Module (SAM)
Projects modality-specific tokens into a unified semantic space
"""
import torch
import torch.nn as nn


class SemanticAbstractionModule(nn.Module):
    """
    Projects modality-specific tokens into a unified semantic space.
    
    According to the paper:
    - zi = A(f_enc^(m)(xi)) where A is a 2-layer MLP
    - zi ∈ R^d' where d' = 512
    """
    
    def __init__(self, input_dim, output_dim=512):
        """
        Args:
            input_dim: Dimension of input tokens (varies by modality)
            output_dim: Dimension of unified semantic space (default: 512)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 2-layer MLP: zi = W2 · ReLU(W1 · f_enc^(m)(xi) + b1) + b2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize MLP weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, tokens):
        """
        Args:
            tokens: (batch_size, num_tokens, input_dim) tensor from modality encoder
        
        Returns:
            abstracted_tokens: (batch_size, num_tokens, output_dim) tensor in unified space
        """
        return self.mlp(tokens)

