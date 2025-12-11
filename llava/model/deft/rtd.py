"""
Recoverable Token Dictionary (RTD)
Enables conditional reconstruction of pruned tokens
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecoverableTokenDictionary(nn.Module):
    """
    Implements token reconstruction via information bottleneck.
    Encodes tokens to bottleneck codes and decodes them conditionally.
    """
    
    def __init__(self, token_dim, bottleneck_ratio=4):
        """
        Args:
            token_dim: Dimension of input tokens
            bottleneck_ratio: Compression ratio (default: 4, so dbottleneck = d/4)
        """
        super().__init__()
        self.token_dim = token_dim
        self.bottleneck_dim = token_dim // bottleneck_ratio
        
        # Encoder: compresses tokens to bottleneck codes
        self.encoder = nn.Sequential(
            nn.Linear(token_dim, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        )
        
        # Decoder: reconstructs tokens from bottleneck codes
        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, token_dim)
        )
        
        # Reconstruction gate: p(reconstruct_i) = σ(Wr · [zi, uncertainty_i, confidence_i] + br)
        self.reconstruct_gate = nn.Sequential(
            nn.Linear(token_dim + 2, 1),  # [zi, uncertainty, confidence]
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize encoder/decoder weights"""
        for module in [self.encoder, self.decoder]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def encode(self, tokens):
        """
        Encode tokens to bottleneck representation.
        
        Args:
            tokens: (batch_size, num_tokens, token_dim) or (num_tokens, token_dim)
        
        Returns:
            codes: (batch_size, num_tokens, bottleneck_dim) or (num_tokens, bottleneck_dim)
        """
        return self.encoder(tokens)
    
    def decode(self, codes):
        """
        Decode bottleneck codes back to tokens.
        
        Args:
            codes: (batch_size, num_tokens, bottleneck_dim) or (num_tokens, bottleneck_dim)
        
        Returns:
            reconstructed_tokens: (batch_size, num_tokens, token_dim) or (num_tokens, token_dim)
        """
        return self.decoder(codes)
    
    def compute_uncertainty(self, logits):
        """
        Compute uncertainty: uncertainty_i = -Σ_c p_c^(i) log p_c^(i)
        
        Args:
            logits: (batch_size, num_tokens, num_classes) or (num_tokens, num_classes)
        
        Returns:
            uncertainty: (batch_size, num_tokens) or (num_tokens)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        uncertainty = -torch.sum(probs * log_probs, dim=-1)
        return uncertainty
    
    def compute_confidence(self, logits):
        """
        Compute confidence: confidence_i = max_c p_c^(i)
        
        Args:
            logits: (batch_size, num_tokens, num_classes) or (num_tokens, num_classes)
        
        Returns:
            confidence: (batch_size, num_tokens) or (num_tokens)
        """
        probs = F.softmax(logits, dim=-1)
        confidence = torch.max(probs, dim=-1)[0]
        return confidence
    
    def should_reconstruct(self, z, uncertainty, confidence, threshold=0.5):
        """
        Decide whether to reconstruct a token based on uncertainty and confidence.
        
        Args:
            z: (batch_size, num_tokens, semantic_dim) or (num_tokens, semantic_dim)
            uncertainty: (batch_size, num_tokens) or (num_tokens)
            confidence: (batch_size, num_tokens) or (num_tokens)
            threshold: Reconstruction threshold (default: 0.5)
        
        Returns:
            reconstruct_mask: (batch_size, num_tokens) or (num_tokens) boolean mask
            reconstruct_probs: (batch_size, num_tokens) or (num_tokens) probabilities
        """
        # Prepare input: [zi, uncertainty_i, confidence_i]
        if z.dim() == 2:
            z = z.unsqueeze(0)
            uncertainty = uncertainty.unsqueeze(0)
            confidence = confidence.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_tokens, dim = z.shape
        uncertainty = uncertainty.unsqueeze(-1)  # (batch_size, num_tokens, 1)
        confidence = confidence.unsqueeze(-1)  # (batch_size, num_tokens, 1)
        
        gate_input = torch.cat([z, uncertainty, confidence], dim=-1)  # (batch_size, num_tokens, dim+2)
        reconstruct_probs = self.reconstruct_gate(gate_input).squeeze(-1)  # (batch_size, num_tokens)
        reconstruct_mask = reconstruct_probs > threshold
        
        if squeeze_output:
            reconstruct_mask = reconstruct_mask.squeeze(0)
            reconstruct_probs = reconstruct_probs.squeeze(0)
        
        return reconstruct_mask, reconstruct_probs
    
    def forward(self, tokens, z=None, logits=None, threshold=0.5):
        """
        Encode tokens and conditionally reconstruct them.
        
        Args:
            tokens: (batch_size, num_tokens, token_dim) original tokens
            z: Optional (batch_size, num_tokens, semantic_dim) for reconstruction gate
            logits: Optional (batch_size, num_tokens, num_classes) for uncertainty/confidence
            threshold: Reconstruction threshold (default: 0.5)
        
        Returns:
            codes: (batch_size, num_tokens, bottleneck_dim) encoded tokens
            reconstructed_tokens: (batch_size, num_tokens, token_dim) reconstructed tokens
            reconstruct_mask: (batch_size, num_tokens) boolean mask for reconstruction
        """
        # Encode tokens
        codes = self.encode(tokens)
        
        # Decode all tokens
        reconstructed_tokens = self.decode(codes)
        
        # Determine which tokens to reconstruct
        if z is not None and logits is not None:
            uncertainty = self.compute_uncertainty(logits)
            confidence = self.compute_confidence(logits)
            reconstruct_mask, reconstruct_probs = self.should_reconstruct(
                z, uncertainty, confidence, threshold=threshold
            )
        else:
            # If no z/logits provided, reconstruct all (for training)
            reconstruct_mask = torch.ones(
                tokens.shape[0], tokens.shape[1],
                device=tokens.device, dtype=torch.bool
            )
            reconstruct_probs = torch.ones(
                tokens.shape[0], tokens.shape[1],
                device=tokens.device, dtype=tokens.dtype
            )
        
        return {
            'codes': codes,
            'reconstructed_tokens': reconstructed_tokens,
            'reconstruct_mask': reconstruct_mask,
            'reconstruct_probs': reconstruct_probs
        }

