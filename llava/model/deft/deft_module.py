"""
DeFT: Decoupled and Feedback-Guided Tokenization
Main module integrating SAM, FGCS, and RTD
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List

from .semantic_abstraction import SemanticAbstractionModule
from .scoring import FeedbackGuidedScoring
from .rtd import RecoverableTokenDictionary


class DeFTModule(nn.Module):
    """
    DeFT: Decoupled and Feedback-Guided Tokenization for Efficient Multimodal Long-Context Modeling
    
    Implements the three-phase pipeline:
    1. Semantic Abstraction: Projects tokens to unified semantic space
    2. Hybrid Scoring: Computes importance scores using semantic, feedback, and group signals
    3. Recoverable Compression: Selects top-k tokens and optionally reconstructs pruned ones
    """
    
    def __init__(
        self,
        token_dim: int,
        semantic_dim: int = 512,
        retention_ratio: float = 0.482,
        enable_rtd: bool = True,
        alpha_var: float = 0.1,
        rtd_bottleneck_ratio: int = 4,
    ):
        """
        Args:
            token_dim: Dimension of input tokens (from encoder)
            semantic_dim: Dimension of unified semantic space (default: 512)
            retention_ratio: Fraction of tokens to retain (default: 0.482 = 48.2%)
            enable_rtd: Whether to enable Recoverable Token Dictionary (default: True)
            alpha_var: Variance penalty for feedback scoring (default: 0.1)
            rtd_bottleneck_ratio: Compression ratio for RTD (default: 4)
        """
        super().__init__()
        self.token_dim = token_dim
        self.semantic_dim = semantic_dim
        self.retention_ratio = retention_ratio
        self.enable_rtd = enable_rtd
        
        # Phase 1: Semantic Abstraction Module (SAM)
        self.sam = SemanticAbstractionModule(
            input_dim=token_dim,
            output_dim=semantic_dim
        )
        
        # Phase 2: Feedback-Guided Compression Scoring (FGCS)
        self.scoring = FeedbackGuidedScoring(
            semantic_dim=semantic_dim,
            alpha_var=alpha_var
        )
        
        # Phase 3: Recoverable Token Dictionary (RTD)
        if enable_rtd:
            self.rtd = RecoverableTokenDictionary(
                token_dim=token_dim,
                bottleneck_ratio=rtd_bottleneck_ratio
            )
        else:
            self.rtd = None
    
    def forward(
        self,
        tokens: torch.Tensor,
        modality: str = 'image',
        spatial_shape: Optional[Tuple[int, int]] = None,
        task_loss: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        retention_ratio: Optional[float] = None,
        inference_mode: str = 'fast',  # 'fast', 'hybrid', 'adaptive'
    ) -> Dict[str, torch.Tensor]:
        """
        DeFT forward pass.
        
        Args:
            tokens: (batch_size, num_tokens, token_dim) input tokens
            modality: 'image', 'video', or 'text'
            spatial_shape: Optional (height, width) for spatial modalities
            task_loss: Optional scalar task loss for feedback scoring (training only)
            logits: Optional (batch_size, num_tokens, num_classes) for RTD reconstruction
            retention_ratio: Override default retention ratio
            inference_mode: 'fast' (retained only), 'hybrid' (retained + reconstructed), 'adaptive'
        
        Returns:
            Dictionary containing:
                - retained_tokens: (batch_size, k, token_dim) selected tokens
                - retained_indices: (batch_size, k) indices of retained tokens
                - reconstructed_tokens: (batch_size, |X~|, token_dim) reconstructed tokens (if RTD enabled)
                - reconstructed_indices: (batch_size, |X~|) indices of reconstructed tokens
                - scores: (batch_size, num_tokens) importance scores
                - scoring_info: Dict with component scores and mixture weights
        """
        batch_size, num_tokens, _ = tokens.shape
        retention_ratio = retention_ratio or self.retention_ratio
        k = int(retention_ratio * num_tokens)
        k = max(1, min(k, num_tokens))  # Ensure 1 <= k <= num_tokens
        
        # Phase 1: Semantic Abstraction
        z = self.sam(tokens)  # (batch_size, num_tokens, semantic_dim)
        
        # Phase 2: Hybrid Scoring
        scores, scoring_info = self.scoring(
            z=z,
            task_loss=task_loss,
            modality=modality,
            spatial_shape=spatial_shape,
            training=self.training
        )  # (batch_size, num_tokens)
        
        # Phase 3: Top-k Selection
        _, topk_indices = torch.topk(scores, k=k, dim=1)  # (batch_size, k)
        
        # Gather retained tokens
        batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1).expand(-1, k)
        retained_tokens = tokens[batch_indices, topk_indices]  # (batch_size, k, token_dim)
        
        # Phase 4: Conditional Reconstruction (if RTD enabled and not fast path)
        reconstructed_tokens = None
        reconstructed_indices = None
        
        if self.enable_rtd and self.rtd is not None and inference_mode != 'fast':
            # Get pruned token indices
            all_indices = torch.arange(num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            pruned_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=tokens.device)
            pruned_mask[batch_indices, topk_indices] = False
            pruned_indices = all_indices[pruned_mask].reshape(batch_size, -1)
            
            # Get pruned tokens
            pruned_batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1).expand(-1, pruned_indices.shape[1])
            pruned_tokens = tokens[pruned_batch_indices, pruned_indices]  # (batch_size, num_pruned, token_dim)
            
            # Get corresponding abstracted tokens and logits for pruned tokens
            pruned_z = z[pruned_batch_indices, pruned_indices]  # (batch_size, num_pruned, semantic_dim)
            pruned_logits = None
            if logits is not None:
                pruned_logits = logits[pruned_batch_indices, pruned_indices]  # (batch_size, num_pruned, num_classes)
            
            # RTD reconstruction
            rtd_output = self.rtd(
                tokens=pruned_tokens,
                z=pruned_z,
                logits=pruned_logits,
                threshold=0.5
            )
            
            # Filter reconstructed tokens based on mask
            reconstruct_mask = rtd_output['reconstruct_mask']  # (batch_size, num_pruned)
            num_reconstructed = reconstruct_mask.sum(dim=1)  # (batch_size,)
            
            # For simplicity, we'll return all reconstructed tokens
            # In practice, you might want to filter by mask
            reconstructed_tokens = rtd_output['reconstructed_tokens']
            reconstructed_indices = pruned_indices
        
        return {
            'retained_tokens': retained_tokens,
            'retained_indices': topk_indices,
            'reconstructed_tokens': reconstructed_tokens,
            'reconstructed_indices': reconstructed_indices,
            'scores': scores,
            'scoring_info': scoring_info,
            'retention_ratio': retention_ratio,
            'num_retained': k,
        }
    
    def get_inference_path(self, tokens, scores, k, logits=None, z=None):
        """
        Determine inference path based on uncertainty and confidence.
        
        Args:
            tokens: (batch_size, num_tokens, token_dim) input tokens
            scores: (batch_size, num_tokens) importance scores
            k: Number of tokens to retain
            logits: Optional (batch_size, num_tokens, num_classes) for uncertainty
            z: Optional (batch_size, num_tokens, semantic_dim) abstracted tokens
        
        Returns:
            path_type: 'fast', 'hybrid', or 'adaptive'
        """
        if not self.enable_rtd or self.rtd is None or logits is None:
            return 'fast'
        
        # Compute average uncertainty and confidence
        uncertainty = self.rtd.compute_uncertainty(logits)  # (batch_size, num_tokens)
        confidence = self.rtd.compute_confidence(logits)  # (batch_size, num_tokens)
        
        avg_uncertainty = uncertainty.mean().item()
        avg_confidence = confidence.mean().item()
        
        # Adaptive decision: use hybrid path if uncertainty is high or confidence is low
        if avg_uncertainty > 0.5 or avg_confidence < 0.7:
            return 'hybrid'
        else:
            return 'fast'
    
    def compute_selection_entropy(self, scores):
        """
        Compute selection entropy: H(S) = -Σ_i p_i log p_i
        where p_i = s_i / Σ_j s_j
        
        Args:
            scores: (batch_size, num_tokens) importance scores
        
        Returns:
            entropy: (batch_size,) selection entropy in bits
        """
        # Convert scores to probabilities
        probs = F.softmax(scores, dim=1)  # (batch_size, num_tokens)
        
        # Compute entropy: H = -Σ p log p
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1)  # (batch_size,)
        
        return entropy

