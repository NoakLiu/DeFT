"""
Feedback-Guided Compression Scoring (FGCS)
Combines semantic saliency, feedback-aware importance, and group interaction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedbackGuidedScoring(nn.Module):
    """
    Computes token importance scores using three signals:
    1. Semantic saliency (s_sem)
    2. Feedback-aware importance (s_fb) - training only
    3. Group-aware scoring (s_group)
    """
    
    def __init__(self, semantic_dim=512, alpha_var=0.1, init_weights=[0.4, 0.4, 0.2]):
        """
        Args:
            semantic_dim: Dimension of semantic space (default: 512)
            alpha_var: Variance penalty coefficient for feedback scoring (default: 0.1)
            init_weights: Initial mixture weights [β1, β2, β3] for [semantic, feedback, group]
        """
        super().__init__()
        self.semantic_dim = semantic_dim
        self.alpha_var = alpha_var
        
        # Semantic saliency MLP: 3-layer MLP
        self.mlp_score = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, 1)
        )
        
        # Learnable mixture weights β1, β2, β3
        self.mixture_logits = nn.Parameter(torch.tensor([
            torch.log(torch.tensor(init_weights[0] / (1 - init_weights[0]))),
            torch.log(torch.tensor(init_weights[1] / (1 - init_weights[1]))),
            torch.log(torch.tensor(init_weights[2] / (1 - init_weights[2])))
        ]))
        
        # Reconstruction gate parameters (for RTD)
        self.reconstruct_gate = nn.Sequential(
            nn.Linear(semantic_dim + 2, 1),  # [zi, uncertainty, confidence]
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize scoring network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def compute_semantic_saliency(self, z):
        """
        Compute semantic saliency: s_sem_i = σ(Ws · MLPscore(zi) + bs)
        
        Args:
            z: (batch_size, num_tokens, semantic_dim) abstracted tokens
        
        Returns:
            s_sem: (batch_size, num_tokens) semantic saliency scores
        """
        s_sem = self.mlp_score(z)  # (batch_size, num_tokens, 1)
        s_sem = torch.sigmoid(s_sem.squeeze(-1))  # (batch_size, num_tokens)
        return s_sem
    
    def compute_feedback_importance(self, z, task_loss, training=True):
        """
        Compute feedback-aware importance: s_fb_i = ||∇zi L_task||_2 · exp(-α_var · Var(∇zi L_task))
        
        Args:
            z: (batch_size, num_tokens, semantic_dim) abstracted tokens
            task_loss: Scalar task loss
            training: Whether in training mode
        
        Returns:
            s_fb: (batch_size, num_tokens) feedback importance scores
        """
        if not training or task_loss is None:
            # At inference, feedback scoring is not used
            return torch.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
        
        # Compute gradients w.r.t. z
        if z.requires_grad:
            try:
                grad_z = torch.autograd.grad(
                    outputs=task_loss,
                    inputs=z,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True
                )[0]
                
                if grad_z is None:
                    return torch.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
            except RuntimeError:
                # If gradient computation fails, return zeros
                return torch.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
        else:
            # If z doesn't require grad, return zeros
            return torch.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
        
        # Compute gradient magnitude: ||∇zi L_task||_2
        grad_magnitude = torch.norm(grad_z, dim=-1)  # (batch_size, num_tokens)
        
        # Compute gradient variance across batch (if batch_size > 1)
        if grad_z.shape[0] > 1:
            grad_variance = torch.var(grad_z, dim=0, keepdim=True)  # (1, num_tokens, semantic_dim)
            grad_variance = torch.mean(grad_variance, dim=-1).squeeze(0)  # (num_tokens,)
            grad_variance = grad_variance.unsqueeze(0).expand_as(grad_magnitude)  # (batch_size, num_tokens)
        else:
            grad_variance = torch.zeros_like(grad_magnitude)
        
        # Feedback score: ||∇zi||_2 · exp(-α_var · Var(∇zi))
        s_fb = grad_magnitude * torch.exp(-self.alpha_var * grad_variance)
        
        return s_fb
    
    def compute_group_scoring(self, z, s_sem, modality='image', spatial_shape=None):
        """
        Compute group-aware scoring: s_group_i = (1/|Gi|) Σ_j∈Gi sim(zi, zj) · s_sem_j
        
        Args:
            z: (batch_size, num_tokens, semantic_dim) abstracted tokens
            s_sem: (batch_size, num_tokens) semantic saliency scores
            modality: 'image', 'video', or 'text'
            spatial_shape: For images/video, (height, width) of spatial layout
        
        Returns:
            s_group: (batch_size, num_tokens) group interaction scores
        """
        batch_size, num_tokens, dim = z.shape
        
        # Compute similarity matrix: sim(zi, zj) = (zi · zj) / (||zi||_2 ||zj||_2)
        z_norm = F.normalize(z, p=2, dim=-1)  # (batch_size, num_tokens, dim)
        similarity = torch.bmm(z_norm, z_norm.transpose(1, 2))  # (batch_size, num_tokens, num_tokens)
        
        # Build neighborhood groups Gi based on modality
        if modality == 'image' and spatial_shape is not None:
            # 8-connected spatial neighbors for images
            h, w = spatial_shape
            neighbors = self._get_spatial_neighbors_8connected(num_tokens, h, w)
        elif modality == 'video' and spatial_shape is not None:
            # Temporal window ±2 for video
            h, w = spatial_shape
            neighbors = self._get_temporal_neighbors(num_tokens, h, w, window=2)
        else:
            # TopK(k=5) semantic neighbors for text
            neighbors = self._get_semantic_neighbors(similarity, k=5)
        
        # Compute group scores: s_group_i = (1/|Gi|) Σ_j∈Gi sim(zi, zj) · s_sem_j
        s_group = torch.zeros_like(s_sem)
        for i in range(num_tokens):
            if i in neighbors:
                neighbor_indices = neighbors[i]
                if len(neighbor_indices) > 0:
                    sim_scores = similarity[:, i, neighbor_indices]  # (batch_size, |Gi|)
                    sem_scores = s_sem[:, neighbor_indices]  # (batch_size, |Gi|)
                    s_group[:, i] = torch.mean(sim_scores * sem_scores, dim=1)
        
        return s_group
    
    def _get_spatial_neighbors_8connected(self, num_tokens, h, w):
        """Get 8-connected spatial neighbors for image tokens"""
        neighbors = {}
        for i in range(num_tokens):
            row = i // w
            col = i % w
            neighbor_list = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbor_list.append(nr * w + nc)
            neighbors[i] = neighbor_list
        return neighbors
    
    def _get_temporal_neighbors(self, num_tokens, h, w, window=2):
        """Get temporal neighbors for video tokens (window ±2)"""
        neighbors = {}
        num_frames = num_tokens // (h * w)
        for i in range(num_tokens):
            frame_idx = i // (h * w)
            spatial_idx = i % (h * w)
            neighbor_list = []
            for dt in range(-window, window + 1):
                if dt == 0:
                    continue
                nf = frame_idx + dt
                if 0 <= nf < num_frames:
                    neighbor_list.append(nf * (h * w) + spatial_idx)
            neighbors[i] = neighbor_list
        return neighbors
    
    def _get_semantic_neighbors(self, similarity, k=5):
        """Get TopK semantic neighbors for text tokens"""
        batch_size, num_tokens, _ = similarity.shape
        neighbors = {}
        
        # Use first batch item for neighbor selection (or average across batch)
        sim_avg = similarity.mean(dim=0)  # (num_tokens, num_tokens)
        
        for i in range(num_tokens):
            # Get top-k most similar tokens (excluding self)
            sim_scores = sim_avg[i]
            sim_scores[i] = -float('inf')  # Exclude self
            _, topk_indices = torch.topk(sim_scores, k=min(k, num_tokens - 1))
            neighbors[i] = topk_indices.tolist()
        
        return neighbors
    
    def normalize_scores(self, scores):
        """
        Z-score normalization: s̃ = (s - μ) / σ
        
        Args:
            scores: (batch_size, num_tokens) raw scores
        
        Returns:
            normalized_scores: (batch_size, num_tokens) z-score normalized scores
        """
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True) + 1e-8
        return (scores - mean) / std
    
    def forward(self, z, task_loss=None, modality='image', spatial_shape=None, training=True):
        """
        Compute unified importance scores: si = β1 · s̃_sem_i + β2 · s̃_fb_i + β3 · s̃_group_i
        
        Args:
            z: (batch_size, num_tokens, semantic_dim) abstracted tokens
            task_loss: Optional scalar task loss for feedback scoring
            modality: 'image', 'video', or 'text'
            spatial_shape: Optional (height, width) for spatial modalities
            training: Whether in training mode
        
        Returns:
            scores: (batch_size, num_tokens) final importance scores
        """
        # Compute individual scoring components
        s_sem = self.compute_semantic_saliency(z)
        s_fb = self.compute_feedback_importance(z, task_loss, training=training) if task_loss is not None else torch.zeros_like(s_sem)
        s_group = self.compute_group_scoring(z, s_sem, modality=modality, spatial_shape=spatial_shape)
        
        # Z-score normalization
        s_sem_norm = self.normalize_scores(s_sem)
        s_fb_norm = self.normalize_scores(s_fb) if task_loss is not None else torch.zeros_like(s_sem_norm)
        s_group_norm = self.normalize_scores(s_group)
        
        # Compute mixture weights: βk = exp(θk) / Σ_j exp(θj)
        mixture_weights = F.softmax(self.mixture_logits, dim=0)
        
        # Combine scores: si = β1 · s̃_sem_i + β2 · s̃_fb_i + β3 · s̃_group_i
        scores = (mixture_weights[0] * s_sem_norm + 
                 mixture_weights[1] * s_fb_norm + 
                 mixture_weights[2] * s_group_norm)
        
        return scores, {
            'semantic': s_sem,
            'feedback': s_fb,
            'group': s_group,
            'mixture_weights': mixture_weights
        }

