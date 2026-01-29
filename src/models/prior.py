"""
Prior construction module for C_anchor estimation

Implements:
- Hard Prior: Type-II rules (structural correspondences)
- Soft Prior: ImageBind-based multimodal embeddings
- Combined Prior: Weighted combination of hard and soft priors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class HardPrior:
    """
    Hard Prior based on Type-II rules
    Encodes structural correspondences between visual and audio features
    """

    def __init__(self, rules: List[Dict], num_heads: int = 6):
        """
        Args:
            rules: List of rule dictionaries, each containing:
                - visual_feature: str
                - audio_feature: str
                - correlation: float (+1 or -1)
                - target_head: str
                - weight: float
            num_heads: Number of control heads
        """
        self.rules = rules
        self.num_heads = num_heads
        self.head_names = ["rhythm", "harmony", "energy", "timbre", "space", "texture"]

        # Build weight matrix W_hard
        self.W_hard = self._build_weight_matrix()

    def _build_weight_matrix(self) -> torch.Tensor:
        """
        Build weight matrix for hard prior
        W_hard[h] = sum of weights for rules targeting head h
        """
        W = torch.zeros(self.num_heads)

        for rule in self.rules:
            target_head = rule['target_head']
            if target_head in self.head_names:
                h_idx = self.head_names.index(target_head)
                W[h_idx] += rule['weight']

        return W

    def compute_saliency(
        self,
        visual_tokens: torch.Tensor,
        visual_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute saliency for each visual token based on rules

        Args:
            visual_tokens: (B, N_v, D) visual token embeddings
            visual_features: Optional dict of extracted visual features

        Returns:
            saliency: (B, N_v) saliency scores
        """
        B, N_v, D = visual_tokens.shape

        # For MVP: uniform saliency
        # In full version, compute based on visual features
        saliency = torch.ones(B, N_v, device=visual_tokens.device)

        if visual_features is not None:
            # TODO: Implement rule-based saliency
            # For each rule, check if visual feature changed significantly
            pass

        return saliency

    def get_weight_matrix(self) -> torch.Tensor:
        """Get the weight matrix"""
        return self.W_hard


class SoftPrior(nn.Module):
    """
    Soft Prior based on ImageBind multimodal embeddings
    Uses pretrained ImageBind to estimate cross-modal coupling
    """

    def __init__(
        self,
        model_name: str = "imagebind",
        num_heads: int = 6,
        head_dim: int = 512,
        freeze: bool = True,
    ):
        """
        Args:
            model_name: Name of pretrained model
            num_heads: Number of control heads
            head_dim: Dimension of embeddings
            freeze: Whether to freeze pretrained weights
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim

        # For MVP: Use CLIP as proxy for ImageBind
        # TODO: Replace with actual ImageBind when available
        try:
            import open_clip
            self.vision_model, _, self.image_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            self.audio_model = self.vision_model  # Placeholder
        except ImportError:
            print("Warning: open_clip not available, using dummy model")
            self.vision_model = nn.Identity()
            self.audio_model = nn.Identity()

        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in self.audio_model.parameters():
                param.requires_grad = False

        # Head queries (learnable or fixed)
        self.head_queries = nn.Parameter(
            torch.randn(num_heads, head_dim),
            requires_grad=False,  # Fixed for MVP
        )

    def initialize_head_queries(self, text_prompts: Dict[str, str]):
        """
        Initialize head queries using CLIP text embeddings

        Args:
            text_prompts: Dictionary mapping head names to text descriptions
        """
        try:
            import open_clip

            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            text_model = self.vision_model  # CLIP text encoder

            with torch.no_grad():
                for h, (head_name, prompt) in enumerate(text_prompts.items()):
                    if h < self.num_heads:
                        tokens = tokenizer([prompt])
                        text_features = text_model.encode_text(tokens)
                        self.head_queries[h] = text_features.squeeze()

        except Exception as e:
            print(f"Failed to initialize head queries: {e}")
            # Keep random initialization

    def extract_tokens(
        self,
        images: torch.Tensor,
        audios: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual and audio tokens using pretrained model

        Args:
            images: (B, 3, H, W) images
            audios: (B, 1, T, F) mel spectrograms

        Returns:
            v_tokens: (B, N_v, D) visual tokens
            a_tokens: (B, N_a, D) audio tokens
        """
        B = images.shape[0]

        with torch.no_grad():
            # Extract visual features
            # Note: This is simplified; actual implementation depends on model
            try:
                v_features = self.vision_model.encode_image(images)
                # Expand to tokens (simplified)
                N_v = 256  # Fixed number of visual tokens
                v_tokens = v_features.unsqueeze(1).expand(-1, N_v, -1)
            except:
                # Dummy implementation
                v_tokens = torch.randn(B, 256, self.head_dim, device=images.device)

            # Extract audio features (placeholder)
            # TODO: Implement proper audio encoding
            N_a = 128
            a_tokens = torch.randn(B, N_a, self.head_dim, device=audios.device)

        return v_tokens, a_tokens

    def compute_coupling(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft coupling matrix C_soft

        Args:
            v_tokens: (B, N_v, D) visual tokens
            a_tokens: (B, N_a, D) audio tokens

        Returns:
            C_soft: (B, N_v, num_heads) coupling matrix
        """
        B, N_v, D = v_tokens.shape
        N_a = a_tokens.shape[1]

        # Step 1: Compute similarity matrix
        # Sim = (B, N_v, N_a)
        v_norm = F.normalize(v_tokens, dim=-1)
        a_norm = F.normalize(a_tokens, dim=-1)
        sim = torch.bmm(v_norm, a_norm.transpose(1, 2))

        # Step 2: Head pooling
        # a_heads = (B, num_heads, N_a)
        Q_norm = F.normalize(self.head_queries, dim=-1)  # (num_heads, D)
        a_norm_t = a_norm.transpose(1, 2)  # (B, D, N_a)
        a_heads = torch.matmul(Q_norm, a_norm_t)  # (num_heads, B, N_a)
        a_heads = a_heads.permute(1, 0, 2)  # (B, num_heads, N_a)

        # Step 3: Coupling
        # C_soft = Sim @ a_heads^T
        # C_soft = (B, N_v, num_heads)
        C_soft = torch.bmm(sim, a_heads.transpose(1, 2))

        # Step 4: Normalize (softmax over heads)
        C_soft = F.softmax(C_soft, dim=-1)

        return C_soft

    def forward(
        self,
        images: torch.Tensor,
        audios: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass to compute C_soft

        Args:
            images: (B, 3, H, W)
            audios: (B, 1, T, F)

        Returns:
            C_soft: (B, N_v, num_heads)
        """
        v_tokens, a_tokens = self.extract_tokens(images, audios)
        C_soft = self.compute_coupling(v_tokens, a_tokens)
        return C_soft


class PriorEstimator(nn.Module):
    """
    Combined prior estimator that merges hard and soft priors
    """

    def __init__(
        self,
        hard_prior: HardPrior,
        soft_prior: SoftPrior,
        alpha: float = 0.3,
        entropy_min: float = 0.5,
        sparsity_max: float = 5.0,
    ):
        """
        Args:
            hard_prior: HardPrior instance
            soft_prior: SoftPrior instance
            alpha: Balance between hard and soft (higher = more hard)
            entropy_min: Minimum entropy constraint
            sparsity_max: Maximum sparsity (L1 norm) constraint
        """
        super().__init__()

        self.hard_prior = hard_prior
        self.soft_prior = soft_prior
        self.alpha = alpha
        self.entropy_min = entropy_min
        self.sparsity_max = sparsity_max

    def forward(
        self,
        images: torch.Tensor,
        audios: torch.Tensor,
        delta_C: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Estimate C_anchor = C_prior + delta_C

        Args:
            images: (B, 3, H, W)
            audios: (B, 1, T, F)
            delta_C: (B, N_v, num_heads) optional perturbation

        Returns:
            C_anchor: (B, N_v, num_heads) coupling matrix
        """
        B = images.shape[0]

        # Compute C_soft
        C_soft = self.soft_prior(images, audios)
        N_v = C_soft.shape[1]

        # Compute hard prior contribution
        W_hard = self.hard_prior.get_weight_matrix().to(images.device)
        v_tokens, _ = self.soft_prior.extract_tokens(images, audios)
        saliency = self.hard_prior.compute_saliency(v_tokens)

        # Broadcast hard prior
        # C_hard = saliency @ W_hard^T
        # C_hard: (B, N_v, num_heads)
        saliency_expanded = saliency.unsqueeze(-1)  # (B, N_v, 1)
        W_hard_expanded = W_hard.unsqueeze(0).unsqueeze(0)  # (1, 1, num_heads)
        C_hard = saliency_expanded * W_hard_expanded

        # Normalize C_hard
        C_hard = C_hard / (C_hard.sum(dim=-1, keepdim=True) + 1e-8)

        # Combine
        C_prior = (1 - self.alpha) * C_soft + self.alpha * C_hard

        # Apply delta_C if provided
        if delta_C is not None:
            C_anchor = C_prior + delta_C
        else:
            C_anchor = C_prior

        # Apply constraints
        C_anchor = self._apply_constraints(C_anchor)

        return C_anchor

    def _apply_constraints(self, C: torch.Tensor) -> torch.Tensor:
        """
        Apply entropy and sparsity constraints

        Args:
            C: (B, N_v, num_heads) coupling matrix

        Returns:
            Constrained coupling matrix
        """
        # Ensure non-negative
        C = torch.clamp(C, min=0)

        # Normalize to sum to 1 across heads for each token
        C = C / (C.sum(dim=-1, keepdim=True) + 1e-8)

        # TODO: Implement entropy and sparsity projection
        # For MVP, just return normalized version

        return C

    def estimate_prior_only(
        self,
        images: torch.Tensor,
        audios: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate C_prior without delta_C (for Stage 2 MVP)
        """
        return self.forward(images, audios, delta_C=None)
