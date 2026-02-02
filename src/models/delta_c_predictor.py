"""
δC Predictor for Stage 2-C: Subjectivity Space Learning

Predicts subjectivity deviation δC given (I_init, A_init, I_edit)
Enables exploration of multiple valid C_anchor interpretations

Based on System Specification v2.1, Section 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DeltaCPredictor(nn.Module):
    """
    δC Predictor: (I_init, A_init, I_edit) → δC

    Purpose:
    - Learn bounded subjectivity space around C_prior
    - Enable UI-based exploration of valid interpretations
    - Predict personalized coupling variations

    Architecture:
    - Uses ImageBind tokens (like Soft Prior)
    - Token-wise prediction: each visual token → per-token δC
    - L2 ball projection: ||δC|| < ε_prior enforced

    Relationship:
    C_anchor = C_prior + δC
    where ||δC||² < ε_prior
    """

    def __init__(
        self,
        visual_dim: int = 1024,  # ImageBind dimension
        audio_dim: int = 1024,   # ImageBind dimension
        hidden_dim: int = 512,
        N_v: int = 256,
        num_heads: int = 6,
        epsilon_prior: float = 0.5,
    ):
        """
        Args:
            visual_dim: Dimension of ImageBind visual tokens
            audio_dim: Dimension of ImageBind audio tokens
            hidden_dim: Hidden layer dimension
            N_v: Number of visual tokens
            num_heads: Number of control heads
            epsilon_prior: Maximum allowed δC magnitude (L2 ball radius)
        """
        super().__init__()

        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.N_v = N_v
        self.num_heads = num_heads
        self.epsilon_prior = epsilon_prior

        # ImageBind encoder (shared with Soft Prior)
        self.imagebind = self._load_imagebind()

        # Per-token MLP for δC prediction
        # Input: [v_init_token, v_edit_token, a_global, Δv_token]
        per_token_input_dim = visual_dim + visual_dim + audio_dim + visual_dim

        self.per_token_mlp = nn.Sequential(
            nn.Linear(per_token_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_heads),
        )

        # NOTE: Can add cross-token attention here for more sophisticated modeling:
        # self.token_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        # This would allow tokens to attend to each other before final prediction

        # Register epsilon as buffer
        self.register_buffer('epsilon_prior', torch.tensor(epsilon_prior))

        # Projection layer for ImageBind patch tokens (1280 -> visual_dim)
        # Fixed seed for reproducibility
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        self._dc_vision_proj = nn.Linear(1280, visual_dim, bias=False)
        nn.init.xavier_uniform_(self._dc_vision_proj.weight)
        self._dc_vision_proj.requires_grad_(False)
        torch.random.set_rng_state(rng_state)

    def _load_imagebind(self):
        """
        Load ImageBind model (same as Soft Prior)

        Returns:
            ImageBind model or fallback
        """
        from src.utils.model_loaders import load_imagebind_or_clip

        model, self._is_imagebind = load_imagebind_or_clip(freeze=True)
        return model

    def encode_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual tokens using ImageBind or CLIP fallback.

        When using ImageBind, extracts 256 spatial patch tokens from the ViT trunk
        (bypassing the CLS-selection head) for spatially-aware token representations.

        Args:
            images: (B, 3, H, W)

        Returns:
            v_tokens: (B, N_v, D) visual tokens
        """
        B = images.shape[0]

        if isinstance(self.imagebind, nn.Identity):
            return torch.randn(B, self.N_v, self.visual_dim, device=images.device)

        with torch.no_grad():
            if getattr(self, '_is_imagebind', False):
                try:
                    from src.utils.model_loaders import (
                        imagebind_preprocess_vision,
                        imagebind_extract_patch_tokens,
                    )

                    # Preprocess: undo ImageNet norm, resize to 224, apply ImageBind norm
                    images_ib = imagebind_preprocess_vision(images)

                    # Extract 256 spatial patch tokens from trunk
                    v_tokens = imagebind_extract_patch_tokens(
                        self.imagebind, images_ib, target_n_tokens=self.N_v
                    )  # (B, N_v, embed_dim=1280)

                    # Project 1280-dim patch tokens to visual_dim
                    v_tokens = self._dc_vision_proj(v_tokens)

                except Exception as e:
                    print(f"ImageBind vision failed: {e}")
                    v_tokens = torch.randn(B, self.N_v, self.visual_dim, device=images.device)
            else:
                # CLIP fallback
                try:
                    v_features = self.imagebind.encode_image(images)  # (B, D)
                    v_tokens = v_features.unsqueeze(1).expand(-1, self.N_v, -1)
                except Exception:
                    v_tokens = torch.randn(B, self.N_v, self.visual_dim, device=images.device)

        return v_tokens

    def encode_audio_global(
        self,
        audios: torch.Tensor,
        waveforms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract audio global embedding using ImageBind or mel-based fallback.

        When waveforms are provided and ImageBind is available, converts to
        Kaldi fbank format (128 mel bins, 204 frames) for proper audio encoding.

        Args:
            audios: (B, 1, T, F) mel spectrograms (fallback)
            waveforms: (B, T_samples) raw waveforms at 16kHz (optional, for ImageBind)

        Returns:
            a_global: (B, D) global audio embedding
        """
        B = audios.shape[0]

        if isinstance(self.imagebind, nn.Identity):
            return torch.randn(B, self.audio_dim, device=audios.device)

        with torch.no_grad():
            if getattr(self, '_is_imagebind', False) and waveforms is not None:
                try:
                    from src.utils.model_loaders import (
                        waveform_to_imagebind_audio,
                        imagebind_extract_audio_embedding,
                    )

                    # Convert waveforms to ImageBind format
                    audio_inputs = []
                    for b in range(B):
                        ib_audio = waveform_to_imagebind_audio(waveforms[b])
                        audio_inputs.append(ib_audio)

                    audio_batch = torch.stack(audio_inputs, dim=0).to(audios.device)
                    # (B, num_clips, 1, 128, 204)

                    a_global = imagebind_extract_audio_embedding(
                        self.imagebind, audio_batch
                    )  # (B, 1024)

                except Exception as e:
                    print(f"ImageBind audio failed: {e}")
                    a_global = torch.randn(B, self.audio_dim, device=audios.device)
            else:
                # CLIP/mel fallback - use mel global pooling as proxy
                mel = audios.squeeze(1)  # (B, T, F)
                a_global = mel.mean(dim=1)  # (B, F)
                # Project to audio_dim if needed
                if a_global.shape[-1] != self.audio_dim:
                    if not hasattr(self, '_audio_proj'):
                        self._audio_proj = nn.Linear(a_global.shape[-1], self.audio_dim).to(audios.device)
                        nn.init.xavier_uniform_(self._audio_proj.weight)
                        self._audio_proj.requires_grad_(False)
                    a_global = self._audio_proj(a_global)

        return a_global

    def forward(
        self,
        I_init: torch.Tensor,
        A_init: torch.Tensor,
        I_edit: torch.Tensor,
        waveforms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict δC for given triplet (token-wise)

        Args:
            I_init: (B, 3, H, W) initial image
            A_init: (B, 1, T, F) initial audio mel spectrogram
            I_edit: (B, 3, H, W) edited image
            waveforms: (B, T_samples) raw waveforms at 16kHz (optional, for ImageBind audio)

        Returns:
            δC: (B, N_v, num_heads) predicted coupling deviation
                with ||δC|| < ε_prior enforced
        """
        B = I_init.shape[0]

        # Extract ImageBind tokens
        v_init_tokens = self.encode_visual_tokens(I_init)  # (B, N_v, D)
        v_edit_tokens = self.encode_visual_tokens(I_edit)  # (B, N_v, D)
        a_global = self.encode_audio_global(A_init, waveforms=waveforms)  # (B, D)

        # Compute per-token visual delta
        Δv_tokens = v_edit_tokens - v_init_tokens  # (B, N_v, D)

        # Broadcast audio to match tokens
        a_global_expanded = a_global.unsqueeze(1).expand(-1, self.N_v, -1)  # (B, N_v, D)

        # Concatenate per-token features
        # [v_init_token, v_edit_token, a_global, Δv_token]
        token_features = torch.cat([
            v_init_tokens,      # (B, N_v, D)
            v_edit_tokens,      # (B, N_v, D)
            a_global_expanded,  # (B, N_v, D)
            Δv_tokens,          # (B, N_v, D)
        ], dim=-1)  # (B, N_v, 4*D)

        # Per-token prediction
        δC_raw = self.per_token_mlp(token_features)  # (B, N_v, num_heads)

        # L2 Ball Projection: ||δC|| < ε_prior
        δC = self._project_to_l2_ball(δC_raw)

        return δC

    def _project_to_l2_ball(self, δC_raw: torch.Tensor) -> torch.Tensor:
        """
        Project δC to L2 ball of radius ε_prior

        Args:
            δC_raw: (B, N_v, num_heads) raw prediction

        Returns:
            δC: (B, N_v, num_heads) projected to ||δC|| < ε_prior
        """
        B = δC_raw.shape[0]

        # Compute L2 norm per sample (flatten spatial dimensions)
        δC_flat = δC_raw.reshape(B, -1)  # (B, N_v * num_heads)
        δC_norm = torch.norm(δC_flat, dim=-1, keepdim=True)  # (B, 1)

        # Compute scaling factor
        # If ||δC|| > ε, scale down to ε
        # If ||δC|| <= ε, keep as is
        scale = torch.clamp(
            self.epsilon_prior / (δC_norm + 1e-8),
            max=1.0
        )  # (B, 1)

        # Apply scaling
        δC = δC_raw * scale.view(B, 1, 1)

        return δC

    def compute_prior_distance(self, δC: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 distance from prior

        Args:
            δC: (B, N_v, num_heads)

        Returns:
            distance: (B,) L2 norm of δC
        """
        B = δC.shape[0]
        δC_flat = δC.reshape(B, -1)
        return torch.norm(δC_flat, dim=-1)

    def sample_subjectivity(
        self,
        I_init: torch.Tensor,
        A_init: torch.Tensor,
        I_edit: torch.Tensor,
        n_samples: int = 5,
        noise_std: float = 0.1,
        waveforms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample multiple δC variations for exploration

        Args:
            I_init: (B, 3, H, W)
            A_init: (B, 1, T, F)
            I_edit: (B, 3, H, W)
            n_samples: Number of samples
            noise_std: Gaussian noise std for sampling
            waveforms: (B, T_samples) raw waveforms (optional, for ImageBind)

        Returns:
            δC_samples: (n_samples, B, N_v, num_heads)
        """
        # Base prediction
        δC_base = self.forward(I_init, A_init, I_edit, waveforms=waveforms)  # (B, N_v, num_heads)

        # Sample variations
        δC_samples = []
        for _ in range(n_samples):
            # Add Gaussian noise
            noise = torch.randn_like(δC_base) * noise_std
            δC_sample = δC_base + noise

            # Project to L2 ball
            δC_sample = self._project_to_l2_ball(δC_sample)

            δC_samples.append(δC_sample)

        return torch.stack(δC_samples, dim=0)


# For testing
if __name__ == "__main__":
    print("Testing δC Predictor with token-wise prediction...\n")

    # Test δC predictor
    predictor = DeltaCPredictor(
        visual_dim=1024,  # ImageBind dimension
        audio_dim=1024,
        hidden_dim=512,
        N_v=256,
        num_heads=6,
        epsilon_prior=0.5,
    )

    # Dummy inputs
    B = 4
    I_init = torch.randn(B, 3, 224, 224)  # ImageBind uses 224x224
    I_edit = torch.randn(B, 3, 224, 224)
    A_init = torch.randn(B, 1, 800, 64)

    # Predict δC
    δC = predictor(I_init, A_init, I_edit)

    print(f"Input shapes:")
    print(f"  I_init: {I_init.shape}")
    print(f"  I_edit: {I_edit.shape}")
    print(f"  A_init: {A_init.shape}")
    print(f"\nOutput δC: {δC.shape}")
    print(f"Expected: ({B}, 256, 6)")

    # Compute distance
    dist = predictor.compute_prior_distance(δC)
    print(f"\nPrior distance (L2 norm):")
    print(f"  {dist}")
    print(f"  All < epsilon_prior (0.5): {(dist < predictor.epsilon_prior).all()}")
    print(f"  Max distance: {dist.max():.4f}")

    # Sample variations
    δC_samples = predictor.sample_subjectivity(I_init, A_init, I_edit, n_samples=5)
    print(f"\nSampled δC: {δC_samples.shape}")
    print(f"Expected: (5, {B}, 256, 6)")

    # Check all samples are within ball
    sample_dists = torch.stack([
        predictor.compute_prior_distance(δC_samples[i])
        for i in range(5)
    ])
    print(f"\nSample distances:")
    print(f"  Mean: {sample_dists.mean():.4f}")
    print(f"  Std: {sample_dists.std():.4f}")
    print(f"  All < epsilon: {(sample_dists < predictor.epsilon_prior).all()}")

    print("\n✓ Test complete!")
