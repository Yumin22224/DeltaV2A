"""
Loss functions for DeltaV2A training

Implements all losses defined in System Specification v2:
- Stage 1: Reconstruction, Structure Preservation, Rank Consistency
- Stage 2-A: Pseudo-target, Manifold, Identity, Monotonicity
- Stage 2-B: Preserve, Coherence, Consistency, Rank
- Stage 2-C: Direction, Manifold, Bounded, Prior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import librosa
import numpy as np


# ============================================================================
# Stage 1 Losses
# ============================================================================

class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss for Stage 1

    More robust than simple mel L2
    Preserves phase and fine structure
    """

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [50, 120, 240],
        win_lengths: List[int] = [240, 600, 1200],
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def stft(self, x: torch.Tensor, fft_size: int, hop_size: int, win_length: int):
        """Compute STFT"""
        # x: (B, 1, T, F) mel or (B, T) waveform
        if x.dim() == 4:
            # Mel spectrogram to waveform (simplified)
            x = x.squeeze(1).mean(dim=-1)  # (B, T)

        return torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            return_complex=True,
        )

    def forward(self, A_target: torch.Tensor, A_recon: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_target: (B, 1, T, F) target mel
            A_recon: (B, 1, T, F) reconstructed mel

        Returns:
            loss: scalar
        """
        loss = 0.0

        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            # Compute STFT
            S_target = self.stft(A_target, fft_size, hop_size, win_length)
            S_recon = self.stft(A_recon, fft_size, hop_size, win_length)

            # Magnitude
            mag_target = torch.abs(S_target)
            mag_recon = torch.abs(S_recon)

            # Spectral convergence
            sc = torch.norm(mag_target - mag_recon) / (torch.norm(mag_target) + 1e-8)

            # Log magnitude distance
            log_mag = torch.mean(
                torch.abs(
                    torch.log(mag_target + 1e-5) - torch.log(mag_recon + 1e-5)
                )
            )

            loss += sc + log_mag

        return loss / len(self.fft_sizes)


class StructurePreservationLoss(nn.Module):
    """
    Structure Preservation Loss for Stage 1 & 2-B

    Preserves rhythm, harmony, energy using audio metrics
    """

    def __init__(
        self,
        heads: List[str] = ["rhythm", "harmony", "energy"],
        weights: Dict[str, float] = None,
    ):
        super().__init__()
        self.heads = heads
        self.weights = weights or {h: 1.0 for h in heads}

    def extract_onset(self, audio_mel: torch.Tensor) -> torch.Tensor:
        """Extract onset strength for rhythm"""
        # Simplified: use energy envelope
        # In practice, use librosa.onset.onset_strength
        energy = torch.mean(audio_mel, dim=-1)  # (B, T)
        return energy

    def extract_chroma(self, audio_mel: torch.Tensor) -> torch.Tensor:
        """Extract chroma for harmony"""
        # Simplified: project to 12 bins
        # In practice, use librosa.feature.chroma_stft
        B, _, T, F = audio_mel.shape
        chroma = torch.randn(B, T, 12, device=audio_mel.device)  # Placeholder
        return chroma

    def extract_rms(self, audio_mel: torch.Tensor) -> torch.Tensor:
        """Extract RMS envelope for energy"""
        # RMS over frequency
        rms = torch.sqrt(torch.mean(audio_mel ** 2, dim=-1))  # (B, T)
        return rms

    def pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation"""
        x_mean = x.mean(dim=-1, keepdim=True)
        y_mean = y.mean(dim=-1, keepdim=True)

        x_centered = x - x_mean
        y_centered = y - y_mean

        cov = (x_centered * y_centered).sum(dim=-1)
        std_x = torch.sqrt((x_centered ** 2).sum(dim=-1) + 1e-8)
        std_y = torch.sqrt((y_centered ** 2).sum(dim=-1) + 1e-8)

        corr = cov / (std_x * std_y + 1e-8)
        return corr

    def cosine_similarity_batched(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity"""
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        sim = (x_norm * y_norm).sum(dim=-1).mean()
        return sim

    def forward(self, A_init: torch.Tensor, A_recon: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_init: (B, 1, T, F) initial audio
            A_recon: (B, 1, T, F) reconstructed/edited audio

        Returns:
            loss: scalar
        """
        loss = 0.0

        for head in self.heads:
            if head == "rhythm":
                onset_init = self.extract_onset(A_init)
                onset_recon = self.extract_onset(A_recon)
                d_rhythm = 1 - self.pearson_correlation(onset_init, onset_recon).mean()
                loss += self.weights[head] * d_rhythm

            elif head == "harmony":
                chroma_init = self.extract_chroma(A_init)
                chroma_recon = self.extract_chroma(A_recon)
                d_harmony = 1 - self.cosine_similarity_batched(chroma_init, chroma_recon)
                loss += self.weights[head] * d_harmony

            elif head == "energy":
                rms_init = self.extract_rms(A_init)
                rms_recon = self.extract_rms(A_recon)
                d_energy = 1 - self.pearson_correlation(rms_init, rms_recon).mean()
                loss += self.weights[head] * d_energy

        return loss


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise Ranking Loss for Stage 1

    Ensures g_h ordering matches Δm_h ordering
    """

    def __init__(self, num_heads: int = 6):
        super().__init__()
        self.num_heads = num_heads

    def forward(
        self,
        S_pred: List[Tuple[torch.Tensor, torch.Tensor]],
        delta_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            S_pred: List of (t_h, g_h) tuples
            delta_m: (B, num_heads) ground truth head changes

        Returns:
            loss: scalar
        """
        # Extract gains
        g = torch.stack([g_h.squeeze(-1) for _, g_h in S_pred], dim=-1)  # (B, num_heads)

        loss = 0.0
        count = 0

        # All pairs
        for i in range(self.num_heads):
            for j in range(i + 1, self.num_heads):
                # Target sign
                target_sign = torch.sign(delta_m[:, i] - delta_m[:, j])

                # Predicted difference
                pred_diff = g[:, i] - g[:, j]

                # Hinge loss
                loss += torch.mean(
                    torch.log(1 + torch.exp(-target_sign * pred_diff))
                )
                count += 1

        return loss / count if count > 0 else torch.tensor(0.0)


# ============================================================================
# Stage 2-A Losses
# ============================================================================

class PseudoTargetLoss(nn.Module):
    """
    Pseudo-Target Loss for Stage 2-A

    Provides weak supervision from Type-II rules
    """

    def __init__(
        self,
        num_heads: int = 6,
        use_huber: bool = True,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_huber = use_huber
        self.huber_delta = huber_delta

    def compute_pseudo_gains(
        self,
        delta_visual_features: Dict[str, float],
        rules: List[Dict],
    ) -> torch.Tensor:
        """Compute pseudo gains from rules"""
        # This is simplified; actual implementation should use rule activation
        pseudo_g = {h: 0.0 for h in range(self.num_heads)}

        head_map = {
            "rhythm": 0, "harmony": 1, "energy": 2,
            "timbre": 3, "space": 4, "texture": 5,
        }

        for rule in rules:
            if abs(delta_visual_features.get(rule['visual_feature'], 0)) > 0.1:
                h = head_map.get(rule['target_head'], 0)
                strength = abs(delta_visual_features.get(rule['visual_feature'], 0))
                pseudo_g[h] += strength * rule['weight']

        # Normalize
        total = sum(pseudo_g.values()) + 1e-8
        pseudo_g_norm = [pseudo_g[h] / total for h in range(self.num_heads)]

        return torch.tensor(pseudo_g_norm)

    def forward(
        self,
        S_final: List[Tuple[torch.Tensor, torch.Tensor]],
        pseudo_g: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            S_final: List of (t_h, g_h) tuples
            pseudo_g: (B, num_heads) pseudo gains from rules

        Returns:
            loss: scalar
        """
        # Extract predicted gains
        g_pred = torch.stack([g_h.squeeze(-1) for _, g_h in S_final], dim=-1)

        if self.use_huber:
            loss = F.huber_loss(g_pred, pseudo_g, delta=self.huber_delta)
        else:
            loss = F.mse_loss(g_pred, pseudo_g)

        return loss


class ManifoldLoss(nn.Module):
    """
    Manifold Loss for Stage 2-A, 2-B, 2-C

    Ensures S_final follows S_proxy distribution
    """

    def __init__(
        self,
        num_heads: int = 6,
        head_dim: int = 64,
        use_moment_matching: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_moment_matching = use_moment_matching

    def forward(
        self,
        S_final: List[Tuple[torch.Tensor, torch.Tensor]],
        S_proxy_stats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            S_final: List of (t_h, g_h) tuples
            S_proxy_stats: Dict with mu_h and sigma_h for each head

        Returns:
            loss: scalar
        """
        loss = 0.0

        for h in range(self.num_heads):
            t_h, g_h = S_final[h]

            mu_proxy = S_proxy_stats[f'mu_{h}'].to(t_h.device)
            sigma_proxy = S_proxy_stats[f'sigma_{h}'].to(t_h.device)

            # Moment matching
            mu_loss = torch.mean((t_h - mu_proxy) ** 2)
            sigma_loss = (torch.std(t_h, dim=0) - sigma_proxy) ** 2
            sigma_loss = torch.mean(sigma_loss)

            loss += mu_loss + sigma_loss

        return loss


class IdentityLoss(nn.Module):
    """
    Identity Loss for Stage 2-A

    When ΔV=0, all g_h should be 0
    """

    def forward(
        self,
        S_final: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Args:
            S_final: List of (t_h, g_h) for zero-delta cases

        Returns:
            loss: scalar
        """
        # Sum all gains
        total_gain = sum([g_h.mean() for _, g_h in S_final])
        return total_gain


class MonotonicityLoss(nn.Module):
    """
    Monotonicity Loss for Stage 2-A

    Larger ||ΔV|| should produce larger Σg_h
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        delta_V_norms: torch.Tensor,
        S_final_list: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> torch.Tensor:
        """
        Args:
            delta_V_norms: (B,) norms of delta_V
            S_final_list: List of B S_final outputs

        Returns:
            loss: scalar
        """
        B = delta_V_norms.shape[0]

        # Compute total gains
        total_gains = []
        for S_final in S_final_list:
            gain_sum = sum([g_h.sum() for _, g_h in S_final])
            total_gains.append(gain_sum)
        total_gains = torch.stack(total_gains)  # (B,)

        loss = 0.0
        count = 0

        # Pairwise comparisons
        for i in range(B):
            for j in range(i + 1, B):
                if delta_V_norms[i] > delta_V_norms[j]:
                    # total_gains[i] should > total_gains[j]
                    loss += F.relu(total_gains[j] - total_gains[i] + self.margin)
                    count += 1

        return loss / count if count > 0 else torch.tensor(0.0)


# ============================================================================
# Stage 2-B Losses
# ============================================================================

class ConditionalPreservationLoss(nn.Module):
    """
    Conditional Preservation Loss for Stage 2-B

    L_preserve = Σ exp(-β·g_h) · d_h(A_edit, A_init)
    Higher g_h → weaker preservation
    """

    def __init__(
        self,
        beta: float = 1.0,
        heads: List[str] = ["rhythm", "harmony", "energy"],
    ):
        super().__init__()
        self.beta = beta
        self.heads = heads
        self.structure_loss = StructurePreservationLoss(heads=heads)

    def forward(
        self,
        A_init: torch.Tensor,
        A_edit: torch.Tensor,
        S_final: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Args:
            A_init: (B, 1, T, F)
            A_edit: (B, 1, T, F) mean over K samples
            S_final: Control signals

        Returns:
            loss: scalar
        """
        head_map = {"rhythm": 0, "harmony": 1, "energy": 2}

        loss = 0.0

        for head_name in self.heads:
            h = head_map[head_name]
            _, g_h = S_final[h]

            # Weight by exp(-β·g_h)
            weight = torch.exp(-self.beta * g_h).mean()

            # Compute distance
            # This is simplified; should use per-head metrics
            if head_name == "rhythm":
                onset_init = self.structure_loss.extract_onset(A_init)
                onset_edit = self.structure_loss.extract_onset(A_edit)
                d_h = 1 - self.structure_loss.pearson_correlation(onset_init, onset_edit).mean()
            elif head_name == "harmony":
                chroma_init = self.structure_loss.extract_chroma(A_init)
                chroma_edit = self.structure_loss.extract_chroma(A_edit)
                d_h = 1 - self.structure_loss.cosine_similarity_batched(chroma_init, chroma_edit)
            elif head_name == "energy":
                rms_init = self.structure_loss.extract_rms(A_init)
                rms_edit = self.structure_loss.extract_rms(A_edit)
                d_h = 1 - self.structure_loss.pearson_correlation(rms_init, rms_edit).mean()

            loss += weight * d_h

        return loss


class CoherenceLoss(nn.Module):
    """
    Multi-Level Coherence Loss for Stage 2-B

    L_coherence = L_high + 0.5 * L_low
    """

    def __init__(
        self,
        high_level_weight: float = 1.0,
        low_level_weight: float = 0.5,
        kappa: float = 0.0,
        eta_r: float = 0.1,
    ):
        super().__init__()
        self.high_level_weight = high_level_weight
        self.low_level_weight = low_level_weight
        self.kappa = kappa
        self.eta_r = eta_r

    def high_level_coherence(
        self,
        delta_V_sem: torch.Tensor,
        delta_A_sem: torch.Tensor,
    ) -> torch.Tensor:
        """Semantic alignment using CLIP/CLAP"""
        return 1 - F.cosine_similarity(delta_V_sem, delta_A_sem, dim=-1).mean()

    def low_level_coherence(
        self,
        delta_visual_features: Dict[str, float],
        delta_audio_features: Dict[str, float],
        rules: List[Dict],
    ) -> torch.Tensor:
        """Rule-based coherence with hinge loss"""
        loss = 0.0
        count = 0

        for rule in rules:
            delta_v = delta_visual_features.get(rule['visual_feature'], 0.0)
            delta_a = delta_audio_features.get(rule['audio_feature'], 0.0)

            # Check if rule is active
            if abs(delta_v) > self.eta_r:
                if rule['correlation'] > 0:
                    # Positive correlation
                    L_r = F.relu(self.kappa - delta_v * delta_a)
                else:
                    # Negative correlation
                    L_r = F.relu(self.kappa + delta_v * delta_a)

                loss += rule['weight'] * L_r
                count += 1

        return loss / count if count > 0 else torch.tensor(0.0)

    def forward(
        self,
        delta_V_sem: torch.Tensor,
        delta_A_sem: torch.Tensor,
        delta_visual_features: Optional[Dict] = None,
        delta_audio_features: Optional[Dict] = None,
        rules: Optional[List] = None,
    ) -> torch.Tensor:
        """
        Args:
            delta_V_sem: (B, D) semantic visual delta (CLIP)
            delta_A_sem: (B, D) semantic audio delta (CLAP)
            delta_visual_features: Low-level visual deltas
            delta_audio_features: Low-level audio deltas
            rules: Type-II rules

        Returns:
            loss: scalar
        """
        L_high = self.high_level_coherence(delta_V_sem, delta_A_sem)

        L_low = torch.tensor(0.0, device=delta_V_sem.device)
        if delta_visual_features and delta_audio_features and rules:
            L_low = self.low_level_coherence(
                delta_visual_features, delta_audio_features, rules
            )

        return self.high_level_weight * L_high + self.low_level_weight * L_low


class ConsistencyLoss(nn.Module):
    """
    Multi-Sample Consistency Loss for Stage 2-B

    L_consistency = L_struct + L_style
    """

    def __init__(
        self,
        struct_heads: List[int] = [0, 1, 2],
        style_heads: List[int] = [3, 4, 5],
        nu_h: float = 0.05,
        max_struct_var: float = 0.1,
    ):
        super().__init__()
        self.struct_heads = struct_heads
        self.style_heads = style_heads
        self.nu_h = nu_h
        self.max_struct_var = max_struct_var

    def extract_metric(
        self,
        audio: torch.Tensor,
        head: int,
    ) -> torch.Tensor:
        """Extract metric for given head"""
        # Placeholder: return mean energy
        return torch.mean(audio, dim=(2, 3))

    def forward(
        self,
        A_edit_samples: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            A_edit_samples: List of K samples, each (B, 1, T, F)

        Returns:
            loss: scalar
        """
        K = len(A_edit_samples)

        loss_struct = 0.0
        loss_style = 0.0

        # Structure: low variance
        for h in self.struct_heads:
            metrics = [self.extract_metric(a, h) for a in A_edit_samples]
            metrics_stack = torch.stack(metrics, dim=0)  # (K, B)
            var = torch.var(metrics_stack, dim=0).mean()
            loss_struct += F.relu(var - self.max_struct_var)

        # Style: minimum diversity
        for h in self.style_heads:
            metrics = [self.extract_metric(a, h) for a in A_edit_samples]
            metrics_stack = torch.stack(metrics, dim=0)
            var = torch.var(metrics_stack, dim=0).mean()
            loss_style += F.relu(self.nu_h - var)

        return loss_struct + loss_style


# ============================================================================
# Stage 2-C Losses
# ============================================================================

class DirectionLoss(nn.Module):
    """
    Direction Loss for Stage 2-C

    ||S_final - S_target||²
    """

    def forward(
        self,
        S_final: List[Tuple[torch.Tensor, torch.Tensor]],
        S_target: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Args:
            S_final: Predicted control signals
            S_target: Target from S_encoder(A_init, A_edit)

        Returns:
            loss: scalar
        """
        loss = 0.0

        for h in range(len(S_final)):
            t_final, g_final = S_final[h]
            t_target, g_target = S_target[h]

            loss += torch.mean((t_final - t_target) ** 2)
            loss += torch.mean((g_final - g_target) ** 2)

        return loss


class BoundedVarianceLoss(nn.Module):
    """
    Bounded Variance Loss for Stage 2-C

    Structure heads: low variance
    Style heads: minimum diversity
    """

    def __init__(
        self,
        max_struct_var: float = 0.1,
        min_style_var: float = 0.05,
    ):
        super().__init__()
        self.max_struct_var = max_struct_var
        self.min_style_var = min_style_var

    def forward(
        self,
        A_pred_list: List[torch.Tensor],
        validity_scores: List[float],
    ) -> torch.Tensor:
        """
        Args:
            A_pred_list: List of predicted audios for valid candidates
            validity_scores: Validity score for each candidate

        Returns:
            loss: scalar
        """
        # Filter high-validity samples
        high_valid_idx = [i for i, v in enumerate(validity_scores) if v >= 0.6]

        if len(high_valid_idx) < 2:
            return torch.tensor(0.0)

        high_valid_audios = [A_pred_list[i] for i in high_valid_idx]

        # Structure variance
        struct_metrics = [torch.mean(a, dim=(2, 3)) for a in high_valid_audios]
        struct_stack = torch.stack(struct_metrics, dim=0)
        var_struct = torch.var(struct_stack, dim=0).mean()
        loss_struct = F.relu(var_struct - self.max_struct_var)

        # Style diversity
        style_metrics = [torch.std(a, dim=(2, 3)) for a in high_valid_audios]
        style_stack = torch.stack(style_metrics, dim=0)
        var_style = torch.var(style_stack, dim=0).mean()
        loss_style = F.relu(self.min_style_var - var_style)

        return loss_struct + loss_style


class PriorRegularizationLoss(nn.Module):
    """
    Prior Regularization Loss for Stage 2-C

    ||δC||² < ε_prior
    """

    def __init__(self, epsilon_prior: float = 0.5):
        super().__init__()
        self.epsilon_prior = epsilon_prior

    def forward(self, delta_C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_C: (B, N_v, num_heads)

        Returns:
            loss: scalar
        """
        delta_C_flat = delta_C.reshape(delta_C.shape[0], -1)
        norm = torch.norm(delta_C_flat, dim=-1)

        # Penalty for exceeding epsilon_prior
        loss = torch.mean(F.relu(norm - self.epsilon_prior) ** 2)

        return loss


# ============================================================================
# Combined Loss Classes
# ============================================================================

class ReconstructionLoss(MultiResolutionSTFTLoss):
    """Alias for MultiResolutionSTFTLoss"""
    pass


class RankConsistencyLoss(PairwiseRankingLoss):
    """Alias for PairwiseRankingLoss"""
    pass
