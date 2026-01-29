"""
Delta Mapping Module (g)

Maps visual delta ΔV and C_anchor to audio control signals S_final
Implements head-wise routing with sensitivity-based gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class DeltaMappingModule(nn.Module):
    """
    Delta Mapping Module: g(ΔV, C_anchor) → S_final

    For each head h:
    1. Compute sensitivity w_h from C_anchor
    2. Project ΔV to head-specific direction u_h
    3. Compute inertia m_h
    4. Route: t_h = (w_h * u_h) / m_h
    5. Compute gain: g_h based on w_h, ||u_h||, stats
    """

    def __init__(
        self,
        delta_dim: int = 512,
        num_heads: int = 6,
        head_dim: int = 64,
        hidden_dim: int = 256,
        inertia_epsilon: float = 1e-3,
        use_manifold_projection: bool = True,
    ):
        """
        Args:
            delta_dim: Dimension of input ΔV
            num_heads: Number of control heads (K=6)
            head_dim: Dimension of each t_h
            hidden_dim: Hidden layer dimension
            inertia_epsilon: Small constant for numerical stability
            use_manifold_projection: Whether to use P_align
        """
        super().__init__()

        self.delta_dim = delta_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.inertia_epsilon = inertia_epsilon
        self.use_manifold_projection = use_manifold_projection

        # Head names (for reference)
        self.head_names = ["rhythm", "harmony", "energy", "timbre", "space", "texture"]

        # Per-head projection MLPs: ΔV → u_h
        self.head_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(delta_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, head_dim),
            )
            for _ in range(num_heads)
        ])

        # Inertia MLP: (ΔV, w_1, ..., w_6) → m_h for each head
        self.inertia_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(delta_dim + num_heads, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus(),  # Ensure positive
            )
            for _ in range(num_heads)
        ])

        # Gain prediction: (w_h, ||u_h||, stats) → g_h
        self.gain_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, hidden_dim // 2),  # 3 features: w_h, ||u_h||, delta_norm
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),  # g_h ∈ [0, 1]
            )
            for _ in range(num_heads)
        ])

        # Manifold projection parameters (P_align)
        if use_manifold_projection:
            self.register_buffer('mu_proxy', torch.zeros(num_heads, head_dim))
            self.register_buffer('sigma_proxy', torch.ones(num_heads, head_dim))

            # Per-head affine transform
            self.W_align = nn.ParameterList([
                nn.Parameter(torch.eye(head_dim))
                for _ in range(num_heads)
            ])
            self.b_align = nn.ParameterList([
                nn.Parameter(torch.zeros(head_dim))
                for _ in range(num_heads)
            ])

    def load_proxy_statistics(self, stats_path: str):
        """
        Load S_proxy statistics from Stage 1

        Args:
            stats_path: Path to saved statistics file
        """
        if not self.use_manifold_projection:
            return

        try:
            stats = torch.load(stats_path)
            for h in range(self.num_heads):
                self.mu_proxy[h] = stats[f'mu_{h}']
                self.sigma_proxy[h] = stats[f'sigma_{h}']
            print(f"Loaded S_proxy statistics from {stats_path}")
        except Exception as e:
            print(f"Failed to load statistics: {e}")

    def compute_sensitivity(self, C_anchor: torch.Tensor) -> torch.Tensor:
        """
        Compute per-head sensitivity from C_anchor

        Args:
            C_anchor: (B, N_v, num_heads)

        Returns:
            w: (B, num_heads) sensitivity scores
        """
        # w_h = mean over visual tokens
        w = C_anchor.mean(dim=1)  # (B, num_heads)
        return w

    def forward_head(
        self,
        h: int,
        delta_V: torch.Tensor,
        w_all: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process single head

        Args:
            h: Head index
            delta_V: (B, delta_dim)
            w_all: (B, num_heads) all sensitivity scores

        Returns:
            t_h_raw: (B, head_dim) direction token
            g_h_raw: (B, 1) gain scalar
        """
        B = delta_V.shape[0]

        # Step 1: Project to head direction
        u_h = self.head_projectors[h](delta_V)  # (B, head_dim)

        # Step 2: Compute inertia
        inertia_input = torch.cat([delta_V, w_all], dim=-1)  # (B, delta_dim + num_heads)
        m_h = self.inertia_mlps[h](inertia_input) + self.inertia_epsilon  # (B, 1)

        # Step 3: Route with sensitivity
        w_h = w_all[:, h:h+1]  # (B, 1)
        t_h_raw = (w_h * u_h) / m_h  # (B, head_dim)

        # Step 4: Compute gain
        u_h_norm = torch.norm(u_h, dim=-1, keepdim=True)  # (B, 1)
        delta_norm = torch.norm(delta_V, dim=-1, keepdim=True)  # (B, 1)
        gain_input = torch.cat([w_h, u_h_norm, delta_norm], dim=-1)  # (B, 3)
        g_h_raw = self.gain_predictors[h](gain_input)  # (B, 1)

        return t_h_raw, g_h_raw

    def apply_manifold_projection(
        self,
        t_h_raw: torch.Tensor,
        g_h_raw: torch.Tensor,
        h: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply P_align to project onto S_proxy manifold

        Args:
            t_h_raw: (B, head_dim)
            g_h_raw: (B, 1)
            h: Head index

        Returns:
            t_h: (B, head_dim) projected
            g_h: (B, 1) clipped
        """
        if not self.use_manifold_projection:
            return t_h_raw, g_h_raw.clamp(0, 1)

        # Normalize
        t_h_norm = F.layer_norm(t_h_raw, (self.head_dim,))

        # Affine transform
        t_h_proj = torch.matmul(t_h_norm, self.W_align[h]) + self.b_align[h]

        # Align to proxy distribution
        mu = self.mu_proxy[h].unsqueeze(0)  # (1, head_dim)
        sigma = self.sigma_proxy[h].unsqueeze(0)  # (1, head_dim)
        t_h = t_h_proj * sigma + mu

        # Clip gain
        g_h = g_h_raw.clamp(0, 1)

        return t_h, g_h

    def forward(
        self,
        delta_V: torch.Tensor,
        C_anchor: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Main forward pass

        Args:
            delta_V: (B, delta_dim) visual delta
            C_anchor: (B, N_v, num_heads) coupling matrix

        Returns:
            S_final: List of 6 tuples (t_h, g_h)
                t_h: (B, head_dim)
                g_h: (B, 1)
        """
        B = delta_V.shape[0]

        # Compute sensitivity for all heads
        w_all = self.compute_sensitivity(C_anchor)  # (B, num_heads)

        # Process each head
        S_raw = []
        for h in range(self.num_heads):
            t_h_raw, g_h_raw = self.forward_head(h, delta_V, w_all)
            S_raw.append((t_h_raw, g_h_raw))

        # Apply manifold projection
        S_final = []
        for h, (t_h_raw, g_h_raw) in enumerate(S_raw):
            t_h, g_h = self.apply_manifold_projection(t_h_raw, g_h_raw, h)
            S_final.append((t_h, g_h))

        return S_final

    def get_head_info(self, S_final: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict:
        """
        Extract information from S_final for logging/debugging

        Args:
            S_final: List of (t_h, g_h) tuples

        Returns:
            Dictionary with head statistics
        """
        info = {}
        for h, (t_h, g_h) in enumerate(S_final):
            head_name = self.head_names[h]
            info[f'{head_name}_gain_mean'] = g_h.mean().item()
            info[f'{head_name}_token_norm'] = torch.norm(t_h, dim=-1).mean().item()

        return info


# For testing
if __name__ == "__main__":
    # Test delta mapping module
    module = DeltaMappingModule(
        delta_dim=512,
        num_heads=6,
        head_dim=64,
        use_manifold_projection=False,  # No stats yet
    )

    # Dummy inputs
    B = 4
    N_v = 256
    delta_V = torch.randn(B, 512)
    C_anchor = torch.rand(B, N_v, 6)
    C_anchor = F.softmax(C_anchor, dim=-1)  # Normalize

    # Forward pass
    S_final = module(delta_V, C_anchor)

    print(f"Input shapes:")
    print(f"  delta_V: {delta_V.shape}")
    print(f"  C_anchor: {C_anchor.shape}")
    print(f"\nOutput S_final (6 heads):")
    for h, (t_h, g_h) in enumerate(S_final):
        print(f"  Head {h}: t_h {t_h.shape}, g_h {g_h.shape}")

    # Get head info
    info = module.get_head_info(S_final)
    print(f"\nHead info:")
    for k, v in info.items():
        print(f"  {k}: {v:.4f}")
