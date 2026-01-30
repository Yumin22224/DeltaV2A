"""
Loss functions for DeltaV2A training
"""

from .losses import (
    # Stage 1
    ReconstructionLoss,
    MultiResolutionSTFTLoss,
    StructurePreservationLoss,
    PairwiseRankingLoss,
    RankConsistencyLoss,
    # Stage 2-A
    PseudoTargetLoss,
    ManifoldLoss,
    IdentityLoss,
    MonotonicityLoss,
    # Stage 2-B
    ConditionalPreservationLoss,
    CoherenceLoss,
    ConsistencyLoss,
    # Stage 2-C
    DirectionLoss,
    BoundedVarianceLoss,
    PriorRegularizationLoss,
)

__all__ = [
    # Stage 1
    "ReconstructionLoss",
    "MultiResolutionSTFTLoss",
    "StructurePreservationLoss",
    "PairwiseRankingLoss",
    "RankConsistencyLoss",
    # Stage 2-A
    "PseudoTargetLoss",
    "ManifoldLoss",
    "IdentityLoss",
    "MonotonicityLoss",
    # Stage 2-B
    "ConditionalPreservationLoss",
    "CoherenceLoss",
    "ConsistencyLoss",
    # Stage 2-C
    "DirectionLoss",
    "BoundedVarianceLoss",
    "PriorRegularizationLoss",
]
