"""
Loss Functions for 3D Reconstruction.

Primary loss: Chamfer Distance — measures the similarity between two
unordered point sets by computing symmetric nearest-neighbour distances.
"""

import torch
import torch.nn as nn


class ChamferDistanceLoss(nn.Module):
    """
    Symmetric Chamfer Distance between two point clouds.

    For point sets P and Q:
        CD(P, Q) = (1/|P|) Σ_{p∈P} min_{q∈Q} ||p - q||²
                 + (1/|Q|) Σ_{q∈Q} min_{p∈P} ||q - p||²

    This is a pure-PyTorch implementation using batched pairwise distances
    for GPU efficiency. No external dependencies required.

    Args:
        reduction: "mean" averages over batch, "sum" sums over batch, "none" per-sample.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), \
            f"Invalid reduction: {reduction}"
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Predicted point cloud [B, N, 3].
            target: Ground truth point cloud [B, M, 3].
        Returns:
            Chamfer Distance scalar (or per-sample if reduction='none').
        """
        # Pairwise squared distances: [B, N, M]
        diff = pred.unsqueeze(2) - target.unsqueeze(1)   # [B, N, M, 3]
        dist_matrix = (diff ** 2).sum(dim=-1)             # [B, N, M]

        # For each point in pred, find nearest in target
        min_pred_to_target = dist_matrix.min(dim=2).values  # [B, N]
        # For each point in target, find nearest in pred
        min_target_to_pred = dist_matrix.min(dim=1).values  # [B, M]

        # Symmetric Chamfer Distance
        cd = min_pred_to_target.mean(dim=1) + min_target_to_pred.mean(dim=1)  # [B]

        if self.reduction == "mean":
            return cd.mean()
        elif self.reduction == "sum":
            return cd.sum()
        else:
            return cd
