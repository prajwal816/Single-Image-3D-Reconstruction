"""
Evaluation Metrics for 3D Point Cloud Reconstruction.

- Chamfer Distance (per-sample)
- IoU via voxelization
- Reconstruction Completeness
"""

import torch
import numpy as np
from typing import Optional


def compute_chamfer_distance(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample symmetric Chamfer Distance.

    Args:
        pred: [B, N, 3] predicted point cloud.
        gt:   [B, M, 3] ground truth point cloud.
    Returns:
        [B] tensor of Chamfer Distances.
    """
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)   # [B, N, M, 3]
    dist = (diff ** 2).sum(dim=-1)                # [B, N, M]

    min_pred = dist.min(dim=2).values.mean(dim=1)  # [B]
    min_gt = dist.min(dim=1).values.mean(dim=1)    # [B]

    return min_pred + min_gt


def compute_iou(
    pred: torch.Tensor,
    gt: torch.Tensor,
    resolution: int = 32,
    padding: float = 0.1,
) -> torch.Tensor:
    """
    Compute IoU between predicted and ground truth point clouds
    via voxelization.

    Both point sets are voxelized into a 3D grid, and volumetric
    IoU is computed as |intersection| / |union|.

    Args:
        pred:       [B, N, 3] predicted point cloud.
        gt:         [B, M, 3] ground truth point cloud.
        resolution: Grid resolution along each axis.
        padding:    Padding around the bounding box (fraction).
    Returns:
        [B] tensor of IoU values.
    """
    batch_size = pred.shape[0]
    ious = []

    for b in range(batch_size):
        p = pred[b].detach().cpu().numpy()  # [N, 3]
        g = gt[b].detach().cpu().numpy()    # [M, 3]

        # Compute common bounding box
        all_pts = np.concatenate([p, g], axis=0)
        bbox_min = all_pts.min(axis=0) - padding
        bbox_max = all_pts.max(axis=0) + padding

        # Voxelize
        pred_vox = _voxelize(p, bbox_min, bbox_max, resolution)
        gt_vox = _voxelize(g, bbox_min, bbox_max, resolution)

        intersection = np.logical_and(pred_vox, gt_vox).sum()
        union = np.logical_or(pred_vox, gt_vox).sum()

        iou = intersection / max(union, 1)
        ious.append(iou)

    return torch.tensor(ious, dtype=torch.float32)


def _voxelize(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """
    Convert point cloud to binary voxel grid.

    Args:
        points:     [N, 3] array.
        bbox_min:   [3] minimum corner.
        bbox_max:   [3] maximum corner.
        resolution: Grid resolution.
    Returns:
        [resolution, resolution, resolution] binary grid.
    """
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    extent = bbox_max - bbox_min
    extent = np.maximum(extent, 1e-6)

    # Normalize points to [0, resolution-1]
    normalized = (points - bbox_min) / extent * (resolution - 1)
    indices = np.clip(normalized.astype(int), 0, resolution - 1)

    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return grid


def compute_reconstruction_completeness(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    """
    Compute reconstruction completeness: fraction of ground truth points
    that have at least one predicted point within `threshold` distance.

    Args:
        pred:      [B, N, 3] predicted point cloud.
        gt:        [B, M, 3] ground truth point cloud.
        threshold: Distance threshold.
    Returns:
        [B] tensor of completeness values in [0, 1].
    """
    diff = gt.unsqueeze(2) - pred.unsqueeze(1)  # [B, M, N, 3]
    dist = (diff ** 2).sum(dim=-1).sqrt()        # [B, M, N]

    min_dist = dist.min(dim=2).values             # [B, M]
    covered = (min_dist < threshold).float()      # [B, M]

    completeness = covered.mean(dim=1)            # [B]
    return completeness
