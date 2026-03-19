"""
Feature Importance — Gradient saliency and occlusion sensitivity.

Lightweight alternatives to SHAP for understanding which image
regions affect 3D reconstruction quality.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class GradientSaliency:
    """
    Vanilla gradient saliency maps.

    Computes |∂output/∂input| to identify pixels with the
    greatest influence on the reconstruction output.

    Args:
        model:  Trained reconstruction model.
        device: Device string.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def compute(self, image: torch.Tensor) -> np.ndarray:
        """
        Compute gradient saliency map.

        Args:
            image: [1, 3, H, W] input image tensor.
        Returns:
            [H, W] saliency map (normalized to [0, 1]).
        """
        image = image.to(self.device).requires_grad_(True)
        pred = self.model(image)
        scalar_out = pred.sum()
        scalar_out.backward()

        grad = image.grad.squeeze(0).abs()  # [3, H, W]
        saliency = grad.mean(dim=0)          # [H, W]
        saliency = saliency.detach().cpu().numpy()

        mx = saliency.max()
        if mx > 0:
            saliency = saliency / mx
        return saliency

    def plot(
        self,
        image: np.ndarray,
        saliency: np.ndarray,
        title: str = "Gradient Saliency",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot saliency map overlay."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        axes[0].imshow(image)
        axes[0].set_title("Input", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(saliency, cmap="hot")
        axes[1].set_title("Saliency Map", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(image)
        axes[2].imshow(saliency, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig


class OcclusionSensitivity:
    """
    Occlusion sensitivity analysis.

    Systematically masks image patches and measures how the
    Chamfer Distance changes. Regions where occlusion causes
    the largest quality drop are most important.

    Args:
        model:      Trained reconstruction model.
        criterion:  Loss function (ChamferDistanceLoss).
        device:     Device string.
        patch_size: Size of the square occlusion patch.
        stride:     Stride for sliding the patch.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: str = "cuda",
        patch_size: int = 16,
        stride: int = 8,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.criterion = criterion.to(device)
        self.device = device
        self.patch_size = patch_size
        self.stride = stride

    @torch.no_grad()
    def compute(
        self,
        image: torch.Tensor,
        gt_points: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute occlusion sensitivity map.

        Args:
            image:     [1, 3, H, W] input image.
            gt_points: [1, M, 3] ground truth point cloud.
        Returns:
            [H', W'] sensitivity map where higher = more important.
        """
        image = image.to(self.device)
        gt_points = gt_points.to(self.device)

        _, _, H, W = image.shape

        # Baseline loss
        pred_base = self.model(image)
        base_loss = self.criterion(pred_base, gt_points).item()

        # Compute sensitivity for each patch position
        rows = list(range(0, H - self.patch_size + 1, self.stride))
        cols = list(range(0, W - self.patch_size + 1, self.stride))

        sensitivity = np.zeros((len(rows), len(cols)))

        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                occluded = image.clone()
                occluded[:, :, r:r + self.patch_size, c:c + self.patch_size] = 0

                pred_occ = self.model(occluded)
                occ_loss = self.criterion(pred_occ, gt_points).item()

                sensitivity[i, j] = max(occ_loss - base_loss, 0)

        # Normalize
        mx = sensitivity.max()
        if mx > 0:
            sensitivity = sensitivity / mx

        return sensitivity

    def plot(
        self,
        image: np.ndarray,
        sensitivity: np.ndarray,
        title: str = "Occlusion Sensitivity",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot occlusion sensitivity overlay."""
        from PIL import Image as PILImage

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        axes[0].imshow(image)
        axes[0].set_title("Input", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Upsample sensitivity to image size
        h, w = image.shape[:2]
        sens_resized = np.array(
            PILImage.fromarray((sensitivity * 255).astype(np.uint8)).resize(
                (w, h), PILImage.BILINEAR
            )
        ).astype(np.float32) / 255.0

        axes[1].imshow(sens_resized, cmap="hot")
        axes[1].set_title("Sensitivity Map", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(image)
        axes[2].imshow(sens_resized, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig
