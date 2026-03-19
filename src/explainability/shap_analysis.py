"""
SHAP Analysis for Single-Image 3D Reconstruction.

Uses gradient-based SHAP explanation to attribute which image regions
drive the 3D reconstruction output. Generates heatmap overlays
that reveal the model's spatial attention.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


class SHAPAnalyzer:
    """
    SHAP-based explainability for the image-to-3D reconstruction model.

    Uses gradient-based attribution (integrated gradients as a lightweight
    SHAP approximation) to identify which image regions most influence
    the predicted 3D output.

    For full SHAP analysis, the `shap` library is used when available.

    Args:
        model:     Trained reconstruction model.
        device:    Device string.
        num_background_samples: Samples for SHAP background distribution.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        num_background_samples: int = 50,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.num_bg = num_background_samples

    def compute_gradient_attribution(
        self,
        image: torch.Tensor,
        num_steps: int = 50,
    ) -> np.ndarray:
        """
        Compute integrated gradients attribution map for a single image.

        Args:
            image: [1, 3, H, W] input image tensor.
            num_steps: Number of interpolation steps.
        Returns:
            [H, W] attribution map (higher = more important).
        """
        image = image.to(self.device).requires_grad_(True)
        baseline = torch.zeros_like(image)

        # Integrated gradients
        total_grad = torch.zeros_like(image)

        for step in range(num_steps):
            alpha = step / num_steps
            interpolated = baseline + alpha * (image - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            pred = self.model(interpolated)
            # Use sum of all predicted coordinates as scalar output
            scalar_output = pred.sum()
            scalar_output.backward()

            total_grad += interpolated.grad

        # Average and multiply by (input - baseline)
        avg_grad = total_grad / num_steps
        attribution = (image.detach() - baseline) * avg_grad

        # Reduce to spatial map: mean over channels, absolute value
        attr_map = attribution.squeeze(0).abs().mean(dim=0)  # [H, W]
        attr_map = attr_map.detach().cpu().numpy()

        # Normalize to [0, 1]
        attr_max = attr_map.max()
        if attr_max > 0:
            attr_map = attr_map / attr_max

        return attr_map

    def compute_shap_values(
        self,
        images: torch.Tensor,
        background: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute SHAP values using the shap library if available.
        Falls back to integrated gradients otherwise.

        Args:
            images:     [B, 3, H, W] images to explain.
            background: [K, 3, H, W] background distribution.
        Returns:
            [B, H, W] SHAP attribution maps.
        """
        try:
            import shap

            def model_fn(x):
                """Wrapper that returns scalar output per sample."""
                with torch.no_grad():
                    x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
                    pred = self.model(x_t)
                    # Mean Chamfer-like scalar per sample
                    return pred.mean(dim=(1, 2)).cpu().numpy()

            if background is None:
                background = torch.randn(self.num_bg, *images.shape[1:]) * 0.1

            bg_np = background.cpu().numpy()
            img_np = images.cpu().numpy()

            explainer = shap.GradientExplainer(
                (self.model, self.model.encoder.features),
                torch.tensor(bg_np, dtype=torch.float32, device=self.device),
            )
            shap_values = explainer.shap_values(
                torch.tensor(img_np, dtype=torch.float32, device=self.device)
            )

            # Average across channels, take abs
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            attr_maps = np.abs(shap_values).mean(axis=1)  # [B, H, W]

            # Normalize
            for i in range(len(attr_maps)):
                mx = attr_maps[i].max()
                if mx > 0:
                    attr_maps[i] /= mx

            return attr_maps

        except Exception:
            # Fallback to integrated gradients
            results = []
            for i in range(images.shape[0]):
                attr = self.compute_gradient_attribution(images[i:i + 1])
                results.append(attr)
            return np.stack(results, axis=0)

    def plot_attribution(
        self,
        image: np.ndarray,
        attribution: np.ndarray,
        title: str = "Attribution Map",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Overlay attribution heatmap on the input image.

        Args:
            image:       [H, W, 3] input image (uint8 or float [0,1]).
            attribution: [H, W] attribution map.
            title:       Plot title.
            save_path:   If provided, save figure.
        Returns:
            matplotlib Figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        if image.max() <= 1.0:
            image_display = (image * 255).astype(np.uint8)
        else:
            image_display = image.astype(np.uint8)

        axes[0].imshow(image_display)
        axes[0].set_title("Input Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Attribution heatmap
        im = axes[1].imshow(attribution, cmap="hot", interpolation="bilinear")
        axes[1].set_title("Attribution Map", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(image_display)
        axes[2].imshow(attribution, cmap="jet", alpha=0.5, interpolation="bilinear")
        axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig

    def analyze_batch(
        self,
        images: torch.Tensor,
        original_images: np.ndarray,
        categories: List[str],
        save_dir: str,
    ) -> List[np.ndarray]:
        """
        Run full SHAP analysis on a batch of images and save results.

        Args:
            images:          [B, 3, H, W] preprocessed images.
            original_images: [B, H, W, 3] original images for display.
            categories:      List of category names.
            save_dir:        Directory to save plots.
        Returns:
            List of [H, W] attribution maps.
        """
        os.makedirs(save_dir, exist_ok=True)
        attr_maps = self.compute_shap_values(images)

        for i in range(len(attr_maps)):
            self.plot_attribution(
                original_images[i],
                attr_maps[i],
                title=f"SHAP Attribution — {categories[i]}",
                save_path=os.path.join(save_dir, f"shap_{categories[i]}_{i}.png"),
            )

        return list(attr_maps)
