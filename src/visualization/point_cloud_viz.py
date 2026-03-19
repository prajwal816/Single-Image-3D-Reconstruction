"""
Point Cloud Visualization — 3D scatter plots and side-by-side comparisons.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple
import torch


class PointCloudVisualizer:
    """
    Generates publication-quality 3D point cloud visualizations.

    Features:
        - Single point cloud rendering
        - Side-by-side: input image | predicted 3D | ground truth 3D
        - Multi-sample grid comparisons
    """

    # Color palette for different categories
    CATEGORY_COLORS = {
        "cube":     "#2878D2",
        "sphere":   "#DC503C",
        "cylinder": "#3CBE5A",
        "cone":     "#F0B428",
        "torus":    "#B43CC8",
    }

    DEFAULT_COLOR = "#5A8FCC"

    @staticmethod
    def plot_point_cloud(
        points: np.ndarray,
        title: str = "",
        color: str = None,
        ax: Optional[plt.Axes] = None,
        point_size: float = 1.5,
        elev: float = 25,
        azim: float = 45,
    ) -> Optional[plt.Figure]:
        """
        Render a single 3D point cloud.

        Args:
            points: [N, 3] array.
            title:  Plot title.
            color:  Point color (hex or matplotlib color).
            ax:     Existing 3D axes (creates new figure if None).
            point_size: Marker size.
            elev:   Elevation angle.
            azim:   Azimuth angle.
        Returns:
            Figure if ax was None, else None.
        """
        created_fig = False
        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")
            created_fig = True

        c = color or PointCloudVisualizer.DEFAULT_COLOR
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=c, s=point_size, alpha=0.7, edgecolors="none")

        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10, fontweight="bold")

        # Clean axes
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

        return fig if created_fig else None

    @staticmethod
    def plot_comparison(
        image: np.ndarray,
        pred_points: np.ndarray,
        gt_points: np.ndarray,
        category: str = "",
        metrics: Optional[dict] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Side-by-side comparison: Input Image | Predicted 3D | Ground Truth 3D.

        Args:
            image:      [H, W, 3] input image (uint8 or float).
            pred_points: [N, 3] predicted point cloud.
            gt_points:   [M, 3] ground truth point cloud.
            category:    Category name for coloring.
            metrics:     Optional dict with {cd, iou, completeness}.
            save_path:   If provided, save figure to this path.
        Returns:
            matplotlib Figure.
        """
        fig = plt.figure(figsize=(15, 5))

        color = PointCloudVisualizer.CATEGORY_COLORS.get(
            category, PointCloudVisualizer.DEFAULT_COLOR
        )

        # Input image
        ax1 = fig.add_subplot(131)
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        ax1.imshow(image)
        ax1.set_title(f"Input Image\n({category})", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # Predicted point cloud
        ax2 = fig.add_subplot(132, projection="3d")
        PointCloudVisualizer.plot_point_cloud(
            pred_points, "Predicted 3D", color=color, ax=ax2
        )

        # Ground truth point cloud
        ax3 = fig.add_subplot(133, projection="3d")
        PointCloudVisualizer.plot_point_cloud(
            gt_points, "Ground Truth 3D", color=color, ax=ax3
        )

        # Add metrics text
        if metrics:
            metric_text = (
                f"CD: {metrics.get('cd', 0):.5f}  |  "
                f"IoU: {metrics.get('iou', 0):.3f}  |  "
                f"Comp: {metrics.get('completeness', 0):.3f}"
            )
            fig.suptitle(metric_text, fontsize=10, y=0.02, color="gray")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig

    @staticmethod
    def plot_comparison_grid(
        samples: List[dict],
        save_path: Optional[str] = None,
        max_cols: int = 4,
    ) -> plt.Figure:
        """
        Grid of comparison triplets.

        Args:
            samples: List of dicts with keys: image, pred_points, gt_points,
                     category, metrics (optional).
            save_path: If provided, save figure.
            max_cols: Maximum columns in the grid.
        """
        n = len(samples)
        n_cols = min(n, max_cols)
        n_rows = (n + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

        for idx, sample in enumerate(samples):
            row = idx // n_cols
            col = idx % n_cols

            # We show a simplified view: just predicted 3D
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
            color = PointCloudVisualizer.CATEGORY_COLORS.get(
                sample.get("category", ""), PointCloudVisualizer.DEFAULT_COLOR
            )
            PointCloudVisualizer.plot_point_cloud(
                sample["pred_points"],
                title=sample.get("category", f"Sample {idx}"),
                color=color,
                ax=ax,
            )

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig
