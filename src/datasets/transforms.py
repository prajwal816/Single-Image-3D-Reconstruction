"""
Data augmentation transforms for images and point clouds.
"""

import torch
import numpy as np
import math
from typing import Optional


class ImageTransforms:
    """Composable image augmentation pipeline for training.

    Args:
        color_jitter: Apply random brightness/contrast/saturation shifts.
        random_crop:  Apply random resized crop.
        normalize:    Normalize to ImageNet statistics.
        image_size:   Target image size after crop.
    """

    def __init__(
        self,
        color_jitter: bool = True,
        random_crop: bool = True,
        normalize: bool = True,
        image_size: int = 224,
    ):
        self.color_jitter = color_jitter
        self.random_crop = random_crop
        self.normalize = normalize
        self.image_size = image_size

        # ImageNet statistics
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor of shape [C, H, W] in range [0, 1].
        Returns:
            Augmented image tensor.
        """
        if self.color_jitter:
            image = self._apply_color_jitter(image)

        if self.random_crop:
            image = self._apply_random_crop(image)

        if self.normalize:
            image = (image - self.mean) / self.std

        return image

    def _apply_color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Random brightness, contrast, and saturation adjustments."""
        # Brightness
        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
        image = torch.clamp(image * brightness, 0, 1)

        # Contrast
        contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
        mean_val = image.mean()
        image = torch.clamp((image - mean_val) * contrast + mean_val, 0, 1)

        return image

    def _apply_random_crop(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate random crop by zooming into a random region."""
        _, h, w = image.shape
        scale = 0.85 + torch.rand(1).item() * 0.15
        new_h, new_w = int(h * scale), int(w * scale)
        top = torch.randint(0, max(h - new_h, 1), (1,)).item()
        left = torch.randint(0, max(w - new_w, 1), (1,)).item()
        image = image[:, top:top + new_h, left:left + new_w]
        # Resize back using interpolation
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)
        return image


class PointCloudTransforms:
    """Augmentation pipeline for 3D point clouds.

    Args:
        jitter_std:  Std-dev of Gaussian noise added to points.
        random_rotation: Apply random SO(3) rotation.
        random_scale: Apply random uniform scaling.
        scale_range:  (min, max) scale factors.
    """

    def __init__(
        self,
        jitter_std: float = 0.02,
        random_rotation: bool = True,
        random_scale: bool = False,
        scale_range: tuple = (0.8, 1.2),
    ):
        self.jitter_std = jitter_std
        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.scale_range = scale_range

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: Tensor of shape [N, 3].
        Returns:
            Augmented point cloud tensor.
        """
        if self.random_rotation:
            points = self._apply_rotation(points)

        if self.random_scale:
            points = self._apply_scale(points)

        if self.jitter_std > 0:
            noise = torch.randn_like(points) * self.jitter_std
            points = points + noise

        return points

    def _apply_rotation(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random rotation around the Y axis."""
        angle = torch.rand(1).item() * 2 * math.pi
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotation = torch.tensor([
            [cos_a,  0, sin_a],
            [0,      1, 0],
            [-sin_a, 0, cos_a],
        ], dtype=points.dtype)
        return points @ rotation.T

    def _apply_scale(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random uniform scaling."""
        scale = self.scale_range[0] + torch.rand(1).item() * (
            self.scale_range[1] - self.scale_range[0]
        )
        return points * scale
