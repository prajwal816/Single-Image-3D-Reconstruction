"""
Image Encoder — ResNet-backbone feature extractor.

Extracts a compact latent vector from an RGB image, suitable for
downstream 3D point cloud prediction.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ImageEncoder(nn.Module):
    """
    ResNet-based image encoder that maps an RGB image to a latent vector.

    Architecture:
        Input [B, 3, H, W]
        → ResNet-18 backbone (all layers except final FC)
        → AdaptiveAvgPool → [B, 512]
        → FC projection → [B, latent_dim]

    Args:
        backbone:   ResNet variant name ("resnet18" or "resnet34").
        pretrained: Whether to use ImageNet-pretrained weights.
        latent_dim: Dimensionality of the output latent vector.
        freeze_backbone: Freeze backbone weights (fine-tune head only).
    """

    BACKBONES = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    }

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        latent_dim: int = 512,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if backbone not in self.BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                             f"Choose from {list(self.BACKBONES.keys())}")

        model_fn, weights, feat_dim = self.BACKBONES[backbone]
        base = model_fn(weights=weights if pretrained else None)

        # Use everything except the final FC layer
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.latent_dim = latent_dim
        self._feat_dim = feat_dim

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, H, W].
        Returns:
            Latent vectors [B, latent_dim].
        """
        feat = self.features(x)          # [B, feat_dim, h, w]
        feat = self.pool(feat)           # [B, feat_dim, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, feat_dim]
        latent = self.projector(feat)    # [B, latent_dim]
        return latent

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return the spatial feature maps before pooling (for explainability)."""
        return self.features(x)
