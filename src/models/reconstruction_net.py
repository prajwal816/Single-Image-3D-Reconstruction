"""
End-to-end Single-Image 3D Reconstruction Network.

Combines the ImageEncoder and PointCloudDecoder into a single module.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .encoder import ImageEncoder
from .decoder import PointCloudDecoder


class SingleImageReconstructionNet(nn.Module):
    """
    Full reconstruction pipeline: RGB image → latent vector → 3D point cloud.

    Architecture:
        ┌──────────────┐    ┌─────────┐    ┌──────────────────┐
        │ RGB Image    │ →  │ ResNet  │ →  │ MLP Decoder      │
        │ [B,3,H,W]   │    │ Encoder │    │ [B, N, 3]        │
        └──────────────┘    └─────────┘    └──────────────────┘

    Args:
        encoder_cfg: Dict with keys: backbone, pretrained, latent_dim.
        decoder_cfg: Dict with keys: hidden_dims, num_points, activation, use_batch_norm.
    """

    def __init__(
        self,
        encoder_cfg: Optional[Dict] = None,
        decoder_cfg: Optional[Dict] = None,
    ):
        super().__init__()

        encoder_cfg = encoder_cfg or {}
        decoder_cfg = decoder_cfg or {}

        latent_dim = encoder_cfg.get("latent_dim", 512)

        self.encoder = ImageEncoder(
            backbone=encoder_cfg.get("backbone", "resnet18"),
            pretrained=encoder_cfg.get("pretrained", True),
            latent_dim=latent_dim,
        )
        self.decoder = PointCloudDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_cfg.get("hidden_dims", [512, 512, 256]),
            num_points=decoder_cfg.get("num_points", 1024),
            activation=decoder_cfg.get("activation", "relu"),
            use_batch_norm=decoder_cfg.get("use_batch_norm", True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Input RGB images [B, 3, H, W].
        Returns:
            Predicted point clouds [B, num_points, 3].
        """
        latent = self.encoder(images)      # [B, latent_dim]
        points = self.decoder(latent)      # [B, num_points, 3]
        return points

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Extract latent representation only."""
        return self.encoder(images)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to point cloud."""
        return self.decoder(latent)

    def get_num_params(self) -> Dict[str, int]:
        """Return parameter counts for each component."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        return {
            "encoder": enc_params,
            "decoder": dec_params,
            "total": enc_params + dec_params,
        }
