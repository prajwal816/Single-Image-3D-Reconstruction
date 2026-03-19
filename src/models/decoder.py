"""
Point Cloud Decoder — MLP-based latent-to-3D decoder.

Maps a latent vector to a set of 3D point coordinates,
predicting the full 3D shape as an unordered point cloud.
"""

import torch
import torch.nn as nn
from typing import List


class PointCloudDecoder(nn.Module):
    """
    MLP decoder that maps a latent vector to a 3D point cloud.

    Architecture:
        Input [B, latent_dim]
        → FC layers with BatchNorm + ReLU
        → Final FC → [B, num_points * 3]
        → Reshape → [B, num_points, 3]
        → Tanh activation (bound output to [-1, 1])

    Args:
        latent_dim:  Dimensionality of the input latent vector.
        hidden_dims: List of hidden layer widths.
        num_points:  Number of 3D points to predict per sample.
        activation:  Activation function name ("relu" or "leaky_relu").
        use_batch_norm: Whether to apply BatchNorm after each hidden layer.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dims: List[int] = None,
        num_points: int = 1024,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 256]

        self.num_points = num_points
        self.latent_dim = latent_dim
        output_dim = num_points * 3

        act_fn = nn.ReLU(inplace=True) if activation == "relu" else nn.LeakyReLU(0.2, inplace=True)

        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn if activation == "relu" else nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.05))
            in_dim = h_dim

        # Final projection
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Tanh())  # Bound output to [-1, 1]

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vectors [B, latent_dim].
        Returns:
            Predicted point clouds [B, num_points, 3].
        """
        out = self.decoder(z)                        # [B, num_points * 3]
        out = out.view(-1, self.num_points, 3)       # [B, num_points, 3]
        return out
