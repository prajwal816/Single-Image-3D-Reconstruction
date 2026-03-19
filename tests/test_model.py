"""
Unit tests for model components.
"""

import pytest
import torch
from src.models.encoder import ImageEncoder
from src.models.decoder import PointCloudDecoder
from src.models.reconstruction_net import SingleImageReconstructionNet


class TestImageEncoder:
    """Test the image encoder."""

    def test_output_shape(self):
        enc = ImageEncoder(backbone="resnet18", pretrained=False, latent_dim=256)
        x = torch.randn(2, 3, 128, 128)
        out = enc(x)
        assert out.shape == (2, 256)

    def test_default_latent_dim(self):
        enc = ImageEncoder(backbone="resnet18", pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = enc(x)
        assert out.shape == (1, 512)

    def test_feature_maps(self):
        enc = ImageEncoder(backbone="resnet18", pretrained=False, latent_dim=128)
        x = torch.randn(1, 3, 224, 224)
        feat = enc.get_feature_maps(x)
        assert feat.ndim == 4
        assert feat.shape[0] == 1

    def test_gradient_flow(self):
        enc = ImageEncoder(backbone="resnet18", pretrained=False, latent_dim=128)
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestPointCloudDecoder:
    """Test the point cloud decoder."""

    def test_output_shape(self):
        dec = PointCloudDecoder(latent_dim=256, hidden_dims=[128, 64],
                                num_points=512)
        z = torch.randn(2, 256)
        out = dec(z)
        assert out.shape == (2, 512, 3)

    def test_output_range(self):
        dec = PointCloudDecoder(latent_dim=128, num_points=256)
        z = torch.randn(4, 128)
        out = dec(z)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_gradient_flow(self):
        dec = PointCloudDecoder(latent_dim=128, num_points=256)
        z = torch.randn(2, 128, requires_grad=True)
        out = dec(z)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None


class TestReconstructionNet:
    """Test the end-to-end model."""

    def test_forward(self):
        model = SingleImageReconstructionNet(
            encoder_cfg={"backbone": "resnet18", "pretrained": False, "latent_dim": 128},
            decoder_cfg={"hidden_dims": [128], "num_points": 256},
        )
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        assert out.shape == (2, 256, 3)

    def test_encode_decode(self):
        model = SingleImageReconstructionNet(
            encoder_cfg={"backbone": "resnet18", "pretrained": False, "latent_dim": 64},
            decoder_cfg={"hidden_dims": [64], "num_points": 128},
        )
        x = torch.randn(1, 3, 64, 64)
        latent = model.encode(x)
        assert latent.shape == (1, 64)

        pts = model.decode(latent)
        assert pts.shape == (1, 128, 3)

    def test_param_counts(self):
        model = SingleImageReconstructionNet(
            encoder_cfg={"backbone": "resnet18", "pretrained": False, "latent_dim": 128},
            decoder_cfg={"hidden_dims": [128], "num_points": 256},
        )
        counts = model.get_num_params()
        assert counts["total"] == counts["encoder"] + counts["decoder"]
        assert counts["total"] > 0

    def test_full_gradient(self):
        model = SingleImageReconstructionNet(
            encoder_cfg={"backbone": "resnet18", "pretrained": False, "latent_dim": 64},
            decoder_cfg={"hidden_dims": [64], "num_points": 64},
        )
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
