"""
Unit tests for the SimulatedShapeNet dataset.
"""

import pytest
import torch
import numpy as np
from src.datasets.shapenet_simulated import (
    SimulatedShapeNetDataset,
    _sample_cube,
    _sample_sphere,
    _sample_cylinder,
    _sample_cone,
    _sample_torus,
)
from src.datasets.transforms import ImageTransforms, PointCloudTransforms


class TestPrimitiveSamplers:
    """Test that each primitive sampler generates valid point clouds."""

    @pytest.mark.parametrize("sampler,name", [
        (_sample_cube, "cube"),
        (_sample_sphere, "sphere"),
        (_sample_cylinder, "cylinder"),
        (_sample_cone, "cone"),
        (_sample_torus, "torus"),
    ])
    def test_shape(self, sampler, name):
        rng = np.random.RandomState(42)
        points = sampler(1024, rng)
        assert points.shape == (1024, 3), f"{name}: expected (1024,3), got {points.shape}"
        assert points.dtype == np.float32

    @pytest.mark.parametrize("sampler", [
        _sample_cube, _sample_sphere, _sample_cylinder, _sample_cone, _sample_torus
    ])
    def test_bounded(self, sampler):
        rng = np.random.RandomState(42)
        points = sampler(2048, rng)
        assert np.all(np.abs(points) < 1.0), "Points should be within [-1, 1] range"


class TestSimulatedShapeNetDataset:
    """Test the full dataset class."""

    def test_length(self):
        ds = SimulatedShapeNetDataset(num_samples=100, num_points=256, image_size=64)
        assert len(ds) == 100

    def test_sample_shapes(self):
        ds = SimulatedShapeNetDataset(num_samples=10, num_points=512, image_size=128)
        sample = ds[0]

        assert sample["image"].shape == (3, 128, 128)
        assert sample["point_cloud"].shape == (512, 3)
        assert sample["category_id"].shape == ()
        assert isinstance(sample["category_name"], str)

    def test_deterministic(self):
        ds1 = SimulatedShapeNetDataset(num_samples=10, num_points=256, image_size=64, seed=42)
        ds2 = SimulatedShapeNetDataset(num_samples=10, num_points=256, image_size=64, seed=42)

        s1 = ds1[5]
        s2 = ds2[5]

        assert torch.allclose(s1["image"], s2["image"])
        assert torch.allclose(s1["point_cloud"], s2["point_cloud"])

    def test_categories(self):
        cats = ["cube", "sphere"]
        ds = SimulatedShapeNetDataset(num_samples=20, categories=cats, num_points=128, image_size=64)
        assert ds.get_categories() == cats
        counts = ds.get_category_counts()
        assert set(counts.keys()).issubset(set(cats))

    def test_image_range(self):
        ds = SimulatedShapeNetDataset(num_samples=5, num_points=128, image_size=64)
        sample = ds[0]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_with_transforms(self):
        img_t = ImageTransforms(color_jitter=True, random_crop=True,
                                normalize=True, image_size=64)
        pc_t = PointCloudTransforms(jitter_std=0.01, random_rotation=True)

        ds = SimulatedShapeNetDataset(
            num_samples=5, num_points=128, image_size=64,
            transform=img_t, point_transform=pc_t,
        )
        sample = ds[0]
        assert sample["image"].shape == (3, 64, 64)
        assert sample["point_cloud"].shape == (128, 3)


class TestTransforms:
    """Test transform modules."""

    def test_image_transform(self):
        t = ImageTransforms(color_jitter=True, random_crop=True, normalize=True)
        img = torch.rand(3, 224, 224)
        out = t(img)
        assert out.shape == (3, 224, 224)

    def test_point_cloud_transform(self):
        t = PointCloudTransforms(jitter_std=0.02, random_rotation=True)
        pc = torch.rand(1024, 3)
        out = t(pc)
        assert out.shape == (1024, 3)
