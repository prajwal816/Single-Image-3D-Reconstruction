"""
Unit tests for evaluation metrics.
"""

import pytest
import torch
import numpy as np
from src.evaluation.metrics import (
    compute_chamfer_distance,
    compute_iou,
    compute_reconstruction_completeness,
)


class TestChamferDistanceMetric:
    """Test the Chamfer Distance evaluation metric."""

    def test_identical(self):
        pts = torch.randn(2, 100, 3)
        cd = compute_chamfer_distance(pts, pts)
        assert cd.shape == (2,)
        assert (cd < 1e-5).all()

    def test_positive(self):
        a = torch.randn(3, 100, 3)
        b = torch.randn(3, 100, 3)
        cd = compute_chamfer_distance(a, b)
        assert (cd > 0).all()


class TestIoU:
    """Test the volumetric IoU metric."""

    def test_identical(self):
        pts = torch.randn(1, 500, 3) * 0.3
        iou = compute_iou(pts, pts, resolution=16)
        assert iou.shape == (1,)
        assert iou[0].item() > 0.9, "IoU should be ~1.0 for identical clouds"

    def test_disjoint(self):
        a = torch.ones(1, 100, 3) * 0.4
        b = torch.ones(1, 100, 3) * -0.4
        iou = compute_iou(a, b, resolution=16)
        assert iou[0].item() < 0.3, "IoU should be low for disjoint clouds"

    def test_batch(self):
        a = torch.randn(4, 200, 3) * 0.3
        b = torch.randn(4, 200, 3) * 0.3
        iou = compute_iou(a, b, resolution=16)
        assert iou.shape == (4,)


class TestReconstructionCompleteness:
    """Test the reconstruction completeness metric."""

    def test_identical(self):
        pts = torch.randn(1, 100, 3) * 0.3
        comp = compute_reconstruction_completeness(pts, pts, threshold=0.1)
        assert comp.shape == (1,)
        assert comp[0].item() > 0.99, "Completeness should be ~1.0 for identical"

    def test_far_apart(self):
        pred = torch.ones(1, 100, 3) * 10.0
        gt = torch.zeros(1, 100, 3)
        comp = compute_reconstruction_completeness(pred, gt, threshold=0.1)
        assert comp[0].item() < 0.01, "Completeness should be ~0 when far apart"

    def test_range(self):
        a = torch.randn(2, 200, 3) * 0.5
        b = torch.randn(2, 200, 3) * 0.5
        comp = compute_reconstruction_completeness(a, b, threshold=0.5)
        assert (comp >= 0).all() and (comp <= 1).all()
