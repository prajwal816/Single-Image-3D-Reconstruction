"""
Unit tests for loss functions.
"""

import pytest
import torch
from src.models.losses import ChamferDistanceLoss


class TestChamferDistanceLoss:
    """Test Chamfer Distance computation."""

    def test_zero_for_identical(self):
        loss_fn = ChamferDistanceLoss(reduction="mean")
        pts = torch.randn(2, 100, 3)
        loss = loss_fn(pts, pts)
        assert loss.item() < 1e-6, "CD should be ~0 for identical point clouds"

    def test_symmetric(self):
        loss_fn = ChamferDistanceLoss(reduction="mean")
        a = torch.randn(2, 100, 3)
        b = torch.randn(2, 100, 3)
        loss_ab = loss_fn(a, b)
        loss_ba = loss_fn(b, a)
        assert torch.allclose(loss_ab, loss_ba, atol=1e-5), \
            "CD should be symmetric"

    def test_positive(self):
        loss_fn = ChamferDistanceLoss(reduction="mean")
        a = torch.randn(4, 200, 3)
        b = torch.randn(4, 200, 3)
        loss = loss_fn(a, b)
        assert loss.item() > 0, "CD should be positive for different clouds"

    def test_different_point_counts(self):
        loss_fn = ChamferDistanceLoss(reduction="mean")
        a = torch.randn(2, 100, 3)
        b = torch.randn(2, 200, 3)
        loss = loss_fn(a, b)
        assert loss.shape == ()

    def test_reduction_none(self):
        loss_fn = ChamferDistanceLoss(reduction="none")
        a = torch.randn(4, 50, 3)
        b = torch.randn(4, 50, 3)
        loss = loss_fn(a, b)
        assert loss.shape == (4,)

    def test_reduction_sum(self):
        loss_fn = ChamferDistanceLoss(reduction="sum")
        a = torch.randn(3, 50, 3)
        b = torch.randn(3, 50, 3)
        loss = loss_fn(a, b)
        assert loss.shape == ()

    def test_gradient_flow(self):
        loss_fn = ChamferDistanceLoss(reduction="mean")
        pred = torch.randn(2, 50, 3, requires_grad=True)
        gt = torch.randn(2, 50, 3)
        loss = loss_fn(pred, gt)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
