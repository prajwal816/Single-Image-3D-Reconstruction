"""
evaluate.py — CLI entry point for evaluating a trained model.

Usage:
    python evaluate.py --config configs/default.yaml --checkpoint experiments/run/best_model.pth
"""

import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.datasets.shapenet_simulated import SimulatedShapeNetDataset
from src.datasets.transforms import ImageTransforms
from src.models.reconstruction_net import SingleImageReconstructionNet
from src.evaluation.evaluator import Evaluator
from src.visualization.point_cloud_viz import PointCloudVisualizer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Single-Image 3D Reconstruction Model"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--num_viz", type=int, default=None,
                        help="Number of visualization samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    eval_cfg = cfg["evaluation"]

    device = cfg["experiment"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Build test dataset
    val_transform = ImageTransforms(
        color_jitter=False, random_crop=False, normalize=True,
        image_size=ds_cfg["image_size"],
    )
    test_ds = SimulatedShapeNetDataset(
        num_samples=ds_cfg["num_test_samples"],
        num_points=ds_cfg["num_points"],
        image_size=ds_cfg["image_size"],
        categories=ds_cfg["categories"],
        transform=val_transform,
        seed=cfg["experiment"]["seed"] + 200000,
    )

    test_loader = DataLoader(
        test_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0,
    )

    # Load model
    model = SingleImageReconstructionNet(
        encoder_cfg=cfg["model"]["encoder"],
        decoder_cfg=cfg["model"]["decoder"],
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("=" * 60)
    print("Single-Image 3D Reconstruction — Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Device: {device}")

    # Evaluate
    evaluator = Evaluator(
        model=model, device=device,
        iou_resolution=eval_cfg["iou_resolution"],
        completeness_threshold=eval_cfg["completeness_threshold"],
    )
    results = evaluator.evaluate(test_loader, test_ds.get_categories())
    Evaluator.print_results(results)

    # Save results
    output_dir = args.output or os.path.dirname(args.checkpoint)
    results_path = os.path.join(output_dir, "evaluation_results.json")
    Evaluator.save_results(results, results_path)

    # Generate visualizations
    num_viz = args.num_viz or eval_cfg["num_visualization_samples"]
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    print(f"\nGenerating {num_viz} visualization samples...")

    # Get raw samples (without normalization for display)
    raw_ds = SimulatedShapeNetDataset(
        num_samples=ds_cfg["num_test_samples"],
        num_points=ds_cfg["num_points"],
        image_size=ds_cfg["image_size"],
        categories=ds_cfg["categories"],
        seed=cfg["experiment"]["seed"] + 200000,
    )

    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i in range(min(num_viz, len(test_ds))):
            sample_raw = raw_ds[i]
            sample_norm = test_ds[i]

            image_input = sample_norm["image"].unsqueeze(0).to(device)
            pred_points = model(image_input).squeeze(0).cpu().numpy()

            image_display = sample_raw["image"].permute(1, 2, 0).numpy()
            gt_points = sample_raw["point_cloud"].numpy()
            category = sample_raw["category_name"]

            sample_metrics = results["per_sample"][i] if i < len(results["per_sample"]) else {}

            PointCloudVisualizer.plot_comparison(
                image=image_display,
                pred_points=pred_points,
                gt_points=gt_points,
                category=category,
                metrics={
                    "cd": sample_metrics.get("chamfer_distance", 0),
                    "iou": sample_metrics.get("iou", 0),
                    "completeness": sample_metrics.get("completeness", 0),
                },
                save_path=os.path.join(viz_dir, f"comparison_{i}_{category}.png"),
            )

    print(f"Visualizations saved to {viz_dir}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
