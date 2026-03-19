"""
visualize.py — CLI entry point for visualization.

Usage:
    python visualize.py --experiment experiments/run --mode training
    python visualize.py --experiment experiments/run --mode reconstruction --checkpoint <path>
"""

import argparse
import os
import yaml
import torch
import numpy as np

from src.visualization.training_viz import TrainingVisualizer
from src.visualization.point_cloud_viz import PointCloudVisualizer
from src.datasets.shapenet_simulated import SimulatedShapeNetDataset
from src.datasets.transforms import ImageTransforms
from src.models.reconstruction_net import SingleImageReconstructionNet


def main():
    parser = argparse.ArgumentParser(
        description="Visualization for 3D Reconstruction"
    )
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment directory")
    parser.add_argument("--mode", type=str, default="training",
                        choices=["training", "reconstruction", "comparison"],
                        help="Visualization mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint (for reconstruction/comparison)")
    parser.add_argument("--num_samples", type=int, default=8)
    args = parser.parse_args()

    output_dir = os.path.join(args.experiment, "plots")
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == "training":
        print("Generating training visualizations...")

        metrics_csv = os.path.join(args.experiment, "metrics.csv")
        train_log = os.path.join(args.experiment, "train_log.csv")

        if os.path.exists(metrics_csv):
            TrainingVisualizer.plot_training_curves(
                metrics_csv,
                save_path=os.path.join(output_dir, "training_curves.png"),
            )
            print("  → training_curves.png")

        if os.path.exists(train_log):
            TrainingVisualizer.plot_step_losses(
                train_log,
                save_path=os.path.join(output_dir, "step_losses.png"),
            )
            print("  → step_losses.png")

    elif args.mode in ("reconstruction", "comparison"):
        if args.checkpoint is None:
            # Try default path
            ckpt = os.path.join(args.experiment, "best_model.pth")
            if not os.path.exists(ckpt):
                print("Error: --checkpoint required for reconstruction mode")
                return
            args.checkpoint = ckpt

        # Load config
        config_path = os.path.join(args.experiment, "config.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        device = cfg["experiment"]["device"]
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        # Load model
        model = SingleImageReconstructionNet(
            encoder_cfg=cfg["model"]["encoder"],
            decoder_cfg=cfg["model"]["decoder"],
        )
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model = model.to(device)

        ds_cfg = cfg["dataset"]

        # Raw dataset for display
        raw_ds = SimulatedShapeNetDataset(
            num_samples=args.num_samples * 2,
            num_points=ds_cfg["num_points"],
            image_size=ds_cfg["image_size"],
            categories=ds_cfg["categories"],
            seed=cfg["experiment"]["seed"] + 500000,
        )

        # Normalized dataset for inference
        norm_transform = ImageTransforms(
            color_jitter=False, random_crop=False, normalize=True,
            image_size=ds_cfg["image_size"],
        )
        norm_ds = SimulatedShapeNetDataset(
            num_samples=args.num_samples * 2,
            num_points=ds_cfg["num_points"],
            image_size=ds_cfg["image_size"],
            categories=ds_cfg["categories"],
            transform=norm_transform,
            seed=cfg["experiment"]["seed"] + 500000,
        )

        print(f"Generating {args.num_samples} reconstruction visualizations...")

        with torch.no_grad():
            for i in range(args.num_samples):
                sample_raw = raw_ds[i]
                sample_norm = norm_ds[i]

                img_input = sample_norm["image"].unsqueeze(0).to(device)
                pred = model(img_input).squeeze(0).cpu().numpy()

                image_display = sample_raw["image"].permute(1, 2, 0).numpy()
                gt_points = sample_raw["point_cloud"].numpy()
                category = sample_raw["category_name"]

                PointCloudVisualizer.plot_comparison(
                    image=image_display,
                    pred_points=pred,
                    gt_points=gt_points,
                    category=category,
                    save_path=os.path.join(output_dir, f"recon_{i}_{category}.png"),
                )
                print(f"  → recon_{i}_{category}.png")

        # Also make a grid
        samples = []
        with torch.no_grad():
            for i in range(min(args.num_samples, 16)):
                sample_norm = norm_ds[i]
                sample_raw = raw_ds[i]
                img_input = sample_norm["image"].unsqueeze(0).to(device)
                pred = model(img_input).squeeze(0).cpu().numpy()
                samples.append({
                    "pred_points": pred,
                    "category": sample_raw["category_name"],
                })

        PointCloudVisualizer.plot_comparison_grid(
            samples,
            save_path=os.path.join(output_dir, "reconstruction_grid.png"),
        )
        print("  → reconstruction_grid.png")

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
