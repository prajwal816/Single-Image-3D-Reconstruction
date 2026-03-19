"""
explain.py — CLI entry point for model explainability analysis.

Usage:
    python explain.py --config configs/default.yaml --checkpoint experiments/run/best_model.pth
    python explain.py --config configs/default.yaml --checkpoint <path> --method gradient
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
from src.models.losses import ChamferDistanceLoss
from src.explainability.shap_analysis import SHAPAnalyzer
from src.explainability.feature_importance import GradientSaliency, OcclusionSensitivity


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Explainability Analysis for 3D Reconstruction"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, default=None,
                        choices=["shap", "gradient", "occlusion"],
                        help="Explainability method")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = cfg["explainability"]
    ds_cfg = cfg["dataset"]

    method = args.method or exp_cfg["method"]
    num_samples = args.num_samples or exp_cfg["num_explain_samples"]

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

    # Build dataset
    val_transform = ImageTransforms(
        color_jitter=False, random_crop=False, normalize=True,
        image_size=ds_cfg["image_size"],
    )
    dataset = SimulatedShapeNetDataset(
        num_samples=max(num_samples * 2, 100),
        num_points=ds_cfg["num_points"],
        image_size=ds_cfg["image_size"],
        categories=ds_cfg["categories"],
        transform=val_transform,
        seed=cfg["experiment"]["seed"] + 300000,
    )
    raw_dataset = SimulatedShapeNetDataset(
        num_samples=max(num_samples * 2, 100),
        num_points=ds_cfg["num_points"],
        image_size=ds_cfg["image_size"],
        categories=ds_cfg["categories"],
        seed=cfg["experiment"]["seed"] + 300000,
    )

    output_dir = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "explainability"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Explainability Analysis — Method: {method.upper()}")
    print("=" * 60)
    print(f"Analyzing {num_samples} samples...")

    if method == "shap":
        analyzer = SHAPAnalyzer(model, device=device,
                                num_background_samples=exp_cfg["num_background_samples"])

        images = torch.stack([dataset[i]["image"] for i in range(num_samples)])
        raw_images = np.stack([
            raw_dataset[i]["image"].permute(1, 2, 0).numpy()
            for i in range(num_samples)
        ])
        categories = [raw_dataset[i]["category_name"] for i in range(num_samples)]

        analyzer.analyze_batch(images, raw_images, categories, output_dir)

    elif method == "gradient":
        saliency = GradientSaliency(model, device=device)

        for i in range(num_samples):
            sample = dataset[i]
            raw_sample = raw_dataset[i]

            image = sample["image"].unsqueeze(0)
            sal_map = saliency.compute(image)

            raw_img = raw_sample["image"].permute(1, 2, 0).numpy()
            category = raw_sample["category_name"]

            saliency.plot(
                raw_img, sal_map,
                title=f"Gradient Saliency — {category}",
                save_path=os.path.join(output_dir, f"saliency_{category}_{i}.png"),
            )

    elif method == "occlusion":
        criterion = ChamferDistanceLoss(reduction="mean")
        occluder = OcclusionSensitivity(
            model, criterion, device=device,
            patch_size=exp_cfg["occlusion_patch_size"],
            stride=exp_cfg["occlusion_stride"],
        )

        for i in range(num_samples):
            sample = dataset[i]
            raw_sample = raw_dataset[i]

            image = sample["image"].unsqueeze(0)
            gt_points = sample["point_cloud"].unsqueeze(0)

            sens_map = occluder.compute(image, gt_points)
            raw_img = raw_sample["image"].permute(1, 2, 0).numpy()
            category = raw_sample["category_name"]

            occluder.plot(
                raw_img, sens_map,
                title=f"Occlusion Sensitivity — {category}",
                save_path=os.path.join(output_dir, f"occlusion_{category}_{i}.png"),
            )

    print(f"\nResults saved to {output_dir}")
    print("Explainability analysis complete!")


if __name__ == "__main__":
    main()
