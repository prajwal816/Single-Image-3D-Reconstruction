"""
Evaluator — runs full evaluation on a dataset split and aggregates metrics.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm

from .metrics import compute_chamfer_distance, compute_iou, compute_reconstruction_completeness


class Evaluator:
    """
    Evaluates a trained reconstruction model on a dataset split.

    Computes per-sample and per-category metrics:
        - Chamfer Distance
        - IoU
        - Reconstruction Completeness

    Args:
        model:       Trained reconstruction model.
        device:      Device string.
        iou_resolution: Voxel grid resolution for IoU.
        completeness_threshold: Distance threshold for completeness.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        iou_resolution: int = 32,
        completeness_threshold: float = 0.05,
    ):
        self.model = model.to(device)
        self.device = device
        self.iou_resolution = iou_resolution
        self.completeness_threshold = completeness_threshold

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        categories: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run evaluation on the entire dataloader.

        Returns:
            Dict with:
                - "overall": {cd, iou, completeness}
                - "per_category": {category_name: {cd, iou, completeness}}
                - "per_sample": [{cd, iou, completeness, category}, ...]
        """
        self.model.eval()

        all_cd = []
        all_iou = []
        all_comp = []
        all_categories = []
        per_sample = []

        cat_metrics = defaultdict(lambda: {"cd": [], "iou": [], "completeness": []})

        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(self.device)
            gt_points = batch["point_cloud"].to(self.device)
            cat_names = batch["category_name"]

            pred_points = self.model(images)

            # Compute metrics
            cd = compute_chamfer_distance(pred_points, gt_points)
            iou = compute_iou(pred_points, gt_points, self.iou_resolution)
            comp = compute_reconstruction_completeness(
                pred_points, gt_points, self.completeness_threshold
            )

            for i in range(len(cd)):
                sample_result = {
                    "chamfer_distance": cd[i].item(),
                    "iou": iou[i].item(),
                    "completeness": comp[i].item(),
                    "category": cat_names[i] if isinstance(cat_names[i], str) else cat_names[i],
                }
                per_sample.append(sample_result)

                cat = sample_result["category"]
                cat_metrics[cat]["cd"].append(cd[i].item())
                cat_metrics[cat]["iou"].append(iou[i].item())
                cat_metrics[cat]["completeness"].append(comp[i].item())

                all_cd.append(cd[i].item())
                all_iou.append(iou[i].item())
                all_comp.append(comp[i].item())

        # Aggregate
        import numpy as np

        overall = {
            "chamfer_distance": float(np.mean(all_cd)),
            "iou": float(np.mean(all_iou)),
            "completeness": float(np.mean(all_comp)),
            "num_samples": len(all_cd),
        }

        per_category = {}
        for cat, metrics in cat_metrics.items():
            per_category[cat] = {
                "chamfer_distance": float(np.mean(metrics["cd"])),
                "iou": float(np.mean(metrics["iou"])),
                "completeness": float(np.mean(metrics["completeness"])),
                "num_samples": len(metrics["cd"]),
            }

        return {
            "overall": overall,
            "per_category": per_category,
            "per_sample": per_sample,
        }

    @staticmethod
    def save_results(results: Dict, output_path: str):
        """Save evaluation results to JSON."""
        # Remove per-sample for compact file, keep overall + per_category
        compact = {
            "overall": results["overall"],
            "per_category": results["per_category"],
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(compact, f, indent=2)
        print(f"Results saved to {output_path}")

    @staticmethod
    def print_results(results: Dict):
        """Pretty-print evaluation results."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        overall = results["overall"]
        print(f"\nOverall ({overall['num_samples']} samples):")
        print(f"  Chamfer Distance : {overall['chamfer_distance']:.6f}")
        print(f"  IoU              : {overall['iou']:.4f}")
        print(f"  Completeness     : {overall['completeness']:.4f}")

        print(f"\nPer-Category:")
        print(f"  {'Category':<15} {'CD':>10} {'IoU':>8} {'Comp':>8} {'N':>6}")
        print(f"  {'-' * 50}")
        for cat, m in sorted(results["per_category"].items()):
            print(f"  {cat:<15} {m['chamfer_distance']:>10.6f} "
                  f"{m['iou']:>8.4f} {m['completeness']:>8.4f} "
                  f"{m['num_samples']:>6}")

        print("=" * 60)
