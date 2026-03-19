"""
train.py — CLI entry point for training the 3D reconstruction model.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume experiments/run/best_model.pth
"""

import argparse
import os
import sys
import logging
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils.logging import setup_logging, get_logger
from src.datasets.shapenet_simulated import SimulatedShapeNetDataset
from src.datasets.transforms import ImageTransforms, PointCloudTransforms
from src.models.reconstruction_net import SingleImageReconstructionNet
from src.models.losses import ChamferDistanceLoss
from src.training.trainer import Trainer
from src.training.experiment import ExperimentTracker
from src.visualization.training_viz import TrainingVisualizer

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_datasets(cfg: dict):
    """Build training and validation datasets from config."""
    ds_cfg = cfg["dataset"]
    aug_cfg = ds_cfg.get("augmentation", {})

    # Image transforms
    train_transform = ImageTransforms(
        color_jitter=aug_cfg.get("color_jitter", True),
        random_crop=aug_cfg.get("random_crop", True),
        normalize=True,
        image_size=ds_cfg["image_size"],
    )
    val_transform = ImageTransforms(
        color_jitter=False, random_crop=False, normalize=True,
        image_size=ds_cfg["image_size"],
    )

    # Point cloud transforms
    train_pc_transform = None
    if aug_cfg.get("point_jitter", 0) > 0 or aug_cfg.get("random_rotation", False):
        train_pc_transform = PointCloudTransforms(
            jitter_std=aug_cfg.get("point_jitter", 0.02),
            random_rotation=aug_cfg.get("random_rotation", True),
        )

    train_ds = SimulatedShapeNetDataset(
        num_samples=ds_cfg["num_train_samples"],
        num_points=ds_cfg["num_points"],
        image_size=ds_cfg["image_size"],
        categories=ds_cfg["categories"],
        transform=train_transform,
        point_transform=train_pc_transform,
        seed=cfg["experiment"]["seed"],
    )

    val_ds = SimulatedShapeNetDataset(
        num_samples=ds_cfg["num_val_samples"],
        num_points=ds_cfg["num_points"],
        image_size=ds_cfg["image_size"],
        categories=ds_cfg["categories"],
        transform=val_transform,
        seed=cfg["experiment"]["seed"] + 100000,
    )

    return train_ds, val_ds


def build_model(cfg: dict) -> SingleImageReconstructionNet:
    """Build model from config."""
    return SingleImageReconstructionNet(
        encoder_cfg=cfg["model"]["encoder"],
        decoder_cfg=cfg["model"]["decoder"],
    )


def build_optimizer(model, cfg: dict):
    """Build optimizer and scheduler from config."""
    train_cfg = cfg["training"]

    # Optimizer
    if train_cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
    elif train_cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            momentum=0.9,
            weight_decay=train_cfg["weight_decay"],
        )

    # Scheduler
    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = None
    if sched_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg["epochs"],
            eta_min=sched_cfg.get("min_lr", 1e-6),
        )
    elif sched_cfg.get("type") == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 30),
            gamma=sched_cfg.get("gamma", 0.1),
        )
    elif sched_cfg.get("type") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sched_cfg.get("gamma", 0.5),
            patience=sched_cfg.get("patience", 10),
        )

    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(
        description="Train Single-Image 3D Reconstruction Model"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    exp_cfg = cfg["experiment"]

    # Setup logging (file log goes into experiment dir)
    log_dir = os.path.join(exp_cfg["output_dir"], exp_cfg["name"])
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(level="INFO", log_file=os.path.join(log_dir, "training.log"))

    # Seed
    seed = exp_cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = exp_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info("=" * 60)
    logger.info("Single-Image 3D Reconstruction — Training")
    logger.info("=" * 60)
    logger.info("Experiment: %s", exp_cfg["name"])
    logger.info("Device:     %s", device)

    # Build components
    train_ds, val_ds = build_datasets(cfg)
    logger.info("Train samples: %d, Val samples: %d", len(train_ds), len(val_ds))
    logger.info("Categories: %s", train_ds.get_categories())

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=exp_cfg.get("num_workers", 0),
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=exp_cfg.get("num_workers", 0),
        pin_memory=(device == "cuda"),
    )

    model = build_model(cfg)
    param_counts = model.get_num_params()
    logger.info(
        "Model parameters: %s (encoder: %s, decoder: %s)",
        f"{param_counts['total']:,}",
        f"{param_counts['encoder']:,}",
        f"{param_counts['decoder']:,}",
    )

    criterion = ChamferDistanceLoss(reduction=cfg["loss"]["reduction"])
    optimizer, scheduler = build_optimizer(model, cfg)

    # Experiment tracker
    tracker = ExperimentTracker(
        experiment_name=exp_cfg["name"],
        output_dir=exp_cfg["output_dir"],
        config=cfg,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        tracker=tracker,
        config=cfg["training"],
    )

    # Resume if needed
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training for %d epochs...", cfg["training"]["epochs"])
    history = trainer.train(
        train_loader, val_loader, cfg["training"]["epochs"]
    )

    # Save training curves
    viz_dir = os.path.join(tracker.get_experiment_dir(), "plots")
    os.makedirs(viz_dir, exist_ok=True)
    TrainingVisualizer.plot_from_history(
        history, save_path=os.path.join(viz_dir, "training_curves.png")
    )
    logger.info("Training curves saved to %s", viz_dir)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
