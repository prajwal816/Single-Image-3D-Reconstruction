"""
Training Visualization — loss curves, IoU curves, LR schedule plots.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class TrainingVisualizer:
    """
    Generates training progress plots from experiment logs.

    Reads CSV-based metric files produced by ExperimentTracker
    and generates publication-quality training curves.
    """

    @staticmethod
    def plot_training_curves(
        metrics_csv: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot training and validation loss curves from metrics.csv.

        Args:
            metrics_csv: Path to the epoch-level metrics CSV.
            save_path:   If provided, save figure.
        Returns:
            matplotlib Figure.
        """
        epochs, train_losses, val_losses, lrs = [], [], [], []

        with open(metrics_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row.get("train_loss", 0)))
                if "val_loss" in row and row["val_loss"]:
                    val_losses.append(float(row["val_loss"]))
                if "lr" in row and row["lr"]:
                    lrs.append(float(row["lr"]))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        ax = axes[0]
        ax.plot(epochs, train_losses, "b-", linewidth=1.5, label="Train Loss",
                alpha=0.9)
        if val_losses:
            val_epochs = epochs[:len(val_losses)]
            ax.plot(val_epochs, val_losses, "r-", linewidth=1.5,
                    label="Val Loss", alpha=0.9)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Chamfer Distance Loss", fontsize=11)
        ax.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # Learning rate schedule
        ax2 = axes[1]
        if lrs:
            ax2.plot(epochs[:len(lrs)], lrs, "g-", linewidth=1.5)
            ax2.set_xlabel("Epoch", fontsize=11)
            ax2.set_ylabel("Learning Rate", fontsize=11)
            ax2.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale("log")
        else:
            ax2.text(0.5, 0.5, "No LR data", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=14, color="gray")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig

    @staticmethod
    def plot_step_losses(
        train_log_csv: str,
        save_path: Optional[str] = None,
        smooth_window: int = 20,
    ) -> plt.Figure:
        """
        Plot step-level training loss with smoothing.

        Args:
            train_log_csv: Path to step-level training log.
            save_path:     If provided, save figure.
            smooth_window: Window size for moving average smoothing.
        Returns:
            matplotlib Figure.
        """
        steps, losses = [], []
        with open(train_log_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))

        fig, ax = plt.subplots(figsize=(10, 4))

        # Raw loss (transparent)
        ax.plot(steps, losses, "b-", alpha=0.15, linewidth=0.5, label="Raw")

        # Smoothed loss
        if len(losses) > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(losses, kernel, mode="valid")
            s_steps = steps[smooth_window - 1:]
            ax.plot(s_steps, smoothed, "b-", linewidth=1.5, label="Smoothed")

        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Step-Level Training Loss", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig

    @staticmethod
    def plot_from_history(
        history: Dict[str, list],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot from the history dict returned by Trainer.train().

        Args:
            history: Dict with train_losses, val_losses, learning_rates.
            save_path: If provided, save figure.
        Returns:
            matplotlib Figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Train loss
        ax = axes[0]
        ax.plot(history["train_losses"], "b-", linewidth=1.5)
        ax.set_title("Train Loss", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Chamfer Distance")
        ax.grid(True, alpha=0.3)

        # Val loss
        ax = axes[1]
        if history.get("val_losses"):
            ax.plot(history["val_losses"], "r-", linewidth=1.5)
        ax.set_title("Validation Loss", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Chamfer Distance")
        ax.grid(True, alpha=0.3)

        # LR
        ax = axes[2]
        if history.get("learning_rates"):
            ax.plot(history["learning_rates"], "g-", linewidth=1.5)
        ax.set_title("Learning Rate", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig
