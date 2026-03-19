"""
Trainer — full training loop with validation, checkpointing, and logging.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from .experiment import ExperimentTracker


class Trainer:
    """
    Manages the full training lifecycle for the reconstruction model.

    Features:
        - Training + validation loops with progress bars
        - LR scheduling (cosine, step, plateau)
        - Gradient clipping
        - Best-model + periodic checkpoint saving
        - Integrated experiment tracking (CSV + TensorBoard)
        - Resume from checkpoint

    Args:
        model:      The reconstruction network.
        criterion:  Loss function (e.g., ChamferDistanceLoss).
        optimizer:  PyTorch optimizer.
        scheduler:  LR scheduler (optional).
        device:     Device string ("cuda" or "cpu").
        tracker:    ExperimentTracker instance.
        config:     Training config dict.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        tracker: Optional[ExperimentTracker] = None,
        config: Optional[Dict] = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.tracker = tracker
        self.config = config or {}

        self.gradient_clip = self.config.get("gradient_clip", 1.0)
        self.checkpoint_interval = self.config.get("checkpoint_interval", 10)
        self.log_interval = self.config.get("log_interval", 50)
        self.val_interval = self.config.get("val_interval", 1)

        self.best_val_loss = float("inf")
        self.global_step = 0
        self.start_epoch = 0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> Dict[str, list]:
        """
        Run the full training loop.

        Returns:
            Dict with keys: train_losses, val_losses, learning_rates.
        """
        history = {
            "train_losses": [],
            "val_losses": [],
            "learning_rates": [],
        }

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()

            # ---- Training phase ----
            train_loss = self._train_epoch(train_loader, epoch, num_epochs)
            history["train_losses"].append(train_loss)
            history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            # ---- Validation phase ----
            val_loss = None
            if (epoch + 1) % self.val_interval == 0:
                val_loss = self._validate(val_loader, epoch)
                history["val_losses"].append(val_loss)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)

            # ---- LR scheduling ----
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # ---- Periodic checkpoint ----
            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch, is_best=False)

            # ---- Log epoch metrics ----
            epoch_time = time.time() - epoch_start
            epoch_metrics = {
                "train_loss": train_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time_s": epoch_time,
            }
            if val_loss is not None:
                epoch_metrics["val_loss"] = val_loss
                epoch_metrics["best_val_loss"] = self.best_val_loss

            if self.tracker is not None:
                self.tracker.log_epoch(epoch, epoch_metrics)

            val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss_str:>10} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")

        # Final checkpoint
        self._save_checkpoint(num_epochs - 1, is_best=False, suffix="final")

        if self.tracker is not None:
            self.tracker.mark_complete({
                "final_train_loss": history["train_losses"][-1],
                "best_val_loss": self.best_val_loss,
            })

        return history

    def _train_epoch(self, loader: DataLoader, epoch: int,
                     num_epochs: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                    leave=False)

        for batch in pbar:
            images = batch["image"].to(self.device)
            gt_points = batch["point_cloud"].to(self.device)

            # Forward
            pred_points = self.model(images)
            loss = self.criterion(pred_points, gt_points)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

            if self.global_step % self.log_interval == 0 and self.tracker:
                self.tracker.log_step(self.global_step, {
                    "loss": loss.item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader, epoch: int) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1} [Val]", leave=False):
            images = batch["image"].to(self.device)
            gt_points = batch["point_cloud"].to(self.device)

            pred_points = self.model(images)
            loss = self.criterion(pred_points, gt_points)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, epoch: int, is_best: bool = False,
                         suffix: str = ""):
        """Save model checkpoint."""
        if self.tracker is None:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        exp_dir = self.tracker.get_experiment_dir()

        if is_best:
            path = os.path.join(exp_dir, "best_model.pth")
            torch.save(checkpoint, path)

        if suffix:
            path = os.path.join(exp_dir, f"checkpoint_{suffix}.pth")
        else:
            path = os.path.join(exp_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """Resume training from a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.global_step = checkpoint.get("global_step", 0)
        self.start_epoch = checkpoint.get("epoch", 0) + 1

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Resumed from epoch {self.start_epoch}, "
              f"global_step {self.global_step}")
