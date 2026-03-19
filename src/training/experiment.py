"""
Experiment Tracker — logs metrics, hyperparams, and config snapshots.

Provides CSV-based metric logging alongside TensorBoard integration
for comprehensive experiment tracking.
"""

import os
import csv
import json
import shutil
import yaml
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class ExperimentTracker:
    """
    Tracks training experiments with structured logging.

    Creates an experiment directory with:
        <output_dir>/<experiment_name>/
        ├── config.yaml          # Frozen config snapshot
        ├── metrics.csv          # Epoch-level metrics
        ├── train_log.csv        # Step-level training log
        ├── metadata.json        # Experiment metadata
        └── tensorboard/         # TensorBoard log directory

    Args:
        experiment_name: Unique experiment identifier.
        output_dir:      Root experiments directory.
        config:          Full configuration dict to snapshot.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "experiments",
        config: Optional[Dict] = None,
    ):
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Save config snapshot
        if config is not None:
            config_path = os.path.join(self.experiment_dir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

        # Metadata
        self._metadata = {
            "experiment_name": experiment_name,
            "created_at": datetime.now().isoformat(),
            "status": "running",
        }
        self._save_metadata()

        # CSV writers
        self._metrics_path = os.path.join(self.experiment_dir, "metrics.csv")
        self._train_log_path = os.path.join(self.experiment_dir, "train_log.csv")
        self._metrics_writer = None
        self._train_log_writer = None
        self._metrics_file = None
        self._train_log_file = None

        # TensorBoard
        self.tb_writer = None
        if HAS_TENSORBOARD:
            tb_dir = os.path.join(self.experiment_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

    def log_step(self, step: int, metrics: Dict[str, float]):
        """Log step-level training metrics (loss, lr, etc.)."""
        if self._train_log_writer is None:
            self._train_log_file = open(self._train_log_path, "w", newline="")
            fieldnames = ["step"] + sorted(metrics.keys())
            self._train_log_writer = csv.DictWriter(
                self._train_log_file, fieldnames=fieldnames
            )
            self._train_log_writer.writeheader()

        row = {"step": step, **metrics}
        self._train_log_writer.writerow(row)
        self._train_log_file.flush()

        # TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"train/{key}", value, step)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch-level metrics (val loss, IoU, etc.)."""
        if self._metrics_writer is None:
            self._metrics_file = open(self._metrics_path, "w", newline="")
            fieldnames = ["epoch"] + sorted(metrics.keys())
            self._metrics_writer = csv.DictWriter(
                self._metrics_file, fieldnames=fieldnames
            )
            self._metrics_writer.writeheader()

        row = {"epoch": epoch, **metrics}
        self._metrics_writer.writerow(row)
        self._metrics_file.flush()

        # TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"epoch/{key}", value, epoch)

    def log_text(self, tag: str, text: str, step: int = 0):
        """Log text data (e.g., model summary)."""
        if self.tb_writer is not None:
            self.tb_writer.add_text(tag, text, step)

    def get_experiment_dir(self) -> str:
        """Return the experiment directory path."""
        return self.experiment_dir

    def mark_complete(self, final_metrics: Optional[Dict] = None):
        """Mark experiment as completed."""
        self._metadata["status"] = "completed"
        self._metadata["completed_at"] = datetime.now().isoformat()
        if final_metrics:
            self._metadata["final_metrics"] = final_metrics
        self._save_metadata()
        self.close()

    def close(self):
        """Close all open file handles."""
        if self._metrics_file is not None:
            self._metrics_file.close()
        if self._train_log_file is not None:
            self._train_log_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()

    def _save_metadata(self):
        """Persist metadata JSON."""
        path = os.path.join(self.experiment_dir, "metadata.json")
        with open(path, "w") as f:
            json.dump(self._metadata, f, indent=2)
