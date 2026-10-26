"""
DermaStack-XAI Training Callbacks and Monitoring Module.
Implements EarlyStopping, ReduceLROnPlateau, Best Model Checkpointing,
and structured JSON Training History persistence.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import keras
from keras import callbacks

from src.utils.logger import get_logger

logger = get_logger("DermaStack-XAI.Callbacks")


class TrainingHistoryLogger(callbacks.Callback):
    """
    Custom Keras callback that records epoch-by-epoch training and validation metrics,
    learning rates, and execution time, serializing history to JSON for reproducibility.
    """

    def __init__(self, history_path: Union[str, Path]):
        super().__init__()
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self._epoch_start_time: float = 0.0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        self.history = []

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        self._epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        elapsed = round(time.time() - self._epoch_start_time, 2)

        # Extract current learning rate
        lr = None
        if hasattr(self.model, "optimizer") and hasattr(self.model.optimizer, "learning_rate"):
            try:
                lr_val = self.model.optimizer.learning_rate
                if hasattr(lr_val, "numpy"):
                    lr = float(lr_val.numpy())
                elif callable(lr_val):
                    lr = float(lr_val(self.model.optimizer.iterations).numpy())
                else:
                    lr = float(lr_val)
            except Exception:
                lr = None

        epoch_record: Dict[str, Any] = {
            "epoch": int(epoch + 1),
            "elapsed_seconds": elapsed,
            "learning_rate": lr
        }

        # Cleanly serialize scalar metrics
        for k, v in logs.items():
            if hasattr(v, "item"):
                epoch_record[k] = float(v.item())
            elif isinstance(v, (float, int)):
                epoch_record[k] = float(v)
            else:
                epoch_record[k] = str(v)

        self.history.append(epoch_record)

        # Write incrementally to disk so partial progress is never lost
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump({
                    "epochs_completed": len(self.history),
                    "history": self.history
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write incremental training history to {self.history_path}: {e}")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump({
                    "epochs_completed": len(self.history),
                    "history": self.history
                }, f, indent=2)
            logger.info(f"Persisted complete training history ({len(self.history)} epochs) to {self.history_path}")
        except Exception as e:
            logger.error(f"Failed to write final training history to {self.history_path}: {e}")


def get_training_callbacks(
    model_name: str,
    fold_idx: int,
    checkpoint_dir: Union[str, Path],
    early_stopping_patience: int = 7,
    reduce_lr_patience: int = 3,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    monitor: str = "val_accuracy",
    mode: str = "max",
    verbose: int = 1
) -> Tuple[List[callbacks.Callback], Path, Path]:
    """
    Constructs the standard suite of training callbacks for a given model and fold.

    Callbacks created:
        1. ModelCheckpoint: Saves the best performing model weights in .keras format.
        2. EarlyStopping: Restores best weights when the selected validation metric stops improving.
        3. ReduceLROnPlateau: Decays learning rate when performance plateaus.
        4. TrainingHistoryLogger: Persists epoch-by-epoch loss and metrics to JSON.

    Args:
        model_name: Architecture identifier ('custom_cnn', 'efficientnet_b0', 'resnet50').
        fold_idx: Fold index (0..4).
        checkpoint_dir: Directory where checkpoints and history logs are saved.
        early_stopping_patience: Number of epochs with no improvement before stopping. Default 7.
        reduce_lr_patience: Number of epochs with no improvement before reducing LR. Default 3.
        reduce_lr_factor: Multiplier for learning rate reduction. Default 0.5.
        min_lr: Lower bound on learning rate. Default 1e-6.
        monitor: Metric to monitor. Default 'val_accuracy'.
        mode: One of 'min' or 'max'. Default 'max' for val_accuracy.
        verbose: Verbosity level.

    Returns:
        (callback_list, checkpoint_path, history_path)
    """
    if mode not in {"min", "max", "auto"}:
        raise ValueError("mode must be 'min', 'max', or 'auto'")
    if not monitor:
        raise ValueError("monitor must be a non-empty metric name")

    chk_dir = Path(checkpoint_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)

    sanitized_name = model_name.lower().replace("-", "_")
    # Versioned filenames prevent the legacy val_loss-selected checkpoints/history
    # from being silently reused after changing the selection criterion.
    metric_tag = monitor.replace("/", "_").replace("-", "_")
    checkpoint_path = chk_dir / f"{sanitized_name}_fold_{fold_idx}_best_{metric_tag}.keras"
    history_path = chk_dir / f"{sanitized_name}_fold_{fold_idx}_history_{metric_tag}.json"

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        verbose=verbose
    )

    early_stop_cb = callbacks.EarlyStopping(
        monitor=monitor,
        patience=early_stopping_patience,
        restore_best_weights=True,
        mode=mode,
        verbose=verbose
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        mode=mode,
        verbose=verbose
    )

    history_cb = TrainingHistoryLogger(history_path=history_path)

    cb_list = [checkpoint_cb, early_stop_cb, reduce_lr_cb, history_cb]
    return cb_list, checkpoint_path, history_path
