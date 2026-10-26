"""
DermaStack-XAI Five-Fold Cross-Validation and Out-Of-Fold (OOF) Prediction Engine.
Coordinates leakage-safe training, checkpointing, evaluation, and OOF probability generation
for Custom CNN, EfficientNetB0, and ResNet50 across 5 Stratified Group folds.
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
import keras

from src.config import DEFAULT_CONFIG, GlobalConfig, HAM10000_CLASSES
from src.data.transforms import (
    CLASS_TO_IDX,
    compute_class_weights,
    create_dataset,
    encode_labels
)
from src.models import (
    build_custom_cnn,
    build_efficientnet_b0,
    build_resnet50,
    validate_output_tensor
)
from src.training.callbacks import get_training_callbacks
from src.utils.logger import get_logger

logger = get_logger("DermaStack-XAI.Trainer")

SUPPORTED_MODELS: List[str] = ["custom_cnn", "efficientnet_b0", "resnet50"]
MODEL_COLUMN_SLICES: Dict[str, Tuple[int, int]] = {
    "custom_cnn": (0, 7),
    "efficientnet_b0": (7, 14),
    "resnet50": (14, 21)
}


class FiveFoldCrossValidationEngine:
    """
    Orchestrates 5-fold cross-validation on the HAM10000 training partition (7,153 images).
    Guarantees strict lesion-group isolation and generates the leakage-safe
    Out-Of-Fold (OOF) prediction probability matrix (7153, 21) for meta-model stacking.
    """

    def __init__(self, config: Optional[GlobalConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.random_seed = self.config.random_seed
        self.paths = self.config.paths
        self.dataset_cfg = self.config.dataset
        self.model_cfg = self.config.model

        # Ensure output directories exist
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.paths.oof_dir.mkdir(parents=True, exist_ok=True)
        self.paths.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata and partition manifest
        self.train_df, self.manifest_data = self._load_and_validate_partitions()

        # Generate 5-fold split indices strictly on training subset
        self.folds = self._generate_5_fold_splits()

    def _load_and_validate_partitions(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads validated metadata CSV and the split manifest, verifying partition sizes.
        Guarantees that only the 7,153 training images are accessed for OOF generation.
        """
        metadata_csv = self.paths.processed_data_dir / "validated_metadata.csv"
        manifest_json = self.paths.splits_dir / "train_val_test_splits.json"

        if not metadata_csv.exists():
            raise FileNotFoundError(f"Missing validated metadata CSV at: {metadata_csv}")
        if not manifest_json.exists():
            raise FileNotFoundError(f"Missing split manifest JSON at: {manifest_json}")

        metadata_df = pd.read_csv(metadata_csv)
        with open(manifest_json, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        train_ids = manifest_data.get("splits", {}).get("train", [])
        val_ids = manifest_data.get("splits", {}).get("val", [])
        test_ids = manifest_data.get("splits", {}).get("test", [])

        # Validate partition counts against contract
        assert len(train_ids) == 7153, f"Expected 7,153 train images, got {len(train_ids)}"
        assert len(val_ids) == 1431, f"Expected 1,431 val images, got {len(val_ids)}"
        assert len(test_ids) == 1431, f"Expected 1,431 test images, got {len(test_ids)}"

        # Strict Holdout Test Protection Assertion
        assert len(set(train_ids).intersection(set(test_ids))) == 0, "CRITICAL: Train and Test IDs overlap!"
        assert len(set(val_ids).intersection(set(test_ids))) == 0, "CRITICAL: Val and Test IDs overlap!"

        # Extract train dataframe preserving manifest order
        train_id_set = set(train_ids)
        train_df = metadata_df[metadata_df["image_id"].isin(train_id_set)].copy()

        # Sort train_df to strictly match manifest train_ids ordering
        id_order = {img_id: idx for idx, img_id in enumerate(train_ids)}
        train_df["manifest_order"] = train_df["image_id"].map(id_order)
        train_df = train_df.sort_values("manifest_order").drop(columns=["manifest_order"]).reset_index(drop=True)

        assert len(train_df) == 7153, f"Mismatch in train_df length: {len(train_df)} vs 7153"
        logger.info(f"Loaded {len(train_df)} training records from manifest.")
        return train_df, manifest_data

    def _generate_5_fold_splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partitions the 7,153 training records into 5 folds using StratifiedGroupKFold on lesion_id.
        Verifies mathematically zero lesion overlap between train and val in every fold.
        """
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        folds = list(sgkf.split(
            X=self.train_df,
            y=self.train_df["dx"],
            groups=self.train_df["lesion_id"]
        ))

        # Validate zero lesion leakage for all 5 folds
        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            tr_lesions = set(self.train_df.iloc[tr_idx]["lesion_id"])
            val_lesions = set(self.train_df.iloc[val_idx]["lesion_id"])
            overlap = tr_lesions.intersection(val_lesions)
            if overlap:
                raise RuntimeError(
                    f"Leakage detected in Fold {fold_idx}: {len(overlap)} lesions appear in both train and val!"
                )
            logger.info(
                f"Fold {fold_idx}: {len(tr_idx)} train ({len(tr_lesions)} lesions), "
                f"{len(val_idx)} val ({len(val_lesions)} lesions), overlap: 0"
            )

        return folds

    def build_fresh_model(
        self,
        model_name: str,
        seed: Optional[int] = None
    ) -> Tuple[keras.Model, bool]:
        """
        Constructs a fresh model instance with no learned weights carried over.
        Returns (model, pretrained_weights_loaded_bool).
        """
        seed_val = seed if seed is not None else self.random_seed
        name = model_name.lower().replace("-", "_")

        if name == "custom_cnn":
            model = build_custom_cnn(
                input_shape=tuple(self.dataset_cfg.image_size) + (3,),
                num_classes=self.dataset_cfg.num_classes,
                dropout_rate=self.model_cfg.dropout_rate,
                seed=seed_val,
                model_name=f"custom_cnn_{seed_val}"
            )
            return model, False

        elif name in ("efficientnet_b0", "efficientnet"):
            try:
                model = build_efficientnet_b0(
                    input_shape=tuple(self.dataset_cfg.image_size) + (3,),
                    num_classes=self.dataset_cfg.num_classes,
                    weights="imagenet",
                    freeze_backbone=True,
                    dropout_rate=self.model_cfg.dropout_rate,
                    seed=seed_val,
                    model_name=f"efficientnet_b0_{seed_val}"
                )
                return model, True
            except Exception as e:
                logger.warning(f"Could not load ImageNet weights for EfficientNetB0 ({e}). Falling back to uninitialized weights.")
                model = build_efficientnet_b0(
                    input_shape=tuple(self.dataset_cfg.image_size) + (3,),
                    num_classes=self.dataset_cfg.num_classes,
                    weights=None,
                    freeze_backbone=True,
                    dropout_rate=self.model_cfg.dropout_rate,
                    seed=seed_val,
                    model_name=f"efficientnet_b0_{seed_val}"
                )
                return model, False

        elif name in ("resnet50", "resnet"):
            try:
                model = build_resnet50(
                    input_shape=tuple(self.dataset_cfg.image_size) + (3,),
                    num_classes=self.dataset_cfg.num_classes,
                    weights="imagenet",
                    freeze_backbone=True,
                    dropout_rate=0.5,
                    seed=seed_val,
                    model_name=f"resnet50_{seed_val}"
                )
                return model, True
            except Exception as e:
                logger.warning(f"Could not load ImageNet weights for ResNet50 ({e}). Falling back to uninitialized weights.")
                model = build_resnet50(
                    input_shape=tuple(self.dataset_cfg.image_size) + (3,),
                    num_classes=self.dataset_cfg.num_classes,
                    weights=None,
                    freeze_backbone=True,
                    dropout_rate=0.5,
                    seed=seed_val,
                    model_name=f"resnet50_{seed_val}"
                )
                return model, False

        else:
            raise ValueError(f"Unknown model_name: {model_name}. Allowed: {SUPPORTED_MODELS}")

    def train_fold(
        self,
        model_name: str,
        fold_idx: int,
        epochs: int = 30,
        batch_size: int = 32,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Trains a single model on a single fold.

        Workflow:
            1. Extract fold training subset and fold validation subset.
            2. Compute class weights strictly on fold training labels.
            3. Build streaming tf.data datasets (augmentation applied exclusively to training subset).
            4. Construct a fresh model instance (weights strictly initialized).
            5. Compile with Adam and categorical cross-entropy.
            6. Train with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, and HistoryLogger.
            7. Reload best saved checkpoint.
            8. Predict probabilities on fold validation subset.
            9. Compute classification metrics (accuracy, macro P/R/F1, per-class metrics, confusion matrix).
        """
        tr_indices, val_indices = self.folds[fold_idx]
        tr_sub = self.train_df.iloc[tr_indices].copy()
        val_sub = self.train_df.iloc[val_indices].copy()

        logger.info(
            f"--- Training {model_name} on Fold {fold_idx} --- "
            f"(Train samples: {len(tr_sub)}, Val samples: {len(val_sub)})"
        )

        # 1. Compute fold-specific class weights strictly from training labels
        class_weights = compute_class_weights(tr_sub["dx"], classes=HAM10000_CLASSES)

        # 2. Encode labels
        tr_labels_oh = encode_labels(tr_sub["dx"], classes=HAM10000_CLASSES)
        val_labels_oh = encode_labels(val_sub["dx"], classes=HAM10000_CLASSES)

        # 3. Build streaming datasets
        tr_ds = create_dataset(
            image_paths=tr_sub["image_path"].tolist(),
            labels=tr_labels_oh,
            model_name=model_name,
            batch_size=batch_size,
            is_training=True,
            shuffle=True,
            seed=self.random_seed + fold_idx,
            target_size=tuple(self.dataset_cfg.image_size)
        )

        val_ds = create_dataset(
            image_paths=val_sub["image_path"].tolist(),
            labels=val_labels_oh,
            model_name=model_name,
            batch_size=batch_size,
            is_training=False,
            shuffle=False,
            seed=self.random_seed,
            target_size=tuple(self.dataset_cfg.image_size)
        )

        if steps_per_epoch is not None:
            tr_ds = tr_ds.take(steps_per_epoch)
        if validation_steps is not None:
            val_ds = val_ds.take(validation_steps)

        # 4. Fresh model construction
        keras.backend.clear_session()
        gc.collect()

        model, pretrained_loaded = self.build_fresh_model(model_name, seed=self.random_seed + fold_idx)
        optimizer = keras.optimizers.Adam(learning_rate=self.model_cfg.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # 5. Callbacks
        cbs, checkpoint_path, history_path = get_training_callbacks(
            model_name=model_name,
            fold_idx=fold_idx,
            checkpoint_dir=self.paths.checkpoints_dir,
            early_stopping_patience=self.model_cfg.early_stopping_patience,
            reduce_lr_patience=self.model_cfg.reduce_lr_patience,
            verbose=verbose
        )

        # 6. Model Training
        start_time = time.time()
        history = model.fit(
            tr_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=cbs,
            verbose=verbose
        )
        train_duration = round(time.time() - start_time, 2)

        # 7. Reload best checkpoint if saved
        if checkpoint_path.exists():
            try:
                model = keras.models.load_model(str(checkpoint_path))
                logger.info(f"Loaded best checkpoint from: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not reload checkpoint {checkpoint_path}: {e}. Evaluating in-memory weights.")

        # 8. Generate validation predictions (deterministic unaugmented dataset)
        val_eval_ds = create_dataset(
            image_paths=val_sub["image_path"].tolist(),
            labels=None,
            model_name=model_name,
            batch_size=batch_size,
            is_training=False,
            shuffle=False,
            seed=self.random_seed,
            target_size=tuple(self.dataset_cfg.image_size)
        )
        if validation_steps is not None:
            val_eval_ds = val_eval_ds.take(validation_steps)

        val_preds = model.predict(val_eval_ds, verbose=0)

        # Slice ground truth to match actual evaluated predictions count (e.g. during smoke testing)
        evaluated_count = len(val_preds)
        val_sub_eval = val_sub.iloc[:evaluated_count]
        y_true_indices = np.array([CLASS_TO_IDX[dx] for dx in val_sub_eval["dx"]])
        y_pred_indices = np.argmax(val_preds, axis=1)

        # 9. Compute validation metrics
        acc = float(accuracy_score(y_true_indices, y_pred_indices))
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true_indices, y_pred_indices, average="macro", zero_division=0
        )
        per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
            y_true_indices, y_pred_indices, average=None, labels=list(range(7)), zero_division=0
        )
        cm = confusion_matrix(y_true_indices, y_pred_indices, labels=list(range(7))).tolist()

        # Checkpoint file size on disk
        chk_size_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0

        fold_result: Dict[str, Any] = {
            "model_name": model_name,
            "fold_idx": fold_idx,
            "train_duration_seconds": train_duration,
            "epochs_run": len(history.history.get("loss", [])),
            "pretrained_weights_loaded": pretrained_loaded,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_size_bytes": chk_size_bytes,
            "history_path": str(history_path),
            "val_sample_count": evaluated_count,
            "class_weights": class_weights,
            "metrics": {
                "accuracy": round(acc, 4),
                "macro_precision": round(float(macro_p), 4),
                "macro_recall": round(float(macro_r), 4),
                "macro_f1": round(float(macro_f1), 4),
                "per_class_recall": {HAM10000_CLASSES[i]: round(float(per_class_r[i]), 4) for i in range(7)},
                "per_class_f1": {HAM10000_CLASSES[i]: round(float(per_class_f1[i]), 4) for i in range(7)},
                "confusion_matrix": cm
            },
            "val_indices": val_indices[:evaluated_count].tolist(),
            "val_predictions": val_preds
        }

        logger.info(
            f"Fold {fold_idx} [{model_name}] Finished in {train_duration}s — "
            f"Acc: {acc:.4f}, Macro-F1: {macro_f1:.4f}, Checkpoint size: {chk_size_bytes / (1024*1024):.2f} MB"
        )
        return fold_result

    def run_smoke_test(
        self,
        epochs: int = 1,
        steps_per_epoch: int = 2,
        validation_steps: int = 2,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Executes a short, memory-safe smoke test across all 3 models on Fold 0.
        Verifies:
            - Pipeline streaming and data augmentation execution.
            - Model compilation and backward pass on CPU.
            - Class-weighted categorical cross-entropy loss execution.
            - Checkpoint serialization and history logging.
            - Prediction generation and metric calculation.
            - Measures execution times, step durations, and disk usage for resource estimation.
        """
        logger.info("==================================================================")
        logger.info("STARTING STAGE 3 SMOKE TEST & RESOURCE ESTIMATION")
        logger.info(f"Parameters: epochs={epochs}, steps_per_epoch={steps_per_epoch}, batch_size={batch_size}")
        logger.info("==================================================================")

        smoke_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "mode": "smoke_test",
            "batch_size": batch_size,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
            "models": {}
        }

        total_smoke_start = time.time()

        for model_name in SUPPORTED_MODELS:
            logger.info(f"\n--- Smoke Testing Architecture: {model_name} ---")
            m_start = time.time()
            fold_res = self.train_fold(
                model_name=model_name,
                fold_idx=0,
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                verbose=1
            )
            m_duration = round(time.time() - m_start, 2)

            # Store summary excluding large prediction arrays
            val_preds = fold_res.pop("val_predictions")
            val_preds_valid, reason = validate_output_tensor(
                val_preds,
                expected_batch_size=len(val_preds),
                expected_classes=7
            )

            # Calculate estimated full epoch duration on CPU
            # Train partition has 5,722 images per fold -> ~179 batches at batch_size=32
            sec_per_step = m_duration / (steps_per_epoch + validation_steps)
            est_epoch_seconds_bs32 = sec_per_step * (5722 / 32)
            est_30_epochs_minutes = (est_epoch_seconds_bs32 * 30) / 60.0

            smoke_results["models"][model_name] = {
                "smoke_duration_seconds": m_duration,
                "seconds_per_step": round(sec_per_step, 3),
                "estimated_full_epoch_duration_seconds": round(est_epoch_seconds_bs32, 1),
                "estimated_30_epochs_per_fold_minutes": round(est_30_epochs_minutes, 1),
                "checkpoint_size_mb": round(fold_res["checkpoint_size_bytes"] / (1024 * 1024), 2),
                "prediction_tensor_valid": val_preds_valid,
                "prediction_tensor_status": reason,
                "metrics": fold_res["metrics"]
            }

        total_smoke_duration = round(time.time() - total_smoke_start, 2)
        smoke_results["total_smoke_duration_seconds"] = total_smoke_duration

        # Estimate total full 15-run 5-fold cross validation time
        total_est_minutes = sum(
            m_info["estimated_30_epochs_per_fold_minutes"] * 5
            for m_info in smoke_results["models"].values()
        )
        smoke_results["total_estimated_15_runs_hours"] = round(total_est_minutes / 60.0, 2)

        # Save smoke test audit to metrics directory
        smoke_audit_path = self.paths.metrics_dir / "stage_3_smoke_test_report.json"
        with open(smoke_audit_path, "w", encoding="utf-8") as f:
            json.dump(smoke_results, f, indent=2)

        logger.info("==================================================================")
        logger.info(f"SMOKE TEST COMPLETE in {total_smoke_duration}s.")
        logger.info(f"Resource report saved to: {smoke_audit_path}")
        return smoke_results

    def run_fold_0_pilot(
        self,
        epochs: int = 5,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Executes a controlled pilot on Fold 0 across all 3 models for up to `epochs` (default 5).
        - Custom CNN: full network trained with class weights.
        - EfficientNetB0: frozen backbone, classification head trained with class weights.
        - ResNet50: frozen backbone, classification head trained with class weights.
        - Evaluates best checkpoint on Fold 0 held-out validation samples (1,431 images).
        - Generates, verifies, and serializes the partial Fold 0 OOF predictions (1431, 21).
        - Measures actual epoch durations, loss, accuracy, macro-F1, per-class recall, and checkpoint sizes.
        - Computes revised runtime projections for 3, 5, 8, and 30 epochs across full 5 folds.
        """
        logger.info("==================================================================")
        logger.info(f"STARTING FOLD 0 PILOT BENCHMARK (Up to {epochs} Epochs per Model)")
        logger.info(f"Batch size: {batch_size}, Models: {SUPPORTED_MODELS}")
        logger.info("==================================================================")

        pilot_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "mode": "fold_0_pilot",
            "fold_idx": 0,
            "target_epochs": epochs,
            "batch_size": batch_size,
            "models": {}
        }

        pilot_start_time = time.time()
        fold_results_map: Dict[str, Dict[str, Any]] = {}

        for model_name in SUPPORTED_MODELS:
            logger.info(f"\n========================================================")
            logger.info(f"PILOT TRAINING: {model_name} on Fold 0 (Epochs: {epochs})")
            logger.info(f"========================================================")

            fold_res = self.train_fold(
                model_name=model_name,
                fold_idx=0,
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=None,       # Full training on all 5,722 images!
                validation_steps=None,      # Full validation on all 1,431 images!
                verbose=verbose
            )

            fold_results_map[model_name] = fold_res

            # Read epoch history from saved JSON
            history_file = Path(fold_res["history_path"])
            epoch_history = []
            if history_file.exists():
                try:
                    with open(history_file, "r", encoding="utf-8") as f:
                        epoch_history = json.load(f).get("history", [])
                except Exception as e:
                    logger.warning(f"Could not read history file {history_file}: {e}")

            epochs_run = fold_res["epochs_run"]
            duration_s = fold_res["train_duration_seconds"]
            avg_epoch_sec = round(duration_s / epochs_run, 2) if epochs_run > 0 else 0.0

            # Store summary in pilot report
            pilot_results["models"][model_name] = {
                "epochs_completed": epochs_run,
                "total_train_seconds": duration_s,
                "average_seconds_per_epoch": avg_epoch_sec,
                "checkpoint_size_mb": round(fold_res["checkpoint_size_bytes"] / (1024 * 1024), 2),
                "pretrained_weights_loaded": fold_res["pretrained_weights_loaded"],
                "metrics": fold_res["metrics"],
                "epoch_history": epoch_history
            }

        total_pilot_duration = round(time.time() - pilot_start_time, 2)
        pilot_results["total_pilot_duration_seconds"] = total_pilot_duration

        # --- ASSEMBLE AND VERIFY PARTIAL FOLD 0 OOF PREDICTIONS (1431, 21) ---
        val_indices = self.folds[0][1]
        n_val = len(val_indices)
        assert n_val == 1431, f"Expected 1431 Fold 0 validation samples, got {n_val}"

        val_sub = self.train_df.iloc[val_indices].copy().reset_index(drop=True)
        partial_oof = np.zeros((n_val, 21), dtype=np.float32)

        for model_name in SUPPORTED_MODELS:
            c_start, c_end = MODEL_COLUMN_SLICES[model_name]
            preds = fold_results_map[model_name]["val_predictions"]
            assert preds.shape == (n_val, 7), f"Expected shape ({n_val}, 7), got {preds.shape}"
            partial_oof[:, c_start:c_end] = preds

        # Verification Assertions
        assert not np.isnan(partial_oof).any(), "Partial OOF matrix contains NaNs!"
        assert not np.isinf(partial_oof).any(), "Partial OOF matrix contains Infs!"
        assert np.all(partial_oof >= 0.0) and np.all(partial_oof <= 1.0 + 1e-4), "Probabilities out of bounds [0, 1]!"

        for model_name, (c_start, c_end) in MODEL_COLUMN_SLICES.items():
            row_sums = partial_oof[:, c_start:c_end].sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-3), (
                f"Pilot OOF probabilities for '{model_name}' do not sum to 1.0!"
            )

        y_true_indices = np.array([CLASS_TO_IDX[dx] for dx in val_sub["dx"]], dtype=np.int32)
        val_image_ids = val_sub["image_id"].tolist()
        val_lesion_ids = val_sub["lesion_id"].tolist()

        # Save partial pilot OOF artifacts with explicit non-complete markers
        pilot_oof_path = self.paths.oof_dir / "pilot_fold_0_partial_oof.npy"
        pilot_labels_path = self.paths.oof_dir / "pilot_fold_0_labels.npy"
        pilot_ids_path = self.paths.oof_dir / "pilot_fold_0_image_ids.json"
        pilot_manifest_path = self.paths.oof_dir / "pilot_fold_0_manifest.json"

        np.save(pilot_oof_path, partial_oof)
        np.save(pilot_labels_path, y_true_indices)
        with open(pilot_ids_path, "w", encoding="utf-8") as f:
            json.dump(val_image_ids, f, indent=2)

        partial_manifest = {
            "created_at": datetime.now().isoformat(),
            "is_complete_oof": False,
            "warning": "PARTIAL FOLD 0 PILOT ONLY - NOT COMPLETE OOF MATRIX",
            "partial_fold_index": 0,
            "evaluated_samples": n_val,
            "total_training_samples": len(self.train_df),
            "coverage_percentage": round((n_val / len(self.train_df)) * 100, 2),
            "matrix_shape": list(partial_oof.shape),
            "classes": HAM10000_CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "model_column_slices": MODEL_COLUMN_SLICES,
            "artifacts": {
                "partial_oof_predictions": str(pilot_oof_path),
                "partial_oof_labels": str(pilot_labels_path),
                "partial_oof_image_ids": str(pilot_ids_path)
            }
        }
        with open(pilot_manifest_path, "w", encoding="utf-8") as f:
            json.dump(partial_manifest, f, indent=2)

        # --- REVISED RUNTIME PROJECTIONS BASED ON ACTUAL MEASURED PILOT DATA ---
        combined_1_epoch_sec = sum(
            m["average_seconds_per_epoch"] for m in pilot_results["models"].values()
        )
        combined_1_epoch_all_5_folds_hours = round((5 * combined_1_epoch_sec) / 3600.0, 3)

        pilot_results["runtime_calibration"] = {
            "measured_combined_1_epoch_seconds_per_fold": round(combined_1_epoch_sec, 2),
            "measured_combined_1_epoch_all_5_folds_hours": combined_1_epoch_all_5_folds_hours,
            "projections_all_15_runs": {
                "3_epochs_total_hours": round(combined_1_epoch_all_5_folds_hours * 3, 2),
                "5_epochs_total_hours": round(combined_1_epoch_all_5_folds_hours * 5, 2),
                "8_epochs_total_hours": round(combined_1_epoch_all_5_folds_hours * 8, 2),
                "30_epochs_total_hours": round(combined_1_epoch_all_5_folds_hours * 30, 2)
            }
        }

        # Save pilot report
        report_path = self.paths.metrics_dir / "stage_3_fold0_pilot_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(pilot_results, f, indent=2)

        logger.info("==================================================================")
        logger.info(f"FOLD 0 PILOT BENCHMARK COMPLETE in {total_pilot_duration}s.")
        logger.info(f"Partial Fold 0 OOF Matrix saved to: {pilot_oof_path}")
        logger.info(f"Pilot Report saved to: {report_path}")
        logger.info(f"Calibrated 1 Combined Epoch (5 Folds): {combined_1_epoch_all_5_folds_hours} hours")
        logger.info(f"Calibrated 5 Epochs (5 Folds, 15 runs): {pilot_results['runtime_calibration']['projections_all_15_runs']['5_epochs_total_hours']} hours")
        logger.info("==================================================================")

        return pilot_results

    def run_5_fold_cross_validation(
        self,
        epochs: int = 5,
        batch_size: int = 32,
        resume: bool = True,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Executes full 5-fold cross-validation across all 3 models (Custom CNN, EfficientNetB0, ResNet50).
        Total model-fold runs: 15 runs.

        Resumption and Pilot Preservation Safeguard:
            If `resume=True`, checks whether a checkpoint and history exist for the current selection metric.
            If both exist, reloads the model, generates validation predictions for that fold,
            computes metrics, and skips retraining. This completely preserves the completed
            Fold 0 pilot and allows safe resumption if training is ever interrupted.

        After all 5 folds for all 3 models are completed:
            Assembles the complete Out-Of-Fold (OOF) matrix (7153, 21), verifies all integrity assertions,
            serializes the complete OOF matrix and manifest ('is_complete_oof: true'),
            computes aggregate cross-fold performance statistics (mean and std),
            and persists the full cross-validation report to `artifacts/metrics/stage_3_full_cv_report.json`.
        """
        logger.info("==================================================================")
        logger.info(f"STARTING FULL 5-FOLD CROSS-VALIDATION (5 Folds x 3 Models = 15 Runs)")
        logger.info(f"Target Epochs per Fold: {epochs}, Batch Size: {batch_size}, Resume: {resume}")
        logger.info(f"Models: {SUPPORTED_MODELS}")
        logger.info("==================================================================")

        total_cv_start = time.time()
        fold_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in SUPPORTED_MODELS}

        full_cv_report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "mode": "full_5_fold_cross_validation",
            "epochs_target": epochs,
            "batch_size": batch_size,
            "total_folds": 5,
            "models": SUPPORTED_MODELS,
            "runs_completed": 0,
            "runs_reused_from_checkpoint": 0,
            "runs_trained_new": 0,
            "fold_details": {m: [] for m in SUPPORTED_MODELS},
            "aggregate_metrics": {}
        }

        # Iterate over all 5 folds
        for fold_idx in range(5):
            logger.info(f"\n##################################################################")
            logger.info(f"--- PROCESSING FOLD {fold_idx} / 5 ---")
            logger.info(f"##################################################################")

            val_indices = self.folds[fold_idx][1]
            n_val = len(val_indices)
            val_sub = self.train_df.iloc[val_indices].copy().reset_index(drop=True)

            for model_name in SUPPORTED_MODELS:
                # Resume only the new val_accuracy-selected artifacts. Legacy
                # *_best.keras / *_history.json files were selected by val_loss.
                chk_path = self.paths.checkpoints_dir / f"{model_name}_fold_{fold_idx}_best_val_accuracy.keras"
                hist_path = self.paths.checkpoints_dir / f"{model_name}_fold_{fold_idx}_history_val_accuracy.json"

                can_resume = resume and chk_path.exists() and hist_path.exists()

                if can_resume:
                    logger.info(
                        f"Reusing existing val_accuracy checkpoint for [{model_name}] on Fold {fold_idx}: {chk_path.name}"
                    )
                    epochs_run = epochs
                    train_duration = 0.0
                    try:
                        with open(hist_path, "r", encoding="utf-8") as f:
                            hist_data = json.load(f)
                        hist_entries = hist_data.get("history", [])
                        epochs_run = len(hist_entries)
                        train_duration = hist_entries[-1].get("elapsed_seconds", 0.0) if hist_entries else 0.0
                    except Exception as e:
                        logger.warning(f"Could not read history for {chk_path.name}: {e}")

                    # Reload model and generate predictions on validation fold.
                    # The versioned filename ensures legacy val_loss-selected
                    # checkpoints are not reused by this corrected run.
                    keras.backend.clear_session()
                    gc.collect()
                    model = keras.models.load_model(str(chk_path))

                    val_eval_ds = create_dataset(
                        image_paths=val_sub["image_path"].tolist(),
                        labels=None,
                        model_name=model_name,
                        batch_size=batch_size,
                        is_training=False,
                        shuffle=False,
                        seed=self.random_seed,
                        target_size=tuple(self.dataset_cfg.image_size)
                    )
                    val_preds = model.predict(val_eval_ds, verbose=0)
                    keras.backend.clear_session()
                    gc.collect()

                    y_true_indices = np.array([CLASS_TO_IDX[dx] for dx in val_sub["dx"]])
                    y_pred_indices = np.argmax(val_preds, axis=1)

                    acc = float(accuracy_score(y_true_indices, y_pred_indices))
                    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                        y_true_indices, y_pred_indices, average="macro", zero_division=0
                    )
                    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
                        y_true_indices, y_pred_indices, average=None, labels=list(range(7)), zero_division=0
                    )
                    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=list(range(7))).tolist()

                    chk_size_bytes = chk_path.stat().st_size if chk_path.exists() else 0

                    fold_res = {
                        "model_name": model_name,
                        "fold_idx": fold_idx,
                        "train_duration_seconds": train_duration,
                        "epochs_run": epochs_run,
                        "pretrained_weights_loaded": None,  # Original load status is not stored in legacy history.
                        "checkpoint_path": str(chk_path),
                        "checkpoint_size_bytes": chk_size_bytes,
                        "history_path": str(hist_path),
                        "val_sample_count": len(val_preds),
                        "metrics": {
                            "accuracy": round(acc, 4),
                            "macro_precision": round(float(macro_p), 4),
                            "macro_recall": round(float(macro_r), 4),
                            "macro_f1": round(float(macro_f1), 4),
                            "per_class_recall": {HAM10000_CLASSES[i]: round(float(per_class_r[i]), 4) for i in range(7)},
                            "per_class_f1": {HAM10000_CLASSES[i]: round(float(per_class_f1[i]), 4) for i in range(7)},
                            "confusion_matrix": cm
                        },
                        "val_indices": val_indices.tolist(),
                        "val_predictions": val_preds
                    }
                    full_cv_report["runs_reused_from_checkpoint"] += 1
                else:
                    logger.info(f"Training [{model_name}] on Fold {fold_idx} for up to {epochs} epochs...")
                    fold_res = self.train_fold(
                        model_name=model_name,
                        fold_idx=fold_idx,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=verbose
                    )
                    full_cv_report["runs_trained_new"] += 1

                fold_results[model_name].append(fold_res)
                full_cv_report["runs_completed"] += 1

                summary_entry = {
                    "model_name": model_name,
                    "fold_idx": fold_idx,
                    "train_duration_seconds": fold_res["train_duration_seconds"],
                    "epochs_run": fold_res["epochs_run"],
                    "checkpoint_path": fold_res["checkpoint_path"],
                    "checkpoint_size_mb": round(fold_res["checkpoint_size_bytes"] / (1024 * 1024), 2),
                    "metrics": fold_res["metrics"]
                }
                full_cv_report["fold_details"][model_name].append(summary_entry)

        # --- ASSEMBLE FULL OOF MATRIX (7153, 21) ---
        logger.info("\n==================================================================")
        logger.info("ASSEMBLING MASTER OUT-OF-FOLD (OOF) MATRIX (7153, 21)")
        logger.info("==================================================================")
        oof_manifest_path = self.assemble_and_save_oof(fold_results, output_dir=self.paths.oof_dir)

        # --- COMPUTE AGGREGATE CROSS-FOLD METRICS (MEAN +/- STD) ---
        for model_name in SUPPORTED_MODELS:
            m_folds = full_cv_report["fold_details"][model_name]
            accs = [f["metrics"]["accuracy"] for f in m_folds]
            macro_ps = [f["metrics"]["macro_precision"] for f in m_folds]
            macro_rs = [f["metrics"]["macro_recall"] for f in m_folds]
            macro_f1s = [f["metrics"]["macro_f1"] for f in m_folds]

            full_cv_report["aggregate_metrics"][model_name] = {
                "accuracy_mean": round(float(np.mean(accs)), 4),
                "accuracy_std": round(float(np.std(accs)), 4),
                "macro_precision_mean": round(float(np.mean(macro_ps)), 4),
                "macro_precision_std": round(float(np.std(macro_ps)), 4),
                "macro_recall_mean": round(float(np.mean(macro_rs)), 4),
                "macro_recall_std": round(float(np.std(macro_rs)), 4),
                "macro_f1_mean": round(float(np.mean(macro_f1s)), 4),
                "macro_f1_std": round(float(np.std(macro_f1s)), 4),
                "per_class_recall_mean": {
                    c: round(float(np.mean([f["metrics"]["per_class_recall"][c] for f in m_folds])), 4)
                    for c in HAM10000_CLASSES
                },
                "per_class_f1_mean": {
                    c: round(float(np.mean([f["metrics"]["per_class_f1"][c] for f in m_folds])), 4)
                    for c in HAM10000_CLASSES
                }
            }

        total_cv_duration = round(time.time() - total_cv_start, 2)
        full_cv_report["total_cv_duration_seconds"] = total_cv_duration
        full_cv_report["total_cv_duration_hours"] = round(total_cv_duration / 3600.0, 2)

        full_report_path = self.paths.metrics_dir / "stage_3_full_cv_report.json"
        with open(full_report_path, "w", encoding="utf-8") as f:
            json.dump(full_cv_report, f, indent=2)

        logger.info(f"Full 5-Fold Cross Validation Complete in {total_cv_duration}s ({full_cv_report['total_cv_duration_hours']} hours).")
        logger.info(f"Report saved to: {full_report_path}")
        logger.info(f"OOF Manifest saved to: {oof_manifest_path}")

        return full_cv_report

    def assemble_and_save_oof(
        self,
        fold_results: Dict[str, List[Dict[str, Any]]],
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Assembles individual fold predictions into the master Out-Of-Fold (OOF) matrix (7153, 21).

        Matrix Structure:
            - Rows (0..7152): Strictly aligned with self.train_df['image_id']
            - Cols (0..6): Custom CNN class probabilities
            - Cols (7..13): EfficientNetB0 class probabilities
            - Cols (14..20): ResNet50 class probabilities

        Assertions Verified:
            1. Shape is exactly (7153, 21).
            2. Every value is finite and in [0.0, 1.0].
            3. Every 7-column block sums to 1.0 +/- 1e-3.
            4. Every sample has exactly one OOF prediction per model.
            5. Aligned labels (7153,) match ground truth.
        """
        out_dir = output_dir or self.paths.oof_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        n_samples = len(self.train_df)
        oof_matrix = np.full((n_samples, 21), fill_value=np.nan, dtype=np.float32)

        for model_name, folds_list in fold_results.items():
            if model_name not in MODEL_COLUMN_SLICES:
                raise ValueError(f"Unrecognized model '{model_name}'. Allowed: {SUPPORTED_MODELS}")
            c_start, c_end = MODEL_COLUMN_SLICES[model_name]

            # Populate rows from each fold's out-of-fold validation predictions
            covered_indices: List[int] = []
            for fold_data in folds_list:
                v_indices = fold_data["val_indices"]
                v_preds = fold_data["val_predictions"]
                oof_matrix[v_indices, c_start:c_end] = v_preds
                covered_indices.extend(v_indices)

            # Assert complete coverage
            if len(set(covered_indices)) != n_samples:
                missing = n_samples - len(set(covered_indices))
                raise ValueError(
                    f"OOF matrix for '{model_name}' has {missing} unpredicted samples out of {n_samples}!"
                )

        # Integrity Assertions
        assert not np.isnan(oof_matrix).any(), "OOF matrix contains unpredicted NaN values!"
        assert not np.isinf(oof_matrix).any(), "OOF matrix contains infinite values!"
        assert np.all(oof_matrix >= 0.0), "OOF matrix contains negative probabilities!"
        assert np.all(oof_matrix <= 1.0 + 1e-4), "OOF matrix contains probabilities > 1.0!"

        # Assert sum to 1.0 for each model block
        for model_name, (c_start, c_end) in MODEL_COLUMN_SLICES.items():
            row_sums = oof_matrix[:, c_start:c_end].sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-3), (
                f"OOF probabilities for '{model_name}' do not sum to 1.0! Min: {row_sums.min()}, Max: {row_sums.max()}"
            )

        # Build aligned ground truth labels
        labels_aligned = np.array([CLASS_TO_IDX[dx] for dx in self.train_df["dx"]], dtype=np.int32)
        image_ids_aligned = self.train_df["image_id"].tolist()
        lesion_ids_aligned = self.train_df["lesion_id"].tolist()

        # Build fold assignments array
        fold_assignments = np.zeros(n_samples, dtype=np.int32)
        for fold_idx, (_, val_idx) in enumerate(self.folds):
            fold_assignments[val_idx] = fold_idx

        # Serialize artifacts
        pred_path = out_dir / "oof_predictions.npy"
        label_path = out_dir / "oof_labels.npy"
        ids_path = out_dir / "oof_image_ids.json"
        manifest_path = out_dir / "oof_manifest.json"

        np.save(pred_path, oof_matrix)
        np.save(label_path, labels_aligned)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(image_ids_aligned, f, indent=2)

        manifest = {
            "created_at": datetime.now().isoformat(),
            "is_complete_oof": True,
            "warning": None,
            "evaluated_samples": n_samples,
            "total_training_samples": n_samples,
            "coverage_percentage": 100.0,
            "matrix_shape": list(oof_matrix.shape),
            "sample_count": n_samples,
            "classes": HAM10000_CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "model_column_slices": MODEL_COLUMN_SLICES,
            "random_seed": self.random_seed,
            "leakage_safety": {
                "training_samples_only": True,
                "external_validation_excluded": True,
                "holdout_test_excluded": True,
                "group_isolation_verified": True
            },
            "artifacts": {
                "oof_predictions": str(pred_path),
                "oof_labels": str(label_path),
                "oof_image_ids": str(ids_path)
            }
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved verified OOF matrix {oof_matrix.shape} to: {pred_path}")
        logger.info(f"Saved OOF manifest to: {manifest_path}")
        return manifest_path


def main():
    parser = argparse.ArgumentParser(description="DermaStack-XAI Training & OOF Engine")
    parser.add_argument("--smoke-test", action="store_true", help="Run short smoke test to verify pipeline & estimate compute")
    parser.add_argument("--pilot", action="store_true", help="Run controlled Fold 0 pilot (up to 5 epochs across all 3 models)")
    parser.add_argument("--full-cv", action="store_true", help="Run full 5-fold cross-validation")
    parser.add_argument("--no-resume", action="store_true", help="Force retraining all folds without reusing existing checkpoints")
    parser.add_argument("--model", type=str, default="all", choices=["all"] + SUPPORTED_MODELS, help="Model to train")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Verify configuration, splits, and paths without training")

    args = parser.parse_args()

    engine = FiveFoldCrossValidationEngine()

    if args.dry_run:
        print("\n[DRY RUN VERIFICATION]")
        print(f"Total training images: {len(engine.train_df)}")
        print(f"Number of folds: {len(engine.folds)}")
        for i, (tr, val) in enumerate(engine.folds):
            print(f"  Fold {i}: train={len(tr)}, val={len(val)}")
        print("Dry run verification passed.\n")
        return

    if args.pilot:
        target_epochs = args.epochs if args.epochs > 0 else 5
        pilot_summary = engine.run_fold_0_pilot(
            epochs=target_epochs,
            batch_size=args.batch_size,
            verbose=1
        )
        print("\n=== STAGE 3 FOLD 0 PILOT SUMMARY ===")
        print(json.dumps(pilot_summary, indent=2))
        return

    if args.smoke_test:
        smoke_summary = engine.run_smoke_test(
            epochs=1,
            steps_per_epoch=2,
            validation_steps=2,
            batch_size=8
        )
        print("\n=== STAGE 3 SMOKE TEST AUDIT SUMMARY ===")
        print(json.dumps(smoke_summary, indent=2))
        return

    if args.full_cv:
        target_epochs = args.epochs if args.epochs > 0 else 5
        cv_summary = engine.run_5_fold_cross_validation(
            epochs=target_epochs,
            batch_size=args.batch_size,
            resume=not args.no_resume,
            verbose=1
        )
        print("\n=== STAGE 3 FULL 5-FOLD CV SUMMARY ===")
        print(json.dumps(cv_summary, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
