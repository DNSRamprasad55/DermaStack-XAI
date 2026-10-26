"""
DermaStack-XAI Stage 6 Holdout Test Evaluation Pipeline.

Executes final benchmark evaluation on the untouched HAM10000 holdout test set (1,431 images):
    1. Loads frozen evaluation protocol.
    2. Streams test images through all 15 base model fold checkpoints (5 folds x 3 architectures).
    3. Computes ensemble mean probability predictions for Custom CNN, EfficientNetB0, and ResNet50.
    4. Constructs the (1431, 21) stacking feature matrix and predicts with both Standard and Balanced meta-models.
    5. Computes Accuracy, Macro-P/R/F1, Weighted-F1, per-class metrics, ROC-AUC, Log Loss, and Brier score.
    6. Generates publication-quality confusion matrices, ROC curves, comparison charts, and calibration plots.
    7. Persists evaluation summary and plot source data.
"""

import argparse
import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import keras
import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, HAM10000_CLASSES, CLASS_NAMES_MAPPING
from src.data.transforms import create_dataset
from src.evaluation.metrics import compute_evaluation_benchmark, validate_evaluation_inputs
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_multiclass_roc_curves,
    plot_model_comparison,
    plot_calibration_curves
)
from src.models.stacking_meta import StackingMetaClassifier, validate_stacking_features
from src.utils.logger import get_logger

logger = get_logger("DermaStack-XAI.Evaluation.Holdout")

SUPPORTED_BASE_MODELS = ["custom_cnn", "efficientnet_b0", "resnet50"]


def verify_holdout_test_isolation(splits_path: Path, metadata_path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Verifies that the holdout test partition is strictly isolated and aligns with metadata.

    Returns:
        Tuple of (test_image_ids, test_labels).
    """
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    test_ids = splits_data.get("splits", {}).get("test", [])
    train_ids = splits_data.get("splits", {}).get("train", [])
    val_ids = splits_data.get("splits", {}).get("val", [])

    assert len(test_ids) == 1431, f"Expected 1,431 test images, got {len(test_ids)}"
    assert len(set(test_ids).intersection(set(train_ids))) == 0, "CRITICAL: Test leaks into train!"
    assert len(set(test_ids).intersection(set(val_ids))) == 0, "CRITICAL: Test leaks into val!"

    meta_df = pd.read_csv(metadata_path).set_index("image_id")
    class_to_idx = {c: i for i, c in enumerate(HAM10000_CLASSES)}

    # Verify lesion-level isolation
    train_lesions = set(meta_df.loc[train_ids, "lesion_id"])
    val_lesions = set(meta_df.loc[val_ids, "lesion_id"])
    test_lesions = set(meta_df.loc[test_ids, "lesion_id"])

    assert len(test_lesions.intersection(train_lesions)) == 0, "CRITICAL: Test lesions overlap with train!"
    assert len(test_lesions.intersection(val_lesions)) == 0, "CRITICAL: Test lesions overlap with val!"

    test_labels = np.array([class_to_idx[meta_df.loc[img_id, "dx"]] for img_id in test_ids], dtype=np.int32)
    logger.info(f"Verified test set isolation: {len(test_ids)} images, {len(test_lesions)} unique lesions, 0 overlap.")
    return test_ids, test_labels


def extract_holdout_test_features(
    test_ids: List[str],
    metadata_path: Path,
    checkpoint_dir: Path,
    batch_size: int = 32,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False
) -> Dict[str, Any]:
    """
    Generates 5-fold ensemble predictions for all 3 base models on the 1,431 test images.

    Returns:
        Dict with 'X_test_stacked', 'y_test', 'test_image_ids', and base model probabilities.
    """
    cache_file = cache_path or (DEFAULT_CONFIG.paths.oof_dir / "holdout_test_ensemble_features.npz")

    if cache_file.exists() and not force_recompute:
        logger.info(f"Loading cached holdout test features from: {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        return {
            "X_test_stacked": cached["X_test_stacked"],
            "y_test": cached["y_test"],
            "test_image_ids": cached["test_image_ids"].tolist(),
            "base_predictions": {
                m: cached[f"preds_{m}"] for m in SUPPORTED_BASE_MODELS
            }
        }

    meta_df = pd.read_csv(metadata_path).set_index("image_id")
    test_image_paths = [str(meta_df.loc[img_id, "image_path"]) for img_id in test_ids]
    class_to_idx = {c: i for i, c in enumerate(HAM10000_CLASSES)}
    test_labels = np.array([class_to_idx[meta_df.loc[img_id, "dx"]] for img_id in test_ids], dtype=np.int32)

    base_predictions: Dict[str, np.ndarray] = {}

    for model_name in SUPPORTED_BASE_MODELS:
        logger.info(f"\n--- Generating Holdout Test Predictions: {model_name} (5 Folds) ---")
        ds = create_dataset(
            image_paths=test_image_paths,
            labels=None,
            model_name=model_name,
            batch_size=batch_size,
            is_training=False,
            shuffle=False
        )

        fold_preds_list: List[np.ndarray] = []
        for fold_idx in range(5):
            ckpt_path = checkpoint_dir / f"{model_name}_fold_{fold_idx}_best.keras"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing required fold checkpoint: {ckpt_path}")

            logger.info(f"Loading {ckpt_path.name}...")
            model = keras.models.load_model(str(ckpt_path), compile=False)

            t0 = time.time()
            f_preds = model.predict(ds, verbose=0)
            elapsed = round(time.time() - t0, 2)
            logger.info(f"  Fold {fold_idx} predicted 1,431 test samples in {elapsed}s.")

            assert f_preds.shape == (1431, 7)
            assert not np.isnan(f_preds).any()
            assert not np.isinf(f_preds).any()

            f_preds = np.clip(f_preds, 0.0, 1.0)
            f_preds = f_preds / np.maximum(f_preds.sum(axis=1, keepdims=True), 1e-12)
            fold_preds_list.append(f_preds)

            del model
            keras.backend.clear_session()

        mean_preds = np.mean(fold_preds_list, axis=0)
        mean_preds = mean_preds / np.maximum(mean_preds.sum(axis=1, keepdims=True), 1e-12)
        base_predictions[model_name] = mean_preds.astype(np.float32)

    X_test_stacked = np.concatenate([
        base_predictions["custom_cnn"],
        base_predictions["efficientnet_b0"],
        base_predictions["resnet50"]
    ], axis=1).astype(np.float32)

    X_test_stacked = validate_stacking_features(X_test_stacked, expected_rows=1431)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        X_test_stacked=X_test_stacked,
        y_test=test_labels,
        test_image_ids=np.array(test_ids),
        preds_custom_cnn=base_predictions["custom_cnn"],
        preds_efficientnet_b0=base_predictions["efficientnet_b0"],
        preds_resnet50=base_predictions["resnet50"]
    )
    logger.info(f"Cached holdout test features to: {cache_file}")

    return {
        "X_test_stacked": X_test_stacked,
        "y_test": test_labels,
        "test_image_ids": test_ids,
        "base_predictions": base_predictions
    }


def execute_stage_6_evaluation(force_recompute: bool = False) -> Dict[str, Any]:
    """
    Executes the complete Stage 6 evaluation pipeline on the holdout test set.
    """
    logger.info("==================================================================")
    logger.info("DERMASTACK-XAI — STAGE 6: FINAL BENCHMARK & EVALUATION")
    logger.info("==================================================================")

    splits_path = DEFAULT_CONFIG.paths.splits_dir / "train_val_test_splits.json"
    meta_path = DEFAULT_CONFIG.paths.processed_data_dir / "validated_metadata.csv"
    chk_dir = DEFAULT_CONFIG.paths.checkpoints_dir
    meta_dir = DEFAULT_CONFIG.paths.meta_models_dir
    outputs_dir = DEFAULT_CONFIG.paths.root / "outputs" / "evaluation"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify Holdout Test Isolation
    test_ids, test_labels = verify_holdout_test_isolation(splits_path, meta_path)

    # 2. Extract / Load Holdout Test Features
    test_data = extract_holdout_test_features(
        test_ids=test_ids,
        metadata_path=meta_path,
        checkpoint_dir=chk_dir,
        batch_size=DEFAULT_CONFIG.dataset.batch_size,
        force_recompute=force_recompute
    )
    X_test = test_data["X_test_stacked"]
    y_test = test_data["y_test"]
    base_preds = test_data["base_predictions"]

    # 3. Load Stacking Meta-Classifiers
    meta_std_path = meta_dir / "stacking_meta_model.joblib"
    meta_bal_path = meta_dir / "stacking_meta_model_balanced.joblib"

    if not meta_std_path.exists() or not meta_bal_path.exists():
        raise FileNotFoundError("Missing required meta-model artifacts.")

    meta_std = StackingMetaClassifier.load(meta_std_path)
    meta_bal = StackingMetaClassifier.load(meta_bal_path)

    probs_std = meta_std.predict_proba(X_test)
    probs_bal = meta_bal.predict_proba(X_test)

    # 4. Compile All 5 Evaluated Models
    models_to_evaluate: Dict[str, np.ndarray] = {
        "Custom CNN": base_preds["custom_cnn"],
        "EfficientNetB0": base_preds["efficientnet_b0"],
        "ResNet50": base_preds["resnet50"],
        "Standard Stacking": probs_std,
        "Balanced Stacking": probs_bal
    }

    # 5. Compute Benchmark Metrics
    test_benchmark_results: Dict[str, Any] = {}
    for name, p_mat in models_to_evaluate.items():
        bench = compute_evaluation_benchmark(
            y_true=y_test,
            y_prob=p_mat,
            classes=HAM10000_CLASSES,
            model_name=name
        )
        test_benchmark_results[name] = bench
        logger.info(
            f"  {name:<20} -> Accuracy: {bench['accuracy']:.4f}, "
            f"Macro-F1: {bench['macro_f1']:.4f}, Macro-Recall: {bench['macro_recall']:.4f}, "
            f"ROC-AUC: {bench['roc_auc_ovr_macro']}, Brier: {bench['multiclass_brier_score']}"
        )

    # 6. Generate Publication Figures
    logger.info("\nGenerating publication-quality figures in outputs/evaluation/...")

    # A. Confusion Matrices (Raw and Normalized) for all 5 models
    for name, p_mat in models_to_evaluate.items():
        clean_name = name.lower().replace(" ", "_")
        cm = test_benchmark_results[name]["confusion_matrix"]
        plot_confusion_matrix(
            cm=cm,
            classes=HAM10000_CLASSES,
            model_name=name,
            output_path=outputs_dir / f"{clean_name}_confusion_matrix_raw.png",
            normalize=False,
            title_suffix="Holdout Test Set"
        )
        plot_confusion_matrix(
            cm=cm,
            classes=HAM10000_CLASSES,
            model_name=name,
            output_path=outputs_dir / f"{clean_name}_confusion_matrix_normalized.png",
            normalize=True,
            title_suffix="Holdout Test Set"
        )

    # B. Multiclass ROC Curves for all 5 models
    for name, p_mat in models_to_evaluate.items():
        clean_name = name.lower().replace(" ", "_")
        plot_multiclass_roc_curves(
            y_true=y_test,
            y_prob=p_mat,
            classes=HAM10000_CLASSES,
            model_name=name,
            output_path=outputs_dir / f"{clean_name}_roc_curves.png",
            title_suffix="Holdout Test Set"
        )

    # C. Cross-Model Benchmark Comparison Chart
    plot_model_comparison(
        models_metrics=test_benchmark_results,
        output_path=outputs_dir / "model_benchmark_comparison_test.png",
        title_suffix="Holdout Test Set (N = 1,431)"
    )

    # D. Calibration Curves
    plot_calibration_curves(
        models_probs=models_to_evaluate,
        y_true=y_test,
        classes=HAM10000_CLASSES,
        output_path=outputs_dir / "model_calibration_curves_test.png",
        title_suffix="Holdout Test Set (N = 1,431)"
    )

    # 7. Load External Validation Metrics for Clear Dual-Partition Reporting
    val_metrics_path = DEFAULT_CONFIG.paths.metrics_dir / "stage_4_stacking_evaluation_report.json"
    val_comparison = {}
    if val_metrics_path.exists():
        with open(val_metrics_path, "r", encoding="utf-8") as f:
            val_comparison = json.load(f).get("external_validation_comparison", {})

    # 8. Save Full Evaluation Summary
    eval_summary_path = DEFAULT_CONFIG.paths.metrics_dir / "evaluation_summary.json"
    final_summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "evaluation_protocol": "artifacts/metrics/frozen_evaluation_protocol.json",
        "sample_counts": {
            "training_partition": 7153,
            "external_validation_partition": 1431,
            "holdout_test_partition": 1431,
            "holdout_test_evaluated_first_time": True
        },
        "canonical_classes": list(HAM10000_CLASSES),
        "holdout_test_benchmark": test_benchmark_results,
        "external_validation_benchmark": val_comparison,
        "figures_directory": str(outputs_dir.resolve())
    }

    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)
    logger.info(f"Saved complete evaluation summary to: {eval_summary_path}")

    return final_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermaStack-XAI Stage 6 Holdout Test Evaluation")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputing test predictions.")
    args = parser.parse_args()

    summary = execute_stage_6_evaluation(force_recompute=args.force_recompute)
    print("\nStage 6 Holdout Test Evaluation Successfully Completed.")
