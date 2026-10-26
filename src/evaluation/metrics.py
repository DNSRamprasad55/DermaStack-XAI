"""
DermaStack-XAI Benchmark Evaluation Metrics Module.

Computes comprehensive, reproducible classification and calibration metrics
for multi-class dermoscopic skin lesion classification:
    - Overall Accuracy
    - Macro Precision, Recall, and F1-score
    - Weighted Precision, Recall, and F1-score
    - Per-Class Precision, Recall, F1-score, and Support
    - Confusion Matrix (7x7)
    - Multiclass One-vs-Rest ROC-AUC (Macro-averaged)
    - Multiclass Log Loss (Cross-Entropy)
    - Multiclass Brier Score
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score
)

from src.config import DEFAULT_CONFIG, HAM10000_CLASSES, CLASS_NAMES_MAPPING
from src.utils.logger import get_logger

logger = get_logger("DermaStack-XAI.Evaluation.Metrics")


def validate_evaluation_inputs(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[List[float]]],
    num_classes: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates evaluation target labels and probability distributions.

    Assertions:
        1. y_true is 1D array of integers in [0, num_classes - 1].
        2. y_prob is 2D float array of shape (N, num_classes).
        3. All probabilities are finite (no NaN, no Inf) and in [0.0, 1.0 + epsilon].
        4. Row sums equal 1.0 +/- 1e-2.

    Returns:
        Tuple of (y_true_clean: np.ndarray, y_prob_clean: np.ndarray).
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_prob, dtype=np.float32)

    if y_t.ndim != 1:
        raise ValueError(f"Expected 1D target label array, got shape {y_t.shape}")

    if y_p.ndim != 2:
        raise ValueError(f"Expected 2D probability matrix, got shape {y_p.shape}")

    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError(f"Sample count mismatch: {y_t.shape[0]} labels vs {y_p.shape[0]} probability rows.")

    if y_p.shape[1] != num_classes:
        raise ValueError(f"Expected {num_classes} probability columns, got {y_p.shape[1]}")

    if np.isnan(y_p).any() or np.isinf(y_p).any():
        raise ValueError("Predicted probabilities contain NaN or Infinite values.")

    if (y_p < -1e-4).any() or (y_p > 1.0 + 1e-3).any():
        raise ValueError(f"Probabilities out of bounds [0, 1]: min={y_p.min()}, max={y_p.max()}")

    # Check for invalid class indices in ground truth
    if (y_t < 0).any() or (y_t >= num_classes).any():
        invalid = y_t[(y_t < 0) | (y_t >= num_classes)]
        raise ValueError(f"Ground truth contains labels outside [0, {num_classes - 1}]: {invalid[:5]}")

    # Clip and renormalize to ensure valid probability simplex
    y_p_clean = np.clip(y_p, 0.0, 1.0)
    row_sums = y_p_clean.sum(axis=1, keepdims=True)
    if np.any(np.abs(row_sums - 1.0) > 1e-2):
        bad_idx = int(np.argmax(np.abs(row_sums - 1.0)))
        raise ValueError(f"Probabilities do not sum to 1.0 at sample {bad_idx}: sum={row_sums[bad_idx, 0]:.4f}")

    y_p_clean = (y_p_clean / np.maximum(row_sums, 1e-12)).astype(np.float32)
    return y_t, y_p_clean


def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 7) -> float:
    """
    Computes the Multiclass Brier Score:
        Brier = (1 / N) * sum_{i=1}^N sum_{c=0}^{C-1} (p_{i, c} - y_{i, c})^2

    where y_{i, c} is the one-hot indicator for sample i and class c.
    Lower Brier score indicates superior probability calibration (range: [0.0, 2.0]).
    """
    N = len(y_true)
    if N == 0:
        return 0.0
    y_one_hot = np.eye(num_classes, dtype=np.float32)[y_true]
    brier_score = float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))
    return round(brier_score, 4)


def compute_evaluation_benchmark(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[List[float]]],
    classes: Optional[List[str]] = None,
    model_name: str = "model"
) -> Dict[str, Any]:
    """
    Computes all classification, ranking, and calibration metrics for a model.

    Args:
        y_true: Ground truth integer class indices (N,).
        y_prob: Predicted probability distributions (N, 7).
        classes: Canonical ordered class names. Defaults to HAM10000_CLASSES.
        model_name: Model identifier string.

    Returns:
        Structured dictionary of evaluation metrics.
    """
    class_list = classes or list(HAM10000_CLASSES)
    num_classes = len(class_list)

    y_t, y_p = validate_evaluation_inputs(y_true, y_prob, num_classes=num_classes)
    y_pred = np.argmax(y_p, axis=1)

    # 1. Overall Accuracy
    acc = float(accuracy_score(y_t, y_pred))

    # 2. Precision, Recall, F1 (Macro and Weighted)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_t, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_t, y_pred, average="weighted", zero_division=0
    )

    # 3. Per-class metrics
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        y_t, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    )

    # 4. Confusion Matrix (7x7)
    cm = confusion_matrix(y_t, y_pred, labels=list(range(num_classes))).tolist()

    # 5. Multiclass One-vs-Rest ROC-AUC (Macro-averaged)
    try:
        roc_auc = float(roc_auc_score(
            y_t,
            y_p,
            multi_class="ovr",
            average="macro",
            labels=list(range(num_classes))
        ))
        roc_auc = round(roc_auc, 4)
    except Exception as e:
        logger.warning(f"Failed to compute ROC-AUC for {model_name}: {e}")
        roc_auc = None

    # 6. Multiclass Log Loss (Cross-Entropy)
    try:
        # Clip to [1e-15, 1 - 1e-15] to prevent log(0)
        p_clipped = np.clip(y_p, 1e-15, 1.0 - 1e-15)
        p_clipped = p_clipped / p_clipped.sum(axis=1, keepdims=True)
        ll = float(log_loss(y_t, p_clipped, labels=list(range(num_classes))))
        ll = round(ll, 4)
    except Exception as e:
        logger.warning(f"Failed to compute Log Loss for {model_name}: {e}")
        ll = None

    # 7. Multiclass Brier Score
    brier = compute_multiclass_brier_score(y_t, y_p, num_classes=num_classes)

    benchmark_results: Dict[str, Any] = {
        "model_name": model_name,
        "sample_count": int(len(y_t)),
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "roc_auc_ovr_macro": roc_auc,
        "multiclass_log_loss": ll,
        "multiclass_brier_score": brier,
        "per_class": {
            cls_name: {
                "precision": round(float(per_p[i]), 4),
                "recall": round(float(per_r[i]), 4),
                "f1": round(float(per_f1[i]), 4),
                "support": int(per_sup[i])
            }
            for i, cls_name in enumerate(class_list)
        },
        "confusion_matrix": cm
    }

    return benchmark_results
