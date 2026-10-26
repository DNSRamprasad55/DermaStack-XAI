"""
DermaStack-XAI Publication Plotting Module.

Generates publication-quality, reproducible figures for dermoscopic skin lesion evaluation:
    1. Multi-Class Confusion Matrix Heatmaps (Raw counts and normalized proportions)
    2. Multi-Class One-vs-Rest ROC Curves (per class + macro-average)
    3. Cross-Model Benchmark Comparison Charts (Accuracy, Macro-F1, Macro-Recall)
    4. Multi-Class Probability Calibration / Reliability Curves
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

from src.config import HAM10000_CLASSES, CLASS_NAMES_MAPPING
from src.utils.logger import get_logger

logger = get_logger("DermaStack-XAI.Evaluation.Plots")

# Publication styling defaults
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def save_plot_source_data(data: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Saves raw numerical source data for a figure to JSON for full reproducibility."""
    out_p = Path(output_path).with_suffix(".json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_p


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    classes: Optional[List[str]] = None,
    model_name: str = "model",
    output_path: Optional[Union[str, Path]] = None,
    normalize: bool = False,
    title_suffix: str = "Holdout Test Set"
) -> Path:
    """
    Plots and saves a publication-quality confusion matrix heatmap.

    Args:
        cm: 7x7 contingency matrix.
        classes: Ordered class names.
        model_name: Identifier for title.
        output_path: Path to save the PNG image.
        normalize: Whether to display percentages normalized by true row support.
        title_suffix: Dataset identifier string.

    Returns:
        Path of the saved image file.
    """
    class_list = classes or list(HAM10000_CLASSES)
    cm_arr = np.asarray(cm, dtype=np.float32)

    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_display = np.where(row_sums > 0, cm_arr / row_sums, 0.0)
        fmt = ".2%"
        vmax = 1.0
        cbar_label = "Recall (Normalized Proportion)"
    else:
        cm_display = cm_arr.astype(int)
        fmt = "d"
        vmax = int(cm_arr.max())
        cbar_label = "Sample Count"

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_list,
        yticklabels=class_list,
        cbar_kws={"label": cbar_label},
        ax=ax,
        linewidths=0.5,
        linecolor="#e0e0e0"
    )

    norm_str = " (Normalized)" if normalize else ""
    ax.set_title(f"{model_name} Confusion Matrix{norm_str}\n({title_suffix}, N = {int(cm_arr.sum())})", pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis", labelpad=8, fontweight="bold")
    ax.set_ylabel("Ground Truth Diagnosis", labelpad=8, fontweight="bold")
    plt.tight_layout()

    out_file = Path(output_path or f"outputs/evaluation/{model_name.lower().replace(' ', '_')}_cm.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)

    # Save source data
    save_plot_source_data({
        "model_name": model_name,
        "classes": class_list,
        "confusion_matrix_raw": cm_arr.tolist(),
        "normalized": normalize
    }, out_file)

    logger.info(f"Saved confusion matrix plot to: {out_file}")
    return out_file


def plot_multiclass_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: Optional[List[str]] = None,
    model_name: str = "model",
    output_path: Optional[Union[str, Path]] = None,
    title_suffix: str = "Holdout Test Set"
) -> Path:
    """
    Plots multi-class One-vs-Rest ROC curves for all classes plus macro-average.

    Args:
        y_true: Ground truth integer class indices (N,).
        y_prob: Predicted probability distributions (N, 7).
        classes: Ordered class names.
        model_name: Display name.
        output_path: Target PNG path.
        title_suffix: Dataset descriptor.

    Returns:
        Path of saved figure.
    """
    class_list = classes or list(HAM10000_CLASSES)
    num_classes = len(class_list)
    y_t = np.asarray(y_true, dtype=np.int32)
    y_p = np.asarray(y_prob, dtype=np.float32)

    y_one_hot = np.eye(num_classes, dtype=np.float32)[y_t]

    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}

    fig, ax = plt.subplots(figsize=(8.5, 7))

    # Palette with distinct colors
    palette = sns.color_palette("tab10", num_classes)

    for i, cls_name in enumerate(class_list):
        fpr_dict[cls_name], tpr_dict[cls_name], _ = roc_curve(y_one_hot[:, i], y_p[:, i])
        roc_auc_dict[cls_name] = auc(fpr_dict[cls_name], tpr_dict[cls_name])
        ax.plot(
            fpr_dict[cls_name],
            tpr_dict[cls_name],
            color=palette[i],
            lw=1.8,
            label=f"{cls_name.upper()} (AUC = {roc_auc_dict[cls_name]:.3f})"
        )

    # Compute macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in class_list]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in class_list:
        mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
    mean_tpr /= num_classes

    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(
        all_fpr,
        mean_tpr,
        color="black",
        linestyle="--",
        lw=2.5,
        label=f"Macro-Average (AUC = {macro_auc:.3f})"
    )

    # Chance baseline
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.2, label="Chance Baseline")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", labelpad=8, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", labelpad=8, fontweight="bold")
    ax.set_title(f"{model_name} One-vs-Rest ROC Curves\n({title_suffix}, N = {len(y_t)})", pad=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_file = Path(output_path or f"outputs/evaluation/{model_name.lower().replace(' ', '_')}_roc.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)

    # Save source data
    save_plot_source_data({
        "model_name": model_name,
        "classes": class_list,
        "roc_auc_per_class": {c: round(float(roc_auc_dict[c]), 4) for c in class_list},
        "macro_auc": round(float(macro_auc), 4)
    }, out_file)

    logger.info(f"Saved ROC curves plot to: {out_file}")
    return out_file


def plot_model_comparison(
    models_metrics: Dict[str, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    title_suffix: str = "Holdout Test Set"
) -> Path:
    """
    Generates a grouped bar chart comparing models across Accuracy, Macro-F1, and Macro-Recall.

    Args:
        models_metrics: Dict mapping display names to metrics dicts.
        output_path: Target PNG path.
        title_suffix: Partition identifier.

    Returns:
        Path of saved figure.
    """
    model_names = list(models_metrics.keys())
    metrics_to_plot = ["accuracy", "macro_f1", "macro_recall"]
    metric_labels = ["Accuracy", "Macro-F1", "Macro-Recall (Sensitivity)"]

    data = {m_label: [models_metrics[m][m_key] for m in model_names] for m_key, m_label in zip(metrics_to_plot, metric_labels)}

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2b5c8f", "#e26d5c", "#38b000"]
    for i, (m_label, values) in enumerate(data.items()):
        offset = (i - 1) * width
        rects = ax.bar(x + offset, values, width, label=m_label, color=colors[i], edgecolor="black", linewidth=0.5)
        # Value labels above bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2%}" if "Accuracy" in m_label else f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=45
            )

    ax.set_ylabel("Score", labelpad=8, fontweight="bold")
    ax.set_title(f"DermaStack-XAI Model Benchmark Comparison\n({title_suffix})", pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontweight="bold")
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_file = Path(output_path or "outputs/evaluation/model_benchmark_comparison.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)

    # Save source data
    save_plot_source_data({
        "models": model_names,
        "metrics": data
    }, out_file)

    logger.info(f"Saved model comparison plot to: {out_file}")
    return out_file


def plot_calibration_curves(
    models_probs: Dict[str, np.ndarray],
    y_true: np.ndarray,
    classes: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    n_bins: int = 10,
    title_suffix: str = "Holdout Test Set"
) -> Path:
    """
    Plots multi-class calibration / reliability curves for evaluated models.

    Computes one-vs-rest empirical probabilities across all classes.

    Args:
        models_probs: Dict mapping model name to (N, 7) probability matrix.
        y_true: Target class indices (N,).
        classes: Ordered class names.
        output_path: Target PNG path.
        n_bins: Calibration bin count.
        title_suffix: Partition identifier.

    Returns:
        Path of saved figure.
    """
    class_list = classes or list(HAM10000_CLASSES)
    num_classes = len(class_list)
    y_one_hot = np.eye(num_classes, dtype=np.float32)[y_true]

    fig, ax = plt.subplots(figsize=(8.5, 7))

    # Perfectly calibrated line
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated", lw=1.5)

    colors = sns.color_palette("tab10", len(models_probs))
    source_records = {}

    for (m_name, probs), col in zip(models_probs.items(), colors):
        # Flatten one-vs-rest across all classes to show global multiclass calibration
        y_true_flat = y_one_hot.ravel()
        probs_flat = probs.ravel()

        prob_true, prob_pred = calibration_curve(y_true_flat, probs_flat, n_bins=n_bins, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", lw=1.8, color=col, label=f"{m_name}")

        source_records[m_name] = {
            "prob_pred": [round(float(v), 4) for v in prob_pred],
            "prob_true": [round(float(v), 4) for v in prob_true]
        }

    ax.set_xlabel("Mean Predicted Probability", labelpad=8, fontweight="bold")
    ax.set_ylabel("Empirical Fraction of Positives", labelpad=8, fontweight="bold")
    ax.set_title(f"Multi-Class Calibration Curves\n({title_suffix})", pad=12, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_file = Path(output_path or "outputs/evaluation/model_calibration_curves.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)

    # Save source data
    save_plot_source_data(source_records, out_file)

    logger.info(f"Saved calibration curves plot to: {out_file}")
    return out_file
