"""
DermaStack-XAI Inference Pipeline.

Provides all business logic for single-image inference against:
  - Individual 5-fold ensemble base models (Custom CNN, EfficientNetB0, ResNet50)
  - Standard Stacking meta-classifier  (class_weight=None)
  - Balanced Stacking meta-classifier  (class_weight='balanced')
  - Grad-CAM visual explanations for all three convolutional architectures

This module is deliberately separated from Streamlit UI code so that every
function can be unit-tested without launching a web server.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ── lazy heavy imports so the module can be imported quickly at the top of the
# Streamlit script without blocking the browser on every rerun
def _load_keras():
    import keras
    return keras

def _load_tf():
    import tensorflow as tf
    return tf

from src.config import (
    DEFAULT_CONFIG,
    HAM10000_CLASSES,
    CLASS_NAMES_MAPPING,
    CLASS_DESCRIPTIONS,
    XAIConfig,
)
from src.models.common import preprocess_for_model
from src.models.stacking_meta import StackingMetaClassifier, validate_stacking_features
from src.utils.logger import get_logger

logger = get_logger("DermaStack-XAI.Inference")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODELS: List[str] = ["custom_cnn", "efficientnet_b0", "resnet50"]
N_FOLDS: int = 5
TARGET_SIZE: Tuple[int, int] = (224, 224)

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "custom_cnn": "Custom Lightweight CNN",
    "efficientnet_b0": "EfficientNetB0",
    "resnet50": "ResNet50",
    "standard_stacking": "Standard Stacking (Logistic Regression)",
    "balanced_stacking": "Balanced Stacking (Logistic Regression, class_weight='balanced')",
}

TARGET_LAYERS: Dict[str, str] = {
    "custom_cnn": "conv4_block",
    "efficientnet_b0": "top_conv",
    "resnet50": "conv5_block3_out",
}

ENSEMBLE_NOTICE: str = (
    "ℹ️ **Ensemble Attribution**: Grad-CAM heatmaps are generated from the individual "
    "convolutional base models (Custom CNN, EfficientNetB0, ResNet50). The Logistic Regression "
    "stacking meta-learner combines their probability outputs numerically and does not operate "
    "on image pixels directly. Therefore base-model heatmaps illustrate what each neural network "
    "attended to, not what the ensemble decision boundary optimised."
)


# ─────────────────────────────────────────────────────────────────────────────
# Image validation and preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_load_image(
    file_bytes_or_path,
    max_pixels: int = 178_956_970,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Validates and loads an uploaded image into a uint8 RGB numpy array.

    Args:
        file_bytes_or_path: Bytes-like object, file path, or PIL Image.
        max_pixels: Maximum allowed image size (default: PIL safety limit).

    Returns:
        Tuple of (rgb_array uint8 (H, W, 3), original_size (W, H)).

    Raises:
        ValueError: For unreadable, non-RGB-convertible, or oversized images.
    """
    try:
        if isinstance(file_bytes_or_path, (str, Path)):
            pil_img = Image.open(file_bytes_or_path)
        elif isinstance(file_bytes_or_path, np.ndarray):
            pil_img = Image.fromarray(file_bytes_or_path.astype(np.uint8))
        elif isinstance(file_bytes_or_path, Image.Image):
            pil_img = file_bytes_or_path
        else:
            pil_img = Image.open(io.BytesIO(file_bytes_or_path.read()
                                           if hasattr(file_bytes_or_path, "read")
                                           else bytes(file_bytes_or_path)))
    except Exception as exc:
        raise ValueError(f"Cannot open image: {exc}") from exc

    w, h = pil_img.size
    if w * h > max_pixels:
        raise ValueError(
            f"Image is too large ({w}×{h} = {w*h:,} pixels). "
            f"Maximum allowed: {max_pixels:,} pixels."
        )
    if w < 32 or h < 32:
        raise ValueError(
            f"Image is too small ({w}×{h}). Minimum dimension: 32×32 pixels."
        )

    rgb_img = pil_img.convert("RGB")
    rgb_array = np.array(rgb_img, dtype=np.uint8)

    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(
            f"Expected 3-channel RGB image; got shape {rgb_array.shape}."
        )
    if np.isnan(rgb_array).any() or np.isinf(rgb_array).any():
        raise ValueError("Image contains non-finite pixel values.")

    return rgb_array, (w, h)


def resize_and_preprocess(
    rgb_array: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """
    Resizes an RGB uint8 image to (224, 224) and applies model-specific preprocessing.

    Args:
        rgb_array: uint8 array of shape (H, W, 3).
        model_name: One of 'custom_cnn', 'efficientnet_b0', 'resnet50'.

    Returns:
        float32 numpy array of shape (1, 224, 224, 3) ready for model.predict().
    """
    tf = _load_tf()
    img_f32 = rgb_array.astype(np.float32)
    # Resize
    tensor = tf.convert_to_tensor(img_f32[np.newaxis], dtype=tf.float32)
    resized = tf.image.resize(tensor, TARGET_SIZE, method="bilinear").numpy()
    # Model-specific preprocessing
    preprocessed = preprocess_for_model(resized, model_name=model_name)
    return np.asarray(preprocessed, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (designed to be wrapped with Streamlit cache_resource in the UI layer)
# ─────────────────────────────────────────────────────────────────────────────

def load_single_fold_model(model_name: str, fold_idx: int):
    """
    Loads one fold checkpoint. Returns the Keras model.

    Raises FileNotFoundError if the checkpoint is missing.
    """
    keras = _load_keras()
    checkpoint_dir = DEFAULT_CONFIG.paths.checkpoints_dir

    new_path = checkpoint_dir / (
        f"{model_name}_fold_{fold_idx}_best_val_accuracy.keras"
    )
    legacy_path = checkpoint_dir / (
        f"{model_name}_fold_{fold_idx}_best.keras"
    )

    if new_path.exists():
        ckpt_path = new_path
    elif legacy_path.exists():
        ckpt_path = legacy_path
    else:
        raise FileNotFoundError(
            f"No checkpoint found for {model_name}, fold {fold_idx}. "
            f"Checked: {new_path} and {legacy_path}"
        )

    logger.info(f"Loading checkpoint: {ckpt_path.name}")
    model = keras.models.load_model(str(ckpt_path), compile=False)
    return model

def load_all_fold_models(model_name: str) -> list:
    """Returns a list of 5 Keras models (one per fold) for the given architecture."""
    return [load_single_fold_model(model_name, fold) for fold in range(N_FOLDS)]


def load_stacking_meta_model(balanced: bool = False) -> StackingMetaClassifier:
    """Loads and returns the standard or balanced stacking meta-classifier."""
    filename = "stacking_meta_model_balanced.joblib" if balanced else "stacking_meta_model.joblib"
    path = DEFAULT_CONFIG.paths.meta_models_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Meta-model not found: {path}. "
            "Ensure Stage 4 stacking has been completed."
        )
    return StackingMetaClassifier.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Single-image prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_single_model(
    rgb_array: np.ndarray,
    model_name: str,
    fold_models: list,
) -> np.ndarray:
    """
    Computes the 5-fold average probability vector for one base architecture.

    Args:
        rgb_array: uint8 (H, W, 3) numpy array.
        model_name: Architecture identifier.
        fold_models: List of 5 loaded Keras models.

    Returns:
        float32 probability array of shape (7,).
    """
    preprocessed = resize_and_preprocess(rgb_array, model_name)
    fold_probs = []
    for model in fold_models:
        p = model.predict(preprocessed, verbose=0)
        p = np.asarray(p, dtype=np.float32).squeeze()  # (7,)
        if p.shape != (7,):
            raise ValueError(f"Unexpected model output shape {p.shape}; expected (7,).")
        if not np.all(np.isfinite(p)):
            raise ValueError("Model output contains non-finite values.")
        p = np.clip(p, 0.0, 1.0)
        p /= np.maximum(p.sum(), 1e-12)
        fold_probs.append(p)

    mean_p = np.mean(fold_probs, axis=0).astype(np.float32)
    mean_p /= np.maximum(mean_p.sum(), 1e-12)
    return mean_p


def predict_fold_probabilities(
    rgb_array: np.ndarray,
    model_name: str,
    fold_models: list,
) -> list:
    """Return one normalized 7-class probability vector per fold, in fold order."""
    preprocessed = resize_and_preprocess(rgb_array, model_name)
    results = []
    for fold_idx, model in enumerate(fold_models):
        p = np.asarray(model.predict(preprocessed, verbose=0), dtype=np.float32).squeeze()
        if p.shape != (7,):
            raise ValueError(
                f"Unexpected prediction shape for {model_name} fold {fold_idx}: {p.shape}; expected (7,)"
            )
        if not np.all(np.isfinite(p)):
            raise ValueError(f"Non-finite predictions for {model_name} fold {fold_idx}.")
        p = np.clip(p, 0.0, 1.0)
        p /= np.maximum(p.sum(), 1e-12)
        results.append(p)
    if len(results) != N_FOLDS:
        raise ValueError(f"Expected {N_FOLDS} fold models for {model_name}, got {len(results)}.")
    return results


def build_stacking_feature_vector(
    cnn_probs: np.ndarray,
    eff_probs: np.ndarray,
    res_probs: np.ndarray,
) -> np.ndarray:
    """
    Concatenates three (7,) base model probability vectors into the (1, 21)
    stacking feature matrix in canonical column order:
        custom_cnn(0:7) | efficientnet_b0(7:14) | resnet50(14:21)

    Returns:
        float32 array of shape (1, 21).
    """
    X = np.concatenate([cnn_probs, eff_probs, res_probs])[np.newaxis].astype(np.float32)
    validate_stacking_features(X, expected_rows=1)
    return X


def run_full_inference(
    rgb_array: np.ndarray,
    fold_models_cnn: list,
    fold_models_eff: list,
    fold_models_res: list,
    meta_std: StackingMetaClassifier,
    meta_bal: StackingMetaClassifier,
) -> Dict[str, Any]:
    """Run inference with fold-aligned stacking features.

    OOF meta-training rows were generated from a single held-out fold model per
    architecture. Therefore, do NOT average the five base-model probabilities
    before feeding them to the meta-classifier: that creates a train/inference
    feature-distribution mismatch. Instead, build one 21-feature row per fold,
    obtain meta predictions per fold, then average the five meta probabilities.
    Base-model display probabilities remain the mean across folds.
    """
    cnn_by_fold = predict_fold_probabilities(rgb_array, "custom_cnn", fold_models_cnn)
    eff_by_fold = predict_fold_probabilities(rgb_array, "efficientnet_b0", fold_models_eff)
    res_by_fold = predict_fold_probabilities(rgb_array, "resnet50", fold_models_res)

    # Base-model outputs shown in the UI remain the standard 5-fold averages.
    cnn_p = np.mean(cnn_by_fold, axis=0).astype(np.float32)
    eff_p = np.mean(eff_by_fold, axis=0).astype(np.float32)
    res_p = np.mean(res_by_fold, axis=0).astype(np.float32)
    for p in (cnn_p, eff_p, res_p):
        p /= np.maximum(p.sum(), 1e-12)

    std_by_fold = []
    bal_by_fold = []
    fold_feature_rows = []
    for fold_idx in range(N_FOLDS):
        X_fold = build_stacking_feature_vector(
            cnn_by_fold[fold_idx], eff_by_fold[fold_idx], res_by_fold[fold_idx]
        )
        fold_feature_rows.append(X_fold.squeeze(0))
        std_by_fold.append(np.asarray(meta_std.predict_proba(X_fold)).squeeze().astype(np.float32))
        bal_by_fold.append(np.asarray(meta_bal.predict_proba(X_fold)).squeeze().astype(np.float32))

    std_p = np.mean(std_by_fold, axis=0).astype(np.float32)
    bal_p = np.mean(bal_by_fold, axis=0).astype(np.float32)
    std_p /= np.maximum(std_p.sum(), 1e-12)
    bal_p /= np.maximum(bal_p.sum(), 1e-12)

    def _fmt(p: np.ndarray) -> Dict[str, float]:
        return {c: round(float(p[i]), 4) for i, c in enumerate(HAM10000_CLASSES)}

    # Keep the legacy single-row key for downstream UI compatibility; it now
    # represents the mean base probabilities. Also expose fold features for audit.
    X_stack_mean = build_stacking_feature_vector(cnn_p, eff_p, res_p)
    return {
        "custom_cnn": {"probs": cnn_p, "prob_dict": _fmt(cnn_p)},
        "efficientnet_b0": {"probs": eff_p, "prob_dict": _fmt(eff_p)},
        "resnet50": {"probs": res_p, "prob_dict": _fmt(res_p)},
        "standard_stacking": {"probs": std_p, "prob_dict": _fmt(std_p)},
        "balanced_stacking": {"probs": bal_p, "prob_dict": _fmt(bal_p)},
        "X_stacking_features": X_stack_mean,
        "X_stacking_features_by_fold": np.asarray(fold_feature_rows, dtype=np.float32),
        "standard_stacking_by_fold": np.asarray(std_by_fold, dtype=np.float32),
        "balanced_stacking_by_fold": np.asarray(bal_by_fold, dtype=np.float32),
        "stacking_feature_strategy": "fold_aligned_meta_predictions_v1",
    }


def top_prediction(probs: np.ndarray) -> Tuple[str, str, float]:
    """
    Returns (class_code, display_name, probability) for the argmax class.
    """
    idx = int(np.argmax(probs))
    code = HAM10000_CLASSES[idx]
    name = CLASS_NAMES_MAPPING.get(code, code)
    return code, name, float(probs[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM integration
# ─────────────────────────────────────────────────────────────────────────────

def run_gradcam(
    rgb_array: np.ndarray,
    model_name: str,
    fold_idx: int,
    class_idx: Optional[int],
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Dict[str, Any]:
    """
    Runs Grad-CAM explanation for one model/fold combination.

    Args:
        rgb_array: uint8 (H, W, 3) source image.
        model_name: 'custom_cnn', 'efficientnet_b0', or 'resnet50'.
        fold_idx:  Fold checkpoint to load (0..4).
        class_idx: Diagnostic class to explain (0..6). None → argmax.
        alpha:     Heatmap blending intensity.
        colormap:  Matplotlib colormap name.

    Returns:
        Dict with keys: overlay_image, heatmap_resized, explained_class_name,
        explained_class_idx, class_probability, all_probabilities, target_layer.
    """
    from src.xai.gradcam import GradCAMExplainer

    model = load_single_fold_model(model_name, fold_idx)
    explainer = GradCAMExplainer(model=model, model_name=model_name)
    img_f32 = rgb_array.astype(np.float32)
    result = explainer.explain(
        image=img_f32,
        class_idx=class_idx,
        alpha=alpha,
        colormap=colormap,
    )

    keras = _load_keras()
    del model
    keras.backend.clear_session()
    return result


def format_probability_table(prob_dict: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Converts a probability dict to a sorted list of row dicts for display.

    Returns list sorted by probability descending.
    """
    rows = []
    for code, prob in prob_dict.items():
        rows.append({
            "Class": code,
            "Diagnosis": CLASS_NAMES_MAPPING.get(code, code),
            "Probability": prob,
            "Probability (%)": f"{prob * 100:.1f}%",
        })
    return sorted(rows, key=lambda r: r["Probability"], reverse=True)
