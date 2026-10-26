"""
DermaStack-XAI Explainable AI (XAI) Module: Grad-CAM Implementation.

Provides Gradient-weighted Class Activation Mapping (Grad-CAM) for deep
convolutional models in DermaStack-XAI:
    - Custom Lightweight CNN (target layer: 'conv4_block')
    - EfficientNetB0 (target layer: 'top_conv')
    - ResNet50 (target layer: 'conv5_block3_out')

Mathematical Formulation:
    1. Feature maps: A^k in R^{U x V}
    2. Class score: y^c (pre-softmax or post-softmax activation for class c)
    3. Channel weights: alpha_k^c = (1 / Z) * sum_i sum_j (d y^c / d A_{i,j}^k)
    4. Heatmap: L_{Grad-CAM}^c = ReLU(sum_k alpha_k^c * A^k)
    5. Normalization: L_norm = (L - min(L)) / (max(L) - min(L) + eps)
"""

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import matplotlib
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import DEFAULT_CONFIG, HAM10000_CLASSES, CLASS_NAMES_MAPPING
from src.models.common import preprocess_for_model
from src.utils.logger import get_logger

logger = get_logger("DermaStack-XAI.XAI.GradCAM")

# Mandatory target layers per project architecture specification
GRADCAM_TARGET_LAYERS: Dict[str, str] = {
    "custom_cnn": "conv4_block",
    "efficientnet_b0": "top_conv",
    "resnet50": "conv5_block3_out"
}

# Explicit ensemble attribution limitation disclaimer
ENSEMBLE_ATTRIBUTION_NOTICE = (
    "ENSEMBLE INTERPRETABILITY NOTICE: Grad-CAM generates visual feature activation heatmaps "
    "specifically for convolutional feature extractors (Custom CNN, EfficientNetB0, ResNet50). "
    "The Stage 4 Stacking Ensemble (Multinomial Logistic Regression) operates on a 21-dimensional "
    "feature space of predicted probability distributions rather than spatial image pixels. "
    "Consequently, base-model heatmaps explain individual neural network feature focus, while "
    "the ensemble decision is an optimal weighted combination of these base model predictions."
)


def validate_image_input(image: Union[np.ndarray, Image.Image, tf.Tensor]) -> np.ndarray:
    """
    Validates and standardizes an input image to a 3-channel RGB float32 array in [0.0, 255.0].

    Args:
        image: Input image as numpy array, PIL Image, or tf.Tensor.

    Returns:
        np.ndarray of shape (H, W, 3), dtype float32, range [0.0, 255.0].
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"), dtype=np.float32)
    elif isinstance(image, tf.Tensor):
        image = image.numpy().astype(np.float32)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.float32)
    else:
        raise TypeError(f"Unsupported image type '{type(image)}'. Expected np.ndarray, PIL.Image, or tf.Tensor.")

    # Remove batch dimension if shape is (1, H, W, 3)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Grad-CAM expects a single image, got batch shape {image.shape}")
        image = image[0]

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel RGB image of shape (H, W, 3), got shape {image.shape}")

    if np.isnan(image).any() or np.isinf(image).any():
        raise ValueError("Input image contains NaN or infinite values.")

    # Auto-scale if input appears normalized to [0, 1]
    if image.max() <= 1.0 and image.min() >= 0.0 and image.max() > 0.0:
        image = image * 255.0

    return image


def find_target_layer(model: keras.Model, layer_name: Optional[str] = None, model_name: Optional[str] = None) -> keras.layers.Layer:
    """
    Locates and validates the target convolutional layer in a Keras model.

    Args:
        model: Loaded Keras model instance.
        layer_name: Explicit layer name. If None, resolves from model_name.
        model_name: Model identifier ('custom_cnn', 'efficientnet_b0', 'resnet50').

    Returns:
        keras.layers.Layer instance.
    """
    resolved_name = layer_name
    if resolved_name is None and model_name is not None:
        clean_name = model_name.lower().replace("-", "_")
        resolved_name = GRADCAM_TARGET_LAYERS.get(clean_name)

    if resolved_name is None:
        # Check if model.name matches known architectures
        clean_model_name = getattr(model, "name", "").lower().replace("-", "_")
        resolved_name = GRADCAM_TARGET_LAYERS.get(clean_model_name)

    if resolved_name is None:
        raise ValueError(
            f"No target layer specified and could not infer from model name '{getattr(model, 'name', '')}'. "
            f"Specify layer_name explicitly or provide one of: {list(GRADCAM_TARGET_LAYERS.keys())}"
        )

    available_layers = [l.name for l in model.layers]
    if resolved_name not in available_layers:
        conv_candidates = [l.name for l in model.layers if "conv" in l.name.lower() or "activation" in l.name.lower()]
        raise ValueError(
            f"Target layer '{resolved_name}' not found in model '{getattr(model, 'name', 'unnamed')}'. "
            f"Available candidate layers: {conv_candidates[-5:]}"
        )

    target_layer = model.get_layer(resolved_name)
    return target_layer


def build_grad_model(model: keras.Model, target_layer_name: str) -> keras.Model:
    """
    Constructs a dual-output gradient model that returns both the target layer's
    activations and the final model predictions.

    Args:
        model: Original trained Keras model.
        target_layer_name: Name of the convolutional layer to extract activations from.

    Returns:
        keras.Model with outputs [target_layer.output, model.output].
    """
    target_layer = find_target_layer(model, layer_name=target_layer_name)
    try:
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
            name=f"{getattr(model, 'name', 'model')}_grad_cam"
        )
        return grad_model
    except Exception as e:
        raise RuntimeError(
            f"Failed to create gradient model for layer '{target_layer_name}' in '{getattr(model, 'name', 'model')}': {e}. "
            "Ensure the target layer is in the forward graph between model.inputs and model.output."
        )


class GradCAMExplainer:
    """
    Grad-CAM Explainer for Deep Convolutional Skin Lesion Classification Models.

    Generates class-discriminative activation maps by computing the gradients of a
    selected diagnostic class with respect to the final convolutional feature maps.
    """

    def __init__(
        self,
        model: keras.Model,
        model_name: str,
        target_layer_name: Optional[str] = None
    ):
        """
        Initializes the Grad-CAM explainer for a given model.

        Args:
            model: Trained Keras model instance.
            model_name: Identifier ('custom_cnn', 'efficientnet_b0', 'resnet50').
            target_layer_name: Optional target layer override. Defaults to GRADCAM_TARGET_LAYERS[model_name].
        """
        self.model = model
        self.model_name = model_name.lower().replace("-", "_")

        if self.model_name in GRADCAM_TARGET_LAYERS:
            self.target_layer_name = target_layer_name or GRADCAM_TARGET_LAYERS[self.model_name]
        else:
            self.target_layer_name = target_layer_name or "conv4_block"

        # Validate layer existence and construct dual-output gradient model
        self.target_layer = find_target_layer(self.model, layer_name=self.target_layer_name)
        self.grad_model = build_grad_model(self.model, target_layer_name=self.target_layer_name)
        self.classes: List[str] = list(HAM10000_CLASSES)

        logger.info(
            f"Initialized GradCAMExplainer for '{self.model_name}' targeting layer '{self.target_layer_name}' "
            f"with output shape {self.target_layer.output.shape}."
        )

    def preprocess_image(self, raw_image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> tf.Tensor:
        """
        Applies exact model-specific preprocessing matching the training pipeline.

        Args:
            raw_image: RGB float array in [0.0, 255.0] of shape (H, W, 3).
            target_size: Optional desired spatial dimension (H, W). If None, infers from model.input_shape.

        Returns:
            tf.Tensor of shape (1, H, W, 3) ready for model forward pass.
        """
        if target_size is None:
            if hasattr(self.model, "input_shape") and self.model.input_shape:
                shape = self.model.input_shape
                if isinstance(shape, list):
                    shape = shape[0]
                h, w = shape[1], shape[2]
                if h is not None and w is not None:
                    target_size = (int(h), int(w))
            if target_size is None:
                target_size = (224, 224)
        # Resize to model input dimensions using bilinear interpolation
        img_tensor = tf.convert_to_tensor(raw_image, dtype=tf.float32)
        if img_tensor.shape[0] != target_size[0] or img_tensor.shape[1] != target_size[1]:
            img_tensor = tf.image.resize(img_tensor, target_size, method=tf.image.ResizeMethod.BILINEAR)

        # Expand batch dimension -> (1, 224, 224, 3)
        batched = tf.expand_dims(img_tensor, axis=0)

        # Apply model-specific preprocessing dispatcher
        preprocessed = preprocess_for_model(batched, model_name=self.model_name)
        return preprocessed

    def compute_heatmap(
        self,
        preprocessed_tensor: tf.Tensor,
        class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """
        Computes the 2D Grad-CAM activation heatmap for a target class.

        Args:
            preprocessed_tensor: 4D tensor (1, 224, 224, 3) preprocessed for this model.
            class_idx: Target class index (0..6). If None, uses the model's top predicted class.

        Returns:
            Tuple of:
                - heatmap: 2D float32 array in [0.0, 1.0] of feature map spatial dimensions.
                - selected_class_idx: Integer index of the explained class.
                - class_probability: Softmax probability for the explained class.
                - all_predictions: Full (7,) probability vector from model.
        """
        if preprocessed_tensor.ndim != 4 or preprocessed_tensor.shape[0] != 1:
            raise ValueError(f"Expected input tensor of shape (1, H, W, 3), got {preprocessed_tensor.shape}")

        with tf.GradientTape() as tape:
            # Forward pass: extract convolutional feature maps and final class predictions
            conv_outputs, predictions = self.grad_model(preprocessed_tensor)

            all_predictions = np.asarray(predictions[0], dtype=np.float32)

            # Determine class to explain
            if class_idx is None:
                selected_class_idx = int(tf.argmax(predictions[0]))
            else:
                if not (0 <= class_idx < len(self.classes)):
                    raise ValueError(
                        f"Target class index {class_idx} out of range [0, {len(self.classes) - 1}]. "
                        f"Allowed classes: {self.classes}"
                    )
                selected_class_idx = int(class_idx)

            # Class score (loss objective for gradient tape)
            class_score = predictions[:, selected_class_idx]
            class_prob = float(all_predictions[selected_class_idx])

        # Backward pass: compute gradients of class score with respect to feature maps
        grads = tape.gradient(class_score, conv_outputs)

        if grads is None:
            raise RuntimeError(
                f"Gradients are None for target layer '{self.target_layer_name}'. "
                "The target layer may be disconnected from the classification output."
            )

        if tf.math.reduce_any(tf.math.is_nan(grads)) or tf.math.reduce_any(tf.math.is_inf(grads)):
            raise ValueError("Non-finite (NaN or Inf) gradients encountered during Grad-CAM backpropagation.")

        # Spatial Global Average Pooling of gradients to compute channel importance weights
        # alpha_k^c = (1 / Z) * sum_i sum_j (d y^c / d A_{i,j}^k)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by importance weights
        # L_{Grad-CAM}^c = sum_k (alpha_k^c * A^k)
        conv_outputs_single = conv_outputs[0]
        heatmap = conv_outputs_single @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply Rectified Linear Unit (ReLU) to isolate positively contributing features
        heatmap = tf.maximum(heatmap, 0.0)

        # Safe normalization to [0.0, 1.0]
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0.0:
            heatmap = heatmap / max_val
        else:
            # Constant or zero activation: return all-zero heatmap without division by zero
            heatmap = tf.zeros_like(heatmap)

        heatmap_arr = heatmap.numpy().astype(np.float32)
        return heatmap_arr, selected_class_idx, class_prob, all_predictions

    def generate_overlay(
        self,
        raw_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "jet"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resizes the heatmap to original image dimensions and creates a blended RGB overlay.

        Args:
            raw_image: Original RGB float array in [0.0, 255.0] of shape (H, W, 3).
            heatmap: 2D float32 array in [0.0, 1.0].
            alpha: Heatmap blend intensity in [0.0, 1.0]. Default 0.4.
            colormap: Matplotlib colormap name. Default 'jet'.

        Returns:
            Tuple of:
                - resized_heatmap: (H, W) float32 array in [0.0, 1.0].
                - overlay: (H, W, 3) uint8 array in [0, 255] ready for display or saving.
        """
        orig_h, orig_w = raw_image.shape[0], raw_image.shape[1]

        # 1. Resize heatmap to match original image dimensions using bilinear interpolation
        heatmap_tensor = tf.convert_to_tensor(heatmap, dtype=tf.float32)[..., tf.newaxis]
        resized_tensor = tf.image.resize(
            heatmap_tensor,
            (orig_h, orig_w),
            method=tf.image.ResizeMethod.BILINEAR
        )
        resized_heatmap = tf.squeeze(resized_tensor).numpy().astype(np.float32)

        # Re-normalize resized heatmap to ensure [0, 1] range
        max_r = resized_heatmap.max()
        if max_r > 0.0:
            resized_heatmap = np.clip(resized_heatmap / max_r, 0.0, 1.0)
        else:
            resized_heatmap = np.zeros_like(resized_heatmap)

        # 2. Colorize heatmap
        try:
            cmap = matplotlib.colormaps[colormap]
        except KeyError:
            logger.warning(f"Colormap '{colormap}' not found, falling back to 'jet'.")
            cmap = matplotlib.colormaps["jet"]

        colored_heatmap = cmap(resized_heatmap)[:, :, :3]  # Drop alpha channel -> (H, W, 3) in [0, 1]

        # 3. Normalize raw image to [0, 1] for blending
        base_norm = np.clip(raw_image / 255.0, 0.0, 1.0)

        # 4. Alpha blend: Overlay = alpha * Heatmap + (1 - alpha) * Base
        overlay_float = alpha * colored_heatmap + (1.0 - alpha) * base_norm
        overlay_uint8 = np.clip(overlay_float * 255.0, 0.0, 255.0).astype(np.uint8)

        return resized_heatmap, overlay_uint8

    def explain(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        class_idx: Optional[int] = None,
        alpha: float = 0.4,
        colormap: str = "jet"
    ) -> Dict[str, Any]:
        """
        Complete end-to-end Grad-CAM explanation pipeline for an input image.

        Args:
            image: Image array, PIL Image, or file path.
            class_idx: Target diagnostic class index (0..6). Defaults to top predicted class.
            alpha: Overlay opacity fraction.
            colormap: Colormap identifier.

        Returns:
            Dictionary containing explanation metadata, probabilities, heatmap, and overlay.
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            raw_image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        else:
            raw_image = validate_image_input(image)

        # 1. Preprocess
        preprocessed = self.preprocess_image(raw_image)

        # 2. Compute Heatmap
        raw_heatmap, target_idx, prob, all_preds = self.compute_heatmap(
            preprocessed, class_idx=class_idx
        )

        # 3. Generate Overlay
        resized_heatmap, overlay_img = self.generate_overlay(
            raw_image, raw_heatmap, alpha=alpha, colormap=colormap
        )

        class_name = self.classes[target_idx]
        class_desc = CLASS_NAMES_MAPPING.get(class_name, class_name)

        return {
            "model_name": self.model_name,
            "target_layer": self.target_layer_name,
            "explained_class_idx": target_idx,
            "explained_class_name": class_name,
            "explained_class_description": class_desc,
            "class_probability": round(float(prob), 4),
            "all_probabilities": {c: round(float(all_preds[i]), 4) for i, c in enumerate(self.classes)},
            "heatmap_feature_dimensions": list(raw_heatmap.shape),
            "heatmap_original_dimensions": list(resized_heatmap.shape),
            "heatmap_raw": raw_heatmap,
            "heatmap_resized": resized_heatmap,
            "overlay_image": overlay_img,
            "original_image": raw_image.astype(np.uint8)
        }


def explain_image_across_models(
    image: Union[np.ndarray, Image.Image, str, Path],
    checkpoint_dir: Optional[Path] = None,
    fold_idx: int = 0,
    class_idx: Optional[int] = None,
    alpha: float = 0.4
) -> Dict[str, Any]:
    """
    Generates comparative Grad-CAM explanations for an image across all three base models:
    Custom CNN, EfficientNetB0, and ResNet50.

    Includes the explicit attribution notice clarifying the distinction between
    convolutional base-model explanations and the Logistic Regression stacking ensemble.

    Args:
        image: Image input.
        checkpoint_dir: Directory containing trained .keras models.
        fold_idx: Fold checkpoint index to load (0..4). Default 0.
        class_idx: Specific class index to explain across all models.
        alpha: Overlay blending parameter.

    Returns:
        Dictionary with per-model explanations and ensemble attribution notice.
    """
    chk_dir = checkpoint_dir or DEFAULT_CONFIG.paths.checkpoints_dir
    results: Dict[str, Any] = {
        "fold_idx": fold_idx,
        "ensemble_attribution_notice": ENSEMBLE_ATTRIBUTION_NOTICE,
        "base_model_explanations": {}
    }

    for model_name in ["custom_cnn", "efficientnet_b0", "resnet50"]:
        ckpt_path = chk_dir / f"{model_name}_fold_{fold_idx}_best.keras"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {ckpt_path}")

        logger.info(f"Loading {ckpt_path.name} for Grad-CAM explanation...")
        model = keras.models.load_model(str(ckpt_path), compile=False)

        explainer = GradCAMExplainer(model=model, model_name=model_name)
        explanation = explainer.explain(image=image, class_idx=class_idx, alpha=alpha)
        results["base_model_explanations"][model_name] = explanation

        # Clean session memory
        del model
        keras.backend.clear_session()

    return results


def run_stage_5_visualizations(
    num_samples_per_class: int = 1,
    output_dir: Optional[Path] = None,
    fold_idx: int = 0
) -> Dict[str, Any]:
    """
    Generates a reproducible suite of research-quality Grad-CAM visualizations
    across all seven diagnostic categories from the validated HAM10000 training partition.

    Args:
        num_samples_per_class: Number of representative images per class (default 1 -> 7 total images).
        output_dir: Directory to save generated PNG overlays and summary manifest.
        fold_idx: Checkpoint fold to use (default 0).

    Returns:
        Summary report dictionary of generated visualizations.
    """
    out_dir = output_dir or (DEFAULT_CONFIG.paths.root / "outputs" / "gradcam")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load validated metadata
    meta_path = DEFAULT_CONFIG.paths.processed_data_dir / "validated_metadata.csv"
    splits_path = DEFAULT_CONFIG.paths.splits_dir / "train_val_test_splits.json"

    if not meta_path.exists() or not splits_path.exists():
        raise FileNotFoundError("Missing metadata or split manifest.")

    meta_df = pd.read_csv(meta_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    # Use only validation or training partition images (STRICTLY NO HOLDOUT TEST)
    val_set = set(splits["val"])
    meta_val = meta_df[meta_df["image_id"].isin(val_set)].copy()

    # Select representative samples per class
    selected_records = []
    for cls_name in HAM10000_CLASSES:
        cls_df = meta_val[meta_val["dx"] == cls_name]
        if len(cls_df) > 0:
            sample = cls_df.iloc[0]
            selected_records.append(sample)

    logger.info(f"Selected {len(selected_records)} representative validation images across 7 classes for Grad-CAM visualization.")

    manifest_entries = []
    classes = list(HAM10000_CLASSES)

    for record in selected_records:
        img_id = record["image_id"]
        true_dx = record["dx"]
        true_idx = classes.index(true_dx)
        img_path = Path(record["image_path"])

        if not img_path.exists():
            logger.warning(f"Image file missing on disk: {img_path}")
            continue

        raw_img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)

        record_entry: Dict[str, Any] = {
            "image_id": img_id,
            "true_diagnosis": true_dx,
            "true_class_index": true_idx,
            "explanations": {}
        }

        for model_name in ["custom_cnn", "efficientnet_b0", "resnet50"]:
            ckpt_path = DEFAULT_CONFIG.paths.checkpoints_dir / f"{model_name}_fold_{fold_idx}_best.keras"
            model = keras.models.load_model(str(ckpt_path), compile=False)

            explainer = GradCAMExplainer(model=model, model_name=model_name)

            # Generate explanation for the true class
            exp_true = explainer.explain(raw_img, class_idx=true_idx)

            # Save overlay image to disk
            overlay_filename = f"{img_id}_{model_name}_{true_dx}_overlay.png"
            overlay_path = out_dir / overlay_filename
            Image.fromarray(exp_true["overlay_image"]).save(overlay_path)

            pred_top_idx = int(np.argmax(list(exp_true["all_probabilities"].values())))
            pred_top_name = classes[pred_top_idx]
            is_correct = (pred_top_idx == true_idx)

            record_entry["explanations"][model_name] = {
                "checkpoint": ckpt_path.name,
                "target_layer": explainer.target_layer_name,
                "predicted_class": pred_top_name,
                "predicted_probability": exp_true["all_probabilities"][pred_top_name],
                "true_class_probability": exp_true["all_probabilities"][true_dx],
                "is_prediction_correct": is_correct,
                "heatmap_shape": exp_true["heatmap_feature_dimensions"],
                "saved_overlay": str(overlay_path.resolve())
            }

            del model
            keras.backend.clear_session()

        manifest_entries.append(record_entry)

    # Save summary report
    summary_report: Dict[str, Any] = {
        "generated_at": datetime.datetime.now().isoformat(),
        "fold_evaluated": fold_idx,
        "target_layers_used": GRADCAM_TARGET_LAYERS,
        "ensemble_attribution_notice": ENSEMBLE_ATTRIBUTION_NOTICE,
        "total_images_visualized": len(manifest_entries),
        "visualizations": manifest_entries
    }

    manifest_file = out_dir / "gradcam_visualization_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"Saved Grad-CAM visualization manifest to: {manifest_file}")
    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermaStack-XAI Stage 5 Grad-CAM Explainer")
    parser.add_argument("--samples-per-class", type=int, default=1, help="Samples to visualize per class.")
    parser.add_argument("--fold", type=int, default=0, help="Fold checkpoint to load (0..4).")
    args = parser.parse_args()

    import pandas as pd
    report = run_stage_5_visualizations(
        num_samples_per_class=args.samples_per_class,
        fold_idx=args.fold
    )
    print(f"\nStage 5 Grad-CAM Visualizations Complete: {report['total_images_visualized']} images generated.")
