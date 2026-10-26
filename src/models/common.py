"""
DermaStack-XAI Common Model Utilities.
Defines shared preprocessing routines, classification head construction,
and model parameter auditing.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import keras
from keras import layers
from src.config import DEFAULT_CONFIG, HAM10000_CLASSES


def get_model_summary_dict(model: keras.Model) -> Dict[str, int]:
    """
    Computes total, trainable, and non-trainable parameter counts.

    Returns:
        Dictionary with 'total_params', 'trainable_params', 'non_trainable_params'.
    """
    total_params = int(model.count_params())
    trainable_params = int(sum(
        int(np.prod(v.shape)) for v in model.trainable_weights
    ))
    non_trainable_params = int(sum(
        int(np.prod(v.shape)) for v in model.non_trainable_weights
    ))
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params
    }


def preprocess_for_model(
    images,
    model_name: str
):
    """
    Applies the model-specific preprocessing function to raw RGB image tensors (values in [0, 255]).
    Supports both NumPy ndarrays and TensorFlow Tensors.

    Args:
        images: Float or uint8 numpy array or tf.Tensor with values in [0, 255], shape (N, H, W, 3) or (H, W, 3).
        model_name: One of 'custom_cnn', 'efficientnet_b0', or 'resnet50'.

    Returns:
        Preprocessed array or tensor formatted specifically for the target architecture.
    """
    import tensorflow as tf
    is_tf = isinstance(images, (tf.Tensor, tf.Variable))

    name = model_name.lower().replace("-", "_")

    if is_tf:
        images_float = tf.cast(images, tf.float32)
        if name == "custom_cnn":
            return images_float / 255.0
        elif name in ("efficientnet_b0", "efficientnet"):
            from keras.applications.efficientnet import preprocess_input as eff_prep
            return eff_prep(images_float)
        elif name in ("resnet50", "resnet"):
            from keras.applications.resnet50 import preprocess_input as res_prep
            return res_prep(images_float)
        else:
            raise ValueError(
                f"Unknown model_name '{model_name}'. "
                f"Expected one of: 'custom_cnn', 'efficientnet_b0', 'resnet50'."
            )
    else:
        images_float = np.asarray(images, dtype=np.float32)
        if name == "custom_cnn":
            # Custom CNN expects inputs normalized to [0.0, 1.0]
            return images_float / 255.0
        elif name in ("efficientnet_b0", "efficientnet"):
            # Keras EfficientNet applications expect RGB float inputs in [0, 255]
            # (Internal rescaling layer handles feature standardization)
            from keras.applications.efficientnet import preprocess_input as eff_prep
            return eff_prep(images_float)
        elif name in ("resnet50", "resnet"):
            # ResNet50 expects BGR zero-centered around ImageNet channel means
            from keras.applications.resnet50 import preprocess_input as res_prep
            return res_prep(images_float.copy())
        else:
            raise ValueError(
                f"Unknown model_name '{model_name}'. "
                f"Expected one of: 'custom_cnn', 'efficientnet_b0', 'resnet50'."
            )


def validate_output_tensor(
    output: np.ndarray,
    expected_batch_size: int,
    expected_classes: int = 7
) -> Tuple[bool, str]:
    """
    Validates that model output is a valid probability distribution.

    Returns:
        (is_valid, error_message)
    """
    if output.ndim != 2:
        return False, f"Expected 2D output tensor, got shape {output.shape}."

    if output.shape != (expected_batch_size, expected_classes):
        return False, f"Expected shape ({expected_batch_size}, {expected_classes}), got {output.shape}."

    if not np.all(np.isfinite(output)):
        return False, "Output contains NaN or infinite values."

    if np.any(output < 0.0) or np.any(output > 1.0 + 1e-5):
        return False, "Output values fall outside valid probability interval [0, 1]."

    sums = np.sum(output, axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        return False, f"Output probabilities do not sum to 1.0 (sums: {sums})."

    return True, "Valid probability distribution."
