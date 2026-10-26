"""
DermaStack-XAI EfficientNetB0 Transfer Learning Architecture.
Implements pre-trained EfficientNetB0 compound-scaled feature extractor
with a custom 7-class dermatoscopic lesion classification head.
"""

from typing import Tuple, Optional
import keras
from keras import layers
from keras.applications import EfficientNetB0
from src.config import DEFAULT_CONFIG


def build_efficientnet_b0(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    weights: Optional[str] = "imagenet",
    freeze_backbone: bool = True,
    dropout_rate: float = 0.4,
    seed: Optional[int] = 42,
    model_name: str = "efficientnet_b0"
) -> keras.Model:
    """
    Builds the EfficientNetB0 transfer learning model.

    Input Expectations:
        - RGB float array with values in [0, 255].
        - Built-in EfficientNet layers handle scaling.

    Grad-CAM Target Layer:
        - 'top_conv' (shape: (batch, 7, 7, 1280)) or 'top_activation'.

    Args:
        input_shape: Input image dimensions (H, W, C). Default (224, 224, 3).
        num_classes: Number of diagnostic categories. Default 7.
        weights: Pretrained weights to load ('imagenet' or None).
        freeze_backbone: Whether to freeze backbone weights during initial training.
        dropout_rate: Dropout fraction in classification head. Default 0.4.
        seed: Random seed for head initialization.
        model_name: Model identifier string.

    Returns:
        keras.Model instance.
    """
    kernel_initializer = keras.initializers.GlorotUniform(seed=seed) if seed is not None else "glorot_uniform"

    # Instantiate base model
    try:
        backbone = EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape
        )
    except Exception as e:
        if weights is not None:
            raise RuntimeError(
                f"Failed to load pretrained weights '{weights}' for EfficientNetB0: {e}. "
                "Ensure local weights are cached or set weights=None for uninitialized weights."
            )
        raise

    # Configure backbone trainability
    backbone.trainable = not freeze_backbone

    # Custom 7-Class Classification Head
    inputs = backbone.input
    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout_rate, seed=seed, name="head_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer=kernel_initializer,
        name="predictions"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


if __name__ == "__main__":
    model = build_efficientnet_b0(weights=None, freeze_backbone=True)
    model.summary()
    print("EfficientNetB0 model created successfully.")
