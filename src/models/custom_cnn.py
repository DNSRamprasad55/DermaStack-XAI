"""
DermaStack-XAI Custom Lightweight CNN Architecture.
A parameter-efficient 4-block convolutional network designed specifically
for 7-class dermatoscopic lesion classification and Grad-CAM interpretability.
"""

from typing import Tuple, Optional
import keras
from keras import layers
from src.config import DEFAULT_CONFIG


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    dropout_rate: float = 0.4,
    seed: Optional[int] = 42,
    model_name: str = "custom_cnn"
) -> keras.Model:
    """
    Builds the Custom Lightweight CNN model.

    Architecture:
        - 4 Convolutional Blocks: [Conv2D -> BatchNorm -> ReLU -> MaxPool2D]
        - Block 4 Conv layer named 'conv4_block' for Grad-CAM access.
        - Global Average Pooling -> Dense(128) -> BatchNorm -> Dropout -> Softmax(num_classes).
        - Total parameters: < 500,000 (well within the < 2,000,000 budget).

    Args:
        input_shape: Input image dimensions (H, W, C). Default (224, 224, 3).
        num_classes: Number of diagnostic categories. Default 7.
        dropout_rate: Dropout fraction in classification head. Default 0.4.
        seed: Random seed for weight initialization.
        model_name: Name identifier for the Keras model.

    Returns:
        keras.Model instance.
    """
    kernel_initializer = keras.initializers.GlorotUniform(seed=seed) if seed is not None else "glorot_uniform"

    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer=kernel_initializer, name="conv1_conv")(inputs)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.ReLU(name="conv1_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="conv1_pool")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=kernel_initializer, name="conv2_conv")(x)
    x = layers.BatchNormalization(name="conv2_bn")(x)
    x = layers.ReLU(name="conv2_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="conv2_pool")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer=kernel_initializer, name="conv3_conv")(x)
    x = layers.BatchNormalization(name="conv3_bn")(x)
    x = layers.ReLU(name="conv3_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="conv3_pool")(x)

    # Block 4 (Final conv layer explicitly named 'conv4_block' for Grad-CAM target)
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer=kernel_initializer, name="conv4_block")(x)
    x = layers.BatchNormalization(name="conv4_bn")(x)
    x = layers.ReLU(name="conv4_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="conv4_pool")(x)

    # Classification Head (Parameter-efficient Global Pooling)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", kernel_initializer=kernel_initializer, name="head_dense")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout_rate, seed=seed, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_initializer=kernel_initializer, name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


if __name__ == "__main__":
    model = build_custom_cnn()
    model.summary()
    print(f"Total Parameters: {model.count_params():,}")
    assert model.count_params() < 2_000_000, "Parameter budget exceeded!"
