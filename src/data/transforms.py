"""
DermaStack-XAI Data Transforms and Streaming Pipeline.
Handles image loading, label encoding, class weight computation,
medically conservative data augmentation, and model-specific preprocessing.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers

from src.config import DEFAULT_CONFIG, HAM10000_CLASSES
from src.models.common import preprocess_for_model
from src.utils.logger import get_logger

logger = get_logger("DermaStack-XAI.Transforms")

# Strict, reproducible class index mappings
CLASS_TO_IDX: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(HAM10000_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls_name for i, cls_name in enumerate(HAM10000_CLASSES)}


def encode_labels(
    labels: Union[List[str], pd.Series],
    classes: Optional[List[str]] = None
) -> np.ndarray:
    """
    Encodes diagnostic category strings into one-hot binary matrices of shape (N, num_classes).

    Args:
        labels: Iterable of diagnosis strings (e.g. ['nv', 'mel', ...]).
        classes: Ordered list of allowed class names. Defaults to HAM10000_CLASSES.

    Returns:
        One-hot float32 ndarray of shape (N, num_classes).
    """
    class_list = classes or HAM10000_CLASSES
    mapping = {c: i for i, c in enumerate(class_list)}
    num_classes = len(class_list)

    indices = []
    for lbl in labels:
        if lbl not in mapping:
            raise ValueError(f"Unrecognized class '{lbl}'. Allowed classes: {class_list}")
        indices.append(mapping[lbl])

    indices_arr = np.array(indices, dtype=np.int32)
    one_hot = np.eye(num_classes, dtype=np.float32)[indices_arr]
    return one_hot


def compute_class_weights(
    train_labels: Union[List[str], pd.Series],
    classes: Optional[List[str]] = None
) -> Dict[int, float]:
    """
    Computes balanced class weights strictly from the provided training partition labels.

    Formula:
        w_c = N / (C * N_c)
        where:
            N = total number of training samples
            C = number of classes (7)
            N_c = count of training samples belonging to class c

    Args:
        train_labels: Training subset diagnosis labels. Never include validation or test labels.
        classes: Ordered list of class names. Defaults to HAM10000_CLASSES.

    Returns:
        Dictionary mapping integer class index (0..6) to float weight.
    """
    class_list = classes or HAM10000_CLASSES
    mapping = {c: i for i, c in enumerate(class_list)}
    num_classes = len(class_list)

    counts: Dict[str, int] = {c: 0 for c in class_list}
    for lbl in train_labels:
        if lbl not in mapping:
            raise ValueError(f"Unrecognized class '{lbl}'. Allowed: {class_list}")
        counts[lbl] += 1

    total_samples = len(train_labels)
    if total_samples == 0:
        raise ValueError("Cannot compute class weights from an empty training partition.")

    class_weights: Dict[int, float] = {}
    for c in class_list:
        idx = mapping[c]
        c_count = counts[c]
        if c_count == 0:
            logger.warning(f"Class '{c}' has 0 training samples in this fold! Assigning default weight 1.0.")
            class_weights[idx] = 1.0
        else:
            weight = float(total_samples / (num_classes * c_count))
            class_weights[idx] = round(weight, 4)

    return class_weights


def get_augmentation_pipeline(
    seed: Optional[int] = 42
) -> keras.Sequential:
    """
    Constructs a medically conservative, label-preserving data augmentation sequence.
    Dermatoscopy lesions exhibit rotational symmetry and arbitrary orientations,
    justifying horizontal/vertical flips and modest rotations/zooms.

    Layers:
        - RandomFlip('horizontal_and_vertical'): 50% chance horizontal, 50% vertical.
        - RandomRotation(0.05): Modest rotation up to +/- 18 degrees (0.05 * 2*pi).
        - RandomZoom((-0.1, 0.1)): Modest zoom in/out by at most 10%.

    Args:
        seed: Random seed for deterministic transformation initialization.

    Returns:
        Keras Sequential pipeline operating on raw RGB float images in [0, 255].
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed, name="aug_random_flip"),
        layers.RandomRotation(0.05, fill_mode="reflect", seed=seed, name="aug_random_rotation"),
        layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="reflect", seed=seed, name="aug_random_zoom")
    ], name="dermatoscopy_augmentation")


def create_dataset(
    image_paths: List[str],
    labels: Optional[np.ndarray] = None,
    model_name: str = "custom_cnn",
    batch_size: int = 32,
    is_training: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    target_size: Tuple[int, int] = (224, 224)
) -> tf.data.Dataset:
    """
    Builds a high-performance tf.data.Dataset streaming images from disk.

    Processing Flow:
        1. Read raw JPEG bytes from disk using tf.io.read_file.
        2. Decode into 3-channel RGB uint8 tensor using tf.image.decode_jpeg.
        3. Deterministically resize to target_size (224, 224), outputting float32 in [0, 255].
        4. If is_training=True: apply label-preserving augmentation (flips, rotation, zoom).
        5. Apply model-specific preprocessing (Custom CNN: /255, EffNet: identity float, ResNet: BGR mean-centered).
        6. Batch and prefetch with tf.data.AUTOTUNE for memory-safe CPU streaming.

    Args:
        image_paths: List of absolute file paths to JPEG images.
        labels: Optional one-hot label array of shape (N, 7). If None, dataset yields features only.
        model_name: Target architecture ('custom_cnn', 'efficientnet_b0', 'resnet50').
        batch_size: Batch size for training or inference.
        is_training: Whether this dataset is for training (enables data augmentation).
        shuffle: Whether to shuffle sample order before batching.
        seed: Random seed for shuffling and augmentations.
        target_size: Spatial dimensions (H, W). Default (224, 224).

    Returns:
        tf.data.Dataset yielding (batch_images, batch_labels) or (batch_images,).
    """
    num_samples = len(image_paths)
    if num_samples == 0:
        raise ValueError("Cannot create dataset from an empty list of image paths.")

    aug_pipeline = get_augmentation_pipeline(seed=seed) if is_training else None

    def _load_raw_image(path_tensor: tf.Tensor) -> tf.Tensor:
        raw_bytes = tf.io.read_file(path_tensor)
        img = tf.image.decode_jpeg(raw_bytes, channels=3)
        img = tf.image.resize(img, target_size, method=tf.image.ResizeMethod.BILINEAR)
        return img

    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    if labels is not None:
        if len(labels) != num_samples:
            raise ValueError(f"Length mismatch: {num_samples} image paths vs {len(labels)} labels.")
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((paths_ds, labels_ds))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(num_samples, 2048), seed=seed, reshuffle_each_iteration=True)

        # 1. Load & resize raw images in parallel
        dataset = dataset.map(
            lambda p, l: (_load_raw_image(p), l),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        # 2. Batch images
        dataset = dataset.batch(batch_size)

        # 3. Vectorized batch augmentation (training only)
        if is_training and aug_pipeline is not None:
            dataset = dataset.map(
                lambda bx, by: (aug_pipeline(bx, training=True), by),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # 4. Model-specific batch preprocessing
        dataset = dataset.map(
            lambda bx, by: (preprocess_for_model(bx, model_name), by),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = paths_ds
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(num_samples, 2048), seed=seed, reshuffle_each_iteration=True)

        dataset = dataset.map(
            _load_raw_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size)

        if is_training and aug_pipeline is not None:
            dataset = dataset.map(
                lambda bx: aug_pipeline(bx, training=True),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.map(
            lambda bx: preprocess_for_model(bx, model_name),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
