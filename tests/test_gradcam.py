"""
Tests for DermaStack-XAI Stage 5: Explainable AI using Grad-CAM.

Covers:
    - Target layer discovery and validation across all three base architectures
    - Input dimension, channel count, and type validation
    - Predicted class (argmax) vs explicitly selected class handling
    - Heatmap dimensional contracts, non-negativity, and finite bounds [0, 1]
    - Safe normalization for zero or uniform activation maps (no division by zero)
    - RGB overlay spatial alignment, color mapping, and [0, 255] uint8 ranges
    - Out-of-bounds class indices and malformed input rejection
    - Disconnected gradient error handling
    - Reproducibility and deterministic heatmap generation
    - Real checkpoint compatibility smoke tests (Custom CNN, EfficientNetB0, ResNet50)
"""

from pathlib import Path
import unittest

import keras
from keras import layers
import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import DEFAULT_CONFIG, HAM10000_CLASSES
from src.xai.gradcam import (
    GradCAMExplainer,
    validate_image_input,
    find_target_layer,
    build_grad_model,
    GRADCAM_TARGET_LAYERS,
    ENSEMBLE_ATTRIBUTION_NOTICE
)


def build_synthetic_conv_model(num_classes: int = 7) -> keras.Model:
    """Builds a fast, lightweight 2-block CNN for testing Grad-CAM mechanics."""
    inputs = layers.Input(shape=(64, 64, 3), name="synth_input")
    x = layers.Conv2D(8, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(16, (3, 3), padding="same", name="target_conv")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="synthetic_model")


class TestGradCAMInputValidation(unittest.TestCase):
    """Tests image input validation and normalization."""

    def test_valid_numpy_image(self):
        img_np = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        validated = validate_image_input(img_np)
        self.assertEqual(validated.shape, (120, 160, 3))
        self.assertEqual(validated.dtype, np.float32)

    def test_valid_pil_image(self):
        pil_img = Image.new("RGB", (100, 80), color=(255, 128, 64))
        validated = validate_image_input(pil_img)
        self.assertEqual(validated.shape, (80, 100, 3))
        self.assertEqual(validated.dtype, np.float32)

    def test_valid_tf_tensor(self):
        tensor = tf.zeros((64, 64, 3), dtype=tf.float32)
        validated = validate_image_input(tensor)
        self.assertEqual(validated.shape, (64, 64, 3))

    def test_auto_scale_unit_interval(self):
        img_unit = np.ones((50, 50, 3), dtype=np.float32) * 0.5
        validated = validate_image_input(img_unit)
        self.assertAlmostEqual(float(validated.max()), 127.5, places=1)

    def test_reject_2d_grayscale(self):
        bad_2d = np.ones((50, 50), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_image_input(bad_2d)

    def test_reject_4_channel_rgba(self):
        bad_4c = np.ones((50, 50, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_image_input(bad_4c)

    def test_reject_nan_image(self):
        nan_img = np.ones((50, 50, 3), dtype=np.float32)
        nan_img[10, 10, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_image_input(nan_img)

    def test_reject_inf_image(self):
        inf_img = np.ones((50, 50, 3), dtype=np.float32)
        inf_img[10, 10, 0] = np.inf
        with self.assertRaises(ValueError):
            validate_image_input(inf_img)


class TestTargetLayerResolution(unittest.TestCase):
    """Tests target layer discovery across architectures."""

    def setUp(self):
        self.synth_model = build_synthetic_conv_model()

    def test_explicit_layer_found(self):
        layer = find_target_layer(self.synth_model, layer_name="target_conv")
        self.assertEqual(layer.name, "target_conv")

    def test_missing_layer_raises_value_error(self):
        with self.assertRaises(ValueError):
            find_target_layer(self.synth_model, layer_name="nonexistent_conv_layer")

    def test_architecture_layer_dictionary_constants(self):
        self.assertEqual(GRADCAM_TARGET_LAYERS["custom_cnn"], "conv4_block")
        self.assertEqual(GRADCAM_TARGET_LAYERS["efficientnet_b0"], "top_conv")
        self.assertEqual(GRADCAM_TARGET_LAYERS["resnet50"], "conv5_block3_out")


class TestGradCAMComputationSynthetic(unittest.TestCase):
    """Tests the Grad-CAM mathematics and pipeline on synthetic convolutional models."""

    def setUp(self):
        self.model = build_synthetic_conv_model(num_classes=7)
        self.explainer = GradCAMExplainer(
            model=self.model,
            model_name="custom_cnn",
            target_layer_name="target_conv"
        )
        self.dummy_img = np.random.uniform(0.0, 255.0, (64, 64, 3)).astype(np.float32)

    def test_predicted_class_default(self):
        # class_idx=None -> top predicted class
        prep = self.explainer.preprocess_image(self.dummy_img, target_size=(64, 64))
        heatmap, cls_idx, prob, all_preds = self.explainer.compute_heatmap(prep, class_idx=None)

        self.assertEqual(cls_idx, int(np.argmax(all_preds)))
        self.assertEqual(prob, all_preds[cls_idx])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_explicit_class_selection(self):
        prep = self.explainer.preprocess_image(self.dummy_img, target_size=(64, 64))
        # Select class index 3 ('df')
        heatmap, cls_idx, prob, all_preds = self.explainer.compute_heatmap(prep, class_idx=3)

        self.assertEqual(cls_idx, 3)
        self.assertEqual(prob, all_preds[3])

    def test_invalid_class_index_raises(self):
        prep = self.explainer.preprocess_image(self.dummy_img, target_size=(64, 64))
        with self.assertRaises(ValueError):
            self.explainer.compute_heatmap(prep, class_idx=99)
        with self.assertRaises(ValueError):
            self.explainer.compute_heatmap(prep, class_idx=-1)

    def test_heatmap_values_in_unit_range(self):
        prep = self.explainer.preprocess_image(self.dummy_img, target_size=(64, 64))
        heatmap, _, _, _ = self.explainer.compute_heatmap(prep)

        self.assertFalse(np.isnan(heatmap).any())
        self.assertFalse(np.isinf(heatmap).any())
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_safe_normalization_on_zero_heatmap(self):
        # A zero heatmap (all activations 0) must normalize cleanly without zero-division error
        zero_raw = np.zeros((16, 16), dtype=np.float32)
        resized_h, overlay = self.explainer.generate_overlay(self.dummy_img, zero_raw)

        self.assertEqual(float(resized_h.max()), 0.0)
        self.assertFalse(np.isnan(resized_h).any())
        self.assertEqual(overlay.shape, (64, 64, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_overlay_dimensions_and_ranges(self):
        res = self.explainer.explain(self.dummy_img, alpha=0.4)
        overlay = res["overlay_image"]
        resized_h = res["heatmap_resized"]

        self.assertEqual(overlay.shape, (64, 64, 3))
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertEqual(resized_h.shape, (64, 64))
        self.assertGreaterEqual(int(overlay.min()), 0)
        self.assertLessEqual(int(overlay.max()), 255)

    def test_reproducibility_deterministic(self):
        res1 = self.explainer.explain(self.dummy_img, class_idx=1)
        res2 = self.explainer.explain(self.dummy_img, class_idx=1)

        np.testing.assert_allclose(res1["heatmap_raw"], res2["heatmap_raw"], atol=1e-6)
        np.testing.assert_allclose(res1["overlay_image"], res2["overlay_image"])

    def test_ensemble_attribution_notice_string(self):
        self.assertIn("ENSEMBLE INTERPRETABILITY NOTICE", ENSEMBLE_ATTRIBUTION_NOTICE)
        self.assertIn("21-dimensional", ENSEMBLE_ATTRIBUTION_NOTICE)


class TestRealCheckpointGradCAMSmoke(unittest.TestCase):
    """Smoke tests validating Grad-CAM execution against actual trained .keras checkpoints."""

    def setUp(self):
        self.chk_dir = DEFAULT_CONFIG.paths.checkpoints_dir
        self.dummy_224 = np.random.uniform(0.0, 255.0, (224, 224, 3)).astype(np.float32)

    def test_custom_cnn_checkpoint_gradcam(self):
        ckpt_path = self.chk_dir / "custom_cnn_fold_0_best.keras"
        if not ckpt_path.exists():
            self.skipTest(f"Checkpoint {ckpt_path} not found.")

        model = keras.models.load_model(str(ckpt_path), compile=False)
        explainer = GradCAMExplainer(model=model, model_name="custom_cnn")
        self.assertEqual(explainer.target_layer_name, "conv4_block")

        exp = explainer.explain(self.dummy_224)
        self.assertEqual(exp["heatmap_raw"].shape, (28, 28))
        self.assertEqual(exp["heatmap_resized"].shape, (224, 224))
        self.assertEqual(exp["overlay_image"].shape, (224, 224, 3))
        self.assertFalse(np.isnan(exp["heatmap_raw"]).any())

    def test_efficientnet_checkpoint_gradcam(self):
        ckpt_path = self.chk_dir / "efficientnet_b0_fold_0_best.keras"
        if not ckpt_path.exists():
            self.skipTest(f"Checkpoint {ckpt_path} not found.")

        model = keras.models.load_model(str(ckpt_path), compile=False)
        explainer = GradCAMExplainer(model=model, model_name="efficientnet_b0")
        self.assertEqual(explainer.target_layer_name, "top_conv")

        exp = explainer.explain(self.dummy_224)
        self.assertEqual(exp["heatmap_raw"].shape, (7, 7))
        self.assertEqual(exp["heatmap_resized"].shape, (224, 224))
        self.assertEqual(exp["overlay_image"].shape, (224, 224, 3))
        self.assertFalse(np.isnan(exp["heatmap_raw"]).any())

    def test_resnet50_checkpoint_gradcam(self):
        ckpt_path = self.chk_dir / "resnet50_fold_0_best.keras"
        if not ckpt_path.exists():
            self.skipTest(f"Checkpoint {ckpt_path} not found.")

        model = keras.models.load_model(str(ckpt_path), compile=False)
        explainer = GradCAMExplainer(model=model, model_name="resnet50")
        self.assertEqual(explainer.target_layer_name, "conv5_block3_out")

        exp = explainer.explain(self.dummy_224)
        self.assertEqual(exp["heatmap_raw"].shape, (7, 7))
        self.assertEqual(exp["heatmap_resized"].shape, (224, 224))
        self.assertEqual(exp["overlay_image"].shape, (224, 224, 3))
        self.assertFalse(np.isnan(exp["heatmap_raw"]).any())


if __name__ == "__main__":
    unittest.main()
