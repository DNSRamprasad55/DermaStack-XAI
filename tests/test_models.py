"""
Automated Test Suite for Base Deep Learning Architectures.
Verifies model construction, forward passes, probability contracts, parameter bounds,
trainability controls, Grad-CAM layer accessibility, and serialization fidelity.
"""

import gc
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import keras

from src.models import (
    build_custom_cnn,
    build_efficientnet_b0,
    build_resnet50,
    get_model_summary_dict,
    preprocess_for_model,
    validate_output_tensor
)
from src.config import DEFAULT_CONFIG


class TestModelArchitectures(unittest.TestCase):
    """Test suite covering the 3 base architectures under CPU constraints."""

    def setUp(self):
        """Set random seeds and prepare minimal dummy RGB tensor (batch_size=2)."""
        keras.utils.set_random_seed(42)
        # Small batch of 2 RGB images with values in [0, 255]
        self.batch_size = 2
        self.input_shape = (224, 224, 3)
        self.num_classes = 7
        self.dummy_images = np.random.uniform(0.0, 255.0, size=(self.batch_size, 224, 224, 3)).astype(np.float32)

    def tearDown(self):
        """Clean up Keras graph state and trigger garbage collection between tests."""
        keras.backend.clear_session()
        gc.collect()

    def test_custom_cnn_construction_and_forward_pass(self):
        """Verify Custom CNN builds, executes forward pass, and obeys probability contract."""
        model = build_custom_cnn(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            dropout_rate=0.4,
            seed=42
        )
        self.assertIsNotNone(model)

        # Preprocess dummy input
        x = preprocess_for_model(self.dummy_images, "custom_cnn")
        self.assertTrue(np.all(x >= 0.0) and np.all(x <= 1.0), "Custom CNN input must be in [0, 1].")

        # Forward pass on CPU
        preds = model(x, training=False).numpy()
        self.assertEqual(preds.shape, (self.batch_size, self.num_classes))

        # Validate probability contract
        valid, msg = validate_output_tensor(preds, self.batch_size, self.num_classes)
        self.assertTrue(valid, msg)

    def test_custom_cnn_parameter_budget(self):
        """Verify Custom CNN has fewer than 2,000,000 total parameters."""
        model = build_custom_cnn(input_shape=self.input_shape, num_classes=self.num_classes)
        summary = get_model_summary_dict(model)
        self.assertLess(
            summary["total_params"],
            2_000_000,
            f"Custom CNN exceeded parameter budget: {summary['total_params']:,} >= 2,000,000"
        )
        self.assertGreater(summary["trainable_params"], 0)

    def test_custom_cnn_gradcam_target_layer_accessible(self):
        """Verify that the target Grad-CAM convolutional layer exists and produces 4D feature map."""
        model = build_custom_cnn(input_shape=self.input_shape, num_classes=self.num_classes)
        target_layer_name = DEFAULT_CONFIG.xai.custom_cnn_layer  # "conv4_block"

        # Assert layer exists
        layer = model.get_layer(target_layer_name)
        self.assertIsNotNone(layer)

        # Sub-model extracting feature map to verify dimension
        feat_extractor = keras.Model(inputs=model.input, outputs=layer.output)
        x = preprocess_for_model(self.dummy_images, "custom_cnn")
        feat_map = feat_extractor(x, training=False).numpy()
        self.assertEqual(feat_map.ndim, 4, "Conv feature map must be 4D (batch, H, W, channels).")
        self.assertEqual(feat_map.shape[0], self.batch_size)

    def test_efficientnet_b0_construction_and_forward_pass(self):
        """Verify EfficientNetB0 builds (weights=None), forward pass works, and probabilities sum to 1."""
        model = build_efficientnet_b0(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            weights=None,  # Offline-safe unit test
            freeze_backbone=True,
            dropout_rate=0.4
        )
        self.assertIsNotNone(model)

        x = preprocess_for_model(self.dummy_images, "efficientnet_b0")
        preds = model(x, training=False).numpy()
        self.assertEqual(preds.shape, (self.batch_size, self.num_classes))

        valid, msg = validate_output_tensor(preds, self.batch_size, self.num_classes)
        self.assertTrue(valid, msg)

    def test_efficientnet_b0_backbone_freezing(self):
        """Verify freeze_backbone controls trainable parameters appropriately."""
        # Frozen
        model_frozen = build_efficientnet_b0(weights=None, freeze_backbone=True)
        summary_frozen = get_model_summary_dict(model_frozen)

        # Unfrozen
        keras.backend.clear_session()
        model_unfrozen = build_efficientnet_b0(weights=None, freeze_backbone=False)
        summary_unfrozen = get_model_summary_dict(model_unfrozen)

        self.assertGreater(
            summary_unfrozen["trainable_params"],
            summary_frozen["trainable_params"],
            "Unfrozen model must have strictly more trainable parameters than frozen model."
        )

    def test_efficientnet_b0_gradcam_target_layer_accessible(self):
        """Verify that EfficientNetB0 target Grad-CAM layer exists and produces 4D feature map."""
        model = build_efficientnet_b0(weights=None, freeze_backbone=True)
        target_layer_name = DEFAULT_CONFIG.xai.efficientnet_layer  # "top_conv"

        layer = model.get_layer(target_layer_name)
        self.assertIsNotNone(layer)

        feat_extractor = keras.Model(inputs=model.input, outputs=layer.output)
        x = preprocess_for_model(self.dummy_images, "efficientnet_b0")
        feat_map = feat_extractor(x, training=False).numpy()
        self.assertEqual(feat_map.ndim, 4)

    def test_resnet50_construction_and_forward_pass(self):
        """Verify ResNet50 builds (weights=None), forward pass works, and probabilities sum to 1."""
        model = build_resnet50(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            weights=None,  # Offline-safe unit test
            freeze_backbone=True,
            dropout_rate=0.5
        )
        self.assertIsNotNone(model)

        x = preprocess_for_model(self.dummy_images, "resnet50")
        preds = model(x, training=False).numpy()
        self.assertEqual(preds.shape, (self.batch_size, self.num_classes))

        valid, msg = validate_output_tensor(preds, self.batch_size, self.num_classes)
        self.assertTrue(valid, msg)

    def test_resnet50_backbone_freezing(self):
        """Verify freeze_backbone controls trainable parameters in ResNet50."""
        model_frozen = build_resnet50(weights=None, freeze_backbone=True)
        summary_frozen = get_model_summary_dict(model_frozen)

        keras.backend.clear_session()
        model_unfrozen = build_resnet50(weights=None, freeze_backbone=False)
        summary_unfrozen = get_model_summary_dict(model_unfrozen)

        self.assertGreater(
            summary_unfrozen["trainable_params"],
            summary_frozen["trainable_params"]
        )

    def test_resnet50_gradcam_target_layer_accessible(self):
        """Verify that ResNet50 target Grad-CAM layer exists and produces 4D feature map."""
        model = build_resnet50(weights=None, freeze_backbone=True)
        target_layer_name = DEFAULT_CONFIG.xai.resnet_layer  # "conv5_block3_out"

        layer = model.get_layer(target_layer_name)
        self.assertIsNotNone(layer)

        feat_extractor = keras.Model(inputs=model.input, outputs=layer.output)
        x = preprocess_for_model(self.dummy_images, "resnet50")
        feat_map = feat_extractor(x, training=False).numpy()
        self.assertEqual(feat_map.ndim, 4)

    def test_models_serialization_fidelity(self):
        """Verify all three models can be saved to .keras and loaded back with matching predictions."""
        models_to_test = [
            ("custom_cnn", build_custom_cnn(num_classes=self.num_classes)),
            ("efficientnet_b0", build_efficientnet_b0(weights=None, freeze_backbone=True)),
            ("resnet50", build_resnet50(weights=None, freeze_backbone=True))
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for name, model in models_to_test:
                x = preprocess_for_model(self.dummy_images, name)
                preds_before = model(x, training=False).numpy()

                save_path = Path(tmpdir) / f"{name}_test.keras"
                model.save(save_path)
                self.assertTrue(save_path.exists())

                loaded_model = keras.models.load_model(save_path)
                preds_after = loaded_model(x, training=False).numpy()

                np.testing.assert_allclose(
                    preds_before,
                    preds_after,
                    rtol=1e-4,
                    atol=1e-5,
                    err_msg=f"Serialization fidelity check failed for {name}!"
                )
                del model, loaded_model
                keras.backend.clear_session()
                gc.collect()

    def test_pretrained_imagenet_weights_loading_integration(self):
        """
        Explicit integration test verifying genuine ImageNet weights load when available locally.
        Runs gracefully without failing if weights are not cached or environment is offline.
        """
        # Test loading cached weights for EfficientNetB0
        try:
            model_eff = build_efficientnet_b0(weights="imagenet", freeze_backbone=True)
            self.assertIsNotNone(model_eff)
            del model_eff
        except RuntimeError as e:
            self.skipTest(f"Pretrained weights unavailable (offline/not cached): {e}")

        keras.backend.clear_session()
        gc.collect()

        # Test loading cached weights for ResNet50
        try:
            model_res = build_resnet50(weights="imagenet", freeze_backbone=True)
            self.assertIsNotNone(model_res)
            del model_res
        except RuntimeError as e:
            self.skipTest(f"Pretrained weights unavailable: {e}")


if __name__ == "__main__":
    unittest.main()
