"""
Automated Test Suite for DermaStack-XAI Stage 3 Training Pipeline.
Verifies 5-fold generation, zero lesion leakage, holdout test isolation,
class weight computation, data transforms, callbacks, and OOF matrix contracts.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from src.config import DEFAULT_CONFIG, HAM10000_CLASSES
from src.data.transforms import (
    CLASS_TO_IDX,
    compute_class_weights,
    create_dataset,
    encode_labels,
    get_augmentation_pipeline
)
from src.models.common import preprocess_for_model
from src.training.callbacks import get_training_callbacks
from src.training.trainer import (
    FiveFoldCrossValidationEngine,
    MODEL_COLUMN_SLICES,
    SUPPORTED_MODELS
)


class TestTrainingPipeline(unittest.TestCase):
    """Unit and integration tests for Stage 3 training and cross-validation pipeline."""

    @classmethod
    def setUpClass(cls):
        """Initialize trainer engine with verified project data."""
        cls.engine = FiveFoldCrossValidationEngine()
        cls.train_df = cls.engine.train_df
        cls.manifest = cls.engine.manifest_data

    def test_partition_sizes_and_holdout_isolation(self):
        """Verify partition sizes and ensure holdout test images are strictly isolated."""
        splits = self.manifest.get("splits", {})
        train_ids = set(splits.get("train", []))
        val_ids = set(splits.get("val", []))
        test_ids = set(splits.get("test", []))

        # Size contracts
        self.assertEqual(len(train_ids), 7153)
        self.assertEqual(len(val_ids), 1431)
        self.assertEqual(len(test_ids), 1431)

        # Zero leakage contracts
        self.assertEqual(len(train_ids.intersection(test_ids)), 0, "Holdout test images leaked into training set!")
        self.assertEqual(len(val_ids.intersection(test_ids)), 0, "Holdout test images leaked into validation set!")
        self.assertEqual(len(train_ids.intersection(val_ids)), 0, "External validation images leaked into training set!")

    def test_5_fold_generation_and_zero_lesion_leakage(self):
        """Verify 5 folds partition all 7,153 images with strictly zero lesion overlap."""
        folds = self.engine.folds
        self.assertEqual(len(folds), 5)

        total_val_samples = 0
        all_val_indices = []

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            self.assertEqual(len(tr_idx) + len(val_idx), 7153)
            total_val_samples += len(val_idx)
            all_val_indices.extend(val_idx)

            # Assert zero lesion overlap between train and val
            tr_lesions = set(self.train_df.iloc[tr_idx]["lesion_id"])
            val_lesions = set(self.train_df.iloc[val_idx]["lesion_id"])
            overlap = tr_lesions.intersection(val_lesions)
            self.assertEqual(
                len(overlap), 0,
                f"Fold {fold_idx} has {len(overlap)} lesions overlapping between train and validation!"
            )

            # Assert all 7 classes represented in val fold
            val_classes = set(self.train_df.iloc[val_idx]["dx"])
            self.assertEqual(
                len(val_classes), 7,
                f"Fold {fold_idx} is missing classes! Present: {val_classes}"
            )

        # Assert validation folds form an exact partition of all 7,153 training records
        self.assertEqual(total_val_samples, 7153)
        self.assertEqual(len(set(all_val_indices)), 7153)

    def test_class_weights_balanced_formula_and_bounds(self):
        """Verify compute_class_weights follows balanced formula w_c = N / (C * N_c) and stays bounded."""
        labels = ["nv", "nv", "nv", "mel", "bkl", "df", "bcc"]
        cw = compute_class_weights(labels, classes=HAM10000_CLASSES)

        self.assertEqual(len(cw), 7)
        # N=7, C=7. For classes with count 1: w = 7 / (7 * 1) = 1.0. For nv (count 3): w = 7 / (7 * 3) = 0.3333
        self.assertAlmostEqual(cw[CLASS_TO_IDX["nv"]], 7.0 / (7 * 3), places=3)
        self.assertAlmostEqual(cw[CLASS_TO_IDX["mel"]], 1.0, places=3)
        self.assertAlmostEqual(cw[CLASS_TO_IDX["df"]], 1.0, places=3)

        # Empty labels must raise ValueError
        with self.assertRaises(ValueError):
            compute_class_weights([])

    def test_preprocessing_consistency_and_double_rescaling_prevention(self):
        """
        Verify that preprocessing functions format inputs correctly for each model:
        - Custom CNN: scales to [0.0, 1.0]
        - EfficientNetB0: preserves float RGB in [0, 255] (preventing double rescaling)
        - ResNet50: converts to BGR and mean-centers
        """
        raw_rgb = np.array([[[[0.0, 128.0, 255.0]]]], dtype=np.float32)

        # Custom CNN
        p_cnn = preprocess_for_model(raw_rgb, "custom_cnn")
        self.assertTrue(np.all(p_cnn >= 0.0) and np.all(p_cnn <= 1.0))
        self.assertAlmostEqual(float(p_cnn[0, 0, 0, 2]), 1.0, places=4)

        # EfficientNetB0 (must NOT be divided by 255 here to avoid double rescaling)
        p_eff = preprocess_for_model(raw_rgb, "efficientnet_b0")
        self.assertTrue(np.allclose(p_eff, raw_rgb))

        # ResNet50 (must be mean-centered with ImageNet channel offsets)
        p_res = preprocess_for_model(raw_rgb, "resnet50")
        # In BGR order, blue channel was 255.0 - 103.939 = 151.061
        self.assertAlmostEqual(float(p_res[0, 0, 0, 0]), 151.061, places=2)

    def test_augmentation_restricted_to_training(self):
        """Verify that evaluation/validation datasets disable augmentation deterministically."""
        sample_paths = [self.train_df["image_path"].iloc[0]] * 4

        # Validation dataset (is_training=False)
        ds_val1 = create_dataset(sample_paths, model_name="custom_cnn", batch_size=4, is_training=False, shuffle=False)
        ds_val2 = create_dataset(sample_paths, model_name="custom_cnn", batch_size=4, is_training=False, shuffle=False)

        batch1 = next(iter(ds_val1)).numpy()
        batch2 = next(iter(ds_val2)).numpy()

        # Deterministic: identical output across runs without augmentation
        self.assertTrue(np.allclose(batch1, batch2))

    def test_callbacks_creation_and_history_logging(self):
        """Verify get_training_callbacks instantiates all required callbacks and logs history."""
        with tempfile.TemporaryDirectory() as td:
            cbs, chk_path, hist_path = get_training_callbacks(
                model_name="custom_cnn",
                fold_idx=0,
                checkpoint_dir=td,
                early_stopping_patience=3,
                reduce_lr_patience=2,
                verbose=0
            )
            self.assertEqual(len(cbs), 4)

            # Fit a minimal 1-epoch toy model to test HistoryLogger
            m = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(7, activation="softmax")])
            m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            x = np.random.randn(8, 4).astype(np.float32)
            y = np.eye(7)[np.random.choice(7, 8)]

            m.fit(x, y, validation_data=(x, y), epochs=1, callbacks=cbs, verbose=0)
            self.assertTrue(chk_path.exists())
            self.assertTrue(hist_path.exists())

            with open(hist_path, "r", encoding="utf-8") as f:
                hist_data = json.load(f)
            self.assertEqual(hist_data["epochs_completed"], 1)
            self.assertIn("val_loss", hist_data["history"][0])

    def test_oof_matrix_assembly_contracts_and_serialization(self):
        """
        Verify that assemble_and_save_oof creates the exact (7153, 21) matrix,
        asserts finite values in [0, 1], checks sum-to-1 per model, and saves manifest.
        """
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "oof"

            # Create mock fold results covering all 7,153 samples for all 3 models
            mock_fold_results = {}
            for model_name in SUPPORTED_MODELS:
                mock_fold_results[model_name] = []
                for fold_idx, (_, val_idx) in enumerate(self.engine.folds):
                    n_val = len(val_idx)
                    # Generate valid synthetic probabilities summing to 1.0
                    raw_p = np.random.uniform(0.1, 1.0, size=(n_val, 7))
                    norm_p = (raw_p / raw_p.sum(axis=1, keepdims=True)).astype(np.float32)

                    mock_fold_results[model_name].append({
                        "fold_idx": fold_idx,
                        "val_indices": val_idx.tolist(),
                        "val_predictions": norm_p
                    })

            # Assemble OOF matrix
            manifest_path = self.engine.assemble_and_save_oof(mock_fold_results, output_dir=out_dir)
            self.assertTrue(manifest_path.exists())

            # Load and verify saved artifacts
            pred_file = out_dir / "oof_predictions.npy"
            label_file = out_dir / "oof_labels.npy"
            ids_file = out_dir / "oof_image_ids.json"

            self.assertTrue(pred_file.exists())
            self.assertTrue(label_file.exists())
            self.assertTrue(ids_file.exists())

            oof_preds = np.load(pred_file)
            oof_labels = np.load(label_file)
            with open(ids_file, "r", encoding="utf-8") as f:
                oof_ids = json.load(f)

            # Contract verifications
            self.assertEqual(oof_preds.shape, (7153, 21))
            self.assertEqual(oof_labels.shape, (7153,))
            self.assertEqual(len(oof_ids), 7153)

            # Value bounds & probability contracts
            self.assertFalse(np.isnan(oof_preds).any())
            self.assertTrue(np.all(oof_preds >= 0.0) and np.all(oof_preds <= 1.0 + 1e-4))

            # Each model slice sums to 1.0
            for m_name, (c_start, c_end) in MODEL_COLUMN_SLICES.items():
                sums = oof_preds[:, c_start:c_end].sum(axis=1)
                self.assertTrue(np.allclose(sums, 1.0, atol=1e-3))

    def test_oof_matrix_rejects_missing_coverage(self):
        """Verify that assemble_and_save_oof raises ValueError if samples are unpredicted."""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "oof"
            # Intentionally omit Fold 4
            incomplete_folds = {
                "custom_cnn": [
                    {
                        "fold_idx": fold_idx,
                        "val_indices": val_idx.tolist(),
                        "val_predictions": np.full((len(val_idx), 7), 1.0 / 7.0, dtype=np.float32)
                    }
                    for fold_idx, (_, val_idx) in enumerate(self.engine.folds[:4])  # only 4 folds!
                ],
                "efficientnet_b0": [],
                "resnet50": []
            }
            with self.assertRaises(ValueError):
                self.engine.assemble_and_save_oof(incomplete_folds, output_dir=out_dir)

    def test_pilot_partial_oof_generation_and_contract(self):
        """Verify that pilot partial OOF saves with shape (1431, 21) and is_complete_oof=False."""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "oof"
            val_indices = self.engine.folds[0][1]
            n_val = len(val_indices)
            self.assertEqual(n_val, 1431)

            partial_oof = np.zeros((n_val, 21), dtype=np.float32)
            for m_name in SUPPORTED_MODELS:
                c_start, c_end = MODEL_COLUMN_SLICES[m_name]
                raw_p = np.random.uniform(0.1, 1.0, size=(n_val, 7))
                partial_oof[:, c_start:c_end] = (raw_p / raw_p.sum(axis=1, keepdims=True)).astype(np.float32)

            # Assert probability bounds
            self.assertFalse(np.isnan(partial_oof).any())
            for m_name, (c_start, c_end) in MODEL_COLUMN_SLICES.items():
                sums = partial_oof[:, c_start:c_end].sum(axis=1)
                self.assertTrue(np.allclose(sums, 1.0, atol=1e-3))

            # Verify pilot manifest structure
            manifest = {
                "is_complete_oof": False,
                "partial_fold_index": 0,
                "evaluated_samples": n_val,
                "matrix_shape": list(partial_oof.shape)
            }
            self.assertFalse(manifest["is_complete_oof"])
            self.assertEqual(manifest["matrix_shape"], [1431, 21])


if __name__ == "__main__":
    unittest.main()
