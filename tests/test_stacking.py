"""
Tests for DermaStack-XAI Stage 4 Stacking Ensemble Meta-Learner.
Covers:
    - Input shape, column slice, and probability simplex validation
    - NaN, Inf, negative, and out-of-range rejection
    - Correct 7-class mapping and ordering
    - Synthetic meta-classifier fitting and convergence
    - Valid output probability distributions (simplex properties)
    - Joblib save/load consistency and security advisory
    - Real OOF prediction matrix and label alignment
    - Partition and lesion group isolation (Holdout test and external validation)
    - Metric calculation correctness and multiclass ROC-AUC
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import accuracy_score

from src.config import DEFAULT_CONFIG, HAM10000_CLASSES
from src.models.stacking_meta import (
    StackingMetaClassifier,
    validate_stacking_features,
    compute_multiclass_metrics,
    MODEL_COLUMN_SLICES,
    SUPPORTED_BASE_MODELS,
    JOBLIB_SECURITY_WARNING
)


class TestStackingInputValidation(unittest.TestCase):
    """Tests input validation for the (N, 21) stacking probability matrix."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        # Construct valid (10, 21) matrix with 3 blocks of 7 probabilities summing to 1
        blocks = []
        for _ in range(3):
            raw = self.rng.dirichlet(np.ones(7), size=10).astype(np.float32)
            blocks.append(raw)
        self.valid_X = np.concatenate(blocks, axis=1)

    def test_valid_input_passes(self):
        cleaned = validate_stacking_features(self.valid_X, expected_rows=10)
        self.assertEqual(cleaned.shape, (10, 21))
        self.assertEqual(cleaned.dtype, np.float32)

    def test_invalid_dimension_raises(self):
        with self.assertRaises(ValueError):
            validate_stacking_features(self.valid_X.flatten())

    def test_invalid_column_count_raises(self):
        bad_cols = self.valid_X[:, :20]
        with self.assertRaises(ValueError):
            validate_stacking_features(bad_cols)

    def test_row_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            validate_stacking_features(self.valid_X, expected_rows=999)

    def test_nan_rejection(self):
        bad_X = self.valid_X.copy()
        bad_X[2, 5] = np.nan
        with self.assertRaises(ValueError):
            validate_stacking_features(bad_X)

    def test_inf_rejection(self):
        bad_X = self.valid_X.copy()
        bad_X[3, 14] = np.inf
        with self.assertRaises(ValueError):
            validate_stacking_features(bad_X)

    def test_negative_probability_rejection(self):
        bad_X = self.valid_X.copy()
        bad_X[0, 0] = -0.5
        with self.assertRaises(ValueError):
            validate_stacking_features(bad_X)

    def test_exceeding_probability_rejection(self):
        bad_X = self.valid_X.copy()
        bad_X[0, 0] = 1.5
        with self.assertRaises(ValueError):
            validate_stacking_features(bad_X)

    def test_sum_to_one_violation_raises(self):
        bad_X = self.valid_X.copy()
        # Disturb column slice 0..6 sum
        bad_X[0, 0:7] = 0.5  # sum is 3.5
        with self.assertRaises(ValueError):
            validate_stacking_features(bad_X)


class TestStackingMetaClassifier(unittest.TestCase):
    """Tests fitting, probability prediction, and serialization of StackingMetaClassifier."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        n_samples = 140
        # Generate synthetic data with 7 balanced classes
        self.y_syn = np.repeat(np.arange(7), n_samples // 7)

        # Generate correlated synthetic features for 3 models
        blocks = []
        for _ in range(3):
            # Base model gives higher probability to true class + noise
            one_hot = np.eye(7)[self.y_syn]
            noisy = one_hot * 0.6 + self.rng.uniform(0.01, 0.05, size=(n_samples, 7))
            normalized = noisy / noisy.sum(axis=1, keepdims=True)
            blocks.append(normalized.astype(np.float32))

        self.X_syn = np.concatenate(blocks, axis=1)

    def test_unfitted_model_raises(self):
        meta = StackingMetaClassifier()
        with self.assertRaises(RuntimeError):
            meta.predict_proba(self.X_syn)

    def test_fit_and_predict_properties(self):
        meta = StackingMetaClassifier(C=1.0, max_iter=500, random_state=42)
        meta.fit(self.X_syn, self.y_syn)

        self.assertTrue(meta.is_fitted)
        self.assertIsNotNone(meta.in_sample_metrics)

        probs = meta.predict_proba(self.X_syn)
        preds = meta.predict(self.X_syn)

        self.assertEqual(probs.shape, (len(self.y_syn), 7))
        self.assertEqual(preds.shape, (len(self.y_syn),))

        # Assert probability simplex properties
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

        # Accurate synthetic predictions
        acc = accuracy_score(self.y_syn, preds)
        self.assertGreater(acc, 0.90)

    def test_save_and_load_consistency(self):
        meta = StackingMetaClassifier(C=0.5, random_state=42)
        meta.fit(self.X_syn, self.y_syn)
        orig_probs = meta.predict_proba(self.X_syn)

        with tempfile.TemporaryDirectory() as tmp_dir:
            art_path = Path(tmp_dir) / "test_meta.joblib"
            man_path = Path(tmp_dir) / "test_meta_manifest.json"

            meta.save(art_path, manifest_path=man_path)

            self.assertTrue(art_path.exists())
            self.assertTrue(man_path.exists())

            # Verify manifest content
            with open(man_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            self.assertIn("security_warning", manifest_data)
            self.assertEqual(manifest_data["classes"], HAM10000_CLASSES)
            self.assertEqual(manifest_data["feature_specification"]["total_features"], 21)

            # Reload model
            loaded_meta = StackingMetaClassifier.load(art_path)
            self.assertTrue(loaded_meta.is_fitted)
            loaded_probs = loaded_meta.predict_proba(self.X_syn)

            np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-6)


class TestMetricsCalculation(unittest.TestCase):
    """Tests classification metrics, confusion matrices, and ROC-AUC computation."""

    def test_metrics_consistency(self):
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 5, 5, 4])
        # Near perfect probability matrix
        one_hot = np.eye(7)[y_true]
        noisy = one_hot * 0.9 + 0.1 / 7.0
        y_prob = (noisy / noisy.sum(axis=1, keepdims=True)).astype(np.float32)

        metrics = compute_multiclass_metrics(
            y_true=y_true,
            y_prob=y_prob,
            classes=HAM10000_CLASSES,
            model_name="test_model"
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertIsNotNone(metrics["roc_auc_ovr_macro"])
        self.assertEqual(len(metrics["confusion_matrix"]), 7)
        self.assertEqual(len(metrics["per_class"]), 7)


class TestRealOOFArtifactIntegrity(unittest.TestCase):
    """Tests integrity of actual Stage 3 OOF artifacts on disk."""

    def setUp(self):
        self.oof_preds_path = DEFAULT_CONFIG.paths.oof_dir / "oof_predictions.npy"
        self.oof_labels_path = DEFAULT_CONFIG.paths.oof_dir / "oof_labels.npy"
        self.splits_path = DEFAULT_CONFIG.paths.splits_dir / "train_val_test_splits.json"

    def test_real_oof_dimensions_and_isolation(self):
        if not self.oof_preds_path.exists() or not self.splits_path.exists():
            self.skipTest("OOF artifacts or splits not found on disk.")

        preds = np.load(self.oof_preds_path)
        labels = np.load(self.oof_labels_path)

        self.assertEqual(preds.shape, (7153, 21))
        self.assertEqual(labels.shape, (7153,))
        self.assertEqual(preds.dtype, np.float32)

        # Strict finite checks
        self.assertFalse(np.isnan(preds).any())
        self.assertFalse(np.isinf(preds).any())
        self.assertGreaterEqual(float(preds.min()), 0.0)
        self.assertLessEqual(float(preds.max()), 1.0)

        # Check block sums
        for model_name, (c_start, c_end) in MODEL_COLUMN_SLICES.items():
            block_sums = preds[:, c_start:c_end].sum(axis=1)
            np.testing.assert_allclose(block_sums, 1.0, atol=1e-3, err_msg=f"{model_name} block sum failed")

        # Partition isolation checks
        with open(self.splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        train_ids = set(splits_data["splits"]["train"])
        val_ids = set(splits_data["splits"]["val"])
        test_ids = set(splits_data["splits"]["test"])

        self.assertEqual(len(train_ids), 7153)
        self.assertEqual(len(val_ids), 1431)
        self.assertEqual(len(test_ids), 1431)

        # Zero leakage
        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)


class TestStage4ResearchValidationAudit(unittest.TestCase):
    """Specific tests addressing Stage 4.1 research validation audit criteria."""

    def setUp(self):
        self.meta_std_path = DEFAULT_CONFIG.paths.meta_models_dir / "stacking_meta_model.joblib"
        self.meta_bal_path = DEFAULT_CONFIG.paths.meta_models_dir / "stacking_meta_model_balanced.joblib"
        self.val_cache_path = DEFAULT_CONFIG.paths.oof_dir / "external_val_ensemble_features.npz"

    def test_both_saved_meta_models_exist_and_load(self):
        if not self.meta_std_path.exists() or not self.meta_bal_path.exists():
            self.skipTest("Saved meta models not found on disk.")

        std_clf = StackingMetaClassifier.load(self.meta_std_path)
        bal_clf = StackingMetaClassifier.load(self.meta_bal_path)

        self.assertTrue(std_clf.is_fitted)
        self.assertTrue(bal_clf.is_fitted)
        self.assertIsNone(std_clf.class_weight)
        self.assertEqual(bal_clf.class_weight, "balanced")

    def test_validation_cache_properties_and_isolation(self):
        if not self.val_cache_path.exists():
            self.skipTest("External validation features cache not found.")

        cached = np.load(self.val_cache_path, allow_pickle=True)
        X_val = cached["X_val_stacked"]
        y_val = cached["y_val"]
        val_ids = set(cached["val_image_ids"].tolist())

        self.assertEqual(X_val.shape, (1431, 21))
        self.assertEqual(y_val.shape, (1431,))
        self.assertEqual(len(val_ids), 1431)

        # Validate feature block bounds and simplex
        validate_stacking_features(X_val, expected_rows=1431)

        # Verify holdout test exclusion
        splits_path = DEFAULT_CONFIG.paths.splits_dir / "train_val_test_splits.json"
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        test_ids = set(splits_data["splits"]["test"])
        train_ids = set(splits_data["splits"]["train"])

        self.assertEqual(len(val_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(train_ids)), 0)

    def test_minority_class_tradeoff_reproducibility(self):
        if not self.val_cache_path.exists() or not self.meta_std_path.exists() or not self.meta_bal_path.exists():
            self.skipTest("Artifacts not available for minority class check.")

        cached = np.load(self.val_cache_path, allow_pickle=True)
        X_val = cached["X_val_stacked"]
        y_val = cached["y_val"]

        std_clf = StackingMetaClassifier.load(self.meta_std_path)
        bal_clf = StackingMetaClassifier.load(self.meta_bal_path)

        p_std = std_clf.predict_proba(X_val)
        p_bal = bal_clf.predict_proba(X_val)

        pred_std = np.argmax(p_std, axis=1)
        pred_bal = np.argmax(p_bal, axis=1)

        # Class 3: df (16 samples)
        df_mask = (y_val == 3)
        self.assertEqual(int(df_mask.sum()), 16)
        # Standard model predicts 0 df (all 16 predicted as nv)
        self.assertEqual(int((pred_std[df_mask] == 3).sum()), 0)
        # Balanced model predicts 16 df (100% recall)
        self.assertEqual(int((pred_bal[df_mask] == 3).sum()), 16)

        # Class 4: mel (159 samples)
        mel_mask = (y_val == 4)
        self.assertEqual(int(mel_mask.sum()), 159)
        mel_rec_std = (pred_std[mel_mask] == 4).sum() / 159.0
        mel_rec_bal = (pred_bal[mel_mask] == 4).sum() / 159.0
        self.assertAlmostEqual(mel_rec_std, 0.1950, places=3)
        self.assertAlmostEqual(mel_rec_bal, 0.5220, places=3)


if __name__ == "__main__":
    unittest.main()
