"""
Tests for DermaStack-XAI Stage 6 Final Benchmark and Evaluation Modules.

Covers:
    - Input validation for target labels and probability distributions
    - Multiclass Brier score mathematical correctness and edge cases
    - Multiclass benchmark computation (Accuracy, Macro/Weighted P/R/F1, Per-class, ROC-AUC, Log Loss)
    - Publication plotting functions (Confusion matrices, ROC curves, Comparison, Calibration)
    - Plot source data persistence (JSON serialization)
    - Frozen evaluation protocol integrity
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.config import DEFAULT_CONFIG, HAM10000_CLASSES
from src.evaluation.metrics import (
    compute_evaluation_benchmark,
    compute_multiclass_brier_score,
    validate_evaluation_inputs
)
from src.evaluation.plots import (
    plot_calibration_curves,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_multiclass_roc_curves,
    save_plot_source_data
)


class TestEvaluationInputValidation(unittest.TestCase):
    """Tests input validation for evaluation targets and probability distributions."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.num_classes = 7
        self.num_samples = 50
        self.valid_y_true = self.rng.integers(0, self.num_classes, size=self.num_samples)
        raw_p = self.rng.dirichlet(np.ones(self.num_classes), size=self.num_samples).astype(np.float32)
        self.valid_y_prob = raw_p / raw_p.sum(axis=1, keepdims=True)

    def test_valid_inputs_pass(self):
        y_t, y_p = validate_evaluation_inputs(self.valid_y_true, self.valid_y_prob, num_classes=self.num_classes)
        self.assertEqual(y_t.shape, (self.num_samples,))
        self.assertEqual(y_p.shape, (self.num_samples, self.num_classes))
        self.assertEqual(y_t.dtype, np.int32)
        self.assertEqual(y_p.dtype, np.float32)

    def test_invalid_label_dimension_raises(self):
        with self.assertRaises(ValueError):
            validate_evaluation_inputs(self.valid_y_true.reshape(-1, 1), self.valid_y_prob)

    def test_invalid_prob_dimension_raises(self):
        with self.assertRaises(ValueError):
            validate_evaluation_inputs(self.valid_y_true, self.valid_y_prob.flatten())

    def test_sample_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            validate_evaluation_inputs(self.valid_y_true[:10], self.valid_y_prob)

    def test_num_classes_mismatch_raises(self):
        with self.assertRaises(ValueError):
            validate_evaluation_inputs(self.valid_y_true, self.valid_y_prob[:, :6], num_classes=7)

    def test_nan_probabilities_raise(self):
        bad_prob = self.valid_y_prob.copy()
        bad_prob[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_evaluation_inputs(self.valid_y_true, bad_prob)

    def test_infinite_probabilities_raise(self):
        bad_prob = self.valid_y_prob.copy()
        bad_prob[0, 0] = np.inf
        with self.assertRaises(ValueError):
            validate_evaluation_inputs(self.valid_y_true, bad_prob)

    def test_negative_probabilities_raise(self):
        bad_prob = self.valid_y_prob.copy()
        bad_prob[0, 0] = -0.1
        with self.assertRaises(ValueError):
            validate_evaluation_inputs(self.valid_y_true, bad_prob)

    def test_invalid_label_index_raises(self):
        bad_labels = self.valid_y_true.copy()
        bad_labels[0] = 7  # Out of bounds for 7 classes [0..6]
        with self.assertRaises(ValueError):
            validate_evaluation_inputs(bad_labels, self.valid_y_prob, num_classes=7)

        bad_labels[0] = -1
        with self.assertRaises(ValueError):
            validate_evaluation_inputs(bad_labels, self.valid_y_prob, num_classes=7)

    def test_non_summing_probabilities_raise(self):
        bad_prob = np.ones((self.num_samples, self.num_classes), dtype=np.float32)  # Sums to 7
        with self.assertRaises(ValueError):
            validate_evaluation_inputs(self.valid_y_true, bad_prob)


class TestBrierScore(unittest.TestCase):
    """Tests Multiclass Brier score calculations."""

    def test_perfect_predictions_give_zero(self):
        y_true = np.array([0, 1, 2, 3], dtype=np.int32)
        y_prob = np.eye(4, dtype=np.float32)
        score = compute_multiclass_brier_score(y_true, y_prob, num_classes=4)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_completely_wrong_predictions_give_two(self):
        # Sample 0: true is 0, predicted 1 with prob 1.0 -> (0-1)^2 + (1-0)^2 = 2.0
        y_true = np.array([0, 0], dtype=np.int32)
        y_prob = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        score = compute_multiclass_brier_score(y_true, y_prob, num_classes=2)
        self.assertAlmostEqual(score, 2.0, places=4)

    def test_uniform_predictions_score(self):
        # 2 samples, 4 classes, uniform prob = 0.25
        # Per sample: (1 - 0.25)^2 + 3 * (0 - 0.25)^2 = 0.5625 + 3 * 0.0625 = 0.75
        y_true = np.array([0, 2], dtype=np.int32)
        y_prob = np.full((2, 4), 0.25, dtype=np.float32)
        score = compute_multiclass_brier_score(y_true, y_prob, num_classes=4)
        self.assertAlmostEqual(score, 0.75, places=4)

    def test_empty_input_returns_zero(self):
        score = compute_multiclass_brier_score(np.array([], dtype=np.int32), np.zeros((0, 7)), num_classes=7)
        self.assertEqual(score, 0.0)


class TestEvaluationBenchmark(unittest.TestCase):
    """Tests the full evaluation benchmark computation."""

    def setUp(self):
        self.rng = np.random.default_rng(123)
        self.num_classes = 7
        self.num_samples = 70
        # Ensure at least 10 samples per class
        self.y_true = np.repeat(np.arange(self.num_classes), 10)
        # Create probabilities with 80% weight on true class
        y_one_hot = np.eye(self.num_classes, dtype=np.float32)[self.y_true]
        noise = self.rng.dirichlet(np.ones(self.num_classes), size=self.num_samples).astype(np.float32)
        self.y_prob = 0.8 * y_one_hot + 0.2 * noise
        self.y_prob /= self.y_prob.sum(axis=1, keepdims=True)

    def test_benchmark_keys_and_ranges(self):
        bench = compute_evaluation_benchmark(
            y_true=self.y_true,
            y_prob=self.y_prob,
            classes=HAM10000_CLASSES,
            model_name="TestModel"
        )

        expected_keys = [
            "model_name", "sample_count", "accuracy",
            "macro_precision", "macro_recall", "macro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1",
            "roc_auc_ovr_macro", "multiclass_log_loss", "multiclass_brier_score",
            "per_class", "confusion_matrix"
        ]
        for key in expected_keys:
            self.assertIn(key, bench)

        self.assertEqual(bench["model_name"], "TestModel")
        self.assertEqual(bench["sample_count"], 70)
        self.assertGreaterEqual(bench["accuracy"], 0.7)
        self.assertLessEqual(bench["accuracy"], 1.0)
        self.assertGreaterEqual(bench["macro_f1"], 0.7)
        self.assertIsNotNone(bench["roc_auc_ovr_macro"])
        self.assertGreaterEqual(bench["roc_auc_ovr_macro"], 0.8)
        self.assertIsNotNone(bench["multiclass_log_loss"])
        self.assertGreaterEqual(bench["multiclass_brier_score"], 0.0)

        # Check per-class support
        for c in HAM10000_CLASSES:
            self.assertIn(c, bench["per_class"])
            self.assertEqual(bench["per_class"][c]["support"], 10)

        # Check confusion matrix dimensions
        cm = np.array(bench["confusion_matrix"])
        self.assertEqual(cm.shape, (7, 7))
        self.assertEqual(cm.sum(), 70)


class TestPlottingFunctions(unittest.TestCase):
    """Tests publication plotting and JSON source data generation."""

    def setUp(self):
        self.rng = np.random.default_rng(999)
        self.num_classes = 7
        self.num_samples = 70
        self.y_true = np.repeat(np.arange(self.num_classes), 10)
        y_one_hot = np.eye(self.num_classes, dtype=np.float32)[self.y_true]
        noise = self.rng.dirichlet(np.ones(self.num_classes), size=self.num_samples).astype(np.float32)
        self.y_prob = 0.75 * y_one_hot + 0.25 * noise
        self.y_prob /= self.y_prob.sum(axis=1, keepdims=True)
        self.cm = np.diag([10] * 7)

    def test_save_plot_source_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_data.png"
            json_path = save_plot_source_data({"metric": 0.95}, out_file)
            self.assertTrue(json_path.exists())
            self.assertEqual(json_path.suffix, ".json")
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["metric"], 0.95)

    def test_plot_confusion_matrix_raw_and_normalized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "cm_raw.png"
            norm_path = Path(tmp_dir) / "cm_norm.png"

            out_raw = plot_confusion_matrix(self.cm, HAM10000_CLASSES, "TestModel", raw_path, normalize=False)
            self.assertTrue(out_raw.exists())
            self.assertTrue(raw_path.with_suffix(".json").exists())

            out_norm = plot_confusion_matrix(self.cm, HAM10000_CLASSES, "TestModel", norm_path, normalize=True)
            self.assertTrue(out_norm.exists())
            self.assertTrue(norm_path.with_suffix(".json").exists())

    def test_plot_multiclass_roc_curves(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            roc_path = Path(tmp_dir) / "roc_curve.png"
            out_roc = plot_multiclass_roc_curves(self.y_true, self.y_prob, HAM10000_CLASSES, "TestModel", roc_path)
            self.assertTrue(out_roc.exists())
            self.assertTrue(roc_path.with_suffix(".json").exists())

    def test_plot_model_comparison(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            comp_path = Path(tmp_dir) / "comparison.png"
            models_metrics = {
                "Model A": {"accuracy": 0.80, "macro_f1": 0.75, "macro_recall": 0.73, "weighted_f1": 0.81},
                "Model B": {"accuracy": 0.85, "macro_f1": 0.79, "macro_recall": 0.78, "weighted_f1": 0.84}
            }
            out_comp = plot_model_comparison(models_metrics, comp_path)
            self.assertTrue(out_comp.exists())
            self.assertTrue(comp_path.with_suffix(".json").exists())

    def test_plot_calibration_curves(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            calib_path = Path(tmp_dir) / "calib.png"
            models_probs = {
                "Model A": self.y_prob,
                "Model B": self.y_prob * 0.9 + 0.1 / 7
            }
            out_calib = plot_calibration_curves(models_probs, self.y_true, HAM10000_CLASSES, calib_path)
            self.assertTrue(out_calib.exists())
            self.assertTrue(calib_path.with_suffix(".json").exists())


class TestFrozenProtocolIntegrity(unittest.TestCase):
    """Verifies the frozen evaluation protocol artifact."""

    def test_frozen_protocol_exists_and_valid(self):
        protocol_path = DEFAULT_CONFIG.paths.metrics_dir / "frozen_evaluation_protocol.json"
        self.assertTrue(protocol_path.exists(), "Frozen protocol artifact missing!")

        with open(protocol_path, "r", encoding="utf-8") as f:
            protocol = json.load(f)

        self.assertEqual(protocol["status"], "FROZEN_BEFORE_HOLDOUT_TEST_EVALUATION")
        self.assertEqual(protocol["dataset"]["expected_sample_count"], 1431)
        self.assertEqual(protocol["dataset"]["canonical_class_order"], list(HAM10000_CLASSES))
        self.assertEqual(len(protocol["evaluated_models"]), 5)
        self.assertIn("accuracy", protocol["metrics_specification"])
        self.assertIn("multiclass_brier_score", protocol["metrics_specification"])
        self.assertEqual(len(protocol["governance_and_integrity_rules"]), 4)


if __name__ == "__main__":
    unittest.main()
