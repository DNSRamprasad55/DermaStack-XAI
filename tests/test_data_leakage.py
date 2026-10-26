"""
Test Data Leakage Prevention and Patient Identifier Auditing.
Verifies that no lesion or patient identifier leaks between train, val, and test splits.
"""

import unittest
import numpy as np
import pandas as pd
from src.data.splitter import (
    split_patient_aware,
    audit_dataset_identifiers,
    determine_effective_group_column,
    InfeasibleSplitError,
    save_split_manifest
)
import tempfile
from pathlib import Path


class TestDataLeakage(unittest.TestCase):
    """Test suite for verifying zero-leakage group splitting."""

    def setUp(self):
        """Create a synthetic dataset reflecting HAM10000 structure."""
        np.random.seed(42)
        classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        class_weights = [0.03, 0.05, 0.11, 0.01, 0.11, 0.67, 0.02]

        # 500 unique lesions across 400 patients
        n_lesions = 500
        records = []

        for i in range(n_lesions):
            lesion_id = f"HAM_{i:05d}"
            patient_id = f"IP_{i % 400:05d}"  # Some patients have multiple lesions
            dx = np.random.choice(classes, p=class_weights)
            num_views = np.random.choice([1, 2, 3], p=[0.70, 0.25, 0.05])
            for v in range(num_views):
                records.append({
                    "lesion_id": lesion_id,
                    "image_id": f"ISIC_{len(records):07d}",
                    "patient_id": patient_id,
                    "dx": dx,
                    "dx_type": "histo",
                    "age": float(np.random.randint(20, 85)),
                    "sex": np.random.choice(["male", "female"]),
                    "localization": "back"
                })

        self.df = pd.DataFrame(records)

    def test_audit_dataset_identifiers(self):
        """Verify audit counts unique lesions and duplicate image views."""
        audit = audit_dataset_identifiers(self.df, group_col="lesion_id")
        self.assertEqual(audit["unique_lesions"], 500)
        self.assertGreater(audit["total_images"], 500)
        self.assertGreater(audit["duplicate_images"], 0)
        self.assertGreater(audit["duplicate_ratio_pct"], 0.0)

    def test_determine_effective_group_column_with_patient_id(self):
        """Verify patient_id is selected when available, explaining patient separation."""
        col, achieved, rationale = determine_effective_group_column(self.df, preferred_group_col="lesion_id")
        self.assertEqual(col, "patient_id")
        self.assertTrue(achieved)
        self.assertIn("patient-level", rationale.lower())

    def test_determine_effective_group_column_without_patient_id(self):
        """Verify fallback to lesion_id when patient_id is absent, with clear limitation note."""
        df_no_patient = self.df.drop(columns=["patient_id"])
        col, achieved, rationale = determine_effective_group_column(df_no_patient, preferred_group_col="lesion_id")
        self.assertEqual(col, "lesion_id")
        self.assertFalse(achieved)
        self.assertIn("cannot be claimed", rationale.lower())

    def test_zero_leakage_lesion_group_split(self):
        """Verify that NO lesion appears in more than one partition when grouping by lesion_id."""
        df_no_patient = self.df.drop(columns=["patient_id"])
        train_df, val_df, test_df, manifest = split_patient_aware(
            df_no_patient,
            group_col="lesion_id",
            target_col="dx",
            test_size=0.15,
            val_size=0.15,
            random_seed=42
        )

        train_lesions = set(train_df["lesion_id"])
        val_lesions = set(val_df["lesion_id"])
        test_lesions = set(test_df["lesion_id"])

        self.assertEqual(len(train_lesions.intersection(val_lesions)), 0)
        self.assertEqual(len(train_lesions.intersection(test_lesions)), 0)
        self.assertEqual(len(val_lesions.intersection(test_lesions)), 0)

        total_partitioned = len(train_lesions) + len(val_lesions) + len(test_lesions)
        self.assertEqual(total_partitioned, 500)
        self.assertFalse(manifest.group_overlap_detected)

    def test_zero_leakage_patient_group_split(self):
        """Verify that NO patient (and hence NO lesion) leaks across partitions when grouping by patient_id."""
        train_df, val_df, test_df, manifest = split_patient_aware(
            self.df,
            patient_col="patient_id",
            group_col="lesion_id",
            target_col="dx",
            test_size=0.15,
            val_size=0.15,
            random_seed=42
        )

        train_patients = set(train_df["patient_id"])
        val_patients = set(val_df["patient_id"])
        test_patients = set(test_df["patient_id"])

        self.assertEqual(len(train_patients.intersection(val_patients)), 0)
        self.assertEqual(len(train_patients.intersection(test_patients)), 0)
        self.assertEqual(len(val_patients.intersection(test_patients)), 0)

        # Furthermore, lesions must also not leak
        train_lesions = set(train_df["lesion_id"])
        val_lesions = set(val_df["lesion_id"])
        test_lesions = set(test_df["lesion_id"])

        self.assertEqual(len(train_lesions.intersection(val_lesions)), 0)
        self.assertEqual(len(train_lesions.intersection(test_lesions)), 0)
        self.assertEqual(len(val_lesions.intersection(test_lesions)), 0)
        self.assertTrue(manifest.patient_level_separation_achieved)

    def test_split_reproducibility(self):
        """Verify that running with the same seed yields identical splits, and different seeds differ."""
        _, _, _, m1 = split_patient_aware(self.df, random_seed=42)
        _, _, _, m2 = split_patient_aware(self.df, random_seed=42)
        _, _, _, m3 = split_patient_aware(self.df, random_seed=999)

        self.assertEqual(m1.splits["train"], m2.splits["train"])
        self.assertEqual(m1.splits["test"], m2.splits["test"])
        self.assertNotEqual(m1.splits["train"], m3.splits["train"])

    def test_missing_group_column_raises_error(self):
        """Verify ValueError is raised if group and patient columns are missing."""
        df_bad = self.df.drop(columns=["lesion_id", "patient_id"])
        with self.assertRaises(ValueError):
            split_patient_aware(df_bad, group_col="lesion_id")

    def test_infeasible_split_handling(self):
        """Verify InfeasibleSplitError is raised when n_splits exceeds group count."""
        tiny_df = pd.DataFrame([
            {"lesion_id": "L1", "image_id": "I1", "dx": "nv"},
            {"lesion_id": "L2", "image_id": "I2", "dx": "nv"}
        ])
        with self.assertRaises(InfeasibleSplitError):
            # test_size = 0.1 -> n_splits = 10, but only 2 groups exist
            split_patient_aware(tiny_df, group_col="lesion_id", test_size=0.1)

    def test_save_split_manifest(self):
        """Verify manifest serialization to JSON."""
        _, _, _, manifest = split_patient_aware(self.df, random_seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = save_split_manifest(manifest, output_dir=Path(tmpdir))
            self.assertTrue(p.exists())
            self.assertTrue(p.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
