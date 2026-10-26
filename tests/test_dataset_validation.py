"""
Test Dataset Ingestion, Validation, and Auditing Pipeline.
Verifies file integrity, metadata compliance, corruption handling, and artifact generation.
"""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from src.data.dataset import (
    HAM10000DatasetAuditor,
    save_audit_artifacts,
    DatasetAuditReport
)


class TestDatasetValidation(unittest.TestCase):
    """Test suite for dataset validation and audit logic using synthetic fixtures."""

    def setUp(self):
        """Create a temporary directory with synthetic image and metadata fixtures."""
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.tmp_dir_obj.name)

        # Create valid subdirectories
        self.img_dir1 = self.raw_dir / "HAM10000_images_part_1"
        self.img_dir1.mkdir(parents=True)

        # Create 10 dummy images (224x224 RGB JPEGs)
        self.valid_images = [f"ISIC_000000{i}" for i in range(10)]
        for img_id in self.valid_images:
            img = Image.new("RGB", (224, 224), color=(np.random.randint(0, 255), 100, 100))
            img.save(self.img_dir1 / f"{img_id}.jpg", "JPEG")

        # Create valid metadata DataFrame
        classes = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df", "nv", "mel", "nv"]
        self.metadata_df = pd.DataFrame({
            "lesion_id": [f"HAM_00000{i % 7}" for i in range(10)],
            "image_id": self.valid_images,
            "dx": classes,
            "dx_type": ["histo"] * 10,
            "age": [45.0 + i for i in range(10)],
            "sex": ["male", "female"] * 5,
            "localization": ["back"] * 10
        })
        self.metadata_path = self.raw_dir / "HAM10000_metadata.csv"
        self.metadata_df.to_csv(self.metadata_path, index=False)

    def tearDown(self):
        """Clean up temporary test directory."""
        self.tmp_dir_obj.cleanup()

    def test_valid_dataset_audit(self):
        """Verify clean audit of a completely valid dataset fixture."""
        auditor = HAM10000DatasetAuditor(raw_dir=self.raw_dir)
        report, validated_df = auditor.audit()

        self.assertTrue(report.dataset_found)
        self.assertEqual(report.total_metadata_rows, 10)
        self.assertEqual(report.valid_records_count, 10)
        self.assertEqual(report.missing_images_count, 0)
        self.assertEqual(report.corrupted_images_count, 0)
        self.assertEqual(report.duplicate_image_ids_count, 0)
        self.assertEqual(report.invalid_labels_count, 0)
        self.assertIsNotNone(validated_df)
        self.assertEqual(len(validated_df), 10)
        self.assertIn("image_path", validated_df.columns)

    def test_missing_dataset_detection(self):
        """Verify auditor handles non-existent raw directory correctly."""
        non_existent = self.raw_dir / "does_not_exist"
        auditor = HAM10000DatasetAuditor(raw_dir=non_existent)
        report, validated_df = auditor.audit()

        self.assertFalse(report.dataset_found)
        self.assertIsNone(validated_df)
        self.assertGreater(len(report.errors), 0)

    def test_missing_image_files_detection(self):
        """Verify detection of image IDs listed in metadata but absent on disk."""
        # Add a row pointing to a missing image
        bad_df = pd.concat([
            self.metadata_df,
            pd.DataFrame([{
                "lesion_id": "HAM_9999",
                "image_id": "ISIC_MISSING_001",
                "dx": "nv",
                "dx_type": "histo",
                "age": 50.0,
                "sex": "male",
                "localization": "back"
            }])
        ], ignore_index=True)
        bad_df.to_csv(self.metadata_path, index=False)

        auditor = HAM10000DatasetAuditor(raw_dir=self.raw_dir)
        report, _ = auditor.audit()

        self.assertEqual(report.missing_images_count, 1)
        self.assertIn("ISIC_MISSING_001", report.missing_images)

    def test_corrupted_image_detection(self):
        """Verify detection of unreadable/corrupted image files."""
        # Create a corrupted file with random binary bytes
        corrupted_id = "ISIC_CORRUPT_001"
        corrupted_file = self.img_dir1 / f"{corrupted_id}.jpg"
        with open(corrupted_file, "wb") as f:
            f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_BYTES_1234567890")

        bad_df = pd.concat([
            self.metadata_df,
            pd.DataFrame([{
                "lesion_id": "HAM_9998",
                "image_id": corrupted_id,
                "dx": "mel",
                "dx_type": "histo",
                "age": 50.0,
                "sex": "female",
                "localization": "chest"
            }])
        ], ignore_index=True)
        bad_df.to_csv(self.metadata_path, index=False)

        auditor = HAM10000DatasetAuditor(raw_dir=self.raw_dir)
        report, validated_df = auditor.audit()

        self.assertEqual(report.corrupted_images_count, 1)
        # Corrupted row should be excluded from validated_df
        self.assertNotIn(corrupted_id, validated_df["image_id"].values)

    def test_invalid_label_detection(self):
        """Verify detection of unknown/unexpected diagnosis labels."""
        bad_df = self.metadata_df.copy()
        bad_df.loc[0, "dx"] = "unknown_skin_condition"
        bad_df.to_csv(self.metadata_path, index=False)

        auditor = HAM10000DatasetAuditor(raw_dir=self.raw_dir)
        report, _ = auditor.audit()

        self.assertEqual(report.invalid_labels_count, 1)
        self.assertIn("unknown_skin_condition", report.invalid_labels)

    def test_duplicate_image_id_detection(self):
        """Verify duplicate image_id entries in metadata are flagged."""
        bad_df = pd.concat([self.metadata_df, self.metadata_df.iloc[[0]]], ignore_index=True)
        bad_df.to_csv(self.metadata_path, index=False)

        auditor = HAM10000DatasetAuditor(raw_dir=self.raw_dir)
        report, _ = auditor.audit()

        self.assertEqual(report.duplicate_image_ids_count, 1)
        self.assertGreater(report.duplicate_metadata_rows_count, 0)

    def test_orphan_images_detection(self):
        """Verify images on disk without metadata entries are flagged as orphans."""
        orphan_img = Image.new("RGB", (224, 224), color=(50, 50, 50))
        orphan_img.save(self.img_dir1 / "ISIC_ORPHAN_999.jpg", "JPEG")

        auditor = HAM10000DatasetAuditor(raw_dir=self.raw_dir)
        report, _ = auditor.audit()

        self.assertEqual(report.orphan_images_count, 1)
        self.assertIn("ISIC_ORPHAN_999", report.orphan_images_sample)

    def test_save_audit_artifacts(self):
        """Verify saving of audit JSON, validated CSV, distribution CSV, and plot."""
        auditor = HAM10000DatasetAuditor(raw_dir=self.raw_dir)
        report, validated_df = auditor.audit()

        out_processed = self.raw_dir / "processed"
        out_plots = self.raw_dir / "plots"

        audit_p, meta_p, dist_p, plot_p = save_audit_artifacts(
            report, validated_df, processed_dir=out_processed, plots_dir=out_plots
        )

        self.assertTrue(audit_p.exists())
        self.assertTrue(meta_p.exists())
        self.assertTrue(dist_p.exists())
        self.assertTrue(plot_p.exists())

        # Verify audit JSON content
        with open(audit_p, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["valid_records_count"], 10)


if __name__ == "__main__":
    unittest.main()
