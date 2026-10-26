"""
DermaStack-XAI Dataset Ingestion, Audit, and Validation Module.
Handles metadata inspection, image discovery, format validation, and integrity checks.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from src.config import DEFAULT_CONFIG, HAM10000_CLASSES, CLASS_NAMES_MAPPING
from src.utils.logger import get_logger

logger = get_logger("DermaStack-XAI.Dataset")


class DatasetValidationError(Exception):
    """Raised when critical dataset validation requirements fail."""
    pass


@dataclass
class DatasetAuditReport:
    """Structured container for dataset audit results."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_found: bool = False
    metadata_path: Optional[str] = None
    total_metadata_rows: int = 0
    valid_records_count: int = 0
    total_images_discovered: int = 0
    missing_images_count: int = 0
    missing_images: List[str] = field(default_factory=list)
    orphan_images_count: int = 0
    orphan_images_sample: List[str] = field(default_factory=list)
    corrupted_images_count: int = 0
    corrupted_images: List[str] = field(default_factory=list)
    duplicate_image_ids_count: int = 0
    duplicate_image_ids: List[str] = field(default_factory=list)
    duplicate_metadata_rows_count: int = 0
    invalid_labels_count: int = 0
    invalid_labels: List[str] = field(default_factory=list)
    patient_identifier_present: bool = False
    patient_identifier_col: Optional[str] = None
    patient_audit_notes: str = ""
    group_identifier_col: str = "lesion_id"
    unique_lesions_count: int = 0
    duplicate_lesion_views_count: int = 0
    duplicate_lesion_ratio_pct: float = 0.0
    class_distribution_counts: Dict[str, int] = field(default_factory=dict)
    class_distribution_percentages: Dict[str, float] = field(default_factory=dict)
    image_formats: Dict[str, int] = field(default_factory=dict)
    image_resolutions: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class HAM10000DatasetAuditor:
    """
    Validates the HAM10000 dataset layout, metadata integrity,
    and image files incrementally without loading entire archives into RAM.
    """

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        metadata_filename: str = "HAM10000_metadata.csv",
        image_subdirs: Optional[Tuple[str, ...]] = None,
        expected_classes: Optional[List[str]] = None,
        group_col: str = "lesion_id",
        patient_col: Optional[str] = None
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else DEFAULT_CONFIG.paths.raw_data_dir
        self.metadata_filename = metadata_filename
        self.image_subdirs = image_subdirs or DEFAULT_CONFIG.dataset.image_subdirs
        self.expected_classes = expected_classes or list(HAM10000_CLASSES)
        self.group_col = group_col
        self.patient_col = patient_col

    def find_metadata_file(self) -> Optional[Path]:
        """Searches for metadata CSV in raw_dir and common locations."""
        candidates = [
            self.raw_dir / self.metadata_filename,
            self.raw_dir / "metadata.csv",
            self.raw_dir / "HAM10000_metadata",
            self.raw_dir / "HAM10000_metadata.CSV"
        ]
        for p in candidates:
            if p.exists() and p.is_file():
                return p
        return None

    def discover_images(self) -> Dict[str, Path]:
        """
        Scans raw_dir and candidate subdirectories for image files.
        Maps normalized image_id (without extension) to absolute Path.
        """
        image_map: Dict[str, Path] = {}
        search_dirs = [self.raw_dir]
        for subdir in self.image_subdirs:
            p = self.raw_dir / subdir
            if p.exists() and p.is_dir():
                search_dirs.append(p)
        # Also check 'images' subfolder if present
        images_dir = self.raw_dir / "images"
        if images_dir.exists() and images_dir.is_dir():
            search_dirs.append(images_dir)

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        for directory in search_dirs:
            if not directory.exists():
                continue
            for item in directory.iterdir():
                if item.is_file() and item.suffix.lower() in valid_extensions:
                    image_id = item.stem
                    if image_id not in image_map:
                        image_map[image_id] = item
        return image_map

    def audit(self, max_inspect_images: Optional[int] = None) -> Tuple[DatasetAuditReport, Optional[pd.DataFrame]]:
        """
        Performs a complete audit of the dataset.

        Returns:
            Tuple of (DatasetAuditReport, validated_df or None)
        """
        report = DatasetAuditReport()
        report.group_identifier_col = self.group_col

        metadata_path = self.find_metadata_file()
        if metadata_path is None:
            report.dataset_found = False
            report.errors.append(
                f"Metadata file '{self.metadata_filename}' not found in '{self.raw_dir}'. "
                "Dataset is absent."
            )
            return report, None

        report.dataset_found = True
        report.metadata_path = str(metadata_path)

        # 1. Read Metadata CSV
        try:
            df = pd.read_csv(metadata_path)
            report.total_metadata_rows = len(df)
        except Exception as e:
            report.errors.append(f"Failed to parse metadata CSV at {metadata_path}: {str(e)}")
            return report, None

        # 2. Check Required Columns
        required_cols = ["image_id", self.group_col, "dx"]
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required:
            report.errors.append(f"Critical columns missing from metadata: {missing_required}")
            return report, None

        # 3. Check Patient Identifier
        potential_patient_cols = ["patient_id", "patient", "subject_id"]
        found_patient_col = None
        for col in potential_patient_cols:
            if col in df.columns:
                found_patient_col = col
                break

        if found_patient_col:
            report.patient_identifier_present = True
            report.patient_identifier_col = found_patient_col
            report.patient_audit_notes = (
                f"Patient identifier column '{found_patient_col}' found. "
                "Patient-level grouping is enabled."
            )
        else:
            report.patient_identifier_present = False
            report.patient_identifier_col = None
            report.patient_audit_notes = (
                f"No explicit patient identifier column ({potential_patient_cols}) found in metadata. "
                f"In the official HAM10000 dataset, '{self.group_col}' serves as the biological lesion group identifier. "
                "Patient-level separation cannot be claimed without explicit patient IDs."
            )

        # 4. Check Duplicate Metadata Rows and Image IDs
        duplicate_rows = df.duplicated().sum()
        report.duplicate_metadata_rows_count = int(duplicate_rows)
        if duplicate_rows > 0:
            report.warnings.append(f"Detected {duplicate_rows} exact duplicate rows in metadata.")

        duplicate_image_ids = df[df["image_id"].duplicated()]["image_id"].tolist()
        report.duplicate_image_ids_count = len(duplicate_image_ids)
        report.duplicate_image_ids = duplicate_image_ids[:20]
        if duplicate_image_ids:
            report.warnings.append(
                f"Detected {len(duplicate_image_ids)} duplicate image_id entries in metadata."
            )

        # 5. Check Diagnosis Labels
        unique_labels = set(df["dx"].dropna().unique())
        invalid_labels = [lbl for lbl in unique_labels if lbl not in self.expected_classes]
        report.invalid_labels_count = len(invalid_labels)
        report.invalid_labels = invalid_labels
        if invalid_labels:
            report.errors.append(
                f"Invalid diagnosis labels found: {invalid_labels}. Expected: {self.expected_classes}"
            )

        # 6. Check Lesion Identifiers & Multi-Image Duplicate Views
        unique_lesions = df[self.group_col].nunique()
        duplicate_views = len(df) - unique_lesions
        report.unique_lesions_count = unique_lesions
        report.duplicate_lesion_views_count = duplicate_views
        report.duplicate_lesion_ratio_pct = round((duplicate_views / len(df)) * 100, 2) if len(df) > 0 else 0.0

        # Class counts in metadata
        class_counts = df["dx"].value_counts().to_dict()
        report.class_distribution_counts = {k: int(v) for k, v in class_counts.items()}
        report.class_distribution_percentages = {
            k: round((float(v) / len(df)) * 100, 2) for k, v in class_counts.items()
        }

        # 7. Discover and Validate Images on Disk
        image_map = self.discover_images()
        report.total_images_discovered = len(image_map)

        metadata_image_ids = set(df["image_id"].astype(str))
        discovered_image_ids = set(image_map.keys())

        missing_images = list(metadata_image_ids - discovered_image_ids)
        orphan_images = list(discovered_image_ids - metadata_image_ids)

        report.missing_images_count = len(missing_images)
        report.missing_images = missing_images[:50]
        report.orphan_images_count = len(orphan_images)
        report.orphan_images_sample = orphan_images[:20]

        if missing_images:
            report.errors.append(
                f"{len(missing_images)} images referenced in metadata are missing from disk."
            )
        if orphan_images:
            report.warnings.append(
                f"{len(orphan_images)} image files discovered on disk that are not in metadata."
            )

        # 8. Incremental Image Readability & Format Verification
        valid_indices = []
        corrupted_images = []
        formats: Dict[str, int] = {}
        resolutions: Dict[str, int] = {}

        # Limit inspection if specified (e.g. for fast sanity checks or tests)
        rows_to_check = df if max_inspect_images is None else df.head(max_inspect_images)

        for idx, row in rows_to_check.iterrows():
            img_id = str(row["image_id"])
            if img_id not in image_map:
                continue

            img_path = image_map[img_id]
            try:
                # Open header only (memory-safe, does not decompress full image buffer into RAM)
                with Image.open(img_path) as img:
                    img_format = str(img.format)
                    img_size = f"{img.size[0]}x{img.size[1]}"
                    formats[img_format] = formats.get(img_format, 0) + 1
                    resolutions[img_size] = resolutions.get(img_size, 0) + 1

                    # Verify image stream integrity
                    img.verify()

                # Add file_path to DataFrame
                valid_indices.append(idx)
            except Exception as e:
                corrupted_images.append(f"{img_id}: {str(e)}")

        report.corrupted_images_count = len(corrupted_images)
        report.corrupted_images = corrupted_images[:20]
        report.image_formats = formats
        report.image_resolutions = resolutions

        if corrupted_images:
            report.errors.append(f"{len(corrupted_images)} images failed integrity verification.")

        # Filter validated records
        validated_df = df.loc[valid_indices].copy()
        validated_df["image_path"] = validated_df["image_id"].map(lambda x: str(image_map.get(str(x), "")))
        report.valid_records_count = len(validated_df)

        return report, validated_df


def save_audit_artifacts(
    report: DatasetAuditReport,
    validated_df: pd.DataFrame,
    processed_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None
) -> Tuple[Path, Path, Path, Optional[Path]]:
    """
    Saves validated dataset artifacts:
    - data/processed/dataset_audit.json
    - data/processed/validated_metadata.csv
    - data/processed/class_distribution.csv
    - outputs/plots/class_distribution.png (if matplotlib available)
    """
    processed_dir = Path(processed_dir) if processed_dir else DEFAULT_CONFIG.paths.processed_data_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save dataset_audit.json
    audit_path = processed_dir / "dataset_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info(f"Saved dataset audit report to {audit_path}")

    # 2. Save validated_metadata.csv
    metadata_path = processed_dir / "validated_metadata.csv"
    validated_df.to_csv(metadata_path, index=False)
    logger.info(f"Saved validated metadata to {metadata_path}")

    # 3. Save class_distribution.csv
    class_dist_path = processed_dir / "class_distribution.csv"
    dist_df = pd.DataFrame([
        {
            "class_code": cls,
            "class_name": CLASS_NAMES_MAPPING.get(cls, cls),
            "count": report.class_distribution_counts.get(cls, 0),
            "percentage": report.class_distribution_percentages.get(cls, 0.0)
        }
        for cls in DEFAULT_CONFIG.dataset.classes
    ])
    dist_df.to_csv(class_dist_path, index=False)
    logger.info(f"Saved class distribution table to {class_dist_path}")

    # 4. Generate class_distribution.png
    plot_file = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        target_plots_dir = Path(plots_dir) if plots_dir else (DEFAULT_CONFIG.paths.root / "outputs" / "plots")
        target_plots_dir.mkdir(parents=True, exist_ok=True)
        plot_file = target_plots_dir / "class_distribution.png"

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            dist_df["class_code"],
            dist_df["count"],
            color=["#2b5c8f", "#3c7bb6", "#569bd8", "#78baf2", "#e07a5f", "#d62828", "#8338ec"]
        )
        ax.set_title("HAM10000 Dataset Class Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Diagnostic Category", fontsize=12)
        ax.set_ylabel("Number of Images", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10
            )

        plt.tight_layout()
        plt.savefig(plot_file, dpi=200)
        plt.close(fig)
        logger.info(f"Saved class distribution plot to {plot_file}")
    except Exception as e:
        logger.warning(f"Could not generate plot artifact: {e}")

    return audit_path, metadata_path, class_dist_path, plot_file


if __name__ == "__main__":
    auditor = HAM10000DatasetAuditor()
    report, validated_df = auditor.audit()

    if not report.dataset_found:
        print("\n[!] DATASET ABSENT: No HAM10000 metadata or image files found in data/raw/.")
        print("Please refer to docs/DATASET_GUIDELINES.md for download instructions.\n")
    else:
        print("\n=== HAM10000 DATASET AUDIT SUMMARY ===")
        print(f"Total Metadata Rows: {report.total_metadata_rows}")
        print(f"Valid Records:       {report.valid_records_count}")
        print(f"Unique Lesions:      {report.unique_lesions_count}")
        print(f"Missing Images:      {report.missing_images_count}")
        print(f"Corrupted Images:    {report.corrupted_images_count}")
        print("Class Distribution:", report.class_distribution_counts)
        if validated_df is not None and len(validated_df) > 0:
            save_audit_artifacts(report, validated_df)
            print("\nGenerating leakage-safe patient/lesion-stratified splits...")
            from src.data.splitter import split_patient_aware, save_split_manifest
            train_df, val_df, test_df, manifest = split_patient_aware(validated_df)
            manifest_path = save_split_manifest(manifest)
            print(f"[OK] Split manifest successfully generated: {manifest_path}")
            print(f"     Train images: {manifest.actual_image_counts['train']} ({manifest.actual_image_percentages['train']}%)")
            print(f"     Val images:   {manifest.actual_image_counts['val']} ({manifest.actual_image_percentages['val']}%)")
            print(f"     Test images:  {manifest.actual_image_counts['test']} ({manifest.actual_image_percentages['test']}%)")
