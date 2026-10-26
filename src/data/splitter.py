"""
DermaStack-XAI Data Splitter:
Implements Patient-Aware and Lesion-Aware Stratified Group Splitting to strictly prevent data leakage.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import DEFAULT_CONFIG
from src.utils.logger import get_logger

logger = get_logger("DermaStack-XAI.Splitter")


class InfeasibleSplitError(Exception):
    """Raised when group constraints or extreme rarity make the requested split infeasible."""
    pass


@dataclass
class SplitManifest:
    """Serializable container for split results and verification metrics."""
    random_seed: int
    group_column_used: str
    patient_level_separation_achieved: bool
    rationale: str
    target_ratios: Dict[str, float]
    actual_image_counts: Dict[str, int]
    actual_image_percentages: Dict[str, float]
    actual_group_counts: Dict[str, int]
    group_overlap_detected: bool
    class_distribution_per_split: Dict[str, Dict[str, int]]
    splits: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def audit_dataset_identifiers(df: pd.DataFrame, group_col: str = "lesion_id") -> Dict[str, int]:
    """
    Audits patient and lesion identifiers in the metadata.
    Returns counts of total records, unique lesions, and duplicate images.
    """
    total_images = len(df)
    unique_lesions = df[group_col].nunique() if group_col in df.columns else 0
    duplicate_images = total_images - unique_lesions
    return {
        "total_images": total_images,
        "unique_lesions": unique_lesions,
        "duplicate_images": duplicate_images,
        "duplicate_ratio_pct": round((duplicate_images / total_images) * 100, 2) if total_images > 0 else 0.0
    }


def determine_effective_group_column(
    df: pd.DataFrame,
    preferred_group_col: str = "lesion_id",
    patient_col: Optional[str] = None
) -> Tuple[str, bool, str]:
    """
    Determines whether to group on patient_id or lesion_id.

    Returns:
        (group_column, patient_level_achieved, rationale)
    """
    # 1. Check if explicit patient column is requested or present
    candidate_patient_cols = [patient_col] if patient_col else ["patient_id", "patient", "subject_id"]
    for col in candidate_patient_cols:
        if col and col in df.columns and df[col].notna().sum() > 0:
            rationale = (
                f"Patient identifier column '{col}' is present. Grouping by '{col}' guarantees "
                f"both patient-level and lesion-level isolation (since all lesions of a patient stay together)."
            )
            return col, True, rationale

    # 2. Check for lesion_id
    if preferred_group_col in df.columns:
        rationale = (
            f"No reliable patient identifier found. Grouping strictly on '{preferred_group_col}'. "
            f"Note: This guarantees lesion-level separation, but patient-level separation cannot be claimed "
            f"without explicit patient identifiers."
        )
        return preferred_group_col, False, rationale

    raise ValueError(
        f"Neither patient identifier nor group column '{preferred_group_col}' found in DataFrame. "
        f"Available columns: {list(df.columns)}"
    )


def split_patient_aware(
    df: pd.DataFrame,
    group_col: str = "lesion_id",
    target_col: str = "dx",
    patient_col: Optional[str] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest]:
    """
    Performs patient-aware or lesion-aware stratified group splitting.
    Guarantees zero group overlap across train, validation, and test sets.

    Args:
        df: Metadata DataFrame.
        group_col: Group column (default 'lesion_id').
        target_col: Diagnosis class column (default 'dx').
        patient_col: Optional explicit patient identifier column.
        test_size: Target fraction for holdout test set (~0.15).
        val_size: Target fraction for validation set (~0.15).
        random_seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df, split_manifest)
    """
    if len(df) == 0:
        raise ValueError("Cannot split empty DataFrame.")

    # 1. Determine effective grouping column
    eff_group_col, patient_level_achieved, rationale = determine_effective_group_column(
        df, preferred_group_col=group_col, patient_col=patient_col
    )

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from DataFrame.")

    # 2. Check Feasibility & Rare Classes
    class_group_counts = df.groupby(target_col)[eff_group_col].nunique()
    min_groups_in_class = class_group_counts.min()
    rare_classes = class_group_counts[class_group_counts < 3].to_dict()

    # Need at least 3 groups per class to guarantee representation in train, val, and test
    if min_groups_in_class < 3:
        logger.warning(
            f"Rare classes with fewer than 3 groups detected: {rare_classes}. "
            "Group constraints will prevent these classes from being represented in all three partitions."
        )

    # 3. Calculate target split parameters
    # Test partition split
    n_splits_test = max(2, int(round(1.0 / test_size)))
    if n_splits_test > df[eff_group_col].nunique():
        raise InfeasibleSplitError(
            f"Cannot perform {n_splits_test}-fold group split: total unique groups "
            f"({df[eff_group_col].nunique()}) is smaller than n_splits ({n_splits_test})."
        )

    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=random_seed)

    groups = df[eff_group_col].values
    y = df[target_col].values
    X = np.zeros(len(df))

    # Split into (Train+Val) and Test
    try:
        train_val_idx, test_idx = next(sgkf_test.split(X, y, groups))
    except Exception as e:
        raise InfeasibleSplitError(f"Stratified group split failed on test partition: {str(e)}")

    train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # 4. Split train_val into Train and Val
    train_ratio = 1.0 - test_size - val_size
    adjusted_val_ratio = val_size / (train_ratio + val_size)
    n_splits_val = max(2, int(round(1.0 / adjusted_val_ratio)))

    if n_splits_val > train_val_df[eff_group_col].nunique():
        raise InfeasibleSplitError(
            f"Cannot perform {n_splits_val}-fold group split on validation set: "
            f"unique groups ({train_val_df[eff_group_col].nunique()}) < n_splits ({n_splits_val})."
        )

    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=random_seed)

    groups_tv = train_val_df[eff_group_col].values
    y_tv = train_val_df[target_col].values
    X_tv = np.zeros(len(train_val_df))

    try:
        train_idx, val_idx = next(sgkf_val.split(X_tv, y_tv, groups_tv))
    except Exception as e:
        raise InfeasibleSplitError(f"Stratified group split failed on validation partition: {str(e)}")

    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

    # 5. Strict Assertion: Zero Group Overlap Across Partitions
    train_groups: Set[str] = set(train_df[eff_group_col])
    val_groups: Set[str] = set(val_df[eff_group_col])
    test_groups: Set[str] = set(test_df[eff_group_col])

    overlap_train_val = train_groups.intersection(val_groups)
    overlap_train_test = train_groups.intersection(test_groups)
    overlap_val_test = val_groups.intersection(test_groups)

    overlap_detected = bool(overlap_train_val or overlap_train_test or overlap_val_test)
    if overlap_detected:
        raise AssertionError(
            f"CRITICAL DATA LEAKAGE: Overlaps detected across splits! "
            f"Train-Val: {len(overlap_train_val)}, Train-Test: {len(overlap_train_test)}, Val-Test: {len(overlap_val_test)}"
        )

    # 6. Build Manifest and Per-Class Distributions
    total_len = len(df)
    counts = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df)
    }
    pcts = {
        k: round((v / total_len) * 100, 2) for k, v in counts.items()
    }
    grp_counts = {
        "train": len(train_groups),
        "val": len(val_groups),
        "test": len(test_groups)
    }

    per_class_dist = {
        "train": {str(k): int(v) for k, v in train_df[target_col].value_counts().to_dict().items()},
        "val": {str(k): int(v) for k, v in val_df[target_col].value_counts().to_dict().items()},
        "test": {str(k): int(v) for k, v in test_df[target_col].value_counts().to_dict().items()}
    }

    # Record warnings if group constraints caused deviation
    warnings = []
    if abs(pcts["train"] - 70.0) > 5.0 or abs(pcts["val"] - 15.0) > 4.0 or abs(pcts["test"] - 15.0) > 4.0:
        warnings.append(
            f"Actual image split percentages ({pcts}) diverge slightly from target (70/15/15) "
            "due to integer group constraints (multi-image lesions staying intact)."
        )

    image_col = "image_id" if "image_id" in df.columns else None
    splits_dict = {}
    if image_col:
        splits_dict = {
            "train": train_df[image_col].astype(str).tolist(),
            "val": val_df[image_col].astype(str).tolist(),
            "test": test_df[image_col].astype(str).tolist()
        }

    manifest = SplitManifest(
        random_seed=random_seed,
        group_column_used=eff_group_col,
        patient_level_separation_achieved=patient_level_achieved,
        rationale=rationale,
        target_ratios={"train": 0.70, "val": 0.15, "test": 0.15},
        actual_image_counts=counts,
        actual_image_percentages=pcts,
        actual_group_counts=grp_counts,
        group_overlap_detected=False,
        class_distribution_per_split=per_class_dist,
        splits=splits_dict,
        warnings=warnings
    )

    return train_df, val_df, test_df, manifest


def save_split_manifest(manifest: SplitManifest, output_dir: Optional[Path] = None) -> Path:
    """Saves the split manifest to data/splits/train_val_test_splits.json."""
    output_dir = Path(output_dir) if output_dir else DEFAULT_CONFIG.paths.splits_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "train_val_test_splits.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    logger.info(f"Saved reproducible split manifest to {manifest_path}")
    return manifest_path
