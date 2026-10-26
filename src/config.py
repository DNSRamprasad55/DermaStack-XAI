"""
DermaStack-XAI: Configuration Module
Defines centralized, reproducible, and type-safe project settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Base Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Seven diagnostic categories from the HAM10000 dataset
HAM10000_CLASSES: List[str] = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

CLASS_NAMES_MAPPING: Dict[str, str] = {
    "akiec": "Actinic Keratoses & Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "akiec": "Precancerous or early non-melanoma skin lesions including actinic keratoses and Bowen's disease.",
    "bcc": "A common type of non-melanoma skin cancer originating from basal cells.",
    "bkl": "Non-cancerous lesions including seborrheic keratoses, solar lentigines, and lichen planus-like keratoses.",
    "df": "A benign fibrous skin nodule typically found on the lower extremities.",
    "mel": "A malignant skin tumor arising from melanocytes, requiring urgent dermatological evaluation.",
    "nv": "Common benign moles formed by clusters of melanocytes.",
    "vasc": "Benign vascular proliferations including cherry angiomas, pyogenic granulomas, and angiokeratomas."
}

MEDICAL_DISCLAIMER: str = (
    "RESEARCH PROTOTYPE DISCLAIMER: DermaStack-XAI is an experimental academic research and "
    "educational tool designed to explore multi-model ensembles and visual explainability (Grad-CAM). "
    "It is NOT a medical device, is NOT CE-marked or FDA-cleared, and MUST NOT be used as a substitute "
    "for professional clinical assessment, dermatoscopy, or histopathological diagnosis. Predictions "
    "and visual heatmaps do not constitute medical confirmation."
)


@dataclass(frozen=True)
class PathConfig:
    """Centralized path definitions."""
    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    splits_dir: Path = PROJECT_ROOT / "data" / "splits"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    checkpoints_dir: Path = PROJECT_ROOT / "artifacts" / "checkpoints"
    oof_dir: Path = PROJECT_ROOT / "artifacts" / "oof"
    meta_models_dir: Path = PROJECT_ROOT / "artifacts" / "meta_models"
    metrics_dir: Path = PROJECT_ROOT / "artifacts" / "metrics"
    figures_dir: Path = PROJECT_ROOT / "artifacts" / "figures"
    logs_dir: Path = PROJECT_ROOT / "logs"


@dataclass
class DatasetConfig:
    """Dataset splitting, resolution, and batch settings."""
    metadata_filename: str = "HAM10000_metadata.csv"
    image_subdirs: Tuple[str, ...] = ("HAM10000_images_part_1", "HAM10000_images_part_2")
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_classes: int = 7
    classes: List[str] = field(default_factory=lambda: list(HAM10000_CLASSES))
    test_ratio: float = 0.15
    val_ratio: float = 0.15
    group_col: str = "lesion_id"
    label_col: str = "dx"
    random_seed: int = 42


@dataclass
class ModelHyperparameters:
    """Hyperparameters for deep learning models."""
    learning_rate: float = 1e-4
    epochs: int = 30
    early_stopping_patience: int = 7
    reduce_lr_patience: int = 3
    dropout_rate: float = 0.4
    l2_reg: float = 1e-4
    k_folds: int = 5


@dataclass
class XAIConfig:
    """Grad-CAM explainability settings."""
    custom_cnn_layer: str = "conv4_block"
    efficientnet_layer: str = "top_conv"
    resnet_layer: str = "conv5_block3_out"
    colormap: str = "jet"
    alpha: float = 0.4


@dataclass
class GlobalConfig:
    """Master configuration container."""
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    xai: XAIConfig = field(default_factory=XAIConfig)
    random_seed: int = 42


DEFAULT_CONFIG = GlobalConfig()
