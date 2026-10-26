"""
DermaStack-XAI Model Architecture Package.
Exposes Custom CNN, EfficientNetB0, and ResNet50 builders, along with
shared preprocessing and model inspection utilities.
"""

from src.models.custom_cnn import build_custom_cnn
from src.models.efficientnet import build_efficientnet_b0
from src.models.resnet import build_resnet50
from src.models.common import (
    get_model_summary_dict,
    preprocess_for_model,
    validate_output_tensor
)

__all__ = [
    "build_custom_cnn",
    "build_efficientnet_b0",
    "build_resnet50",
    "get_model_summary_dict",
    "preprocess_for_model",
    "validate_output_tensor"
]
