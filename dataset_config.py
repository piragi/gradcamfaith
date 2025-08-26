"""
Dataset Configuration Module

This module defines the configuration for different datasets, providing a unified interface
for dataset-specific settings like class names, transforms, and model configurations.

IMPORTANT: All preprocessing is centralized here through callable transforms.
This is the SINGLE SOURCE OF TRUTH for all dataset preprocessing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torchvision.transforms as transforms
from PIL import Image

# Handle different torchvision versions
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    # Fallback for older torchvision versions
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

# ============================================================================
# Dataset-specific transform functions
# Each dataset has its own preprocessing requirements
# ============================================================================


def create_covidquex_transform(_) -> transforms.Compose:
    """
    Create CovidQUEX-specific transforms.
    
    CovidQUEX uses ViT only (no CLIP):
    - Resize(256) + CenterCrop(224) pipeline
    - Custom normalization: mean=[0.56, 0.56, 0.56], std=[0.21, 0.21, 0.21]
    - Augmentations only for training
    """
    # CovidQUEX-specific normalization
    mean = [0.56, 0.56, 0.56]
    std = [0.21, 0.21, 0.21]
    interpolation = BILINEAR

    return transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def create_hyperkvasir_transform(split: str = 'test') -> transforms.Compose:
    """
    Create HyperKvasir-specific transforms.
    
    HyperKvasir uses ViT only (no CLIP):
    - Direct resize to 224x224 (images are already preprocessed to 224x224)
    - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - Extensive augmentations for training (ColorJitter, GaussianBlur, flips, rotation)
    """
    # ImageNet normalization for HyperKvasir
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    interpolation = BILINEAR

    if split == 'train':
        # Extensive augmentations as per SSL4GIE paper
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=interpolation),
            transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # No augmentations for val/test
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def create_waterbirds_transform(_) -> transforms.Compose:
    """
    Create Waterbirds-specific transforms.
    
    Waterbirds uses CLIP only (no ViT):
    - Direct resize to 224x224
    - CLIP normalization
    - No augmentations (natural images)
    """

    from vit_prisma.transforms import get_clip_val_transforms
    return get_clip_val_transforms()


def create_oxford_pets_transform(_) -> transforms.Compose:
    """
    Create Oxford Pets-specific transforms.
    
    Oxford Pets uses CLIP only (no ViT):
    - Direct resize to 224x224
    - CLIP normalization
    - No augmentations (natural images)
    """
    from vit_prisma.transforms import get_clip_val_transforms
    return get_clip_val_transforms()


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset with centralized preprocessing."""

    name: str  # Dataset identifier
    num_classes: int  # Number of classes
    class_names: List[str]  # Original class names (for display/logging)
    class_to_idx: Dict[str, int]  # Mapping from original names to indices
    idx_to_class: Dict[int, str]  # Reverse mapping

    # Model configuration
    model_checkpoint: str  # Path to the trained model

    # Transform callable - THE SINGLE SOURCE OF TRUTH for preprocessing
    transform_fn: Callable[[str], transforms.Compose] = field(default=None)

    # Data paths (set during runtime)
    prepared_data_path: Optional[Path] = None

    # DEPRECATED - kept for backward compatibility only
    # DO NOT USE THESE - use transform_fn instead
    img_size: int = 224
    normalize_mean: List[float] = None
    normalize_std: List[float] = None

    def __post_init__(self):
        """Set default normalization values if not provided (DEPRECATED)."""
        if self.normalize_mean is None:
            self.normalize_mean = [0.485, 0.456, 0.406]  # ImageNet defaults
        if self.normalize_std is None:
            self.normalize_std = [0.229, 0.224, 0.225]  # ImageNet defaults

    def get_transforms(self, split: str = 'test', use_clip: bool = False):
        """
        Get transforms for a specific split (train/val/test/dev).
        
        This is THE centralized method for all preprocessing.
        
        Args:
            split: The data split ('train', 'val', 'test', 'dev')
            use_clip: If True, use CLIP-specific normalization values
        """
        if self.transform_fn is not None:
            # Use the new callable transform (THIS IS THE PREFERRED PATH)
            return self.transform_fn(split)

        # If no transform_fn is defined, raise an error
        raise ValueError(
            f"No transform function defined for dataset {self.name}. "
            "All datasets must define a transform_fn."
        )


# CovidQUEX Dataset Configuration
COVIDQUEX_CONFIG = DatasetConfig(
    name="covidquex",
    num_classes=3,
    class_names=["COVID-19", "Non-COVID", "Normal"],
    class_to_idx={
        "COVID-19": 0,
        "Non-COVID": 1,
        "Normal": 2
    },
    idx_to_class={
        0: "COVID-19",
        1: "Non-COVID",
        2: "Normal"
    },
    model_checkpoint="./models/covidquex/covidquex_model.pth",
    transform_fn=create_covidquex_transform,  # Use the centralized transform
    # DEPRECATED - kept for reference only
    normalize_mean=[0.56, 0.56, 0.56],
    normalize_std=[0.21, 0.21, 0.21]
)

# HyperKvasir Dataset Configuration
HYPERKVASIR_CONFIG = DatasetConfig(
    name="hyperkvasir",
    num_classes=6,
    class_names=["cecum", "ileum", "retroflex-rectum", "pylorus", "retroflex-stomach", "z-line"],
    class_to_idx={
        "cecum": 0,
        "ileum": 1,
        "retroflex-rectum": 2,
        "pylorus": 3,
        "retroflex-stomach": 4,
        "z-line": 5
    },
    idx_to_class={
        0: "cecum",
        1: "ileum",
        2: "retroflex-rectum",
        3: "pylorus",
        4: "retroflex-stomach",
        5: "z-line"
    },
    model_checkpoint="./models/hyperkvasir/hyperkvasir_vit_model.pth",
    transform_fn=create_hyperkvasir_transform  # Use the centralized transform
)

# Waterbirds Dataset Configuration
WATERBIRDS_CONFIG = DatasetConfig(
    name="waterbirds",
    num_classes=2,
    class_names=["landbird", "waterbird"],
    class_to_idx={
        "landbird": 0,
        "waterbird": 1
    },
    idx_to_class={
        0: "landbird",
        1: "waterbird"
    },
    model_checkpoint="",  # Will use CLIP, no checkpoint needed
    transform_fn=create_waterbirds_transform  # Use the centralized transform
)

# Oxford-IIIT Pet Dataset Configuration (Binary: Cat vs Dog)
OXFORD_PETS_CONFIG = DatasetConfig(
    name="oxford_pets",
    num_classes=2,
    class_names=["cat", "dog"],
    class_to_idx={
        "cat": 0,
        "dog": 1
    },
    idx_to_class={
        0: "cat",
        1: "dog"
    },
    model_checkpoint="",  # Will use CLIP, no checkpoint needed
    transform_fn=create_oxford_pets_transform  # Use the centralized transform
)

# Dataset registry for easy access
DATASET_CONFIGS = {
    "covidquex": COVIDQUEX_CONFIG,
    "hyperkvasir": HYPERKVASIR_CONFIG,
    "waterbirds": WATERBIRDS_CONFIG,
    "oxford_pets": OXFORD_PETS_CONFIG,
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset ('covidquex' or 'hyperkvasir')
        
    Returns:
        DatasetConfig object for the specified dataset
        
    Raises:
        ValueError: If dataset_name is not recognized
    """
    if dataset_name.lower() not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    return DATASET_CONFIGS[dataset_name.lower()]
