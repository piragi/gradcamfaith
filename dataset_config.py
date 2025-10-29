"""
Dataset Configuration Module

This module defines the configuration for different datasets, providing a unified interface
for dataset-specific settings like class names, transforms, and model configurations.

IMPORTANT: All preprocessing is centralized here through callable transforms.
This is the SINGLE SOURCE OF TRUTH for all dataset preprocessing.
"""

import json
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
    """CovidQUEX transforms: Resize(256) + CenterCrop(224) + custom normalization."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.56, 0.56, 0.56], std=[0.21, 0.21, 0.21])
    ])


def create_hyperkvasir_transform(split: str = 'test') -> transforms.Compose:
    """HyperKvasir transforms: ImageNet normalization + training augmentations."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    base_transforms = [
        transforms.Resize((224, 224), interpolation=BILINEAR),
    ]

    if split == 'train':
        base_transforms.extend([
            transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
        ])

    base_transforms.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    return transforms.Compose(base_transforms)


def create_waterbirds_transform(_) -> transforms.Compose:
    """Waterbirds transforms: CLIP preprocessing."""
    from vit_prisma.transforms import get_clip_val_transforms
    return get_clip_val_transforms()


def create_imagenet_transform(_) -> transforms.Compose:
    """ImageNet transforms: CLIP preprocessing (same as waterbirds)."""
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

    def get_transforms(self, split: str = 'test'):
        """Get transforms for a specific split (train/val/test/dev)."""
        if self.transform_fn is not None:
            return self.transform_fn(split)
        raise ValueError(f"No transform function defined for dataset {self.name}")


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
    transform_fn=create_covidquex_transform
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
    transform_fn=create_hyperkvasir_transform
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
    model_checkpoint="",
    transform_fn=create_waterbirds_transform
)

_IMAGENET_CONFIG = None  # module-level cache


def _create_imagenet_config() -> "DatasetConfig":
    """
    Build ImageNet-1k config using saved class_names.json if present,
    otherwise fall back to placeholders (only used until refresh).
    """
    class_names_file = Path("./data/imagenet/raw/class_names.json")
    if class_names_file.exists():
        with open(class_names_file, "r") as f:
            class_names = json.load(f)
        print(f"✓ Loaded {len(class_names)} ImageNet class names from {class_names_file}")
    else:
        # Quiet fallback; we'll refresh after download
        class_names = [f"class_{i}" for i in range(1000)]

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for i, name in enumerate(class_names)}

    return DatasetConfig(
        name="imagenet",
        num_classes=1000,
        class_names=class_names,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        model_checkpoint="",
        transform_fn=create_imagenet_transform,
    )


def get_imagenet_config() -> "DatasetConfig":
    """Return cached ImageNet config (build once, then reuse)."""
    global _IMAGENET_CONFIG
    if _IMAGENET_CONFIG is None:
        _IMAGENET_CONFIG = _create_imagenet_config()
    return _IMAGENET_CONFIG


def refresh_imagenet_config() -> None:
    """Rebuild the cached config (call after download writes class_names.json)."""
    global _IMAGENET_CONFIG
    _IMAGENET_CONFIG = _create_imagenet_config()


# Dataset registry for easy access
DATASET_CONFIGS = {
    "covidquex": COVIDQUEX_CONFIG,
    "hyperkvasir": HYPERKVASIR_CONFIG,
    "waterbirds": WATERBIRDS_CONFIG,
    "imagenet": get_imagenet_config,
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    Return a concrete DatasetConfig for the given dataset.
    Supports both direct objects and callables (lazy factories).
    """
    key = dataset_name.lower()
    if key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    cfg_or_factory = DATASET_CONFIGS[key]
    return cfg_or_factory() if callable(cfg_or_factory) else cfg_or_factory
