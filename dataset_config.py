"""
Dataset Configuration Module

This module defines the configuration for different datasets, providing a unified interface
for dataset-specific settings like class names, transforms, and model configurations.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path
import torchvision.transforms as transforms


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    
    name: str  # Dataset identifier
    num_classes: int  # Number of classes
    class_names: List[str]  # Original class names (for display/logging)
    class_to_idx: Dict[str, int]  # Mapping from original names to indices
    idx_to_class: Dict[int, str]  # Reverse mapping
    
    # Model configuration
    model_checkpoint: str  # Path to the trained model
    
    # Data paths (set during runtime)
    prepared_data_path: Optional[Path] = None
    
    # Transform configurations
    img_size: int = 224
    normalize_mean: List[float] = None
    normalize_std: List[float] = None
    
    def __post_init__(self):
        """Set default normalization values if not provided."""
        if self.normalize_mean is None:
            self.normalize_mean = [0.485, 0.456, 0.406]  # ImageNet defaults
        if self.normalize_std is None:
            self.normalize_std = [0.229, 0.224, 0.225]  # ImageNet defaults
    
    def get_transforms(self, split: str = 'test'):
        """Get transforms for a specific split (train/val/test/dev)."""
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])
        else:  # val, test, or dev
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
            ])


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
    model_checkpoint="./model/model_best.pth.tar"
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
    model_checkpoint="./model/vit_b-ImageNet_class_init-frozen_False-dataset_Hyperkvasir_anatomical.pth"
)


# Dataset registry for easy access
DATASET_CONFIGS = {
    "covidquex": COVIDQUEX_CONFIG,
    "hyperkvasir": HYPERKVASIR_CONFIG,
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