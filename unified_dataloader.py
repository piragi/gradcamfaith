"""
Unified DataLoader Module

This module provides a simple, unified interface for loading medical imaging datasets
that have been converted to the standard format using dataset_converters.py.

Uses PyTorch's built-in ImageFolder for simplicity and reliability.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torchvision.datasets import ImageFolder
from PIL import Image

from dataset_config import DatasetConfig, get_dataset_config


class UnifiedMedicalDataset:
    """
    Unified dataset loader for medical imaging datasets.
    
    This class provides a simple interface to load datasets that have been
    converted to the standard format (class folders within train/val/test).
    """
    
    def __init__(
        self,
        data_path: Path,
        dataset_config: DatasetConfig,
    ):
        """
        Initialize the unified dataset loader.
        
        Args:
            data_path: Path to the prepared dataset (containing train/val/test folders)
            dataset_config: Configuration object for the dataset
        """
        self.data_path = Path(data_path)
        self.config = dataset_config
        # Verify the data path exists and has the expected structure
        self._verify_data_structure()
        
        # Build cached metadata for each split
        self.numeric_samples: Dict[str, List[Tuple[Path, int]]] = {}

        # Check for all possible splits including dev
        for split in ['train', 'val', 'test', 'dev']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue

            # Skip splits that contain no images (placeholder folders)
            if not any(split_path.glob("class_*/*")):
                continue

            # Special-case: ImageNet test split is unlabeled; use class_-1 only
            if self.config.name == 'imagenet' and split == 'test':
                cls_dir = split_path / 'class_-1'
                files = []
                for pattern in ('*.JPEG','*.jpeg','*.jpg','*.png'):
                    files.extend(sorted(cls_dir.glob(pattern)))
                self.numeric_samples[split] = [(f, -1) for f in files]
            else:
                dataset = ImageFolder(
                    root=split_path,
                    transform=self.config.get_transforms(split)
                )

                self.numeric_samples[split] = self._build_numeric_samples(dataset)
    
    def _verify_data_structure(self):
        """Verify that the data path has the expected structure."""
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        # Check for at least one split
        splits = ['train', 'val', 'test', 'dev']
        existing_splits = [s for s in splits if (self.data_path / s).exists()]
        
        if not existing_splits:
            raise ValueError(
                f"No valid splits found in {self.data_path}. "
                f"Expected at least one of: {splits}"
            )
        
        # Verify class folders
        for split in existing_splits:
            split_path = self.data_path / split
            class_folders = list(split_path.glob("class_*"))
            # Allow ImageNet test to include an extra 'class_-1' folder
            effective_count = len(class_folders)
            if self.config.name == 'imagenet' and split == 'test' and any(f.name == 'class_-1' for f in class_folders):
                effective_count -= 1

            if effective_count != self.config.num_classes:
                print(
                    f"Warning: Expected {self.config.num_classes} classes in {split}, "
                    f"but found {len(class_folders)}"
                )
    
    def _parse_class_index(self, class_folder: str) -> int:
        """Extract integer class index from a folder name (expects class_<idx>)."""
        try:
            return int(class_folder.split('_')[-1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Unable to parse class index from folder name {class_folder!r}") from exc

    def _build_numeric_samples(self, dataset: ImageFolder) -> List[Tuple[Path, int]]:
        """Convert dataset samples to use numeric labels derived from folder names."""
        numeric_samples: List[Tuple[Path, int]] = []
        for path_str, _ in dataset.samples:
            sample_path = Path(path_str)
            label_idx = self._parse_class_index(sample_path.parent.name)
            numeric_samples.append((sample_path, label_idx))
        return numeric_samples

    def get_numeric_samples(self, split: str) -> List[Tuple[Path, int]]:
        """Return samples with numeric class indices derived from folder names."""
        if split not in self.numeric_samples:
            raise ValueError(f"Split '{split}' not available. Available: {list(self.numeric_samples.keys())}")
        return list(self.numeric_samples[split])

    
    
    def get_num_samples(self, split: str) -> int:
        """Get the number of samples in a split."""
        samples = self.numeric_samples.get(split)
        return len(samples) if samples is not None else 0

    def get_class_counts(self, split: str) -> Dict[int, int]:
        """Get the number of samples per class in a split."""
        samples = self.numeric_samples.get(split)
        if samples is None:
            return {}

        class_counts = {i: 0 for i in range(self.config.num_classes)}
        for _, label_idx in samples:
            class_counts[label_idx] += 1
        return class_counts

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the dataset."""
        return {
            'dataset_name': self.config.name,
            'num_classes': self.config.num_classes,
            'class_names': self.config.class_names,
            'splits': {
                split: {
                    'total_samples': len(samples),
                    'class_distribution': {
                        self.config.idx_to_class[idx]: count
                        for idx, count in self.get_class_counts(split).items()
                    }
                }
                for split, samples in self.numeric_samples.items()
            }
        }


def create_dataloader(
    dataset_name: str,
    data_path: Path
) -> UnifiedMedicalDataset:
    """
    Convenience helper to build the unified dataset wrapper for a given dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'covidquex', 'hyperkvasir', 'imagenet')
        data_path: Path to the prepared dataset directory

    Returns:
        UnifiedMedicalDataset instance
    """
    config = get_dataset_config(dataset_name)
    return UnifiedMedicalDataset(
        data_path=data_path,
        dataset_config=config
    )


def get_single_image_loader(
    image_path: Path,
    dataset_config: DatasetConfig,
    use_clip: bool = False
) -> torch.Tensor:
    """
    Load and preprocess a single image for inference.
    
    Args:
        image_path: Path to the image
        dataset_config: Configuration for preprocessing
        use_clip: If True, use CLIP-specific preprocessing
        
    Returns:
        Preprocessed image tensor ready for model input
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply test transforms
    transform = dataset_config.get_transforms('test')
    image_tensor = transform(image)
    
    # Add batch dimension
    return image_tensor.unsqueeze(0)
