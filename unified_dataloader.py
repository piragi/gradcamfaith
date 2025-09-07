"""
Unified DataLoader Module

This module provides a simple, unified interface for loading medical imaging datasets
that have been converted to the standard format using dataset_converters.py.

Uses PyTorch's built-in ImageFolder for simplicity and reliability.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
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
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_clip: bool = False
    ):
        """
        Initialize the unified dataset loader.
        
        Args:
            data_path: Path to the prepared dataset (containing train/val/test folders)
            dataset_config: Configuration object for the dataset
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for CUDA
            use_clip: If True, use CLIP-specific preprocessing
        """
        self.data_path = Path(data_path)
        self.config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_clip = use_clip
        
        # Verify the data path exists and has the expected structure
        self._verify_data_structure()
        
        # Create datasets for each split
        self.datasets = {}
        self.dataloaders = {}
        
        # Check for all possible splits including dev
        for split in ['train', 'val', 'test', 'dev']:
            split_path = self.data_path / split
            if split_path.exists():
                # Get appropriate transforms for this split
                transform = self.config.get_transforms(split)
                
                # Create ImageFolder dataset
                dataset = ImageFolder(
                    root=split_path,
                    transform=transform
                )
                
                self.datasets[split] = dataset
                
                # Create DataLoader
                self.dataloaders[split] = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=(split == 'train'),
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=(split == 'train')  # Drop incomplete batches in training
                )
    
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
            
            if len(class_folders) != self.config.num_classes:
                print(
                    f"Warning: Expected {self.config.num_classes} classes in {split}, "
                    f"but found {len(class_folders)}"
                )
    
    def get_dataloader(self, split: str) -> DataLoader:
        """
        Get DataLoader for a specific split.
        
        Args:
            split: One of 'train', 'val', or 'test'
            
        Returns:
            DataLoader for the specified split
        """
        if split not in self.dataloaders:
            raise ValueError(f"Split '{split}' not available. Available: {list(self.dataloaders.keys())}")
        
        return self.dataloaders[split]
    
    def get_dataset(self, split: str) -> ImageFolder:
        """
        Get Dataset for a specific split.
        
        Args:
            split: One of 'train', 'val', or 'test'
            
        Returns:
            ImageFolder dataset for the specified split
        """
        if split not in self.datasets:
            raise ValueError(f"Split '{split}' not available. Available: {list(self.datasets.keys())}")
        
        return self.datasets[split]
    
    def get_num_samples(self, split: str) -> int:
        """Get the number of samples in a split."""
        if split in self.datasets:
            return len(self.datasets[split])
        return 0
    
    def get_class_counts(self, split: str) -> Dict[int, int]:
        """Get the number of samples per class in a split."""
        if split not in self.datasets:
            return {}
        
        class_counts = {i: 0 for i in range(self.config.num_classes)}
        for _, label in self.datasets[split].samples:
            class_counts[label] += 1
        return class_counts
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the dataset."""
        return {
            'dataset_name': self.config.name,
            'num_classes': self.config.num_classes,
            'class_names': self.config.class_names,
            'splits': {
                split: {
                    'total_samples': len(self.datasets[split]),
                    'class_distribution': {
                        self.config.idx_to_class[idx]: count 
                        for idx, count in self.get_class_counts(split).items()
                    }
                } for split in self.datasets
            }
        }


def create_dataloader(
    dataset_name: str,
    data_path: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    use_clip: bool = False,
    **kwargs
) -> UnifiedMedicalDataset:
    """
    Convenience function to create a dataloader for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset ('covidquex' or 'hyperkvasir')
        data_path: Path to the prepared dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        use_clip: If True, use CLIP-specific preprocessing
        **kwargs: Additional arguments passed to UnifiedMedicalDataset
        
    Returns:
        UnifiedMedicalDataset instance
    """
    config = get_dataset_config(dataset_name)
    return UnifiedMedicalDataset(
        data_path=data_path,
        dataset_config=config,
        batch_size=batch_size,
        num_workers=num_workers,
        use_clip=use_clip,
        **kwargs
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


# Example usage
if __name__ == "__main__":
    dataset = create_dataloader("covidquex", Path("./data/covidquex_unified"))
    stats = dataset.get_statistics()
    print(f"Dataset: {stats['dataset_name']}")
    print(f"Classes: {stats['num_classes']} - {stats['class_names']}")
    
    for split, split_stats in stats['splits'].items():
        print(f"{split}: {split_stats['total_samples']} samples")
        for class_name, count in split_stats['class_distribution'].items():
            print(f"  {class_name}: {count}")