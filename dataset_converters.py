"""
Dataset Converters Module

This module contains functions to convert various medical imaging datasets
into a unified format that can be consumed by the unified dataloader.

Standard format:
    data/
    ├── train/
    │   ├── class_0/
    │   │   ├── img_0001.jpg
    │   │   └── ...
    │   └── class_1/
    ├── val/
    └── test/
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dataset_config import DatasetConfig, COVIDQUEX_CONFIG, HYPERKVASIR_CONFIG
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
import random


def prepare_covidquex(
    source_path: Path,
    output_path: Path,
    config: DatasetConfig = COVIDQUEX_CONFIG
) -> Dict:
    """
    Convert CovidQUEX dataset to unified format.
    Images are preprocessed to 224x224 using resize(256) -> center crop(224).
    
    Expects exact source structure:
    - source_path/
        ├── Train/
        │   ├── COVID-19/
        │   │   └── images/
        │   │       ├── image1.png
        │   │       └── ...
        │   ├── Non-COVID/
        │   │   └── ...
        │   └── Normal/
        │   │   └── ...
        ├── Val/
        │   └── ...
        │       
        └── Test/
            └── ...
    
    Args:
        source_path: Path to source dataset (typically ./lung)
        output_path: Path where unified format will be created
        config: Dataset configuration
        
    Returns:
        Dictionary with conversion metadata
    """
    output_path = Path(output_path)
    source_path = Path(source_path)
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        for class_idx in range(config.num_classes):
            (output_path / split / f"class_{class_idx}").mkdir(parents=True, exist_ok=True)
    
    conversion_stats = {
        'dataset': 'covidquex',
        'total_images': 0,
        'splits': {'train': 0, 'val': 0, 'test': 0},
        'classes': {name: 0 for name in config.class_names},
        'class_mapping': config.class_to_idx
    }
    
    # Map source split names to our standard names
    split_mapping = {
        'Train': 'train',
        'Val': 'val',
        'Test': 'test'
    }
    
    # Process each split and class combination
    for source_split, target_split in split_mapping.items():
        split_dir = source_path / source_split
        if not split_dir.exists():
            raise ValueError(f"Expected split directory not found: {split_dir}")
        
        for class_name, class_idx in config.class_to_idx.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} not found")
                continue
            
            # Images are in the images subdirectory
            images_dir = class_dir / "images"
            images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            
            if not images:
                print(f"Warning: No images found in {class_dir}")
                continue
            
            print(f"Found {len(images)} images in {source_split}/{class_name}")
            
            for idx, img_path in enumerate(tqdm(images, desc=f"{source_split}/{class_name}")):
                new_name = f"img_{class_idx:02d}_{target_split}_{idx:05d}.png"  # Always save as PNG
                dest_path = output_path / target_split / f"class_{class_idx}" / new_name
                
                # Apply same preprocessing as get_default_processor: resize(256) -> center crop(224)
                img = Image.open(img_path).convert('RGB')
                # Resize so smaller dimension is 256
                w, h = img.size
                if w < h:
                    new_w = 256
                    new_h = int(h * 256 / w)
                else:
                    new_h = 256
                    new_w = int(w * 256 / h)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                # Center crop to 224x224
                left = (new_w - 224) // 2
                top = (new_h - 224) // 2
                right = left + 224
                bottom = top + 224
                img = img.crop((left, top, right, bottom))
                img.save(dest_path, 'PNG')
                
                conversion_stats['total_images'] += 1
                conversion_stats['splits'][target_split] += 1
                conversion_stats['classes'][class_name] += 1
    
    # Save metadata
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(conversion_stats, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total images: {conversion_stats['total_images']}")
    print(f"Splits: {conversion_stats['splits']}")
    print(f"Classes: {conversion_stats['classes']}")
    
    return conversion_stats


def split_ids(len_ids):
    """
    Reproduce the exact same split as in the SSL4GIE paper
    """
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))
    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )
    train_indices, val_indices = train_test_split(train_indices, test_size=test_size, random_state=42)
    return train_indices, test_indices, val_indices


def prepare_hyperkvasir(
    source_path: Path,
    output_path: Path,
    config: DatasetConfig = HYPERKVASIR_CONFIG,
    csv_path: Optional[Path] = None
) -> Dict:
    """
    Convert HyperKvasir dataset to unified format.
    Images are preprocessed to 224x224 using direct resize (SSL4GIE methodology).
    
    Uses the CSV file (image-labels.csv) to identify and organize images.
    
    Args:
        source_path: Path to source dataset (HyperKvasir root directory)
        output_path: Path where unified format will be created
        config: Dataset configuration
        csv_path: Path to image-labels.csv (if None, looks for it in source_path)
        
    Returns:
        Dictionary with conversion metadata
    """
    output_path = Path(output_path)
    source_path = Path(source_path)
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        for class_idx in range(config.num_classes):
            (output_path / split / f"class_{class_idx}").mkdir(parents=True, exist_ok=True)
    
    conversion_stats = {
        'dataset': 'hyperkvasir',
        'total_images': 0,
        'splits': {'train': 0, 'val': 0, 'test': 0},
        'classes': {name: 0 for name in config.class_names},
        'class_mapping': config.class_to_idx
    }
    
    # If csv_path not provided, look for it in source directory
    if csv_path is None:
        csv_path = source_path / "image-labels.csv"
    
    if not csv_path.exists():
        raise ValueError(f"CSV file not found at {csv_path}. Please provide the path to image-labels.csv")
    
    # Load and process the CSV
    df = pd.read_csv(csv_path)
    
    # Filter for anatomical landmarks
    df_filtered = df[df['Classification'] == 'anatomical-landmarks'].copy()
    
    # Filter for our specific classes
    df_filtered = df_filtered[df_filtered['Finding'].isin(config.class_names)]
    
    print(f"Found {len(df_filtered)} images for classes: {config.class_names}")
    
    # Create splits using SSL4GIE method
    # Apply the same splits as SSL4GIE paper
    train_idx, test_idx, val_idx = split_ids(len(df_filtered))
    
    # Process each split
    split_indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    
    for split, indices in split_indices.items():
        split_df = df_filtered.iloc[indices]
        
        for _, row in tqdm(split_df.iterrows(), desc=f"Processing {split}", total=len(split_df)):
            video_file = row['Video file']
            finding = row['Finding']
            
            # Map finding to class index
            class_idx = config.class_to_idx[finding]
            
            # Find the actual image file
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_path = source_path / f"{video_file}{ext}"
                if potential_path.exists():
                    img_path = potential_path
                    break
                # Also check in subdirectories
                matches = list(source_path.rglob(f"{video_file}{ext}"))
                if matches:
                    img_path = matches[0]
                    break
            
            if img_path is None:
                print(f"Warning: Could not find image for {video_file}")
                continue
            
            # Create new standardized name
            img_count = conversion_stats['splits'][split]
            new_name = f"img_{class_idx:02d}_{split}_{img_count:05d}{img_path.suffix}"
            dest_path = output_path / split / f"class_{class_idx}" / new_name
            
            # SSL4GIE methodology: resize directly to 224x224  
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            if split == 'train':
                # For training, create multiple augmented versions with fixed seeds
                # This matches the SSL4GIE training augmentations
                
                # Define the augmentation pipeline
                augment_transform = transforms.Compose([
                    transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01),
                    transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(180),
                ])
                
                # Save original
                img.save(dest_path, 'PNG')
                conversion_stats['total_images'] += 1
                conversion_stats['splits'][split] += 1
                conversion_stats['classes'][finding] += 1
                
                # Generate augmented versions with different seeds
                # We'll create 4 augmented versions per image to get ~16k total
                num_augmentations = 4
                for aug_idx in range(num_augmentations):
                    # Set seed for reproducibility (base_seed + image_index + aug_index)
                    seed = 42 + img_count * 10 + aug_idx
                    torch.manual_seed(seed)
                    random.seed(seed)
                    np.random.seed(seed)
                    
                    # Convert PIL to tensor, apply augmentations, convert back
                    img_tensor = transforms.ToTensor()(img)
                    # Add batch dimension for transforms
                    img_tensor = img_tensor.unsqueeze(0)
                    
                    # Apply augmentations (need to convert back from tensor)
                    augmented_pil = augment_transform(img)
                    
                    # Save augmented image
                    aug_count = conversion_stats['splits'][split] + aug_idx + 1
                    aug_name = f"img_{class_idx:02d}_{split}_{aug_count:05d}_aug{aug_idx}.png"
                    aug_dest = output_path / split / f"class_{class_idx}" / aug_name
                    augmented_pil.save(aug_dest, 'PNG')
                    
                    conversion_stats['total_images'] += 1
                    conversion_stats['splits'][split] += 1
                    conversion_stats['classes'][finding] += 1
            else:
                # For val/test, just save the original
                img.save(dest_path, 'PNG')
                conversion_stats['total_images'] += 1
                conversion_stats['splits'][split] += 1
                conversion_stats['classes'][finding] += 1
    
    # Save metadata
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(conversion_stats, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total images: {conversion_stats['total_images']}")
    print(f"Splits: {conversion_stats['splits']}")
    print(f"Classes: {conversion_stats['classes']}")
    
    return conversion_stats


def convert_dataset(
    dataset_name: str,
    source_path: Path,
    output_path: Path,
    **kwargs
) -> Dict:
    """
    Main entry point for dataset conversion.
    
    Args:
        dataset_name: Name of the dataset ('covidquex' or 'hyperkvasir')
        source_path: Path to source dataset
        output_path: Path where unified format will be created
        **kwargs: Additional arguments passed to specific converter
        
    Returns:
        Dictionary with conversion metadata
    """
    converters = {
        'covidquex': prepare_covidquex,
        'hyperkvasir': prepare_hyperkvasir,
    }
    
    if dataset_name.lower() not in converters:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(converters.keys())}")
    
    converter_func = converters[dataset_name.lower()]
    return converter_func(source_path, output_path, **kwargs)
