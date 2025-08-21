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
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset_config import COVIDQUEX_CONFIG, HYPERKVASIR_CONFIG, DatasetConfig


def prepare_covidquex(source_path: Path, output_path: Path, config: DatasetConfig = COVIDQUEX_CONFIG) -> Dict:
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
        'splits': {
            'train': 0,
            'val': 0,
            'test': 0
        },
        'classes': {
            name: 0
            for name in config.class_names
        },
        'class_mapping': config.class_to_idx
    }

    # Map source split names to our standard names
    split_mapping = {'Train': 'train', 'Val': 'val', 'Test': 'test'}

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
        'splits': {
            'train': 0,
            'val': 0,
            'test': 0
        },
        'classes': {
            name: 0
            for name in config.class_names
        },
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


def prepare_waterbirds(source_path: Path, output_path: Path, config: Optional['DatasetConfig'] = None) -> Dict:
    """
    Convert Waterbirds dataset to unified format.
    
    The Waterbirds dataset consists of bird images on different backgrounds.
    Dataset structure expected:
    - waterbird_complete95_forest2water2/
        ├── metadata.csv (contains file paths, labels, groups, and splits)
        └── (image files referenced in metadata)
    
    Args:
        source_path: Path to Waterbirds dataset directory (waterbird_complete95_forest2water2)
        output_path: Path where unified format will be created
        config: Dataset configuration (will be created if not provided)
        
    Returns:
        Dictionary with conversion metadata
    """
    output_path = Path(output_path)
    source_path = Path(source_path)

    # If config not provided, get it from dataset_config
    if config is None:
        from dataset_config import WATERBIRDS_CONFIG
        config = WATERBIRDS_CONFIG

    # Create output directory structure
    for split in ['train', 'val', 'test']:
        for class_idx in range(config.num_classes):
            (output_path / split / f"class_{class_idx}").mkdir(parents=True, exist_ok=True)

    conversion_stats = {
        'dataset': 'waterbirds',
        'total_images': 0,
        'splits': {
            'train': 0,
            'val': 0,
            'test': 0
        },
        'classes': {
            name: 0
            for name in config.class_names
        },
        'groups': {
            'landbird_land': 0,
            'landbird_water': 0,
            'waterbird_land': 0,
            'waterbird_water': 0
        },
        'class_mapping': config.class_to_idx
    }

    # Read metadata file
    metadata_path = source_path / "metadata.csv"
    if not metadata_path.exists():
        raise ValueError(f"Metadata file not found at {metadata_path}")

    df = pd.read_csv(metadata_path)
    print(f"Found {len(df)} images in metadata")

    # The metadata contains:
    # - img_filename: path to the image file
    # - y: target label (0 for landbird, 1 for waterbird)
    # - place: background type (0 for land, 1 for water)
    # - split: 0=train, 1=val, 2=test

    split_mapping = {0: 'train', 1: 'val', 2: 'test'}

    # Process each image
    for _, row in tqdm(df.iterrows(), desc="Processing Waterbirds", total=len(df)):
        img_filename = row['img_filename']
        y = int(row['y'])  # 0=landbird, 1=waterbird
        place = int(row['place'])  # 0=land, 1=water
        split_idx = int(row['split'])

        if split_idx not in split_mapping:
            print(f"Warning: Unknown split index {split_idx}, skipping")
            continue

        split = split_mapping[split_idx]
        class_name = config.idx_to_class[y]

        # Track group statistics
        group_name = f"{class_name}_{'land' if place == 0 else 'water'}"
        conversion_stats['groups'][group_name] += 1

        # Find the image file
        img_path = source_path / img_filename
        if not img_path.exists():
            print(f"Warning: Image not found at {img_path}")
            continue

        # Create new standardized name
        img_count = conversion_stats['splits'][split]
        new_name = f"img_{y:02d}_{split}_{img_count:05d}.jpg"
        dest_path = output_path / split / f"class_{y}" / new_name

        # Process and save image
        img = Image.open(img_path).convert('RGB')
        # Resize to 224x224 (standard for vision models)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img.save(dest_path, 'JPEG')

        conversion_stats['total_images'] += 1
        conversion_stats['splits'][split] += 1
        conversion_stats['classes'][class_name] += 1

    # Save metadata
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(conversion_stats, f, indent=2)

    print(f"\nConversion complete!")
    print(f"Total images: {conversion_stats['total_images']}")
    print(f"Splits: {conversion_stats['splits']}")
    print(f"Classes: {conversion_stats['classes']}")
    print(f"Groups: {conversion_stats['groups']}")

    return conversion_stats


def prepare_oxford_pets(
    source_path: Path,
    output_path: Path,
    config: Optional['DatasetConfig'] = None,
    splits: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Prepare Oxford-IIIT Pet dataset for unified pipeline.
    
    Args:
        source_path: Path to the extracted Oxford pets directory
        output_path: Path where the prepared dataset should be saved
        config: Optional dataset configuration
        splits: Optional dictionary with train/val/test split ratios
        
    Returns:
        Dictionary with dataset statistics
    """
    print(f"Preparing Oxford-IIIT Pet dataset from {source_path}")

    # Default splits if not provided
    if splits is None:
        splits = {"train": 0.7, "val": 0.15, "test": 0.15}

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    for split in splits.keys():
        for class_name in ["class_0", "class_1"]:  # cat, dog
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)

    # Read annotations
    trainval_file = source_path / "annotations" / "trainval.txt"
    test_file = source_path / "annotations" / "test.txt"

    if not trainval_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Annotation files not found in {source_path / 'annotations'}")

    # Parse annotations
    def parse_annotations(file_path):
        """Parse Oxford pets annotation file."""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    class_id = int(parts[1])  # 1-37 for breeds
                    species = int(parts[2])  # 1=cat, 2=dog
                    data.append({
                        'image': f"{image_name}.jpg",
                        'breed_id': class_id,
                        'species': species - 1  # Convert to 0-indexed (0=cat, 1=dog)
                    })
        return data

    # Load all annotations
    trainval_data = parse_annotations(trainval_file)
    test_data = parse_annotations(test_file)

    # Split trainval into train and val
    import random
    random.seed(42)
    random.shuffle(trainval_data)

    train_size = int(len(trainval_data) * (splits["train"] / (splits["train"] + splits["val"])))
    train_data = trainval_data[:train_size]
    val_data = trainval_data[train_size:]

    # Organize data by split
    split_data = {"train": train_data, "val": val_data, "test": test_data}

    # Copy images to output directories
    images_dir = source_path / "images"
    conversion_stats = {"total_images": 0, "splits": {"train": 0, "val": 0, "test": 0}, "classes": {"cat": 0, "dog": 0}}

    for split_name, data in split_data.items():
        for item in data:
            src_image = images_dir / item['image']
            if src_image.exists():
                # Determine target class directory (0=cat, 1=dog)
                class_dir = f"class_{item['species']}"
                class_name = "cat" if item['species'] == 0 else "dog"

                # Load, resize to 224x224, and save image
                dst_image = output_path / split_name / class_dir / f"img_{item['species']:02d}_{split_name}_{conversion_stats['splits'][split_name]:05d}.jpg"

                import shutil
                shutil.copy2(src_image, dst_image)
                # Open image, convert to RGB, resize to 224x224
                # from PIL import Image
                # img = Image.open(src_image).convert('RGB')
                # img = img.resize((224, 224), Image.Resampling.LANCZOS)
                # img.save(dst_image, 'JPEG')

                conversion_stats["total_images"] += 1
                conversion_stats["splits"][split_name] += 1
                conversion_stats["classes"][class_name] += 1

    # Save metadata
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(conversion_stats, f, indent=2)

    print(f"\nConversion complete!")
    print(f"Total images: {conversion_stats['total_images']}")
    print(f"Splits: {conversion_stats['splits']}")
    print(f"Classes: {conversion_stats['classes']}")

    return conversion_stats


def convert_dataset(dataset_name: str, source_path: Path, output_path: Path, **kwargs) -> Dict:
    """
    Main entry point for dataset conversion.
    
    Args:
        dataset_name: Name of the dataset ('covidquex', 'hyperkvasir', or 'waterbirds')
        source_path: Path to source dataset
        output_path: Path where unified format will be created
        **kwargs: Additional arguments passed to specific converter
        
    Returns:
        Dictionary with conversion metadata
    """
    converters = {
        'covidquex': prepare_covidquex,
        'hyperkvasir': prepare_hyperkvasir,
        'waterbirds': prepare_waterbirds,
        'oxford_pets': prepare_oxford_pets,
    }

    if dataset_name.lower() not in converters:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(converters.keys())}")

    converter_func = converters[dataset_name.lower()]
    return converter_func(source_path, output_path, **kwargs)
