import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

import vit.preprocessing as preprocessing


def get_ssl4gie_pil_transforms(split_type: str):
    """
    Get SSL4GIE transforms for PIL images (no tensor conversion/normalization).
    These are applied after resize(224,224) to match SSL4GIE methodology.
    
    Args:
        split_type: 'train' for augmentation, 'val'/'test' for no augmentation
    """
    if split_type == 'train':
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
        ])
    else:
        # No augmentation for val/test
        return transforms.Compose([])


def get_ssl4gie_tensor_transforms():
    """
    Get tensor conversion and normalization transforms.
    Apply these during training/inference, not during preprocessing.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


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


def create_class_directories(base_dir: Path, class_names: List[str]):
    """
    Create class subdirectories within the base directory.
    
    Args:
        base_dir: The split directory (train/val/test)
        class_names: List of class names (findings) to create directories for
    """
    for class_name in class_names:
        class_dir = base_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)


def preprocess_hyperkvasir_with_ssl4gie_transforms(
    config,
    csv_path: Path,
    source_dir: Path,
    classification_filter: str = "anatomical-landmarks",
    save_multiple_train_augmentations: bool = False,
    num_train_augmentations: int = 5
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Process HyperKvasir images using SSL4GIE transform methodology and save to train/val/test folders
    with class subdirectories.
    
    Args:
        config: Pipeline configuration
        csv_path: Path to the CSV file with image metadata
        source_dir: Directory containing the original Hyperkvasir image folders
        classification_filter: Only process images with this classification
        save_multiple_train_augmentations: Whether to save multiple augmented versions of training images
        num_train_augmentations: Number of augmented versions to save per training image
        
    Returns:
        Tuple of (train_paths, val_paths, test_paths)
    """

    # Create split directories
    train_dir = source_dir / "preprocessed" / "train"
    val_dir = source_dir / "preprocessed" / "val"
    test_dir = source_dir / "preprocessed" / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV and filter for specified classification
    df = pd.read_csv(csv_path)
    filtered_df = df[df['Classification'] == classification_filter].copy().reset_index(drop=True)

    print(f"Found {len(filtered_df)} images with classification '{classification_filter}'")
    print(f"Findings distribution:")
    print(filtered_df['Finding'].value_counts())

    # Get unique class names (findings) to create directories
    unique_classes = filtered_df['Finding'].unique().tolist()
    print(f"\nUnique classes found: {unique_classes}")

    # Create class directories in each split
    create_class_directories(train_dir, unique_classes)
    create_class_directories(val_dir, unique_classes)
    create_class_directories(test_dir, unique_classes)

    # Apply the same splits as SSL4GIE paper
    train_indices, test_indices, val_indices = split_ids(len(filtered_df))

    print(f"\nSplit sizes (matching SSL4GIE paper):")
    print(f"Train: {len(train_indices)} images")
    print(f"Val: {len(val_indices)} images")
    print(f"Test: {len(test_indices)} images")

    # Create split dataframes
    train_df = filtered_df.iloc[train_indices].copy()
    val_df = filtered_df.iloc[val_indices].copy()
    test_df = filtered_df.iloc[test_indices].copy()

    # Verify split distribution
    print(f"\nTrain split findings distribution:")
    print(train_df['Finding'].value_counts())

    # Process each split with SSL4GIE transforms
    train_paths = process_split_with_ssl4gie_methodology(
        train_df, source_dir, train_dir, "train", config, save_multiple_train_augmentations, num_train_augmentations
    )
    val_paths = process_split_with_ssl4gie_methodology(val_df, source_dir, val_dir, "val", config, False, 1)
    test_paths = process_split_with_ssl4gie_methodology(test_df, source_dir, test_dir, "test", config, False, 1)

    # Save split information for reference
    save_split_metadata(train_df, val_df, test_df, source_dir)

    # Save transform information
    save_ssl4gie_transform_info(source_dir)

    print(f"\n=== PREPROCESSING COMPLETE ===")
    print(f"Train: {len(train_paths)} images in {train_dir}")
    print(f"Val: {len(val_paths)} images in {val_dir}")
    print(f"Test: {len(test_paths)} images in {test_dir}")
    print(f"\nDirectory structure created:")
    print(f"  train/")
    for class_name in unique_classes:
        train_class_count = len([p for p in train_paths if class_name in str(p)])
        print(f"    ├── {class_name}/ ({train_class_count} images)")
    print(f"  val/")
    for class_name in unique_classes:
        val_class_count = len([p for p in val_paths if class_name in str(p)])
        print(f"    ├── {class_name}/ ({val_class_count} images)")
    print(f"  test/")
    for class_name in unique_classes:
        test_class_count = len([p for p in test_paths if class_name in str(p)])
        print(f"    ├── {class_name}/ ({test_class_count} images)")

    print(f"\nSSL4GIE Transform Methodology Applied:")
    print(f"  - All images: Resize to 224x224 first (matching SSL4GIE Dataset.__getitem__)")
    print(f"  - Train: ColorJitter + GaussianBlur + RandomFlips + RandomRotation(180°)")
    print(f"  - Val/Test: No augmentation (resize only)")
    print(f"  - Tensor conversion/normalization: Apply during training with provided transforms")

    return train_paths, val_paths, test_paths


def process_split_with_ssl4gie_methodology(
    split_df: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
    split_name: str,
    config,
    save_multiple_augmentations: bool = False,
    num_augmentations: int = 1
) -> List[Path]:
    """
    Process images using SSL4GIE methodology: resize first, then apply transforms.
    Save images in class-specific subdirectories.
    """
    processed_paths = []
    success_count = 0
    failed_files = []

    # Get SSL4GIE PIL transforms (no tensor conversion)
    ssl4gie_pil_transform = get_ssl4gie_pil_transforms(split_name)

    # Common image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    print(f"\nProcessing {split_name} split ({len(split_df)} images) with SSL4GIE methodology...")
    if split_name == 'train' and save_multiple_augmentations:
        print(f"  Saving {num_augmentations} augmented versions per training image")

    for idx, row in split_df.iterrows():
        video_file = row['Video file']
        finding = row['Finding']
        organ = row['Organ']

        # Create class-specific output directory path
        class_output_dir = output_dir / finding

        # Search for the original image file
        original_image_path = None
        for ext in extensions:
            search_pattern = f"{video_file}{ext}"
            matches = list(source_dir.rglob(search_pattern))
            if matches:
                original_image_path = matches[0]
                break

        if not original_image_path:
            failed_files.append(f"{video_file} ({finding})")
            continue

        try:
            # SSL4GIE methodology: Load and resize to 224x224 FIRST
            img = Image.open(original_image_path).convert('RGB')
            img_resized = img.resize((224, 224))  # This matches SSL4GIE Dataset.__getitem__

            if split_name == 'train' and save_multiple_augmentations:
                # Save multiple augmented versions for training data
                for aug_idx in range(num_augmentations):
                    new_filename = f"{video_file}_aug{aug_idx}.jpg"
                    output_path = class_output_dir / new_filename

                    if output_path.exists():
                        processed_paths.append(output_path)
                        continue

                    # Apply SSL4GIE transforms to the resized image
                    processed_img = ssl4gie_pil_transform(img_resized)
                    processed_img.save(output_path, quality=95)
                    processed_paths.append(output_path)

            else:
                # Save single version
                new_filename = f"{video_file}.jpg"
                output_path = class_output_dir / new_filename

                if output_path.exists():
                    processed_paths.append(output_path)
                    continue

                # Apply SSL4GIE transforms (for val/test, this is no-op)
                processed_img = ssl4gie_pil_transform(img_resized)
                processed_img.save(output_path, quality=95)
                processed_paths.append(output_path)

            success_count += 1
            if success_count % 50 == 0:
                print(f"  {split_name}: {success_count}/{len(split_df)} processed...")

        except Exception as e:
            failed_files.append(f"{video_file} ({finding}): {e}")
            continue

    print(f"{split_name} complete: {len(processed_paths)} files saved, {len(failed_files)} failed")

    if failed_files:
        print(f"  Failed files in {split_name}:")
        for fail in failed_files[:5]:
            print(f"    - {fail}")
        if len(failed_files) > 5:
            print(f"    ... and {len(failed_files) - 5} more")

    return processed_paths


def save_split_metadata(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path):
    """
    Save metadata about the splits for reproducibility and analysis.
    """
    # Add split column to each dataframe
    train_df_copy = train_df.copy()
    val_df_copy = val_df.copy()
    test_df_copy = test_df.copy()

    train_df_copy['split'] = 'train'
    val_df_copy['split'] = 'val'
    test_df_copy['split'] = 'test'

    # Save individual split files
    train_df_copy.to_csv(output_dir / "train_metadata.csv", index=False)
    val_df_copy.to_csv(output_dir / "val_metadata.csv", index=False)
    test_df_copy.to_csv(output_dir / "test_metadata.csv", index=False)

    # Save combined file
    all_splits_df = pd.concat([train_df_copy, val_df_copy, test_df_copy], ignore_index=True)
    all_splits_df.to_csv(output_dir / "all_splits_metadata.csv", index=False)

    # Save split statistics
    stats = {
        'total_images': len(all_splits_df),
        'train_count': len(train_df_copy),
        'val_count': len(val_df_copy),
        'test_count': len(test_df_copy),
        'train_findings': train_df_copy['Finding'].value_counts().to_dict(),
        'val_findings': val_df_copy['Finding'].value_counts().to_dict(),
        'test_findings': test_df_copy['Finding'].value_counts().to_dict()
    }

    import json
    with open(output_dir / "split_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved split metadata to {output_dir}/")


def save_ssl4gie_transform_info(output_dir: Path):
    """
    Save information about SSL4GIE transforms used for reproducibility.
    """
    transform_info = {
        "methodology": "SSL4GIE preprocessing with exact transform replication",
        "directory_structure":
        "Class-based subdirectories within each split (train/class_name/, val/class_name/, test/class_name/)",
        "preprocessing_transforms": {
            "all_splits": {
                "resize": "(224, 224) - applied first (matches SSL4GIE Dataset.__getitem__)"
            },
            "train_only": {
                "color_jitter": {
                    "brightness": 0.4,
                    "contrast": 0.5,
                    "saturation": 0.25,
                    "hue": 0.01
                },
                "gaussian_blur": {
                    "kernel_size": "(25, 25)",
                    "sigma": "(0.001, 2.0)"
                },
                "random_horizontal_flip": True,
                "random_vertical_flip": True,
                "random_rotation": 180
            },
            "val_test": {
                "augmentation": "None (resize only)"
            }
        },
        "runtime_transforms": {
            "tensor_conversion": "transforms.ToTensor()",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "note": "Apply during training/inference using get_ssl4gie_tensor_transforms()"
            }
        },
        "paper_reference": "SSL4GIE: Self-Supervised Learning for Gastrointestinal Image Enhancement",
        "split_method": "sklearn train_test_split with random_state=42",
        "split_ratios": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1
        }
    }

    import json
    with open(output_dir / "ssl4gie_preprocessing_info.json", 'w') as f:
        json.dump(transform_info, f, indent=2)


def run_hyperkvasir_ssl4gie_preprocessing(
    config,
    csv_path: Path,
    source_dir: Path,
    classification_filter: str = "anatomical-landmarks",
    save_multiple_train_augmentations: bool = False,
    num_train_augmentations: int = 5
):
    """
    Main function to preprocess HyperKvasir dataset using SSL4GIE transform methodology.
    Saves processed images to train/val/test folders with class subdirectories.
    
    Args:
        config: Pipeline configuration
        csv_path: Path to image-labels.csv
        source_dir: Path to HyperKvasir root directory
        classification_filter: Which class to process ('anatomical-landmarks' or 'pathological-findings')
        save_multiple_train_augmentations: Whether to save multiple augmented versions of training images
        num_train_augmentations: How many augmented versions to save per training image
    """
    print("Converting HyperKvasir dataset with SSL4GIE transform methodology...")
    print("Creating class-based directory structure within each split...")
    print("Using exact same splits as SSL4GIE paper (random_state=42)")
    print(f"Processing: {classification_filter}")

    if save_multiple_train_augmentations:
        print(f"Will save {num_train_augmentations} augmented versions per training image")

    train_paths, val_paths, test_paths = preprocess_hyperkvasir_with_ssl4gie_transforms(
        config, csv_path, source_dir, classification_filter, save_multiple_train_augmentations, num_train_augmentations
    )

    print(f"\nSSL4GIE-methodology dataset preprocessing complete!")
    print(f"Directory structure created with class folders:")
    print(f"   {source_dir}/")
    print(f"   ├── train/ ({len(train_paths)} images)")
    print(f"   │   ├── z-line/")
    print(f"   │   ├── retroflex-rectum/")
    print(f"   │   └── ... (other anatomical landmarks)")
    print(f"   ├── val/ ({len(val_paths)} images)")
    print(f"   │   ├── z-line/")
    print(f"   │   ├── retroflex-rectum/")
    print(f"   │   └── ... (other anatomical landmarks)")
    print(f"   └── test/ ({len(test_paths)} images)")
    print(f"       ├── z-line/")
    print(f"       ├── retroflex-rectum/")
    print(f"       └── ... (other anatomical landmarks)")

    print(f"\nFor training, apply tensor transforms:")
    print(f"   tensor_transform = get_ssl4gie_tensor_transforms()")
    print(f"   # This applies ToTensor() + ImageNet normalization")

    return train_paths, val_paths, test_paths


# Usage example
def main():
    """
    Example of how to use the SSL4GIE preprocessing with save-to-disk functionality
    and class-based directory structure
    """
    from config import PipelineConfig

    config = PipelineConfig()
    csv_path = Path("./hyper-kvasir/image-labels.csv")
    source_dir = Path("./hyper-kvasir/")

    run_hyperkvasir_ssl4gie_preprocessing(
        config,
        csv_path,
        source_dir,
        classification_filter="anatomical-landmarks",
        save_multiple_train_augmentations=True,
        num_train_augmentations=5
    )


if __name__ == "__main__":
    main()
