"""
Unified Data Setup Module

This module handles both dataset downloads and conversion to unified format.
Combines functionality from setup.py and dataset_converters.py.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import gdown
import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset_config import COVIDQUEX_CONFIG, HYPERKVASIR_CONFIG, DatasetConfig


def download_with_progress(url: str, filename: Path) -> None:
    """Download file with progress bar using requests."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='')
        print()  # New line after download
    except Exception as e:
        raise


def download_from_gdrive(file_id: str, output_path: Path, description: str) -> None:
    """Download file from Google Drive using gdown."""
    if output_path.exists():
        return

    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(output_path), quiet=False)
    except Exception as e:
        raise


def extract_zip(zip_path: Path, extract_to: Path, remove_after: bool = True) -> None:
    """Extract zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        if remove_after:
            zip_path.unlink()
    except Exception as e:
        raise


def extract_tar_gz(tar_path: Path, extract_to: Path, remove_after: bool = True) -> None:
    """Extract tar.gz file."""
    try:
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
        if remove_after:
            tar_path.unlink()
    except Exception as e:
        raise


def download_hyperkvasir(data_dir: Path, models_dir: Path) -> None:
    """Download Hyperkvasir dataset and model."""
    print("\nDownloading Hyperkvasir...")

    # Create hyperkvasir subdirectories
    hk_data_dir = data_dir / "hyperkvasir"
    hk_models_dir = models_dir / "hyperkvasir"
    hk_data_dir.mkdir(exist_ok=True, parents=True)
    hk_models_dir.mkdir(exist_ok=True, parents=True)

    # Download Hyperkvasir dataset
    dataset_url = "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip"
    dataset_path = hk_data_dir / "hyper-kvasir-labeled-images.zip"

    if not dataset_path.exists():
        try:
            # Try wget first (faster for large files)
            subprocess.run(["wget", "-O", str(dataset_path), dataset_url], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to requests if wget is not available
            download_with_progress(dataset_url, dataset_path)

    # Extract dataset
    extracted_dir = hk_data_dir / "labeled-images"
    if not extracted_dir.exists():
        extract_zip(dataset_path, hk_data_dir)

    # Download Hyperkvasir model
    model_info = {
        "name": "hyperkvasir_vit_model.pth",
        "id": "1gT4Z0qD09ClPOcfgXo0rMAsjvoD-tpDE",
        "description": "ViT model for Hyperkvasir"
    }

    output_path = hk_models_dir / model_info["name"]
    download_from_gdrive(model_info["id"], output_path, model_info["description"])


def download_waterbirds(data_dir: Path, models_dir: Path) -> None:
    """Download Waterbirds dataset."""
    print("\nDownloading Waterbirds...")

    # Create waterbirds subdirectories
    wb_data_dir = data_dir / "waterbirds"
    wb_models_dir = models_dir / "waterbirds"
    wb_data_dir.mkdir(exist_ok=True, parents=True)
    wb_models_dir.mkdir(exist_ok=True, parents=True)

    # Download Waterbirds dataset
    # The dataset is available from the group_DRO repository
    dataset_url = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
    dataset_path = wb_data_dir / "waterbird_complete95_forest2water2.tar.gz"

    if not dataset_path.exists():
        print(f"Downloading Waterbirds dataset from {dataset_url}")
        try:
            # Try wget first (faster for large files)
            subprocess.run(["wget", "-O", str(dataset_path), dataset_url], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to requests if wget is not available
            download_with_progress(dataset_url, dataset_path)

    # Extract dataset
    extracted_dir = wb_data_dir / "waterbird_complete95_forest2water2"
    if not extracted_dir.exists():
        print("Extracting Waterbirds dataset...")
        extract_tar_gz(dataset_path, wb_data_dir)
        print(f"Dataset extracted to {extracted_dir}")

    # Download CLIP model (will be cached by transformers library)
    # We don't need to download it manually, but we'll check if transformers is installed
    try:
        import transformers
        print("✓ Transformers library available for CLIP model loading")
    except ImportError:
        print("⚠ Warning: transformers library not installed. Install with: pip install transformers")


def download_covidquex(data_dir: Path, models_dir: Path) -> None:
    """Download CovidQueX dataset and model."""
    print("\nDownloading CovidQueX...")

    # Create covidquex subdirectories
    cq_data_dir = data_dir / "covidquex"
    cq_models_dir = models_dir / "covidquex"
    cq_data_dir.mkdir(exist_ok=True, parents=True)
    cq_models_dir.mkdir(exist_ok=True, parents=True)

    # Download CovidQueX dataset
    dataset_info = {
        "name": "covidquex_data.tar.gz",
        "id": "1XrCWP3ICQvurchnJjyVweYy2jHQM0BHO",
        "description": "CovidQueX dataset (tar.gz)"
    }

    dataset_path = cq_data_dir / dataset_info["name"]
    download_from_gdrive(dataset_info["id"], dataset_path, dataset_info["description"])

    # Extract dataset if it's a tar.gz file
    if dataset_path.suffix == '.gz' and dataset_path.exists():
        extracted_dir = cq_data_dir / "extracted"
        if not extracted_dir.exists():
            extract_tar_gz(dataset_path, cq_data_dir)

    # Download CovidQueX model
    model_info = {
        "name": "covidquex_model.pth",
        "id": "1JZM5ZRncaV3iFX9L6NFT1P0-APyHbBV0",
        "description": "CovidQueX model"
    }

    output_path = cq_models_dir / model_info["name"]
    download_from_gdrive(model_info["id"], output_path, model_info["description"])

    # Check if the downloaded model is a tar.gz archive and extract if needed
    if output_path.exists():
        import shutil

        # Check if file is a tar/gzip archive
        try:
            with open(output_path, 'rb') as f:
                magic = f.read(2)
                if magic == b'\x1f\x8b':  # gzip magic number - it's a tar.gz file
                    print(f"Extracting compressed model archive: {output_path}")

                    # First extract the tar.gz file
                    with tarfile.open(output_path, 'r:gz') as tar:
                        tar.extractall(cq_models_dir)

                    # Now extract model_best.pth.tar
                    model_tar_path = cq_models_dir / "results_model" / "model_best.pth.tar"
                    if model_tar_path.exists():
                        print(f"Extracting model from: {model_tar_path}")

                        # Extract the actual model from the .pth.tar file
                        # The .pth.tar file contains the actual PyTorch model state dict
                        final_model_path = cq_models_dir / "covidquex_model.pth"

                        # Copy the .pth.tar file directly as the model
                        # (PyTorch can load .pth.tar files directly)
                        shutil.copy(str(model_tar_path), str(final_model_path))
                        print(f"Model copied to: {final_model_path}")

                        # Clean up - remove the original tar.gz file and extracted directories
                        results_dir = cq_models_dir / "results_model"
                        if results_dir.exists():
                            shutil.rmtree(results_dir)
                    else:
                        print(f"Warning: Expected model file not found at {model_tar_path}")
        except Exception as e:
            print(f"Warning: Could not check/extract model file: {e}")


def download_sae_checkpoints(data_dir: Path) -> None:
    """Download SAE checkpoints from HuggingFace for all layers."""
    print("\n" + "=" * 50)
    print("Downloading SAE Checkpoints from HuggingFace")
    print("=" * 50)
    
    # Map each layer to its specific repository
    layer_repo_map = {
        1: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_1-hook_resid_post-64-82",
        2: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_2-hook_resid_post-64-80",
        3: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_3-hook_resid_post-64-80",
        4: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_4-hook_resid_post-64-80",
        5: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_5-hook_resid_post-64-81",
        6: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_6-hook_resid_post-64-81",
        7: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_7-hook_resid_post-64-83",
        8: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_8-hook_resid_post-64-84",
        9: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_9-hook_resid_post-64-86",
        10: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_10-hook_resid_post-64-85"
    }
    
    sae_base_dir = data_dir / "sae_waterbirds_clip_b32"
    successful_layers = []
    
    for layer_num, repo_id in layer_repo_map.items():
        print(f"Downloading Layer {layer_num} SAE checkpoint...")
        
        try:
            downloaded_path = hf_hub_download(repo_id=repo_id, filename="weights.pt", cache_dir="./hf_cache")
            target_dir = sae_base_dir / f"layer_{layer_num}"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy(downloaded_path, target_dir / "weights.pt")
            
            # Download config if available
            try:
                config_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir="./hf_cache")
                shutil.copy(config_path, target_dir / "config.json")
            except:
                pass
            
            successful_layers.append(layer_num)
            print(f"✓ Layer {layer_num} complete")
            
        except Exception as e:
            print(f"✗ Layer {layer_num} failed: {str(e)}")
    
    print(f"\nDownloaded {len(successful_layers)}/10 SAE layers to {sae_base_dir}")


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def _create_output_structure(output_path: Path, num_classes: int) -> None:
    """Create standard output directory structure for all datasets."""
    for split in ['train', 'val', 'test']:
        for class_idx in range(num_classes):
            (output_path / split / f"class_{class_idx}").mkdir(parents=True, exist_ok=True)


def _create_conversion_stats(dataset_name: str, config: DatasetConfig) -> Dict:
    """Create initial conversion statistics dictionary."""
    return {
        'dataset': dataset_name,
        'total_images': 0,
        'splits': {'train': 0, 'val': 0, 'test': 0},
        'classes': {name: 0 for name in config.class_names},
        'class_mapping': config.class_to_idx
    }


def _save_metadata(output_path: Path, stats: Dict) -> None:
    """Save conversion metadata to JSON file."""
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total images: {stats['total_images']}")
    print(f"Splits: {stats['splits']}")
    print(f"Classes: {stats['classes']}")


def _process_image(img_path: Path, dest_path: Path, stats: Dict, split: str, class_name: str, copy_only: bool = False) -> bool:
    """Process and save a single image, updating statistics. Returns True if successful."""
    try:
        if img_path.stat().st_size == 0:
            print(f"Warning: Skipping empty file: {img_path}")
            return False
        
        if copy_only:
            shutil.copy2(img_path, dest_path)
        else:
            img = Image.open(img_path).convert('RGB')
            img.save(dest_path, 'PNG')
        
        stats['total_images'] += 1
        stats['splits'][split] += 1
        stats['classes'][class_name] += 1
        return True
    except Exception as e:
        print(f"Warning: Failed to process {img_path}: {e}")
        return False


def split_ids(len_ids):
    """Reproduce the exact same split as in the SSL4GIE paper."""
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


def prepare_covidquex(source_path: Path, output_path: Path, config: DatasetConfig = COVIDQUEX_CONFIG) -> Dict:
    """Convert CovidQUEX dataset to unified format."""
    output_path, source_path = Path(output_path), Path(source_path)
    
    _create_output_structure(output_path, config.num_classes)
    conversion_stats = _create_conversion_stats('covidquex', config)
    
    split_mapping = {'Train': 'train', 'Val': 'val', 'Test': 'test'}
    
    for source_split, target_split in split_mapping.items():
        split_dir = source_path / source_split
        if not split_dir.exists():
            raise ValueError(f"Expected split directory not found: {split_dir}")

        for class_name, class_idx in config.class_to_idx.items():
            images_dir = split_dir / class_name / "images"
            if not images_dir.exists():
                print(f"Warning: Images directory {images_dir} not found")
                continue
                
            images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            if not images:
                continue
                
            print(f"Found {len(images)} images in {source_split}/{class_name}")
            
            for idx, img_path in enumerate(tqdm(images, desc=f"{source_split}/{class_name}")):
                new_name = f"img_{class_idx:02d}_{target_split}_{idx:05d}.png"
                dest_path = output_path / target_split / f"class_{class_idx}" / new_name
                _process_image(img_path, dest_path, conversion_stats, target_split, class_name)

    _save_metadata(output_path, conversion_stats)
    return conversion_stats


def prepare_hyperkvasir(
    source_path: Path,
    output_path: Path,
    config: DatasetConfig = HYPERKVASIR_CONFIG,
    csv_path: Optional[Path] = None
) -> Dict:
    """Convert HyperKvasir dataset to unified format using CSV metadata."""
    output_path, source_path = Path(output_path), Path(source_path)
    
    _create_output_structure(output_path, config.num_classes)
    conversion_stats = _create_conversion_stats('hyperkvasir', config)
    
    # Load CSV metadata
    csv_path = csv_path or source_path / "image-labels.csv"
    if not csv_path.exists():
        raise ValueError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    df_filtered = df[
        (df['Classification'] == 'anatomical-landmarks') & 
        (df['Finding'].isin(config.class_names))
    ].copy()
    
    print(f"Found {len(df_filtered)} images for classes: {config.class_names}")
    
    # Create splits using SSL4GIE method  
    train_idx, test_idx, val_idx = split_ids(len(df_filtered))
    split_indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    for split, indices in split_indices.items():
        split_df = df_filtered.iloc[indices]
        
        for _, row in tqdm(split_df.iterrows(), desc=f"Processing {split}", total=len(split_df)):
            video_file, finding = row['Video file'], row['Finding']
            class_idx = config.class_to_idx[finding]
            
            # Find the actual image file
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                if (potential_path := source_path / f"{video_file}{ext}").exists():
                    img_path = potential_path
                    break
                if matches := list(source_path.rglob(f"{video_file}{ext}")):
                    img_path = matches[0]
                    break
            
            if img_path is None:
                print(f"Warning: Could not find image for {video_file}")
                continue
            
            img_count = conversion_stats['splits'][split]
            new_name = f"img_{class_idx:02d}_{split}_{img_count:05d}.png"
            dest_path = output_path / split / f"class_{class_idx}" / new_name
            _process_image(img_path, dest_path, conversion_stats, split, finding)

    _save_metadata(output_path, conversion_stats)
    return conversion_stats


def prepare_waterbirds(source_path: Path, output_path: Path, config: Optional['DatasetConfig'] = None) -> Dict:
    """Convert Waterbirds dataset to unified format."""
    output_path, source_path = Path(output_path), Path(source_path)
    
    if config is None:
        from dataset_config import WATERBIRDS_CONFIG
        config = WATERBIRDS_CONFIG
    
    _create_output_structure(output_path, config.num_classes)
    conversion_stats = _create_conversion_stats('waterbirds', config)
    conversion_stats['groups'] = {'landbird_land': 0, 'landbird_water': 0, 'waterbird_land': 0, 'waterbird_water': 0}
    
    # Load metadata
    metadata_path = source_path / "metadata.csv"
    if not metadata_path.exists():
        raise ValueError(f"Metadata file not found at {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    print(f"Found {len(df)} images in metadata")
    
    split_mapping = {0: 'train', 1: 'val', 2: 'test'}
    
    for _, row in tqdm(df.iterrows(), desc="Processing Waterbirds", total=len(df)):
        y, place, split_idx = int(row['y']), int(row['place']), int(row['split'])
        
        if split_idx not in split_mapping:
            print(f"Warning: Unknown split index {split_idx}, skipping")
            continue
        
        split = split_mapping[split_idx]
        class_name = config.idx_to_class[y]
        
        # Track group statistics (unique to Waterbirds)
        group_name = f"{class_name}_{'land' if place == 0 else 'water'}"
        conversion_stats['groups'][group_name] += 1
        
        img_path = source_path / row['img_filename']
        if not img_path.exists():
            print(f"Warning: Image not found at {img_path}")
            continue
        
        img_count = conversion_stats['splits'][split]
        new_name = f"img_{y:02d}_{split}_{img_count:05d}.jpg"
        dest_path = output_path / split / f"class_{y}" / new_name
        _process_image(img_path, dest_path, conversion_stats, split, class_name, copy_only=True)
    
    # Custom metadata saving for Waterbirds (includes group stats)
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(conversion_stats, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total images: {conversion_stats['total_images']}")
    print(f"Splits: {conversion_stats['splits']}")
    print(f"Classes: {conversion_stats['classes']}")
    print(f"Groups: {conversion_stats['groups']}")
    
    return conversion_stats


def convert_dataset(dataset_name: str, source_path: Path, output_path: Path, **kwargs) -> Dict:
    """Main entry point for dataset conversion."""
    converters = {
        'covidquex': prepare_covidquex,
        'hyperkvasir': prepare_hyperkvasir,
        'waterbirds': prepare_waterbirds,
    }

    if dataset_name.lower() not in converters:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(converters.keys())}")

    converter_func = converters[dataset_name.lower()]
    return converter_func(source_path, output_path, **kwargs)


def print_summary(data_dir: Path, models_dir: Path) -> None:
    """Print summary of downloaded files."""
    print("\nSummary:")

    # Count files
    total_files = 0
    for path in [data_dir, models_dir]:
        if path.exists():
            total_files += sum(1 for _ in path.rglob('*') if _.is_file())

    print(f"  Total files downloaded: {total_files}")
    print(f"  Data directory: {data_dir.absolute()}")
    print(f"  Models directory: {models_dir.absolute()}")


def main():
    """Main function to orchestrate all downloads."""
    data_dir, models_dir = Path("./data"), Path("./models")
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    print("Dataset & Model Setup")
    print("=" * 50)

    try:
        # Download datasets (uncomment as needed)
        download_hyperkvasir(data_dir, models_dir)
        download_covidquex(data_dir, models_dir)
        download_waterbirds(data_dir, models_dir)
        
        download_sae_checkpoints(data_dir)
        print_summary(data_dir, models_dir)
        print("\nSetup completed successfully.")

    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check for required packages
    try:
        import gdown
        import requests
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print("Missing required package. Install: pip install gdown requests huggingface-hub")
        sys.exit(1)

    main()
