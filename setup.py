#!/usr/bin/env python3
"""
Setup script to download Hyperkvasir and CovidQueX datasets and model files.
Run: python setup_downloads.py

Downloads:
- Hyperkvasir: dataset (zip) and model from Google Drive
- CovidQueX: dataset (tar.gz) and model from Google Drive
"""
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List

import gdown
import requests


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
        print(f"\nError downloading {url}: {e}")
        raise


def download_from_gdrive(file_id: str, output_path: Path, description: str) -> None:
    """Download file from Google Drive using gdown."""
    if output_path.exists():
        print(f"   ✓ Already exists: {output_path.name}")
        return

    print(f"   Downloading: {description}")
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(output_path), quiet=False)
        print(f"   ✓ Downloaded: {output_path.name}")
    except Exception as e:
        print(f"   ❌ Failed to download: {e}")
        raise


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file with progress indication."""
    print(f"   Extracting: {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"   ✓ Extracted to: {extract_to}")
    except Exception as e:
        print(f"   ❌ Failed to extract: {e}")
        raise


def extract_tar_gz(tar_path: Path, extract_to: Path) -> None:
    """Extract tar.gz file with progress indication."""
    print(f"   Extracting: {tar_path.name}")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
        print(f"   ✓ Extracted to: {extract_to}")
    except Exception as e:
        print(f"   ❌ Failed to extract: {e}")
        raise


def download_hyperkvasir(data_dir: Path, models_dir: Path) -> None:
    """Download Hyperkvasir dataset and model."""
    print("\n" + "=" * 60)
    print("HYPERKVASIR DOWNLOADS")
    print("=" * 60)

    # Create hyperkvasir subdirectories
    hk_data_dir = data_dir / "hyperkvasir"
    hk_models_dir = models_dir / "hyperkvasir"
    hk_data_dir.mkdir(exist_ok=True, parents=True)
    hk_models_dir.mkdir(exist_ok=True, parents=True)

    # 1. Download Hyperkvasir dataset
    print("\n1. Downloading Hyperkvasir dataset...")
    dataset_url = "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip"
    dataset_path = hk_data_dir / "hyper-kvasir-labeled-images.zip"

    if not dataset_path.exists():
        print(f"   Source: {dataset_url}")
        try:
            # Try wget first (faster for large files)
            subprocess.run(["wget", "-O", str(dataset_path), dataset_url], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to requests if wget is not available
            print("   wget not found, using requests...")
            download_with_progress(dataset_url, dataset_path)
        print(f"   ✓ Downloaded: {dataset_path.name}")
    else:
        print(f"   ✓ Already exists: {dataset_path.name}")

    # Extract dataset
    extracted_dir = hk_data_dir / "labeled-images"
    if not extracted_dir.exists():
        extract_zip(dataset_path, hk_data_dir)
    else:
        print(f"   ✓ Already extracted: labeled-images/")

    # 2. Download Hyperkvasir model
    print("\n2. Downloading Hyperkvasir model...")
    model_info = {
        "name": "hyperkvasir_vit_model.pth",
        "id": "1gT4Z0qD09ClPOcfgXo0rMAsjvoD-tpDE",
        "description": "ViT model for Hyperkvasir"
    }

    output_path = hk_models_dir / model_info["name"]
    download_from_gdrive(model_info["id"], output_path, model_info["description"])


def download_covidquex(data_dir: Path, models_dir: Path) -> None:
    """Download CovidQueX dataset and model."""
    print("\n" + "=" * 60)
    print("COVIDQUEX DOWNLOADS")
    print("=" * 60)

    # Create covidquex subdirectories
    cq_data_dir = data_dir / "covidquex"
    cq_models_dir = models_dir / "covidquex"
    cq_data_dir.mkdir(exist_ok=True, parents=True)
    cq_models_dir.mkdir(exist_ok=True, parents=True)

    # 1. Download CovidQueX dataset
    print("\n1. Downloading CovidQueX dataset...")
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
        else:
            print(f"   ✓ Already extracted")

    # 2. Download CovidQueX model
    print("\n2. Downloading CovidQueX model...")
    model_info = {
        "name": "covidquex_model.pth",
        "id": "1JZM5ZRncaV3iFX9L6NFT1P0-APyHbBV0",
        "description": "CovidQueX model"
    }

    output_path = cq_models_dir / model_info["name"]
    download_from_gdrive(model_info["id"], output_path, model_info["description"])


def print_summary(data_dir: Path, models_dir: Path) -> None:
    """Print summary of downloaded files."""
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print("\n📁 Directory Structure:")

    # Check Hyperkvasir files
    print("\n  Hyperkvasir:")
    hk_data = data_dir / "hyperkvasir"
    hk_models = models_dir / "hyperkvasir"

    if hk_data.exists():
        for item in hk_data.iterdir():
            size = item.stat().st_size / (1024**2) if item.is_file() else 0
            print(f"    ✓ data/hyperkvasir/{item.name}" + (f" ({size:.1f} MB)" if size > 0 else ""))

    if hk_models.exists():
        for item in hk_models.iterdir():
            size = item.stat().st_size / (1024**2)
            print(f"    ✓ models/hyperkvasir/{item.name} ({size:.1f} MB)")

    # Check CovidQueX files
    print("\n  CovidQueX:")
    cq_data = data_dir / "covidquex"
    cq_models = models_dir / "covidquex"

    if cq_data.exists():
        for item in cq_data.iterdir():
            size = item.stat().st_size / (1024**2) if item.is_file() else 0
            print(f"    ✓ data/covidquex/{item.name}" + (f" ({size:.1f} MB)" if size > 0 else ""))

    if cq_models.exists():
        for item in cq_models.iterdir():
            size = item.stat().st_size / (1024**2)
            print(f"    ✓ models/covidquex/{item.name} ({size:.1f} MB)")


def main():
    """Main function to orchestrate all downloads."""
    # Create main directories
    data_dir = Path("data")
    models_dir = Path("models")
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("MEDICAL DATASET & MODEL DOWNLOADER")
    print("=" * 60)
    print(f"\n📂 Data directory: {data_dir.absolute()}")
    print(f"📂 Models directory: {models_dir.absolute()}")

    try:
        # Download Hyperkvasir
        download_hyperkvasir(data_dir, models_dir)

        # Download CovidQueX
        download_covidquex(data_dir, models_dir)

        # Print summary
        print_summary(data_dir, models_dir)

        print("\n" + "=" * 60)
        print("✅ ALL DOWNLOADS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space")
        print("3. Try installing required packages: pip install gdown requests")
        print("4. For Google Drive files, ensure they are publicly accessible")
        sys.exit(1)


if __name__ == "__main__":
    # Check for required packages
    try:
        import gdown
        import requests
    except ImportError as e:
        print("❌ Missing required package!")
        print("Please install: pip install gdown requests")
        sys.exit(1)

    main()
