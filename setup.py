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
    # Create main directories
    data_dir = Path("./scratch/data")
    models_dir = Path("./scratch/models")
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    print("Medical Dataset & Model Setup")

    try:
        # Download Hyperkvasir
        download_hyperkvasir(data_dir, models_dir)

        # Download CovidQueX
        download_covidquex(data_dir, models_dir)

        # Print summary
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
    except ImportError as e:
        print("Missing required package. Install: pip install gdown requests")
        sys.exit(1)

    main()
