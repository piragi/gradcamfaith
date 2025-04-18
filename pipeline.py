import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

import perturbation
import transformer as trans

DATA_DIR = "./images"
OUTPUT_DIR = "./results"
CACHE_DIR = "./cache"
PATCH_DIR = "./results/patches"


def preprocess_dataset(
    source_dir: str = "./COVID-QU-Ex",
    dest_dir: str = "./images",
    target_size: tuple = (224, 224)) -> List[Path]:
    """
    Preprocess JPG images by resizing them to target_size and saving them to dest_dir.
    Can recursively search through subfolders and filter for frontal X-rays.
    
    Args:
        source_dir: Directory containing original images
        dest_dir: Directory to save resized images
        target_size: Target size (width, height) for the resized images
        recursive: If True, search recursively through all subfolders
        only_frontal: If True, only process images with '_frontal' in the filename
        
    Returns:
        List of paths to the processed images
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    ensure_directories([dest_path])

    image_files = list(source_path.glob("**/*.png"))
    print(f"Found {len(image_files)} frontal X-rays")

    processed_paths = []
    for image_file in image_files:
        # Create a unique output filename to avoid conflicts from different subfolders
        output_filename = f"{image_file.name}"
        output_path = dest_path / output_filename

        if output_path.exists():
            print(
                f"Skipping {output_path.name} - already exists in {dest_dir}")
            processed_paths.append(output_path)
            continue

        try:
            img = Image.open(image_file)
            resized_img = img.resize(target_size, Image.LANCZOS)
            resized_img.save(output_path)
            processed_paths.append(output_path)

            print(f"Processed: {image_file.name} -> {output_path}")
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")

    print(
        f"Preprocessing complete. {len(processed_paths)} images saved to {dest_dir}"
    )
    return processed_paths


def ensure_directories(directories: List[Path]) -> None:
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)


def try_load_from_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def save_to_cache(cache_path: Path, result: Dict[str, Any]) -> None:
    with open(cache_path, 'w') as f:
        json.dump(result, f)


def classify_explain_single_image(image_path: Path,
                                  vit: trans.ViT,
                                  dirs: Dict[str, Path],
                                  output_suffix: str,
                                  pretransform: bool = False,
                                  gini_params: Tuple[float, float,
                                                     float] = (0.65, 8.0, 0.5),
                                  use_cached: bool = True) -> Dict[str, Any]:
    cache_path = dirs[
        'cache'] / f"{image_path.stem}_classification{output_suffix}.json"
    cached_result = try_load_from_cache(cache_path)
    if use_cached and cached_result: return cached_result

    image, classification = vit.classify_image(image_path=str(image_path))
    vis_path = dirs["attribution"] / f"{image_path.stem}_vis.png"
    image.save(vis_path)

    image, (attribution,
            attribution_neg), ffn_activity = trans.transmm_with_ffn_activity(
                vit,
                image_path,
                classification['predicted_class_idx'],
                pretransform=pretransform,
                gini_params=gini_params)

    attribution_path = dirs[
        "attribution"] / f"{image_path.stem}_attribution.npy"
    np.save(attribution_path, attribution)

    attribution_neg_path = dirs[
        "attribution"] / f"{image_path.stem}_attribution_neg.npy"
    np.save(attribution_neg_path, attribution_neg)

    ffn_activity_path = dirs[
        "attribution"] / f"{image_path.stem}_ffn_activity.npy"
    np.save(ffn_activity_path, ffn_activity)
    # explainer.visualize(image, attribution, save_path=str(vis_path))

    vit_input_path = dirs[
        "vit_inputs"] / f"{image_path.stem}{output_suffix}.png"
    result = {
        "image_path":
        str(image_path),
        "vit_input_path":
        str(vit_input_path),
        "predicted_class":
        classification["predicted_class_label"],
        "predicted_class_idx":
        classification["predicted_class_idx"],
        "confidence":
        float(classification["probabilities"][
            classification["predicted_class_idx"]]),
        "probabilities":
        classification["probabilities"].tolist(),
        "attribution_path":
        str(attribution_path),
        "attribution_neg_path":
        str(attribution_neg_path),
        "attribution_vis_path":
        str(vis_path),
        "ffn_activity_path":
        str(ffn_activity_path)
    }

    save_to_cache(cache_path, result)

    return result


def classify(data_directory: str, output_suffix: str = "") -> pd.DataFrame:
    """
    Classify images and generate attribution maps.
    
    Args:
        data_directory: Optional custom directory containing images to classify.
                       If None, uses default DATA_DIR.
        output_suffix: Optional suffix for output files to distinguish different runs
        
    Returns:
        DataFrame with classification results and paths to attribution maps.
    """
    output_dir = Path(OUTPUT_DIR)
    cache_dir = Path(CACHE_DIR)
    attribution_dir = output_dir / f"attributions{output_suffix}"
    vit_inputs_dir = output_dir / "vit_inputs"
    data_dir = Path(data_directory)

    dirs = {
        "output": output_dir,
        "cache": cache_dir,
        "attribution": attribution_dir,
        "vit_inputs": vit_inputs_dir
    }

    ensure_directories(list(dirs.values()))
    image_paths = list(data_dir.glob("*.png"))

    vit = trans.ViT(method="transmm")
    results = []

    for image_path in image_paths:
        result = classify_explain_single_image(image_path, vit, dirs,
                                               output_suffix)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / f"classification_results{output_suffix}.csv",
              index=False)

    return df


def generate_perturbed_identifier(original_filename: str,
                                  patch_id: int,
                                  x: int,
                                  y: int,
                                  strength: float,
                                  method: str = "sd") -> str:
    if method == "sd":
        return f"{original_filename}_patch{patch_id}_x{x}_y{y}_s{strength}"
    else:
        return f"{original_filename}_patch{patch_id}_x{x}_y{y}_mean"


def perturb_single_patch(original_filename: str,
                         patch_info: Tuple[int, int, int],
                         sd_pipe: StableDiffusionInpaintPipeline,
                         output_dirs: Dict[str, Path],
                         method: str = "sd",
                         strength: float = 0.2) -> Optional[Path]:
    patch_id, x, y = patch_info
    patch_size = 16  # Standard patch size

    perturbed_id = generate_perturbed_identifier(original_filename, patch_id,
                                                 x, y, strength, method)
    perturbed_image_path = output_dirs["perturbed"] / f'{perturbed_id}.png'
    mask_path = output_dirs["masks"] / f'{perturbed_id}_mask.npy'

    # Skip if already exists
    if perturbed_image_path.exists():
        return perturbed_image_path

    try:
        if method == "sd":
            result_image, np_mask = perturbation.perturb_single_patch(
                sd_pipe,
                original_filename, (x, y, patch_size),
                strength=strength)
        else:
            result_image, np_mask = perturbation.perturb_patch_mean(
                original_filename, (x, y, patch_size))

        result_image.save(perturbed_image_path)
        np.save(mask_path, np_mask)
        return perturbed_image_path

    except Exception as e:
        print(f"Error processing {perturbed_id}: {e}")
        return None


def perturb_image_patches(image_path: str,
                          sd_pipe: Optional[StableDiffusionInpaintPipeline],
                          output_dirs: Dict[str, Path],
                          patch_size: int = 16,
                          strength: float = 0.2,
                          method: str = "mean") -> List[Path]:
    perturbed_paths = []

    processed_filename = Path(image_path).name
    original_filename = processed_filename.split('.')[0]

    # Load image and attribution
    image = Image.open(image_path).convert('RGB')
    width, height = image.size

    num_patches_x = width // patch_size

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch_id = (y // patch_size) * num_patches_x + (x // patch_size)

            # Process with SD perturbation
            perturbed_path = perturb_single_patch(
                original_filename=original_filename,
                patch_info=(patch_id, x, y),
                sd_pipe=sd_pipe,
                output_dirs=output_dirs,
                method=method,
                strength=strength)
            if perturbed_path:
                perturbed_paths.append(perturbed_path)

    return perturbed_paths


def perturb_all_patches(
        results_df: pd.DataFrame,
        sd_pipe: Optional[StableDiffusionInpaintPipeline] = None,
        patch_size: int = 16,
        strength: float = 0.2,
        max_images: Optional[int] = None,
        method: str = "mean") -> List[Path]:
    """
    Perturbs patches in all images and saves the results.
    
    Args:
        results_df: DataFrame with classification results
        sd_pipe: StableDiffusionInpaintPipeline instance
        patch_size: Size of each patch to perturb
        strength: Perturbation strength
        max_images: Maximum number of images to process
        
    Returns:
        List of paths to perturbed images
    """
    output_dir = Path(OUTPUT_DIR)
    perturbed_dir = output_dir / "patches"
    mask_dir = output_dir / "patch_masks"

    output_dirs = {"perturbed": perturbed_dir, "masks": mask_dir}
    ensure_directories(list(output_dirs.values()))

    perturbed_image_paths = []
    processing_df = results_df.head(max_images) if max_images else results_df

    for idx, row in enumerate(processing_df.iterrows()):
        _, row_data = row
        image_path = row_data["image_path"]

        print(
            f"Processing image {idx+1}/{len(processing_df)}: {Path(image_path).name}"
        )

        image_paths = perturb_image_patches(image_path=image_path,
                                            sd_pipe=sd_pipe,
                                            output_dirs=output_dirs,
                                            patch_size=patch_size,
                                            strength=strength,
                                            method=method)

        perturbed_image_paths.extend(image_paths)

    return perturbed_image_paths
