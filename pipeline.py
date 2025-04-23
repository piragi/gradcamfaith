# pipeline.py
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

import io_utils
import perturbation
import vit.attribution as attribution
import vit.model as model
import vit.preprocessing as preprocessing
from config import FileConfig, PipelineConfig


def preprocess_dataset(config: PipelineConfig, source_dir: Path) -> List[Path]:
    """
    Preprocess images by resizing them to target_size and saving them to the data directory.
    
    Args:
        config: Pipeline configuration
        source_dir: Directory containing original images
        
    Returns:
        List of paths to the processed images
    """
    io_utils.ensure_directories([config.file.data_dir])

    image_files = list(source_dir.glob("**/*.png"))
    print(f"Found {len(image_files)} X-ray images")

    processed_paths = []
    for image_file in image_files:
        output_path = config.file.data_dir / image_file.name

        if output_path.exists():
            print(
                f"Skipping {output_path.name} - already exists in {config.file.data_dir}"
            )
            processed_paths.append(output_path)
            continue

        try:
            img = preprocessing.load_image(str(image_file))
            processor = preprocessing.get_default_processor(
                img_size=config.classify.target_size[0])
            # Apply the PIL transforms (resize, crop) without tensor conversion
            pil_transform = processor.transforms[0]
            processed_img = pil_transform(img)
            processed_img.save(output_path)

            processed_paths.append(output_path)
            print(f"Processed: {image_file.name} -> {output_path}")
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")

    print(
        f"Preprocessing complete. {len(processed_paths)} images saved to {config.file.data_dir}"
    )
    return processed_paths


def save_attribution_results(config: FileConfig, image_path: Path,
                             attribution_map: np.ndarray,
                             attribution_neg: np.ndarray,
                             ffn_activity: List[Dict]) -> Dict[str, str]:
    """Save attribution results to files.
    
    Args:
        config: File configuration
        image_path: Path to the original image
        attribution_map: Attribution map
        attribution_neg: Negative attribution map
        ffn_activity: FFN activity data
        
    Returns:
        Dictionary with paths to saved files
    """
    attribution_path = config.attribution_dir / f"{image_path.stem}_attribution.npy"
    attribution_neg_path = config.attribution_dir / f"{image_path.stem}_attribution_neg.npy"
    ffn_activity_path = config.attribution_dir / f"{image_path.stem}_ffn_activity.npy"

    # Convert ffn_activities to numpy array format for saving
    ffn_activity_data = np.array([{
        'layer': activity['layer'],
        'mean_activity': activity['mean_activity'],
        'cls_activity': activity['cls_activity'],
        'activity': activity['activity']
    } for activity in ffn_activity],
                                 dtype=object)

    np.save(attribution_path, attribution_map)
    np.save(attribution_neg_path, attribution_neg)
    np.save(ffn_activity_path, ffn_activity_data)

    return {
        "attribution_path": str(attribution_path),
        "attribution_neg_path": str(attribution_neg_path),
        "ffn_activity_path": str(ffn_activity_path)
    }


def classify_explain_single_image(config: PipelineConfig, image_path: Path,
                                  vit_model: model.VisionTransformer,
                                  device: torch.device) -> Dict[str, Any]:
    """
    Classify a single image and generate attribution.
    
    Args:
        config: Pipeline configuration
        image_path: Path to the image to classify
        vit_model: Pre-loaded Vision Transformer model
        device: Device to run on
        
    Returns:
        Dictionary with classification results and paths
    """
    cache_path = io_utils.build_cache_path(
        config.file.cache_dir, image_path,
        f"_classification{config.file.output_suffix}")

    # Try to load from cache
    cached_result = io_utils.try_load_from_cache(cache_path)
    if config.file.use_cached and cached_result:
        return cached_result

    original_image, input_tensor = preprocessing.preprocess_image(
        str(image_path), img_size=config.classify.target_size[0])

    prediction = model.get_prediction(vit_model, input_tensor, device=device)

    attribution_result = attribution.generate_attribution(
        model=vit_model,
        input_tensor=input_tensor,
        method="transmm",
        target_class=prediction['predicted_class_idx'],
        device=device,
        img_size=config.classify.target_size[0],
        gini_params=config.classify.gini_params)

    # Extract attribution maps and FFN activities
    pos_attr = attribution_result["attribution_positive"]
    neg_attr = attribution_result["attribution_negative"]
    ffn_activities = attribution_result["ffn_activity"]

    # Save attribution results
    attribution_paths = save_attribution_results(
        config.file,
        image_path,
        pos_attr,  # Positive attribution map
        neg_attr,  # Negative attribution map
        ffn_activities  # FFN activity data
    )

    # Save the model input
    vit_input_path = config.file.vit_inputs_dir / f"{image_path.stem}{config.file.output_suffix}.png"
    original_image.save(vit_input_path)

    # Prepare the result
    result = {
        "image_path":
        str(image_path),
        "vit_input_path":
        str(vit_input_path),
        "predicted_class":
        prediction["predicted_class_label"],
        "predicted_class_idx":
        prediction["predicted_class_idx"],
        "confidence":
        float(prediction["probabilities"][prediction["predicted_class_idx"]]),
        "probabilities":
        prediction["probabilities"].tolist(),
        **attribution_paths
    }

    # Cache the result
    io_utils.save_to_cache(cache_path, result)

    return result


def classify_dataset(config: PipelineConfig,
                     vit_model: model.VisionTransformer,
                     device: torch.device) -> pd.DataFrame:
    """
    Classify images in the data directory and generate attribution maps.
    
    Args:
        config: Pipeline configuration
        vit_model: Pre-loaded Vision Transformer model
        device: Device to run on
        
    Returns:
        DataFrame with classification results and paths to attribution maps.
    """
    io_utils.ensure_directories(config.directories)
    image_paths = list(config.file.data_dir.glob("*.png"))
    results = []

    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")
        result = classify_explain_single_image(config, image_path, vit_model,
                                               device)
        results.append(result)

    df = pd.DataFrame(results)
    results_path = config.file.output_dir / f"classification_results{config.file.output_suffix}.csv"
    df.to_csv(results_path, index=False)

    return df


# Perturbation functions
def generate_perturbed_identifier(config: PipelineConfig,
                                  original_filename: str, patch_id: int,
                                  x: int, y: int) -> str:
    """Generate an identifier for a perturbed image.
    
    Args:
        config: Pipeline configuration
        original_filename: Original filename without extension
        patch_id: Unique identifier for the patch
        x: X-coordinate of the patch
        y: Y-coordinate of the patch
        
    Returns:
        Identifier string for the perturbed image
    """
    if config.perturb.method == "sd":
        return f"{original_filename}_patch{patch_id}_x{x}_y{y}_s{config.perturb.strength}"
    else:
        return f"{original_filename}_patch{patch_id}_x{x}_y{y}_mean"


def perturb_single_patch(
    config: PipelineConfig,
    original_filename: str,
    patch_info: Tuple[int, int, int],
    sd_pipe: Optional[StableDiffusionInpaintPipeline] = None
) -> Optional[Path]:
    """
    Perturb a single patch in an image.
    
    Args:
        config: Pipeline configuration
        original_filename: Original filename without extension
        patch_info: Tuple of (patch_id, x, y)
        sd_pipe: Optional StableDiffusionInpaintPipeline for SD method
        
    Returns:
        Path to the perturbed image if successful, None otherwise
    """
    patch_id, x, y = patch_info
    patch_size = config.perturb.patch_size

    perturbed_id = generate_perturbed_identifier(config, original_filename,
                                                 patch_id, x, y)
    perturbed_image_path = config.file.perturbed_dir / f'{perturbed_id}.png'
    mask_path = config.file.mask_dir / f'{perturbed_id}_mask.npy'

    # Skip if already exists
    if perturbed_image_path.exists():
        return perturbed_image_path

    try:
        if config.perturb.method == "sd":
            if sd_pipe is None:
                raise ValueError(
                    "SD pipe is required for SD perturbation but was not provided"
                )

            result_image, np_mask = perturbation.perturb_patch_sd(
                sd_pipe,
                original_filename, (x, y, patch_size),
                strength=config.perturb.strength)
        else:
            result_image, np_mask = perturbation.perturb_patch_mean(
                original_filename, (x, y, patch_size))

        result_image.save(perturbed_image_path)
        np.save(mask_path, np_mask)
        return perturbed_image_path

    except Exception as e:
        print(f"Error processing {perturbed_id}: {e}")
        return None


def generate_patch_coordinates(image: Image.Image,
                               patch_size: int) -> List[Tuple[int, int, int]]:
    """Generate patch coordinates for an image.
    
    Args:
        image: Image to generate patches for
        patch_size: Size of patches
        
    Returns:
        List of patch coordinates as (patch_id, x, y)
    """
    width, height = image.size
    num_patches_x = width // patch_size
    patches = []

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch_id = (y // patch_size) * num_patches_x + (x // patch_size)
            patches.append((patch_id, x, y))

    return patches


def perturb_image_patches(
        config: PipelineConfig,
        image_path: Path,
        sd_pipe: Optional[StableDiffusionInpaintPipeline] = None
) -> List[Path]:
    """
    Perturb patches in a single image.
    
    Args:
        config: Pipeline configuration
        image_path: Path to the image to perturb
        sd_pipe: Optional StableDiffusionInpaintPipeline for SD method
        
    Returns:
        List of paths to perturbed images
    """
    perturbed_paths = []

    original_filename = image_path.stem
    image = preprocessing.load_image(str(image_path))

    patches = generate_patch_coordinates(image, config.perturb.patch_size)

    for patch_info in patches:
        perturbed_path = perturb_single_patch(
            config,
            original_filename=original_filename,
            patch_info=patch_info,
            sd_pipe=sd_pipe)

        if perturbed_path:
            perturbed_paths.append(perturbed_path)

    return perturbed_paths


def perturb_dataset(
        config: PipelineConfig,
        results_df: pd.DataFrame,
        sd_pipe: Optional[StableDiffusionInpaintPipeline] = None
) -> List[Path]:
    """
    Perturb patches in all images.
    
    Args:
        config: Pipeline configuration
        results_df: DataFrame with classification results
        sd_pipe: Optional StableDiffusionInpaintPipeline for SD method
        
    Returns:
        List of paths to perturbed images
    """
    io_utils.ensure_directories(
        [config.file.perturbed_dir, config.file.mask_dir])

    perturbed_image_paths = []
    processing_df = results_df.head(
        config.perturb.max_images) if config.perturb.max_images else results_df

    for idx, row in enumerate(processing_df.iterrows()):
        _, row_data = row
        image_path = Path(row_data["image_path"])

        print(
            f"Processing image {idx+1}/{len(processing_df)}: {image_path.name}"
        )

        image_paths = perturb_image_patches(config, image_path, sd_pipe)
        perturbed_image_paths.extend(image_paths)

    return perturbed_image_paths


def run_classification(config: PipelineConfig,
                       device: torch.device) -> pd.DataFrame:
    """
    Run the classification pipeline.
    
    Args:
        config: Pipeline configuration
        device: Device to run on
        
    Returns:
        DataFrame with classification results
    """
    io_utils.ensure_directories(config.directories)

    vit_model = model.load_vit_model(num_classes=3, device=device)
    vit_model = model.register_model_hooks(vit_model)
    print("Model loaded and hooks registered for attribution")

    return classify_dataset(config, vit_model, device)


def run_perturbation(config: PipelineConfig,
                     results_df: pd.DataFrame) -> List[Path]:
    """
    Run the perturbation pipeline.
    
    Args:
        config: Pipeline configuration
        results_df: Classification results
        
    Returns:
        List of paths to perturbed images
    """
    # Load the SD pipeline if needed
    sd_pipe = None
    if config.perturb.method == "sd":
        sd_pipe = perturbation.load_sd_model()

    return perturb_dataset(config, results_df, sd_pipe)


def run_pipeline(
    config: PipelineConfig,
    source_dir: Optional[Path] = None,
    device: Optional[torch.device] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete pipeline: preprocess, classify, and perturb.
    
    Args:
        config: Pipeline configuration
        source_dir: Optional directory containing source images to preprocess
        device: Device to run on
        
    Returns:
        Tuple of (classification_results, perturbed_paths)
    """
    io_utils.ensure_directories(config.directories)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Preprocess if source directory is provided
    if source_dir:
        preprocess_dataset(config, source_dir)

    results_df = run_classification(config, device)

    perturbed_paths = run_perturbation(config, results_df)
    print(f"Generated {len(perturbed_paths)} perturbed patch images")

    config.file.output_suffix = "_perturbed"
    config.file.data_dir = Path("./results/patches")
    perturbed_df = run_classification(config, device)

    return results_df, perturbed_df
