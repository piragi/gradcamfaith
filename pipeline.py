from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import numpy as np

from config import PipelineConfig, FileConfig, ClassificationConfig, PerturbationConfig
import io_utils
import transformer as trans
import perturbation


# Preprocessing functions
def preprocess_dataset(config: PipelineConfig, source_dir: Path) -> List[Path]:
    """
    Preprocess images by resizing them to target_size and saving them to the data directory.
    
    Args:
        config: Pipeline configuration
        source_dir: Directory containing original images
        
    Returns:
        List of paths to the processed images
    """
    io_utils.ensure_directories([config.file_config.data_dir])
    
    image_files = list(source_dir.glob("**/*.png"))
    print(f"Found {len(image_files)} X-ray images")
    
    processed_paths = []
    for image_file in image_files:
        output_path = config.file_config.data_dir / image_file.name
        
        if output_path.exists():
            print(f"Skipping {output_path.name} - already exists in {config.file_config.data_dir}")
            processed_paths.append(output_path)
            continue
        
        try:
            img = Image.open(image_file)
            resized_img = img.resize(config.classification_config.target_size, Image.LANCZOS)
            resized_img.save(output_path)
            processed_paths.append(output_path)
            
            print(f"Processed: {image_file.name} -> {output_path}")
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
    
    print(f"Preprocessing complete. {len(processed_paths)} images saved to {config.file_config.data_dir}")
    return processed_paths


# Attribution functions
def save_attribution_results(
    config: FileConfig,
    image_path: Path, 
    attribution_map: np.ndarray, 
    attribution_neg: np.ndarray, 
    ffn_activity: np.ndarray
) -> Dict[str, str]:
    """Save attribution results to files.
    
    Args:
        config: File configuration
        image_path: Path to the original image
        attribution_map: Attribution map
        attribution_neg: Negative attribution map
        ffn_activity: FFN activity map
        
    Returns:
        Dictionary with paths to saved files
    """
    attribution_path = config.attribution_dir / f"{image_path.stem}_attribution.npy"
    attribution_neg_path = config.attribution_dir / f"{image_path.stem}_attribution_neg.npy"
    ffn_activity_path = config.attribution_dir / f"{image_path.stem}_ffn_activity.npy"
    
    np.save(attribution_path, attribution_map)
    np.save(attribution_neg_path, attribution_neg)
    np.save(ffn_activity_path, ffn_activity)
    
    return {
        "attribution_path": str(attribution_path),
        "attribution_neg_path": str(attribution_neg_path),
        "ffn_activity_path": str(ffn_activity_path)
    }

# Classification functions
def classify_explain_single_image(
    config: PipelineConfig,
    image_path: Path,
    vit: trans.ViT
) -> Dict[str, Any]:
    """
    Classify a single image and generate attribution.
    
    Args:
        config: Pipeline configuration
        image_path: Path to the image to classify
        vit: Vision Transformer model
        
    Returns:
        Dictionary with classification results and paths
    """
    cache_path = io_utils.build_cache_path(
        config.file_config.cache_dir, 
        image_path, 
        f"_classification{config.file_config.output_suffix}"
    )
    
    # Try to load from cache
    cached_result = io_utils.try_load_from_cache(cache_path)
    if config.file_config.use_cached and cached_result:
        return cached_result
    
    # Classify the image
    image, classification = vit.classify_image(image_path=str(image_path))
    
    # Save the visualized image
    vis_path = config.file_config.attribution_dir / f"{image_path.stem}_vis.png"
    image.save(vis_path)
    
    # Generate attribution
    image, (attribution_map, attribution_neg), ffn_activity = trans.transmm_with_ffn_activity(
        vit,
        image_path,
        classification['predicted_class_idx'],
        pretransform=config.classification_config.pretransform,
        gini_params=config.classification_config.gini_params
    )
    
    # Save attribution results
    attribution_paths = save_attribution_results(
        config.file_config,
        image_path, 
        attribution_map, 
        attribution_neg, 
        ffn_activity
    )
    
    # Save the VIT input
    vit_input_path = config.vit_inputs_dir / f"{image_path.stem}{config.output_suffix}.png"
    
    # Prepare the result
    result = {
        "image_path": str(image_path),
        "vit_input_path": str(vit_input_path),
        "predicted_class": classification["predicted_class_label"],
        "predicted_class_idx": classification["predicted_class_idx"],
        "confidence": float(classification["probabilities"][classification["predicted_class_idx"]]),
        "probabilities": classification["probabilities"].tolist(),
        "attribution_vis_path": str(vis_path),
        **attribution_paths
    }
    
    # Cache the result
    io_utils.save_to_cache(cache_path, result)
    
    return result


def classify_dataset(config: PipelineConfig, vit: trans.ViT) -> pd.DataFrame:
    """
    Classify images in the data directory and generate attribution maps.
    
    Args:
        config: Pipeline configuration
        vit: Vision Transformer model
        
    Returns:
        DataFrame with classification results and paths to attribution maps.
    """
    io_utils.ensure_directories(config.directories)
    image_paths = list(config.data_dir.glob("*.png"))
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")
        result = classify_explain_single_image(config, image_path, vit)
        results.append(result)
    
    df = pd.DataFrame(results)
    results_path = config.output_dir / f"classification_results{config.output_suffix}.csv"
    df.to_csv(results_path, index=False)
    
    return df


# Perturbation functions
def generate_perturbed_identifier(
    config: PipelineConfig,
    original_filename: str,
    patch_id: int,
    x: int,
    y: int
) -> str:
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
    if config.perturbation_method == "sd":
        return f"{original_filename}_patch{patch_id}_x{x}_y{y}_s{config.perturbation_strength}"
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
    patch_size = config.patch_size

    perturbed_id = generate_perturbed_identifier(config, original_filename, patch_id, x, y)
    perturbed_image_path = config.perturbed_dir / f'{perturbed_id}.png'
    mask_path = config.mask_dir / f'{perturbed_id}_mask.npy'

    # Skip if already exists
    if perturbed_image_path.exists():
        return perturbed_image_path

    try:
        if config.perturbation_method == "sd":
            if sd_pipe is None:
                raise ValueError("SD pipe is required for SD perturbation but was not provided")
            
            result_image, np_mask = perturbation.perturb_single_patch(
                sd_pipe,
                original_filename, 
                (x, y, patch_size),
                strength=config.perturbation_strength
            )
        else:
            result_image, np_mask = perturbation.perturb_patch_mean(
                original_filename, 
                (x, y, patch_size)
            )
        
        result_image.save(perturbed_image_path)
        np.save(mask_path, np_mask)
        return perturbed_image_path
        
    except Exception as e:
        print(f"Error processing {perturbed_id}: {e}")
        return None


def generate_patch_coordinates(
    image: Image.Image,
    patch_size: int
) -> List[Tuple[int, int, int]]:
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
    image = Image.open(image_path).convert("RGB")
    
    patches = generate_patch_coordinates(image, config.patch_size)
    
    for patch_info in patches:
        perturbed_path = perturb_single_patch(
            config,
            original_filename=original_filename,
            patch_info=patch_info,
            sd_pipe=sd_pipe
        )
        
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
    io_utils.ensure_directories([config.perturbed_dir, config.mask_dir])
    
    perturbed_image_paths = []
    processing_df = results_df.head(config.perturbation_config.max_images) if config.perturbation_config.max_images else results_df
    
    for idx, row in enumerate(processing_df.iterrows()):
        _, row_data = row
        image_path = Path(row_data["image_path"])
        
        print(f"Processing image {idx+1}/{len(processing_df)}: {image_path.name}")
        
        image_paths = perturb_image_patches(config, image_path, sd_pipe)
        perturbed_image_paths.extend(image_paths)
    
    return perturbed_image_paths


# Factory function
def create_vit_model(config: PipelineConfig) -> trans.ViT:
    """Create and initialize a ViT model.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Initialized ViT model
    """
    model_config = trans.ModelConfig(
        model_type=trans.ModelType(config.classification_config.model_type),
        num_classes=config.classification_config.num_classes,
        device=config.classification_config.device
    )
    return trans.ViT(config=model_config, method=config.classification_config.attribution_method)


def run_classification(config: PipelineConfig) -> pd.DataFrame:
    """
    Run the classification pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        DataFrame with classification results
    """
    io_utils.ensure_directories(config.directories)
    vit_model = create_vit_model(config)
    return classify_dataset(config, vit_model)


def run_perturbation(config: PipelineConfig, results_df: pd.DataFrame) -> List[Path]:
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
    if config.perturbation_method == "sd":
        sd_pipe = perturbation.create_sd_pipeline()
    
    return perturb_dataset(config, results_df, sd_pipe)


def run_pipeline(config: PipelineConfig, source_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, List[Path]]:
    """
    Run the complete pipeline: preprocess, classify, and perturb.
    
    Args:
        config: Pipeline configuration
        source_dir: Optional directory containing source images to preprocess
        
    Returns:
        Tuple of (classification_results, perturbed_paths)
    """
    io_utils.ensure_directories(config.directories)
    
    # Preprocess if source directory is provided
    if source_dir:
        preprocess_dataset(config, source_dir)
    
    # Classify
    results_df = run_classification(config)
    
    # Perturb
    perturbed_paths = run_perturbation(config, results_df)
    
    return results_df, perturbed_paths
