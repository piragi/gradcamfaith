# pipeline.py
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import quantus
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from tqdm import tqdm

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

import analysis
import io_utils
import perturbation
import visualization
import vit.model as model
import vit.preprocessing as preprocessing
from config import FileConfig, PipelineConfig
from data_types import (
    AttributionDataBundle, AttributionOutputPaths, ClassEmbeddingRepresentationItem, ClassificationPrediction,
    ClassificationResult, FFNActivityItem, HeadContributionItem, PerturbationPatchInfo, PerturbedImageRecord
)
from faithfulness import evaluate_and_report_faithfulness
from translrp.ViT_new import VisionTransformer
from transmm_sfaf import (generate_attribution_prisma, load_models, load_steering_resources)


def preprocess_dataset(config: PipelineConfig, source_dir: Path) -> List[Path]:
    """
    Preprocess images by resizing them to target_size and saving them to the data directory.
    Includes class and split information in the output filename.
    
    Args:
        config: Pipeline configuration
        source_dir: Directory containing organized images (e.g., ham10k/organized/train/)
        
    Returns:
        List of paths to the processed images
    """
    io_utils.ensure_directories([config.file.data_dir])

    image_files = list(source_dir.glob("**/*.jpg"))
    print(f"Found {len(image_files)} images")

    processed_paths = []
    for image_file in image_files:
        try:
            # Extract class and split information from directory structure
            # Assuming structure: source_dir/split/class/image.jpg or source_dir/class/image.jpg
            class_name = image_file.parent.name

            # Check if we have a split directory (train/val/test)
            potential_split = image_file.parent.parent.name
            if potential_split in ['train', 'val', 'test']:
                split_name = potential_split
                # Create filename with split and class: train_akiec_ISIC_0024306.jpg
                new_filename = f"{class_name}_{image_file.name}"
            else:
                # No split directory, just use class: akiec_ISIC_0024306.jpg
                new_filename = f"{class_name}_{image_file.name}"

            output_path = config.file.data_dir / new_filename

            # Skip if exists
            if output_path.exists():
                processed_paths.append(output_path)
                continue

            # Load and process image
            img = preprocessing.load_image(str(image_file))
            processor = preprocessing.get_default_processor(img_size=config.classify.target_size[0])
            # Apply the PIL transforms (resize, crop) without tensor conversion
            pil_transform = processor.transforms[0]
            processed_img = pil_transform(img)
            processed_img.save(output_path)

            processed_paths.append(output_path)
            print(f"Processed: {image_file.name} -> {new_filename} (class: {class_name})")

        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            continue

    print(f"Preprocessing complete. {len(processed_paths)} images saved to {config.file.data_dir}")

    # Print class distribution summary
    class_counts = {}
    for path in processed_paths:
        # Extract class from filename (assumes format: [split_]class_originalname.jpg)
        filename_parts = path.stem.split('_')
        if len(filename_parts) >= 2:
            if filename_parts[0] in ['train', 'val', 'test']:
                class_name = filename_parts[1]
            else:
                class_name = filename_parts[0]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print(f"\nProcessed class distribution:")

    return processed_paths


def save_attribution_bundle_to_files(
    image_stem: str, attribution_bundle: AttributionDataBundle, file_config: FileConfig
) -> AttributionOutputPaths:
    """Save attribution bundle contents to .npy files."""

    # Ensure attribution directory exists
    io_utils.ensure_directories([file_config.attribution_dir])

    attribution_path = file_config.attribution_dir / f"{image_stem}_attribution.npy"
    raw_attribution_path = file_config.attribution_dir / f"{image_stem}_raw_attribution.npy"
    logits_path = Path("")
    ffn_activity_path = Path("")
    class_embedding_path = Path("")
    head_contribution_path = Path("")

    # Save positive attribution
    np.save(attribution_path, attribution_bundle.positive_attribution)
    np.save(raw_attribution_path, attribution_bundle.raw_attribution)

    # Save logits (save empty array if None)
    if attribution_bundle.logits is not None:
        logits_path = file_config.attribution_dir / f"{image_stem}_logits.npy"
        np.save(logits_path, attribution_bundle.logits)

    # Save FFN activities
    # Convert List[FFNActivityItem] to a NumPy array of objects (dicts) for saving
    if attribution_bundle.ffn_activities:
        ffn_activity_path = file_config.attribution_dir / f"{image_stem}_ffn_activity.npy"
        ffn_data_to_save = []
        for item in attribution_bundle.ffn_activities:
            ffn_data_to_save.append({
                'layer': item.layer,
                'mean_activity': item.mean_activity,
                'cls_activity': item.cls_activity,
                'activity_data': item.activity_data
            })
        np.save(ffn_activity_path, np.array(ffn_data_to_save, dtype=object))

    if attribution_bundle.head_contribution:
        head_contribution_path = file_config.attribution_dir / f"{image_stem}_head_contribution.npz"

        # Stack all layers into a single array instead of object array
        layers = []
        layer_indices = []

        for item in attribution_bundle.head_contribution:
            layers.append(item.stacked_contribution)
            layer_indices.append(item.layer)

        # Stack into single array: [num_layers, num_heads, batch_size, num_tokens, head_dim]
        stacked_contributions = np.stack(layers, axis=0)
        layer_indices = np.array(layer_indices)

        # Save as separate arrays (much more compact than object arrays)
        np.savez(head_contribution_path, contributions=stacked_contributions, layer_indices=layer_indices)
    # Save class embedding representations
    if attribution_bundle.class_embedding_representations:
        class_embedding_path = file_config.attribution_dir / f"{image_stem}_class_embedding_representation.npy"
        class_embedding_data_to_save = []
        for item in attribution_bundle.class_embedding_representations:
            class_embedding_data_to_save.append({
                'layer': item.layer,
                'attention_class_representation_input': item.attention_class_representation_input,
                'mlp_class_representation_input': item.mlp_class_representation_input,
                'attention_class_representation_output': item.attention_class_representation_output,
                'attention_map': item.attention_map,
                'mlp_class_representation_output': item.mlp_class_representation_output
            })
        np.save(class_embedding_path, np.array(class_embedding_data_to_save, dtype=object))

    return AttributionOutputPaths(
        attribution_path=attribution_path,
        raw_attribution_path=raw_attribution_path,
        logits=logits_path,
        ffn_activity_path=ffn_activity_path,
        class_embedding_path=class_embedding_path,
        head_contribution_path=head_contribution_path
    )


def classify_single_image(
    config: PipelineConfig, image_path: Path, vit_model: model.VisionTransformer, device: torch.device
) -> ClassificationResult:
    """
    Classifies a single image and returns a ClassificationResult with prediction only.
    Uses its own caching mechanism for classification-only results.
    """
    cache_path = io_utils.build_cache_path(
        config.file.cache_dir, image_path, f"_classification_explained{config.file.output_suffix}.json"
    )

    # Try to load from cache
    loaded_result = io_utils.try_load_from_cache(cache_path)
    if config.file.use_cached_perturbed and loaded_result:
        if loaded_result.attribution_paths is not None:
            print(
                f"Warning: Cache for classify_single_image contained attribution paths for {image_path.name}. Ignoring them."
            )
            loaded_result.attribution_paths = None
        return loaded_result

    _, input_tensor = preprocessing.preprocess_image(str(image_path), img_size=config.classify.target_size[0])
    input_tensor = input_tensor.to(device)

    prediction_dict = model.get_prediction(vit_model, input_tensor, device=device, eval=True)

    current_prediction = ClassificationPrediction(
        predicted_class_label=prediction_dict["predicted_class_label"],
        predicted_class_idx=prediction_dict["predicted_class_idx"],
        confidence=float(prediction_dict["probabilities"][prediction_dict["predicted_class_idx"]].item()),
        probabilities=prediction_dict["probabilities"].tolist()
    )

    result = ClassificationResult(image_path=image_path, prediction=current_prediction, attribution_paths=None)

    # Cache the result
    io_utils.save_to_cache(cache_path, result)

    return result


def classify_explain_single_image(
    config: PipelineConfig,
    image_path: Path,
    vit_model: model.VisionTransformer,  # Hooks should be registered on this model instance
    device: torch.device,
    sae,
    class_specific_features,
    steering_resources: Optional[Dict[int, Dict[str, Any]]],  # ADDED
    class_analysis
) -> ClassificationResult:
    """
    Classifies a single image AND generates explanations.
    Relies on vit.attribution.generate_attribution to return both prediction and attribution data.
    Manages caching for the combined ClassificationResult.
    """

    cache_path = io_utils.build_cache_path(
        config.file.cache_dir, image_path, f"_classification_explained{config.file.output_suffix}.json"
    )

    # Try to load from cache
    loaded_result = io_utils.try_load_from_cache(cache_path)
    if config.file.use_cached_original and loaded_result:
        if loaded_result.attribution_paths is not None:
            return loaded_result, None

    # 1. Preprocess image
    _, input_tensor = preprocessing.preprocess_image(str(image_path), img_size=config.classify.target_size[0])
    input_tensor = input_tensor.to(device)

    # 2. Call attribution.generate_attribution
    raw_attribution_result_dict = generate_attribution_prisma(
        model=vit_model,
        input_tensor=input_tensor,
        config=config,
        device=device,
        sae=sae,
        sf_af_dict=class_specific_features,
        steering_resources=steering_resources,
        enable_steering=config.file.weighted,
        class_analysis=class_analysis,
    )

    # Extract gradient analysis info if present
    raw_attr = raw_attribution_result_dict.get("raw_attribution", {})

    # 3. Instantiate ClassificationPrediction from the "predictions" field
    prediction_data = raw_attribution_result_dict["predictions"]
    current_prediction = ClassificationPrediction(
        predicted_class_label=prediction_data["predicted_class_label"],
        predicted_class_idx=prediction_data["predicted_class_idx"],
        confidence=float(prediction_data["probabilities"][prediction_data["predicted_class_idx"]]
                         ),  # Assuming probabilities is a list/tensor
        probabilities=prediction_data["probabilities"]
    )  # Needs to be a list

    # 4. Instantiate FFNActivityItem and ClassEmbeddingRepresentationItem lists
    ffn_activity_items: List[FFNActivityItem] = []
    if "ffn_activity" in raw_attribution_result_dict and raw_attribution_result_dict["ffn_activity"]:
        for ffn_dict in raw_attribution_result_dict["ffn_activity"]:
            ffn_activity_items.append(
                FFNActivityItem(
                    layer=ffn_dict["layer"],
                    mean_activity=ffn_dict["mean_activity"],
                    cls_activity=ffn_dict["cls_activity"],
                    activity_data=ffn_dict["activity"]
                )
            )

    class_embedding_items: List[ClassEmbeddingRepresentationItem] = []
    if "class_embedding_representation" in raw_attribution_result_dict and raw_attribution_result_dict[
        "class_embedding_representation"]:
        for cer_dict in raw_attribution_result_dict["class_embedding_representation"]:
            class_embedding_items.append(
                ClassEmbeddingRepresentationItem(
                    layer=cer_dict["layer"],
                    attention_class_representation_output=cer_dict["attention_class_representation_output"],
                    mlp_class_representation_output=cer_dict["mlp_class_representation_output"],
                    attention_class_representation_input=cer_dict["attention_class_representation_input"],
                    attention_map=cer_dict["attention_map"],
                    mlp_class_representation_input=cer_dict["mlp_class_representation_input"]
                )
            )

    head_contribution_items: List[HeadContributionItem] = []
    if "head_contribution" in raw_attribution_result_dict and raw_attribution_result_dict["head_contribution"]:
        for head_contribution_dict in raw_attribution_result_dict["head_contribution"]:
            head_contribution_items.append(
                HeadContributionItem(
                    layer=head_contribution_dict["layer"],
                    stacked_contribution=head_contribution_dict["stacked_contribution"]
                )
            )

    # 5. Instantiate AttributionDataBundle
    attribution_bundle = AttributionDataBundle(
        positive_attribution=raw_attribution_result_dict["attribution_positive"],
        raw_attribution=raw_attr,
        logits=raw_attribution_result_dict.get("logits"),
        ffn_activities=ffn_activity_items,
        class_embedding_representations=class_embedding_items,
        head_contribution=head_contribution_items,
    )

    # 6. Save attribution bundle to files
    saved_attribution_paths = save_attribution_bundle_to_files(image_path.stem, attribution_bundle, config.file)

    # 7. Create final ClassificationResult
    final_result = ClassificationResult(
        image_path=image_path, prediction=current_prediction, attribution_paths=saved_attribution_paths
    )

    # 8. Cache the combined result
    io_utils.save_to_cache(cache_path, final_result)

    return final_result, None


def classify_and_explain_dataset(
    config: PipelineConfig,
    vit_model: model.VisionTransformer,
    device: torch.device,
    image_paths_to_process: List[Path],
    output_results_csv_path: Path,
    steering_resources: Optional[Dict[int, Dict[str, Any]]],  # ADDED
    sae,
    class_specific_features,
    class_analysis
) -> Tuple[List[ClassificationResult], List[Dict[str, Any]]]:  # Return gradient infos too
    collected_results: List[ClassificationResult] = []
    gradient_infos: List[Dict[str, Any]] = []

    for image_path in tqdm(
        image_paths_to_process, desc=f"Classifying & Explaining (suffix: '{config.file.output_suffix}')"
    ):
        try:
            result, gradient_info = classify_explain_single_image(
                config, image_path, vit_model, device, sae, class_specific_features, steering_resources, class_analysis
            )
            collected_results.append(result)
            gradient_infos.append(gradient_info)
        except Exception as e:
            print(f"Error C&E {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if collected_results:
        io_utils.save_classification_results_to_csv(collected_results, output_results_csv_path)
    else:
        print("No images successfully C&E.")

    # Analyze gradient patterns across all images
    if gradient_infos:
        gradient_summary = collect_gradient_analysis(gradient_infos)
        if gradient_summary:
            print("\n=== GRADIENT ANALYSIS SUMMARY ===")
            print(f"Analyzed {gradient_summary['num_samples']} images with SAE boosting")
            print(f"Average Gradient Magnitude: {gradient_summary['avg_boosted_grad_magnitude']:.6f}")
            print(f"Average Grad/Act Ratio: {gradient_summary['avg_grad_to_act_ratio']:.6f}")
            print(f"Average Gradient Percentile: {gradient_summary['avg_grad_percentile']:.1f}%")
            print(f"Average Gradient Sparsity: {gradient_summary['avg_gradient_sparsity']:.3f}")

            # Save detailed gradient info to JSON for further analysis
            gradient_json_path = config.file.output_dir / f"gradient_analysis{config.file.output_suffix}.json"
            with open(gradient_json_path, 'w') as f:
                json.dump({
                    'summary':
                    gradient_summary,
                    'per_image': [{
                        'image': str(path.name),
                        **info
                    } for path, info in zip(image_paths_to_process, gradient_infos) if info]
                },
                          f,
                          indent=2,
                          default=str)
            print(f"\nDetailed gradient analysis saved to: {gradient_json_path}")

    return collected_results, gradient_infos


def classify_dataset_only(
    config: PipelineConfig, vit_model: model.VisionTransformer, device: torch.device,
    image_paths_to_process: List[Path], output_results_csv_path: Path
) -> List[ClassificationResult]:
    collected_results: List[ClassificationResult] = []
    
    
    for image_path in tqdm(image_paths_to_process, desc=f"Classifying dataset (suffix: '{config.file.output_suffix}')"):
        try:
            result = classify_single_image(config, image_path, vit_model, device)
            collected_results.append(result)
            
            
                
        except Exception as e:
            print(f"Error classifying {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    if collected_results:
        io_utils.save_classification_results_to_csv(collected_results, output_results_csv_path)
    else:
        print("No images successfully classified.")
    return collected_results


def generate_perturbed_identifier(
    config: PipelineConfig, original_image_stem: str, patch_info: PerturbationPatchInfo
) -> str:
    if config.perturb.method == "sd":
        return f"{original_image_stem}_patch{patch_info.patch_id}_x{patch_info.x}_y{patch_info.y}_s{config.perturb.strength}"
    else:
        return f"{original_image_stem}_patch{patch_info.patch_id}_x{patch_info.x}_y{patch_info.y}_{config.perturb.method}"


def perturb_single_patch(
    config: PipelineConfig,
    original_image_path: Path,
    patch_info_dc: PerturbationPatchInfo,
    sd_pipe: Optional[StableDiffusionInpaintPipeline] = None
) -> Optional[PerturbedImageRecord]:
    original_image_stem = original_image_path.stem
    patch_size = config.perturb.patch_size
    perturbed_id = generate_perturbed_identifier(config, original_image_stem, patch_info_dc)
    perturbed_image_path = config.file.perturbed_dir / f'{perturbed_id}.png'
    mask_path = config.file.mask_dir / f'{perturbed_id}_mask.npy'

    if config.file.use_cached_perturbed and perturbed_image_path.exists() and mask_path.exists():
        return PerturbedImageRecord(
            original_image_path=original_image_path,
            perturbed_image_path=perturbed_image_path,
            mask_path=mask_path,
            patch_info=patch_info_dc,
            perturbation_method=config.perturb.method,
            perturbation_strength=config.perturb.strength if config.perturb.method == "sd" else None
        )
    try:
        io_utils.ensure_directories([config.file.perturbed_dir, config.file.mask_dir])
        patch_coordinates = (patch_info_dc.x, patch_info_dc.y, patch_size)
        if config.perturb.method == "sd":
            raise Exception("do not include sd anymore")
        elif config.perturb.method == "mean":
            result_image, np_mask = perturbation.perturb_patch_mean(
                str(original_image_path), patch_coordinates, config.file
            )
        else:
            raise ValueError(f"Unknown perturbation method: {config.perturb.method}")
        result_image.save(perturbed_image_path)
        np.save(mask_path, np_mask)
        return PerturbedImageRecord(
            original_image_path=original_image_path,
            perturbed_image_path=perturbed_image_path,
            mask_path=mask_path,
            patch_info=patch_info_dc,
            perturbation_method=config.perturb.method,
            perturbation_strength=config.perturb.strength if config.perturb.method == "sd" else None
        )
    except Exception as e:
        print(f"Error perturbing patch for {original_image_stem} (ID {patch_info_dc.patch_id}): {e}")
        return None


def generate_patch_coordinates(image: Image.Image, patch_size: int) -> List[Tuple[int, int, int]]:
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
    original_image_path: Path,
    sd_pipe: Optional[StableDiffusionInpaintPipeline] = None
) -> List[PerturbedImageRecord]:
    """
    Perturb patches in a single image.
    
    Args:
        config: Pipeline configuration
        image_path: Path to the image to perturb
        sd_pipe: Optional StableDiffusionInpaintPipeline for SD method
        
    Returns:
        List of paths to perturbed images
    """
    perturbed_records: List[PerturbedImageRecord] = []
    try:
        image = preprocessing.load_image(str(original_image_path))
    except Exception as e:
        print(f"Error loading image {original_image_path} for perturbation: {e}")
        return []
    patch_coordinates_list = generate_patch_coordinates(image, config.perturb.patch_size)
    for patch_id, x, y in patch_coordinates_list:
        patch_info_dc = PerturbationPatchInfo(patch_id=patch_id, x=x, y=y)
        record = perturb_single_patch(config, original_image_path, patch_info_dc, sd_pipe)
        if record: perturbed_records.append(record)
    return perturbed_records


def perturb_dataset(
    config: PipelineConfig,
    image_paths_to_perturb: List[Path],
    sd_pipe: Optional[StableDiffusionInpaintPipeline] = None
) -> List[PerturbedImageRecord]:
    """
    Perturb patches in all images.
    
    Args:
        config: Pipeline configuration
        results_df: DataFrame with classification results
        sd_pipe: Optional StableDiffusionInpaintPipeline for SD method
        
    Returns:
        List of paths to perturbed images
    """
    io_utils.ensure_directories([config.file.perturbed_dir, config.file.mask_dir])
    all_perturbed_records: List[PerturbedImageRecord] = []
    for image_path in tqdm(image_paths_to_perturb, desc="Perturbing dataset"):
        records_for_image = perturb_image_patches(config, image_path, sd_pipe)
        all_perturbed_records.extend(records_for_image)
    print(f"Generated {len(all_perturbed_records)} perturbed image records.")
    return all_perturbed_records


def run_perturbation(config: PipelineConfig, image_paths: List[Path]) -> List[PerturbedImageRecord]:
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

    return perturb_dataset(config, image_paths, sd_pipe)


def visualize_attributions(results_df: pd.DataFrame):
    for _, row in results_df.iterrows():
        # Extract file locations from the DataFrame row
        image_path = row["image_path"]
        attribution_path = row["attribution_path"]

        # Load the original image
        original_image = Image.open(image_path)

        # Load the attribution map from the stored .npy file
        attribution_map = np.load(attribution_path)

        # Call the visualization function
        visualization.visualize_attribution_map(
            attribution_map,
            original_image,
            save_path=f'./results/vit_inputs_unweighted/{Path(image_path).stem}_visualization.png'
        )


def run_classification_standalone(
    config: PipelineConfig,
    device: torch.device,
    image_source_dir: Optional[Path] = None,
    output_suffix_override: Optional[str] = None,
    explain: bool = True
) -> List[ClassificationResult]:
    io_utils.ensure_directories(config.directories)
    vit_model = model.load_vit_model(device=device)
    print("ViT model loaded")

    if image_source_dir:
        images_to_proc = sorted(list(image_source_dir.glob("*.png")))
    else:
        images_to_proc = sorted(list(config.file.data_dir.glob("*.png")))
    print(f"Processing {len(images_to_proc)} images from {'source' if image_source_dir else 'data_dir'}.")

    orig_suffix = config.file.output_suffix
    if output_suffix_override is not None:
        config.file.output_suffix = output_suffix_override

    explain_tag = "_explained" if explain else "_classified_only"
    csv_path = config.file.output_dir / f"results{explain_tag}{config.file.output_suffix}.csv"

    if explain:
        print(f"Running C&E (suffix: '{config.file.output_suffix}')")
        results = classify_and_explain_dataset(config, vit_model, device, images_to_proc, csv_path)
    else:
        print(f"Running Classify Only (suffix: '{config.file.output_suffix}')")
        results = classify_dataset_only(config, vit_model, device, images_to_proc, csv_path)

    if output_suffix_override is not None:
        config.file.output_suffix = orig_suffix
    return results


# ===== OPTIMIZED SACO FUNCTIONS =====

def create_batched_masks(image_size: Tuple[int, int], patch_size: int, 
                        patch_coords: List[Tuple[int, int]]) -> torch.Tensor:
    """Create all patch masks for an image in a single batched tensor - MATCHES original approach exactly."""
    height, width = image_size
    num_patches = len(patch_coords)
    
    masks = torch.zeros((num_patches, height, width), dtype=torch.bool)
    
    for i, (x, y) in enumerate(patch_coords):
        # Match original approach: NO bounds checking (like original perturbation.py:149)
        # np_mask[y:y + patch_size, x:x + patch_size] = True
        y_end = y + patch_size
        x_end = x + patch_size
        
        # But we still need to prevent out-of-bounds access in tensor operations
        y_end = min(y_end, height)
        x_end = min(x_end, width)
        masks[i, y:y_end, x:x_end] = True
        
    return masks


def create_batched_masks_vectorized(image_size: Tuple[int, int], patch_size: int, 
                                   patch_coords: List[Tuple[int, int]]) -> torch.Tensor:
    """Create all patch masks using vectorized operations - much faster than loop-based approach."""
    height, width = image_size
    num_patches = len(patch_coords)
    
    if num_patches == 0:
        return torch.zeros((0, height, width), dtype=torch.bool)
    
    # Convert coordinates to tensors for vectorized operations
    coords_tensor = torch.tensor(patch_coords, dtype=torch.long)  # [num_patches, 2] (x, y)
    x_coords = coords_tensor[:, 0]  # [num_patches]
    y_coords = coords_tensor[:, 1]  # [num_patches]
    
    # Create coordinate grids
    y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    y_grid = y_grid.unsqueeze(0)  # [1, height, width]
    x_grid = x_grid.unsqueeze(0)  # [1, height, width]
    
    # Broadcast patch coordinates for comparison
    x_start = x_coords.view(-1, 1, 1)  # [num_patches, 1, 1]
    y_start = y_coords.view(-1, 1, 1)  # [num_patches, 1, 1]
    x_end = x_start + patch_size
    y_end = y_start + patch_size
    
    # Vectorized mask creation: check if each pixel is within any patch bounds
    x_in_range = (x_grid >= x_start) & (x_grid < torch.clamp(x_end, max=width))
    y_in_range = (y_grid >= y_start) & (y_grid < torch.clamp(y_end, max=height))
    masks = x_in_range & y_in_range  # [num_patches, height, width]
    
    return masks


def apply_batched_perturbations_exact_match(image_tensor: torch.Tensor, 
                                          masks: torch.Tensor, 
                                          perturbation_method: str = "mean",
                                          original_pil_image: Image.Image = None,
                                          patch_coords: List[Tuple[int, int]] = None,
                                          patch_size: int = 16) -> torch.Tensor:
    """Apply perturbations using EXACT same method as original - ensures 100% identical results."""
    num_patches, height, width = masks.shape
    channels = image_tensor.shape[0]
    
    if num_patches == 0:
        return torch.empty((0, channels, height, width), dtype=image_tensor.dtype, device=image_tensor.device)
    
    if perturbation_method == "mean" and original_pil_image is not None:
        from PIL import ImageStat
        
        # EXACT reproduction of perturbation.py:perturb_patch_mean()
        processor = preprocessing.get_processor_for_precached_224_images()
        
        # Calculate mean color once using EXACT original method  
        mean_channels = ImageStat.Stat(original_pil_image).mean
        mean_color = int(mean_channels[0])  # Take only first channel like original
        
        # Create grayscale patch template once (performance optimization)
        patch_template = Image.new("L", (patch_size, patch_size), mean_color)
        
        # Batch preprocessing: collect all PIL images first, then process together
        pil_images = []
        for i in range(num_patches):
            x, y = patch_coords[i] if patch_coords else (0, 0)
            
            # Create perturbed PIL image using EXACT original method
            result_pil = original_pil_image.copy()
            
            # Paste patch exactly like original 
            result_pil.paste(patch_template, (x, y))
            pil_images.append(result_pil)
        
        # Batch apply preprocessing (this can potentially be optimized further)
        perturbed_tensors = [processor(pil_img) for pil_img in pil_images]
        
        # Stack into batch tensor
        return torch.stack(perturbed_tensors)
            
    else:
        # Fallback: use tensor-only operations for when we don't have PIL image
        base_tensor = image_tensor.unsqueeze(0).repeat(num_patches, 1, 1, 1)
        mean_value = image_tensor.mean().item()
        
        # Apply perturbation using tensor operations
        masks_expanded = masks.unsqueeze(1).expand(-1, channels, -1, -1)
        perturbation_values = torch.full_like(base_tensor, mean_value)
        
        return torch.where(masks_expanded, perturbation_values, base_tensor)


def batched_model_inference(model_instance: model.VisionTransformer,
                         image_batch: torch.Tensor,
                         device: torch.device,
                         batch_size: int = 32) -> List[Dict]:
  """Run model inference on a batch of images efficiently."""
  model_instance.eval()
  all_predictions = []

  with torch.no_grad():
      for i in range(0, len(image_batch), batch_size):
          batch_chunk = image_batch[i:i + batch_size].to(device)

          logits = model_instance(batch_chunk)
          probabilities = torch.softmax(logits, dim=1)
          predicted_indices = torch.argmax(probabilities, dim=1)

          for j in range(len(batch_chunk)):
              pred_dict = {
                  "predicted_class_idx": predicted_indices[j].item(),
                  "probabilities": probabilities[j],
                  "confidence": probabilities[j, predicted_indices[j]].item()
              }
              all_predictions.append(pred_dict)

  return all_predictions


def calculate_saco_vectorized(attributions: np.ndarray, confidence_impacts: np.ndarray) -> float:
    """
    Vectorized SaCo calculation - eliminates O(n²) nested loops.
    
    Args:
        attributions: Array of attribution values per patch (already sorted descending)
        confidence_impacts: Array of confidence impact values per patch
        
    Returns:
        SaCo score
    """
    n = len(attributions)
    if n < 2:
        return 0.0
    
    # Create all pairwise differences using broadcasting
    attr_diffs = attributions[:, None] - attributions[None, :]  # [n, n]
    impact_diffs = confidence_impacts[:, None] - confidence_impacts[None, :]  # [n, n]
    
    # Only consider upper triangular part (i < j comparisons)
    upper_tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Extract upper triangular values
    attr_diffs_tri = attr_diffs[upper_tri_mask]  # [n*(n-1)/2]
    impact_diffs_tri = impact_diffs[upper_tri_mask]  # [n*(n-1)/2]
    
    # Calculate faithfulness: impact_i >= impact_j (since we're looking at i < j)
    is_faithful = impact_diffs_tri >= 0
    
    # Calculate weights
    weights = np.where(is_faithful, attr_diffs_tri, -attr_diffs_tri)
    
    # Calculate SaCo score
    total_weight = np.sum(np.abs(weights))
    if total_weight > 0:
        return np.sum(weights) / total_weight
    else:
        return 0.0


def calculate_patch_saco_vectorized(attributions: np.ndarray, confidence_impacts: np.ndarray, patch_ids: np.ndarray) -> Dict[int, float]:
    """
    Vectorized calculation of patch-specific SaCo scores.
    
    Args:
        attributions: Array of attribution values per patch (already sorted descending)
        confidence_impacts: Array of confidence impact values per patch  
        patch_ids: Array of patch IDs corresponding to each attribution/impact
        
    Returns:
        Dictionary mapping patch_id to patch_saco_score
    """
    n = len(attributions)
    if n < 2:
        return {int(pid): 0.0 for pid in patch_ids}
    
    # First compute all pairwise data vectorized
    attr_diffs = attributions[:, None] - attributions[None, :]  # [n, n]
    impact_diffs = confidence_impacts[:, None] - confidence_impacts[None, :]  # [n, n]
    
    # Only consider upper triangular part (i < j comparisons)
    upper_tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Extract pair data
    i_indices, j_indices = np.where(upper_tri_mask)
    attr_diffs_pairs = attr_diffs[upper_tri_mask]
    impact_diffs_pairs = impact_diffs[upper_tri_mask]
    
    # Calculate faithfulness and weights
    is_faithful = impact_diffs_pairs >= 0
    weights = np.where(is_faithful, attr_diffs_pairs, -attr_diffs_pairs)
    
    # Get patch IDs for each pair
    patch_i_ids = patch_ids[i_indices]
    patch_j_ids = patch_ids[j_indices]
    
    # Calculate patch-specific SaCo scores
    patch_saco_scores = {}
    unique_patch_ids = np.unique(patch_ids)
    
    for patch_id in unique_patch_ids:
        # Find pairs involving this patch
        involved_as_i = (patch_i_ids == patch_id)
        involved_as_j = (patch_j_ids == patch_id)
        involved = involved_as_i | involved_as_j
        
        if not np.any(involved):
            patch_saco_scores[int(patch_id)] = 0.0
            continue
        
        # Get weights for this patch's pairs
        patch_weights = weights[involved]
        
        # Calculate patch-specific SaCo
        sum_weights = np.sum(patch_weights)
        sum_abs_weights = np.sum(np.abs(patch_weights))
        
        if sum_abs_weights > 0:
            patch_saco_scores[int(patch_id)] = sum_weights / sum_abs_weights
        else:
            patch_saco_scores[int(patch_id)] = 0.0
    
    return patch_saco_scores




def calculate_saco_for_single_image(
    original_result: ClassificationResult,
    vit_model: model.VisionTransformer,
    config: PipelineConfig,
    device: torch.device,
    max_patches_per_batch: int = 512
) -> List[Dict]:
    """Calculate SaCo metrics for a single image using optimized batched operations."""
    image_path = original_result.image_path
    
    # Load and preprocess original image once
    original_pil, original_tensor = preprocessing.preprocess_image(
        str(image_path), img_size=config.classify.target_size[0]
    )
    
    # Load attribution map once
    attribution_path = original_result.attribution_paths.attribution_path
    attribution_map = np.load(attribution_path)
    
    # Generate all patch coordinates - MATCH ORIGINAL APPROACH EXACTLY
    width, height = original_pil.size
    patch_coords = []
    patch_infos = []
    
    # Use SAME patch ID calculation as original approach (generate_patch_coordinates)
    num_patches_x = width // config.perturb.patch_size
    
    for y in range(0, height - config.perturb.patch_size + 1, config.perturb.patch_size):
        for x in range(0, width - config.perturb.patch_size + 1, config.perturb.patch_size):
            # Match original: patch_id = (y // patch_size) * num_patches_x + (x // patch_size)
            patch_id = (y // config.perturb.patch_size) * num_patches_x + (x // config.perturb.patch_size)
            patch_coords.append((x, y))
            patch_infos.append(PerturbationPatchInfo(patch_id=patch_id, x=x, y=y))
    
    if not patch_coords:
        return []
    
    # Create all masks in batch - use vectorized approach for performance
    masks = create_batched_masks_vectorized(
        image_size=(height, width),
        patch_size=config.perturb.patch_size,
        patch_coords=patch_coords
    )
    
    # Memory-efficient processing: handle large patch counts by batching
    if len(patch_coords) > max_patches_per_batch:
        # Process in chunks to avoid memory issues
        all_predictions = []
        for i in range(0, len(patch_coords), max_patches_per_batch):
            end_idx = min(i + max_patches_per_batch, len(patch_coords))
            batch_masks = masks[i:end_idx]
            
            # Apply perturbations for this batch using exact match method
            batch_coords = patch_coords[i:end_idx]
            batch_perturbed = apply_batched_perturbations_exact_match(
                original_tensor, batch_masks, config.perturb.method, original_pil, 
                batch_coords, config.perturb.patch_size
            )
            
            # Run inference for this batch
            batch_predictions = batched_model_inference(
                vit_model, batch_perturbed, device, batch_size=min(256, len(batch_perturbed))
            )
            all_predictions.extend(batch_predictions)
            
            # Clear GPU memory for this batch to prevent accumulation
            del batch_perturbed, batch_masks
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        perturbed_predictions = all_predictions
    else:
        # Apply all perturbations using exact match approach
        perturbed_batch = apply_batched_perturbations_exact_match(
            original_tensor, masks, config.perturb.method, original_pil,
            patch_coords, config.perturb.patch_size
        )
        
        # Run batched inference but ensure compatibility with original approach
        perturbed_predictions = batched_model_inference(
            vit_model, perturbed_batch, device, batch_size=min(1024, len(perturbed_batch))
        )
    
    
    # Calculate results for each patch
    results = []
    original_pred = original_result.prediction
    
    for i, (patch_info, (x, y)) in enumerate(zip(patch_infos, patch_coords)):
        perturbed_pred = perturbed_predictions[i]
        
        # Calculate mean attribution in patch
        y_end = min(y + config.perturb.patch_size, attribution_map.shape[0])
        x_end = min(x + config.perturb.patch_size, attribution_map.shape[1])
        patch_region = attribution_map[y:y_end, x:x_end]
        mean_patch_attr = float(np.mean(patch_region))
        
        
        result = {
            "original_image": str(image_path),
            "patch_id": patch_info.patch_id,
            "x": patch_info.x,
            "y": patch_info.y,
            "mean_attribution": mean_patch_attr,
            "original_predicted_idx": original_pred.predicted_class_idx,
            "original_confidence": original_pred.confidence,
            "perturbed_predicted_idx": perturbed_pred["predicted_class_idx"],
            "perturbed_confidence": perturbed_pred["confidence"],
            "class_changed": (original_pred.predicted_class_idx != perturbed_pred["predicted_class_idx"]),
            "confidence_delta": perturbed_pred["confidence"] - original_pred.confidence,
            "confidence_delta_abs": abs(perturbed_pred["confidence"] - original_pred.confidence)
        }
        results.append(result)
        
        
    return results


def run_saco_from_pipeline_outputs_optimized(
    config: PipelineConfig,
    original_pipeline_results: List[ClassificationResult],
    vit_model: VisionTransformer,
    device: torch.device,
    generate_visualizations: bool = False,
    save_analysis_results: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run SaCo analysis using optimized batched processing and in-memory operations.
    """
    print("=== SACO ANALYSIS ===")
    saco_analysis_results: Dict[str, pd.DataFrame] = {}
    
    # Run SaCo calculation for all images
    all_perturbation_results = []
    print("Running SaCo calculations...")
    
    for original_result in tqdm(original_pipeline_results, desc="Processing images"):
        try:
            image_results = calculate_saco_for_single_image(
                original_result, vit_model, config, device
            )
            all_perturbation_results.extend(image_results)
        except Exception as e:
            print(f"Error processing {original_result.image_path.name}: {e}")
            continue
    
    # Convert to DataFrame
    perturbation_comparison_df = pd.DataFrame(all_perturbation_results)
    
    # Calculate SaCo scores and patch-level analysis data using vectorized approach
    saco_scores = {}
    patch_analysis_rows = []  # For the patch-level CSV that saco_feature_analysis.py expects
    
    for image_name, image_data in perturbation_comparison_df.groupby('original_image'):
        image_data = image_data.sort_values('mean_attribution', ascending=False).reset_index(drop=True)
        
        attributions = image_data['mean_attribution'].values
        confidence_impacts = image_data['confidence_delta_abs'].values
        patch_ids = image_data['patch_id'].values
        
        # Calculate overall SaCo score using vectorized approach
        saco_score = calculate_saco_vectorized(attributions, confidence_impacts)
        saco_scores[image_name] = saco_score
        # Calculate patch-specific SaCo scores using vectorized approach  
        patch_sacos = calculate_patch_saco_vectorized(attributions, confidence_impacts, patch_ids)
        
        # Add patch data rows
        for patch_id, patch_saco in patch_sacos.items():
            patch_analysis_rows.append({
                'image_name': image_name,
                'patch_id': int(patch_id),
                'faithful_pairs_count': 0,  # Not needed for saco_feature_analysis.py
                'unfaithful_pairs_count': 0,  # Not needed for saco_feature_analysis.py
                'faithful_pairs_pct': 0,  # Not needed for saco_feature_analysis.py
                'patch_saco': patch_saco
            })
    
    saco_df = pd.DataFrame({'image_name': list(saco_scores.keys()), 'saco_score': list(saco_scores.values())})
    saco_analysis_results["saco_scores"] = saco_df
    
    # Create and save patch analysis DataFrame (compatible with saco_feature_analysis.py)
    patch_analysis_df = pd.DataFrame(patch_analysis_rows)
    saco_analysis_results["patch_analysis"] = patch_analysis_df
    
    saco_scores_map: Dict[str, float] = pd.Series(
        saco_analysis_results["saco_scores"].saco_score.values, 
        index=saco_analysis_results["saco_scores"].image_name
    ).to_dict()
    
    faithfulness_df = analysis.analyze_faithfulness_vs_correctness_from_objects(
        saco_scores_map, original_pipeline_results
    )
    saco_analysis_results["faithfulness_correctness"] = faithfulness_df
    
    patterns_df = analysis.analyze_key_attribution_patterns(
        saco_analysis_results["faithfulness_correctness"], vit_model, config
    )
    saco_analysis_results["attribution_patterns"] = patterns_df
    
    if save_analysis_results:
        print("Saving analysis results...")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        for name, df_to_save in saco_analysis_results.items():
            if isinstance(df_to_save, pd.DataFrame) and not df_to_save.empty:
                if name == "patch_analysis":
                    # Save patch analysis with the expected filename format
                    save_path = config.file.output_dir / f"saco_patch_analysis_mean_optimized_{timestamp}.csv"
                else:
                    save_path = config.file.output_dir / f"analysis_{name}_optimized_{timestamp}.csv"
                df_to_save.to_csv(save_path, index=False)
                print(f"Saved {name} to {save_path}")
                
                # Also save the main patch analysis with the exact filename expected by saco_feature_analysis.py
                if name == "patch_analysis":
                    expected_path = config.file.output_dir / "saco_patch_analysis_mean.csv"
                    df_to_save.to_csv(expected_path, index=False)
                    print(f"Also saved patch analysis to expected path: {expected_path}")
    
    avg_saco = np.mean(list(saco_scores.values())) if saco_scores else 0
    print(f"Average optimized SaCo score: {avg_saco:.4f} (over {len(saco_scores)} images)")
    
    print("=== OPTIMIZED SACO ANALYSIS COMPLETE ===")
    return saco_analysis_results


def run_saco_from_pipeline_outputs(
    config: PipelineConfig,
    original_pipeline_results: List[ClassificationResult],
    vit_model: VisionTransformer,
    device: torch.device,
    generate_visualizations: bool = False,
    save_analysis_results: bool = True
):
    """
    Run SaCo (Saliency Correlation) analysis using direct outputs from the pipeline.
    
    Args:
        pipeline_config: The configuration used for the pipeline run (for paths, params).
        original_pipeline_results: List of ClassificationResult for original images.
        all_perturbed_image_records: List of PerturbedImageRecord for all generated perturbations.
        perturbed_pipeline_results: List of ClassificationResult for perturbed images.
        model_instance: The loaded VisionTransformer model, if needed for pattern analysis.
        generate_visualizations: Whether to generate comparison visualizations.
        save_analysis_results: Whether to save all generated analysis DataFrames.
        
    Returns:
        Dictionary with analysis DataFrames.
    """
    print("Perturbations for SaCo")
    paths_for_perturbation = [res.image_path for res in original_pipeline_results]

    perturbed_image_records: List[PerturbedImageRecord] = run_perturbation(config, paths_for_perturbation)

    perturbed_image_paths_to_classify = [
        rec.perturbed_image_path for rec in perturbed_image_records if rec.perturbed_image_path.exists()
    ]

    perturbed_csv_path = config.file.output_dir / f"classification_results_perturbed_classified_only{config.file.output_suffix}_perturbed.csv"
    print(f"Running Classify ONLY for perturbed images (suffix: '{config.file.output_suffix}_perturbed')")

    perturbed_pipeline_results = classify_dataset_only(
        config, vit_model, device, perturbed_image_paths_to_classify, perturbed_csv_path
    )

    saco_analysis_results: Dict[str, pd.DataFrame] = {}

    print("Building analysis context...")
    analysis_context = analysis.AnalysisContext.build(
        config=config,
        original_results=original_pipeline_results,
        all_perturbed_records=perturbed_image_records,
        perturbed_classification_results=perturbed_pipeline_results
    )

    print("Generating perturbation comparison DataFrame for SaCo...")
    perturbation_comparison_df = analysis.generate_perturbation_comparison_dataframe(
        analysis_context, generate_visualizations=generate_visualizations
    )

    print("Running core SaCo calculations...")
    saco_scores_dict, _, _ = analysis.run_saco_analysis(
        context=analysis_context,
        perturbation_comparison_df=perturbation_comparison_df,
        perturb_method_filter=config.perturb.method
    )

    saco_df = pd.DataFrame({'image_name': list(saco_scores_dict.keys()), 'saco_score': list(saco_scores_dict.values())})
    saco_analysis_results["saco_scores"] = saco_df

    saco_scores_map_for_analysis: Dict[str, float] = pd.Series(
        saco_analysis_results["saco_scores"].saco_score.values, index=saco_analysis_results["saco_scores"].image_name
    ).to_dict()

    faithfulness_df = analysis.analyze_faithfulness_vs_correctness_from_objects(
        saco_scores_map_for_analysis,  # Pass the dictionary: {str(image_path): score}
        original_pipeline_results  # Pass the List[ClassificationResult]
    )
    saco_analysis_results["faithfulness_correctness"] = faithfulness_df

    print("Analyzing key attribution patterns...")
    patterns_df = analysis.analyze_key_attribution_patterns(
        saco_analysis_results["faithfulness_correctness"], vit_model, config
    )
    saco_analysis_results["attribution_patterns"] = patterns_df

    if save_analysis_results:
        print("Saving analysis results...")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        for name, df_to_save in saco_analysis_results.items():
            if isinstance(df_to_save, pd.DataFrame) and not df_to_save.empty:
                # Use a consistent naming convention, incorporating the mode and output_file_tag
                save_path = config.file.output_dir / f"analysis_{name}_{timestamp}.csv"
                df_to_save.to_csv(save_path, index=False)
                print(f"Saved {name} to {save_path}")


# Add this function to collect gradient info across images
def collect_gradient_analysis(gradient_info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate gradient analysis across multiple images to identify patterns."""
    if not gradient_info_list:
        return {}

    # Filter out empty gradient info
    valid_infos = [info for info in gradient_info_list if info and 'boosted_feature_idx' in info]
    if not valid_infos:
        return {}

    # Compute aggregated statistics
    aggregated = {
        'num_samples': len(valid_infos),
        'avg_boosted_grad_magnitude': np.mean([info['boosted_grad_magnitude'] for info in valid_infos]),
        'avg_boosted_activation_strength': np.mean([info['boosted_activation_strength'] for info in valid_infos]),
        'avg_grad_to_act_ratio': np.mean([info['boosted_grad_to_act_ratio'] for info in valid_infos]),
        'avg_grad_percentile': np.mean([info['boosted_grad_percentile'] for info in valid_infos]),
        'avg_gradient_sparsity': np.mean([info['gradient_sparsity'] for info in valid_infos]),
        'feature_distribution': defaultdict(int)
    }

    # Count which features were boosted
    for info in valid_infos:
        aggregated['feature_distribution'][info['boosted_feature_idx']] += 1

    return aggregated


def run_pipeline(config: PipelineConfig,
                 source_dir_for_preprocessing: Path,
                 device: Optional[torch.device] = None) -> List[ClassificationResult]:
    """
    Streamlined full pipeline:
    1. (Optional) Preprocess dataset.
    2. Classify AND Explain original images.
    3. Run perturbation on original images.
    4. Classify ONLY perturbed images.
    Returns Tuple[List[ClassificationResult] (originals, explained), List[ClassificationResult] (perturbed, classified_only)].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    io_utils.ensure_directories(config.directories)

    original_image_paths = preprocess_dataset(config, source_dir_for_preprocessing)
    print(f"Found {len(original_image_paths)} original images for processing.")

    # vit_model = model.load_vit_model(model_path="./model/vit_b_hyperkvasir_anatomical_for_translrp.pth", device=device)
    sae, vit_model = load_models()
    # class_specific_features = find_class_specific_features(vit_model, sae)
    # 2. Load all steering resources (SAEs and Dictionaries) ONCE
    # You can control which layers to use from your config file
    steering_layers_from_config = getattr(config.classify, 'steering_layers', [8])
    steering_resources = load_steering_resources(steering_layers_from_config)
    print("All steering resources loaded.")
    class_analysis = None
    print("ViT model loaded, hooks registered, and set to eval mode.")

    originals_csv_path = config.file.output_dir / f"classification_results_originals_explained{config.file.output_suffix}.csv"
    print(f"Running Classify & Explain for original images")

    original_results_explained, gradient_infos = classify_and_explain_dataset(
        config, vit_model, device, original_image_paths, originals_csv_path, steering_resources, sae, None,
        class_analysis
    )

    # Quick hypothesis test: Check if low gradient ratios correlate with attribution improvement
    if gradient_infos:
        low_ratio_threshold = 0.1  # Adjust based on your data
        low_ratio_features = [
            info for info in gradient_infos if info and info.get('boosted_grad_to_act_ratio', 1.0) < low_ratio_threshold
        ]
        print(f"\n{len(low_ratio_features)} images had low grad/act ratio (<{low_ratio_threshold})")
        print("These might be cases where gradient-based attribution struggles without boosting")

    if config.classify.analysis:
        print("Evaluate Faithfulness Pipeline")
        evaluate_and_report_faithfulness(config, vit_model, device, original_results_explained)

    # print("Run Head Analysis")
    # run_head_analysis(original_results_explained, vit_model, config)

    print("Compare attributions")
    # For debugging, let's run both approaches on dev dataset to compare
    if "dev" in str(config.file.data_dir).lower():
        print("DEV MODE: Running both original and optimized approaches for comparison")
        # Initialize patch logger for comparison
        # initialize_patch_logger(config.file.output_dir / "patch_comparison_logs")
        print("\n--- Running Original Approach ---")
        # run_saco_from_pipeline_outputs(config, original_results_explained, vit_model, device)
        print("\n--- Running Optimized Approach ---") 
        run_saco_from_pipeline_outputs(config, original_results_explained, vit_model, device)
    else:
        # For production (val dataset), use optimized version only
        run_saco_from_pipeline_outputs_optimized(config, original_results_explained, vit_model, device)

    print("Full pipeline finished.")
    return original_results_explained
