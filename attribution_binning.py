"""
Corrected Attribution Binning Implementation for SaCo

This implements the binning approach as described in the SaCo paper:
- Group patches by attribution value ranges (not spatial location)
- Perturb all patches within each attribution bin together
- Calculate SaCo based on bin comparisons
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from data_types import ClassificationResult
import vit.preprocessing as preprocessing
import analysis

def batched_model_inference(model_instance, image_batch: torch.Tensor, device: torch.device, batch_size: int = 32) -> List[Dict]:
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

@dataclass
class BinInfo:
    """Information about an attribution bin"""
    bin_id: int
    min_value: float
    max_value: float
    patch_indices: List[int]  # Which of the 196 patches belong to this bin
    mean_attribution: float
    total_attribution: float
    n_patches: int


def create_attribution_bins_from_patches(
    raw_attributions: np.ndarray,  # Shape: (196,)
    n_bins: int = 20
) -> List[BinInfo]:
    """
    Create bins based on attribution value ranges, ensuring equal sizes.
    
    This groups the 196 patches into n_bins based on their attribution values.
    It sorts the patches by attribution and divides them into equal-sized chunks,
    which is a more direct implementation of the SaCo paper's description of
    "equally sized pixel subsets".
    
    Args:
        raw_attributions: Array of 196 attribution values (one per ViT patch)
        n_bins: Number of bins to create
        
    Returns:
        List of BinInfo objects describing each bin
    """
    if n_bins > len(raw_attributions):
        n_bins = len(raw_attributions)

    # Sort patch indices based on their attribution values (descending)
    sorted_patch_indices = np.argsort(raw_attributions)[::-1]
    
    # Split the sorted indices into n_bins of roughly equal size
    binned_indices = np.array_split(sorted_patch_indices, n_bins)

    bins = []
    # Bins are created from highest attribution to lowest, so we sort them
    # by their mean attribution to keep a consistent bin_id order later.
    temp_bins = []
    for patch_indices_list in binned_indices:
        patch_indices = patch_indices_list.tolist()
        if len(patch_indices) == 0:
            continue
        bin_attributions = raw_attributions[patch_indices]
        temp_bins.append({
            "indices": patch_indices,
            "attributions": bin_attributions
        })
    
    # Sort bins by their mean attribution value to assign IDs consistently
    # (lowest attribution bin gets id 0)
    temp_bins.sort(key=lambda b: np.mean(b['attributions']))

    for bin_id, b_data in enumerate(temp_bins):
        patch_indices = b_data['indices']
        bin_attributions = b_data['attributions']
        
        bin_info = BinInfo(
            bin_id=bin_id,
            min_value=float(np.min(bin_attributions)),
            max_value=float(np.max(bin_attributions)),
            patch_indices=patch_indices,
            mean_attribution=float(np.mean(bin_attributions)),
            total_attribution=float(np.sum(bin_attributions)), # This is s(Gi) from the paper
            n_patches=len(patch_indices)
        )
        bins.append(bin_info)
    
    return bins

def create_spatial_mask_for_bin(
    bin_info: BinInfo,
    image_size: Tuple[int, int] = (224, 224),
    patch_size: int = 16
) -> torch.Tensor:
    """
    Create a spatial mask for all patches in a bin.
    
    Args:
        bin_info: Information about the bin
        image_size: Size of the image (height, width)
        patch_size: Size of each ViT patch (16x16 for standard ViT)
        
    Returns:
        Boolean mask of shape (H, W)
    """
    height, width = image_size
    mask = torch.zeros((height, width), dtype=torch.bool)
    
    # Each patch index corresponds to a position in the 14x14 grid
    grid_size = 14  # 224 / 16 = 14
    
    for patch_idx in bin_info.patch_indices:
        # Convert linear index to grid position
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        
        # Convert to pixel coordinates
        y_start = row * patch_size
        y_end = min(y_start + patch_size, height)
        x_start = col * patch_size
        x_end = min(x_start + patch_size, width)
        
        mask[y_start:y_end, x_start:x_end] = True
    
    return mask


def apply_binned_perturbation(
    original_tensor: torch.Tensor,
    bin_mask: torch.Tensor,
    perturbation_method: str = "mean",
    original_pil_image=None
) -> torch.Tensor:
    """
    Apply perturbation to all patches in a bin.
    
    Uses grayscale perturbation method for consistency with patch-wise approach.
    """
    if perturbation_method == "mean":
        if original_pil_image is not None:
            # Use consistent grayscale perturbation method
            from PIL import Image, ImageStat
            import vit.preprocessing as preprocessing # Make sure this import is available
            
            # 1. Calculate mean, but take only the first channel for grayscale value
            mean_channels = ImageStat.Stat(original_pil_image).mean
            mean_color_value = int(mean_channels[0])

            # 2. Create a full-size RGB layer with the mean value
            # We create a large mask and then apply it, which is more efficient
            # than creating and pasting many small patches.
            # Use RGB mode to match the original image
            grayscale_layer = Image.new("RGB", original_pil_image.size, (mean_color_value, mean_color_value, mean_color_value))

            # 3. Create a perturbed PIL image by pasting the mean value only where the mask is True
            # The mask needs to be converted to a PIL image to be used here.
            # The bin_mask is a Tensor, so we convert it.
            pil_mask = Image.fromarray(bin_mask.numpy().astype('uint8') * 255, mode='L')
            
            result_pil = original_pil_image.copy()
            result_pil.paste(grayscale_layer, (0, 0), mask=pil_mask)

            # 4. Preprocess the final perturbed PIL image back to a tensor
            # Check the size of the result_pil to determine which processor to use
            if result_pil.size == (224, 224):
                processor = preprocessing.get_processor_for_precached_224_images()
            else:
                processor = preprocessing.get_default_processor(img_size=224)
            perturbed_tensor = processor(result_pil)
            return perturbed_tensor
        else:
            # Fallback tensor-only logic (this part will likely not be used but is good to have)
            perturbed = original_tensor.clone()
            mean_value = original_tensor[0].mean().item() # Grayscale-like mean from first channel
            for c in range(original_tensor.shape[0]):
                perturbed[c][bin_mask] = mean_value
            return perturbed

    # Default case if method is not "mean"
    return original_tensor.clone()

def calculate_binned_saco_for_image(
    original_result: ClassificationResult,
    vit_model,
    config,
    device: torch.device,
    n_bins: int = 20,
    debug: bool = False,
    include_patch_level: bool = False
) -> Tuple[float, List[Dict], List[Dict]]:
    """
    Calculate SaCo score using attribution binning for a single image.
    
    Args:
        include_patch_level: If True, also generate patch-level SaCo scores for compatibility
    
    Returns:
        Tuple of (saco_score, bin_results, patch_results)
    """
    image_path = original_result.image_path
    
    # Load image and attribution
    original_pil, original_tensor = preprocessing.preprocess_image(
        str(image_path), img_size=config.classify.target_size[0]
    )
    
    # Load raw attribution (196 values, one per patch)
    raw_attribution_path = original_result.attribution_paths.raw_attribution_path
    raw_attributions = np.load(raw_attribution_path)  # Shape: (196,)
    
    # Create bins based on attribution values
    bins = create_attribution_bins_from_patches(raw_attributions, n_bins)
    
    if debug and "dev_0aec60bd" in str(image_path):
        print(f"\nBINNED DEBUG - Image: {image_path.name}")
        print(f"Number of bins created: {len(bins)}")
        print(f"Raw attributions shape: {raw_attributions.shape}")
        print(f"Attribution range: [{raw_attributions.min():.4f}, {raw_attributions.max():.4f}]")
    
    # Apply perturbations for each bin
    perturbed_tensors = []
    valid_bins = []
    
    for bin_info in bins:
        # Create spatial mask for this bin
        mask = create_spatial_mask_for_bin(bin_info)
        
        # Apply perturbation
        perturbed = apply_binned_perturbation(
            original_tensor, mask, config.perturb.method, original_pil
        )
        
        perturbed_tensors.append(perturbed)
        valid_bins.append(bin_info)
    
    # Batch inference
    if not perturbed_tensors:
        return 0.0, []
    
    batch_tensor = torch.stack(perturbed_tensors)
    predictions = batched_model_inference(vit_model, batch_tensor, device, batch_size=len(batch_tensor))
    
    # Calculate results
    bin_results = []
    original_pred = original_result.prediction
    
    for bin_info, pred in zip(valid_bins, predictions):
        confidence_impact = pred["confidence"] - original_pred.confidence
        
        result = {
            "bin_id": bin_info.bin_id,
            "mean_attribution": bin_info.mean_attribution,
            "total_attribution": bin_info.total_attribution,
            "n_patches": bin_info.n_patches,
            "confidence_delta": confidence_impact,
            "confidence_delta_abs": abs(confidence_impact),
            "class_changed": (pred["predicted_class_idx"] != original_pred.predicted_class_idx)
        }
        bin_results.append(result)
    
    # Sort by mean attribution (descending)
    bin_results.sort(key=lambda x: x["total_attribution"], reverse=True)
    
    # Calculate bin-level SaCo scores (individual score per bin)
    if len(bin_results) >= 2:
        attributions = np.array([r["total_attribution"] for r in bin_results])
        impacts = np.array([r["confidence_delta_abs"] for r in bin_results])
        bin_ids = np.array([r["bin_id"] for r in bin_results])
        
        # Calculate bin-level SaCo score using vectorized approach  
        bin_saco_score = calculate_saco_vectorized(attributions, impacts)
        
        # Add bin-level SaCo scores to results (same score for all bins in this image)
        for result in bin_results:
            result["bin_saco_score"] = bin_saco_score
    else:
        # If only one bin, SaCo score is 0
        for result in bin_results:
            result["bin_saco_score"] = 0.0
    
    # Extract arrays for overall image SaCo calculation
    attributions = np.array([r["total_attribution"] for r in bin_results])
    impacts = np.array([r["confidence_delta_abs"] for r in bin_results])
    
    if debug and "dev_0aec60bd" in str(image_path):
        print(f"BINNED - Number of bins: {len(bin_results)}")
        print(f"BINNED - Attributions[:5]: {attributions[:5]}")
        print(f"BINNED - Impacts[:5]: {impacts[:5]}")
        print(f"BINNED - Attribution sum: {np.sum(attributions):.4f}")
        print(f"BINNED - Impact sum: {np.sum(impacts):.4f}")
    
    # Calculate overall image SaCo score using the same function as patch-wise
    saco_score = calculate_saco_vectorized(attributions, impacts)
    
    # Generate patch-level results if requested (for compatibility with saco_feature_analysis.py)
    patch_results = []
    if include_patch_level:
        # For patch-level compatibility, we need to calculate individual patch SaCo scores
        # This requires creating perturbations for each of the 196 patches individually
        patch_results = calculate_patch_level_saco(
            original_result, vit_model, config, device, raw_attributions
        )
    
    return saco_score, bin_results, patch_results


def _get_or_compute_binned_results(
    config: Any,
    original_results: List[ClassificationResult],
    vit_model: Any,
    device: torch.device,
    n_bins: int,
) -> pd.DataFrame:
    """
    Gets binned SaCo results by either loading from cache or computing them.

    If results are computed from scratch, they are automatically saved to the
    cache file for future use (if save_on_compute is True).

    Returns:
        A pd.DataFrame containing the detailed bin results.
    """

    # --- ATTEMPT TO LOAD FROM CACHE ---
    if config.file.use_cached_perturbed:
        cache_path = config.file.output_dir / config.file.use_cached_perturbed
        print(f"Attempting to load cached results from: {cache_path}")
        if cache_path.exists():
            print(f"Cache file found! Loading...")
            return pd.read_csv(cache_path)
        else:
            print(f"Cache file not found. Proceeding with computation.")

    # --- COMPUTE FROM SCRATCH ---
    print(f"Computing binned SaCo results for {len(original_results)} images...")
    all_bin_results = []
    
    for original_result in tqdm(original_results, desc=f"Processing with {n_bins} bins"):
        try:
            saco_score, bin_results, _ = calculate_binned_saco_for_image(
                original_result, vit_model, config, device, n_bins
            )
            for result in bin_results:
                result["image_name"] = str(original_result.image_path)
                result["saco_score"] = saco_score
                all_bin_results.append(result)
        except Exception as e:
            import traceback
            print(f"Error processing {original_result.image_path.name}: {e}")
            if "do not match" in str(e):
                traceback.print_exc()
            continue
            
    bin_results_df = pd.DataFrame(all_bin_results)

    return bin_results_df


def run_binned_saco_analysis(
    config,
    original_results: List[ClassificationResult],
    vit_model,
    device: torch.device,
    n_bins: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    Run binned SaCo analysis for entire dataset, with optional caching.

    This function first gets the raw data (either by computing it or loading
    from a cache), and then performs all subsequent analysis steps.
    It will also save derived analysis files (e.g., faithfulness).

    Returns:
        A dictionary with analysis results as DataFrames.
    """
    print(f"=== BINNED SACO ANALYSIS (n_bins={n_bins}) ===")
    
    # 1. Get data by either loading from cache or computing (and saving).
    bin_results_df = _get_or_compute_binned_results(
        config, original_results, vit_model, device, n_bins,
    )

    if bin_results_df.empty:
        print("No results were generated or loaded. Aborting analysis.")
        return {}

    # --- START OF ANALYSIS SECTION ---
    print("\n--- Performing post-hoc analysis on results ---")
    analysis_results = {"bin_results": bin_results_df}
    
    # 2. Derive SaCo scores DataFrame
    saco_df = bin_results_df[['image_name', 'saco_score']].drop_duplicates().reset_index(drop=True)
    analysis_results["saco_scores"] = saco_df
    
    # 3. Faithfulness vs Correctness Analysis
    saco_scores_map = pd.Series(saco_df.saco_score.values, index=saco_df.image_name).to_dict()
    faithfulness_df = analysis.analyze_faithfulness_vs_correctness_from_objects(saco_scores_map, original_results)
    analysis_results["faithfulness_correctness"] = faithfulness_df
    
    # 4. Key Attribution Patterns Analysis
    patterns_df = analysis.analyze_key_attribution_patterns(analysis_results["faithfulness_correctness"], vit_model, config)
    analysis_results["attribution_patterns"] = patterns_df
    
    # 5. Summary statistics
    if not saco_df.empty:
        avg_saco = saco_df['saco_score'].mean()
        std_saco = saco_df['saco_score'].std()
        print(f"\nBinned SaCo Analysis Summary:")
        print(f"  Number of images: {len(saco_df)}")
        print(f"  Number of bins: {n_bins}")
        print(f"  Average SaCo: {avg_saco:.4f}")
        print(f"  Std SaCo: {std_saco:.4f}")
    
    print("\nSaving derived analysis files...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    for name, df_to_save in analysis_results.items():
        if isinstance(df_to_save, pd.DataFrame) and not df_to_save.empty:
            if name == "bin_results":
                # Save bin results with dataset prefix for saco_feature_analysis
                dataset_name = config.file.output_dir.name  # Assumes output_dir ends with dataset name
                save_path = config.file.output_dir / f"{dataset_name}_bin_results.csv"
                df_to_save.to_csv(save_path, index=False)
                print(f"Saved bin results to {save_path}")
            
            # Also save with timestamp for archival
            save_path = config.file.output_dir / f"analysis_{name}_binned_{timestamp}.csv"
            df_to_save.to_csv(save_path, index=False)
            print(f"Saved {name} to {save_path}")

    print("=== BINNED SACO ANALYSIS COMPLETE ===")
    return analysis_results

# For easy integration with your pipeline, add this wrapper function:
def run_binned_attribution_analysis(
    config,
    vit_model,
    original_results: List[ClassificationResult],
    device: torch.device,
    n_bins: int = 20
) -> Dict[str, pd.DataFrame]:
    """
    Wrapper function that matches your pipeline interface.
    """
    vit_model.to(device)
    vit_model.eval()
    
    # Run binned analysis
    results = run_binned_saco_analysis(
        config, original_results, vit_model, device, n_bins
    )
    return results    
