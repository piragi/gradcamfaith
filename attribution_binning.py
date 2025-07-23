"""
Corrected Attribution Binning Implementation for SaCo

This implements the binning approach as described in the SaCo paper:
- Group patches by attribution value ranges (not spatial location)
- Perturb all patches within each attribution bin together
- Calculate SaCo based on bin comparisons
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd

from data_types import ClassificationResult
import vit.preprocessing as preprocessing
import pipeline

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
    
    *** QUICK TEST VERSION ***
    This version uses the EXACT SAME GRAYSCALE PERTURBATION as the 
    'optimized' patch-wise pipeline to check for consistency.
    """
    if perturbation_method == "mean":
        if original_pil_image is not None:
            # This logic is now identical to pipeline.py's apply_batched_perturbations_exact_match
            from PIL import Image, ImageStat
            import vit.preprocessing as preprocessing # Make sure this import is available
            
            # 1. Calculate mean, but take only the first channel for grayscale value
            mean_channels = ImageStat.Stat(original_pil_image).mean
            mean_color_value = int(mean_channels[0])

            # 2. Create a full-size grayscale layer with the mean value
            # We create a large mask and then apply it, which is more efficient
            # than creating and pasting many small patches.
            grayscale_layer = Image.new("L", original_pil_image.size, mean_color_value)

            # 3. Create a perturbed PIL image by pasting the mean value only where the mask is True
            # The mask needs to be converted to a PIL image to be used here.
            # The bin_mask is a Tensor, so we convert it.
            pil_mask = Image.fromarray(bin_mask.numpy().astype('uint8') * 255, mode='L')
            
            result_pil = original_pil_image.copy()
            result_pil.paste(grayscale_layer, (0, 0), mask=pil_mask)

            # 4. Preprocess the final perturbed PIL image back to a tensor
            processor = preprocessing.get_processor_for_precached_224_images()
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
    debug: bool = False
) -> Tuple[float, List[Dict]]:
    """
    Calculate SaCo score using attribution binning for a single image.
    
    Returns:
        Tuple of (saco_score, bin_results)
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
    predictions = pipeline.batched_model_inference(vit_model, batch_tensor, device)
    
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
    
    # Extract arrays for SaCo calculation
    attributions = np.array([r["total_attribution"] for r in bin_results])
    impacts = np.array([r["confidence_delta_abs"] for r in bin_results])
    
    if debug and "dev_0aec60bd" in str(image_path):
        print(f"BINNED - Number of bins: {len(bin_results)}")
        print(f"BINNED - Attributions[:5]: {attributions[:5]}")
        print(f"BINNED - Impacts[:5]: {impacts[:5]}")
        print(f"BINNED - Attribution sum: {np.sum(attributions):.4f}")
        print(f"BINNED - Impact sum: {np.sum(impacts):.4f}")
    
    # Calculate SaCo score using the same function as patch-wise
    saco_score = pipeline.calculate_saco_vectorized(attributions, impacts)
    
    return saco_score, bin_results


def run_binned_saco_analysis(
    config,
    original_results: List[ClassificationResult],
    vit_model,
    device: torch.device,
    n_bins: int = 20,
    save_results: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run binned SaCo analysis for entire dataset.
    
    Returns dictionary with analysis results DataFrames.
    """
    print(f"=== BINNED SACO ANALYSIS (n_bins={n_bins}) ===")
    
    all_saco_scores = {}
    all_bin_results = []
    
    # Process each image
    for original_result in tqdm(original_results, desc=f"Processing with {n_bins} bins"):
        try:
            image_name = str(original_result.image_path)
            saco_score, bin_results = calculate_binned_saco_for_image(
                original_result, vit_model, config, device, n_bins,
                debug=True  # Enable debug for specific images
            )
            
            all_saco_scores[image_name] = saco_score
            
            # Add image name to each bin result
            for result in bin_results:
                result["image_name"] = image_name
                result["saco_score"] = saco_score
                all_bin_results.append(result)
                
        except Exception as e:
            print(f"Error processing {original_result.image_path.name}: {e}")
            continue
    
    # Create analysis DataFrames
    analysis_results = {}
    
    # 1. SaCo scores DataFrame
    saco_df = pd.DataFrame([
        {"image_name": name, "saco_score": score}
        for name, score in all_saco_scores.items()
    ])
    analysis_results["saco_scores"] = saco_df
    
    # 2. Detailed bin results DataFrame
    bin_results_df = pd.DataFrame(all_bin_results)
    analysis_results["bin_results"] = bin_results_df
    
    # 3. Summary statistics
    if len(all_saco_scores) > 0:
        avg_saco = np.mean(list(all_saco_scores.values()))
        std_saco = np.std(list(all_saco_scores.values()))
        
        print(f"\nBinned SaCo Analysis Summary:")
        print(f"  Number of images: {len(all_saco_scores)}")
        print(f"  Number of bins: {n_bins}")
        print(f"  Average SaCo: {avg_saco:.4f}")
        print(f"  Std SaCo: {std_saco:.4f}")
        
        # Compare with patch-wise if available
        patch_wise_file = config.file.output_dir / "analysis_saco_scores_optimized_2024-12-19_14-30.csv"
        if patch_wise_file.exists():
            patch_wise_df = pd.read_csv(patch_wise_file)
            patch_wise_avg = patch_wise_df["saco_score"].mean()
            print(f"  Patch-wise average: {patch_wise_avg:.4f}")
            print(f"  Difference: {avg_saco - patch_wise_avg:.4f}")
    
    if save_results:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, df in analysis_results.items():
            if len(df) > 0:
                filename = f"binned_{name}_{n_bins}bins_{timestamp}.csv"
                save_path = config.file.output_dir / filename
                df.to_csv(save_path, index=False)
                print(f"Saved {name} to {save_path}")
    
    return analysis_results


# For easy integration with your pipeline, add this wrapper function:
def run_binned_attribution_analysis(
    config,
    original_results: List[ClassificationResult],
    device: torch.device,
    n_bins: int = 20
) -> None:
    """
    Wrapper function that matches your pipeline interface.
    """
    from transmm_sfaf import load_models
    
    # Load model
    sae, vit_model = load_models()
    vit_model.to(device)
    vit_model.eval()
    
    # Run binned analysis
    results = run_binned_saco_analysis(
        config, original_results, vit_model, device, n_bins
    )
    
    # Optional: Run comparison with different bin counts
    if n_bins == 20:  # Only do comparison for default run
        print("\n=== Running bin count comparison ===")
        bin_counts = [28, 49, 98, 196]
        comparison_results = []
        
        for n in bin_counts:
            print(f"\nTesting with {n} bins...")
            temp_results = run_binned_saco_analysis(
                config, original_results, vit_model, device, n, save_results=False
            )
            
            avg_saco = temp_results["saco_scores"]["saco_score"].mean()
            comparison_results.append({
                "n_bins": n,
                "avg_saco": avg_saco,
                "n_images": len(temp_results["saco_scores"])
            })
        
        comparison_df = pd.DataFrame(comparison_results)
        print("\nBin count comparison:")
        print(comparison_df)
        
        # Save comparison
        comparison_path = config.file.output_dir / "binned_saco_bin_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved comparison to {comparison_path}")
