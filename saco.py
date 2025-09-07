"""
Corrected Attribution Binning Implementation for SaCo

This implements the binning approach as described in the SaCo paper:
- Group patches by attribution value ranges (not spatial location)
- Perturb all patches within each attribution bin together
- Calculate SaCo based on bin comparisons

Refactored for better testability with separated concerns:
- Loading: Load image and attributions from files
- Binning: Create bins from attribution values
- Perturbation: Apply perturbations to bins
- Impact calculation: Measure confidence changes
- SaCo calculation: Compute the faithfulness score
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_types import ClassificationResult


def batched_model_inference(model_instance,
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


def calculate_saco_vectorized_with_bias(attributions: np.ndarray,
                                        confidence_impacts: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Simplified: Only return overall SaCo and attribution bias per bin.
    """
    n = len(attributions)
    if n < 2:
        return 0.0, np.zeros(n)

    # Create all pairwise differences using broadcasting
    attr_diffs = attributions[:, None] - attributions[None, :]
    impact_diffs = confidence_impacts[:, None] - confidence_impacts[None, :]

    # Upper triangular mask (unique comparisons only)
    upper_tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    # Identify violations
    violations = (impact_diffs * attr_diffs < 0) & upper_tri_mask

    # Weight violations by attribution difference
    violation_weights = violations * attr_diffs  # Signed weights!

    # Attribution bias:
    # Negative when bin i is over-attributed (has violations as higher-attr bin)
    # Positive when bin j is under-attributed (has violations as lower-attr bin)
    bin_attribution_bias = np.zeros(n)

    # Sum contributions
    # Row sum: when bin is first element (higher attribution in violation = over-attributed)
    bin_attribution_bias -= np.sum(violation_weights, axis=1)

    # Column sum: when bin is second element (lower attribution in violation = under-attributed)
    bin_attribution_bias += np.sum(violation_weights, axis=0)

    # Calculate overall SaCo for reference
    is_faithful = ~violations & upper_tri_mask
    weights = np.where(is_faithful, attr_diffs, -attr_diffs)
    weights_upper = np.where(upper_tri_mask, weights, 0)
    total_abs_weight = np.sum(np.abs(weights_upper))
    overall_saco = np.sum(weights_upper) / total_abs_weight if total_abs_weight > 0 else 0.0

    # Normalize bias by total weight
    if total_abs_weight > 0:
        bin_attribution_bias = bin_attribution_bias / total_abs_weight

    return overall_saco, bin_attribution_bias


@dataclass
class BinInfo:
    """Information about an attribution bin"""
    bin_id: int
    min_value: float
    max_value: float
    patch_indices: List[int]  # Which patches belong to this bin (196 for B-16, 49 for B-32)
    mean_attribution: float
    total_attribution: float
    n_patches: int


def create_attribution_bins_from_patches(
    raw_attributions: np.ndarray,  # Shape: (n_patches,) - 196 for B-16, 49 for B-32
    n_bins: int = 20
) -> List[BinInfo]:
    """
    Create bins based on attribution value ranges, ensuring equal sizes.
    
    This groups the patches into n_bins based on their attribution values.
    It sorts the patches by attribution and divides them into equal-sized chunks,
    which is a more direct implementation of the SaCo paper's description of
    "equally sized pixel subsets".
    
    Args:
        raw_attributions: Array of attribution values (one per ViT patch)
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
        temp_bins.append({"indices": patch_indices, "attributions": bin_attributions})

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
            total_attribution=float(np.sum(bin_attributions)),  # This is s(Gi) from the paper
            n_patches=len(patch_indices)
        )
        bins.append(bin_info)

    return bins


def create_spatial_mask_for_bin(
    bin_info: BinInfo, image_size: Tuple[int, int] = (224, 224), patch_size: int = 32
) -> torch.Tensor:
    """
    Create a spatial mask for all patches in a bin.
    
    Args:
        bin_info: Information about the bin
        image_size: Size of the image (height, width)
        patch_size: Size of each ViT patch (16 for B-16, 32 for B-32)
        
    Returns:
        Boolean mask of shape (H, W)
    """
    height, width = image_size
    mask = torch.zeros((height, width), dtype=torch.bool)

    # Each patch index corresponds to a position in the grid
    grid_size = image_size[0] // patch_size  # 14 for B-16, 7 for B-32

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
    original_pil_image=None,
    dataset_name: str = None
) -> torch.Tensor:
    """
    Apply perturbation to all patches in a bin.
    
    Uses grayscale perturbation method for consistency with patch-wise approach.
    Uses dataset-specific preprocessing from centralized dataset_config.
    """
    if perturbation_method == "mean":
        if original_pil_image is not None:
            # Use consistent grayscale perturbation method
            from PIL import Image, ImageStat


            # 1. Calculate mean, but take only the first channel for grayscale value
            mean_channels = ImageStat.Stat(original_pil_image).mean
            mean_color_value = int(mean_channels[0])

            # 2. Ensure the original image is in RGB mode
            if original_pil_image.mode != 'RGB':
                original_pil_image = original_pil_image.convert('RGB')

            # 3. Create a full-size RGB layer with the mean value
            # We create a large mask and then apply it, which is more efficient
            # than creating and pasting many small patches.
            grayscale_layer = Image.new(
                "RGB", original_pil_image.size, (mean_color_value, mean_color_value, mean_color_value)
            )

            # 4. Create a perturbed PIL image by pasting the mean value only where the mask is True
            # The mask needs to be converted to a PIL image to be used here.
            # The bin_mask is a Tensor, so we convert it.
            # Ensure the mask has the same size as the image
            if bin_mask.shape != (original_pil_image.height, original_pil_image.width):
                # Resize the mask to match the image size
                mask_array = bin_mask.numpy().astype('uint8') * 255
                pil_mask = Image.fromarray(mask_array, mode='L')
                pil_mask = pil_mask.resize(original_pil_image.size, Image.NEAREST)
            else:
                pil_mask = Image.fromarray(bin_mask.numpy().astype('uint8') * 255, mode='L')

            result_pil = original_pil_image.copy()
            result_pil.paste(grayscale_layer, (0, 0), mask=pil_mask)

            # 5. Preprocess the final perturbed PIL image back to a tensor
            # Use dataset-specific transforms from centralized dataset_config
            if dataset_name:
                from dataset_config import get_dataset_config
                dataset_config = get_dataset_config(dataset_name)
                # Use test transforms (no augmentations) for perturbation evaluation
                # The dataset config will handle whether to use CLIP or ViT preprocessing
                processor = dataset_config.get_transforms('test')
            else:
                raise ValueError("dataset_name is required for proper preprocessing")

            perturbed_tensor = processor(result_pil)
            return perturbed_tensor
        else:
            # Fallback tensor-only logic (this part will likely not be used but is good to have)
            perturbed = original_tensor.clone()
            mean_value = original_tensor[0].mean().item()  # Grayscale-like mean from first channel
            for c in range(original_tensor.shape[0]):
                perturbed[c][bin_mask] = mean_value
            return perturbed

    # Default case if method is not "mean"
    return original_tensor.clone()


# ============= DATA CONTAINERS FOR SEPARATED CONCERNS =============


@dataclass
class ImageData:
    """Container for loaded image data."""
    pil_image: Any
    tensor: torch.Tensor
    raw_attributions: np.ndarray
    original_confidence: float
    original_class_idx: int


@dataclass
class BinnedPerturbationData:
    """Container for perturbation data."""
    bins: List[BinInfo]
    perturbed_tensors: List[torch.Tensor]


@dataclass
class BinImpactResult:
    """Results from measuring bin impacts."""
    bin_results: List[Dict]
    saco_score: float
    bin_biases: np.ndarray


# ============= SEPARATED CONCERN FUNCTIONS =============


def load_image_and_attributions(classification_result: ClassificationResult, target_size: int = 224) -> ImageData:
    """
    Load image and attribution data from files.
    
    NOTE: We only load the raw PIL image here, no preprocessing.
    Preprocessing will be done after perturbation using dataset-specific transforms.
    
    Separated concern: File I/O and data loading
    """
    # Load raw image without any preprocessing
    image_path = classification_result.image_path
    from PIL import Image
    pil_image = Image.open(image_path).convert('RGB')

    # Load attributions
    attr_path = classification_result.attribution_paths.raw_attribution_path
    if attr_path is None:
        raise ValueError(f"No attribution path for {image_path}")
    raw_attributions = np.load(attr_path)

    # Get original prediction info
    original_pred = classification_result.prediction

    return ImageData(
        pil_image=pil_image,
        tensor=None,  # We don't preprocess yet - will do after perturbation
        raw_attributions=raw_attributions,
        original_confidence=original_pred.confidence,
        original_class_idx=original_pred.predicted_class_idx
    )


def create_binned_perturbations(
    image_data: ImageData,
    n_bins: int,
    perturbation_method: str = "mean",
    patch_size: int = 32,
    dataset_name: str = None
) -> BinnedPerturbationData:
    """
    Create bins and apply perturbations.
    
    Separated concern: Binning and perturbation logic
    """
    # Create bins from attributions
    bins = create_attribution_bins_from_patches(image_data.raw_attributions, n_bins)

    # Apply perturbations for each bin
    perturbed_tensors = []
    for bin_info in bins:
        # Create spatial mask
        mask = create_spatial_mask_for_bin(bin_info, patch_size=patch_size)

        # Apply perturbation with dataset-specific preprocessing
        perturbed = apply_binned_perturbation(
            image_data.tensor, mask, perturbation_method, image_data.pil_image, dataset_name
        )
        perturbed_tensors.append(perturbed)

    return BinnedPerturbationData(bins=bins, perturbed_tensors=perturbed_tensors)


def measure_bin_impacts(
    perturbation_data: BinnedPerturbationData,
    image_data: ImageData,
    model: Any,
    device: torch.device,
    batch_size: int = 32
) -> List[Dict]:
    """
    Measure the impact of each bin on model confidence.
    
    Separated concern: Model inference and impact measurement
    """
    if not perturbation_data.perturbed_tensors:
        return []

    # Batch inference
    batch_tensor = torch.stack(perturbation_data.perturbed_tensors)
    predictions = batched_model_inference(model, batch_tensor, device, batch_size=len(batch_tensor))

    # Calculate impacts for each bin
    bin_results = []
    for bin_info, pred in zip(perturbation_data.bins, predictions):
        # Paper formula: ∇pred = p(ŷ|original) - p(ŷ|perturbed)
        confidence_impact = image_data.original_confidence - pred["confidence"]

        result = {
            "bin_id": bin_info.bin_id,
            "mean_attribution": bin_info.mean_attribution,
            "total_attribution": bin_info.total_attribution,
            "n_patches": bin_info.n_patches,
            "confidence_delta": confidence_impact,
            "confidence_delta_abs": abs(confidence_impact),
            "class_changed": (pred["predicted_class_idx"] != image_data.original_class_idx)
        }
        bin_results.append(result)

    return bin_results


def compute_saco_from_impacts(bin_results: List[Dict], compute_bias: bool = True) -> BinImpactResult:
    """
    Compute SaCo score from bin impact results.
    
    Separated concern: SaCo calculation
    """
    if len(bin_results) < 2:
        # Cannot compute SaCo with less than 2 bins
        saco_score = 0.0
        bin_biases = np.zeros(len(bin_results))

        # Add default values
        for result in bin_results:
            result["bin_attribution_bias"] = 0.0
    else:
        # Sort by total attribution for consistent ordering
        bin_results.sort(key=lambda x: x["total_attribution"], reverse=True)

        # Extract arrays for SaCo calculation
        attributions = np.array([r["total_attribution"] for r in bin_results])
        impacts = np.array([r["confidence_delta"] for r in bin_results])  # Signed!

        # Calculate SaCo and bias
        saco_score, bin_biases = calculate_saco_vectorized_with_bias(attributions, impacts)

        # Add bias to results if requested
        if compute_bias:
            for i, result in enumerate(bin_results):
                result["bin_attribution_bias"] = float(bin_biases[i])

    return BinImpactResult(bin_results=bin_results, saco_score=saco_score, bin_biases=bin_biases)


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
    
    This function orchestrates the separated concern functions for better testability.
    
    Args:
        original_result: Classification result with attribution paths
        vit_model: The model to use for inference
        config: Configuration object
        device: Device to run inference on
        n_bins: Number of bins to create
        debug: Whether to print debug information
        include_patch_level: If True, also generate patch-level SaCo scores for compatibility
    
    Returns:
        Tuple of (saco_score, bin_results, patch_results)
    """
    try:
        # Get dataset name from config for proper preprocessing
        dataset_name = config.file.dataset_name if hasattr(config.file, 'dataset_name') else None

        # Step 1: Load image and attributions (no preprocessing yet)
        image_data = load_image_and_attributions(original_result, target_size=config.classify.target_size[0])

        # Step 2: Create bins and apply perturbations (with dataset-specific preprocessing)
        # Determine patch_size from model architecture
        if hasattr(vit_model, 'cfg') and hasattr(vit_model.cfg, 'patch_size'):
            patch_size = vit_model.cfg.patch_size
        else:
            # Infer from dataset/model type
            # For CLIP models using B-32 architecture
            if dataset_name == 'waterbirds':
                patch_size = 32
            else:
                patch_size = 16  # Default for standard ViT-B-16
        
        if debug:
            print(f"Using patch_size={patch_size} for {dataset_name}")
        perturbation_data = create_binned_perturbations(
            image_data,
            n_bins,
            perturbation_method=config.perturb.method,
            patch_size=patch_size,
            dataset_name=dataset_name
        )

        # Step 3: Measure impacts through model inference
        vit_model.eval()
        bin_results = measure_bin_impacts(perturbation_data, image_data, vit_model, device, batch_size=32)

        # Step 4: Compute SaCo score and biases
        impact_result = compute_saco_from_impacts(bin_results, compute_bias=True)

        # Empty patch results for compatibility
        # TODO: Remove this when patch-level analysis is fully deprecated
        patch_results = []

        return impact_result.saco_score, impact_result.bin_results, patch_results

    except Exception as e:
        print(f"Error in calculate_binned_saco_for_image: {e}")
        raise


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
        config,
        original_results,
        vit_model,
        device,
        n_bins,
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
    faithfulness_df = analyze_faithfulness_vs_correctness_from_objects(saco_scores_map, original_results)
    analysis_results["faithfulness_correctness"] = faithfulness_df

    # 4. Key Attribution Patterns Analysis
    patterns_df = analyze_key_attribution_patterns(
        analysis_results["faithfulness_correctness"], vit_model, config
    )
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
                # Save bin results with dataset prefix
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
    results = run_binned_saco_analysis(config, original_results, vit_model, device, n_bins)
    return results


def extract_true_class_from_filename(filename):
    """
    Simple function to extract true class from filename.
    For the unified dataloader format, the true label is already available in ClassificationResult.
    This is a fallback that returns None if we can't determine the class.
    """
    # For waterbirds dataset, filenames typically start with class index
    filepath_str = str(filename)
    if 'img_00_' in filepath_str:
        return 'landbird'
    elif 'img_01_' in filepath_str:
        return 'waterbird'
    else:
        # Return None - the true_label should be available in ClassificationResult
        return None


def analyze_faithfulness_vs_correctness_from_objects(
    saco_scores: Dict[str, float],
    original_classification_results: List[ClassificationResult]
) -> pd.DataFrame:
    """
    Analyze the relationship between attribution faithfulness (SaCo) and prediction correctness.
    
    Args:
        saco_scores: Dictionary mapping string original image paths to SaCo scores.
        original_classification_results: List of ClassificationResult objects for original images.
        
    Returns:
        DataFrame with SaCo scores, correctness, confidence, and paths for further analysis.
    """
    analysis_data_list = []

    for original_res in original_classification_results:
        image_path_str = str(original_res.image_path)
        saco_score = saco_scores.get(image_path_str)

        # Use true_label from ClassificationResult if available, otherwise fall back to extraction
        true_class_label = original_res.true_label if original_res.true_label else extract_true_class_from_filename(original_res.image_path)

        if original_res.prediction is None:
            continue

        prediction_info = original_res.prediction
        attribution_paths_info = original_res.attribution_paths

        row_data = {
            'filename': image_path_str,
            'saco_score': saco_score,
            'predicted_class': prediction_info.predicted_class_label,
            'predicted_idx': prediction_info.predicted_class_idx,
            'true_class': true_class_label,
            'is_correct': prediction_info.predicted_class_label == true_class_label,
            'confidence': prediction_info.confidence,
            'attribution_path': str(attribution_paths_info.attribution_path) if attribution_paths_info else None,
            'probabilities': prediction_info.probabilities
        }
        analysis_data_list.append(row_data)

    return pd.DataFrame(analysis_data_list)


def analyze_key_attribution_patterns(df: pd.DataFrame, model, config) -> pd.DataFrame:
    """
    Simplified version of attribution patterns analysis.
    For now, just return the input DataFrame with basic processing.
    """
    # Clean data
    df_clean = df.dropna(subset=['saco_score'])
    
    # Add class column for compatibility
    if 'filename' in df_clean.columns:
        df_clean['class'] = df_clean['true_class']
    
    # Only analyze correct predictions for now
    df_clean = df_clean[df_clean['is_correct']]
    
    return df_clean
