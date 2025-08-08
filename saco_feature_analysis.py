"""
SaCo-based Feature Analysis Prototype (Refactored)

Identifies SAE features that are highly active in patches with a significant
discrepancy between their attribution and classification impact (SaCo score).

This version processes each image only once, analyzing both over-represented
(negative SaCo) and under-represented (positive SaCo) patches in a single
forward pass per image for improved efficiency and clarity.
"""

import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.sae import SparseAutoencoder
from vit.preprocessing import get_processor_for_precached_224_images

# Restore your original imports. Make sure this path is correct for your project structure.
from transmm_sfaf import (SAE_CONFIG, load_models)
import logging

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- NEW HELPER FUNCTION TO FIX PICKLING ERROR ---
def defaultdict_to_dict(d: Any) -> Any:
    """
    Recursively convert defaultdicts to regular dicts to allow pickling.
    """
    if isinstance(d, defaultdict):
        # Convert the children first
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d
# --------------------------------------------------


def extract_class_from_image_name(image_name: str) -> str:
    """Extract anatomical class from image filename."""
    filename = Path(image_name).stem
    parts = filename.split('_')
    # Training format: train_uuid_class_aug0 -> class is parts[2]
    # Validation format: val_uuid_class -> class is parts[2]
    if len(parts) >= 3:
        return parts[2]  # Class is always the third part
    else:
        return 'unknown'


def load_saco_bin_data(csv_path: str) -> pd.DataFrame:
    """Load and process SaCo bin analysis data from attribution_binning.py output."""
    print("Loading bin-level CSV data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} bin records")
    
    # The bin data should have columns: image_name, bin_id, mean_attribution, total_attribution, 
    # n_patches, confidence_delta, confidence_delta_abs, class_changed, saco_score
    required_cols = ['image_name', 'bin_id', 'mean_attribution', 'confidence_delta_abs']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    return df


def classify_bins(df: pd.DataFrame, attr_quantiles: tuple = (0.3, 0.7), impact_quantiles: tuple = (0.3, 0.7)) -> pd.DataFrame:
    """
    Classify bins into 'over_attributed', 'under_attributed', or 'normal' based on 
    per-image quantiles of attribution and confidence impact.
    
    A bin is problematic if it has:
    - High attribution (>70th percentile) + Low impact (<30th percentile) = over-attributed
    - Low attribution (<30th percentile) + High impact (>70th percentile) = under-attributed
    
    Args:
        df: DataFrame with bin data including 'mean_attribution' and 'confidence_delta_abs'
        attr_quantiles: (low, high) quantiles for attribution thresholds 
        impact_quantiles: (low, high) quantiles for confidence impact thresholds
    """
    # Calculate per-image quantile thresholds
    attr_low_q, attr_high_q = attr_quantiles
    impact_low_q, impact_high_q = impact_quantiles
    
    attr_high_threshold = df.groupby('image_name')['mean_attribution'].transform(lambda x: x.quantile(attr_high_q))
    attr_low_threshold = df.groupby('image_name')['mean_attribution'].transform(lambda x: x.quantile(attr_low_q))
    impact_high_threshold = df.groupby('image_name')['confidence_delta_abs'].transform(lambda x: x.quantile(impact_high_q))
    impact_low_threshold = df.groupby('image_name')['confidence_delta_abs'].transform(lambda x: x.quantile(impact_low_q))
    
    conditions = [
        # High attribution + Low impact = over-attributed
        (df['mean_attribution'] > attr_high_threshold) & (df['confidence_delta_abs'] < impact_low_threshold),
        # Low attribution + High impact = under-attributed  
        (df['mean_attribution'] < attr_low_threshold) & (df['confidence_delta_abs'] > impact_high_threshold)
    ]
    choices = ['over_attributed', 'under_attributed']
    df['problem_type'] = np.select(conditions, choices, default='normal')
    return df


def get_patches_for_bin(image_name: str, bin_id: int, raw_attributions: np.ndarray, n_bins: int = 20) -> List[int]:
    """
    Recreate which patches belong to a specific bin for a given image.
    This replicates the binning logic from attribution_binning.py
    """
    if n_bins > len(raw_attributions):
        n_bins = len(raw_attributions)
    
    # Sort patch indices by attribution values (descending)
    sorted_patch_indices = np.argsort(raw_attributions)[::-1]
    
    # Split into equal-sized bins
    binned_indices = np.array_split(sorted_patch_indices, n_bins)
    
    # Sort bins by their mean attribution to match bin_id assignment
    temp_bins = []
    for patch_indices_list in binned_indices:
        if len(patch_indices_list) == 0:
            continue
        bin_attributions = raw_attributions[patch_indices_list]
        temp_bins.append({
            "indices": patch_indices_list,
            "mean_attr": np.mean(bin_attributions)
        })
    
    # Sort by mean attribution (lowest first to match bin_id assignment)
    temp_bins.sort(key=lambda b: b['mean_attr'])
    
    if bin_id < len(temp_bins):
        return temp_bins[bin_id]["indices"].tolist()
    else:
        return []


def map_bins_to_spatial_regions(problematic_bins_df: pd.DataFrame, attribution_data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Map problematic bins back to their constituent patches for spatial analysis.
    OPTIMIZED: Vectorized operations and batch processing.
    """
    # Pre-compute all patch mappings to avoid repeated computation
    image_patch_mappings = {}
    
    patch_records = []
    
    # Group by image to minimize repeated computations
    for image_name, image_bins in problematic_bins_df.groupby('image_name'):
        if image_name not in attribution_data:
            logging.warning(f"No attribution data found for {image_name}")
            continue
        
        raw_attrs = attribution_data[image_name]
        
        # Pre-compute binning for this image once
        if image_name not in image_patch_mappings:
            n_bins = len(image_bins['bin_id'].unique())
            sorted_patch_indices = np.argsort(raw_attrs)[::-1]
            binned_indices = np.array_split(sorted_patch_indices, n_bins)
            
            # Create mapping from bin_id to patch_ids
            temp_bins = []
            for patch_indices_list in binned_indices:
                if len(patch_indices_list) == 0:
                    continue
                bin_attributions = raw_attrs[patch_indices_list]
                temp_bins.append({
                    "indices": patch_indices_list,
                    "mean_attr": np.mean(bin_attributions)
                })
            
            temp_bins.sort(key=lambda b: b['mean_attr'])
            image_patch_mappings[image_name] = {i: temp_bins[i]["indices"] for i in range(len(temp_bins))}
        
        # Use pre-computed mapping
        patch_mapping = image_patch_mappings[image_name]
        
        # Vectorized creation of patch records
        for _, bin_row in image_bins.iterrows():
            bin_id = bin_row['bin_id']
            if bin_id in patch_mapping:
                patch_ids = patch_mapping[bin_id]
                
                # Create records for all patches in this bin at once
                patch_records.extend([{
                    'image_name': image_name,
                    'patch_id': int(patch_id),
                    'bin_id': bin_id,
                    'problem_type': bin_row['problem_type'],
                    'mean_attribution': raw_attrs[patch_id],
                    'bin_saco_score': bin_row.get('saco_score', 0.0)
                } for patch_id in patch_ids])
    
    return pd.DataFrame(patch_records)


def _calculate_feature_metrics_for_patches(
    feature_activations: torch.Tensor,
    target_patch_indices: torch.Tensor,
    target_patch_saco_scores: torch.Tensor,
    activation_threshold: float = 0.1,
) -> Dict[int, Dict[str, Any]]:
    """
    Internal helper to compute feature overlap metrics for a specific set of target patches.
    """
    if len(target_patch_indices) == 0:
        return {}

    target_patch_activations = feature_activations[target_patch_indices]
    is_active_in_target = target_patch_activations.abs() > activation_threshold
    features_to_analyze = is_active_in_target.any(dim=0).nonzero(as_tuple=True)[0]

    if len(features_to_analyze) == 0:
        return {}

    feature_metrics = {}
    for feat_idx in features_to_analyze:
        feat_activations_all = feature_activations[:, feat_idx]
        feat_activations_target = target_patch_activations[:, feat_idx]

        active_in_target_mask = feat_activations_target.abs() > activation_threshold
        target_active_count = active_in_target_mask.sum().item()
        
        if target_active_count == 0:
            continue

        all_active_count = (feat_activations_all.abs() > activation_threshold).sum().item()
        overlap_ratio = target_active_count / max(all_active_count, 1)
        active_target_values = feat_activations_target[active_in_target_mask]
        
        target_weights = feat_activations_target.abs() / (feat_activations_target.abs().sum() + 1e-8)
        weighted_saco_score = (target_patch_saco_scores * target_weights).sum().item()

        feature_metrics[feat_idx.item()] = {
            'total_active_patches': all_active_count,
            'problematic_active_patches': target_active_count,
            'overlap_ratio': overlap_ratio,
            'mean_target_activation': active_target_values.mean().item(),
            'max_target_activation': active_target_values.max().item(),
            'weighted_saco_score': weighted_saco_score,
        }
    return feature_metrics


def compute_saco_adjusted_confidence_score(
    mean_saco: float,
    overlap_ratio: float, 
    n_occurrences: int,
    image_saco_scores: List[float],
    confidence_threshold: int = 20
) -> float:
    """
    Compute SaCo-adjusted confidence score that prioritizes features from 
    problematic (low SaCo) images.
    
    Args:
        mean_saco: Mean weighted SaCo score for this feature
        overlap_ratio: Mean overlap ratio
        n_occurrences: Number of occurrences
        image_saco_scores: List of SaCo scores from images where this feature appears
        confidence_threshold: Sample size for full confidence (default: 20)
    
    Returns:
        SaCo-adjusted confidence score (higher = more important to analyze)
    """
    # Base quality score (higher absolute SaCo + higher overlap = better)
    base_quality = abs(mean_saco) * overlap_ratio
    
    # Confidence penalty: linear ramp up to confidence_threshold
    confidence_factor = min(n_occurrences / confidence_threshold, 1.0)
    
    # Additional penalty for very low sample sizes (< 5 samples)
    if n_occurrences < 5:
        confidence_factor *= 0.5  # Heavy penalty for unreliable features
    
    # NEW: SaCo adjustment - prioritize features from low SaCo (problematic) images
    if len(image_saco_scores) > 0:
        mean_image_saco = np.mean(image_saco_scores)
        # Higher penalty for features from low-SaCo images (more problematic = higher priority)
        saco_boost = 1.0 + (1.0 - max(0.0, mean_image_saco))  # Boost between 1.0 and 2.0
    else:
        saco_boost = 1.0
    
    return base_quality * confidence_factor * saco_boost


def _aggregate_results(
    raw_results: Dict[int, Dict[str, List]],
    min_occurrences: int = 2,
    confidence_threshold: int = 20,
    image_saco_scores: Dict[str, float] = None
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Aggregate raw occurrence data into final statistics with confidence-adjusted scoring."""
    aggregated = defaultdict(dict)
    
    for feat_id, data_by_type in raw_results.items():
        for patch_type, occurrences in data_by_type.items():
            if len(occurrences) < min_occurrences:
                continue

            overlap_ratios = [occ['overlap_ratio'] for occ in occurrences]
            weighted_sacos = [occ['weighted_saco_score'] for occ in occurrences]
            mean_activations = [occ['mean_target_activation'] for occ in occurrences]
            classes_affected = [occ['image_class'] for occ in occurrences]
            images_affected = [occ['image'] for occ in occurrences]
            class_counts = Counter(classes_affected)
            
            # Compute standard statistics (optimized with numpy)
            n_occ = len(occurrences)
            mean_overlap = float(np.mean(overlap_ratios))
            mean_saco = float(np.mean(weighted_sacos))
            mean_activation = float(np.mean(mean_activations))
            
            # Collect image SaCo scores for this feature
            feature_image_saco_scores = []
            if image_saco_scores:
                for img_name in images_affected:
                    if img_name in image_saco_scores:
                        feature_image_saco_scores.append(image_saco_scores[img_name])
            
            # NEW: Compute SaCo-adjusted confidence score
            confidence_score = compute_saco_adjusted_confidence_score(
                mean_saco, mean_overlap, n_occ, feature_image_saco_scores, confidence_threshold
            )
            
            stats = {
                'n_occurrences': n_occ,
                'mean_overlap_ratio': mean_overlap,
                'mean_weighted_saco': mean_saco,
                'mean_activation_in_problematic': mean_activation,
                'saco_adjusted_score': confidence_score,  # NEW: Updated name to reflect SaCo adjustment
                'images_affected': images_affected,
                'class_distribution': dict(class_counts),
                'dominant_class': class_counts.most_common(1)[0][0] if class_counts else 'unknown',
                'mean_image_saco': float(np.mean(feature_image_saco_scores)) if feature_image_saco_scores else 0.0,  # NEW
                'image_saco_scores': feature_image_saco_scores,  # NEW: For debugging/analysis
            }
            aggregated[patch_type][feat_id] = stats

    # Sort by SaCo-adjusted score instead of raw SaCo score
    sorted_results = {}
    for patch_type, feature_stats in aggregated.items():
        sorted_features = sorted(
            feature_stats.items(),
            key=lambda item: item[1]['saco_adjusted_score'],
            reverse=True  # Higher SaCo-adjusted score = better
        )
        sorted_results[patch_type] = dict(sorted_features)

    return sorted_results


def process_image_batch(
    batch_items: List[Tuple[str, Any]], 
    model: HookedSAEViT, 
    sae: SparseAutoencoder, 
    layer_idx: int,
    device: torch.device,
    transform: Any,
    activation_threshold: float,
    min_overlap_ratio: float
) -> Dict[int, Dict[str, List]]:
    """Process a batch of images efficiently - simple approach with batched GPU ops."""
    
    # Prepare batch data
    batch_images = []
    batch_metadata = []
    
    for image_name, image_df in batch_items:
        try:
            if image_name.startswith('results/'):
                image_path = Path(image_name)
            else:
                image_path = Path(f"results/val/preprocessed/{Path(image_name).name}")
            
            if not image_path.exists():
                logging.warning(f"Image not found, skipping: {image_path}")
                continue

            image = Image.open(image_path).convert('RGB')
            img_tensor = transform(image)
            batch_images.append(img_tensor)
            batch_metadata.append((image_name, image_df))
            
        except Exception as e:
            logging.warning(f"Error loading {image_name}: {e}")
            continue
    
    if not batch_images:
        return {}
    
    # Batch forward pass (the main optimization)
    batch_tensor = torch.stack(batch_images).to(device)  # [B, C, H, W]
    resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
    
    with torch.no_grad():
        _, cache = model.run_with_cache(batch_tensor, names_filter=[resid_hook_name])
        resid = cache[resid_hook_name]  # [B, T+1, D]
        _, codes = sae.encode(resid)    # [B, T+1, K]
    
    # Simple per-image processing (original approach)
    batch_results = {}
    
    for batch_idx, (image_name, image_df) in enumerate(batch_metadata):
        feature_activations = codes[batch_idx, 1:]  # Skip CLS token
        image_class = extract_class_from_image_name(image_name)
        
        for patch_type in ['over_attributed', 'under_attributed']:
            target_patches_df = image_df[image_df['problem_type'] == patch_type]
            if target_patches_df.empty:
                continue

            target_patch_indices = torch.tensor(target_patches_df['patch_id'].values, dtype=torch.long).to(device)
            target_saco_scores = torch.tensor(target_patches_df['bin_saco_score'].values, dtype=torch.float32).to(device)

            feature_metrics = _calculate_feature_metrics_for_patches(
                feature_activations,
                target_patch_indices,
                target_saco_scores,
                activation_threshold
            )

            for feat_id, metrics in feature_metrics.items():
                if metrics['overlap_ratio'] >= min_overlap_ratio:
                    if feat_id not in batch_results:
                        batch_results[feat_id] = {'over_attributed': [], 'under_attributed': []}
                    
                    batch_results[feat_id][patch_type].append({
                        'image': image_name,
                        'image_class': image_class,
                        **metrics
                    })
    
    return batch_results


def analyze_saco_features_bin_based(
    saco_bin_csv_path: str,
    attribution_dir: str,  # Directory containing raw attribution files
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    layer_idx: int = 9,
    attr_quantiles: tuple = (0.3, 0.7),
    impact_quantiles: tuple = (0.3, 0.7),
    n_images: Optional[int] = 50,
    min_overlap_ratio: float = 0.3,
    min_occurrences: int = 2,
    activation_threshold: float = 0.1,
    batch_size: int = 4,
    n_bins: int = 20,  # Number of bins used in binning analysis
) -> Dict[str, Any]:
    """
    Main analysis function to identify features spanning problematic SaCo bins
    using the bin-centric approach from attribution_binning.py.
    """
    device = next(model.parameters()).device
    transform = get_processor_for_precached_224_images()
    
    # Load bin-level data from attribution_binning.py output
    saco_df = load_saco_bin_data(saco_bin_csv_path)
    saco_df = classify_bins(saco_df, attr_quantiles, impact_quantiles)
    problematic_df = saco_df[saco_df['problem_type'] != 'normal']
    
    # Extract image-level SaCo scores for weighting feature importance
    image_saco_scores = {}
    if 'saco_score' in saco_df.columns:
        for _, row in saco_df.iterrows():
            image_saco_scores[row['image_name']] = row['saco_score']
    
    logging.info(f"Found {len(problematic_df[problematic_df['problem_type'] == 'over_attributed'])} over-attributed bins and "
                f"{len(problematic_df[problematic_df['problem_type'] == 'under_attributed'])} under-attributed bins.")
    
    # Load raw attribution data for mapping bins back to patches
    attribution_data = {}
    for image_name in problematic_df['image_name'].unique():
        # Try to find the corresponding raw attribution file
        image_stem = Path(image_name).stem
        attr_file = Path(attribution_dir) / f"{image_stem}_raw_attribution.npy"
        if attr_file.exists():
            attribution_data[image_name] = np.load(attr_file)
        else:
            logging.warning(f"No attribution file found for {image_name}")
    
    # Map problematic bins back to their constituent patches
    problematic_patches_df = map_bins_to_spatial_regions(problematic_df, attribution_data)
    
    logging.info(
        f"Mapped bins to {len(problematic_patches_df[problematic_patches_df['problem_type'] == 'over_attributed'])} over-attributed patches and "
        f"{len(problematic_patches_df[problematic_patches_df['problem_type'] == 'under_attributed'])} under-attributed patches."
    )

    # Early filtering: Only process images with significant numbers of problematic patches
    patch_counts = problematic_patches_df['image_name'].value_counts()
    min_patches_per_image = 3  # Skip images with < 3 problematic patches
    significant_images = patch_counts[patch_counts >= min_patches_per_image].index.tolist()
    
    filtered_df = problematic_patches_df[problematic_patches_df['image_name'].isin(significant_images)]
    logging.info(f"Filtered from {len(problematic_patches_df['image_name'].unique())} to {len(significant_images)} images with ≥{min_patches_per_image} problematic patches")
    
    image_groups = filtered_df.groupby('image_name')
    raw_feature_results = defaultdict(lambda: defaultdict(list))
    
    images_to_process = list(image_groups)[:n_images] if n_images is not None else list(image_groups)
    
    # Process images in batches for better GPU utilization
    total_batches = (len(images_to_process) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Processing image batches", unit="batch"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(images_to_process))
        batch_items = images_to_process[start_idx:end_idx]
        
        try:
            # Process batch
            batch_results = process_image_batch(
                batch_items, model, sae, layer_idx, device, transform,
                activation_threshold, min_overlap_ratio
            )
            
            # Merge batch results into main results
            for feat_id, type_data in batch_results.items():
                for patch_type, occurrences in type_data.items():
                    raw_feature_results[feat_id][patch_type].extend(occurrences)
                    
        except Exception as e:
            logging.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)
            
        # Optional: Clear GPU cache every few batches to prevent memory issues
        if (batch_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    logging.info(f"\n--- Aggregating results from {len(images_to_process)} processed images ---")
    final_results = _aggregate_results(raw_feature_results, min_occurrences, confidence_threshold=20, image_saco_scores=image_saco_scores)

    for patch_type, sorted_features in final_results.items():
        logging.info(f"\nTop 10 features for '{patch_type}' patches (found {len(sorted_features)} total):")
        for i, (feat_id, stats) in enumerate(list(sorted_features.items())[:20]):
             logging.info(
                f"  {i+1}. Feature {feat_id}: "
                f"saco_adj_score={stats['saco_adjusted_score']:.3f}, "
                f"mean_img_saco={stats['mean_image_saco']:.3f}, "
                f"w_saco={stats['mean_weighted_saco']:.3f}, "
                f"overlap={stats['mean_overlap_ratio']:.2f}, "
                f"n_occ={stats['n_occurrences']}, "
                f"dom_class={stats['dominant_class']}"
            )
    
    # --- APPLY FIX HERE ---
    # Convert defaultdict to a regular dict to make it picklable before returning
    picklable_raw_results = defaultdict_to_dict(raw_feature_results)

    return {
        'results_by_type': final_results,
        'analysis_params': {
            'layer_idx': layer_idx,
            'attr_quantiles': attr_quantiles,
            'impact_quantiles': impact_quantiles,
            'n_bins': n_bins,
            'min_overlap_ratio': min_overlap_ratio,
            'min_occurrences': min_occurrences,
            'n_images_processed': len(images_to_process)
        },
        'raw_results': picklable_raw_results # Use the converted, picklable version
    }


def save_analysis_results(results: Dict[str, Any], save_path: str):
    """Save analysis results to file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, save_path)
    logging.info(f"Analysis results saved to {save_path}")


if __name__ == "__main__":
    model = load_models()
    
    layer_idx = 7
    sae_path = Path(SAE_CONFIG[layer_idx]["sae_path"])
    sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
    sae.to(next(model.parameters()).device)
    
    # Use bin-level CSV from attribution_binning.py output
    saco_bin_csv_path = "results/Train/saco_bin_analysis_binned_49bins_2025-07-24_17-34.csv"  # Update with actual file
    attribution_dir = "results/Train/attributions"  # Directory with raw attribution files
    
    results = analyze_saco_features_bin_based(
        saco_bin_csv_path=saco_bin_csv_path,
        attribution_dir=attribution_dir,
        model=model,
        sae=sae,
        layer_idx=layer_idx,
        attr_quantiles=(0.2, 0.8),   # 30th and 70th percentiles for attribution thresholds
        impact_quantiles=(0.2, 0.8), # 30th and 70th percentiles for confidence impact thresholds
        n_images=None,
        min_overlap_ratio=0.7,        # 30% overlap is sufficient for bin-based analysis
        min_occurrences=10,            # Minimal cutoff: confidence weighting handles reliability
        batch_size=32,                # Optimized batch size for vectorized processing
        n_bins=49,                    # Match the number of bins used in attribution_binning.py
    )
    
    save_path = f"results/saco_problematic_features_bins_l{layer_idx}.pt"
    save_analysis_results(results, save_path)
