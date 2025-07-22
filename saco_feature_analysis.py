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

# Restore your original imports. Make sure this path is correct for your project structure.
from transmm_sfaf import (SAE_CONFIG, get_processor_for_precached_224_images, load_models)

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


def patch_id_to_coordinates(patch_id: int, n_patches_per_side: int = 14) -> Tuple[int, int]:
    """Convert patch ID to (row, col) coordinates."""
    return patch_id // n_patches_per_side, patch_id % n_patches_per_side


def load_saco_patch_data(csv_path: str) -> pd.DataFrame:
    """Load and process SaCo patch analysis data."""
    print("Loading CSV data...")
    df = pd.read_csv(csv_path)
    print("Processing patch coordinates...")
    # Vectorized coordinate conversion - much faster than apply
    n_patches_per_side = 14
    df['patch_row'] = df['patch_id'] // n_patches_per_side
    df['patch_col'] = df['patch_id'] % n_patches_per_side
    return df


def classify_patches(df: pd.DataFrame, negative_threshold: float, positive_threshold: float) -> pd.DataFrame:
    """
    Classify patches into 'over_attributed', 'under_attributed', or 'normal'.
    """
    conditions = [
        df['patch_saco'] < negative_threshold,
        df['patch_saco'] > positive_threshold
    ]
    choices = ['over_attributed', 'under_attributed']
    df['problem_type'] = np.select(conditions, choices, default='normal')
    return df


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


def compute_confidence_adjusted_score(
    saco_score: float, 
    overlap_ratio: float, 
    n_occurrences: int,
    confidence_threshold: int = 20
) -> float:
    """
    Compute confidence-adjusted quality score that balances statistical reliability 
    with feature quality.
    
    Args:
        saco_score: Mean weighted SaCo score 
        overlap_ratio: Mean overlap ratio
        n_occurrences: Number of occurrences
        confidence_threshold: Sample size for full confidence (default: 20)
    
    Returns:
        Confidence-adjusted quality score
    """
    # Base quality score (higher absolute SaCo + higher overlap = better)
    base_quality = abs(saco_score) * overlap_ratio
    
    # Confidence penalty: linear ramp up to confidence_threshold
    confidence_factor = min(n_occurrences / confidence_threshold, 1.0)
    
    # Additional penalty for very low sample sizes (< 5 samples)
    if n_occurrences < 5:
        confidence_factor *= 0.5  # Heavy penalty for unreliable features
    
    return base_quality * confidence_factor


def _aggregate_results(
    raw_results: Dict[int, Dict[str, List]],
    min_occurrences: int = 2,
    confidence_threshold: int = 20
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
            class_counts = Counter(classes_affected)
            
            # Compute standard statistics (optimized with numpy)
            n_occ = len(occurrences)
            mean_overlap = float(np.mean(overlap_ratios))
            mean_saco = float(np.mean(weighted_sacos))
            mean_activation = float(np.mean(mean_activations))
            
            # NEW: Compute confidence-adjusted score
            confidence_score = compute_confidence_adjusted_score(
                mean_saco, mean_overlap, n_occ, confidence_threshold
            )
            
            stats = {
                'n_occurrences': n_occ,
                'mean_overlap_ratio': mean_overlap,
                'mean_weighted_saco': mean_saco,
                'mean_activation_in_problematic': mean_activation,
                'confidence_adjusted_score': confidence_score,  # NEW
                'images_affected': [occ['image'] for occ in occurrences],
                'class_distribution': dict(class_counts),
                'dominant_class': class_counts.most_common(1)[0][0] if class_counts else 'unknown',
            }
            aggregated[patch_type][feat_id] = stats

    # Sort by confidence-adjusted score instead of raw SaCo score
    sorted_results = {}
    for patch_type, feature_stats in aggregated.items():
        sorted_features = sorted(
            feature_stats.items(),
            key=lambda item: item[1]['confidence_adjusted_score'],
            reverse=True  # Higher confidence score = better
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
            target_saco_scores = torch.tensor(target_patches_df['patch_saco'].values, dtype=torch.float32).to(device)

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


def analyze_saco_features_single_pass(
    saco_csv_path: str,
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    layer_idx: int = 9,
    negative_threshold: float = -0.3,
    positive_threshold: float = 0.3,
    n_images: Optional[int] = 50,
    min_overlap_ratio: float = 0.3,
    min_occurrences: int = 2,
    activation_threshold: float = 0.1,
    batch_size: int = 4,  # NEW: batch processing
) -> Dict[str, Any]:
    """
    Main analysis function to identify features spanning problematic SaCo patches
    using a single forward pass per image.
    """
    device = next(model.parameters()).device
    transform = get_processor_for_precached_224_images()
    
    saco_df = load_saco_patch_data(saco_csv_path)
    saco_df = classify_patches(saco_df, negative_threshold, positive_threshold)
    problematic_df = saco_df[saco_df['problem_type'] != 'normal']
    
    logging.info(
        f"Found {len(problematic_df[problematic_df['problem_type'] == 'over_attributed'])} over-attributed patches and "
        f"{len(problematic_df[problematic_df['problem_type'] == 'under_attributed'])} under-attributed patches."
    )

    # Only group images that actually have problematic patches - saves memory
    unique_problematic_images = problematic_df['image_name'].unique()
    logging.info(f"Processing {len(unique_problematic_images)} unique images with problematic patches")
    image_groups = problematic_df.groupby('image_name')
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
    final_results = _aggregate_results(raw_feature_results, min_occurrences, confidence_threshold=20)

    for patch_type, sorted_features in final_results.items():
        logging.info(f"\nTop 10 features for '{patch_type}' patches (found {len(sorted_features)} total):")
        for i, (feat_id, stats) in enumerate(list(sorted_features.items())[:20]):
             logging.info(
                f"  {i+1}. Feature {feat_id}: "
                f"conf_score={stats['confidence_adjusted_score']:.3f}, "
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
            'negative_threshold': negative_threshold,
            'positive_threshold': positive_threshold,
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
    _, model = load_models()
    
    layer_idx = 8
    sae_path = Path(SAE_CONFIG[layer_idx]["sae_path"])
    sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
    sae.to(next(model.parameters()).device)
    
    saco_csv_path = "results/train/saco_patch_analysis_mean_optimized_2025-07-22_13-27.csv"
    
    results = analyze_saco_features_single_pass(
        saco_csv_path=saco_csv_path,
        model=model,
        sae=sae,
        layer_idx=layer_idx,
        negative_threshold=-0.5,     # More inclusive: catch moderate over-attribution
        positive_threshold=0.5,      # More inclusive: catch moderate under-attribution  
        n_images=50000,
        min_overlap_ratio=0.7,       # More inclusive: 30% overlap is sufficient
        min_occurrences=5,           # Minimal cutoff: confidence weighting handles reliability
        batch_size=32,                 # Optimized batch size for vectorized processing
    )
    
    save_path = f"results/saco_problematic_features_l{layer_idx}_moderate.pt"
    save_analysis_results(results, save_path)
