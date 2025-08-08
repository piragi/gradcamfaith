"""
Feature-based Patch Impact Analysis

This module implements a cleaner approach to identifying SAE features that correspond
to patches with misaligned attribution vs. classification impact. Instead of using bins,
it directly analyzes each active feature and classifies their patches by the ratio of
classification impact to attribution strength.

Workflow:
1. For each image, identify active features  
2. For each active feature, calculate log(class impact/attribution) per patch
3. Aggregate across all images to build the final dictionary
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.sae import SparseAutoencoder
from vit.preprocessing import get_processor_for_precached_224_images

from transmm_sfaf import SAE_CONFIG, load_models

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_class_from_image_name(image_name: str) -> str:
    """Extract class from image filename.
    
    Expected format: COVID-19_sub-S12091_ses-E24868_run-1_bp-chest_vp-ap_cr.png
    The class is the first part before the first underscore.
    """
    filename = Path(image_name).stem
    
    # The class is the first part of the filename
    # Handle multi-word classes like "COVID-19" that contain hyphens
    if filename.startswith('COVID-19'):
        return 'COVID-19'
    elif filename.startswith('Non-COVID'):
        return 'Non-COVID'
    elif filename.startswith('Normal'):
        return 'Normal'
    else:
        # Fallback: take everything before the first underscore
        parts = filename.split('_')
        if parts:
            return parts[0]
        return 'unknown'


def recreate_bin_to_patch_mapping(raw_attributions: np.ndarray, n_bins: int = 49) -> Dict[int, List[int]]:
    """
    Recreate the bin-to-patch mapping based on attribution values.
    This replicates the binning logic from attribution_binning.py
    """
    n_patches = len(raw_attributions)
    if n_bins > n_patches:
        n_bins = n_patches
    
    # Sort patch indices by attribution values (descending)
    sorted_patch_indices = np.argsort(raw_attributions)[::-1]
    
    # Split into equal-sized bins
    binned_indices = np.array_split(sorted_patch_indices, n_bins)
    
    # Create bins sorted by mean attribution
    temp_bins = []
    for patch_indices in binned_indices:
        if len(patch_indices) == 0:
            continue
        bin_attributions = raw_attributions[patch_indices]
        temp_bins.append({
            "indices": patch_indices,
            "mean_attr": np.mean(bin_attributions)
        })
    
    # Sort by mean attribution (lowest first to match bin_id assignment)
    temp_bins.sort(key=lambda b: b['mean_attr'])
    
    # Create mapping from bin_id to patch indices
    bin_to_patches = {}
    for bin_id, bin_data in enumerate(temp_bins):
        bin_to_patches[bin_id] = bin_data["indices"].tolist()
    
    return bin_to_patches


def load_saco_bin_data(csv_path: str, attribution_dir: str = "results/Val/attributions") -> Dict[str, Dict[str, Any]]:
    """Load SaCo bin data and reconstruct patch-level confidence impacts."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Group by image to get per-image data
    image_data = {}
    
    for image_name, group in df.groupby('image_name'):
        # Load raw attribution for this image
        image_stem = Path(image_name).stem
        attr_file = Path(attribution_dir) / f"{image_stem}_raw_attribution.npy"
        
        if not attr_file.exists():
            # Try alternative paths
            attr_file = Path("results/Train/attributions") / f"{image_stem}_raw_attribution.npy"
            if not attr_file.exists():
                logging.warning(f"No attribution file found for {image_name}")
                continue
        
        raw_attrs = np.load(attr_file)
        
        # Recreate bin-to-patch mapping
        n_bins = len(group)  # Number of bins for this image
        bin_to_patches = recreate_bin_to_patch_mapping(raw_attrs, n_bins)
        
        # Create patch-level confidence impact data
        patch_impacts = {}
        
        for _, row in group.iterrows():
            bin_id = int(row['bin_id'])
            # confidence_delta is negative when removing the bin hurts confidence
            # so positive confidence_delta means the bin helps classification
            conf_impact = row['confidence_delta']
            
            # Assign the same impact to all patches in the bin
            if bin_id in bin_to_patches:
                for patch_id in bin_to_patches[bin_id]:
                    patch_impacts[patch_id] = conf_impact
        
        # Get overall image info
        saco_score = group['saco_score'].iloc[0]
        
        image_data[image_name] = {
            'raw_attributions': raw_attrs,
            'patch_impacts': patch_impacts,
            'saco_score': saco_score,
            'n_bins': n_bins,
            'bin_data': group.to_dict('records')  # Keep original bin data for reference
        }
    
    return image_data


@torch.no_grad()
def analyze_feature_patch_ratios(
    image_path: str,
    image_data: Dict[str, Any],
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    layer_idx: int,
    device: torch.device,
    transform: Any,
    activation_threshold: float = 0.01,
    min_patches_per_feature: int = 3,
    epsilon: float = 1e-6
) -> Dict[int, Dict[str, Any]]:
    """
    For a single image, analyze each active feature and calculate log(impact/attribution) for its patches.
    
    Returns:
        Dict mapping feature_id -> {
            'patch_ratios': List of (patch_id, log_ratio) tuples,
            'mean_log_ratio': float,
            'n_active_patches': int,
            'feature_strength': float (mean activation strength)
        }
    """
    # Load and process image
    if image_path.startswith('results/'):
        img_path = Path(image_path)
    else:
        img_path = Path(f"results/val/preprocessed/{Path(image_path).name}")
    
    if not img_path.exists():
        logging.warning(f"Image not found: {img_path}")
        return {}
    
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get SAE features
    resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
    _, cache = model.run_with_cache(img_tensor, names_filter=[resid_hook_name])
    resid = cache[resid_hook_name]
    _, codes = sae.encode(resid)
    
    # Remove CLS token
    feature_activations = codes[0, 1:]  # [n_patches, n_features]
    n_patches = feature_activations.shape[0]
    
    # Get attribution and impact data
    raw_attrs = image_data['raw_attributions']
    patch_impacts = image_data['patch_impacts']
    
    # Normalize attribution to [0, 1] for ratio calculation
    attr_min, attr_max = raw_attrs.min(), raw_attrs.max()
    if attr_max - attr_min > epsilon:
        norm_attrs = (raw_attrs - attr_min) / (attr_max - attr_min) + epsilon
    else:
        norm_attrs = np.ones_like(raw_attrs) * epsilon
    
    # Analyze each feature
    feature_analysis = {}
    
    # Find active features (any patch with activation > threshold)
    active_features = (feature_activations > activation_threshold).any(dim=0).nonzero(as_tuple=True)[0]
    
    for feat_idx in active_features:
        feat_activations = feature_activations[:, feat_idx]
        active_patches = (feat_activations > activation_threshold).nonzero(as_tuple=True)[0]
        
        if len(active_patches) < min_patches_per_feature:
            continue
        
        # Calculate log(impact/attribution) for each active patch
        log_ratios = []
        for patch_idx in active_patches:
            patch_id = patch_idx.item()
            
            # Get impact (positive = helps classification)
            impact = patch_impacts.get(patch_id, 0.0)
            
            # Get normalized attribution
            attr = norm_attrs[patch_id]
            
            # Calculate log ratio (handle edge cases)
            if abs(impact) < epsilon:
                # Near-zero impact
                log_ratio = -5.0 if attr > 0.1 else 0.0
            else:
                # Normal case: log(|impact|/attr)
                ratio = abs(impact) / attr
                log_ratio = np.log(ratio)
                # Adjust sign based on impact direction
                if impact < 0:  # Negative impact (hurts classification)
                    log_ratio = -log_ratio
            
            log_ratios.append(log_ratio)
        
        # Aggregate statistics (don't store individual patch ratios)
        mean_log_ratio = np.mean(log_ratios)
        sum_log_ratio = np.sum(log_ratios)  # Sum across patches for this feature in this image
        feature_strength = feat_activations[active_patches].mean().item()
        
        feature_analysis[feat_idx.item()] = {
            'mean_log_ratio': mean_log_ratio,
            'sum_log_ratio': sum_log_ratio,  # Sum of log ratios across patches
            'n_active_patches': len(active_patches),
            'feature_strength': feature_strength,
            'std_log_ratio': np.std(log_ratios)
        }
    
    return feature_analysis


def aggregate_feature_statistics(
    all_results: Dict[str, Dict[int, Dict[str, Any]]],
    min_occurrences: int = 10
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Aggregate feature statistics across all images.
    
    Returns dictionary with two categories:
    - 'under_attributed': Features with positive mean log ratio (high impact, low attribution)
    - 'over_attributed': Features with negative mean log ratio (low impact, high attribution)
    """
    # Collect all occurrences by feature
    feature_occurrences = defaultdict(list)
    
    for image_name, image_features in all_results.items():
        image_class = extract_class_from_image_name(image_name)
        
        for feat_id, feat_data in image_features.items():
            feature_occurrences[feat_id].append({
                'image': image_name,
                'class': image_class,
                'mean_log_ratio': feat_data['mean_log_ratio'],
                'sum_log_ratio': feat_data['sum_log_ratio'],  # NEW: Per-image sum
                'n_patches': feat_data['n_active_patches'],
                'feature_strength': feat_data['feature_strength'],
                'std_log_ratio': feat_data['std_log_ratio']
            })
    
    # Aggregate and classify features
    classified_features = {
        'under_attributed': {},
        'over_attributed': {}
    }
    
    for feat_id, occurrences in feature_occurrences.items():
        if len(occurrences) < min_occurrences:
            continue
        
        # Calculate aggregate statistics
        mean_log_ratios = [occ['mean_log_ratio'] for occ in occurrences]
        overall_mean_log_ratio = np.mean(mean_log_ratios)
        
        # Calculate confidence score based on consistency and strength
        consistency = 1.0 / (1.0 + np.std(mean_log_ratios))
        avg_n_patches = np.mean([occ['n_patches'] for occ in occurrences])
        avg_strength = np.mean([occ['feature_strength'] for occ in occurrences])
        
        # Confidence increases with more occurrences, consistency, and patch coverage
        confidence_score = (
            min(len(occurrences) / 50.0, 1.0) *  # Occurrence factor
            consistency *                         # Consistency factor
            min(avg_n_patches / 10.0, 1.0) *     # Patch coverage factor
            abs(overall_mean_log_ratio)          # Effect size
        )
        
        # Class distribution
        class_counts = defaultdict(int)
        for occ in occurrences:
            class_counts[occ['class']] += 1
        
        # Determine dominant class
        dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
        
        # Create feature statistics
        feature_stats = {
            'mean_log_ratio': overall_mean_log_ratio,
            'std_log_ratio': np.std(mean_log_ratios),
            'n_occurrences': len(occurrences),
            'confidence_score': confidence_score,
            'avg_n_patches': avg_n_patches,
            'avg_strength': avg_strength,
            'class_distribution': dict(class_counts),
            'dominant_class': dominant_class
        }
        
        # Classify based on mean log ratio
        if overall_mean_log_ratio > 0:
            classified_features['under_attributed'][feat_id] = feature_stats
        else:
            classified_features['over_attributed'][feat_id] = feature_stats
    
    # Sort by confidence score (can be changed to 'sum_of_sums' or 'sum_of_means' for comparison)
    sort_metric = 'confidence_score'  # Options: 'confidence_score', 'sum_of_sums', 'sum_of_means', 'mean_log_ratio'
    
    for category in classified_features:
        classified_features[category] = dict(
            sorted(
                classified_features[category].items(),
                key=lambda x: abs(x[1][sort_metric]),  # Use absolute value for sorting
                reverse=True
            )
        )
    
    return classified_features


def analyze_saco_features_direct(
    saco_bin_csv: str,
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    layer_idx: int = 7,
    n_images: Optional[int] = None,
    activation_threshold: float = 0.01,
    min_patches_per_feature: int = 3,
    min_occurrences: int = 10,
    attribution_dir: str = "results/Val/attributions",
    memory_efficient: bool = True,  # NEW: Enable memory-efficient processing
    batch_size: int = 500,  # NEW: Process in batches for memory efficiency
) -> Dict[str, Any]:
    """
    Main analysis function using direct feature-to-patch ratio calculation.
    Now with optional memory-efficient processing.
    """
    device = next(model.parameters()).device
    transform = get_processor_for_precached_224_images()
    
    # Load SaCo bin data
    logging.info("Loading SaCo bin data and reconstructing patch impacts...")
    image_data_map = load_saco_bin_data(saco_bin_csv, attribution_dir)
    
    images_to_process = list(image_data_map.keys())
    if n_images is not None:
        images_to_process = images_to_process[:n_images]
    
    logging.info(f"Processing {len(images_to_process)} images...")
    
    if memory_efficient:
        # Memory-efficient path: aggregate on-the-fly without batching
        logging.info("Using memory-efficient processing with online aggregation...")
        
        # Initialize online aggregator
        feature_occurrences = defaultdict(list)
        processed_count = 0
        
        # Process all images in one go, but aggregate online
        for image_name in tqdm(images_to_process, desc="Analyzing features"):
            try:
                image_features = analyze_feature_patch_ratios(
                    image_path=image_name,
                    image_data=image_data_map[image_name],
                    model=model,
                    sae=sae,
                    layer_idx=layer_idx,
                    device=device,
                    transform=transform,
                    activation_threshold=activation_threshold,
                    min_patches_per_feature=min_patches_per_feature
                )
                
                if image_features:
                    # Aggregate features online instead of storing raw results
                    image_class = extract_class_from_image_name(image_name)
                    for feat_id, feat_data in image_features.items():
                        # Only store essential stats, not patch-level details
                        feature_occurrences[feat_id].append({
                            'image': image_name,
                            'class': image_class,
                            'mean_log_ratio': feat_data['mean_log_ratio'],
                            'sum_log_ratio': feat_data['sum_log_ratio'],
                            'n_patches': feat_data['n_active_patches'],
                            'feature_strength': feat_data['feature_strength'],
                            'std_log_ratio': feat_data['std_log_ratio']
                        })
                    processed_count += 1
                    
            except Exception as e:
                logging.error(f"Error processing {image_name}: {e}")
                continue
        
        logging.info("Finished processing all images, preparing final statistics...")
        all_results = feature_occurrences  # Use the aggregated data directly
        
    else:
        # Original path: store all results (memory-intensive)
        all_results = {}
        
        for image_name in tqdm(images_to_process, desc="Analyzing features"):
            try:
                image_features = analyze_feature_patch_ratios(
                    image_path=image_name,
                    image_data=image_data_map[image_name],
                    model=model,
                    sae=sae,
                    layer_idx=layer_idx,
                    device=device,
                    transform=transform,
                    activation_threshold=activation_threshold,
                    min_patches_per_feature=min_patches_per_feature
                )
                
                if image_features:
                    all_results[image_name] = image_features
                    
            except Exception as e:
                logging.error(f"Error processing {image_name}: {e}")
                continue
    
    logging.info(f"Processed {processed_count if memory_efficient else len(all_results)} images successfully")
    
    # Aggregate results - now handles both formats
    logging.info("Aggregating feature statistics...")
    if memory_efficient:
        # Use the memory-efficient aggregation
        classified_features = aggregate_feature_statistics_memory_efficient(all_results, min_occurrences)
    else:
        classified_features = aggregate_feature_statistics(all_results, min_occurrences)
    
    # Log summary
    for category, features in classified_features.items():
        logging.info(f"\n{category.upper()} features: {len(features)} total")
        for i, (feat_id, stats) in enumerate(list(features.items())[:10]):
            logging.info(
                f"  {i+1}. Feature {feat_id}: "
                f"mean_log={stats['mean_log_ratio']:.3f}, "
                f"balanced={stats.get('balanced_score', 0):.3f}, "
                f"sum_means={stats['sum_of_means']:.3f}, "
                f"sum_all={stats['sum_of_sums']:.3f}, "
                f"conf={stats['confidence_score']:.3f}, "
                f"n_occ={stats['n_occurrences']}, "
                f"dom_class={stats['dominant_class']}"
            )
    
    return {
        'results_by_type': classified_features,
        'analysis_params': {
            'layer_idx': layer_idx,
            'activation_threshold': activation_threshold,
            'min_patches_per_feature': min_patches_per_feature,
            'min_occurrences': min_occurrences,
            'n_images_processed': processed_count if memory_efficient else len(all_results)
        }
        # Removed raw_results to reduce file size
    }


def aggregate_feature_statistics_memory_efficient(
    feature_occurrences: Dict[int, List[Dict[str, Any]]],
    min_occurrences: int = 10
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Memory-efficient aggregation for online-collected feature occurrences.
    Compatible with boosting strategy output format.
    """
    # Aggregate and classify features
    classified_features = {
        'under_attributed': {},
        'over_attributed': {}
    }
    
    for feat_id, occurrences in feature_occurrences.items():
        if len(occurrences) < min_occurrences:
            continue
        
        # Calculate aggregate statistics (same as original)
        mean_log_ratios = [occ['mean_log_ratio'] for occ in occurrences]
        sum_log_ratios_per_image = [occ['sum_log_ratio'] for occ in occurrences]
        
        overall_mean_log_ratio = np.mean(mean_log_ratios)
        sum_of_means = np.sum(mean_log_ratios)
        sum_of_sums = np.sum(sum_log_ratios_per_image)
        
        # Calculate confidence score based on consistency and strength
        consistency = 1.0 / (1.0 + np.std(mean_log_ratios))
        avg_n_patches = np.mean([occ['n_patches'] for occ in occurrences])
        avg_strength = np.mean([occ['feature_strength'] for occ in occurrences])
        
        # Confidence increases with more occurrences, consistency, and patch coverage
        confidence_score = (
            min(len(occurrences) / 50.0, 1.0) *
            consistency *
            min(avg_n_patches / 10.0, 1.0) *
            abs(overall_mean_log_ratio)
        )
        
        # Class distribution
        class_counts = defaultdict(int)
        for occ in occurrences:
            class_counts[occ['class']] += 1
        
        # Determine dominant class
        dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
        
        # Create feature statistics (SAME FORMAT as original for compatibility)
        balanced_score = abs(overall_mean_log_ratio) * np.sqrt(len(occurrences))
        
        feature_stats = {
            'mean_log_ratio': overall_mean_log_ratio,
            'sum_of_means': sum_of_means,
            'sum_of_sums': sum_of_sums,
            'std_log_ratio': np.std(mean_log_ratios),
            'n_occurrences': len(occurrences),
            'confidence_score': confidence_score,
            'balanced_score': balanced_score,
            'avg_n_patches': avg_n_patches,
            'avg_strength': avg_strength,
            'class_distribution': dict(class_counts),
            'dominant_class': dominant_class
        }
        
        # Classify based on mean log ratio
        if overall_mean_log_ratio > 0:
            classified_features['under_attributed'][feat_id] = feature_stats
        else:
            classified_features['over_attributed'][feat_id] = feature_stats
    
    # Sort by confidence score
    sort_metric = 'confidence_score'
    
    for category in classified_features:
        classified_features[category] = dict(
            sorted(
                classified_features[category].items(),
                key=lambda x: abs(x[1][sort_metric]),
                reverse=True
            )
        )
    
    return classified_features


def aggregate_feature_statistics(
    all_results: Dict[str, Dict[int, Dict[str, Any]]],  
    min_occurrences: int = 10
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Aggregate feature statistics across all images.
    
    Returns dictionary with two categories:
    - 'under_attributed': Features with positive mean log ratio (high impact, low attribution)
    - 'over_attributed': Features with negative mean log ratio (low impact, high attribution)
    """
    # Collect all occurrences by feature
    feature_occurrences = defaultdict(list)
    
    for image_name, image_features in all_results.items():
        image_class = extract_class_from_image_name(image_name)
        
        for feat_id, feat_data in image_features.items():
            feature_occurrences[feat_id].append({
                'image': image_name,
                'class': image_class,
                'mean_log_ratio': feat_data['mean_log_ratio'],
                'sum_log_ratio': feat_data['sum_log_ratio'],  # NEW: Per-image sum
                'n_patches': feat_data['n_active_patches'],
                'feature_strength': feat_data['feature_strength'],
                'std_log_ratio': feat_data['std_log_ratio']
            })
    
    # Aggregate and classify features
    classified_features = {
        'under_attributed': {},
        'over_attributed': {}
    }
    
    for feat_id, occurrences in feature_occurrences.items():
        if len(occurrences) < min_occurrences:
            continue
        
        # Calculate aggregate statistics
        mean_log_ratios = [occ['mean_log_ratio'] for occ in occurrences]
        sum_log_ratios_per_image = [occ['sum_log_ratio'] for occ in occurrences]
        
        overall_mean_log_ratio = np.mean(mean_log_ratios)
        sum_of_means = np.sum(mean_log_ratios)  # Sum of mean log ratios across images
        sum_of_sums = np.sum(sum_log_ratios_per_image)  # Sum of all patch log ratios across all images
        
        # Calculate confidence score based on consistency and strength
        consistency = 1.0 / (1.0 + np.std(mean_log_ratios))
        avg_n_patches = np.mean([occ['n_patches'] for occ in occurrences])
        avg_strength = np.mean([occ['feature_strength'] for occ in occurrences])
        
        # Confidence increases with more occurrences, consistency, and patch coverage
        confidence_score = (
            min(len(occurrences) / 50.0, 1.0) *  # Occurrence factor
            consistency *                         # Consistency factor
            min(avg_n_patches / 10.0, 1.0) *     # Patch coverage factor
            abs(overall_mean_log_ratio)          # Effect size
        )
        
        # Class distribution
        class_counts = defaultdict(int)
        for occ in occurrences:
            class_counts[occ['class']] += 1
        
        # Determine dominant class
        dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
        
        # Create feature statistics
        # Add balanced score: mean * sqrt(n_occurrences)
        balanced_score = abs(overall_mean_log_ratio) * np.sqrt(len(occurrences))
        
        feature_stats = {
            'mean_log_ratio': overall_mean_log_ratio,
            'sum_of_means': sum_of_means,  # Sum of mean log ratios across images
            'sum_of_sums': sum_of_sums,    # Sum of all patch log ratios across all images
            'std_log_ratio': np.std(mean_log_ratios),
            'n_occurrences': len(occurrences),
            'confidence_score': confidence_score,
            'balanced_score': balanced_score,  # NEW: mean * sqrt(n_occurrences)
            'avg_n_patches': avg_n_patches,
            'avg_strength': avg_strength,
            'class_distribution': dict(class_counts),
            'dominant_class': dominant_class
        }
        
        # Classify based on mean log ratio
        if overall_mean_log_ratio > 0:
            classified_features['under_attributed'][feat_id] = feature_stats
        else:
            classified_features['over_attributed'][feat_id] = feature_stats
    
    # Sort by confidence score (can be changed to 'sum_of_sums' or 'sum_of_means' for comparison)
    sort_metric = 'confidence_score'  # Options: 'confidence_score', 'sum_of_sums', 'sum_of_means', 'mean_log_ratio'
    
    for category in classified_features:
        classified_features[category] = dict(
            sorted(
                classified_features[category].items(),
                key=lambda x: abs(x[1][sort_metric]),  # Use absolute value for sorting
                reverse=True
            )
        )
    
    return classified_features


def save_analysis_results(results: Dict[str, Any], save_path: str):
    """Save analysis results to file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, save_path)
    logging.info(f"Analysis results saved to {save_path}")


if __name__ == "__main__":
    try:
        # Load model and SAE
        model = load_models()
        
        layer_idx = 6
        sae_path = Path(SAE_CONFIG[layer_idx]["sae_path"])
        sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
        sae.to(next(model.parameters()).device)
        
        # Path to SaCo bin analysis CSV
        saco_bin_csv = "results/Train/saco_bin_analysis_binned_49bins_2025-07-24_17-34.csv"
        attribution_dir = "results/Train/attributions"
        
        results = analyze_saco_features_direct(
            saco_bin_csv=saco_bin_csv,
            model=model,
            sae=sae,
            layer_idx=layer_idx,
            n_images=None,  # Process all images
            activation_threshold=0.1,
            min_patches_per_feature=3,
            min_occurrences=1,
            attribution_dir=attribution_dir,
            memory_efficient=True,  # Use memory-efficient processing
            batch_size=500  # Process 500 images at a time
        )
        
        save_path = f"results/saco_features_direct_l{layer_idx}.pt"
        save_analysis_results(results, save_path)
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save partial results if available
        if 'results' in locals() and results:
            save_path = f"results/saco_features_direct_l{layer_idx}_partial.pt"
            logging.info(f"Attempting to save partial results to {save_path}")
            try:
                save_analysis_results(results, save_path)
            except Exception as save_error:
                logging.error(f"Failed to save partial results: {save_error}")
