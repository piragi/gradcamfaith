"""
Optimized Feature-based Patch Impact Analysis

Memory-efficient version for processing large datasets.
Key optimizations:
- Online aggregation without storing raw results
- Streaming processing to handle any dataset size
- Compact numpy arrays for intermediate storage
- Compatible output format for transmm_sfaf.py
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
    """Extract class from image filename."""
    filename = Path(image_name).stem
    
    if filename.startswith('COVID-19'):
        return 'COVID-19'
    elif filename.startswith('Non-COVID'):
        return 'Non-COVID'
    elif filename.startswith('Normal'):
        return 'Normal'
    else:
        parts = filename.split('_')
        if parts:
            return parts[0]
        return 'unknown'


def recreate_bin_to_patch_mapping(raw_attributions: np.ndarray, n_bins: int = 49) -> Dict[int, List[int]]:
    """Recreate the bin-to-patch mapping based on attribution values."""
    n_patches = len(raw_attributions)
    if n_bins > n_patches:
        n_bins = n_patches
    
    sorted_patch_indices = np.argsort(raw_attributions)[::-1]
    binned_indices = np.array_split(sorted_patch_indices, n_bins)
    
    temp_bins = []
    for patch_indices in binned_indices:
        if len(patch_indices) == 0:
            continue
        bin_attributions = raw_attributions[patch_indices]
        temp_bins.append({
            "indices": patch_indices,
            "mean_attr": np.mean(bin_attributions)
        })
    
    temp_bins.sort(key=lambda b: b['mean_attr'])
    
    bin_to_patches = {}
    for bin_id, bin_data in enumerate(temp_bins):
        bin_to_patches[bin_id] = bin_data["indices"].tolist()
    
    return bin_to_patches


def load_saco_bin_data_streaming(csv_path: str, attribution_dir: str = "results/Val/attributions"):
    """Generator that streams SaCo bin data one image at a time."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    for image_name, group in df.groupby('image_name'):
        image_stem = Path(image_name).stem
        attr_file = Path(attribution_dir) / f"{image_stem}_raw_attribution.npy"
        
        if not attr_file.exists():
            attr_file = Path("results/Train/attributions") / f"{image_stem}_raw_attribution.npy"
            if not attr_file.exists():
                logging.warning(f"No attribution file found for {image_name}")
                continue
        
        raw_attrs = np.load(attr_file)
        n_bins = len(group)
        bin_to_patches = recreate_bin_to_patch_mapping(raw_attrs, n_bins)
        
        patch_impacts = {}
        for _, row in group.iterrows():
            bin_id = int(row['bin_id'])
            conf_impact = row['confidence_delta']
            
            if bin_id in bin_to_patches:
                for patch_id in bin_to_patches[bin_id]:
                    patch_impacts[patch_id] = conf_impact
        
        saco_score = group['saco_score'].iloc[0]
        
        yield image_name, {
            'raw_attributions': raw_attrs,
            'patch_impacts': patch_impacts,
            'saco_score': saco_score,
            'n_bins': n_bins
        }


@torch.no_grad()
def compute_feature_stats(
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
) -> Dict[int, np.ndarray]:
    """
    Compute compact feature statistics for a single image.
    Returns only essential stats as numpy arrays for memory efficiency.
    """
    if image_path.startswith('results/'):
        img_path = Path(image_path)
    else:
        img_path = Path(f"results/val/preprocessed/{Path(image_path).name}")
    
    if not img_path.exists():
        return {}
    
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
    _, cache = model.run_with_cache(img_tensor, names_filter=[resid_hook_name])
    resid = cache[resid_hook_name]
    _, codes = sae.encode(resid)
    
    feature_activations = codes[0, 1:]  # Remove CLS token
    
    raw_attrs = image_data['raw_attributions']
    patch_impacts = image_data['patch_impacts']
    
    # Normalize attribution
    attr_min, attr_max = raw_attrs.min(), raw_attrs.max()
    if attr_max - attr_min > epsilon:
        norm_attrs = (raw_attrs - attr_min) / (attr_max - attr_min) + epsilon
    else:
        norm_attrs = np.ones_like(raw_attrs) * epsilon
    
    feature_stats = {}
    active_features = (feature_activations > activation_threshold).any(dim=0).nonzero(as_tuple=True)[0]
    
    for feat_idx in active_features:
        feat_activations = feature_activations[:, feat_idx]
        active_patches = (feat_activations > activation_threshold).nonzero(as_tuple=True)[0]
        
        if len(active_patches) < min_patches_per_feature:
            continue
        
        log_ratios = []
        for patch_idx in active_patches:
            patch_id = patch_idx.item()
            impact = patch_impacts.get(patch_id, 0.0)
            attr = norm_attrs[patch_id]
            
            if abs(impact) < epsilon:
                log_ratio = -5.0 if attr > 0.1 else 0.0
            else:
                ratio = abs(impact) / attr
                log_ratio = np.log(ratio)
                if impact < 0:
                    log_ratio = -log_ratio
            
            log_ratios.append(log_ratio)
        
        # Store only essential stats as compact numpy array
        # Format: [mean_log_ratio, sum_log_ratio, std_log_ratio, n_patches, avg_strength]
        feature_stats[feat_idx.item()] = np.array([
            np.mean(log_ratios),
            np.sum(log_ratios),
            np.std(log_ratios),
            len(active_patches),
            feat_activations[active_patches].mean().item()
        ], dtype=np.float32)
    
    return feature_stats


class OnlineFeatureAggregator:
    """Maintains running statistics for features without storing all occurrences."""
    
    def __init__(self):
        self.feature_data = defaultdict(lambda: {
            'sum_mean_log': 0.0,
            'sum_sum_log': 0.0,
            'sum_sq_mean_log': 0.0,  # For computing std
            'sum_n_patches': 0.0,
            'sum_strength': 0.0,
            'n_occurrences': 0,
            'class_counts': defaultdict(int)
        })
    
    def update(self, image_name: str, feature_stats: Dict[int, np.ndarray]):
        """Update running statistics with new feature observations."""
        image_class = extract_class_from_image_name(image_name)
        
        for feat_id, stats in feature_stats.items():
            mean_log, sum_log, std_log, n_patches, strength = stats
            
            data = self.feature_data[feat_id]
            data['sum_mean_log'] += mean_log
            data['sum_sum_log'] += sum_log
            data['sum_sq_mean_log'] += mean_log ** 2
            data['sum_n_patches'] += n_patches
            data['sum_strength'] += strength
            data['n_occurrences'] += 1
            data['class_counts'][image_class] += 1
    
    def get_final_stats(self, min_occurrences: int = 10) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Compute final statistics from running aggregates."""
        classified_features = {
            'under_attributed': {},
            'over_attributed': {}
        }
        
        for feat_id, data in self.feature_data.items():
            n_occ = data['n_occurrences']
            if n_occ < min_occurrences:
                continue
            
            # Compute final statistics
            mean_log_ratio = data['sum_mean_log'] / n_occ
            sum_of_means = data['sum_mean_log']
            sum_of_sums = data['sum_sum_log']
            
            # Standard deviation using online formula
            variance = (data['sum_sq_mean_log'] / n_occ) - (mean_log_ratio ** 2)
            std_log_ratio = np.sqrt(max(0, variance))
            
            avg_n_patches = data['sum_n_patches'] / n_occ
            avg_strength = data['sum_strength'] / n_occ
            
            # Confidence score
            consistency = 1.0 / (1.0 + std_log_ratio)
            confidence_score = (
                min(n_occ / 50.0, 1.0) *
                consistency *
                min(avg_n_patches / 10.0, 1.0) *
                abs(mean_log_ratio)
            )
            
            # Class distribution
            class_counts = dict(data['class_counts'])
            dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
            
            feature_stats = {
                'mean_log_ratio': float(mean_log_ratio),
                'sum_of_means': float(sum_of_means),
                'sum_of_sums': float(sum_of_sums),
                'std_log_ratio': float(std_log_ratio),
                'n_occurrences': int(n_occ),
                'confidence_score': float(confidence_score),
                'balanced_score': float(abs(mean_log_ratio) * np.sqrt(n_occ)),
                'avg_n_patches': float(avg_n_patches),
                'avg_strength': float(avg_strength),
                'class_distribution': class_counts,
                'dominant_class': dominant_class
            }
            
            # Classify
            if mean_log_ratio > 0:
                classified_features['under_attributed'][feat_id] = feature_stats
            else:
                classified_features['over_attributed'][feat_id] = feature_stats
        
        # Sort by confidence score
        for category in classified_features:
            classified_features[category] = dict(
                sorted(
                    classified_features[category].items(),
                    key=lambda x: abs(x[1]['confidence_score']),
                    reverse=True
                )
            )
        
        return classified_features


def analyze_saco_features_optimized(
    saco_bin_csv: str,
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    layer_idx: int = 7,
    n_images: Optional[int] = None,
    activation_threshold: float = 0.01,
    min_patches_per_feature: int = 3,
    min_occurrences: int = 10,
    attribution_dir: str = "results/Val/attributions"
) -> Dict[str, Any]:
    """
    Optimized analysis function with streaming processing and online aggregation.
    Compatible with transmm_sfaf.py expected format.
    """
    device = next(model.parameters()).device
    transform = get_processor_for_precached_224_images()
    
    logging.info("Starting streaming analysis of SaCo features...")
    
    # Initialize online aggregator
    aggregator = OnlineFeatureAggregator()
    
    # Count total images for progress bar
    import pandas as pd
    total_images = len(pd.read_csv(saco_bin_csv)['image_name'].unique())
    if n_images:
        total_images = min(total_images, n_images)
    
    processed_count = 0
    
    # Stream through data
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for image_name, image_data in load_saco_bin_data_streaming(saco_bin_csv, attribution_dir):
            if n_images and processed_count >= n_images:
                break
            
            try:
                feature_stats = compute_feature_stats(
                    image_path=image_name,
                    image_data=image_data,
                    model=model,
                    sae=sae,
                    layer_idx=layer_idx,
                    device=device,
                    transform=transform,
                    activation_threshold=activation_threshold,
                    min_patches_per_feature=min_patches_per_feature
                )
                
                if feature_stats:
                    aggregator.update(image_name, feature_stats)
                    processed_count += 1
                
                pbar.update(1)
                
            except Exception as e:
                logging.error(f"Error processing {image_name}: {e}")
                pbar.update(1)
                continue
    
    logging.info(f"Processed {processed_count} images successfully")
    
    # Get final statistics
    logging.info("Computing final feature statistics...")
    classified_features = aggregator.get_final_stats(min_occurrences)
    
    # Log summary
    for category, features in classified_features.items():
        logging.info(f"\n{category.upper()} features: {len(features)} total")
        for i, (feat_id, stats) in enumerate(list(features.items())[:10]):
            logging.info(
                f"  {i+1}. Feature {feat_id}: "
                f"mean_log={stats['mean_log_ratio']:.3f}, "
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
            'n_images_processed': processed_count
        }
    }


def save_analysis_results(results: Dict[str, Any], save_path: str):
    """Save analysis results in torch format compatible with transmm_sfaf.py."""
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
        
        results = analyze_saco_features_optimized(
            saco_bin_csv=saco_bin_csv,
            model=model,
            sae=sae,
            layer_idx=layer_idx,
            n_images=None,  # Process all images
            activation_threshold=0.1,
            min_patches_per_feature=3,
            min_occurrences=1,
            attribution_dir=attribution_dir
        )
        
        # Save in torch format for compatibility with transmm_sfaf.py
        save_path = f"results/saco_features_l{layer_idx}_optimized.pt"
        save_analysis_results(results, save_path)
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save partial results if available
        if 'results' in locals() and results:
            save_path = f"results/saco_features_l{layer_idx}_partial.pt"
            logging.info(f"Attempting to save partial results to {save_path}")
            try:
                save_analysis_results(results, save_path)
            except Exception as save_error:
                logging.error(f"Failed to save partial results: {save_error}")