"""
Simplified SaCo Feature Analysis
Identifies SAE features corresponding to patches with misaligned attribution vs. impact.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from vit.preprocessing import get_processor_for_precached_224_images
from dataset_config import get_dataset_config
from pipeline_unified import load_model_for_dataset
from vit_prisma.sae import SparseAutoencoder

# Suppress debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

# ============ CONFIG ============
config = {
    'dataset': 'covidquex',  # 'covidquex' or 'hyperkvasir'
    'layer': 6,                 # Which layer's SAE to analyze
    'n_images': None,          # None for all, or specific number
    'activation_threshold': 0.1,  # Min activation to consider feature active
    'min_patches': 1,          # Min patches per feature
    'min_occurrences': 1,      # Min times feature must appear across dataset
    'max_features_per_image': 500,  # Process only top-k features per image (huge speedup!)
    
    # Path configuration
    'results_dir': './data/covidquex_unified/results/train',  # Where SaCo results are stored
    'sae_base_dir': 'data',        # Base directory for SAE models (data/sae_hyperkvasir/...)
    'image_base_dir': 'data',      # Base directory for images (data/hyperkvasir_unified/...)
    'split': 'train',                # Which split to analyze: 'train', 'val', or 'test'
}
# ================================


def recreate_bin_to_patch_mapping(raw_attributions, n_bins=49):
    """Recreate bin-to-patch mapping from attribution values."""
    n_patches = len(raw_attributions)
    n_bins = min(n_bins, n_patches)
    
    # Sort patches by attribution (descending)
    sorted_indices = np.argsort(raw_attributions)[::-1]
    binned_indices = np.array_split(sorted_indices, n_bins)
    
    # Create bins sorted by mean attribution
    temp_bins = []
    for indices in binned_indices:
        if len(indices) > 0:
            temp_bins.append({
                "indices": indices,
                "mean_attr": np.mean(raw_attributions[indices])
            })
    
    temp_bins.sort(key=lambda b: b['mean_attr'])
    
    # Map bin_id to patch indices
    bin_to_patches = {}
    for bin_id, bin_data in enumerate(temp_bins):
        bin_to_patches[bin_id] = bin_data["indices"].tolist()
    
    return bin_to_patches


def load_saco_data(dataset_name, results_dir):
    """Load SaCo analysis results including bin-level data."""
    results_path = Path(results_dir)
    
    # Get class mapping for the dataset
    from dataset_config import get_dataset_config
    dataset_config = get_dataset_config(dataset_name)
    idx_to_class = {i: name for i, name in enumerate(dataset_config.class_names)}
    
    # Load bin results file
    bin_csv = results_path / f"{results_dir.split('/')[-1]}_bin_results.csv"
    if not bin_csv.exists():
        # Try to find any bin_results file
        bin_files = list(results_path.glob("*_bin_results.csv"))
        if not bin_files:
            raise FileNotFoundError(f"No bin results file found in {results_path}")
        bin_csv = bin_files[0]
    
    print(f"Loading bin results from {bin_csv}")
    df_bins = pd.read_csv(bin_csv)
    
    # Load classification results for true labels
    class_csv = results_path / "classification_results_originals_explained.csv"
    if not class_csv.exists():
        # Try to find faithfulness analysis as fallback
        faith_files = list(results_path.glob("analysis_faithfulness_correctness_binned_*.csv"))
        if faith_files:
            class_csv = sorted(faith_files)[-1]
    
    df_class = pd.read_csv(class_csv) if class_csv.exists() else pd.DataFrame()
    
    # Map images to true labels
    image_to_label = {}
    if not df_class.empty:
        if 'filename' in df_class.columns:
            for _, row in df_class.iterrows():
                image_to_label[row['filename']] = row.get('true_class', 'unknown')
        elif 'image_path' in df_class.columns:
            for _, row in df_class.iterrows():
                image_to_label[row['image_path']] = row.get('true_label', 'unknown')
    
    attr_dir = results_path / "attributions"
    
    # Group by image
    image_data = {}
    for image_name, group in df_bins.groupby('image_name'):
        image_stem = Path(image_name).stem
        
        # Load raw attributions - MUST be patch-level (196,) not pixel-level (224,224)
        attr_file = attr_dir / f"{image_stem}_raw_attribution.npy"
        if not attr_file.exists():
            continue
            
        raw_attrs = np.load(attr_file)
        
        # Verify correct shape
        if raw_attrs.shape != (196,):
            print(f"Warning: Skipping {image_stem} - wrong attribution shape {raw_attrs.shape}, expected (196,)")
            continue
        bin_to_patches = recreate_bin_to_patch_mapping(raw_attrs)
        
        # Normalize attribution to [0, 1] for ratio calculation
        epsilon = 1e-6
        attr_min, attr_max = raw_attrs.min(), raw_attrs.max()
        if attr_max - attr_min > epsilon:
            norm_attrs = (raw_attrs - attr_min) / (attr_max - attr_min) + epsilon
        else:
            norm_attrs = np.ones_like(raw_attrs) * epsilon
        
        # Create patch-level data: each patch gets its bin's impact but individual attribution
        patch_data = {}
        
        # First, create bin_data for reference
        bin_data = {}
        for _, row in group.iterrows():
            bin_id = row['bin_id']
            patch_indices = bin_to_patches.get(bin_id, [])
            
            bin_data[bin_id] = {
                'patch_indices': patch_indices,
                'mean_attribution': row['mean_attribution'],
                'confidence_delta': row['confidence_delta'],
                'confidence_delta_abs': row['confidence_delta_abs']
            }
            
            # Now calculate log_ratio for each patch individually
            # Use signed impact to properly identify under vs over-attributed
            impact = row['confidence_delta']  # Positive = helps classification
            
            for patch_id in patch_indices:
                # Use individual patch's normalized attribution
                patch_norm_attr = norm_attrs[patch_id]
                
                # Calculate log ratio considering impact direction
                if abs(impact) < epsilon:
                    # Near-zero impact
                    log_ratio = -5.0 if patch_norm_attr > 0.1 else 0.0
                elif patch_norm_attr > epsilon:
                    # Normal case: log(|impact|/attr)
                    ratio = abs(impact) / patch_norm_attr
                    log_ratio = np.log(ratio)
                    # Positive impact with low attribution = under-attributed (positive log ratio)
                    # Negative impact or low impact = over-attributed (negative log ratio)
                    if impact < 0:
                        log_ratio = -log_ratio
                else:
                    log_ratio = 0.0
                
                # Clamp extreme values
                if not np.isfinite(log_ratio) or abs(log_ratio) > 10:
                    log_ratio = np.clip(log_ratio, -10, 10) if np.isfinite(log_ratio) else 0.0
                
                patch_data[patch_id] = {
                    'log_ratio': log_ratio,
                    'attribution': raw_attrs[patch_id],
                    'norm_attribution': patch_norm_attr,
                    'impact': impact,
                    'bin_id': bin_id
                }
        
        # Get true label from path structure: .../class_X/...
        true_label = 'unknown'
        if 'class_' in image_name:
            # Extract class_X from path
            parts = Path(image_name).parts
            for part in parts:
                if part.startswith('class_'):
                    class_idx = int(part.split('_')[1])
                    true_label = idx_to_class.get(class_idx, f'class_{class_idx}')
                    break
        
        # Fallback to image_to_label if available
        if true_label == 'unknown' and image_name in image_to_label:
            true_label = image_to_label[image_name]
        
        image_data[image_name] = {
            'bin_data': bin_data,
            'patch_data': patch_data,  # Add patch-level data
            'true_label': true_label,
            'raw_attributions': raw_attrs,
            'saco_score': group.iloc[0]['saco_score'] if 'saco_score' in group.columns else 0.0
        }
    
    return image_data


def load_sae(dataset_name, layer_idx):
    """Load trained SAE for given dataset and layer."""
    sae_dir = Path(config['sae_base_dir']) / f"sae_{dataset_name}" / f"layer_{layer_idx}"
    
    # Find SAE file - look in subdirectories
    sae_files = list(sae_dir.glob("*/n_images_*.pt"))
    
    # Filter out log_feature_sparsity files
    sae_files = [f for f in sae_files if 'log_feature_sparsity' not in str(f)]
    
    if not sae_files:
        raise FileNotFoundError(f"No SAE found in {sae_dir}")
    
    # Use the most recent one if multiple exist
    sae_path = sorted(sae_files)[-1]
    print(f"Loading SAE from {sae_path}")
    
    sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
    sae.cuda().eval()
    return sae


def load_or_create_feature_dict(dataset_name, layer_idx):
    """Load or create feature dictionary."""
    dict_path = Path(f"data/featuredict_{dataset_name}/layer_{layer_idx}_features.pt")
    
    if dict_path.exists():
        return torch.load(dict_path, weights_only=False)
    else:
        dict_path.parent.mkdir(parents=True, exist_ok=True)
        return {}


def save_feature_dict(feature_dict, dataset_name, layer_idx):
    """Save feature dictionary."""
    dict_path = Path(f"data/featuredict_{dataset_name}/layer_{layer_idx}_features.pt")
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feature_dict, dict_path)
    print(f"Saved feature dict to {dict_path}")


def get_feature_activations(image_path, model, sae, layer_idx, device, transform):
    """Get SAE feature activations for a single image."""
    # Load and process image
    if Path(image_path).exists():
        img = Image.open(image_path).convert('RGB')
    else:
        # Try in data directory with configured paths
        img_path = Path(config['image_base_dir']) / f"{config['dataset']}_unified" / config['split'] / image_path
        if not img_path.exists():
            return None
        img = Image.open(img_path).convert('RGB')
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get activations at specified layer
    with torch.no_grad():
        resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
        _, cache = model.run_with_cache(img_tensor, names_filter=[resid_hook_name])
        resid = cache[resid_hook_name]  # [1, 197, 768]
        
        # Get SAE features using encode method
        _, codes = sae.encode(resid)  # codes shape: [1, 197, d_sae]
        
        # Remove batch dimension and CLS token
        feature_acts = codes[0, 1:]  # [196, d_sae] - patches only, no CLS
        
    return feature_acts


def analyze_image_features(feature_acts, image_data):
    """Analyze SAE features for a single image using pre-computed feature activations."""
    if feature_acts is None:
        return None
        
    
    # Analyze features per patch - OPTIMIZED VERSION
    feature_patch_ratios = {}
    patch_data = image_data['patch_data']  # Use patch-level data
    
    # OPTIMIZATION: Process only top-k most active features instead of all 49k
    # Get maximum activation per feature across all patches
    max_acts_per_feature = feature_acts.max(dim=0).values
    
    # Get top-k features (much faster than checking all 49k)
    k = min(config['max_features_per_image'], feature_acts.shape[1])  # Process only top-k features per image
    top_values, top_indices = torch.topk(max_acts_per_feature, k=k)
    
    # Filter by threshold
    threshold_mask = top_values > config['activation_threshold']
    active_features = top_indices[threshold_mask]
    
    for feat_idx in active_features:
        feat_activations = feature_acts[:, feat_idx]
        active_patches = (feat_activations > config['activation_threshold']).nonzero(as_tuple=True)[0]
        
        
        if len(active_patches) < config['min_patches']:
            continue
            
        
        # Get log ratios for active patches using patch-level data
        log_ratios = []
        for patch_idx in active_patches:
            # Convert to regular Python int to avoid numpy comparison issues
            patch_id = patch_idx.item()
            
            # Get patch-specific log ratio
            if patch_id in patch_data:
                log_ratios.append(patch_data[patch_id]['log_ratio'])
            
        
        if log_ratios:
            # Calculate feature strength as mean of active patches (not max)
            feature_strength = feat_activations[active_patches].mean().item()
            
            feature_patch_ratios[feat_idx.item()] = {
                'mean_log_ratio': np.mean(log_ratios),
                'sum_log_ratio': np.sum(log_ratios),  # Important for aggregation
                'std_log_ratio': np.std(log_ratios),
                'n_active_patches': len(active_patches),
                'feature_strength': feature_strength
            }
    
    
    return feature_patch_ratios



def save_correlation_results(correlation_results, config):
    """
    Save and summarize attribution-SAE correlation results.
    
    Args:
        correlation_results: List of correlation metrics per image
        config: Configuration dictionary
    """
    if not correlation_results:
        print("No correlation results to save.")
        return
    
    # Convert to DataFrame for easier analysis
    df_corr = pd.DataFrame(correlation_results)
    
    # Save correlation results
    corr_save_path = Path(f"data/featuredict_{config['dataset']}/layer_{config['layer']}_attribution_sae_correlation.csv")
    corr_save_path.parent.mkdir(parents=True, exist_ok=True)
    df_corr.to_csv(corr_save_path, index=False)
    print(f"Correlation results saved to {corr_save_path}")
    
    # Compute and print summary statistics
    print("\n" + "="*50)
    print("ATTRIBUTION-SAE CORRELATION SUMMARY")
    print("="*50)
    
    # Average correlations
    corr_cols = ['pearson_max', 'spearman_max', 'pearson_mean', 'spearman_mean', 'pearson_active_count']
    for col in corr_cols:
        if col in df_corr.columns:
            mean_val = df_corr[col].mean()
            std_val = df_corr[col].std()
            pos_ratio = (df_corr[col] > 0).mean()
            print(f"{col}: {mean_val:.3f} ± {std_val:.3f} (positive: {pos_ratio*100:.1f}%)")
    
    # Top-k overlap
    print("\nTop-k Patch Overlap:")
    for k in [10, 20, 50]:
        col = f'top{k}_overlap'
        if col in df_corr.columns:
            mean_overlap = df_corr[col].mean()
            std_overlap = df_corr[col].std()
            print(f"  Top-{k}: {mean_overlap*100:.1f}% ± {std_overlap*100:.1f}%")
    
    # Feature alignment
    if 'mean_feature_alignment' in df_corr.columns:
        mean_align = df_corr['mean_feature_alignment'].mean()
        aligned_ratio = (df_corr['mean_feature_alignment'] > 1.0).mean()
        print(f"\nFeature Alignment:")
        print(f"  Mean alignment: {mean_align:.3f}")
        print(f"  Aligned (>1.0): {aligned_ratio*100:.1f}% of images")
    
    # Save summary to JSON
    summary = {
        'dataset': config['dataset'],
        'layer': config['layer'],
        'n_images': len(df_corr),
        'correlations': {},
        'top_k_overlap': {},
        'feature_alignment': {}
    }
    
    for col in corr_cols:
        if col in df_corr.columns:
            summary['correlations'][col] = {
                'mean': float(df_corr[col].mean()),
                'std': float(df_corr[col].std()),
                'positive_ratio': float((df_corr[col] > 0).mean())
            }
    
    for k in [10, 20, 50]:
        col = f'top{k}_overlap'
        if col in df_corr.columns:
            summary['top_k_overlap'][f'top_{k}'] = {
                'mean': float(df_corr[col].mean()),
                'std': float(df_corr[col].std())
            }
    
    if 'mean_feature_alignment' in df_corr.columns:
        summary['feature_alignment'] = {
            'mean': float(df_corr['mean_feature_alignment'].mean()),
            'std': float(df_corr['mean_feature_alignment'].std()),
            'aligned_ratio': float((df_corr['mean_feature_alignment'] > 1.0).mean())
        }
    
    import json
    summary_path = Path(f"data/featuredict_{config['dataset']}/layer_{config['layer']}_correlation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nCorrelation summary saved to {summary_path}")
    
    return summary


def compute_attribution_sae_correlation(
    raw_attributions: np.ndarray,
    feature_acts: torch.Tensor,
    activation_threshold: float = 0.1
) -> Dict[str, float]:
    """
    Compute correlation metrics between attribution values and SAE activations.
    
    Args:
        raw_attributions: Attribution values per patch [196]
        feature_acts: SAE feature activations [196, d_sae]
        activation_threshold: Minimum activation to consider feature active
        
    Returns:
        Dictionary of correlation metrics
    """
    from scipy.stats import spearmanr, pearsonr
    
    # Convert to numpy
    sae_acts_np = feature_acts.cpu().numpy()
    
    # Aggregate SAE activations across features
    max_acts_per_patch = sae_acts_np.max(axis=1)  # [196]
    mean_acts_per_patch = sae_acts_np.mean(axis=1)  # [196]
    active_features_per_patch = (sae_acts_np > activation_threshold).sum(axis=1)
    
    metrics = {}
    
    # Pearson correlation (linear relationship)
    try:
        pearson_max, p_val_max = pearsonr(raw_attributions, max_acts_per_patch)
        metrics['pearson_max'] = pearson_max
        metrics['pearson_max_pval'] = p_val_max
        
        pearson_mean, p_val_mean = pearsonr(raw_attributions, mean_acts_per_patch)
        metrics['pearson_mean'] = pearson_mean
        metrics['pearson_mean_pval'] = p_val_mean
        
        # Spearman correlation (monotonic relationship)
        spearman_max, sp_val_max = spearmanr(raw_attributions, max_acts_per_patch)
        metrics['spearman_max'] = spearman_max
        metrics['spearman_max_pval'] = sp_val_max
        
        spearman_mean, sp_val_mean = spearmanr(raw_attributions, mean_acts_per_patch)
        metrics['spearman_mean'] = spearman_mean
        metrics['spearman_mean_pval'] = sp_val_mean
        
        # Correlation with number of active features
        pearson_count, p_val_count = pearsonr(raw_attributions, active_features_per_patch)
        metrics['pearson_active_count'] = pearson_count
        metrics['pearson_active_count_pval'] = p_val_count
        
        # Top-k overlap analysis
        k_values = [10, 20, 50]
        for k in k_values:
            # Top-k patches by attribution
            top_k_attr = np.argsort(raw_attributions)[-k:]
            # Top-k patches by SAE activation
            top_k_sae = np.argsort(max_acts_per_patch)[-k:]
            
            # Compute overlap
            overlap = len(set(top_k_attr) & set(top_k_sae))
            overlap_ratio = overlap / k
            metrics[f'top{k}_overlap'] = overlap_ratio
            
            # Jaccard similarity
            jaccard = overlap / len(set(top_k_attr) | set(top_k_sae))
            metrics[f'top{k}_jaccard'] = jaccard
        
        # Feature alignment: Check if important features are in highly attributed patches
        feature_importance = sae_acts_np.max(axis=0)  # Max activation per feature
        top_features = np.argsort(feature_importance)[-100:]  # Top 100 features
        
        alignment_scores = []
        for feat_idx in top_features[-20:]:  # Top 20 most important features
            feat_acts = sae_acts_np[:, feat_idx]
            active_patches = np.where(feat_acts > activation_threshold)[0]
            
            if len(active_patches) > 0:
                # Average attribution of patches where this feature is active
                avg_attr = raw_attributions[active_patches].mean()
                overall_avg = raw_attributions.mean()
                
                # Relative attribution (> 1 means feature is in highly attributed areas)
                relative_attr = avg_attr / (overall_avg + 1e-8)
                alignment_scores.append(relative_attr)
        
        if alignment_scores:
            metrics['mean_feature_alignment'] = np.mean(alignment_scores)
            metrics['std_feature_alignment'] = np.std(alignment_scores)
            
    except Exception as e:
        print(f"Error computing correlations: {e}")
    
    return metrics


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_config = get_dataset_config(config['dataset'])
    
    print(f"Analyzing {config['dataset']} dataset, layer {config['layer']}")
    print(f"Settings: activation_threshold={config['activation_threshold']}, max_features={config['max_features_per_image']}")
    
    # Load model and SAE
    model = load_model_for_dataset(dataset_config, device)
    sae = load_sae(config['dataset'], config['layer'])
    sae.eval()
    
    # Load SaCo data
    print("Loading SaCo analysis results...")
    image_data_map = load_saco_data(config['dataset'], config['results_dir'])
    
    # Limit images if specified
    image_list = list(image_data_map.keys())
    if config['n_images']:
        image_list = image_list[:config['n_images']]
    
    print(f"Processing {len(image_list)} images...")
    
    # Load/create feature dictionary
    feature_dict = load_or_create_feature_dict(config['dataset'], config['layer'])
    
    # Setup transform
    transform = get_processor_for_precached_224_images()
    
    # Analyze each image
    feature_occurrences = defaultdict(list)
    correlation_results = []  # Store correlation analysis results
    
    for image_name in tqdm(image_list, desc="Analyzing images"):
        try:
            # Get feature activations once
            feature_acts = get_feature_activations(
                image_name, 
                model, sae, config['layer'], 
                device, transform
            )
            
            if feature_acts is None:
                continue
            
            # 1. Compute attribution-SAE correlation
            raw_attrs = image_data_map[image_name]['raw_attributions']
            corr_metrics = compute_attribution_sae_correlation(
                raw_attrs, 
                feature_acts, 
                activation_threshold=config['activation_threshold']
            )
            
            if corr_metrics:
                corr_metrics['image'] = image_name
                corr_metrics['class'] = image_data_map[image_name]['true_label']
                correlation_results.append(corr_metrics)
            
            # 2. Analyze features for SaCo (using same feature_acts)
            image_features = analyze_image_features(
                feature_acts,
                image_data_map[image_name]
            )
            
            if image_features:
                image_class = image_data_map[image_name]['true_label']
                
                for feat_id, feat_data in image_features.items():
                    feature_occurrences[feat_id].append({
                        'image': image_name,
                        'class': image_class,
                        'mean_log_ratio': feat_data['mean_log_ratio'],
                        'sum_log_ratio': feat_data['sum_log_ratio'],  # Important for v2 compatibility
                        'n_patches': feat_data['n_active_patches'],
                        'feature_strength': feat_data['feature_strength'],
                        'std_log_ratio': feat_data['std_log_ratio']
                    })
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
    
    # Aggregate results
    print("\nAggregating feature statistics...")
    aggregated_features = {}
    
    for feat_id, occurrences in feature_occurrences.items():
        if len(occurrences) >= config['min_occurrences']:
            mean_ratios = [occ['mean_log_ratio'] for occ in occurrences]
            
            # Get class distribution
            class_counts = {}
            for occ in occurrences:
                cls = occ['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            # Determine dominant class
            dominant_class = max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else 'unknown'
            
            aggregated_features[feat_id] = {
                'mean_log_ratio': np.mean(mean_ratios),
                'std_log_ratio': np.std(mean_ratios),
                'n_occurrences': len(occurrences),
                'classes': list(set(occ['class'] for occ in occurrences)),
                'dominant_class': dominant_class,
                'mean_feature_strength': np.mean([occ['feature_strength'] for occ in occurrences])
            }
    
    # Classify features by ratio (match v2 format)
    under_attributed = {}  # High impact/low attribution (boost) - positive log ratio
    over_attributed = {}   # Low impact/high attribution (suppress) - negative log ratio
    
    for feat_id, info in aggregated_features.items():
        ratio = info['mean_log_ratio']
        if ratio > 0:
            under_attributed[feat_id] = info
        else:
            over_attributed[feat_id] = info
    
    # Update and save feature dictionary
    for feat_id, feat_info in aggregated_features.items():
        if feat_id not in feature_dict:
            feature_dict[feat_id] = {}
        
        feature_dict[feat_id].update({
            'mean_log_ratio': feat_info['mean_log_ratio'],
            'n_occurrences': feat_info['n_occurrences'],
            'classes': feat_info['classes'],
            'dataset': config['dataset'],
            'layer': config['layer']
        })
    
    save_feature_dict(feature_dict, config['dataset'], config['layer'])
    
    # Sort features by confidence/importance (match v2)
    # You can use different metrics: 'mean_log_ratio', 'n_occurrences', etc.
    sort_metric = 'mean_log_ratio'
    
    under_attributed = dict(
        sorted(under_attributed.items(), 
               key=lambda x: abs(x[1][sort_metric]), 
               reverse=True)
    )
    
    over_attributed = dict(
        sorted(over_attributed.items(),
               key=lambda x: abs(x[1][sort_metric]),
               reverse=True)
    )
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total features analyzed: {len(aggregated_features)}")
    print(f"Under-attributed (boost): {len(under_attributed)} features")
    print(f"Over-attributed (suppress): {len(over_attributed)} features")
    
    # Print top features for each category
    print("\nTop 5 UNDER-ATTRIBUTED features (high impact, low attribution):")
    for i, (feat_id, stats) in enumerate(list(under_attributed.items())[:5]):
        print(f"  {i+1}. Feature {feat_id}: mean_log={stats['mean_log_ratio']:.3f}, "
              f"n_occ={stats['n_occurrences']}, classes={stats['classes']}")
    
    print("\nTop 5 OVER-ATTRIBUTED features (low impact, high attribution):")
    for i, (feat_id, stats) in enumerate(list(over_attributed.items())[:5]):
        print(f"  {i+1}. Feature {feat_id}: mean_log={stats['mean_log_ratio']:.3f}, "
              f"n_occ={stats['n_occurrences']}, classes={stats['classes']}")
    
    # Save results in v2 format
    results = {
        'results_by_type': {
            'under_attributed': under_attributed,
            'over_attributed': over_attributed
        },
        'analysis_params': {
            'layer_idx': config['layer'],
            'activation_threshold': config['activation_threshold'],
            'min_patches_per_feature': config['min_patches'],
            'min_occurrences': config['min_occurrences'],
            'n_images_processed': len(image_list),
            'dataset': config['dataset']
        }
    }
    
    # Save to the correct location for transmm_sfaf.py to find it
    save_path = Path(f"data/featuredict_{config['dataset']}/layer_{config['layer']}_saco_features.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, save_path)
    print(f"\nResults saved to {save_path}")
    
    # Save and summarize correlation results
    save_correlation_results(correlation_results, config)


if __name__ == "__main__":
    main()
