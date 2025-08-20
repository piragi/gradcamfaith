"""
Simplified SaCo Feature Analysis
Identifies SAE features corresponding to patches with misaligned attribution vs. impact.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from vit_prisma.sae import SparseAutoencoder

from dataset_config import get_dataset_config
from pipeline_unified import load_model_for_dataset
from vit.preprocessing import get_processor_for_precached_224_images

# Suppress debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

# ============ CONFIG ============
config = {
    'datasets': ['covidquex', 'hyperkvasir'],  # List of datasets to analyze
    'layers': [4, 5, 6, 7, 8, 9, 10],  # List of layers to analyze
    'n_images': None,  # None for all, or specific number
    'activation_threshold': 0.1,  # Min activation to consider feature active
    'min_patches': 1,  # Min patches per feature
    'min_occurrences': 1,  # Min times feature must appear across dataset
    'max_features_per_image': 64,  # Process only top-k features per image

    # Path configuration
    'sae_base_dir': 'data',  # Base directory for SAE models (data/sae_hyperkvasir/...)
    'image_base_dir': 'data',  # Base directory for images (data/hyperkvasir_unified/...)
    'split': 'train',  # Which split to analyze: 'train', 'val', or 'test'
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
            temp_bins.append({"indices": indices, "mean_attr": np.mean(raw_attributions[indices])})

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
        if raw_attrs.shape != (196, ):
            print(f"Warning: Skipping {image_stem} - wrong attribution shape {raw_attrs.shape}, expected (196,)")
            continue
        bin_to_patches = recreate_bin_to_patch_mapping(raw_attrs)

        # Define proper normalization function
        def normalize_fn(x):
            """Normalize array to [0, 1] with proper epsilon handling."""
            x_min, x_max = np.min(x), np.max(x)
            if (x_max - x_min) > 1e-8:
                return (x - x_min) / (x_max - x_min + 1e-8)
            else:
                return x  # Keep original if no variation

        # Normalize attribution to [0, 1] for ratio calculation
        # norm_attrs = normalize_fn(raw_attrs)

        # Create patch-level data: each patch gets its bin's impact but individual attribution
        patch_data = {}

        # Collect all impacts for normalization (use absolute values)
        all_impacts = np.array([abs(row['confidence_delta']) for _, row in group.iterrows()])
        # norm_impacts_array = normalize_fn(all_impacts)

        # Create bin_data for reference
        bin_data = {}
        bin_idx = 0
        for _, row in group.iterrows():
            bin_id = row['bin_id']
            patch_indices = bin_to_patches.get(bin_id, [])

            bin_data[bin_id] = {
                'patch_indices': patch_indices,
                'mean_attribution': row['mean_attribution'],
                'confidence_delta': row['confidence_delta'],
                'confidence_delta_abs': row['confidence_delta_abs'],
                'bin_attribution_bias': row.get('bin_attribution_bias', 0.0)  # Get bin bias score
            }

            # Get bin attribution bias - this replaces log_ratio calculation
            bin_bias = row.get('bin_attribution_bias', 0.0)
            impact_raw = row['confidence_delta']  # Keep for reference
            
            # Assign the same bin bias to all patches in this bin
            for patch_id in patch_indices:
                patch_data[patch_id] = {
                    'log_ratio': bin_bias,  # Use bin_attribution_bias instead of log_ratio
                    'attribution': raw_attrs[patch_id],
                    'norm_attribution': raw_attrs[patch_id],  # Keep original attribution
                    'impact': impact_raw,  # Keep raw impact for reference
                    'norm_impact': abs(impact_raw),  # Use absolute impact
                    'bin_id': bin_id,
                    'bin_attribution_bias': bin_bias  # Also store explicitly
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


def load_and_process_image(image_path, transform, device, dataset_name):
    """Load and preprocess an image - reusable function."""
    if Path(image_path).exists():
        img = Image.open(image_path).convert('RGB')
    else:
        # Try in data directory with configured paths
        img_path = Path(config['image_base_dir']) / f"{dataset_name}_unified" / config['split'] / image_path
        if not img_path.exists():
            return None
        img = Image.open(img_path).convert('RGB')

    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def get_sae_encoding(img_tensor, model, sae, layer_idx):
    """Get SAE encoding for an image - reusable function."""
    with torch.no_grad():
        resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
        _, cache = model.run_with_cache(img_tensor, names_filter=[resid_hook_name])
        resid = cache[resid_hook_name]  # [1, 197, 768]

        # Get SAE features using encode method
        _, codes = sae.encode(resid)  # codes shape: [1, 197, d_sae]

    return codes, resid


def get_feature_activations(image_path, model, sae, layer_idx, device, transform, dataset_name):
    """Get SAE feature activations for a single image."""
    img_tensor = load_and_process_image(image_path, transform, device, dataset_name)
    if img_tensor is None:
        return None

    codes, _ = get_sae_encoding(img_tensor, model, sae, layer_idx)

    # Remove batch dimension and CLS token
    feature_acts = codes[0, 1:]  # [196, d_sae] - patches only, no CLS

    return feature_acts


def get_batch_feature_activations(image_paths, model, sae, layer_idx, device, transform, dataset_name, batch_size=8):
    """Get SAE feature activations for a batch of images."""
    valid_images = []
    valid_paths = []

    # Load and validate images
    for image_path in image_paths:
        img_tensor = load_and_process_image(image_path, transform, device, dataset_name)
        if img_tensor is not None:
            valid_images.append(img_tensor)
            valid_paths.append(image_path)

    if not valid_images:
        return {}

    # Process in batches
    all_feature_acts = {}

    for i in range(0, len(valid_images), batch_size):
        batch_imgs = valid_images[i:i + batch_size]
        batch_paths = valid_paths[i:i + batch_size]

        # Stack into batch tensor
        batch_tensor = torch.cat(batch_imgs, dim=0)  # [batch, 3, 224, 224]

        with torch.no_grad():
            resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
            _, cache = model.run_with_cache(batch_tensor, names_filter=[resid_hook_name])
            resid = cache[resid_hook_name]  # [batch, 197, 768]

            # Get SAE features using encode method
            _, codes = sae.encode(resid)  # codes shape: [batch, 197, d_sae]

            # Process each image in batch
            for j, path in enumerate(batch_paths):
                # Remove CLS token
                feature_acts = codes[j, 1:]  # [196, d_sae] - patches only, no CLS
                all_feature_acts[path] = feature_acts

    return all_feature_acts


def analyze_image_features(feature_acts, image_data):
    """
    Analyze SAE features for a single image using pre-computed feature activations.
    Vectorized version for better performance.
    """
    if feature_acts is None:
        return None

    feature_analysis = {}
    patch_data = image_data['patch_data']

    # Pre-compute patch data as tensors for vectorized operations
    n_patches = 196
    log_ratios_tensor = torch.zeros(n_patches, device=feature_acts.device)
    impacts_tensor = torch.zeros(n_patches, device=feature_acts.device)
    valid_mask = torch.zeros(n_patches, dtype=torch.bool, device=feature_acts.device)

    for patch_id, data in patch_data.items():
        if patch_id < n_patches:  # Safety check
            log_ratios_tensor[patch_id] = float(data['log_ratio'])
            impacts_tensor[patch_id] = float(data['impact'])
            valid_mask[patch_id] = True

    # Get top-k features based on MAX activation across all patches
    # Take the maximum SAE activation across all 196 patches to find most active features
    max_activation_per_feature = feature_acts.max(dim=0).values  # [d_sae]
    k = min(config['max_features_per_image'], feature_acts.shape[1])

    # Select top-k features by maximum activation
    top_max_acts, top_indices = torch.topk(max_activation_per_feature, k=k)

    # Filter by threshold
    threshold_mask = top_max_acts > config['activation_threshold']
    active_features = top_indices[threshold_mask]

    if len(active_features) == 0:
        return feature_analysis

    # Process all active features at once
    threshold = config['activation_threshold']
    min_patches = config['min_patches']

    # Get activations for all active features at once
    active_feat_acts = feature_acts[:, active_features]  # [196, n_active_features]

    # Create masks for active patches per feature
    active_masks = active_feat_acts > threshold  # [196, n_active_features]

    # Process each feature
    for i, feat_idx in enumerate(active_features):
        feat_mask = active_masks[:, i]  # [196]
        combined_mask = feat_mask & valid_mask  # Only patches that are active AND have data

        n_active = combined_mask.sum().item()
        if n_active < min_patches:
            continue

        # Get activations for this feature
        feat_acts = active_feat_acts[:, i]  # [196]
        active_feat_acts_masked = feat_acts[combined_mask]

        # Vectorized computation of metrics
        active_log_ratios = log_ratios_tensor[combined_mask]
        active_impacts = impacts_tensor[combined_mask]

        # Compute weighted impacts
        activation_weights = active_feat_acts_masked
        weighted_impacts = active_impacts * activation_weights
        total_activation = activation_weights.sum().item()

        feature_analysis[feat_idx.item()] = {
            # Bin bias metrics (positive = under-attributed, negative = over-attributed)
            'mean_log_ratio': active_log_ratios.mean().item(),  # Now using bin_attribution_bias
            'sum_log_ratio': active_log_ratios.sum().item(),
            'std_log_ratio': active_log_ratios.std().item() if n_active > 1 else 0,

            # Pure impact metrics
            'mean_impact': active_impacts.mean().item(),
            'weighted_impact': weighted_impacts.sum().item() / total_activation if total_activation > 0 else None,
            'std_impact': active_impacts.std().item() if n_active > 1 else 0,
            'max_impact': active_impacts.abs().max().item(),

            # Common metrics
            'n_active_patches': n_active,
            'feature_strength': active_feat_acts_masked.mean().item(),
            'mean_activation': active_feat_acts_masked.mean().item(),
            'max_activation': active_feat_acts_masked.max().item()
        }

    return feature_analysis


def process_images_batch(image_list, image_data_map, model, sae, layer_idx, device, transform, dataset_name):
    """Process all images and extract feature occurrences."""
    feature_occurrences = defaultdict(list)
    batch_size = 32  # Increased since analysis is now faster

    print(f"Processing images in batches of {batch_size}...")

    for batch_start in tqdm(range(0, len(image_list), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(image_list))
        batch_images = image_list[batch_start:batch_end]

        # Get feature activations for entire batch
        batch_feature_acts = get_batch_feature_activations(
            batch_images,
            model,
            sae,
            layer_idx,
            device,
            transform,
            dataset_name,
            batch_size=min(16, batch_size)  # Sub-batch for memory efficiency
        )

        # Analyze each image in the batch
        for image_name in batch_images:
            try:
                if image_name not in batch_feature_acts:
                    continue

                feature_acts = batch_feature_acts[image_name]

                # Analyze features for SaCo
                image_features = analyze_image_features(feature_acts, image_data_map[image_name])

                if image_features:
                    image_class = image_data_map[image_name]['true_label']

                    for feat_id, feat_data in image_features.items():
                        feature_occurrences[feat_id].append({
                            'image': image_name,
                            'class': image_class,
                            'mean_log_ratio': feat_data['mean_log_ratio'],
                            'sum_log_ratio': feat_data['sum_log_ratio'],
                            'n_patches': feat_data['n_active_patches'],
                            'feature_strength': feat_data['feature_strength'],
                            'std_log_ratio': feat_data['std_log_ratio'],
                            'mean_impact': feat_data['mean_impact'],
                            'weighted_impact': feat_data['weighted_impact'],
                            'std_impact': feat_data['std_impact'],
                            'max_impact': feat_data['max_impact']
                        })
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return feature_occurrences


def aggregate_feature_statistics(feature_occurrences):
    """Aggregate feature occurrences into statistics."""
    aggregated_features = {}

    for feat_id, occurrences in feature_occurrences.items():
        if len(occurrences) >= config['min_occurrences']:
            mean_ratios = [occ['mean_log_ratio'] for occ in occurrences]
            mean_impacts = [occ['mean_impact'] for occ in occurrences if occ['mean_impact'] is not None]

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
                'mean_impact': np.mean(mean_impacts) if mean_impacts else None,
                'std_impact': np.std(mean_impacts) if len(mean_impacts) > 1 else None,
                'n_occurrences': len(occurrences),
                'classes': list(set(occ['class'] for occ in occurrences)),
                'dominant_class': dominant_class,
                'mean_feature_strength': np.mean([occ['feature_strength'] for occ in occurrences])
            }

    return aggregated_features


def classify_features(aggregated_features):
    """Classify features into under/over-attributed categories."""
    under_attributed = {}  # High impact/low attribution (boost) - positive log ratio
    over_attributed = {}  # Low impact/high attribution (suppress) - negative log ratio

    for feat_id, info in aggregated_features.items():
        ratio = info['mean_log_ratio']
        if ratio > 0:
            under_attributed[feat_id] = info
        else:
            over_attributed[feat_id] = info

    # Sort by mean_log_ratio
    sort_metric = 'mean_log_ratio'
    under_attributed = dict(sorted(under_attributed.items(), key=lambda x: abs(x[1][sort_metric]), reverse=True))
    over_attributed = dict(sorted(over_attributed.items(), key=lambda x: abs(x[1][sort_metric]), reverse=True))

    return under_attributed, over_attributed


def save_results(aggregated_features, under_attributed, over_attributed, dataset_name, layer_idx):
    """Save analysis results and print summary."""
    # Update and save feature dictionary
    feature_dict = load_or_create_feature_dict(dataset_name, layer_idx)

    for feat_id, feat_info in aggregated_features.items():
        if feat_id not in feature_dict:
            feature_dict[feat_id] = {}

        feature_dict[feat_id].update({
            'mean_log_ratio': feat_info['mean_log_ratio'],
            'n_occurrences': feat_info['n_occurrences'],
            'classes': feat_info['classes'],
            'dataset': dataset_name,
            'layer': layer_idx
        })

    save_feature_dict(feature_dict, dataset_name, layer_idx)

    # Print summary
    print("\n" + "=" * 50)
    print(f"ANALYSIS SUMMARY - {dataset_name} Layer {layer_idx}")
    print("=" * 50)
    print(f"Total features analyzed: {len(aggregated_features)}")
    print(f"Under-attributed (boost): {len(under_attributed)} features")
    print(f"Over-attributed (suppress): {len(over_attributed)} features")

    # Print top features
    print("\nTop 5 UNDER-ATTRIBUTED features (positive bias, need boost):")
    for i, (feat_id, stats) in enumerate(list(under_attributed.items())[:5]):
        impact_str = f"impact={stats['mean_impact']:.4f}, " if stats.get('mean_impact') is not None else ""
        print(
            f"  {i+1}. Feature {feat_id}: {impact_str}bin_bias={stats['mean_log_ratio']:.3f}, "
            f"n_occ={stats['n_occurrences']}, classes={stats['classes']}"
        )

    print("\nTop 5 OVER-ATTRIBUTED features (negative bias, need suppression):")
    for i, (feat_id, stats) in enumerate(list(over_attributed.items())[:5]):
        impact_str = f"impact={stats['mean_impact']:.4f}, " if stats.get('mean_impact') is not None else ""
        print(
            f"  {i+1}. Feature {feat_id}: {impact_str}bin_bias={stats['mean_log_ratio']:.3f}, "
            f"n_occ={stats['n_occurrences']}, classes={stats['classes']}"
        )

    # Save results in v2 format
    results = {
        'results_by_type': {
            'under_attributed': under_attributed,
            'over_attributed': over_attributed
        },
        'analysis_params': {
            'layer_idx': layer_idx,
            'activation_threshold': config['activation_threshold'],
            'min_patches_per_feature': config['min_patches'],
            'min_occurrences': config['min_occurrences'],
            'n_images_processed': len(aggregated_features),
            'dataset': dataset_name
        }
    }

    save_path = Path(f"data/featuredict_{dataset_name}/layer_{layer_idx}_saco_features.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, save_path)
    print(f"\nResults saved to {save_path}")


def process_dataset_layer(dataset_name, layer_idx, device):
    """Process a single dataset-layer combination."""
    print(f"\n{'='*80}")
    print(f"Processing: Dataset={dataset_name}, Layer={layer_idx}")
    print(f"{'='*80}\n")

    dataset_config = get_dataset_config(dataset_name)

    print(f"Analyzing {dataset_name} dataset, layer {layer_idx}")
    print(
        f"Settings: activation_threshold={config['activation_threshold']}, max_features={config['max_features_per_image']}"
    )

    # Load model and SAE
    model = load_model_for_dataset(dataset_config, device)
    sae = load_sae(dataset_name, layer_idx)
    sae.eval()

    # Load SaCo data
    results_dir = f'./data/{dataset_name}_unified/results/train'
    print(f"Loading SaCo analysis results from {results_dir}...")
    image_data_map = load_saco_data(dataset_name, results_dir)

    # Limit images if specified
    image_list = list(image_data_map.keys())
    if config['n_images']:
        image_list = image_list[:config['n_images']]

    print(f"Processing {len(image_list)} images...")

    # Setup transform
    transform = get_processor_for_precached_224_images()

    # Process images
    feature_occurrences = process_images_batch(
        image_list, image_data_map, model, sae, layer_idx, device, transform, dataset_name
    )

    # Aggregate and classify features
    print("\nAggregating feature statistics...")
    aggregated_features = aggregate_feature_statistics(feature_occurrences)
    under_attributed, over_attributed = classify_features(aggregated_features)

    # Save results
    save_results(aggregated_features, under_attributed, over_attributed, dataset_name, layer_idx)

    # Clean up
    del model, sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


def main():
    """Main entry point - coordinates processing of all datasets and layers."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process each dataset and layer combination
    for dataset_name in config['datasets']:
        for layer_idx in config['layers']:
            process_dataset_layer(dataset_name, layer_idx, device)

    print(f"\n{'='*80}")
    print("All datasets and layers processed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
