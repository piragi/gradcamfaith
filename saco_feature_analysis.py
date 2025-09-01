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
from pipeline import load_model_for_dataset

# Suppress debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

# ============ CONFIG ============
config = {
    'datasets': ['covidquex'],  # List of datasets to analyze
    'layers': [4],  # List of layers to analyze
    'n_images': None,  # None for all, or specific number
    'activation_threshold': 0.1,  # Min activation to consider feature active
    'min_patches': 1,  # Min patches per feature
    'min_occurrences': 5,  # Min times feature must appear across dataset (reduced for per-class analysis)
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
        bin_to_patches = recreate_bin_to_patch_mapping(raw_attrs, n_bins=49)

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

    # TEST: Use the ImageNet CLIP B-32 SAE for waterbirds
    if dataset_name == "waterbirds" and layer_idx == 4:
        print("=" * 60)
        print("TEST MODE: Using ImageNet CLIP B-32 SAE for Waterbirds")
        print("=" * 60)
        sae_path = Path("data/sae_waterbirds_clip_b32/layer_4/weights.pt")
        if not sae_path.exists():
            raise FileNotFoundError(f"ImageNet SAE not found at {sae_path}. Please run download_paper_sae.py first.")
        print(f"Loading ImageNet SAE from {sae_path}")
        sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
        sae.cuda().eval()
        return sae

    # Original logic (commented out for test)
    sae_dir = Path("data") / f"sae_{dataset_name}" / f"layer_{layer_idx}" / f"bd0c9507-vit_unified_sae"
    sae_files = list(sae_dir.glob("n_images_*.pt"))

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

    # For other datasets, raise an error for now
    # raise NotImplementedError(f"SAE loading for {dataset_name} layer {layer_idx} not implemented in test mode")


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
    # Note: B-32 has 49 patches (7x7), B-16 has 196 patches (14x14)
    feature_acts = codes[0, 1:]  # [num_patches, d_sae] - patches only, no CLS

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
                # Note: B-32 has 49 patches (7x7), B-16 has 196 patches (14x14)
                feature_acts = codes[j, 1:]  # [num_patches, d_sae] - patches only, no CLS
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
    # Dynamically determine number of patches from feature_acts
    n_patches = feature_acts.shape[0]  # Will be 49 for B-32, 196 for B-16
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
    # NEW: Track occurrences per class
    class_feature_occurrences = defaultdict(lambda: defaultdict(list))
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
                        occurrence_data = {
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
                        }
                        # Add to global feature occurrences
                        feature_occurrences[feat_id].append(occurrence_data)
                        # NEW: Add to class-specific occurrences
                        class_feature_occurrences[image_class][feat_id].append(occurrence_data)
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return feature_occurrences, class_feature_occurrences


def aggregate_feature_statistics(feature_occurrences, compute_class_aware=False):
    """Aggregate feature occurrences into statistics.
    
    Args:
        feature_occurrences: Dict of feature_id -> list of occurrences
        compute_class_aware: If True, expects nested dict with class -> feature_id -> occurrences
    """
    if compute_class_aware:
        # Process class-aware occurrences
        return aggregate_class_aware_features(feature_occurrences)

    aggregated_features = {}

    for feat_id, occurrences in feature_occurrences.items():
        if len(occurrences) >= config['min_occurrences']:
            mean_ratios = [occ['mean_log_ratio'] for occ in occurrences]
            mean_impacts = [occ['mean_impact'] for occ in occurrences if occ['mean_impact'] is not None]

            # Collect all individual log ratios across all occurrences for detailed stats
            all_ratios = []
            for occ in occurrences:
                # Each occurrence already has mean_log_ratio, but we also want sum and other stats
                all_ratios.append(occ['mean_log_ratio'])

            # Separate positive and negative values for cancellation analysis
            positive_ratios = [r for r in all_ratios if r > 0]
            negative_ratios = [r for r in all_ratios if r < 0]

            # Get class distribution
            class_counts = {}
            for occ in occurrences:
                cls = occ['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1

            # Determine dominant class
            dominant_class = max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else 'unknown'

            aggregated_features[feat_id] = {
                # Original statistics
                'mean_log_ratio':
                np.mean(mean_ratios),
                'std_log_ratio':
                np.std(mean_ratios),
                'mean_impact':
                np.mean(mean_impacts) if mean_impacts else None,
                'std_impact':
                np.std(mean_impacts) if len(mean_impacts) > 1 else None,
                'n_occurrences':
                len(occurrences),
                'classes':
                list(set(occ['class'] for occ in occurrences)),
                'dominant_class':
                dominant_class,
                'mean_feature_strength':
                np.mean([occ['feature_strength'] for occ in occurrences]),

                # New detailed statistics
                'min_log_ratio':
                np.min(all_ratios) if all_ratios else 0,
                'max_log_ratio':
                np.max(all_ratios) if all_ratios else 0,
                'median_log_ratio':
                np.median(all_ratios) if all_ratios else 0,

                # Cancellation analysis
                'sum_positive_ratios':
                np.sum(positive_ratios) if positive_ratios else 0,
                'sum_negative_ratios':
                np.sum(negative_ratios) if negative_ratios else 0,
                'n_positive_occurrences':
                len(positive_ratios),
                'n_negative_occurrences':
                len(negative_ratios),
                'net_ratio_sum':
                np.sum(all_ratios),  # Net effect after cancellation
                'cancellation_ratio':
                abs(np.sum(negative_ratios)) /
                np.sum(positive_ratios) if positive_ratios and np.sum(positive_ratios) > 0 else 0,

                # Percentiles for distribution understanding
                'percentile_25':
                np.percentile(all_ratios, 25) if all_ratios else 0,
                'percentile_75':
                np.percentile(all_ratios, 75) if all_ratios else 0,
                'iqr':
                np.percentile(all_ratios, 75) - np.percentile(all_ratios, 25) if all_ratios else 0,
            }

    return aggregated_features


def aggregate_class_aware_features(class_feature_occurrences):
    """Aggregate feature statistics per class."""
    class_aggregated = {}

    for class_name, feature_occurrences in class_feature_occurrences.items():
        class_aggregated[class_name] = {}

        for feat_id, occurrences in feature_occurrences.items():
            if len(occurrences) >= config['min_occurrences']:
                mean_ratios = [occ['mean_log_ratio'] for occ in occurrences]
                mean_impacts = [occ['mean_impact'] for occ in occurrences if occ['mean_impact'] is not None]

                # Collect all individual log ratios
                all_ratios = [occ['mean_log_ratio'] for occ in occurrences]

                # Separate positive and negative values
                positive_ratios = [r for r in all_ratios if r > 0]
                negative_ratios = [r for r in all_ratios if r < 0]

                class_aggregated[class_name][feat_id] = {
                    'mean_log_ratio': np.mean(mean_ratios),
                    'std_log_ratio': np.std(mean_ratios),
                    'mean_impact': np.mean(mean_impacts) if mean_impacts else None,
                    'std_impact': np.std(mean_impacts) if len(mean_impacts) > 1 else None,
                    'n_occurrences': len(occurrences),
                    'class': class_name,
                    'mean_feature_strength': np.mean([occ['feature_strength'] for occ in occurrences]),

                    # Detailed statistics
                    'min_log_ratio': np.min(all_ratios) if all_ratios else 0,
                    'max_log_ratio': np.max(all_ratios) if all_ratios else 0,
                    'median_log_ratio': np.median(all_ratios) if all_ratios else 0,

                    # Positive/negative analysis
                    'n_positive_occurrences': len(positive_ratios),
                    'n_negative_occurrences': len(negative_ratios),
                    'net_ratio_sum': np.sum(all_ratios),

                    # Percentiles
                    'percentile_25': np.percentile(all_ratios, 25) if all_ratios else 0,
                    'percentile_75': np.percentile(all_ratios, 75) if all_ratios else 0,
                    'iqr': np.percentile(all_ratios, 75) - np.percentile(all_ratios, 25) if all_ratios else 0,
                }

    return class_aggregated


def classify_features(aggregated_features, class_aware=False):
    """Classify features into under/over-attributed categories.
    
    Args:
        aggregated_features: Either dict of features or dict of class -> features
        class_aware: If True, process per-class features
    """
    if class_aware:
        return classify_class_aware_features(aggregated_features)

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


def classify_class_aware_features(class_aggregated_features):
    """Classify features per class into under/over-attributed/neutral categories."""
    class_classifications = {}

    for class_name, features in class_aggregated_features.items():
        under_attributed = {}
        over_attributed = {}
        neutral = {}  # Features with near-zero bias

        for feat_id, info in features.items():
            ratio = info['mean_log_ratio']
            # Use a threshold to determine if a feature is neutral
            if abs(ratio) < 0.01:  # Threshold for "neutral"
                neutral[feat_id] = info
            elif ratio > 0:
                under_attributed[feat_id] = info
            else:
                over_attributed[feat_id] = info

        # Sort by absolute mean_log_ratio
        sort_metric = 'mean_log_ratio'
        under_attributed = dict(sorted(under_attributed.items(), key=lambda x: abs(x[1][sort_metric]), reverse=True))
        over_attributed = dict(sorted(over_attributed.items(), key=lambda x: abs(x[1][sort_metric]), reverse=True))
        neutral = dict(sorted(neutral.items(), key=lambda x: abs(x[1][sort_metric])))

        class_classifications[class_name] = {
            'under_attributed': under_attributed,
            'over_attributed': over_attributed,
            'neutral': neutral
        }

    return class_classifications


def compute_shrinkage_weight(n: int, n0: float = 50.0) -> float:
    """Compute empirical Bayes shrinkage weight."""
    return n / (n + n0)


def compute_consistency(pos_count: int, neg_count: int) -> float:
    """Compute consistency: fraction of occurrences with dominant sign."""
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return max(pos_count, neg_count) / total


def create_robust_inference_dict(
    class_aggregated_features,
    dataset_name,
    layer_idx,
    min_occ_threshold=10,
    consistency_threshold=0.7,
    shrinkage_n0=50.0,
    bias_min_threshold=0.02
):
    """Create robust inference dictionary with shrinkage and reliability filtering."""
    inference_dict = {
        'dataset': dataset_name,
        'layer': layer_idx,
        'config': {
            'min_occurrences': min_occ_threshold,
            'consistency_threshold': consistency_threshold,
            'shrinkage_n0': shrinkage_n0,
            'bias_min_threshold': bias_min_threshold,
            # Inference parameters (can be tuned)
            'top_L': 5,  # Top features per patch
            'strength_k': 3.0,  # Exponential strength
            'clamp_min': 0.3,
            'clamp_max': 3.0
        },
        'by_class': {},
        'shrinkage_targets': {}  # Per-class empirical Bayes targets
    }

    # Process each class
    for class_name, features in class_aggregated_features.items():
        # Compute empirical Bayes shrinkage target (median of all biases in class)
        all_biases = [info['mean_log_ratio'] for info in features.values()]
        shrinkage_target = np.median(all_biases) if all_biases else 0.0
        inference_dict['shrinkage_targets'][class_name] = float(shrinkage_target)

        # Process features with shrinkage and reliability filtering
        reliable_features = {}

        for feat_id, info in features.items():
            n_occ = info['n_occurrences']

            # Skip if too few occurrences
            if n_occ < min_occ_threshold:
                continue

            # Compute consistency
            pos_count = info.get('n_positive_occurrences', 0)
            neg_count = info.get('n_negative_occurrences', 0)
            consistency = compute_consistency(pos_count, neg_count)

            # Skip if inconsistent
            if consistency < consistency_threshold:
                continue

            # Apply shrinkage to mean bias
            raw_bias = info['mean_log_ratio']
            shrinkage_weight = compute_shrinkage_weight(n_occ, shrinkage_n0)
            shrunk_bias = shrinkage_weight * raw_bias + (1 - shrinkage_weight) * shrinkage_target

            # Skip if bias too small after shrinkage
            if abs(shrunk_bias) < bias_min_threshold:
                continue

            # Store reliable feature
            reliable_features[feat_id] = {
                'bias': float(shrunk_bias),
                'raw_bias': float(raw_bias),
                'n': n_occ,
                'consistency': float(consistency),
                'weight': float(shrinkage_weight)
            }

        inference_dict['by_class'][class_name] = reliable_features

    # Add metadata
    inference_dict['metadata'] = {
        'total_features': sum(len(features) for features in class_aggregated_features.values()),
        'reliable_features': sum(len(features) for features in inference_dict['by_class'].values()),
        'classes': list(inference_dict['by_class'].keys())
    }

    return inference_dict


def save_class_aware_results(class_classifications, class_aggregated_features, dataset_name, layer_idx):
    """Save class-aware analysis results and print summary."""
    print("\n" + "=" * 60)
    print(f"CLASS-AWARE ANALYSIS SUMMARY - {dataset_name} Layer {layer_idx}")
    print("=" * 60)

    # Analyze cross-class feature behavior
    feature_class_behavior = defaultdict(dict)

    for class_name, features in class_aggregated_features.items():
        for feat_id, info in features.items():
            feature_class_behavior[feat_id][class_name] = {
                'mean_log_ratio':
                info['mean_log_ratio'],
                'n_occurrences':
                info['n_occurrences'],
                'category':
                'under' if info['mean_log_ratio'] > 0.01 else ('over' if info['mean_log_ratio'] < -0.01 else 'neutral')
            }

    # Find features with different behaviors across classes
    mixed_behavior_features = []
    consistent_under_features = []
    consistent_over_features = []

    for feat_id, class_info in feature_class_behavior.items():
        categories = set(info['category'] for info in class_info.values())

        if len(categories) > 1:
            mixed_behavior_features.append(feat_id)
        elif 'under' in categories:
            consistent_under_features.append(feat_id)
        elif 'over' in categories:
            consistent_over_features.append(feat_id)

    print(f"\nCross-class Feature Behavior:")
    print(f"  Features with mixed behavior: {len(mixed_behavior_features)}")
    print(f"  Consistently under-attributed: {len(consistent_under_features)}")
    print(f"  Consistently over-attributed: {len(consistent_over_features)}")

    # Print examples of mixed behavior features
    if mixed_behavior_features:
        print(f"\nExamples of features with class-dependent behavior:")
        for feat_id in mixed_behavior_features[:5]:
            print(f"\n  Feature {feat_id}:")
            for class_name, info in feature_class_behavior[feat_id].items():
                print(
                    f"    {class_name}: {info['category']} (ratio={info['mean_log_ratio']:.3f}, n={info['n_occurrences']})"
                )

    # Print per-class summaries
    for class_name, classification in class_classifications.items():
        print(f"\n{class_name} Summary:")
        print(f"  Under-attributed: {len(classification['under_attributed'])} features")
        print(f"  Over-attributed: {len(classification['over_attributed'])} features")
        print(f"  Neutral: {len(classification['neutral'])} features")

        # Top features for this class
        if classification['under_attributed']:
            print(f"\n  Top 3 under-attributed features for {class_name}:")
            for i, (feat_id, stats) in enumerate(list(classification['under_attributed'].items())[:3]):
                print(f"    {i+1}. Feature {feat_id}: bias={stats['mean_log_ratio']:.3f}, n={stats['n_occurrences']}")

        if classification['over_attributed']:
            print(f"\n  Top 3 over-attributed features for {class_name}:")
            for i, (feat_id, stats) in enumerate(list(classification['over_attributed'].items())[:3]):
                print(f"    {i+1}. Feature {feat_id}: bias={stats['mean_log_ratio']:.3f}, n={stats['n_occurrences']}")

    # Save class-aware results
    class_aware_results = {
        'class_classifications': class_classifications,
        'class_aggregated_features': class_aggregated_features,
        'feature_class_behavior': dict(feature_class_behavior),
        'mixed_behavior_features': mixed_behavior_features,
        'analysis_params': {
            'layer_idx': layer_idx,
            'activation_threshold': config['activation_threshold'],
            'min_patches_per_feature': config['min_patches'],
            'min_occurrences': config['min_occurrences'],
            'dataset': dataset_name
        }
    }

    save_path = Path(f"data/featuredict_{dataset_name}/layer_{layer_idx}_class_aware_saco_features.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(class_aware_results, save_path)
    print(f"\nClass-aware results saved to {save_path}")

    # Create and save robust inference dictionary
    print("\nCreating robust inference dictionary...")
    inference_dict = create_robust_inference_dict(
        class_aggregated_features,
        dataset_name,
        layer_idx,
        min_occ_threshold=config['min_occurrences'],
        consistency_threshold=0.7,
        shrinkage_n0=50.0,
        bias_min_threshold=0.02
    )

    # Save inference dictionary
    inference_path = Path(f"data/featuredict_{dataset_name}/layer_{layer_idx}_inference_dict.pt")
    torch.save(inference_dict, inference_path)
    print(f"Inference dictionary saved to {inference_path}")

    # Print inference dict summary
    print(f"\nInference Dictionary Summary:")
    for class_name, features in inference_dict['by_class'].items():
        print(
            f"  {class_name}: {len(features)} reliable features (shrinkage target: {inference_dict['shrinkage_targets'][class_name]:.4f})"
        )
    print(f"  Total reliable features: {inference_dict['metadata']['reliable_features']}")
    print(
        f"  Reduction: {inference_dict['metadata']['total_features']} -> {inference_dict['metadata']['reliable_features']}"
    )

    return class_aware_results


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

    # Print detailed analytics summary
    print("\n" + "=" * 50)
    print("FEATURE DISTRIBUTION ANALYTICS")
    print("=" * 50)

    # Calculate overall statistics across all features
    all_mean_ratios = [stats['mean_log_ratio'] for stats in aggregated_features.values()]
    all_std_ratios = [stats['std_log_ratio'] for stats in aggregated_features.values()]
    all_min_ratios = [stats['min_log_ratio'] for stats in aggregated_features.values()]
    all_max_ratios = [stats['max_log_ratio'] for stats in aggregated_features.values()]
    all_cancellation_ratios = [
        stats['cancellation_ratio'] for stats in aggregated_features.values() if stats['cancellation_ratio'] > 0
    ]

    # Features with mixed signals (both positive and negative occurrences)
    mixed_features = [
        feat_id for feat_id, stats in aggregated_features.items()
        if stats['n_positive_occurrences'] > 0 and stats['n_negative_occurrences'] > 0
    ]

    print(f"\nGlobal Statistics:")
    print(f"  Mean of all feature ratios: {np.mean(all_mean_ratios):.4f}")
    print(f"  Std of all feature ratios: {np.std(all_mean_ratios):.4f}")
    print(f"  Min feature ratio: {np.min(all_min_ratios):.4f}")
    print(f"  Max feature ratio: {np.max(all_max_ratios):.4f}")
    print(f"  Mean within-feature std: {np.mean(all_std_ratios):.4f}")

    print(f"\nCancellation Analysis:")
    print(
        f"  Features with mixed signals: {len(mixed_features)} ({100*len(mixed_features)/len(aggregated_features):.1f}%)"
    )
    if all_cancellation_ratios:
        print(f"  Mean cancellation ratio: {np.mean(all_cancellation_ratios):.3f}")
        print(f"  Max cancellation ratio: {np.max(all_cancellation_ratios):.3f}")

    # Find features with highest cancellation
    high_cancellation = [(feat_id, stats['cancellation_ratio']) for feat_id, stats in aggregated_features.items()
                         if stats['cancellation_ratio'] > 0.5]
    high_cancellation.sort(key=lambda x: x[1], reverse=True)

    if high_cancellation:
        print(f"\n  Top features with high cancellation (>0.5):")
        for feat_id, cancel_ratio in high_cancellation[:5]:
            stats = aggregated_features[feat_id]
            print(
                f"    Feature {feat_id}: cancellation={cancel_ratio:.2f}, "
                f"pos_sum={stats['sum_positive_ratios']:.3f}, neg_sum={stats['sum_negative_ratios']:.3f}, "
                f"net={stats['net_ratio_sum']:.3f}"
            )

    # Distribution spread analysis
    print(f"\nDistribution Spread:")
    iqr_values = [stats['iqr'] for stats in aggregated_features.values()]
    print(f"  Mean IQR across features: {np.mean(iqr_values):.4f}")
    print(f"  Max IQR: {np.max(iqr_values):.4f}")

    # Features with high variance
    high_variance_features = [(feat_id, stats['std_log_ratio']) for feat_id, stats in aggregated_features.items()
                              if stats['std_log_ratio'] > np.percentile(all_std_ratios, 90)]
    high_variance_features.sort(key=lambda x: x[1], reverse=True)

    if high_variance_features:
        print(f"\n  Top features with high variance (>90th percentile):")
        for feat_id, std in high_variance_features[:5]:
            stats = aggregated_features[feat_id]
            print(
                f"    Feature {feat_id}: std={std:.3f}, range=[{stats['min_log_ratio']:.3f}, {stats['max_log_ratio']:.3f}]"
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
    model_result = load_model_for_dataset(dataset_config, device)

    # Handle CLIP model (returns tuple) vs regular model
    if isinstance(model_result, tuple):
        model, processor = model_result
        print("CLIP model loaded - using vision encoder for SAE analysis")
        # For SAE analysis, we just need the vision encoder (model)
        # No need for the full CLIP classifier wrapper here
    else:
        model = model_result
        processor = None

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

    # Setup transform using dataset-specific configuration
    dataset_config = get_dataset_config(dataset_name)
    # Use test transforms (no augmentations) and let dataset config handle CLIP/ViT
    transform = dataset_config.get_transforms('test')

    # Process images
    feature_occurrences, class_feature_occurrences = process_images_batch(
        image_list, image_data_map, model, sae, layer_idx, device, transform, dataset_name
    )

    # Aggregate and classify features (global)
    print("\nAggregating global feature statistics...")
    aggregated_features = aggregate_feature_statistics(feature_occurrences)
    under_attributed, over_attributed = classify_features(aggregated_features)

    # Save global results
    save_results(aggregated_features, under_attributed, over_attributed, dataset_name, layer_idx)

    # Aggregate and classify features (per-class)
    print("\nAggregating class-aware feature statistics...")
    class_aggregated_features = aggregate_feature_statistics(class_feature_occurrences, compute_class_aware=True)
    class_classifications = classify_features(class_aggregated_features, class_aware=True)

    # Save class-aware results
    save_class_aware_results(class_classifications, class_aggregated_features, dataset_name, layer_idx)

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
