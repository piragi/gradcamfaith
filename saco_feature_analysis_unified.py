"""
Feature-based Patch Impact Analysis - Unified Version

This module is adapted from saco_feature_analysis_v2.py to work with the unified
dataset system. It identifies SAE features that correspond to patches with misaligned 
attribution vs. classification impact.

Main changes from v2:
- Works with unified dataset structure
- Uses true_label from ClassificationResult
- Compatible with binned SaCo analysis output from unified pipeline
- Organized SAE/feature dict storage under ./data
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from vit.preprocessing import get_processor_for_precached_224_images
from dataset_config import get_dataset_config

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def load_saco_bin_data_unified(
    bin_results_csv: str,
    results_csv: str,
    attribution_dir: str
) -> Dict[str, Dict[str, Any]]:
    """
    Load SaCo bin data from unified pipeline output.
    
    Args:
        bin_results_csv: Path to the bin results CSV from attribution_binning
        results_csv: Path to the classification results CSV with true labels
        attribution_dir: Directory containing raw attribution files
    """
    # Load bin results
    df_bins = pd.read_csv(bin_results_csv)
    
    # Load classification results for true labels
    df_results = pd.read_csv(results_csv)
    
    # Create mapping from image path to true label
    image_to_label = {}
    for _, row in df_results.iterrows():
        image_to_label[row['image_path']] = row.get('true_label', 'unknown')
    
    # Group by image to get per-image data
    image_data = {}
    
    for image_name, group in df_bins.groupby('image_name'):
        # Extract image stem for attribution file
        image_stem = Path(image_name).stem
        attr_file = Path(attribution_dir) / f"{image_stem}_raw_attribution.npy"
        
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
            # confidence_delta is the impact on confidence
            conf_impact = row['confidence_delta']
            
            # Assign the same impact to all patches in the bin
            if bin_id in bin_to_patches:
                for patch_id in bin_to_patches[bin_id]:
                    patch_impacts[patch_id] = conf_impact
        
        # Get overall image info
        saco_score = group['saco_score'].iloc[0]
        true_label = image_to_label.get(image_name, 'unknown')
        
        image_data[image_name] = {
            'raw_attributions': raw_attrs,
            'patch_impacts': patch_impacts,
            'saco_score': saco_score,
            'true_label': true_label,
            'n_bins': n_bins,
            'bin_data': group.to_dict('records')
        }
    
    return image_data


def load_model_and_sae(
    dataset_name: str,
    layer_idx: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sae_filename: str = "n_images_49276.pt"
) -> Tuple[torch.nn.Module, Any]:
    """
    Load the ViT model and SAE for a specific dataset and layer.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'covidquex', 'hyperkvasir')
        layer_idx: Layer index for the SAE
        device: Device to load models on
        sae_filename: Name of the SAE file to load (default: 'n_images_49276.pt')
        
    Returns:
        Tuple of (model, sae)
    """
    # Load dataset config to get model checkpoint path
    dataset_config = get_dataset_config(dataset_name)
    
    # Load ViT model using vit_prisma
    try:
        from vit_prisma.models.base_vit import HookedViT
        
        # Initialize model
        model = HookedViT.from_pretrained(
            "vit_base_patch16_224",
            num_classes=dataset_config.num_classes
        )
    except ImportError:
        # Fallback to regular ViT if vit_prisma not available
        logging.warning("vit_prisma not available, using standard ViT")
        from vit.model import VisionTransformer
        model = VisionTransformer(num_classes=dataset_config.num_classes)
    
    # Load checkpoint if available
    if hasattr(dataset_config, 'model_checkpoint') and Path(dataset_config.model_checkpoint).exists():
        checkpoint = torch.load(dataset_config.model_checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logging.info(f"Loaded model checkpoint from {dataset_config.model_checkpoint}")
    
    model = model.to(device)
    model.eval()
    
    # Load SAE
    sae_dir = Path(f"data/sae_{dataset_name}/layer_{layer_idx}")
    sae_file = sae_dir / sae_filename
    
    if not sae_file.exists():
        # Try alternative path structure (from models/sweep)
        alt_sae_path = Path(f"models/sweep/sae_l{layer_idx}_k64_exp64_lr2e-05")
        if alt_sae_path.exists():
            # Find the SAE file in the sweep directory
            sae_files = list(alt_sae_path.glob(f"*/{sae_filename}"))
            if sae_files:
                sae_file = sae_files[0]
                logging.info(f"Using SAE from sweep directory: {sae_file}")
    
    if not sae_file.exists():
        raise FileNotFoundError(f"SAE file not found. Looked for:\n  - {sae_dir / sae_filename}\n  - models/sweep/sae_l{layer_idx}_*/*/{sae_filename}")
    
    logging.info(f"Loading SAE from: {sae_file}")
    
    # Load SAE (adjust based on your SAE implementation)
    try:
        from vit_prisma.sae import SparseAutoencoder
        sae = SparseAutoencoder.load_from_pretrained(str(sae_file))
    except ImportError:
        logging.warning("Using placeholder SAE - please implement proper SAE loading")
        sae = None  # You'll need to implement this based on your SAE format
    
    if sae is not None:
        sae = sae.to(device)
    
    return model, sae


def load_feature_dict(
    dataset_name: str,
    layer_idx: int,
    create_if_missing: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Load or create feature dictionary for a dataset and layer.
    
    Args:
        dataset_name: Name of the dataset
        layer_idx: Layer index
        create_if_missing: Whether to create an empty dict if file doesn't exist
        
    Returns:
        Feature dictionary mapping feature_id to feature info
    """
    feature_dict_dir = Path(f"data/featuredict_{dataset_name}")
    feature_dict_dir.mkdir(parents=True, exist_ok=True)
    
    feature_dict_path = feature_dict_dir / f"layer_{layer_idx}_features.pt"
    
    if feature_dict_path.exists():
        return torch.load(feature_dict_path)
    elif create_if_missing:
        return {}
    else:
        raise FileNotFoundError(f"Feature dictionary not found at {feature_dict_path}")


def save_feature_dict(
    feature_dict: Dict[int, Dict[str, Any]],
    dataset_name: str,
    layer_idx: int
):
    """
    Save feature dictionary for a dataset and layer.
    """
    feature_dict_dir = Path(f"data/featuredict_{dataset_name}")
    feature_dict_dir.mkdir(parents=True, exist_ok=True)
    
    feature_dict_path = feature_dict_dir / f"layer_{layer_idx}_features.pt"
    torch.save(feature_dict, feature_dict_path)
    logging.info(f"Saved feature dictionary to {feature_dict_path}")


@torch.no_grad()
def analyze_feature_patch_ratios(
    image_path: str,
    image_data: Dict[str, Any],
    model: torch.nn.Module,
    sae: Any,  # SparseAutoencoder
    layer_idx: int,
    device: torch.device,
    transform: Any,
    activation_threshold: float = 0.01,
    min_patches_per_feature: int = 3,
    epsilon: float = 1e-6
) -> Dict[int, Dict[str, Any]]:
    """
    For a single image, analyze each active feature and calculate log(impact/attribution) for its patches.
    """
    # Load image from the path
    img_path = Path(image_path)
    
    if not img_path.exists():
        logging.warning(f"Image not found: {img_path}")
        return {}
    
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Check if SAE is available
    if sae is None:
        logging.warning("SAE not loaded, skipping feature analysis")
        return {}
    
    # Get SAE features (assuming model has the necessary hooks)
    try:
        resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
        _, cache = model.run_with_cache(img_tensor, names_filter=[resid_hook_name])
        resid = cache[resid_hook_name]
        _, codes = sae.encode(resid)
    except AttributeError:
        logging.warning("Model doesn't support run_with_cache, skipping feature analysis")
        return {}
    
    # Remove CLS token
    feature_activations = codes[0, 1:]  # [n_patches, n_features]
    n_patches = feature_activations.shape[0]
    
    # Get attribution and impact data
    raw_attrs = image_data['raw_attributions']
    patch_impacts = image_data['patch_impacts']
    
    # Normalize attributions to [0, 1]
    attrs_min = raw_attrs.min()
    attrs_max = raw_attrs.max()
    if attrs_max > attrs_min:
        normalized_attrs = (raw_attrs - attrs_min) / (attrs_max - attrs_min)
    else:
        normalized_attrs = np.ones_like(raw_attrs)
    
    # Calculate log(impact/attribution) for each feature's active patches
    feature_results = {}
    
    for feat_id in range(feature_activations.shape[1]):
        feat_acts = feature_activations[:, feat_id].cpu().numpy()
        
        # Find patches where this feature is active
        active_mask = feat_acts > activation_threshold
        active_patches = np.where(active_mask)[0]
        
        if len(active_patches) < min_patches_per_feature:
            continue
        
        # Calculate log ratios for active patches
        patch_ratios = []
        for patch_id in active_patches:
            if patch_id >= len(raw_attrs):
                continue
                
            attr = normalized_attrs[patch_id]
            impact = patch_impacts.get(patch_id, 0.0)
            
            # Calculate log(|impact| / (attr + epsilon))
            # Using absolute impact to handle both positive and negative impacts
            ratio = abs(impact) / (attr + epsilon)
            log_ratio = np.log(ratio + epsilon)
            
            patch_ratios.append((patch_id, log_ratio))
        
        if patch_ratios:
            log_ratios = [r[1] for r in patch_ratios]
            feature_results[feat_id] = {
                'patch_ratios': patch_ratios,
                'mean_log_ratio': np.mean(log_ratios),
                'sum_log_ratio': np.sum(log_ratios),
                'std_log_ratio': np.std(log_ratios),
                'n_active_patches': len(patch_ratios),
                'feature_strength': np.mean(feat_acts[active_mask])
            }
    
    return feature_results


def classify_features_by_ratio(
    aggregated_features: Dict[int, Dict[str, Any]],
    percentile_thresholds: Tuple[float, float] = (10, 90)
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Classify features into categories based on their mean log ratio.
    """
    # Calculate percentiles across all features
    all_mean_ratios = [stats['mean_log_ratio'] for stats in aggregated_features.values()]
    
    if not all_mean_ratios:
        return {'high_impact_low_attr': {}, 'balanced': {}, 'low_impact_high_attr': {}}
    
    low_threshold = np.percentile(all_mean_ratios, percentile_thresholds[0])
    high_threshold = np.percentile(all_mean_ratios, percentile_thresholds[1])
    
    classified_features = {
        'high_impact_low_attr': {},  # Features with high impact but low attribution
        'balanced': {},               # Features with balanced impact/attribution
        'low_impact_high_attr': {}    # Features with low impact but high attribution
    }
    
    for feat_id, stats in aggregated_features.items():
        mean_ratio = stats['mean_log_ratio']
        
        if mean_ratio >= high_threshold:
            classified_features['high_impact_low_attr'][feat_id] = stats
        elif mean_ratio <= low_threshold:
            classified_features['low_impact_high_attr'][feat_id] = stats
        else:
            classified_features['balanced'][feat_id] = stats
    
    # Sort within each category
    for category in classified_features:
        sort_metric = 'mean_log_ratio' if category != 'balanced' else 'n_occurrences'
        classified_features[category] = dict(
            sorted(
                classified_features[category].items(),
                key=lambda x: abs(x[1][sort_metric]),
                reverse=True
            )
        )
    
    return classified_features


def analyze_saco_features_unified(
    dataset_name: str,
    layer_idx: int = 6,
    n_images: Optional[int] = None,
    activation_threshold: float = 0.01,
    min_patches_per_feature: int = 3,
    min_occurrences: int = 10,
    results_dir: str = "results/dev_weighted",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Main analysis function for unified pipeline.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'covidquex', 'hyperkvasir')
        layer_idx: SAE layer to analyze
        n_images: Number of images to process (None for all)
        activation_threshold: Minimum activation to consider a feature active
        min_patches_per_feature: Minimum patches for a feature to be included
        min_occurrences: Minimum times a feature must appear across images
        results_dir: Directory containing results from unified pipeline
        device: Device to use for computation
    """
    # Get dataset config
    dataset_config = get_dataset_config(dataset_name)
    
    # Find the latest bin results CSV
    results_path = Path(results_dir)
    bin_csv_files = list(results_path.glob("saco_bin_analysis_binned_*.csv"))
    
    if not bin_csv_files:
        raise FileNotFoundError(f"No bin analysis CSV found in {results_dir}")
    
    latest_bin_csv = sorted(bin_csv_files)[-1]
    
    # Find classification results CSV
    results_csv = results_path / f"results_{dataset_name}_unified.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    
    # Attribution directory
    attribution_dir = results_path / "attributions"
    
    logging.info(f"Loading data from {latest_bin_csv}")
    logging.info(f"Using attributions from {attribution_dir}")
    
    # Load SaCo bin data
    image_data_map = load_saco_bin_data_unified(
        bin_results_csv=str(latest_bin_csv),
        results_csv=str(results_csv),
        attribution_dir=str(attribution_dir)
    )
    
    images_to_process = list(image_data_map.keys())
    if n_images is not None:
        images_to_process = images_to_process[:n_images]
    
    logging.info(f"Processing {len(images_to_process)} images...")
    
    # Load model and SAE
    logging.info(f"Loading model and SAE for {dataset_name}, layer {layer_idx}")
    try:
        model, sae = load_model_and_sae(dataset_name, layer_idx, device, args.sae_filename)
    except FileNotFoundError as e:
        logging.error(f"Failed to load model/SAE: {e}")
        logging.info("Please ensure SAEs are organized in data/sae_{dataset_name}/layer_{layer_idx}/")
        # Continue without SAE analysis
        model, sae = None, None
    
    # Load existing feature dictionary (if any)
    feature_dict = load_feature_dict(dataset_name, layer_idx, create_if_missing=True)
    
    # Initialize aggregator
    feature_occurrences = defaultdict(list)
    processed_count = 0
    
    # Get transform for preprocessed images
    transform = get_processor_for_precached_224_images()
    
    # Process images
    for image_name in tqdm(images_to_process, desc="Analyzing features"):
        try:
            if model is not None and sae is not None:
                # Analyze features for this image
                image_features = analyze_feature_patch_ratios(
                    image_path=image_name,
                    image_data=image_data_map[image_name],
                    model=model,
                    sae=sae,
                    layer_idx=layer_idx,
                    device=torch.device(device),
                    transform=transform,
                    activation_threshold=activation_threshold,
                    min_patches_per_feature=min_patches_per_feature
                )
                
                if image_features:
                    # Aggregate features
                    image_class = image_data_map[image_name]['true_label']
                    for feat_id, feat_data in image_features.items():
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
            logging.error(f"Error processing {image_name}: {e}", exc_info=True)
    
    logging.info(f"Processed {processed_count} images successfully")
    
    # Aggregate feature statistics
    aggregated_features = {}
    for feat_id, occurrences in feature_occurrences.items():
        if len(occurrences) >= min_occurrences:
            mean_log_ratios = [occ['mean_log_ratio'] for occ in occurrences]
            aggregated_features[feat_id] = {
                'mean_log_ratio': np.mean(mean_log_ratios),
                'std_log_ratio': np.std(mean_log_ratios),
                'n_occurrences': len(occurrences),
                'occurrences': occurrences,
                'classes': list(set(occ['class'] for occ in occurrences)),
                'mean_feature_strength': np.mean([occ['feature_strength'] for occ in occurrences])
            }
    
    # Classify features by their impact/attribution ratio
    classified_features = classify_features_by_ratio(aggregated_features) if aggregated_features else {}
    
    # Update feature dictionary
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
    
    # Save updated feature dictionary
    if aggregated_features:
        save_feature_dict(feature_dict, dataset_name, layer_idx)
    
    # Aggregate results
    results = {
        'dataset': dataset_name,
        'layer_idx': layer_idx,
        'n_images': processed_count,
        'n_features_analyzed': len(aggregated_features),
        'aggregated_features': aggregated_features,
        'classified_features': classified_features,
        'feature_dict': feature_dict,
        'dataset_config': dataset_config.to_dict() if hasattr(dataset_config, 'to_dict') else str(dataset_config)
    }
    
    return results


def save_analysis_results(results: Dict[str, Any], save_path: str):
    """Save analysis results to file."""
    torch.save(results, save_path)
    logging.info(f"Results saved to {save_path}")


def create_sae_directory_structure(dataset_name: str):
    """
    Create the directory structure for SAEs and feature dictionaries.
    """
    base_dir = Path("data")
    sae_dir = base_dir / f"sae_{dataset_name}"
    feature_dict_dir = base_dir / f"featuredict_{dataset_name}"
    
    sae_dir.mkdir(parents=True, exist_ok=True)
    feature_dict_dir.mkdir(parents=True, exist_ok=True)
    
    # Create layer subdirectories for SAEs
    for layer_idx in range(12):  # Assuming ViT-B with 12 layers
        layer_dir = sae_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)
    
    logging.info(f"Created directory structure for {dataset_name}")
    logging.info(f"  SAEs: {sae_dir}")
    logging.info(f"  Feature dicts: {feature_dict_dir}")
    
    return sae_dir, feature_dict_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze SaCo features for unified pipeline")
    parser.add_argument("--dataset", type=str, default="covidquex", help="Dataset name")
    parser.add_argument("--layer", type=int, default=6, help="Layer index")
    parser.add_argument("--n-images", type=int, default=None, help="Number of images to process")
    parser.add_argument("--results-dir", type=str, default="results/dev_weighted", help="Results directory")
    parser.add_argument("--create-dirs", action="store_true", help="Create directory structure")
    parser.add_argument("--sae-filename", type=str, default="n_images_49276.pt", help="SAE filename to load")
    
    args = parser.parse_args()
    
    if args.create_dirs:
        create_sae_directory_structure(args.dataset)
    else:
        # Run analysis
        results = analyze_saco_features_unified(
            dataset_name=args.dataset,
            layer_idx=args.layer,
            n_images=args.n_images,
            activation_threshold=0.1,
            min_patches_per_feature=3,
            min_occurrences=1,
            results_dir=args.results_dir
        )
        
        save_path = f"results/saco_features_unified_{args.dataset}_l{args.layer}.pt"
        save_analysis_results(results, save_path)
