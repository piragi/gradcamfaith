import glob
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision  # Required for the dataset with paths
from scipy.stats import pearsonr
from tqdm import tqdm
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.sae import SparseAutoencoder

from transmm_sfaf import (IDX2CLS, get_processor_for_precached_224_images, load_models)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def batch_gini(features: torch.Tensor) -> torch.Tensor:
    """
    Calculates Gini coefficients for multiple features at once.
    Args:
        features: (n_patches, n_features) tensor
    Returns:
        (n_features,) tensor of Gini coefficients
    """
    # Handle edge cases
    if features.shape[1] == 0:
        return torch.tensor([])

    # Sort each feature column
    sorted_features, _ = torch.sort(features.abs(), dim=0)
    n = features.shape[0]
    cumsum = torch.cumsum(sorted_features, dim=0)

    # Avoid division by zero
    cumsum_last = cumsum[-1].clamp(min=1e-8)

    return (n + 1 - 2 * cumsum.sum(dim=0) / cumsum_last) / n


def batch_dice(features: torch.Tensor, target: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Calculates Dice coefficients for multiple features against a target.
    Args:
        features: (n_patches, n_features) tensor
        target: (n_patches,) tensor
        threshold: threshold for binarization
    Returns:
        (n_features,) tensor of Dice coefficients
    """
    feat_masks = (features > threshold).float()
    target_mask = (target > threshold).float().unsqueeze(1)

    intersections = (feat_masks * target_mask).sum(dim=0)
    unions = feat_masks.sum(dim=0) + target_mask.sum()

    return (2 * intersections / (unions + 1e-8))


# --- Main Dictionary Building Function ---


def build_attribution_aligned_feature_dictionary(
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    attribution_dir: str,
    n_samples: int = 1000,
    layer_idx: int = 9,
    patch_size: int = 16,
    min_occurrences: int = 5,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Builds a dictionary of features based on their alignment with baseline attribution maps,
    using vectorized operations for efficiency.

    Args:
        model: Hooked SAE Vision Transformer.
        sae: Sparse Autoencoder.
        dataloader: A DataLoader that yields (image, label, path).
        attribution_dir: Directory where pre-computed high-res attribution .npy files are stored.
        n_samples: Maximum number of samples to process.
        layer_idx: Transformer layer to analyze.
        patch_size: The size of a single patch (e.g., 16 for ViT-B/16).
        min_occurrences: Minimum times a feature must appear to be considered reliable.
        save_path: Optional path to save the final dictionary.

    Returns:
        A dictionary containing features and their alignment/localization metrics.
    """
    device = next(model.parameters()).device
    feature_occurrences = defaultdict(list)
    samples_processed = 0

    resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
    pbar = tqdm(dataloader, total=min(n_samples, len(dataloader)), desc="Analyzing feature alignment")
    reliable_features = {}

    for imgs, labels, paths in pbar:
        if samples_processed >= n_samples:
            break

        # Process one image at a time
        image, label, path = imgs[0:1].to(device), labels[0].item(), paths[0]

        # --- 1. Get SAE feature activations ---
        with torch.no_grad():
            _, cache = model.run_with_cache(image, names_filter=[resid_hook_name])
            resid = cache[resid_hook_name]
            _, codes = sae.encode(resid)

        # Patch tokens only: (196, n_features)
        feature_activations = codes[0, 1:]

        # --- 2. Load and DOWNSAMPLE the corresponding attribution map ---
        try:
            # More robustly find the attribution file
            img_filename_stem = Path(path).stem
            parts = img_filename_stem.split('_')
            prefix = 'train'
            uuid = parts[0]
            aug_part = parts[1]

            # Construct a glob pattern
            attr_pattern = str(Path(attribution_dir) / f"{prefix}_{uuid}_*_{aug_part}_attribution.npy")
            attr_files = glob.glob(attr_pattern)

            if not attr_files:
                logging.warning(f"Attribution not found for pattern {attr_pattern}, skipping.")
                continue

            attr_path = attr_files[0]
            if len(attr_files) > 1:
                logging.warning(f"Multiple attributions for {img_filename_stem}, using first: {attr_path}")

            # Load and correctly downsample the map
            attr_map_high_res = np.load(attr_path)

            # Ensure it's a 2D map before processing
            if attr_map_high_res.ndim != 2:
                logging.warning(
                    f"Unexpected attribution map shape {attr_map_high_res.shape} for {attr_path}, skipping."
                )
                continue

            # Reshape for pooling: (N, C, H, W) -> (1, 1, 224, 224)
            attr_tensor_high_res = torch.from_numpy(attr_map_high_res).unsqueeze(0).unsqueeze(0).float().to(device)

            # Use average pooling to downsample to patch resolution
            attr_tensor_patch_level_2d = F.avg_pool2d(attr_tensor_high_res, kernel_size=patch_size, stride=patch_size)

            # Flatten to a vector for correlation: (1, 1, 14, 14) -> (196,)
            attr_vec = attr_tensor_patch_level_2d.flatten()

            if attr_vec.shape[0] != feature_activations.shape[0]:
                logging.warning(
                    f"Attribution map shape mismatch after downsampling for {img_filename_stem} "
                    f"({attr_vec.shape[0]}) vs patch count "
                    f"({feature_activations.shape[0]}). Check patch_size. Skipping."
                )
                continue

            # Normalize for stable dot products and correlation
            attr_vec = (attr_vec - attr_vec.mean()) / (attr_vec.std() + 1e-8)

        except Exception as e:
            logging.warning(f"Error loading or processing attribution for {img_filename_stem}: {e}, skipping.")
            continue

        # --- 3. VECTORIZED computation of alignment metrics for active features ---
        active_feature_indices = (feature_activations.abs().sum(dim=0) > 1e-6).nonzero(as_tuple=True)[0]

        if len(active_feature_indices) == 0:
            continue

        # Get all active features at once
        active_features = feature_activations[:, active_feature_indices]  # (196, n_active)

        # 1. Vectorized correlations
        # Normalize features
        feat_means = active_features.mean(dim=0, keepdim=True)
        feat_stds = active_features.std(dim=0, keepdim=True)
        # Handle features with zero std
        valid_std_mask = feat_stds.squeeze() > 1e-6

        if valid_std_mask.sum() == 0:
            continue

        # Filter to only valid features
        valid_indices = active_feature_indices[valid_std_mask]
        valid_features = active_features[:, valid_std_mask]
        valid_feat_means = feat_means[:, valid_std_mask]
        valid_feat_stds = feat_stds[:, valid_std_mask]

        # Normalize
        active_features_norm = (valid_features - valid_feat_means) / (valid_feat_stds + 1e-8)

        # Compute correlations using matrix multiplication
        n_patches = len(attr_vec)
        correlations = torch.matmul(attr_vec.unsqueeze(0), active_features_norm).squeeze() / (n_patches - 1)

        # 2. Vectorized AWA scores
        awa_scores = torch.matmul(attr_vec.unsqueeze(0), valid_features).squeeze()

        # 3. Vectorized Gini coefficients
        gini_scores = batch_gini(valid_features)

        # 4. Vectorized Dice scores
        dice_scores = batch_dice(valid_features, attr_vec)

        # Store results
        for i, feat_idx in enumerate(valid_indices):
            feature_occurrences[feat_idx.item()].append({
                'pfac_corr': correlations[i].item(),
                'awa_score': awa_scores[i].item(),
                'gini_score': gini_scores[i].item(),
                'dice_score': dice_scores[i].item(),
                'class': label
            })

        samples_processed += 1
        pbar.set_postfix({"Processed": samples_processed, "Active Features": len(valid_indices)})

    # --- 4. Aggregate results and build final dictionary ---
    for feat_id, occurrences in feature_occurrences.items():
        if len(occurrences) < min_occurrences:
            continue

        pfac_corrs = [o['pfac_corr'] for o in occurrences]
        awa_scores = [o['awa_score'] for o in occurrences]
        gini_scores = [o['gini_score'] for o in occurrences]
        dice_scores = [o['dice_score'] for o in occurrences]
        classes = [o['class'] for o in occurrences]

        mean_pfac = np.mean(pfac_corrs)
        cv_pfac = np.std(pfac_corrs) / (abs(mean_pfac) + 1e-6)
        consistency_score = mean_pfac * (1 - cv_pfac) * np.log1p(len(occurrences))

        reliable_features[feat_id] = {
            'mean_pfac_corr': mean_pfac,
            'std_pfac_corr': np.std(pfac_corrs),
            'cv_pfac_corr': cv_pfac,
            'mean_awa_score': np.mean(awa_scores),
            'std_awa_score': np.std(awa_scores),
            'mean_gini_score': np.mean(gini_scores),
            'mean_dice_score': np.mean(dice_scores),
            'consistency_score': consistency_score,
            'occurrences': len(occurrences),
            'classes_activated': list(set(classes)),
            'raw_metrics': {
                'pfac_corrs': pfac_corrs,
                'awa_scores': awa_scores,
                'gini_scores': gini_scores,
                'dice_scores': dice_scores,
                'classes': classes
            }
        }

    # --- 5. Finalize and Save Dictionary ---
    if reliable_features:
        sorted_features = sorted(reliable_features.items(), key=lambda item: item[1]['consistency_score'], reverse=True)
        final_dict = {
            'feature_stats': dict(sorted_features),
            'metadata': {
                'layer_idx': layer_idx,
                'patch_size': patch_size,
                'n_samples_processed': samples_processed,
                'min_occurrences': min_occurrences,
                'attribution_dir': attribution_dir
            },
            'metric_definitions': {
                'pfac_corr': "Pearson Correlation between feature activations and downsampled attribution map.",
                'awa_score': "Attribution-Weighted Activation (dot product).",
                'gini_score': "Spatial concentration of feature activations.",
                'dice_score': "Spatial overlap (Dice) of thresholded maps.",
                'consistency_score': "Overall score rewarding high, stable alignment and frequency."
            }
        }
    else:
        final_dict = {'feature_stats': {}, 'metadata': {}}
        logging.warning("No reliable features found meeting the criteria.")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_dict, save_path)
        logging.info(f"Attribution alignment dictionary saved to {save_path}")

    return final_dict


sae, model = load_models()
# Build new dictionary
label_map = {2: 3, 3: 2}


def custom_target_transform(target):
    return label_map.get(target, target)


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None: sample = self.transform(sample)
        # Assuming you have a target_transform defined elsewhere
        # if self.target_transform is not None: target = self.target_transform(target)
        return sample, target, path


train_dataset = ImageFolderWithPaths("./hyper-kvasir_imagefolder/train", get_processor_for_precached_224_images())
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
layer_id = 6
stealth_dict = build_attribution_aligned_feature_dictionary(
    model,
    sae,
    dataloader,
    n_samples=50000,
    attribution_dir="./results/train/attributions",
    layer_idx=layer_id,
    min_occurrences=10,  # Must appear 3+ times
    save_path=f"./sae_dictionaries/sfaf_stealth_l{layer_id}_alignment_min10.pt"
)
