import glob
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.sae import SparseAutoencoder

from transmm_sfaf import (SAE_CONFIG, get_processor_for_precached_224_images, load_models)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_feature_steerability(
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    img: torch.Tensor,
    target_class: int,
    feature_indices: torch.Tensor,
    resid_hook_name: str,
    layer_idx: int,
    batch_size: int = 32
) -> torch.Tensor:
    """
    Compute steerability metric for features by measuring how much blacking them out hurts target prediction.
    
    Args:
        model: The hooked SAE vision transformer
        sae: The sparse autoencoder
        img: Input image tensor (1, C, H, W)
        target_class: Target class index
        feature_indices: Indices of features to test (n_features,)
        resid_hook_name: Name of the residual hook
        layer_idx: Layer index
        batch_size: Batch size for vectorized computation
        
    Returns:
        steerability_scores: Drop in target class logit for each feature (n_features,)
    """
    device = img.device
    n_features = len(feature_indices)

    # Get baseline prediction
    with torch.no_grad():
        baseline_logits = model(img)  # (1, n_classes)
        baseline_target_logit = baseline_logits[0, target_class]

    steerability_scores = torch.zeros(n_features, device=device)

    # Process features in batches for efficiency
    for start_idx in range(0, n_features, batch_size):
        end_idx = min(start_idx + batch_size, n_features)
        batch_features = feature_indices[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx

        # Create intervention functions for this batch
        def intervention_fn(activations, hook):
            # activations has shape (batch_size_actual, n_patches, d_model)
            batch_size_current = activations.shape[0]
            n_patches = activations.shape[1]

            # Process each item in the batch separately to avoid SAE batch size issues
            modified_activations_list = []

            for i in range(batch_size_current):
                # Get SAE codes for this single item
                single_activation = activations[i:i + 1]  # (1, n_patches, d_model)
                _, codes = sae.encode(single_activation)  # (1, n_patches, n_features)

                # Zero out the specific feature for this batch item
                feat_idx = batch_features[i]
                codes[0, :, feat_idx] = 0.0  # Zero out this feature across all patches

                # Decode back to activations
                modified_activation = sae.decode(codes)  # (1, n_patches, d_model)
                modified_activations_list.append(modified_activation)

            # Concatenate all modified activations
            modified_activations = torch.cat(modified_activations_list, dim=0)  # (batch_size, n_patches, d_model)
            return modified_activations

        # Run forward pass with interventions
        with torch.no_grad():
            # Repeat input for batch
            batch_img = img.repeat(batch_size_actual, 1, 1, 1)  # (batch_size, C, H, W)

            # Run with intervention
            steered_logits = model.run_with_hooks(
                batch_img, fwd_hooks=[(resid_hook_name, intervention_fn)]
            )  # (batch_size, n_classes)

            steered_target_logits = steered_logits[:, target_class]  # (batch_size,)

            # Compute drop in target class logit (positive = feature helps target class)
            logit_drops = baseline_target_logit - steered_target_logits  # (batch_size,)
            steerability_scores[start_idx:end_idx] = logit_drops

    return steerability_scores


def compute_feature_locality(feature_activations: torch.Tensor, threshold: float = 0.05) -> Dict[int, Dict[str, Any]]:
    """
    Compute locality metrics for features based on their spatial activation patterns.
    
    Args:
        feature_activations: Tensor of shape (n_patches, n_features)
        threshold: Minimum activation value to consider a patch "active"
    
    Returns:
        Dictionary mapping feature_id to locality metrics
    """
    n_patches, n_features = feature_activations.shape
    locality_metrics = {}

    # Find active features
    active_mask = feature_activations.abs() > threshold
    active_features = active_mask.any(dim=0).nonzero(as_tuple=True)[0]

    for feat_idx in active_features:
        feat_mask = active_mask[:, feat_idx]
        active_patches = feat_mask.nonzero(as_tuple=True)[0]
        n_active_patches = len(active_patches)

        if n_active_patches > 0:
            # Compute locality score (inverse of spatial spread)
            # Lower values = more local, higher values = more distributed
            locality_score = n_active_patches / n_patches

            # Store patch indices and activations for later analysis
            patch_activations = feature_activations[active_patches, feat_idx]

            locality_metrics[feat_idx.item()] = {
                'n_active_patches': n_active_patches,
                'locality_score': locality_score,  # 0-1, lower is more local
                'active_patch_indices': active_patches.cpu().tolist(),
                'patch_activation_values': patch_activations.cpu().tolist(),
                'max_activation': patch_activations.max().item(),
                'mean_activation': patch_activations.mean().item()
            }

    return locality_metrics


def compute_combined_feature_score(
    steerability: float,
    locality: float,
    frequency: int,
    pfac_corr: float,
    steer_weight: float = 0.5,
    local_weight: float = 0.3,
    freq_weight: float = 0.2,
    corr_weight: float = 0.2,
    freq_penalty_threshold: int = 100,
    locality_preference: str = "local"  # "local" or "distributed"
) -> float:
    """
    Compute a combined score that optimizes for steerability, locality, and frequency.
    
    Args:
        steerability: Steerability score (higher = more important for target)
        locality: Locality score (0-1, lower = more local)
        frequency: Number of occurrences
        pfac_corr: PFAC correlation score
        *_weight: Weights for each component
        freq_penalty_threshold: Penalize features that are too common
        locality_preference: Whether to prefer local or distributed features
    
    Returns:
        Combined score (higher is better)
    """
    # Normalize steerability (assume it's already in a reasonable range)
    steer_norm = np.clip(steerability, 0, 1)

    # Transform locality based on preference
    if locality_preference == "local":
        # Prefer local features (fewer patches)
        local_norm = 1 - locality  # Invert so lower locality -> higher score
    else:
        # Prefer distributed features
        local_norm = locality

    # Frequency score with penalty for being too common
    # Note: We use the feature's total frequency here, as a globally common
    # feature is still common, even when analyzing its role in a specific class.
    if frequency < freq_penalty_threshold:
        # Reward moderate frequency
        freq_norm = np.log1p(frequency) / np.log1p(freq_penalty_threshold)
    else:
        # Penalize excessive frequency
        freq_norm = 1 - (frequency - freq_penalty_threshold) / frequency
    freq_norm = np.clip(freq_norm, 0, 1)

    # Weighted combination
    combined_score = (steer_weight * steer_norm + local_weight * local_norm + freq_weight * freq_norm)

    return combined_score


def build_attribution_aligned_feature_dictionary(
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    attribution_dir: str,
    n_samples: int = 1000,
    layer_idx: int = 9,
    patch_size: int = 16,
    min_occurrences: int = 5,
    threshold: float = 0.10,
    locality_threshold: float = 0.05,
    save_path: Optional[str] = None,
    compute_combined_scores: bool = True,
    score_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    feature_occurrences = defaultdict(list)
    samples_processed = 0
    resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"

    # Default score weights
    if score_weights is None:
        score_weights = {'steer_weight': 0.3, 'local_weight': 0.3, 'freq_weight': 0.2, 'corr_weight': 0.2}

    pbar = tqdm(dataloader, total=min(n_samples, len(dataloader)), desc="Analyzing feature alignment")

    for imgs, labels, paths in pbar:
        if samples_processed >= n_samples:
            break

        img, label, path = imgs[0:1].to(device), labels[0].item(), paths[0]

        # SAE codes
        with torch.no_grad():
            _, cache = model.run_with_cache(img, names_filter=[resid_hook_name])
            resid = cache[resid_hook_name]
            _, codes = sae.encode(resid)

        feature_activations = codes[0, 1:]  # (n_patches, n_features)

        # Compute locality metrics for all active features
        locality_metrics = compute_feature_locality(feature_activations, locality_threshold)

        # Attribution map
        try:
            stem = Path(path).stem
            parts = stem.split('_')
            prefix = 'train'
            uuid, aug = parts[0], parts[1]
            pattern = str(Path(attribution_dir) / f"{prefix}_{uuid}_*_{aug}_attribution.npy")
            attr_files = glob.glob(pattern)

            if not attr_files:
                logging.warning(f"No attribution for pattern {pattern}")
                continue

            attr_map = np.load(attr_files[0])
            if attr_map.ndim != 2:
                logging.warning(f"Unexpected map shape {attr_map.shape} for {stem}")
                continue

            attr_tensor = F.avg_pool2d(
                torch.as_tensor(attr_map).float().unsqueeze_(0).unsqueeze_(0).to(device),
                kernel_size=patch_size,
                stride=patch_size
            ).flatten()  # (n_patches,)

            attr_tensor = (attr_tensor - attr_tensor.mean()) / (attr_tensor.std() + 1e-8)

        except Exception as e:
            logging.warning(f"Attribution load error for {stem}: {e}")
            continue

        # Compute PFAC and Steerability
        active_idx = (feature_activations.abs().sum(0) > 1e-6).nonzero(as_tuple=True)[0]
        if active_idx.numel() == 0:
            continue

        feats = feature_activations[:, active_idx]  # (n_patches, n_active)
        mu, sigma = feats.mean(0, keepdim=True), feats.std(0, keepdim=True)
        valid = (sigma.squeeze() > 1e-6).nonzero(as_tuple=True)[0]

        if valid.numel() == 0:
            continue

        idx_valid = active_idx[valid]
        feats_norm = (feats[:, valid] - mu[:, valid]) / (sigma[:, valid] + 1e-8)
        n_patches = attr_tensor.numel()
        pfac_vec = (attr_tensor @ feats_norm) / (n_patches - 1)  # (n_valid,)

        # Compute steerability metric
        steerability_scores = compute_feature_steerability(
            model, sae, img, label, idx_valid, resid_hook_name, layer_idx, batch_size=512
        )

        # Store results including locality information
        for i, fid in enumerate(idx_valid):
            fid_item = fid.item()

            # Get locality info if available
            locality_info = locality_metrics.get(fid_item, {})

            feature_occurrences[fid_item].append({
                'pfac_corr':
                pfac_vec[i].item(),
                'class':
                label,
                'steerability':
                steerability_scores[i].item() if i < len(steerability_scores) else 0.0,
                'locality_score':
                locality_info.get('locality_score', 1.0),  # Default to distributed
                'n_active_patches':
                locality_info.get('n_active_patches', 0),
                'active_patch_indices':
                locality_info.get('active_patch_indices', []),
                'max_activation':
                locality_info.get('max_activation', 0.0),
                'mean_activation':
                locality_info.get('mean_activation', 0.0)
            })

        samples_processed += 1
        pbar.set_postfix({"Processed": samples_processed})

    # Aggregate results
    reliable_features: Dict[int, Any] = {}

    for fid, occ in tqdm(feature_occurrences.items(), desc="Aggregating feature stats"):
        if len(occ) < min_occurrences:
            continue

        pfacs = [o['pfac_corr'] for o in occ]
        steers = [o['steerability'] for o in occ]
        localities = [o['locality_score'] for o in occ]
        n_patches_list = [o['n_active_patches'] for o in occ]
        classes = [o['class'] for o in occ]

        mean_pfac = float(np.mean(pfacs))
        mean_steer = float(np.mean(steers))
        mean_locality = float(np.mean(localities))
        mean_n_patches = float(np.mean(n_patches_list))

        total_frequency = len(occ)

        class_count_map = Counter(classes)
        class_mean_pfac = {}
        class_mean_steer = {}
        class_mean_locality = {}

        for cls, cnt in class_count_map.items():
            pfac_vals = [p for p, c in zip(pfacs, classes) if c == cls]
            steer_vals = [s for s, c in zip(steers, classes) if c == cls]
            local_vals = [l for l, c in zip(localities, classes) if c == cls]

            class_mean_pfac[cls] = float(np.mean(pfac_vals))
            class_mean_steer[cls] = float(np.mean(steer_vals))
            class_mean_locality[cls] = float(np.mean(local_vals))

        all_patch_indices = set()
        for o in occ:
            all_patch_indices.update(o['active_patch_indices'])

        feature_info = {
            'mean_pfac_corr': mean_pfac,
            'mean_steerability': mean_steer,
            'mean_locality_score': mean_locality,
            'mean_n_patches_active': mean_n_patches,
            'occurrences': total_frequency,
            'frequency': total_frequency,  # Alias for clarity
            'class_count_map': dict(class_count_map),
            'class_mean_pfac': class_mean_pfac,
            'class_mean_steerability': class_mean_steer,
            'class_mean_locality': class_mean_locality,
            'unique_patch_indices': sorted(list(all_patch_indices)),
            'n_unique_patches': len(all_patch_indices),
            'raw_pfacs': pfacs,
            'raw_steerability': steers,
            'raw_locality': localities
        }

        # --- MODIFICATION START ---
        # Compute combined score PER CLASS if requested
        if compute_combined_scores:
            class_combined_scores = {}
            for cls in class_mean_steer.keys():
                score = compute_combined_feature_score(
                    steerability=class_mean_steer[cls],
                    locality=class_mean_locality[cls],
                    frequency=total_frequency,  # Use total frequency for penalty
                    pfac_corr=class_mean_pfac[cls],
                    **score_weights
                )
                class_combined_scores[cls] = score

            feature_info['class_combined_scores'] = class_combined_scores
        # --- MODIFICATION END ---

        reliable_features[fid] = feature_info

    # --- MODIFICATION START ---
    # Rank features by their MAXIMUM combined score across any class
    if compute_combined_scores:
        feature_ranking = []
        for fid, info in reliable_features.items():
            if 'class_combined_scores' in info and info['class_combined_scores']:
                max_score = max(info['class_combined_scores'].values())
                feature_ranking.append((fid, max_score))

        feature_ranking = sorted(feature_ranking, key=lambda x: x[1], reverse=True)
    else:
        feature_ranking = []
    # --- MODIFICATION END ---

    final_dict = {
        'feature_stats': reliable_features,
        'feature_ranking': feature_ranking,  # List of (feature_id, max_score) tuples
        'metadata': {  # ... (rest of the function is the same)
            # ...
        },
        'metric_definitions': {
            'pfac_corr': "Pearson correlation between feature activations and attribution map",
            'steerability':
            "Drop in target class logit when feature is zeroed out (higher = more necessary for target)",
            'locality_score': "Fraction of patches where feature is active (0-1, lower = more local)",
            'frequency': "Number of times feature appears across samples",
            'class_combined_scores': "Weighted combination of steerability, locality, etc., computed per class."
        }
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_dict, save_path)
        logging.info(f"Dictionary saved to {save_path}")

        if compute_combined_scores and len(feature_ranking) > 0:
            logging.info(f"Top 10 features by combined score:")
            for i, (fid, score) in enumerate(feature_ranking[:10]):
                info = reliable_features[fid]
                logging.info(
                    f"  {i+1}. Feature {fid}: score={score:.3f}, "
                    f"steer={info['mean_steerability']:.3f}, "
                    f"local={info['mean_locality_score']:.3f}, "
                    f"freq={info['frequency']}, "
                    f"pfac={info['mean_pfac_corr']:.3f}"
                )

    return final_dict


# Main execution
if __name__ == "__main__":
    _, model = load_models()

    # Build new dictionary
    label_map = {2: 3, 3: 2}

    def custom_target_transform(target):
        return label_map.get(target, target)

    class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

        def __getitem__(self, index):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            # Assuming you have a target_transform defined elsewhere
            # if self.target_transform is not None: target = self.target_transform(target)
            return sample, target, path

    train_dataset = ImageFolderWithPaths("./hyper-kvasir_imagefolder/train", get_processor_for_precached_224_images())
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

    layers = range(2, 11)

    # Custom score weights - adjust these based on your priorities
    score_weights = {
        'steer_weight': 0.5,  # Prioritize steerability
        'local_weight': 0.3,  # Spatial locality is important
        'freq_weight': 0.2,  # Moderate frequency preference
        'corr_weight': 0.2  # Attribution correlation
    }

    for layer_idx in layers:
        sae_path = Path(SAE_CONFIG[layer_idx]["sae_path"])
        sae = SparseAutoencoder.load_from_pretrained(str(sae_path))

        build_attribution_aligned_feature_dictionary(
            model,
            sae,
            dataloader,
            n_samples=50000,
            attribution_dir="./results/train/attributions",
            layer_idx=layer_idx,
            min_occurrences=3,
            locality_threshold=0.05,  # Minimum activation to consider a patch active
            save_path=f"./sae_dictionaries/steer_corr_local_l{layer_idx}_alignment_min1_64k64.pt",
            compute_combined_scores=True,
            score_weights=score_weights
        )
