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
    threshold: float = 0.10,
    save_path: Optional[str] = None
) -> Dict[str, Any]:

    device = next(model.parameters()).device
    feature_occurrences = defaultdict(list)
    samples_processed = 0

    resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
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

        # Compute PFAC only
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
        for i, fid in enumerate(idx_valid):
            feature_occurrences[fid.item()].append({'pfac_corr': pfac_vec[i].item(), 'class': label})

        samples_processed += 1
        pbar.set_postfix({"Processed": samples_processed})

    # Aggregate results
    reliable_features: Dict[int, Any] = {}
    for fid, occ in feature_occurrences.items():
        if len(occ) < min_occurrences:
            continue

        pfacs = [o['pfac_corr'] for o in occ]
        classes = [o['class'] for o in occ]

        mean_pfac = float(np.mean(pfacs))
        class_count_map = Counter(classes)
        class_mean_pfac = {}
        for cls, cnt in class_count_map.items():
            vals = [p for p, c in zip(pfacs, classes) if c == cls]
            class_mean_pfac[cls] = float(np.mean(vals))

        reliable_features[fid] = {
            'mean_pfac_corr': mean_pfac,
            'occurrences': len(occ),
            'class_count_map': dict(class_count_map),
            'class_mean_pfac': class_mean_pfac,
            'raw_pfacs': pfacs
        }

    final_dict = {
        'feature_stats': reliable_features,
        'metadata': {
            'layer_idx': layer_idx,
            'patch_size': patch_size,
            'n_samples_processed': samples_processed,
            'min_occurrences': min_occurrences,
            'attribution_dir': attribution_dir,
            'threshold': threshold
        },
        'metric_definitions': {
            'pfac_corr': "Pearson correlation between feature activations and attribution map"
        }
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_dict, save_path)
        logging.info(f"Dictionary saved to {save_path}")

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
