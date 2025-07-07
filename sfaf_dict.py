from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.sae import SparseAutoencoder


def build_stealth_feature_dictionary(
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    n_samples: int = 10000,
    layer_idx: int = 9,
    s_percentile: int = 75,
    a_percentile: int = 50,
    min_logit_impact: float = 0.01,
    min_consistency_score: float = 0.3,
    min_occurrences: int = 1,
    save_path: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Build dictionary of reliable stealth features that consistently exhibit:
    - High steerability (S_f) relative to other active features
    - Low attention relevance (A_f) relative to other active features  
    - High impact on model predictions when ablated
    - Consistent behavior across multiple images
    
    Args:
        model: Hooked SAE Vision Transformer
        sae: Sparse Autoencoder 
        dataloader: Training data loader
        n_samples: Maximum number of samples to process
        layer_idx: Transformer layer to analyze
        s_percentile: Percentile threshold for S_f among active features
        a_percentile: Percentile threshold for A_f among active features (stealth = below this)
        min_logit_impact: Minimum logit change required for inclusion
        min_consistency_score: Minimum consistency score for reliability
        min_occurrences: Minimum times a feature must appear to be considered
        save_path: Optional path to save the dictionary
        
    Returns:
        Dictionary containing stealth feature characteristics and metrics
    """
    device = next(model.parameters()).device
    n_features = sae.cfg.d_sae
    n_classes = model.head.out_features

    # Track per-image stealth feature occurrences and impacts
    feature_occurrences = defaultdict(list)
    samples_processed = 0

    resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
    attn_hook_name = f"blocks.{layer_idx}.attn.hook_pattern"

    for imgs, labels in tqdm(dataloader, desc="Analyzing stealth features"):
        if samples_processed >= n_samples:
            break

        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        for i in range(batch_size):
            if samples_processed >= n_samples:
                break

            # Process single image
            image = imgs[i:i + 1]
            label = labels[i].item()

            # Get S_f, A_f, and codes for this image
            S_f, A_f, codes, original_logits = _compute_single_image_sa(
                model, sae, image, label, layer_idx, resid_hook_name, attn_hook_name
            )

            # Identify stealth features for this image
            stealth_indices = _identify_stealth_features(S_f, A_f, s_percentile, a_percentile)

            if len(stealth_indices) == 0:
                samples_processed += 1
                continue

            # Test impact of each stealth feature
            for feat_idx in stealth_indices:
                logit_impact = _measure_feature_impact(model, image, label, codes, feat_idx, original_logits)

                feature_occurrences[feat_idx.item()].append({
                    'logit_impact': logit_impact,
                    's_f': S_f[feat_idx].item(),
                    's_f_sign': torch.sign(S_f[feat_idx]).item(),
                    'a_f': A_f[feat_idx].item(),
                    'class': label
                })

            samples_processed += 1

    # Analyze consistency and build final dictionary
    reliable_features = {}
    feature_stats = {}

    for feat_id, occurrences in feature_occurrences.items():
        if len(occurrences) < min_occurrences:
            continue

        # Extract metrics
        impacts = [occ['logit_impact'] for occ in occurrences]
        s_f_values = [occ['s_f'] for occ in occurrences]
        a_f_values = [occ['a_f'] for occ in occurrences]
        classes = [occ['class'] for occ in occurrences]

        # Calculate consistency metrics
        mean_impact = np.mean(impacts)
        std_impact = np.std(impacts)
        abs_mean_impact = abs(mean_impact)

        # More nuanced consistency score
        if len(impacts) >= min_occurrences:
            # Components of consistency
            impact_magnitude = abs_mean_impact
            impact_reliability = 1.0 / (1.0 + std_impact / abs_mean_impact) if abs_mean_impact > 0 else 0
            frequency_bonus = np.log(len(impacts)) / np.log(10)  # log scale for occurrences

            # Combine factors
            consistency_score = impact_magnitude * impact_reliability * (1 + frequency_bonus)

            signs = np.sign(s_f_values)
            reliable_features[feat_id] = {
                'mean_logit_impact': mean_impact,
                'std_logit_impact': std_impact,
                'cv_logit_impact': std_impact / (abs_mean_impact + 1e-6),
                'impact_magnitude': impact_magnitude,
                'impact_reliability': impact_reliability,
                'consistency_score': consistency_score,
                'mean_s_f': np.mean(s_f_values),
                'std_s_f': np.std(s_f_values),
                'mean_a_f': np.mean(a_f_values),
                'std_a_f': np.std(a_f_values),
                'occurrences': len(occurrences),
                'classes_affected': list(set(classes)),
                'feature_type': 'constructive' if np.mean(s_f_values) > 0 else 'destructive',
                'sign_consistency': np.mean(signs == np.sign(np.mean(s_f_values)))
            }

            feature_stats[feat_id] = {
                'all_impacts': impacts,
                'all_s_f': s_f_values,
                'all_a_f': a_f_values,
                'all_classes': classes
            }

    # Convert to tensors and create final dictionary
    if reliable_features:
        feature_ids = torch.tensor(list(reliable_features.keys()), dtype=torch.long)

        # Create consolidated metrics tensors
        metrics_tensor = torch.zeros(len(feature_ids), 7)  # 7 key metrics
        for i, feat_id in enumerate(feature_ids):
            stats = reliable_features[feat_id.item()]
            metrics_tensor[i] = torch.tensor([
                stats['mean_logit_impact'], stats['cv_logit_impact'],
                stats['consistency_score'], stats['mean_s_f'], stats['mean_a_f'], stats['occurrences'],
                len(stats['classes_affected'])
            ])

        dictionary = {
            'feature_ids':
            feature_ids,
            'metrics':
            metrics_tensor,
            'metrics_names': [
                'mean_logit_impact', 'cv_logit_impact',  'consistency_score', 'mean_s_f', 'mean_a_f',
                'occurrences', 'n_classes_affected'
            ],
            'detailed_stats':
            reliable_features,
            'raw_data':
            feature_stats,
            'layer_idx':
            layer_idx,
            'n_samples_processed':
            samples_processed,
            'selection_criteria': {
                's_percentile': s_percentile,
                'a_percentile': a_percentile,
                'min_logit_impact': min_logit_impact,
                'min_consistency_score': min_consistency_score,
                'min_occurrences': min_occurrences
            }
        }
    else:
        dictionary = {
            'feature_ids': torch.tensor([], dtype=torch.long),
            'metrics': torch.tensor([]),
            'metrics_names': [],
            'detailed_stats': {},
            'raw_data': {},
            'layer_idx': layer_idx,
            'n_samples_processed': samples_processed,
            'selection_criteria': {
                's_percentile': s_percentile,
                'a_percentile': a_percentile,
                'min_logit_impact': min_logit_impact,
                'min_consistency_score': min_consistency_score,
                'min_occurrences': min_occurrences
            }
        }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dictionary, save_path)

    return dictionary


def _compute_single_image_sa(
    model: HookedSAEViT, sae: SparseAutoencoder, image: torch.Tensor, label: int, layer_idx: int, resid_hook_name: str,
    attn_hook_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute S_f and A_f for a single image."""
    device = next(model.parameters()).device
    n_classes = model.head.out_features

    resid_storage, attn_storage = {}, {}

    def save_resid_hook(tensor, hook):
        tensor.requires_grad_(True)
        resid_storage['resid'] = tensor

    def save_attn_hook(tensor, hook):
        tensor.requires_grad_(True)
        attn_storage['attn'] = tensor

    fwd_hooks = [(resid_hook_name, save_resid_hook), (attn_hook_name, save_attn_hook)]

    model.zero_grad()
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(image)

    resid, attn = resid_storage['resid'], attn_storage['attn']

    with torch.no_grad():
        _, codes = sae.encode(resid)

    # Compute gradients
    target = logits[0, label]
    resid_grad, attn_grad = torch.autograd.grad(outputs=target, inputs=[resid, attn])

    with torch.no_grad():
        # A_f calculation
        grad_weighted_attn = (attn * attn_grad.abs()).sum(dim=1)
        cls_to_patch_attn = grad_weighted_attn[0, 0, 1:]
        active_codes_mask = codes[0, 1:, :] > 0.0
        A_f = torch.einsum('t,tf->f', cls_to_patch_attn, active_codes_mask.float())

        # S_f calculation
        dir_deriv = torch.einsum('d,fd->f', resid_grad[0, 0, :], sae.W_dec)
        codes_no_cls = codes[0, 1:, :]
        S_f = (codes_no_cls * dir_deriv.unsqueeze(0)).sum(0)

    return S_f, A_f, codes, logits


def _identify_stealth_features(
    S_f: torch.Tensor, A_f: torch.Tensor, s_percentile: int, a_percentile: int
) -> torch.Tensor:
    """Identify stealth features using active-feature-relative thresholds."""
    active_mask = (S_f.abs() > 1e-6) | (A_f.abs() > 1e-6)

    if active_mask.sum() < 10:
        return torch.tensor([], dtype=torch.long)

    s_threshold = torch.quantile(S_f.abs()[active_mask], s_percentile / 100.0)
    a_threshold = torch.quantile(A_f[active_mask], a_percentile / 100.0)

    stealth_mask = active_mask & (S_f.abs() > s_threshold) & (A_f < a_threshold)
    return stealth_mask.nonzero(as_tuple=True)[0]


def _measure_feature_impact(
    model: HookedSAEViT,
    image: torch.Tensor,
    label: int,
    codes: torch.Tensor,
    feat_idx: torch.Tensor,
    original_logits: torch.Tensor,
    activation_threshold: float = 0.05
) -> float:
    """Measure logit impact of ablating a specific feature."""
    codes_patches = codes[0, 1:]  # (196, F)

    # Find active patches for this feature
    active_patches = (codes_patches[:, feat_idx] > activation_threshold).nonzero(as_tuple=True)[0]

    if len(active_patches) == 0:
        return 0.0

    # Create masked image
    masked_image = image.clone()
    patch_size = 16

    for patch_idx in active_patches:
        row = (patch_idx // 14) * patch_size
        col = (patch_idx % 14) * patch_size

        masked_image[0, :, row:row + patch_size, col:col + patch_size] = 0

    # Forward pass with mask
    with torch.no_grad():
        masked_logits = model(masked_image)
        logit_change = (original_logits - masked_logits)[0, label].item()

    return logit_change
