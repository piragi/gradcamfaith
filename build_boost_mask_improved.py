"""
Improved Boosting Strategy Based on Analysis Findings

Key insights incorporated:
1. Focus on 10-100 occurrence features (best signal-to-noise)
2. Only boost/suppress features with |bin bias| >= 1.5
3. Class-aware corrections (COVID over-attributed, Non-COVID under-attributed)
4. Balance frequency and impact in selection

Note: The field 'mean_log_ratio' now contains bin_attribution_bias values:
- Positive values = under-attributed (need boost)
- Negative values = over-attributed (need suppression)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch


def get_active_features(codes: torch.Tensor, min_activation: float = 0.05, topk_active: Optional[int] = None) -> set:
    """
    Get set of active features in current sample.
    
    Args:
        codes: SAE codes [n_patches, n_feats]
        min_activation: Minimum activation threshold
        topk_active: If specified, only consider top-k most active features
        
    Returns:
        Set of active feature indices
    """
    if topk_active is None:
        # Use all active features
        active_features = (codes > min_activation).any(dim=0).nonzero(as_tuple=True)[0]
        return set(active_features.cpu().tolist())
    else:
        # Get top-k most active features by max activation
        max_acts_per_feature = codes.max(dim=0).values
        k = min(topk_active, codes.shape[1])
        top_values, top_indices = torch.topk(max_acts_per_feature, k=k)

        # Filter by minimum activation threshold
        threshold_mask = top_values > min_activation
        active_features = top_indices[threshold_mask]
        return set(active_features.cpu().tolist())


def get_sorted_saco_features(
    features_dict: Dict[int, Dict],
    cache_attr: str,
    min_occurrences: int,
    max_occurrences: int,
    min_log_ratio: float,
    use_balanced_score: bool,
    use_abs_ratio: bool = True,
    debug: bool = True
) -> List[Tuple[int, Dict, float]]:
    """
    Get sorted features from SaCo results with caching.
    
    Args:
        features_dict: Dictionary of features from SaCo analysis
        cache_attr: Attribute name for caching (e.g., '_cached_sorted_over')
        min_occurrences: Minimum feature occurrences
        max_occurrences: Maximum feature occurrences
        min_log_ratio: Minimum bin bias threshold (now using bin_attribution_bias)
        use_balanced_score: Whether to use balanced scoring
        use_abs_ratio: Whether to use absolute value of bin bias
        debug: Whether to print debug information
        
    Returns:
        List of (feat_id, stats, score) tuples sorted by score
    """
    # Check cache first
    if hasattr(build_boost_mask_improved, cache_attr):
        cached = getattr(build_boost_mask_improved, cache_attr)
        if debug:
            print(f"  Using cached {cache_attr}: {len(cached)} features")
        return cached

    if debug:
        print(f"\n  Building sorted features for {cache_attr}:")
        print(f"    Total features in dict: {len(features_dict)}")
        print(f"    Filters: occurrences=[{min_occurrences}, {max_occurrences}], min_bias={min_log_ratio}")

    # Build and cache sorted list
    presorted = []
    filtered_reasons = {'occ_low': 0, 'occ_high': 0, 'ratio_low': 0}

    for feat_id, stats in features_dict.items():
        n_occ = stats.get('n_occurrences', 0)
        log_ratio = abs(stats.get('mean_log_ratio', 0)) if use_abs_ratio else stats.get('mean_log_ratio', 0)

        # Apply filters with tracking
        if n_occ < min_occurrences:
            filtered_reasons['occ_low'] += 1
            continue
        if n_occ > max_occurrences:
            filtered_reasons['occ_high'] += 1
            continue
        if log_ratio < min_log_ratio:
            filtered_reasons['ratio_low'] += 1
            continue

        # Compute score
        if use_balanced_score:
            score = log_ratio * math.log(n_occ + 1)
        else:
            score = log_ratio

        presorted.append((feat_id, stats, score))

    # Sort by score
    presorted.sort(key=lambda x: x[2], reverse=True)

    if debug:
        print(f"    Filtered out: {filtered_reasons}")
        print(f"    Remaining features: {len(presorted)}")
        if presorted:
            print(f"    Top 3 features:")
            for i, (fid, stats, score) in enumerate(presorted[:3]):
                print(
                    f"      {i+1}. Feature {fid}: bias={stats.get('mean_log_ratio', 0):.3f}, "
                    f"occ={stats.get('n_occurrences', 0)}, score={score:.3f}"
                )

    # Cache for future use
    setattr(build_boost_mask_improved, cache_attr, presorted)

    return presorted


def apply_modulation(
    codes: torch.Tensor,
    boost_mask: torch.Tensor,
    features: List[Tuple[int, Dict, float]],
    mode: str,
    strength: float,
    min_activation: float,
    debug: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Apply boost or suppress modulation to features with CONSTANT strength.
    
    Args:
        codes: SAE codes [n_patches, n_feats]
        boost_mask: Current boost mask to modify
        features: List of (feat_id, stats, score) tuples
        mode: 'boost' or 'suppress'
        strength: Constant strength for modulation (no adaptation)
        min_activation: Minimum activation threshold
        debug: Whether to print debug info
        
    Returns:
        Updated boost_mask and list of selected feature IDs
    """
    selected_features = []

    for feat_id, stats, score in features:
        feat_activations = codes[:, feat_id]
        active_mask = feat_activations > min_activation

        # Use CONSTANT strength - no adaptation based on bias magnitude
        if mode == 'boost':
            # Apply constant boost to patches where feature is active
            # For active patches, multiply by strength directly
            patch_modulation = torch.ones_like(feat_activations)
            patch_modulation[active_mask] = strength
        else:  # suppress
            # Apply constant suppression to patches where feature is active
            # For active patches, divide by strength (or multiply by 1/strength)
            patch_modulation = torch.ones_like(feat_activations)
            patch_modulation[active_mask] = 1.0 / strength if strength > 0 else 0.1

        boost_mask *= patch_modulation
        selected_features.append(feat_id)

        if debug:
            n_active = active_mask.sum().item()
            actual_strength = strength if mode == 'boost' else (1.0 / strength if strength > 0 else 0.1)
            print(
                f"  {mode.upper()} {feat_id}: ratio={stats.get('mean_log_ratio', 0):.4f}, "
                f"occ={stats.get('n_occurrences', 0)}, class={stats.get('dominant_class', 'unknown')}, "
                f"constant_strength={actual_strength:.2f}, patches={n_active}"
            )

    return boost_mask, selected_features


def select_topk_activation_features(codes: torch.Tensor, max_boost: int,
                                    min_activation: float) -> List[Tuple[int, Dict, float]]:
    """
    Select features based on top-k activation magnitude.
    
    Args:
        codes: SAE codes [n_patches, n_feats]
        max_boost: Number of features to select
        min_activation: Minimum activation threshold
        
    Returns:
        List of (feat_id, stats_dict, score) tuples
    """
    # Compute maximum activation per feature
    max_acts_per_feature = codes.max(dim=0).values

    # Get top-k most active features
    k = min(max_boost, codes.shape[1])
    top_values, top_indices = torch.topk(max_acts_per_feature, k=k)

    # Filter by minimum activation threshold
    threshold_mask = top_values > min_activation

    features = []
    for i in range(len(top_indices)):
        if threshold_mask[i]:
            feat_id = top_indices[i].item()
            max_act = top_values[i].item()
            # Create a simple stats dict for compatibility
            stats = {'max_activation': max_act, 'mean_log_ratio': max_act}  # Use activation as pseudo-ratio
            features.append((feat_id, stats, max_act))

    return features


def select_random_features(
    codes: torch.Tensor,
    max_suppress: int,
    max_boost: int,
    min_activation: float,
    seed: Optional[int] = None
) -> Tuple[List[Tuple[int, Dict, float]], List[Tuple[int, Dict, float]]]:
    """
    Randomly select active features for boosting/suppression.
    
    Args:
        codes: SAE codes [n_patches, n_feats]
        max_suppress: Number of features to suppress
        max_boost: Number of features to boost
        min_activation: Minimum activation threshold
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (suppress_features, boost_features)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Find all active features in this sample
    active_mask = (codes > min_activation).any(dim=0)
    active_features = active_mask.nonzero(as_tuple=True)[0]

    if len(active_features) == 0:
        return [], []

    # Randomly sample features
    n_total = min(max_suppress + max_boost, len(active_features))

    if n_total > 0:
        # Random permutation
        perm = torch.randperm(len(active_features))[:n_total]
        selected_indices = active_features[perm]

        # Split into suppress and boost
        suppress_features = []
        boost_features = []

        for i, feat_id in enumerate(selected_indices):
            feat_id_int = feat_id.item()
            # Get max activation for this feature
            max_act = codes[:, feat_id].max().item()

            # Create stats dict for compatibility
            stats = {'max_activation': max_act, 'mean_log_ratio': max_act}

            if i < max_suppress:
                suppress_features.append((feat_id_int, stats, max_act))
            else:
                boost_features.append((feat_id_int, stats, max_act))

        return suppress_features, boost_features

    return [], []


def select_class_aware_features(
    codes: torch.Tensor,
    inference_dict: Dict[str, Any],
    predicted_class: str,
    top_L_per_patch: int = 5,
    strength_k: float = 3.0,
    debug: bool = True
) -> Dict[int, Dict[str, float]]:
    """
    Select features using class-aware robust statistics.
    
    Args:
        codes: SAE codes [n_patches, n_feats]
        inference_dict: Robust inference dictionary with per-class features
        predicted_class: The predicted class name
        top_L_per_patch: Number of top features to consider per patch
        strength_k: Exponential strength multiplier
        debug: Whether to print debug info
        
    Returns:
        Dictionary mapping feature_id to correction info
    """
    if debug:
        print(f"\n=== CLASS-AWARE FEATURE SELECTION ===")
        print(f"Predicted class: {predicted_class}")

    # Get class-specific features
    class_features = inference_dict.get('by_class', {}).get(predicted_class, {})

    if not class_features:
        if debug:
            print(f"No reliable features for class {predicted_class}")
        return {}

    if debug:
        print(f"Available reliable features for {predicted_class}: {len(class_features)}")

    # Get config
    config = inference_dict.get('config', {})
    top_L = min(top_L_per_patch, config.get('top_L', 5))
    k = strength_k if strength_k is not None else config.get('strength_k', 3.0)

    # Collect features that are active in this sample
    n_patches = codes.shape[0]
    selected_features = {}

    for patch_idx in range(n_patches):
        patch_acts = codes[patch_idx]  # [n_feats]

        # Find top-L activated features in this patch
        active_mask = patch_acts > 0
        if not active_mask.any():
            continue

        # Get top-L features by activation
        top_k = min(top_L, active_mask.sum().item())
        if top_k == 0:
            continue

        top_values, top_indices = torch.topk(patch_acts, k=top_k)

        # Check which are in our reliable set
        for i in range(top_k):
            feat_id = top_indices[i].item()
            activation = top_values[i].item()

            if feat_id in class_features:
                feat_info = class_features[feat_id]

                if feat_id not in selected_features:
                    selected_features[feat_id] = {
                        'bias': feat_info['bias'],
                        'raw_bias': feat_info.get('raw_bias', feat_info['bias']),
                        'n_occurrences': feat_info.get('n', 0),
                        'consistency': feat_info.get('consistency', 1.0),
                        'patches': [],
                        'max_activation': activation
                    }

                # Track patch activation
                selected_features[feat_id]['patches'].append(patch_idx)
                selected_features[feat_id]['max_activation'] = max(
                    selected_features[feat_id]['max_activation'], activation
                )

    if debug:
        print(f"Selected {len(selected_features)} reliable features active in this sample")
        if selected_features:
            # Show top features by bias magnitude
            sorted_feats = sorted(selected_features.items(), key=lambda x: abs(x[1]['bias']), reverse=True)
            print(f"\nTop features by |bias|:")
            for feat_id, info in sorted_feats[:5]:
                print(
                    f"  Feature {feat_id}: bias={info['bias']:.3f}, "
                    f"n_patches={len(info['patches'])}, n_occ={info['n_occurrences']}"
                )

    return selected_features


def select_saco_features(
    codes: torch.Tensor,
    saco_results: Dict[str, Any],
    active_set: set,
    max_suppress: int,
    max_boost: int,
    min_occurrences: int,
    max_occurrences: int,
    min_log_ratio: float,
    use_balanced_score: bool,
    suppress_priority: float = 1.0,
    boost_priority: float = 1.0,
    debug: bool = True
) -> Tuple[List[Tuple[int, Dict, float]], List[Tuple[int, Dict, float]]]:
    """
    Select features based on SaCo analysis.
    
    Returns:
        Tuple of (suppress_features, boost_features)
    """
    n_feats = codes.shape[1]
    results_by_type = saco_results.get('results_by_type', {})

    if debug:
        print(f"\n=== SELECT SACO FEATURES DEBUG ===")
        print(f"Active features in current sample: {len(active_set)}")
        print(f"Max suppress: {max_suppress}, Max boost: {max_boost}")
        print(f"SAE feature dimension: {n_feats}")

    suppress_features = []
    boost_features = []

    # Process over-attributed features (suppress)
    if max_suppress > 0:
        over_attributed = results_by_type.get('over_attributed', {})
        if debug:
            print(f"\nProcessing OVER-ATTRIBUTED features for suppression:")
            print(f"  Total over-attributed features available: {len(over_attributed)}")

        if over_attributed:
            sorted_features = get_sorted_saco_features(
                over_attributed,
                '_cached_sorted_over',
                min_occurrences,
                max_occurrences,
                min_log_ratio,
                use_balanced_score,
                use_abs_ratio=True,
                debug=debug
            )

            # Track filtering
            not_active = 0
            out_of_bounds = 0

            # Filter by active features
            for feat_id, stats, score in sorted_features:
                if feat_id >= n_feats:
                    out_of_bounds += 1
                    continue
                if feat_id in active_set:
                    suppress_features.append((feat_id, stats, score * suppress_priority))
                    if len(suppress_features) >= max_suppress:
                        break
                else:
                    not_active += 1

            if debug:
                print(f"  Filter results: {not_active} not active, {out_of_bounds} out of bounds")
                print(f"  Selected {len(suppress_features)} features for suppression")

    # Process under-attributed features (boost)
    if max_boost > 0:
        under_attributed = results_by_type.get('under_attributed', {})
        if debug:
            print(f"\nProcessing UNDER-ATTRIBUTED features for boosting:")
            print(f"  Total under-attributed features available: {len(under_attributed)}")

        if under_attributed:
            sorted_features = get_sorted_saco_features(
                under_attributed,
                '_cached_sorted_under',
                min_occurrences,
                max_occurrences,
                min_log_ratio,
                use_balanced_score,
                use_abs_ratio=False,
                debug=debug
            )

            # Track filtering
            not_active = 0
            out_of_bounds = 0

            # Filter by active features
            for feat_id, stats, score in sorted_features:
                if feat_id >= n_feats:
                    out_of_bounds += 1
                    continue
                if feat_id in active_set:
                    boost_features.append((feat_id, stats, score * boost_priority))
                    if len(boost_features) >= max_boost:
                        break
                else:
                    not_active += 1

            if debug:
                print(f"  Filter results: {not_active} not active, {out_of_bounds} out of bounds")
                print(f"  Selected {len(boost_features)} features for boosting")

    if debug:
        print(f"\nFINAL: {len(suppress_features)} suppress, {len(boost_features)} boost")

    return suppress_features, boost_features


def precache_sorted_features(
    saco_results: Dict[str, Any],
    min_occurrences: int = 10,
    max_occurrences: int = 100000,
    min_log_ratio: float = 0.,
    use_balanced_score: bool = True
):
    """
    Pre-cache sorted feature lists for faster runtime performance.
    Call this once when loading the saco_results dictionary.
    """
    results_by_type = saco_results.get('results_by_type', {})

    # Clear any existing cache
    if hasattr(build_boost_mask_improved, '_cached_sorted_over'):
        delattr(build_boost_mask_improved, '_cached_sorted_over')
    if hasattr(build_boost_mask_improved, '_cached_sorted_under'):
        delattr(build_boost_mask_improved, '_cached_sorted_under')

    # Pre-cache over-attributed features
    over_attributed = results_by_type.get('over_attributed', {})
    if over_attributed:
        sorted_over = get_sorted_saco_features(
            over_attributed,
            '_cached_sorted_over',
            min_occurrences,
            max_occurrences,
            min_log_ratio,
            use_balanced_score,
            use_abs_ratio=True
        )
        print(f"Pre-cached {len(sorted_over)} over-attributed features")

    # Pre-cache under-attributed features
    under_attributed = results_by_type.get('under_attributed', {})
    if under_attributed:
        sorted_under = get_sorted_saco_features(
            under_attributed,
            '_cached_sorted_under',
            min_occurrences,
            max_occurrences,
            min_log_ratio,
            use_balanced_score,
            use_abs_ratio=False
        )
        print(f"Pre-cached {len(sorted_under)} under-attributed features")


def precache_bias_multiplicative_features(
    saco_results: Dict[str, Any], min_occurrences: int = 10, max_occurrences: int = 100, min_abs_bias: float = 0.0
):
    """
    Pre-cache sorted features for bias multiplicative correction.
    Call this once when loading the saco_results dictionary.
    """
    results_by_type = saco_results.get('results_by_type', {})
    all_features = {**results_by_type.get('under_attributed', {}), **results_by_type.get('over_attributed', {})}

    # Pre-filter and sort features
    sorted_features = []
    for feat_id, stats in all_features.items():
        n_occ = stats.get('n_occurrences', 0)
        if n_occ < min_occurrences or n_occ > max_occurrences:
            continue

        bin_bias = abs(stats.get('mean_log_ratio', 0))
        if bin_bias < min_abs_bias:
            continue

        sorted_features.append((feat_id, stats, bin_bias))

    # Sort by bias magnitude (descending)
    sorted_features.sort(key=lambda x: x[2], reverse=True)

    # Cache with the standard parameters
    cache_key = f'_cached_bias_mult_{min_occurrences}_{max_occurrences}_{min_abs_bias}'
    setattr(build_bias_multiplicative_mask, cache_key, sorted_features)

    print(f"Pre-cached {len(sorted_features)} features for bias multiplicative correction")
    if sorted_features:
        top_biases = [f[2] for f in sorted_features[:10]]
        print(f"  Top 10 bias magnitudes: {[f'{b:.4f}' for b in top_biases]}")


@torch.no_grad()
def build_class_aware_mask(
    codes: torch.Tensor,
    inference_dict: Dict[str, Any],
    predicted_class: str,
    top_L_per_patch: int = 5,
    strength_k: float = 3.0,
    clamp_min: float = 0.3,
    clamp_max: float = 3.0,
    debug: bool = False
) -> torch.Tensor:
    """
    Build multiplicative mask using class-aware robust features.
    
    Args:
        codes: SAE codes [n_patches, n_feats]
        inference_dict: Robust inference dictionary
        predicted_class: Predicted class name
        top_L_per_patch: Top features per patch
        strength_k: Exponential strength
        clamp_min/max: Clamp range for patch multipliers
        debug: Whether to print debug info
        
    Returns:
        Multiplicative mask [n_patches]
    """
    device = codes.device
    n_patches = codes.shape[0]

    # Get class-specific reliable features
    class_features = inference_dict.get('by_class', {}).get(predicted_class, {})

    if not class_features:
        if debug:
            print(f"No reliable features for class {predicted_class}, returning neutral mask")
        return torch.ones(n_patches, device=device)

    # Get config
    config = inference_dict.get('config', {})
    top_L = min(top_L_per_patch, config.get('top_L', 5))
    k = strength_k if strength_k is not None else config.get('strength_k', 3.0)
    clamp_min = clamp_min if clamp_min is not None else config.get('clamp_min', 0.3)
    clamp_max = clamp_max if clamp_max is not None else config.get('clamp_max', 3.0)

    # Initialize mask
    patch_multipliers = torch.ones(n_patches, device=device)

    # Process each patch
    for patch_idx in range(n_patches):
        patch_acts = codes[patch_idx]  # [n_feats]

        # Find top-L features in this patch
        active_mask = patch_acts > 0
        if not active_mask.any():
            continue

        top_k = min(top_L, active_mask.sum().item())
        if top_k == 0:
            continue

        top_values, top_indices = torch.topk(patch_acts, k=top_k)

        # Normalize activations within patch
        if top_k > 1:
            # Z-score normalization -> sigmoid to [0,1]
            act_mean = top_values.mean()
            act_std = top_values.std() + 1e-8
            norm_acts = torch.sigmoid((top_values - act_mean) / act_std)
        else:
            norm_acts = torch.ones_like(top_values)

        # Compute multiplier contributions
        multipliers = []
        for i in range(top_k):
            feat_id = top_indices[i].item()

            if feat_id in class_features:
                feat_info = class_features[feat_id]
                bias = feat_info['bias']
                activation = norm_acts[i].item()

                # Exponential correction: exp(k * bias * activation)
                m = math.exp(k * bias * activation)
                multipliers.append(m)

        if multipliers:
            # Geometric mean of contributions
            import numpy as np
            patch_mult = np.exp(np.mean(np.log(multipliers)))
            # Clamp to range
            patch_mult = max(clamp_min, min(clamp_max, patch_mult))
            patch_multipliers[patch_idx] = patch_mult

    if debug:
        print(f"\nMask statistics:")
        print(f"  Mean: {patch_multipliers.mean():.3f}")
        print(f"  Std: {patch_multipliers.std():.3f}")
        print(f"  Min: {patch_multipliers.min():.3f}")
        print(f"  Max: {patch_multipliers.max():.3f}")
        print(f"  Patches boosted (>1.1): {(patch_multipliers > 1.1).sum().item()}")
        print(f"  Patches suppressed (<0.9): {(patch_multipliers < 0.9).sum().item()}")

    return patch_multipliers


def build_boost_mask_improved(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    predicted_class: int,
    idx_to_class: Dict[int, str],
    device: torch.device,
    *,
    # Frequency filtering
    min_occurrences: int = 1,
    max_occurrences: int = 10000000,

    # Ratio thresholds
    min_log_ratio: float = 0.,

    # Class-specific behavior
    class_aware: bool = False,

    # Strength parameters
    suppress_strength: float = 0.2,
    boost_strength: float = 2.5,

    # Selection limits
    max_suppress: int = 0,
    max_boost: int = 10,

    # Activation threshold
    min_activation: float = 0.1,

    # Top-k filtering for active features
    topk_active: Optional[int] = None,

    # Weighting strategy
    use_balanced_score: bool = True,

    # Selection method
    selection_method: str = 'saco',  # 'saco', 'topk_activation', or 'random'

    # Random seed (for random selection)
    random_seed: Optional[int] = 42,
    debug: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Improved boosting strategy based on comprehensive analysis.
    
    Args:
        sae_codes: SAE feature codes [1+T, k]
        saco_results: SaCo analysis results (required even for topk method)
        predicted_class: Predicted class index
        idx_to_class: Class index to name mapping
        device: Device to run on
        selection_method: 'saco' for SaCo-based, 'topk_activation' for top-k baseline, 'random' for random baseline
        random_seed: Random seed for reproducible random selection
        
    Returns:
        boost_mask: Multiplicative mask for patches
        selected_features: List of selected feature IDs
    """
    codes = sae_codes[0, 1:].to(device)  # Remove CLS token
    n_patches, n_feats = codes.shape
    boost_mask = torch.ones(n_patches, device=device)

    predicted_class_name = idx_to_class.get(predicted_class, f'class_{predicted_class}')

    # Use new class-aware method if inference dict is provided
    if use_class_aware_v2 and inference_dict is not None:
        if debug:
            print(f"\n=== USING CLASS-AWARE V2 WITH ROBUST FEATURES ===")
            print(f"Predicted class: {predicted_class_name}")

        # Build mask using robust class-aware features
        mask = build_class_aware_mask(
            codes,
            inference_dict,
            predicted_class_name,
            top_L_per_patch=5,
            strength_k=3.0,
            clamp_min=0.3,
            clamp_max=3.0,
            debug=debug
        )
        return mask

    if debug:
        print(f"\n=== Boosting for {predicted_class_name} (method: {selection_method}) ===")

    # Get active features in current sample
    active_set = get_active_features(codes, min_activation, topk_active)

    # Class-aware adjustments (optional)
    suppress_priority = 1.0
    boost_priority = 1.0
    actual_max_suppress = max_suppress
    actual_max_boost = max_boost

    # Select features based on method
    if selection_method == 'topk_activation':
        # Top-k activation baseline
        boost_features = select_topk_activation_features(codes, actual_max_boost, min_activation)
        suppress_features = []

        if debug:
            print(f"Top-k activation: Selected {len(boost_features)} features to boost")

    elif selection_method == 'random':
        # Random baseline
        suppress_features, boost_features = select_random_features(
            codes, actual_max_suppress, actual_max_boost, min_activation, random_seed
        )

        if debug:
            print(f"Random selection: {len(suppress_features)} suppress, {len(boost_features)} boost")

    else:  # Default to 'saco'
        # SaCo-based selection
        suppress_features, boost_features = select_saco_features(
            codes, saco_results, active_set, actual_max_suppress, actual_max_boost, min_occurrences, max_occurrences,
            min_log_ratio, use_balanced_score, suppress_priority, boost_priority
        )

    # Apply suppression
    selected_features = []
    if suppress_features:
        boost_mask, suppress_ids = apply_modulation(
            codes, boost_mask, suppress_features, 'suppress', suppress_strength, min_activation, debug
        )
        selected_features.extend(suppress_ids)

    # Apply boosting
    if boost_features:
        boost_mask, boost_ids = apply_modulation(
            codes, boost_mask, boost_features, 'boost', boost_strength, min_activation, debug
        )
        selected_features.extend(boost_ids)

    if debug:
        total_active = (codes > min_activation).any(dim=0).sum().item()
        print(
            f"Mask summary: {len(selected_features)} features selected "
            f"(suppress={len(suppress_features)}, boost={len(boost_features)}) "
            f"from {len(active_set)} active out of {total_active} total"
        )

    return boost_mask, selected_features


@torch.no_grad()
def build_additive_correction_mask(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    predicted_class: int,
    idx_to_class: Dict[int, str],
    device: torch.device,
    *,
    # Activation threshold
    min_activation: float = 0.1,

    # Scaling factor for corrections
    scaling_factor: float = 1.0,

    # Frequency filtering
    min_occurrences: int = 1,
    max_occurrences: int = 10000000,

    # Minimum absolute bias to consider
    min_abs_bias: float = 0.0,

    # Top-k filtering for active features
    topk_active: Optional[int] = None,

    # Aggregation method when multiple features active at same patch
    aggregation: str = 'weighted_mean',  # 'weighted_mean', 'sum', 'max'
    debug: bool = False
) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
    """
    Build an additive correction mask based on bin bias values.
    
    This function directly uses the bin_attribution_bias as an additive correction,
    since it represents how much the attribution should be adjusted when a feature is active.
    
    Args:
        sae_codes: SAE feature codes [1+T, k]
        saco_results: SaCo analysis results with bin biases
        predicted_class: Predicted class index
        idx_to_class: Class index to name mapping
        device: Device to run on
        min_activation: Minimum activation threshold to consider feature active
        scaling_factor: Scale the correction strength (1.0 = use bias directly)
        min_occurrences: Minimum feature occurrences in dataset
        max_occurrences: Maximum feature occurrences in dataset
        min_abs_bias: Minimum absolute bias value to apply correction
        topk_active: If specified, only consider top-k most active features
        aggregation: How to combine corrections when multiple features are active
        debug: Whether to print debug information
        
    Returns:
        correction_mask: Additive corrections for each patch
        selected_features: List of feature IDs that contributed to corrections
        debug_info: Dictionary with debug information
    """
    codes = sae_codes[0, 1:].to(device)  # Remove CLS token
    n_patches, n_feats = codes.shape

    predicted_class_name = idx_to_class.get(predicted_class, f'class_{predicted_class}')

    if debug:
        print(f"\n=== ADDITIVE Correction for {predicted_class_name} ===")
        print(f"Patches: {n_patches}, Features: {n_feats}")
        print(f"Scaling factor: {scaling_factor}, Aggregation: {aggregation}")

    # Initialize correction mask with zeros (no correction)
    correction_mask = torch.zeros(n_patches, device=device)

    # Get active features in current sample
    active_set = get_active_features(codes, min_activation, topk_active)

    if debug:
        print(f"Active features in sample: {len(active_set)}")

    # Combine all features (both under and over-attributed)
    results_by_type = saco_results.get('results_by_type', {})
    all_features = {**results_by_type.get('under_attributed', {}), **results_by_type.get('over_attributed', {})}

    # Track corrections per patch for aggregation
    patch_corrections = [[] for _ in range(n_patches)]
    patch_weights = [[] for _ in range(n_patches)]
    selected_features = []

    # Statistics for debugging
    features_used = 0
    features_filtered = {'not_active': 0, 'out_of_bounds': 0, 'low_occ': 0, 'high_occ': 0, 'low_bias': 0}

    for feat_id, stats in all_features.items():
        # Check if feature index is valid
        if feat_id >= n_feats:
            features_filtered['out_of_bounds'] += 1
            continue

        # Check if feature is active in this sample
        if feat_id not in active_set:
            features_filtered['not_active'] += 1
            continue

        # Apply occurrence filters
        n_occ = stats.get('n_occurrences', 0)
        if n_occ < min_occurrences:
            features_filtered['low_occ'] += 1
            continue
        if n_occ > max_occurrences:
            features_filtered['high_occ'] += 1
            continue

        # Get bin bias (this is the attribution error)
        bin_bias = stats.get('mean_log_ratio', 0)

        # Apply minimum bias threshold
        if abs(bin_bias) < min_abs_bias:
            features_filtered['low_bias'] += 1
            continue

        # Get feature activations for each patch
        feat_activations = codes[:, feat_id]
        active_mask = feat_activations > min_activation

        if not active_mask.any():
            continue

        features_used += 1
        selected_features.append(feat_id)

        # Store corrections and weights for each active patch
        for patch_idx in active_mask.nonzero(as_tuple=True)[0]:
            activation_strength = feat_activations[patch_idx].item()
            # The correction is the bin bias scaled by the scaling factor
            correction = bin_bias * scaling_factor
            patch_corrections[patch_idx].append(correction)
            patch_weights[patch_idx].append(activation_strength)

        if debug and features_used <= 5:  # Show first 5 features
            n_active_patches = active_mask.sum().item()
            print(
                f"  Feature {feat_id}: bias={bin_bias:.3f}, "
                f"occ={n_occ}, patches={n_active_patches}, "
                f"class={stats.get('dominant_class', 'unknown')}"
            )

    # Aggregate corrections per patch
    patches_corrected = 0
    max_correction = 0
    min_correction = 0

    for patch_idx in range(n_patches):
        if not patch_corrections[patch_idx]:
            continue

        patches_corrected += 1
        corrections = torch.tensor(patch_corrections[patch_idx], device=device)
        weights = torch.tensor(patch_weights[patch_idx], device=device)

        if aggregation == 'weighted_mean':
            # Weight by activation strength
            weights_norm = weights / weights.sum()
            patch_correction = (corrections * weights_norm).sum()
        elif aggregation == 'sum':
            # Simple sum of all corrections
            patch_correction = corrections.sum()
        elif aggregation == 'max':
            # Use the correction with maximum absolute value
            idx = corrections.abs().argmax()
            patch_correction = corrections[idx]
        else:
            # Default to mean
            patch_correction = corrections.mean()

        correction_mask[patch_idx] = patch_correction
        max_correction = max(max_correction, patch_correction.item())
        min_correction = min(min_correction, patch_correction.item())

    if debug:
        print(f"\nCorrection Summary:")
        print(f"  Features used: {features_used}/{len(all_features)}")
        print(f"  Features filtered: {features_filtered}")
        print(f"  Patches corrected: {patches_corrected}/{n_patches}")
        print(f"  Correction range: [{min_correction:.3f}, {max_correction:.3f}]")
        print(f"  Mean abs correction: {correction_mask.abs().mean().item():.3f}")

    # Prepare debug info
    debug_info = {
        'features_used': features_used,
        'features_filtered': features_filtered,
        'patches_corrected': patches_corrected,
        'correction_range': (min_correction, max_correction),
        'mean_abs_correction': correction_mask.abs().mean().item()
    }

    return correction_mask, selected_features, debug_info


@torch.no_grad()
def build_bias_multiplicative_mask(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    device: torch.device,
    *,
    # Activation threshold
    min_activation: float = 0.1,

    # Correction method
    correction_method: str = 'bounded',  # 'direct', 'bounded', 'sigmoid', 'clamped'

    # Scale factor for bias
    scale_factor: float = 1.0,

    # Minimum absolute bias to apply correction
    min_abs_bias: float = 0.0,

    # Frequency filtering
    min_occurrences: int = 1,
    max_occurrences: int = 10000000,

    # Top-k filtering for active features
    topk_active: Optional[int] = None,

    # Aggregation method for multiple features
    aggregation: str = 'geometric_mean',  # 'geometric_mean', 'arithmetic_mean', 'max'
    debug: bool = False
) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
    """
    Build multiplicative mask using (1 + bin_bias) formula with proper handling.
    
    This function uses the bin_attribution_bias as a multiplicative correction factor,
    interpreting it as the proportional adjustment needed for proper attribution.
    
    Args:
        sae_codes: SAE feature codes [1+T, k]
        saco_results: SaCo analysis results with bin biases
        device: Device to run on
        min_activation: Minimum activation threshold to consider feature active
        correction_method: How to convert bias to multiplier:
            - 'direct': multiplier = 1 + bias (clamped at 0.1)
            - 'bounded': multiplier = exp(bias) (always positive)
            - 'sigmoid': multiplier = 0.5 + 1.5*sigmoid(bias) (range [0.5, 2])
            - 'clamped': multiplier = 1 + bias, clamped to [0.2, 3]
        scale_factor: Scale the bias before applying (for fine-tuning strength)
        min_abs_bias: Minimum absolute bias value to apply correction
        min_occurrences: Minimum feature occurrences in dataset
        max_occurrences: Maximum feature occurrences in dataset
        topk_active: If specified, only consider top-k most active features
        aggregation: How to combine multiple feature corrections
        debug: Whether to print debug information
        
    Returns:
        multiplier_mask: Multiplicative corrections for each patch
        selected_features: List of feature IDs that contributed to corrections
        debug_info: Dictionary with debug information
    """
    codes = sae_codes[0, 1:].to(device)  # Remove CLS token
    n_patches, n_feats = codes.shape

    if debug:
        print(f"\n=== BIAS MULTIPLICATIVE Correction ===")
        print(f"Patches: {n_patches}, Features: {n_feats}")
        print(f"Method: {correction_method}, Scale: {scale_factor}, Aggregation: {aggregation}")

    # Initialize with ones (no correction)
    multiplier_mask = torch.ones(n_patches, device=device)

    # Get active features in current sample
    active_set = get_active_features(codes, min_activation, topk_active)

    if debug:
        print(f"Active features in sample: {len(active_set)}")

    # Combine all features (both under and over-attributed)
    results_by_type = saco_results.get('results_by_type', {})
    all_features = {**results_by_type.get('under_attributed', {}), **results_by_type.get('over_attributed', {})}

    # OPTIMIZATION: Pre-filter and sort features once
    # Check if we have a cached sorted list
    cache_key = f'_cached_bias_mult_{min_occurrences}_{max_occurrences}_{min_abs_bias}'
    if hasattr(build_bias_multiplicative_mask, cache_key):
        sorted_features = getattr(build_bias_multiplicative_mask, cache_key)
        if debug:
            print(f"  Using cached sorted features: {len(sorted_features)} features")
    else:
        # Pre-filter features by occurrences and bias
        sorted_features = []
        for feat_id, stats in all_features.items():
            n_occ = stats.get('n_occurrences', 0)
            if n_occ < min_occurrences or n_occ > max_occurrences:
                continue

            bin_bias = abs(stats.get('mean_log_ratio', 0))
            if bin_bias < min_abs_bias:
                continue

            sorted_features.append((feat_id, stats, bin_bias))

        # Sort by bias magnitude (descending)
        sorted_features.sort(key=lambda x: x[2], reverse=True)

        # Cache for future use
        setattr(build_bias_multiplicative_mask, cache_key, sorted_features)

        if debug:
            print(f"  Pre-filtered to {len(sorted_features)} features from {len(all_features)}")

    # Track multipliers per patch for aggregation
    patch_multipliers = [[] for _ in range(n_patches)]
    patch_weights = [[] for _ in range(n_patches)]
    patch_biases = [[] for _ in range(n_patches)]  # Store original biases for max_impact
    selected_features = []

    # Statistics for debugging
    features_used = 0
    features_filtered = {'not_active': 0, 'out_of_bounds': 0}

    if debug:
        print(f"  Sorted features available: {len(sorted_features)}")
        print(f"  Active features in image: {len(active_set)}")
        # Find overlap
        overlap = sum(1 for feat_id, _, _ in sorted_features if feat_id in active_set and feat_id < n_feats)
        print(f"  Overlap (features both in dict AND active): {overlap}")

    # Limit to top N features by bias if too many
    max_features_to_check = 512  # Don't check more than 100 features for speed
    features_to_process = sorted_features[:max_features_to_check] if len(
        sorted_features
    ) > max_features_to_check else sorted_features

    for feat_id, stats, _ in features_to_process:
        # Check if feature index is valid
        if feat_id >= n_feats:
            features_filtered['out_of_bounds'] += 1
            continue

        # Check if feature is active in this sample
        if feat_id not in active_set:
            features_filtered['not_active'] += 1
            continue

        # Get bin bias and scale it (already pre-filtered by occurrences and min_bias)
        bin_bias = stats.get('mean_log_ratio', 0) * scale_factor

        # Get feature activations for each patch
        feat_activations = codes[:, feat_id]
        active_mask = feat_activations > min_activation

        if not active_mask.any():
            continue

        features_used += 1
        selected_features.append(feat_id)

        # Calculate multiplier based on method
        if correction_method == 'direct':
            # Direct but with safety floor
            multiplier = max(0.1, 1 + bin_bias)
        elif correction_method == 'bounded':
            # Exponential: always positive, symmetric
            multiplier = torch.exp(torch.tensor(bin_bias, device=device))
        elif correction_method == 'sigmoid':
            # Sigmoid bounded to [0.5, 2]
            multiplier = 0.5 + 1.5 * torch.sigmoid(torch.tensor(bin_bias, device=device))
        elif correction_method == 'clamped':
            # Direct with reasonable bounds
            multiplier = torch.clamp(torch.tensor(1 + bin_bias, device=device), 0.2, 3.0)
        else:
            multiplier = 1.0

        multiplier = multiplier.item() if torch.is_tensor(multiplier) else multiplier

        # Store multipliers, weights, and biases for each active patch
        for patch_idx in active_mask.nonzero(as_tuple=True)[0]:
            activation_strength = feat_activations[patch_idx].item()
            patch_multipliers[patch_idx].append(multiplier)
            patch_weights[patch_idx].append(activation_strength)
            patch_biases[patch_idx].append(bin_bias)  # Store original scaled bias

        if debug and features_used <= 5:  # Show first 5 features
            n_active_patches = active_mask.sum().item()
            print(
                f"  Feature {feat_id}: bias={bin_bias/scale_factor:.3f}, "
                f"multiplier={multiplier:.3f}, occ={n_occ}, patches={n_active_patches}"
            )

        # Debug: Track bias distribution
        if features_used == 1:
            all_biases = [
                abs(s.get('mean_log_ratio', 0)) for s in all_features.values()
                if s.get('n_occurrences', 0) >= min_occurrences
            ]
            if all_biases:
                print(f"\n  DEBUG - Bias distribution of {len(all_biases)} features:")
                print(
                    f"    Max: {max(all_biases):.4f}, Min: {min(all_biases):.4f}, "
                    f"Mean: {sum(all_biases)/len(all_biases):.4f}"
                )
                print(f"    Features with |bias| > 0.5: {sum(1 for b in all_biases if b > 0.5)}")
                print(f"    Features with |bias| > 0.1: {sum(1 for b in all_biases if b > 0.1)}")
                print(f"    Features with |bias| > 0.01: {sum(1 for b in all_biases if b > 0.01)}")

    # Aggregate multipliers per patch
    patches_corrected = 0
    max_multiplier = 1.0
    min_multiplier = 1.0

    for patch_idx in range(n_patches):
        if not patch_multipliers[patch_idx]:
            continue

        patches_corrected += 1
        multipliers = torch.tensor(patch_multipliers[patch_idx], device=device)
        weights = torch.tensor(patch_weights[patch_idx], device=device)

        if aggregation == 'geometric_mean':
            # Weighted geometric mean (appropriate for multiplicative corrections)
            weights_norm = weights / weights.sum()
            # Use log-space for numerical stability
            log_multipliers = torch.log(multipliers)
            patch_multiplier = torch.exp((log_multipliers * weights_norm).sum())
        elif aggregation == 'arithmetic_mean':
            # Weighted arithmetic mean
            weights_norm = weights / weights.sum()
            patch_multiplier = (multipliers * weights_norm).sum()
        elif aggregation == 'max':
            # Use the strongest correction (furthest from 1.0)
            idx = (multipliers - 1).abs().argmax()  # Furthest from 1
            patch_multiplier = multipliers[idx]
        elif aggregation == 'max_impact':
            # NEW: Use the feature with highest impact (strongest absolute bias)
            # This avoids dilution by letting the "expert" feature make the decision
            biases = torch.tensor(patch_biases[patch_idx], device=device)
            abs_biases = biases.abs()
            max_impact_idx = abs_biases.argmax()
            patch_multiplier = multipliers[max_impact_idx]
        else:
            # Simple mean as fallback
            patch_multiplier = multipliers.mean()

        multiplier_mask[patch_idx] = patch_multiplier
        max_multiplier = max(max_multiplier, patch_multiplier.item())
        min_multiplier = min(min_multiplier, patch_multiplier.item())

    if debug:
        print(f"\nMultiplier Summary:")
        print(f"  Features used: {features_used}/{len(all_features)}")
        print(f"  Features filtered: {features_filtered}")
        print(f"  Patches corrected: {patches_corrected}/{n_patches}")
        print(f"  Multiplier range: [{min_multiplier:.3f}, {max_multiplier:.3f}]")
        print(f"  Mean multiplier: {multiplier_mask.mean().item():.3f}")

        # Show distribution of multipliers
        boosted = (multiplier_mask > 1.05).sum().item()
        suppressed = (multiplier_mask < 0.95).sum().item()
        unchanged = n_patches - boosted - suppressed
        print(f"  Patches: {boosted} boosted, {suppressed} suppressed, {unchanged} unchanged")

    # Prepare debug info
    debug_info = {
        'features_used': features_used,
        'features_filtered': features_filtered,
        'patches_corrected': patches_corrected,
        'multiplier_range': (min_multiplier, max_multiplier),
        'mean_multiplier': multiplier_mask.mean().item(),
        'patches_boosted': (multiplier_mask > 1.05).sum().item(),
        'patches_suppressed': (multiplier_mask < 0.95).sum().item()
    }

    return multiplier_mask, selected_features, debug_info
