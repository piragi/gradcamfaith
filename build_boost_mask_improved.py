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
    use_abs_ratio: bool = True
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
        
    Returns:
        List of (feat_id, stats, score) tuples sorted by score
    """
    # Check cache first
    if hasattr(build_boost_mask_improved, cache_attr):
        return getattr(build_boost_mask_improved, cache_attr)

    # Build and cache sorted list
    presorted = []
    for feat_id, stats in features_dict.items():
        n_occ = stats.get('n_occurrences', 0)
        log_ratio = abs(stats.get('mean_log_ratio', 0)) if use_abs_ratio else stats.get('mean_log_ratio', 0)

        # Apply filters
        if n_occ < min_occurrences or n_occ > max_occurrences:
            continue
        if log_ratio < min_log_ratio:
            continue

        # Compute score
        if use_balanced_score:
            score = log_ratio * math.log(n_occ + 1)
        else:
            score = log_ratio

        presorted.append((feat_id, stats, score))

    # Sort by score
    presorted.sort(key=lambda x: x[2], reverse=True)

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
    Apply boost or suppress modulation to features.
    
    Args:
        codes: SAE codes [n_patches, n_feats]
        boost_mask: Current boost mask to modify
        features: List of (feat_id, stats, score) tuples
        mode: 'boost' or 'suppress'
        strength: Base strength for modulation
        min_activation: Minimum activation threshold
        debug: Whether to print debug info
        
    Returns:
        Updated boost_mask and list of selected feature IDs
    """
    selected_features = []

    for feat_id, stats, score in features:
        feat_activations = codes[:, feat_id]
        active_mask = feat_activations > min_activation

        # Get ratio magnitude for adaptive strength
        if mode == 'boost':
            ratio_magnitude = stats.get('mean_log_ratio', 0)  # Now contains bin_attribution_bias
            # Adaptive boost based on bin bias magnitude
            adaptive_strength = 1.0 + (strength - 1.0) * (0.5 + 0.5 * min(ratio_magnitude / 2.0, 1.0))
            # Apply boost
            patch_modulation = 1.0 + feat_activations.clamp(0, 1) * (adaptive_strength - 1.0)
        else:  # suppress
            ratio_magnitude = abs(stats.get('mean_log_ratio', 0))  # Now contains bin_attribution_bias
            # Adaptive suppression based on bin bias magnitude
            adaptive_strength = strength * (0.5 + 0.5 * min(ratio_magnitude / 2.0, 1.0))
            # Apply suppression
            patch_modulation = 1.0 - feat_activations.clamp(0, 1) * (1.0 - adaptive_strength)

        boost_mask *= patch_modulation
        selected_features.append(feat_id)

        if debug:
            n_active = active_mask.sum().item()
            print(
                f"  {mode.upper()} {feat_id}: ratio={stats.get('mean_log_ratio', 0):.2f}, "
                f"occ={stats.get('n_occurrences', 0)}, class={stats.get('dominant_class', 'unknown')}, "
                f"strength={adaptive_strength:.2f}, patches={n_active}"
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
    boost_priority: float = 1.0
) -> Tuple[List[Tuple[int, Dict, float]], List[Tuple[int, Dict, float]]]:
    """
    Select features based on SaCo analysis.
    
    Returns:
        Tuple of (suppress_features, boost_features)
    """
    n_feats = codes.shape[1]
    results_by_type = saco_results.get('results_by_type', {})

    suppress_features = []
    boost_features = []

    # Process over-attributed features (suppress)
    if max_suppress > 0:
        over_attributed = results_by_type.get('over_attributed', {})
        if over_attributed:
            sorted_features = get_sorted_saco_features(
                over_attributed,
                '_cached_sorted_over',
                min_occurrences,
                max_occurrences,
                min_log_ratio,
                use_balanced_score,
                use_abs_ratio=True
            )

            # Filter by active features
            for feat_id, stats, score in sorted_features:
                if feat_id >= n_feats:
                    continue
                if feat_id in active_set:
                    suppress_features.append((feat_id, stats, score * suppress_priority))
                    if len(suppress_features) >= max_suppress:
                        break

    # Process under-attributed features (boost)
    if max_boost > 0:
        under_attributed = results_by_type.get('under_attributed', {})
        if under_attributed:
            sorted_features = get_sorted_saco_features(
                under_attributed,
                '_cached_sorted_under',
                min_occurrences,
                max_occurrences,
                min_log_ratio,
                use_balanced_score,
                use_abs_ratio=False
            )

            # Filter by active features
            for feat_id, stats, score in sorted_features:
                if feat_id >= n_feats:
                    continue
                if feat_id in active_set:
                    boost_features.append((feat_id, stats, score * boost_priority))
                    if len(boost_features) >= max_boost:
                        break

    return suppress_features, boost_features


def precache_sorted_features(
    saco_results: Dict[str, Any],
    min_occurrences: int = 1,
    max_occurrences: int = 100000,
    min_log_ratio: float = 1.,
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


@torch.no_grad()
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
    min_log_ratio: float = 1.5,

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
