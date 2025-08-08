"""
Improved Boosting Strategy Based on Analysis Findings

Key insights incorporated:
1. Focus on 10-100 occurrence features (best signal-to-noise)
2. Only boost/suppress features with |log ratio| >= 1.5
3. Class-aware corrections (COVID over-attributed, Non-COVID under-attributed)
4. Balance frequency and impact in selection
"""

import torch
from typing import Dict, Any, List, Tuple, Optional


# Class names mapping (from vit/model.py)
IDX2CLS = {0: 'COVID-19', 1: 'Non-COVID', 2: 'Normal'}


def precache_sorted_features(
    saco_results: Dict[str, Any],
    min_occurrences: int = 5,
    max_occurrences: int = 100000,
    min_log_ratio: float = 1.8,
    use_balanced_score: bool = True
):
    """
    Pre-cache sorted feature lists for faster runtime performance.
    Call this once when loading the saco_results dictionary.
    """
    results_by_type = saco_results.get('results_by_type', {})
    
    # Cache over-attributed features
    over_attributed = results_by_type.get('over_attributed', {})
    if over_attributed:
        presorted = []
        for feat_id, stats in over_attributed.items():
            n_occ = stats.get('n_occurrences', 0)
            log_ratio = abs(stats.get('mean_log_ratio', 0))
            
            if n_occ < min_occurrences or n_occ > max_occurrences:
                continue
            if log_ratio < min_log_ratio:
                continue
            
            score = log_ratio * (n_occ ** 0.5) if use_balanced_score else log_ratio
            presorted.append((feat_id, stats, score))
        
        presorted.sort(key=lambda x: x[2], reverse=True)
        build_boost_mask_improved._cached_sorted_over = presorted
        print(f"Pre-cached {len(presorted)} over-attributed features")
    
    # Cache under-attributed features
    under_attributed = results_by_type.get('under_attributed', {})
    if under_attributed:
        presorted = []
        for feat_id, stats in under_attributed.items():
            n_occ = stats.get('n_occurrences', 0)
            log_ratio = stats.get('mean_log_ratio', 0)
            
            if n_occ < min_occurrences or n_occ > max_occurrences:
                continue
            if log_ratio < min_log_ratio:
                continue
            
            score = log_ratio * (n_occ ** 0.5) if use_balanced_score else log_ratio
            presorted.append((feat_id, stats, score))
        
        presorted.sort(key=lambda x: x[2], reverse=True)
        build_boost_mask_improved._cached_sorted_under = presorted
        print(f"Pre-cached {len(presorted)} under-attributed features")


@torch.no_grad()
def build_boost_mask_improved(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    *,
    # Frequency filtering - MUCH WIDER RANGE
    min_occurrences: int = 5,        # Include rarer features
    max_occurrences: int = 100000,      # Include more common features
    
    # Ratio thresholds - LOWER THRESHOLD
    min_log_ratio: float = 2.,      # Include more moderate misalignments
    
    # Class-specific behavior (disabled by default for general testing)
    class_aware: bool = False,
    
    # Strength parameters - STRONGER EFFECTS
    suppress_strength: float = 0.2,  # Stronger suppression (lower = stronger)
    boost_strength: float = 10.0,     # Much stronger boost
    
    # Selection limits - MORE FEATURES
    max_suppress: int = 5,          # More suppressions
    max_boost: int = 15,             # Many more boosts
    
    # Activation threshold - LOWER TO CATCH MORE
    min_activation: float = 0.05,    # Lower threshold to include more patches
    
    # Weighting strategy
    use_balanced_score: bool = True,  # Use mean * sqrt(n_occurrences) for ranking
    
    debug: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Improved boosting strategy based on comprehensive analysis.
    
    Key improvements:
    1. Filters features by occurrence range (10-100 preferred)
    2. Only boosts/suppresses features with meaningful misalignment (|ratio| >= 1.5)
    3. Class-aware: different strategies for COVID vs Non-COVID vs Normal
    4. Uses balanced scoring to weight both frequency and impact
    """
    
    codes = sae_codes[0, 1:].to(device)  # Remove CLS token
    n_patches, n_feats = codes.shape
    boost_mask = torch.ones(n_patches, device=device)
    selected_features = []
    
    results_by_type = saco_results.get('results_by_type', {})
    predicted_class_name = IDX2CLS.get(predicted_class, 'unknown')
    
    if debug:
        print(f"\n=== Improved Boosting for {predicted_class_name} ===")
    
    # For now, use general strategy without class-specific adjustments
    suppress_priority = 1.0
    boost_priority = 1.0
    actual_max_suppress = max_suppress
    actual_max_boost = max_boost
    
    # Optional: mild class awareness (commented out for testing)
    # if class_aware and predicted_class_name == 'Normal':
    #     # Only be more conservative with Normal class
    #     min_log_ratio = 2.0  # Higher threshold for Normal
    #     actual_max_suppress = int(max_suppress * 0.7)
    #     actual_max_boost = int(max_boost * 0.7)
    
    # Process OVER-ATTRIBUTED features (SUPPRESS)
    over_attributed = results_by_type.get('over_attributed', {})
    if over_attributed and actual_max_suppress > 0:
        
        # PRE-COMPUTE active features set for O(1) lookup
        active_features = (codes > min_activation).any(dim=0).nonzero(as_tuple=True)[0]
        active_set = set(active_features.cpu().tolist())
        
        # Pre-filter and pre-sort features ONCE (can be cached in future)
        if not hasattr(build_boost_mask_improved, '_cached_sorted_over'):
            # Cache the sorted list on first call
            presorted = []
            for feat_id, stats in over_attributed.items():
                n_occ = stats.get('n_occurrences', 0)
                log_ratio = abs(stats.get('mean_log_ratio', 0))
                
                # Apply filters
                if n_occ < min_occurrences or n_occ > max_occurrences:
                    continue
                if log_ratio < min_log_ratio:
                    continue
                
                # Pre-compute score
                if use_balanced_score:
                    score = log_ratio * (n_occ ** 0.5)
                else:
                    score = log_ratio
                
                presorted.append((feat_id, stats, score))
            
            # Sort once by score
            presorted.sort(key=lambda x: x[2], reverse=True)
            build_boost_mask_improved._cached_sorted_over = presorted
        
        # Now just filter by active features - VERY FAST
        filtered_features = []
        for feat_id, stats, score in build_boost_mask_improved._cached_sorted_over:
            if feat_id >= n_feats:
                continue
            if feat_id in active_set:  # O(1) lookup
                filtered_features.append((feat_id, stats, score * suppress_priority))
                if len(filtered_features) >= actual_max_suppress:
                    break  # Stop early once we have enough
        
        suppress_count = 0
        for feat_id, stats, score in filtered_features[:actual_max_suppress]:
            # We already checked activity in filtering, get activations directly
            feat_activations = codes[:, feat_id]
            active_mask = feat_activations > min_activation
            
            # Adaptive suppression based on log ratio magnitude
            ratio_magnitude = abs(stats.get('mean_log_ratio', 0))
            # Make suppression stronger and less gradual
            adaptive_suppress = suppress_strength * (0.5 + 0.5 * min(ratio_magnitude / 2.0, 1.0))
            
            # Apply suppression
            patch_suppression = 1.0 - feat_activations.clamp(0, 1) * (1.0 - adaptive_suppress)
            boost_mask *= patch_suppression
            
            selected_features.append(feat_id)
            suppress_count += 1
            
            if debug:
                n_active = active_mask.sum().item()
                print(f"  SUPPRESS {feat_id}: ratio={stats['mean_log_ratio']:.2f}, "
                      f"occ={stats['n_occurrences']}, class={stats['dominant_class']}, "
                      f"strength={adaptive_suppress:.2f}, patches={n_active}")
    
    # Process UNDER-ATTRIBUTED features (BOOST)
    under_attributed = results_by_type.get('under_attributed', {})
    if under_attributed and actual_max_boost > 0:
        
        # Reuse active set if already computed, otherwise compute it
        if 'active_set' not in locals():
            active_features = (codes > min_activation).any(dim=0).nonzero(as_tuple=True)[0]
            active_set = set(active_features.cpu().tolist())
        
        # Pre-filter and pre-sort features ONCE (can be cached in future)
        if not hasattr(build_boost_mask_improved, '_cached_sorted_under'):
            # Cache the sorted list on first call
            presorted = []
            for feat_id, stats in under_attributed.items():
                n_occ = stats.get('n_occurrences', 0)
                log_ratio = stats.get('mean_log_ratio', 0)
                
                # Apply filters
                if n_occ < min_occurrences or n_occ > max_occurrences:
                    continue
                if log_ratio < min_log_ratio:
                    continue
                
                # Pre-compute score
                if use_balanced_score:
                    score = log_ratio * (n_occ ** 0.5)
                else:
                    score = log_ratio
                
                presorted.append((feat_id, stats, score))
            
            # Sort once by score
            presorted.sort(key=lambda x: x[2], reverse=True)
            build_boost_mask_improved._cached_sorted_under = presorted
        
        # Now just filter by active features - VERY FAST
        filtered_features = []
        for feat_id, stats, score in build_boost_mask_improved._cached_sorted_under:
            if feat_id >= n_feats:
                continue
            if feat_id in active_set:  # O(1) lookup
                filtered_features.append((feat_id, stats, score * boost_priority))
                if len(filtered_features) >= actual_max_boost:
                    break  # Stop early once we have enough
        
        boost_count = 0
        for feat_id, stats, score in filtered_features[:actual_max_boost]:
            # We already checked activity in filtering, get activations directly
            feat_activations = codes[:, feat_id]
            active_mask = feat_activations > min_activation
            
            # Adaptive boost based on log ratio magnitude
            ratio_magnitude = stats.get('mean_log_ratio', 0)
            # Make boost stronger and less gradual
            adaptive_boost = 1.0 + (boost_strength - 1.0) * (0.5 + 0.5 * min(ratio_magnitude / 2.0, 1.0))
            
            # Apply boost
            patch_boost = 1.0 + feat_activations.clamp(0, 1) * (adaptive_boost - 1.0)
            boost_mask *= patch_boost
            
            selected_features.append(feat_id)
            boost_count += 1
            
            if debug:
                n_active = active_mask.sum().item()
                print(f"  BOOST {feat_id}: ratio={stats['mean_log_ratio']:.2f}, "
                      f"occ={stats['n_occurrences']}, class={stats['dominant_class']}, "
                      f"strength={adaptive_boost:.2f}, patches={n_active}")
    
    if debug:
        total_active = (codes > min_activation).any(dim=0).sum().item()
        print(f"Improved mask: {len(selected_features)} features selected "
              f"(from {total_active} active), "
              f"suppress={suppress_count}, boost={boost_count}")
        print(f"Class strategy: {predicted_class_name} - "
              f"suppress_priority={suppress_priority:.1f}, "
              f"boost_priority={boost_priority:.1f}")
    
    return boost_mask, selected_features


@torch.no_grad()
def build_boost_mask_conservative(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    *,
    # Very strict thresholds
    min_occurrences: int = 20,
    max_occurrences: int = 80,
    min_log_ratio: float = 2.0,  # Only very misaligned features
    
    # Conservative strengths
    suppress_strength: float = 0.3,
    boost_strength: float = 2.0,
    
    # Fewer features
    max_suppress: int = 5,
    max_boost: int = 5,
    
    min_activation: float = 0.15,  # Higher activation threshold
    
    debug: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Conservative variant: Only corrects the most egregious misalignments.
    Good for when you want minimal intervention.
    """
    return build_boost_mask_improved(
        sae_codes=sae_codes,
        saco_results=saco_results,
        predicted_class=predicted_class,
        device=device,
        min_occurrences=min_occurrences,
        max_occurrences=max_occurrences,
        min_log_ratio=min_log_ratio,
        class_aware=True,
        suppress_strength=suppress_strength,
        boost_strength=boost_strength,
        max_suppress=max_suppress,
        max_boost=max_boost,
        min_activation=min_activation,
        use_balanced_score=True,
        debug=debug
    )


@torch.no_grad()
def build_boost_mask_aggressive(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    *,
    # Wider frequency range
    min_occurrences: int = 5,
    max_occurrences: int = 100000,
    min_log_ratio: float = 2.,  # Lower threshold
    
    # Stronger corrections
    suppress_strength: float = 0.2,  # Stronger suppression
    boost_strength: float = 5.0,     # Much stronger boost
    
    # More features
    max_suppress: int = 15,
    max_boost: int = 20,
    
    min_activation: float = 0.05,  # Lower threshold
    
    debug: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Aggressive variant: Corrects more features with stronger adjustments.
    Good for when attribution quality is very poor.
    """
    return build_boost_mask_improved(
        sae_codes=sae_codes,
        saco_results=saco_results,
        predicted_class=predicted_class,
        device=device,
        min_occurrences=min_occurrences,
        max_occurrences=max_occurrences,
        min_log_ratio=min_log_ratio,
        class_aware=False,
        suppress_strength=suppress_strength,
        boost_strength=boost_strength,
        max_suppress=max_suppress,
        max_boost=max_boost,
        min_activation=min_activation,
        use_balanced_score=True,
        debug=debug
    )


@torch.no_grad()
def build_boost_mask_ultra_aggressive(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    *,
    debug: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Ultra-aggressive variant: Maximum intervention for testing impact.
    - Very wide frequency range (1-1000)
    - Low ratio threshold (0.5)
    - Many features (50 boost, 30 suppress)
    - Strong effects
    """
    return build_boost_mask_improved(
        sae_codes=sae_codes,
        saco_results=saco_results,
        predicted_class=predicted_class,
        device=device,
        min_occurrences=1,        # Include ALL features
        max_occurrences=10000,    # No upper limit
        min_log_ratio=0.5,        # Very low threshold
        class_aware=False,
        suppress_strength=0.1,    # Very strong suppression
        boost_strength=8.0,       # Very strong boost
        max_suppress=30,          # Many suppressions
        max_boost=50,             # MANY boosts
        min_activation=0.01,      # Very low activation threshold
        use_balanced_score=False, # Just use raw ratio for maximum impact
        debug=debug
    )
