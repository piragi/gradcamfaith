"""
Optimized boost mask builder that pre-computes and caches feature lists
"""

import torch
from typing import Dict, List, Tuple, Any, Optional


class CachedBoostMaskBuilder:
    """Pre-computes and caches the sorted feature lists for faster boosting"""
    
    def __init__(self, saco_results: Dict[str, Any]):
        self.saco_results = saco_results
        self._cache = {}
        self._precompute_sorted_features()
    
    def _precompute_sorted_features(self):
        """Pre-compute sorted feature lists for each category"""
        results_by_type = self.saco_results.get('results_by_type', {})
        n_images = 5417  # Total images in dataset
        
        # Pre-sort over-attributed features
        over_attributed = results_by_type.get('over_attributed', {})
        if over_attributed:
            # Filter by frequency: keep only 2-30% range
            filtered_over = [
                (fid, stats) for fid, stats in over_attributed.items()
                if 0.02 * n_images <= stats['n_occurrences'] <= 0.3 * n_images
            ]
            # Compute n^0.2 score on the fly
            self._cache['over_attributed'] = sorted(
                filtered_over, 
                key=lambda x: abs(x[1]['mean_log_ratio']) * (x[1]['n_occurrences'] ** 0.2), 
                reverse=True  # Highest score first
            )
            print(f"Filtered over-attributed: {len(over_attributed)} -> {len(self._cache['over_attributed'])} features")
        else:
            self._cache['over_attributed'] = []
        
        # Pre-sort under-attributed features  
        under_attributed = results_by_type.get('under_attributed', {})
        if under_attributed:
            # Filter by frequency: keep only 2-30% range
            filtered_under = [
                (fid, stats) for fid, stats in under_attributed.items()
                if 0.02 * n_images <= stats['n_occurrences'] <= 0.3 * n_images
            ]
            # Compute n^0.2 score on the fly
            self._cache['under_attributed'] = sorted(
                filtered_under,
                key=lambda x: abs(x[1]['mean_log_ratio']) * (x[1]['n_occurrences'] ** 0.2),
                reverse=True  # Highest score first
            )
            print(f"Filtered under-attributed: {len(under_attributed)} -> {len(self._cache['under_attributed'])} features")
            
            # Show top 10 features with new scoring
            print("\nTop 10 under-attributed features by n^0.2 scoring:")
            for i, (fid, stats) in enumerate(self._cache['under_attributed'][:10]):
                n02_score = abs(stats['mean_log_ratio']) * (stats['n_occurrences'] ** 0.2)
                print(f"  {i+1}. Feature {fid}: n^0.2_score={n02_score:.3f}, "
                      f"mean_log={stats['mean_log_ratio']:.3f}, n_occ={stats['n_occurrences']}")
        else:
            self._cache['under_attributed'] = []
    
    @torch.no_grad()
    def build_boost_mask(
        self,
        sae_codes: torch.Tensor,
        device: torch.device,
        *,
        suppress_strength: float = 0.5,
        boost_strength: float = 2.0,
        min_activation: float = 0.05,
        top_k_suppress: int = 10,
        top_k_boost: int = 8,
        use_log_ratio_weighting: bool = True,
        debug: bool = False
    ) -> Tuple[torch.Tensor, List[int]]:
        """Build boost mask using pre-cached sorted features"""
        
        codes = sae_codes[0, 1:].to(device)  # Remove CLS token
        n_patches, n_feats = codes.shape
        boost_mask = torch.ones(n_patches, device=device)
        selected_features = []
        
        # Use pre-cached sorted features for suppression
        suppress_count = 0
        for feat_id, stats in self._cache['over_attributed']:
            if suppress_count >= top_k_suppress:
                break
                
            # Check if feature exists and is active
            if feat_id >= n_feats:
                continue
                
            feat_activations = codes[:, feat_id]
            active_mask = feat_activations > min_activation
            
            if not active_mask.any():
                continue
            
            # Weight suppression by log ratio magnitude if enabled
            if use_log_ratio_weighting:
                # Use mean_log_ratio for weighting instead of sum_of_means
                log_ratio_weight = min(abs(stats['mean_log_ratio']) / 3.0, 1.0)
                effective_suppress = suppress_strength * log_ratio_weight
            else:
                effective_suppress = suppress_strength
            
            # Apply suppression
            patch_suppression = 1.0 - feat_activations.clamp(0, 1) * (1.0 - effective_suppress)
            boost_mask *= patch_suppression
            
            selected_features.append(feat_id)
            suppress_count += 1
            
            if debug:
                n_active = active_mask.sum().item()
                n02_score = abs(stats['mean_log_ratio']) * (stats['n_occurrences'] ** 0.2)
                print(f"  SUPPRESS feature {feat_id}: "
                      f"mean_log={stats['mean_log_ratio']:.3f}, "
                      f"n^0.2_score={n02_score:.3f}, "
                      f"n_occ={stats['n_occurrences']}, "
                      f"strength={effective_suppress:.3f}, "
                      f"active_patches={n_active}")
        
        # Use pre-cached sorted features for boosting
        boost_count = 0
        for feat_id, stats in self._cache['under_attributed']:
            if boost_count >= top_k_boost:
                break
                
            # Check if feature exists and is active
            if feat_id >= n_feats:
                continue
                
            feat_activations = codes[:, feat_id] 
            active_mask = feat_activations > min_activation
            
            if not active_mask.any():
                continue
            
            # Weight boost by log ratio magnitude if enabled
            if use_log_ratio_weighting:
                # Use mean_log_ratio for weighting instead of sum_of_means
                log_ratio_weight = min(stats['mean_log_ratio'] / 3.0, 1.0)
                effective_boost = 1.0 + (boost_strength - 1.0) * log_ratio_weight
            else:
                effective_boost = boost_strength
            
            # Apply boost
            patch_boost = 1.0 + feat_activations.clamp(0, 1) * (effective_boost - 1.0)
            boost_mask *= patch_boost
            
            selected_features.append(feat_id)
            boost_count += 1
            
            if debug:
                n_active = active_mask.sum().item()
                n02_score = abs(stats['mean_log_ratio']) * (stats['n_occurrences'] ** 0.2)
                print(f"  BOOST feature {feat_id}: "
                      f"mean_log={stats['mean_log_ratio']:.3f}, "
                      f"n^0.2_score={n02_score:.3f}, "
                      f"n_occ={stats['n_occurrences']}, "
                      f"strength={effective_boost:.3f}, "
                      f"active_patches={n_active}")
        
        if debug:
            total_active = (codes.abs() > min_activation).any(dim=0).sum().item()
            print(f"Direct feature mask: {len(selected_features)} total features selected "
                  f"(from {total_active} active features)")
        
        return boost_mask, selected_features


# Global cache for the boost mask builder
_cached_builder: Optional[CachedBoostMaskBuilder] = None


def get_cached_boost_mask_builder(saco_results: Dict[str, Any], force_rebuild: bool = False) -> CachedBoostMaskBuilder:
    """Get or create a cached boost mask builder"""
    global _cached_builder
    
    # Check if we need to create a new builder (results changed or forced)
    if _cached_builder is None or force_rebuild:
        _cached_builder = CachedBoostMaskBuilder(saco_results)
    
    return _cached_builder


@torch.no_grad()
def build_boost_mask_saco_direct_cached(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    device: torch.device,
    **kwargs
) -> Tuple[torch.Tensor, List[int]]:
    """Wrapper function that uses the cached builder"""
    builder = get_cached_boost_mask_builder(saco_results)
    return builder.build_boost_mask(sae_codes, device, **kwargs)