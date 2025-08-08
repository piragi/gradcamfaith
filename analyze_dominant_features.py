"""
Analyze the specific features that dominate when using sum_of_means metric
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List

def analyze_dominant_features(results_path: str, feature_ids: List[int]):
    """Analyze specific features that keep appearing in boosting."""
    
    # Load results
    results = torch.load(results_path, weights_only=False)
    under_attr = results['results_by_type']['under_attributed']
    over_attr = results['results_by_type']['over_attributed']
    
    print("ANALYSIS OF DOMINANT FEATURES")
    print("=" * 80)
    
    # Combine both dictionaries for analysis
    all_features = {}
    for fid in feature_ids:
        if fid in under_attr:
            all_features[fid] = ('under', under_attr[fid])
        elif fid in over_attr:
            all_features[fid] = ('over', over_attr[fid])
        else:
            print(f"Feature {fid}: NOT FOUND")
            continue
    
    # Analyze each feature
    for fid, (category, stats) in all_features.items():
        print(f"\nFeature {fid} ({category}-attributed):")
        print(f"  Mean log ratio: {stats['mean_log_ratio']:.3f}")
        print(f"  Sum of means: {stats['sum_of_means']:.3f}")
        print(f"  Occurrences: {stats['n_occurrences']} ({stats['n_occurrences']/5417*100:.1f}% of images)")
        print(f"  Confidence score: {stats['confidence_score']:.3f}")
        print(f"  Dominant class: {stats['dominant_class']}")
        
        if 'class_distribution' in stats:
            total = sum(stats['class_distribution'].values())
            print("  Class distribution:")
            for cls, count in stats['class_distribution'].items():
                print(f"    {cls}: {count} ({count/total*100:.1f}%)")
    
    # Calculate what percentage of total boost/suppress effect these features account for
    print("\n" + "=" * 80)
    print("IMPACT ANALYSIS")
    print("=" * 80)
    
    # Total sum_of_means across all features
    total_under_sum = sum(f['sum_of_means'] for f in under_attr.values())
    total_over_sum = sum(abs(f['sum_of_means']) for f in over_attr.values())
    
    # Sum for our dominant features
    dominant_under_sum = sum(stats['sum_of_means'] for fid, (cat, stats) in all_features.items() if cat == 'under')
    dominant_over_sum = sum(abs(stats['sum_of_means']) for fid, (cat, stats) in all_features.items() if cat == 'over')
    
    print(f"\nUnder-attributed features:")
    print(f"  Total sum across all {len(under_attr)} features: {total_under_sum:.1f}")
    print(f"  Sum from these {sum(1 for _, (cat, _) in all_features.items() if cat == 'under')} features: {dominant_under_sum:.1f}")
    print(f"  Percentage: {dominant_under_sum/total_under_sum*100:.1f}%")
    
    print(f"\nOver-attributed features:")
    print(f"  Total sum across all {len(over_attr)} features: {total_over_sum:.1f}")
    print(f"  Sum from these {sum(1 for _, (cat, _) in all_features.items() if cat == 'over')} features: {dominant_over_sum:.1f}")
    print(f"  Percentage: {dominant_over_sum/total_over_sum*100:.1f}%")
    
    # Suggest better candidates
    print("\n" + "=" * 80)
    print("BETTER CANDIDATE FEATURES")
    print("=" * 80)
    
    # Find features with high mean, moderate frequency
    print("\nUnder-attributed candidates (high impact, moderate frequency):")
    candidates = [(fid, f) for fid, f in under_attr.items() 
                  if 30 <= f['n_occurrences'] <= 500 and f['mean_log_ratio'] > 2.0]
    candidates.sort(key=lambda x: x[1]['mean_log_ratio'], reverse=True)
    
    for fid, stats in candidates[:10]:
        print(f"  Feature {fid}: mean={stats['mean_log_ratio']:.3f}, "
              f"n_occ={stats['n_occurrences']} ({stats['n_occurrences']/5417*100:.1f}%), "
              f"class={stats['dominant_class']}")
    
    print("\nOver-attributed candidates (high impact, moderate frequency):")
    candidates = [(fid, f) for fid, f in over_attr.items() 
                  if 30 <= f['n_occurrences'] <= 500 and f['mean_log_ratio'] < -2.0]
    candidates.sort(key=lambda x: x[1]['mean_log_ratio'])
    
    for fid, stats in candidates[:10]:
        print(f"  Feature {fid}: mean={stats['mean_log_ratio']:.3f}, "
              f"n_occ={stats['n_occurrences']} ({stats['n_occurrences']/5417*100:.1f}%), "
              f"class={stats['dominant_class']}")

if __name__ == "__main__":
    # The features that keep appearing when using sum_of_means
    dominant_features = [16371, 30636, 37592, 5473, 17544, 38446, 21844, 42805, 15984, 17586, 15163, 48122, 8895]
    
    results_path = "results/saco_features_direct_l7.pt"
    analyze_dominant_features(results_path, dominant_features)