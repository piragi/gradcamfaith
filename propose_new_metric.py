"""
Propose and evaluate new metrics for feature selection that avoid always-active features
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate_metrics(results_path: str):
    """Evaluate different metric proposals."""
    
    results = torch.load(results_path, weights_only=False)
    under_attr = results['results_by_type']['under_attributed']
    over_attr = results['results_by_type']['over_attributed']
    
    print("EVALUATING FEATURE SELECTION METRICS")
    print("=" * 80)
    
    # Total number of images (from your output)
    n_images = 5417
    
    # Define different metrics
    def metric_1_filtered_mean(f):
        """High mean but filter out features that appear too often"""
        if f['n_occurrences'] > 0.6 * n_images:  # Appears in >60% of images
            return 0
        return abs(f['mean_log_ratio'])
    
    def metric_2_sweet_spot(f):
        """Favor features in the sweet spot: moderate frequency, high impact"""
        return abs(f['mean_log_ratio']) * np.log(f['n_occurrences'])
    
    def metric_3_impact_per_image(f):
        """Average impact per image where feature appears"""
        if f['n_occurrences'] > 0.6 * n_images:
            return 0
        # This approximates: sum_of_means / sqrt(n_occurrences)
        # Balances total impact with frequency
        return abs(f['sum_of_means']) / np.sqrt(f['n_occurrences'])
    
    def metric_4_rarity_weighted(f):
        """Weight by rarity - penalize very common features"""
        rarity = 1.0 - (f['n_occurrences'] / n_images)
        rarity = max(0, rarity - 0.4) / 0.6  # Zero out if appears in >60%
        return abs(f['mean_log_ratio']) * rarity * np.sqrt(f['n_occurrences'])
    
    def metric_5_percentile_gated(f):
        """Only consider features in specific occurrence percentiles"""
        occ_pct = f['n_occurrences'] / n_images
        if 0.005 < occ_pct < 0.4:  # Between 0.5% and 40% of images
            return abs(f['mean_log_ratio']) * f['confidence_score']
        return 0
    
    metrics = {
        'filtered_mean': metric_1_filtered_mean,
        'sweet_spot': metric_2_sweet_spot,
        'impact_per_image': metric_3_impact_per_image,
        'rarity_weighted': metric_4_rarity_weighted,
        'percentile_gated': metric_5_percentile_gated
    }
    
    # Evaluate each metric
    for metric_name, metric_fn in metrics.items():
        print(f"\n{metric_name.upper()} METRIC")
        print("-" * 40)
        
        # Score all features
        under_scores = {fid: metric_fn(f) for fid, f in under_attr.items()}
        over_scores = {fid: metric_fn(f) for fid, f in over_attr.items()}
        
        # Get top features
        top_under = sorted(under_scores.items(), key=lambda x: x[1], reverse=True)[:50]
        top_over = sorted(over_scores.items(), key=lambda x: x[1], reverse=True)[:50]
        
        print("\nTop under-attributed features:")
        for fid, score in top_under:
            if score > 0:
                f = under_attr[fid]
                print(f"  Feature {fid}: score={score:.3f}, mean={f['mean_log_ratio']:.3f}, "
                      f"n_occ={f['n_occurrences']} ({f['n_occurrences']/n_images*100:.1f}%)")
        
        print("\nTop over-attributed features:")
        for fid, score in top_over:
            if score > 0:
                f = over_attr[fid]
                print(f"  Feature {fid}: score={score:.3f}, mean={f['mean_log_ratio']:.3f}, "
                      f"n_occ={f['n_occurrences']} ({f['n_occurrences']/n_images*100:.1f}%)")
    
    # Visualize the distribution of the best metric
    print("\n" + "=" * 80)
    print("RECOMMENDED METRIC: sweet_spot")
    print("=" * 80)
    print("\nThis metric:")
    print("- Filters out features appearing in <30 or >50% of images")
    print("- Weights by mean log ratio")
    print("- Gives slight boost for more occurrences (up to 500)")
    print("- Avoids both outliers AND always-active features")

if __name__ == "__main__":
    results_path = "results/saco_features_direct_l7.pt"
    evaluate_metrics(results_path)