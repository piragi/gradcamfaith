"""
Analyze the numerical properties of features from the SaCo analysis
to help determine the best metric for feature selection.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, Any

def analyze_feature_properties(results_path: str):
    """Load and analyze the feature statistics."""
    
    # Load results
    results = torch.load(results_path, weights_only=False)
    
    # Extract feature data
    under_attr = results['results_by_type']['under_attributed']
    over_attr = results['results_by_type']['over_attributed']
    
    print(f"Total under-attributed features: {len(under_attr)}")
    print(f"Total over-attributed features: {len(over_attr)}")
    print("\n" + "="*80 + "\n")
    
    # Analyze under-attributed features (high impact, low attribution)
    analyze_feature_category("UNDER-ATTRIBUTED", under_attr)
    
    print("\n" + "="*80 + "\n")
    
    # Analyze over-attributed features (low impact, high attribution)
    analyze_feature_category("OVER-ATTRIBUTED", over_attr)
    
    # Create correlation plots
    create_correlation_plots(under_attr, over_attr)
    
    # Analyze metric distributions
    analyze_metric_distributions(under_attr, over_attr)
    
    # Find outliers and patterns
    find_patterns(under_attr, over_attr)

def analyze_feature_category(category: str, features: Dict[int, Dict[str, Any]]):
    """Analyze a category of features."""
    print(f"{category} FEATURES ANALYSIS")
    print("-" * 40)
    
    if not features:
        print("No features in this category")
        return
    
    # Extract metrics
    mean_log_ratios = [f['mean_log_ratio'] for f in features.values()]
    sum_of_means = [f['sum_of_means'] for f in features.values()]
    sum_of_sums = [f['sum_of_sums'] for f in features.values()]
    n_occurrences = [f['n_occurrences'] for f in features.values()]
    avg_n_patches = [f.get('avg_n_patches', 0) for f in features.values()]
    confidence_scores = [f['confidence_score'] for f in features.values()]
    
    # Basic statistics
    print(f"\nMean Log Ratio:")
    print(f"  Range: [{np.min(mean_log_ratios):.3f}, {np.max(mean_log_ratios):.3f}]")
    print(f"  Mean: {np.mean(mean_log_ratios):.3f}, Std: {np.std(mean_log_ratios):.3f}")
    
    print(f"\nSum of Means:")
    print(f"  Range: [{np.min(sum_of_means):.3f}, {np.max(sum_of_means):.3f}]")
    print(f"  Mean: {np.mean(sum_of_means):.3f}, Std: {np.std(sum_of_means):.3f}")
    
    print(f"\nOccurrences:")
    print(f"  Range: [{np.min(n_occurrences)}, {np.max(n_occurrences)}]")
    print(f"  Mean: {np.mean(n_occurrences):.1f}, Std: {np.std(n_occurrences):.1f}")
    
    if any(avg_n_patches):  # Only show if data exists
        print(f"\nAverage Patches per Feature:")
        print(f"  Range: [{np.min(avg_n_patches):.1f}, {np.max(avg_n_patches):.1f}]")
        print(f"  Mean: {np.mean(avg_n_patches):.1f}, Std: {np.std(avg_n_patches):.1f}")
    
    # Analyze correlations
    print(f"\nCorrelations:")
    print(f"  mean_log_ratio vs n_occurrences: {np.corrcoef(mean_log_ratios, n_occurrences)[0,1]:.3f}")
    if any(avg_n_patches):
        print(f"  mean_log_ratio vs avg_n_patches: {np.corrcoef(mean_log_ratios, avg_n_patches)[0,1]:.3f}")
        print(f"  n_occurrences vs avg_n_patches: {np.corrcoef(n_occurrences, avg_n_patches)[0,1]:.3f}")
    
    # Top features by different metrics
    print(f"\nTop 5 features by different metrics:")
    
    # Sort by absolute mean_log_ratio
    sorted_by_mean = sorted(features.items(), key=lambda x: abs(x[1]['mean_log_ratio']), reverse=True)[:5]
    print(f"\n  By |mean_log_ratio|:")
    for feat_id, stats in sorted_by_mean:
        avg_p = stats.get('avg_n_patches', 'N/A')
        avg_p_str = f"{avg_p:.1f}" if isinstance(avg_p, (int, float)) else avg_p
        print(f"    Feature {feat_id}: mean={stats['mean_log_ratio']:.3f}, n_occ={stats['n_occurrences']}, avg_patches={avg_p_str}")
    
    # Sort by sum_of_means
    sorted_by_sum = sorted(features.items(), key=lambda x: abs(x[1]['sum_of_means']), reverse=True)[:5]
    print(f"\n  By |sum_of_means|:")
    for feat_id, stats in sorted_by_sum:
        print(f"    Feature {feat_id}: sum_means={stats['sum_of_means']:.3f}, mean={stats['mean_log_ratio']:.3f}, n_occ={stats['n_occurrences']}")
    
    # Sort by a balanced metric: |mean| * sqrt(n_occurrences)
    balanced_metric = {fid: abs(f['mean_log_ratio']) * np.sqrt(f['n_occurrences']) for fid, f in features.items()}
    sorted_by_balanced = sorted(balanced_metric.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  By balanced metric (|mean| * sqrt(n_occ)):")
    for feat_id, score in sorted_by_balanced:
        stats = features[feat_id]
        print(f"    Feature {feat_id}: score={score:.3f}, mean={stats['mean_log_ratio']:.3f}, n_occ={stats['n_occurrences']}")

def create_correlation_plots(under_attr: Dict, over_attr: Dict):
    """Create scatter plots to visualize relationships between metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for row, (features, title) in enumerate([(under_attr, "Under-attributed"), (over_attr, "Over-attributed")]):
        if not features:
            continue
            
        # Extract data
        mean_log = [abs(f['mean_log_ratio']) for f in features.values()]
        n_occ = [f['n_occurrences'] for f in features.values()]
        avg_patches = [f.get('avg_n_patches', 0) for f in features.values()]
        
        # Plot 1: |mean_log_ratio| vs n_occurrences
        ax = axes[row, 0]
        ax.scatter(n_occ, mean_log, alpha=0.5)
        ax.set_xlabel('N Occurrences')
        ax.set_ylabel('|Mean Log Ratio|')
        ax.set_title(f'{title}: |Mean| vs Occurrences')
        ax.set_xscale('log')
        
        # Plot 2: |mean_log_ratio| vs avg_n_patches
        ax = axes[row, 1]
        ax.scatter(avg_patches, mean_log, alpha=0.5)
        ax.set_xlabel('Avg Patches')
        ax.set_ylabel('|Mean Log Ratio|')
        ax.set_title(f'{title}: |Mean| vs Avg Patches')
        
        # Plot 3: n_occurrences vs avg_n_patches
        ax = axes[row, 2]
        ax.scatter(n_occ, avg_patches, alpha=0.5)
        ax.set_xlabel('N Occurrences')
        ax.set_ylabel('Avg Patches')
        ax.set_title(f'{title}: Occurrences vs Patches')
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('feature_correlations.png', dpi=150)
    print("\nSaved correlation plots to feature_correlations.png")

def analyze_metric_distributions(under_attr: Dict, over_attr: Dict):
    """Analyze the distribution of different metrics."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for row, (features, title) in enumerate([(under_attr, "Under-attributed"), (over_attr, "Over-attributed")]):
        if not features:
            continue
        
        # Different potential metrics
        mean_log = [abs(f['mean_log_ratio']) for f in features.values()]
        n_occ = [f['n_occurrences'] for f in features.values()]
        
        # Metric 1: Balanced score
        balanced = [abs(f['mean_log_ratio']) * np.sqrt(f['n_occurrences']) for f in features.values()]
        
        # Metric 2: Harmonic mean of effect and frequency
        harmonic = [2 * abs(f['mean_log_ratio']) * f['n_occurrences'] / (abs(f['mean_log_ratio']) + f['n_occurrences'] + 1e-6) for f in features.values()]
        
        # Metric 3: Effect size with frequency threshold
        effect_with_threshold = [abs(f['mean_log_ratio']) if f['n_occurrences'] >= 10 else 0 for f in features.values()]
        
        # Metric 4: Percentile-based composite
        mean_percentile = np.array([np.percentile(mean_log, 100 * (i+1)/len(mean_log)) for i in range(len(mean_log))])
        occ_percentile = np.array([np.percentile(n_occ, 100 * (i+1)/len(n_occ)) for i in range(len(n_occ))])
        composite = mean_percentile * occ_percentile / 100
        
        # Plot histograms
        metrics = [balanced, harmonic, effect_with_threshold, composite]
        names = ['Balanced\n(|mean|*sqrt(n))', 'Harmonic Mean', 'Thresholded\n(n>=10)', 'Percentile\nComposite']
        
        for i, (metric, name) in enumerate(zip(metrics, names)):
            ax = axes[row, i]
            ax.hist(metric, bins=30, alpha=0.7)
            ax.set_xlabel(name)
            ax.set_ylabel('Count')
            ax.set_title(f'{title}')
    
    plt.tight_layout()
    plt.savefig('metric_distributions.png', dpi=150)
    print("Saved metric distributions to metric_distributions.png")

def find_patterns(under_attr: Dict, over_attr: Dict):
    """Find interesting patterns in the data."""
    print("\n" + "="*80)
    print("INTERESTING PATTERNS")
    print("="*80)
    
    # Find features that appear frequently but have low effect
    print("\nFeatures with HIGH frequency but LOW effect (potential noise):")
    for features, category in [(under_attr, "under-attributed"), (over_attr, "over-attributed")]:
        high_freq_low_effect = [(fid, f) for fid, f in features.items() 
                                if f['n_occurrences'] > 100 and abs(f['mean_log_ratio']) < 0.5]
        print(f"\n{category}: {len(high_freq_low_effect)} features")
        for fid, f in high_freq_low_effect[:3]:
            print(f"  Feature {fid}: n_occ={f['n_occurrences']}, mean={f['mean_log_ratio']:.3f}")
    
    # Find features that appear rarely but have high effect
    print("\n\nFeatures with LOW frequency but HIGH effect (potential outliers):")
    for features, category in [(under_attr, "under-attributed"), (over_attr, "over-attributed")]:
        low_freq_high_effect = [(fid, f) for fid, f in features.items() 
                                if f['n_occurrences'] < 20 and abs(f['mean_log_ratio']) > 2.0]
        print(f"\n{category}: {len(low_freq_high_effect)} features")
        for fid, f in low_freq_high_effect[:3]:
            print(f"  Feature {fid}: n_occ={f['n_occurrences']}, mean={f['mean_log_ratio']:.3f}")
    
    # Find the "sweet spot" features
    print("\n\nFeatures in the 'sweet spot' (moderate frequency, strong effect):")
    for features, category in [(under_attr, "under-attributed"), (over_attr, "over-attributed")]:
        sweet_spot = [(fid, f) for fid, f in features.items() 
                      if 30 <= f['n_occurrences'] <= 200 and abs(f['mean_log_ratio']) > 1.0]
        print(f"\n{category}: {len(sweet_spot)} features")
        for fid, f in sorted(sweet_spot, key=lambda x: abs(x[1]['mean_log_ratio']), reverse=True)[:5]:
            avg_p = f.get('avg_n_patches', 'N/A')
            avg_p_str = f", avg_patches={avg_p:.1f}" if isinstance(avg_p, (int, float)) else ""
            print(f"  Feature {fid}: n_occ={f['n_occurrences']}, mean={f['mean_log_ratio']:.3f}{avg_p_str}")
    
    # Analyze class distribution patterns
    print("\n\nClass-specific features:")
    for features, category in [(under_attr, "under-attributed"), (over_attr, "over-attributed")]:
        class_specific = [(fid, f) for fid, f in features.items() 
                          if 'class_distribution' in f and max(f['class_distribution'].values()) / sum(f['class_distribution'].values()) > 0.8]
        print(f"\n{category}: {len(class_specific)} class-specific features")
        for fid, f in class_specific[:3]:
            print(f"  Feature {fid}: dominant_class={f['dominant_class']}, distribution={f['class_distribution']}")

if __name__ == "__main__":
    results_path = "results/saco_features_direct_l7_test.pt"
    analyze_feature_properties(results_path)