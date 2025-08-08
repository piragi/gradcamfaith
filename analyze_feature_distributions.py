"""
Analyze the distribution of features in the SaCo dictionary
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_feature_distributions(results_path: str):
    """Analyze how features are distributed across different dimensions."""
    
    # Load results
    results = torch.load(results_path, weights_only=False)
    under_attr = results['results_by_type']['under_attributed']
    over_attr = results['results_by_type']['over_attributed']
    
    n_images = 5417
    
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"Total under-attributed features: {len(under_attr)}")
    print(f"Total over-attributed features: {len(over_attr)}")
    
    # Extract data for analysis
    def extract_data(features):
        data = {
            'n_occurrences': [],
            'mean_log_ratio': [],
            'frequency_pct': [],
            'feature_ids': []
        }
        
        for fid, f in features.items():
            data['n_occurrences'].append(f['n_occurrences'])
            data['mean_log_ratio'].append(f['mean_log_ratio'])
            data['frequency_pct'].append(f['n_occurrences'] / n_images * 100)
            data['feature_ids'].append(fid)
        
        return data
    
    under_data = extract_data(under_attr)
    over_data = extract_data(over_attr)
    
    # 1. OCCURRENCE DISTRIBUTION
    print("\n1. OCCURRENCE DISTRIBUTION")
    print("-" * 40)
    
    def analyze_occurrences(data, category):
        n_occ = np.array(data['n_occurrences'])
        
        print(f"\n{category}:")
        print(f"  Min occurrences: {n_occ.min()}")
        print(f"  Max occurrences: {n_occ.max()}")
        print(f"  Mean occurrences: {n_occ.mean():.1f}")
        print(f"  Median occurrences: {np.median(n_occ):.1f}")
        print(f"  Std occurrences: {n_occ.std():.1f}")
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\n  Percentiles:")
        for p in percentiles:
            val = np.percentile(n_occ, p)
            print(f"    {p}%: {val:.0f} occurrences ({val/n_images*100:.1f}% of images)")
        
        # Frequency bins
        freq_pct = np.array(data['frequency_pct'])
        bins = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        hist, edges = np.histogram(freq_pct, bins=bins)
        
        print(f"\n  Distribution by frequency:")
        for i in range(len(hist)):
            if hist[i] > 0:
                print(f"    {edges[i]:>3.0f}-{edges[i+1]:>3.0f}%: {hist[i]:>5} features ({hist[i]/len(n_occ)*100:>5.1f}%)")
    
    analyze_occurrences(under_data, "Under-attributed")
    analyze_occurrences(over_data, "Over-attributed")
    
    # 2. LOG RATIO DISTRIBUTION
    print("\n\n2. LOG RATIO DISTRIBUTION")
    print("-" * 40)
    
    def analyze_log_ratios(data, category):
        log_ratios = np.array(data['mean_log_ratio'])
        
        print(f"\n{category}:")
        print(f"  Min log ratio: {log_ratios.min():.3f}")
        print(f"  Max log ratio: {log_ratios.max():.3f}")
        print(f"  Mean log ratio: {log_ratios.mean():.3f}")
        print(f"  Median log ratio: {np.median(log_ratios):.3f}")
        print(f"  Std log ratio: {log_ratios.std():.3f}")
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\n  Percentiles:")
        for p in percentiles:
            print(f"    {p}%: {np.percentile(log_ratios, p):.3f}")
        
        # Distribution by magnitude
        abs_ratios = np.abs(log_ratios)
        bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 20.0]
        hist, edges = np.histogram(abs_ratios, bins=bins)
        
        print(f"\n  Distribution by |log ratio|:")
        for i in range(len(hist)):
            if hist[i] > 0:
                print(f"    {edges[i]:>3.1f}-{edges[i+1]:>3.1f}: {hist[i]:>5} features ({hist[i]/len(log_ratios)*100:>5.1f}%)")
    
    analyze_log_ratios(under_data, "Under-attributed")
    analyze_log_ratios(over_data, "Over-attributed")
    
    # 3. JOINT DISTRIBUTION ANALYSIS
    print("\n\n3. JOINT DISTRIBUTION: LOG RATIO vs OCCURRENCE")
    print("-" * 40)
    
    def analyze_joint_distribution(data, category):
        n_occ = np.array(data['n_occurrences'])
        log_ratios = np.abs(data['mean_log_ratio'])
        freq_pct = np.array(data['frequency_pct'])
        
        # Create 2D histogram
        freq_bins = [0, 2, 5, 10, 20, 40, 60, 100]
        ratio_bins = [0, 1, 1.5, 2, 3, 20]
        
        print(f"\n{category} - 2D distribution (|log ratio| vs frequency %):")
        print(f"{'Freq %':<12}", end='')
        for i in range(len(ratio_bins)-1):
            print(f"|log|:{ratio_bins[i]}-{ratio_bins[i+1]:<4}", end='  ')
        print()
        print("-" * 80)
        
        for i in range(len(freq_bins)-1):
            freq_mask = (freq_pct >= freq_bins[i]) & (freq_pct < freq_bins[i+1])
            print(f"{freq_bins[i]:>2}-{freq_bins[i+1]:<3}%:    ", end='')
            
            for j in range(len(ratio_bins)-1):
                ratio_mask = (log_ratios >= ratio_bins[j]) & (log_ratios < ratio_bins[j+1])
                count = np.sum(freq_mask & ratio_mask)
                print(f"{count:>8}", end='  ')
            print()
        
        # Find sweet spots
        print(f"\n  Sweet spot candidates (10-40% freq, |log ratio| > 2):")
        sweet_mask = (freq_pct >= 10) & (freq_pct <= 40) & (log_ratios > 2)
        sweet_indices = np.where(sweet_mask)[0]
        
        if len(sweet_indices) > 0:
            # Sort by log ratio
            sorted_sweet = sorted(sweet_indices, key=lambda i: log_ratios[i], reverse=True)
            for idx in sorted_sweet[:5]:
                fid = data['feature_ids'][idx]
                print(f"    Feature {fid}: |log|={log_ratios[idx]:.3f}, freq={freq_pct[idx]:.1f}%, n={n_occ[idx]}")
    
    analyze_joint_distribution(under_data, "Under-attributed")
    analyze_joint_distribution(over_data, "Over-attributed")
    
    # 4. SCORE DISTRIBUTIONS WITH DIFFERENT METRICS
    print("\n\n4. SCORE DISTRIBUTIONS WITH DIFFERENT METRICS")
    print("-" * 40)
    
    def analyze_metric_distributions(data, category):
        n_occ = np.array(data['n_occurrences'])
        log_ratios = np.abs(data['mean_log_ratio'])
        freq_pct = np.array(data['frequency_pct']) / 100  # Convert to fraction
        
        # Calculate different metrics
        metrics = {
            'current (mean*sqrt(n))': log_ratios * np.sqrt(n_occ),
            'mean*log(n)': log_ratios * np.log(n_occ + 1),
            'mean*n^0.3': log_ratios * (n_occ ** 0.3),
            'mean*n^0.2': log_ratios * (n_occ ** 0.2),
        }
        
        print(f"\n{category} - Top 10 features by each metric:")
        
        for metric_name, scores in metrics.items():
            print(f"\n  {metric_name}:")
            top_indices = np.argsort(scores)[-10:][::-1]
            
            for rank, idx in enumerate(top_indices[:5]):
                fid = data['feature_ids'][idx]
                print(f"    {rank+1}. Feature {fid}: score={scores[idx]:.1f}, "
                      f"|log|={log_ratios[idx]:.3f}, freq={freq_pct[idx]*100:.1f}%, n={n_occ[idx]}")
    
    analyze_metric_distributions(under_data, "Under-attributed")
    
    # 5. VISUALIZATION
    print("\n\n5. CREATING VISUALIZATIONS...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Occurrence histogram (log scale)
    ax = axes[0, 0]
    ax.hist(under_data['n_occurrences'], bins=50, alpha=0.7, label='Under-attr', color='blue')
    ax.hist(over_data['n_occurrences'], bins=50, alpha=0.7, label='Over-attr', color='red')
    ax.set_yscale('log')
    ax.set_xlabel('Number of occurrences')
    ax.set_ylabel('Count (log scale)')
    ax.set_title('Occurrence Distribution')
    ax.legend()
    
    # Plot 2: Log ratio histogram
    ax = axes[0, 1]
    ax.hist(np.abs(under_data['mean_log_ratio']), bins=50, alpha=0.7, label='Under-attr', color='blue')
    ax.hist(np.abs(over_data['mean_log_ratio']), bins=50, alpha=0.7, label='Over-attr', color='red')
    ax.set_xlabel('|Mean log ratio|')
    ax.set_ylabel('Count')
    ax.set_title('Log Ratio Distribution')
    ax.legend()
    
    # Plot 3: 2D scatter - log ratio vs occurrence
    ax = axes[0, 2]
    ax.scatter(under_data['frequency_pct'], np.abs(under_data['mean_log_ratio']), 
               alpha=0.3, s=10, label='Under-attr', color='blue')
    ax.scatter(over_data['frequency_pct'], np.abs(over_data['mean_log_ratio']), 
               alpha=0.3, s=10, label='Over-attr', color='red')
    ax.set_xlabel('Frequency (%)')
    ax.set_ylabel('|Mean log ratio|')
    ax.set_title('Log Ratio vs Frequency')
    ax.set_ylim(0, 6)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(2, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    
    # Plot 4: 2D density plot for under-attributed
    ax = axes[1, 0]
    ax.hexbin(under_data['frequency_pct'], np.abs(under_data['mean_log_ratio']), 
              gridsize=30, cmap='Blues', mincnt=1)
    ax.set_xlabel('Frequency (%)')
    ax.set_ylabel('|Mean log ratio|')
    ax.set_title('Under-attributed Density')
    ax.set_ylim(0, 6)
    
    # Plot 5: Current metric distribution
    ax = axes[1, 1]
    current_scores_under = np.abs(under_data['mean_log_ratio']) * np.sqrt(under_data['n_occurrences'])
    ax.hist(current_scores_under, bins=50, alpha=0.7)
    ax.set_xlabel('Current metric score (mean * sqrt(n))')
    ax.set_ylabel('Count')
    ax.set_title('Current Metric Distribution')
    ax.set_yscale('log')
    
    # Plot 6: Frequency percentiles
    ax = axes[1, 2]
    percentiles = np.arange(0, 101, 1)
    under_pct_values = np.percentile(under_data['frequency_pct'], percentiles)
    over_pct_values = np.percentile(over_data['frequency_pct'], percentiles)
    ax.plot(percentiles, under_pct_values, label='Under-attr', color='blue')
    ax.plot(percentiles, over_pct_values, label='Over-attr', color='red')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Frequency Percentiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=150)
    print("Saved visualizations to feature_distributions.png")

if __name__ == "__main__":
    results_path = "results/saco_features_direct_l7.pt"
    analyze_feature_distributions(results_path)