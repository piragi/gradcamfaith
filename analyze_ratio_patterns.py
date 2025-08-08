"""
Enhanced Analysis Focusing on Log Ratio Patterns and Concentration

This script provides deeper insights into the log ratio distributions
and their relationship with feature occurrences.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_and_prepare_data(file_path: str):
    """Load results and prepare dataframes for analysis."""
    results = torch.load(file_path, map_location='cpu', weights_only=False)
    
    under_attr = results['results_by_type']['under_attributed']
    over_attr = results['results_by_type']['over_attributed']
    
    # Convert to dataframes
    data_under = []
    for feat_id, feat_stats in under_attr.items():
        data_under.append({
            'feature_id': feat_id,
            'log_ratio': feat_stats['mean_log_ratio'],
            'occurrences': feat_stats['n_occurrences'],
            'std_ratio': feat_stats.get('std_log_ratio', 0),
            'avg_patches': feat_stats.get('avg_n_patches', 0),
            'dominant_class': feat_stats.get('dominant_class', 'unknown'),
            'category': 'under_attributed'
        })
    
    data_over = []
    for feat_id, feat_stats in over_attr.items():
        data_over.append({
            'feature_id': feat_id,
            'log_ratio': feat_stats['mean_log_ratio'],
            'occurrences': feat_stats['n_occurrences'],
            'std_ratio': feat_stats.get('std_log_ratio', 0),
            'avg_patches': feat_stats.get('avg_n_patches', 0),
            'dominant_class': feat_stats.get('dominant_class', 'unknown'),
            'category': 'over_attributed'
        })
    
    df_under = pd.DataFrame(data_under)
    df_over = pd.DataFrame(data_over)
    
    return df_under, df_over, results


def analyze_ratio_concentration(df_under, df_over):
    """Analyze how concentrated the log ratios are."""
    
    print("\n" + "="*80)
    print("LOG RATIO CONCENTRATION ANALYSIS")
    print("="*80)
    
    for category, df in [('Under-attributed', df_under), ('Over-attributed', df_over)]:
        if df.empty:
            continue
            
        ratios = df['log_ratio'].values
        abs_ratios = np.abs(ratios)
        
        print(f"\n{category.upper()}:")
        print("-"*40)
        
        # Basic statistics
        print(f"Total features: {len(df)}")
        print(f"\nLog Ratio Statistics:")
        print(f"  Mean: {np.mean(ratios):.3f}")
        print(f"  Median: {np.median(ratios):.3f}")
        print(f"  Std Dev: {np.std(ratios):.3f}")
        print(f"  IQR: {np.percentile(ratios, 75) - np.percentile(ratios, 25):.3f}")
        print(f"  95% CI: [{np.percentile(ratios, 2.5):.3f}, {np.percentile(ratios, 97.5):.3f}]")
        
        # Concentration metrics
        print(f"\nConcentration Metrics:")
        print(f"  Gini coefficient: {gini_coefficient(abs_ratios):.3f}")
        print(f"  Coefficient of variation: {np.std(ratios) / (np.mean(np.abs(ratios)) + 1e-10):.3f}")
        
        # Percentage within different ranges
        print(f"\nPercentage of features within absolute log ratio ranges:")
        ranges = [(0, 0.5), (0, 1.0), (0, 1.5), (0, 2.0), (0, 3.0)]
        for low, high in ranges:
            pct = 100 * np.sum((abs_ratios >= low) & (abs_ratios <= high)) / len(abs_ratios)
            print(f"  |log ratio| ≤ {high}: {pct:.1f}%")
        
        # Extreme values
        print(f"\nExtreme values:")
        n_extreme = 10
        if category == 'Under-attributed':
            top_features = df.nlargest(n_extreme, 'log_ratio')
            print(f"  Top {n_extreme} highest ratios:")
            for _, row in top_features.iterrows():
                print(f"    Feature {int(row['feature_id']):5d}: ratio={row['log_ratio']:.3f}, "
                      f"occ={int(row['occurrences']):4d}, class={row['dominant_class']}")
        else:
            top_features = df.nsmallest(n_extreme, 'log_ratio')
            print(f"  Top {n_extreme} most negative ratios:")
            for _, row in top_features.iterrows():
                print(f"    Feature {int(row['feature_id']):5d}: ratio={row['log_ratio']:.3f}, "
                      f"occ={int(row['occurrences']):4d}, class={row['dominant_class']}")


def gini_coefficient(x):
    """Calculate Gini coefficient for concentration measurement."""
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n


def analyze_ratio_vs_occurrence(df_under, df_over):
    """Analyze relationship between log ratio and occurrence frequency."""
    
    print("\n" + "="*80)
    print("LOG RATIO VS OCCURRENCE ANALYSIS")
    print("="*80)
    
    for category, df in [('Under-attributed', df_under), ('Over-attributed', df_over)]:
        if df.empty:
            continue
            
        print(f"\n{category.upper()}:")
        print("-"*40)
        
        # Correlation analysis
        corr = np.corrcoef(np.abs(df['log_ratio']), df['occurrences'])[0, 1]
        spearman_corr = stats.spearmanr(np.abs(df['log_ratio']), df['occurrences'])[0]
        
        print(f"Correlation between |log ratio| and occurrences:")
        print(f"  Pearson: {corr:.3f}")
        print(f"  Spearman: {spearman_corr:.3f}")
        
        # Binned analysis
        occurrence_bins = [1, 10, 50, 100, 500, 1000, 10000]
        df['occ_bin'] = pd.cut(df['occurrences'], bins=occurrence_bins, include_lowest=True)
        
        print(f"\nMean |log ratio| by occurrence frequency:")
        for bin_label in df['occ_bin'].cat.categories:
            bin_data = df[df['occ_bin'] == bin_label]
            if not bin_data.empty:
                mean_ratio = np.mean(np.abs(bin_data['log_ratio']))
                std_ratio = np.std(np.abs(bin_data['log_ratio']))
                count = len(bin_data)
                print(f"  {str(bin_label):15s}: mean={mean_ratio:.3f} ± {std_ratio:.3f} (n={count})")
        
        # High frequency vs low frequency comparison
        median_occ = df['occurrences'].median()
        high_freq = df[df['occurrences'] > median_occ]
        low_freq = df[df['occurrences'] <= median_occ]
        
        print(f"\nHigh vs Low frequency features (split at median={median_occ:.0f}):")
        print(f"  High frequency: mean |ratio|={np.mean(np.abs(high_freq['log_ratio'])):.3f}, n={len(high_freq)}")
        print(f"  Low frequency:  mean |ratio|={np.mean(np.abs(low_freq['log_ratio'])):.3f}, n={len(low_freq)}")
        
        # Statistical test
        statistic, p_value = stats.mannwhitneyu(
            np.abs(high_freq['log_ratio']), 
            np.abs(low_freq['log_ratio']),
            alternative='two-sided'
        )
        print(f"  Mann-Whitney U test p-value: {p_value:.4e}")


def create_enhanced_visualizations(df_under, df_over, save_dir="analysis_plots"):
    """Create enhanced visualizations focusing on log ratios."""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Log Ratio vs Occurrences Scatter
    ax1 = plt.subplot(2, 3, 1)
    if not df_under.empty:
        ax1.scatter(df_under['occurrences'], df_under['log_ratio'], 
                   alpha=0.3, s=10, color='#ff7f0e', label='Under-attr')
    if not df_over.empty:
        ax1.scatter(df_over['occurrences'], df_over['log_ratio'], 
                   alpha=0.3, s=10, color='#2ca02c', label='Over-attr')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Occurrences (log scale)')
    ax1.set_ylabel('Log Ratio')
    ax1.set_title('Log Ratio vs Feature Occurrences')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Absolute Log Ratio Distribution (KDE)
    ax2 = plt.subplot(2, 3, 2)
    if not df_under.empty:
        sns.kdeplot(np.abs(df_under['log_ratio']), ax=ax2, label='Under-attr', color='#ff7f0e')
    if not df_over.empty:
        sns.kdeplot(np.abs(df_over['log_ratio']), ax=ax2, label='Over-attr', color='#2ca02c')
    
    ax2.set_xlabel('|Log Ratio|')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Absolute Log Ratios')
    ax2.legend()
    ax2.set_xlim(0, 5)
    
    # 3. Cumulative Distribution
    ax3 = plt.subplot(2, 3, 3)
    if not df_under.empty:
        sorted_ratios = np.sort(np.abs(df_under['log_ratio']))
        cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        ax3.plot(sorted_ratios, cdf, label='Under-attr', color='#ff7f0e', linewidth=2)
    
    if not df_over.empty:
        sorted_ratios = np.sort(np.abs(df_over['log_ratio']))
        cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        ax3.plot(sorted_ratios, cdf, label='Over-attr', color='#2ca02c', linewidth=2)
    
    ax3.set_xlabel('|Log Ratio|')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution of |Log Ratios|')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, 5)
    
    # 4. Hexbin plot for density
    ax4 = plt.subplot(2, 3, 4)
    df_all = pd.concat([df_under, df_over])
    if not df_all.empty:
        hexbin = ax4.hexbin(df_all['occurrences'], np.abs(df_all['log_ratio']), 
                            xscale='log', gridsize=30, cmap='YlOrRd', mincnt=1)
        plt.colorbar(hexbin, ax=ax4, label='Feature Count')
    
    ax4.set_xlabel('Number of Occurrences (log scale)')
    ax4.set_ylabel('|Log Ratio|')
    ax4.set_title('Feature Density Heatmap')
    ax4.set_ylim(0, 5)
    
    # 5. Box plots by occurrence bins
    ax5 = plt.subplot(2, 3, 5)
    occurrence_bins = [1, 10, 50, 100, 500, 1000, 10000]
    bin_labels = ['1-10', '10-50', '50-100', '100-500', '500-1K', '1K+']
    
    df_all['occ_bin'] = pd.cut(df_all['occurrences'], bins=occurrence_bins, 
                               labels=bin_labels, include_lowest=True)
    df_all['abs_log_ratio'] = np.abs(df_all['log_ratio'])
    
    box_data = []
    positions = []
    colors = []
    pos = 0
    
    for category, color in [('under_attributed', '#ff7f0e'), ('over_attributed', '#2ca02c')]:
        cat_data = df_all[df_all['category'] == category]
        for i, bin_label in enumerate(bin_labels):
            bin_data = cat_data[cat_data['occ_bin'] == bin_label]['abs_log_ratio']
            if not bin_data.empty:
                box_data.append(bin_data.values)
                positions.append(pos)
                colors.append(color)
            pos += 1
        pos += 0.5  # Gap between categories
    
    bp = ax5.boxplot(box_data, positions=positions, widths=0.6, 
                     patch_artist=True, showfliers=False)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.set_xlabel('Occurrence Bins')
    ax5.set_ylabel('|Log Ratio|')
    ax5.set_title('|Log Ratio| Distribution by Occurrence Frequency')
    ax5.set_ylim(0, 3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ff7f0e', alpha=0.7, label='Under-attr'),
                      Patch(facecolor='#2ca02c', alpha=0.7, label='Over-attr')]
    ax5.legend(handles=legend_elements)
    
    # 6. Class-specific ratio distributions
    ax6 = plt.subplot(2, 3, 6)
    classes = df_all['dominant_class'].unique()
    x_pos = np.arange(len(classes))
    width = 0.35
    
    under_means = []
    over_means = []
    under_stds = []
    over_stds = []
    
    for cls in classes:
        under_data = df_under[df_under['dominant_class'] == cls]['log_ratio']
        over_data = df_over[df_over['dominant_class'] == cls]['log_ratio']
        
        under_means.append(np.mean(under_data) if not under_data.empty else 0)
        over_means.append(np.abs(np.mean(over_data)) if not over_data.empty else 0)
        under_stds.append(np.std(under_data) if not under_data.empty else 0)
        over_stds.append(np.std(over_data) if not over_data.empty else 0)
    
    ax6.bar(x_pos - width/2, under_means, width, yerr=under_stds, 
           label='Under-attr', color='#ff7f0e', alpha=0.7, capsize=5)
    ax6.bar(x_pos + width/2, over_means, width, yerr=over_stds,
           label='Over-attr |ratio|', color='#2ca02c', alpha=0.7, capsize=5)
    
    ax6.set_xlabel('Dominant Class')
    ax6.set_ylabel('Mean Log Ratio (absolute for over-attr)')
    ax6.set_title('Mean Ratios by Class')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(classes, rotation=45, ha='right')
    ax6.legend()
    
    plt.suptitle('Enhanced Log Ratio Analysis', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    save_file = save_path / "enhanced_ratio_analysis.png"
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    print(f"\nEnhanced visualizations saved to: {save_file}")
    
    plt.show()


def main():
    """Main analysis function."""
    
    # Analyze the most recent layer 6 file
    file_path = "results/saco_features_direct_l6.pt"
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"Analyzing: {file_path}")
    
    # Load and prepare data
    df_under, df_over, results = load_and_prepare_data(file_path)
    
    # Run analyses
    analyze_ratio_concentration(df_under, df_over)
    analyze_ratio_vs_occurrence(df_under, df_over)
    
    # Create visualizations
    create_enhanced_visualizations(df_under, df_over)
    
    # Save detailed data for further analysis
    df_all = pd.concat([df_under, df_over])
    df_all.to_csv("detailed_ratio_analysis.csv", index=False)
    print(f"\nDetailed data saved to: detailed_ratio_analysis.csv")


if __name__ == "__main__":
    main()