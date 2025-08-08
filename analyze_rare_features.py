"""
Deep Analysis of Rare Features with High Misalignment

This script investigates whether rare features with high log ratios are noise
or carry important information about systematic attribution failures.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_full_results(file_path: str) -> Dict:
    """Load complete results including raw data if available."""
    results = torch.load(file_path, map_location='cpu', weights_only=False)
    return results


def analyze_cumulative_impact(results: Dict) -> pd.DataFrame:
    """Analyze the cumulative impact of features by frequency bins."""
    
    under_attr = results['results_by_type']['under_attributed']
    over_attr = results['results_by_type']['over_attributed']
    
    # Define frequency bins
    freq_bins = [
        (1, 10, "1-10 (very rare)"),
        (11, 50, "11-50 (rare)"),
        (51, 100, "51-100 (uncommon)"),
        (101, 500, "101-500 (common)"),
        (501, 1000, "501-1000 (very common)"),
        (1001, 10000, "1000+ (ubiquitous)")
    ]
    
    analysis_data = []
    
    for category, features in [('under_attributed', under_attr), ('over_attributed', over_attr)]:
        for min_occ, max_occ, bin_label in freq_bins:
            bin_features = {
                fid: fdata for fid, fdata in features.items()
                if min_occ <= fdata['n_occurrences'] <= max_occ
            }
            
            if not bin_features:
                continue
            
            # Calculate statistics for this bin
            n_features = len(bin_features)
            total_occurrences = sum(f['n_occurrences'] for f in bin_features.values())
            
            # Impact metrics
            log_ratios = [f['mean_log_ratio'] for f in bin_features.values()]
            mean_ratio = np.mean(log_ratios)
            median_ratio = np.median(log_ratios)
            std_ratio = np.std(log_ratios)
            
            # Weighted impact (occurrences * |log_ratio|)
            weighted_impacts = [
                f['n_occurrences'] * abs(f['mean_log_ratio']) 
                for f in bin_features.values()
            ]
            total_weighted_impact = sum(weighted_impacts)
            
            # Class distribution
            class_counts = defaultdict(int)
            for f in bin_features.values():
                class_counts[f.get('dominant_class', 'unknown')] += 1
            
            # Features with extreme ratios (|ratio| > 2)
            extreme_features = sum(1 for r in log_ratios if abs(r) > 2.0)
            extreme_pct = 100 * extreme_features / n_features if n_features > 0 else 0
            
            analysis_data.append({
                'category': category,
                'freq_bin': bin_label,
                'min_occ': min_occ,
                'max_occ': max_occ,
                'n_features': n_features,
                'total_occurrences': total_occurrences,
                'mean_log_ratio': mean_ratio,
                'median_log_ratio': median_ratio,
                'std_log_ratio': std_ratio,
                'total_weighted_impact': total_weighted_impact,
                'extreme_features': extreme_features,
                'extreme_pct': extreme_pct,
                'dominant_class': max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else 'unknown'
            })
    
    return pd.DataFrame(analysis_data)


def analyze_middle_frequency_features(results: Dict, min_occ: int = 10, max_occ: int = 100):
    """Deep dive into features with moderate frequency."""
    
    print(f"\n{'='*80}")
    print(f"DEEP ANALYSIS: FEATURES WITH {min_occ}-{max_occ} OCCURRENCES")
    print('='*80)
    
    for category_name, features in results['results_by_type'].items():
        # Filter to target frequency range
        mid_freq_features = {
            fid: fdata for fid, fdata in features.items()
            if min_occ <= fdata['n_occurrences'] <= max_occ
        }
        
        if not mid_freq_features:
            continue
        
        print(f"\n{category_name.upper().replace('_', ' ')}:")
        print("-"*40)
        print(f"Total features in range: {len(mid_freq_features)}")
        
        # Analyze by log ratio magnitude
        ratio_bins = [
            (0, 0.5, "minimal"),
            (0.5, 1.0, "mild"),
            (1.0, 1.5, "moderate"),
            (1.5, 2.0, "strong"),
            (2.0, 3.0, "severe"),
            (3.0, float('inf'), "extreme")
        ]
        
        print("\nDistribution by misalignment severity:")
        for min_r, max_r, severity in ratio_bins:
            count = sum(1 for f in mid_freq_features.values() 
                       if min_r <= abs(f['mean_log_ratio']) < max_r)
            pct = 100 * count / len(mid_freq_features) if mid_freq_features else 0
            
            # Get example features
            examples = [
                (fid, f) for fid, f in mid_freq_features.items()
                if min_r <= abs(f['mean_log_ratio']) < max_r
            ]
            examples.sort(key=lambda x: abs(x[1]['mean_log_ratio']), reverse=True)
            
            print(f"  {severity:10s} ({min_r:.1f}-{max_r:.1f}): {count:4d} features ({pct:5.1f}%)")
            
            if examples and severity in ['severe', 'extreme']:
                print(f"    Top examples:")
                for fid, f in examples[:3]:
                    print(f"      Feature {fid:5d}: ratio={f['mean_log_ratio']:+.3f}, "
                          f"occ={f['n_occurrences']:3d}, class={f.get('dominant_class', 'unknown')}")
        
        # Consistency analysis
        print("\nConsistency metrics (std of log ratio):")
        std_ratios = [f.get('std_log_ratio', 0) for f in mid_freq_features.values()]
        print(f"  Mean std: {np.mean(std_ratios):.3f}")
        print(f"  Median std: {np.median(std_ratios):.3f}")
        
        # High consistency features (low std)
        consistent_features = [
            (fid, f) for fid, f in mid_freq_features.items()
            if f.get('std_log_ratio', 1) < 0.5 and abs(f['mean_log_ratio']) > 1.5
        ]
        
        if consistent_features:
            print(f"\nHighly consistent features with strong misalignment (std<0.5, |ratio|>1.5):")
            print(f"  Found {len(consistent_features)} features")
            consistent_features.sort(key=lambda x: abs(x[1]['mean_log_ratio']), reverse=True)
            for fid, f in consistent_features[:5]:
                print(f"    Feature {fid:5d}: ratio={f['mean_log_ratio']:+.3f}, "
                      f"std={f.get('std_log_ratio', 0):.3f}, "
                      f"occ={f['n_occurrences']:3d}, "
                      f"class={f.get('dominant_class', 'unknown')}")


def analyze_feature_clustering(results: Dict):
    """Check if rare features tend to appear together in specific images."""
    
    print(f"\n{'='*80}")
    print("FEATURE CO-OCCURRENCE ANALYSIS")
    print('='*80)
    
    # This would require the raw_results which might not be saved
    # We'll analyze based on the aggregated data instead
    
    for category_name, features in results['results_by_type'].items():
        print(f"\n{category_name.upper().replace('_', ' ')}:")
        print("-"*40)
        
        # Group features by occurrence ranges
        rare_features = [fid for fid, f in features.items() if f['n_occurrences'] <= 10]
        uncommon_features = [fid for fid, f in features.items() if 10 < f['n_occurrences'] <= 100]
        common_features = [fid for fid, f in features.items() if f['n_occurrences'] > 100]
        
        print(f"Feature distribution:")
        print(f"  Rare (≤10): {len(rare_features)} features")
        print(f"  Uncommon (11-100): {len(uncommon_features)} features")
        print(f"  Common (>100): {len(common_features)} features")
        
        # Analyze class specificity
        for freq_label, feature_list in [
            ("Rare", rare_features),
            ("Uncommon", uncommon_features),
            ("Common", common_features)
        ]:
            if not feature_list:
                continue
                
            class_dominant = defaultdict(int)
            for fid in feature_list:
                dom_class = features[fid].get('dominant_class', 'unknown')
                class_dominant[dom_class] += 1
            
            print(f"\n  {freq_label} features by dominant class:")
            for cls, count in sorted(class_dominant.items(), key=lambda x: x[1], reverse=True):
                pct = 100 * count / len(feature_list)
                print(f"    {cls:10s}: {count:4d} ({pct:5.1f}%)")


def create_rare_feature_visualizations(results: Dict, save_dir: str = "analysis_plots"):
    """Create visualizations focusing on rare features."""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Prepare cumulative impact data
    impact_df = analyze_cumulative_impact(results)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Cumulative impact by frequency bin
    ax1 = plt.subplot(2, 3, 1)
    
    for category in ['under_attributed', 'over_attributed']:
        cat_data = impact_df[impact_df['category'] == category]
        if not cat_data.empty:
            x = range(len(cat_data))
            color = '#ff7f0e' if category == 'under_attributed' else '#2ca02c'
            
            ax1.bar(x, cat_data['total_weighted_impact'], 
                   label=category.replace('_', ' ').title(), 
                   alpha=0.7, color=color)
    
    ax1.set_xlabel('Frequency Bin')
    ax1.set_ylabel('Total Weighted Impact (occurrences × |log ratio|)')
    ax1.set_title('Cumulative Impact by Feature Frequency')
    ax1.set_xticks(range(len(cat_data)))
    ax1.set_xticklabels(cat_data['freq_bin'].values, rotation=45, ha='right')
    ax1.legend()
    ax1.set_yscale('log')
    
    # 2. Feature count vs total occurrences
    ax2 = plt.subplot(2, 3, 2)
    
    for category in ['under_attributed', 'over_attributed']:
        cat_data = impact_df[impact_df['category'] == category]
        if not cat_data.empty:
            color = '#ff7f0e' if category == 'under_attributed' else '#2ca02c'
            ax2.scatter(cat_data['n_features'], cat_data['total_occurrences'],
                       s=cat_data['mean_log_ratio'].abs() * 100,
                       alpha=0.6, color=color,
                       label=category.replace('_', ' ').title())
            
            # Add labels for each point
            for _, row in cat_data.iterrows():
                ax2.annotate(row['freq_bin'].split()[0], 
                           (row['n_features'], row['total_occurrences']),
                           fontsize=8, alpha=0.7)
    
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Total Occurrences')
    ax2.set_title('Features vs Total Appearances\n(bubble size = mean |log ratio|)')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. Percentage of extreme features by bin
    ax3 = plt.subplot(2, 3, 3)
    
    under_data = impact_df[impact_df['category'] == 'under_attributed']
    over_data = impact_df[impact_df['category'] == 'over_attributed']
    
    x = np.arange(len(under_data))
    width = 0.35
    
    if not under_data.empty:
        ax3.bar(x - width/2, under_data['extreme_pct'], width, 
               label='Under-attr', color='#ff7f0e', alpha=0.7)
    if not over_data.empty:
        ax3.bar(x + width/2, over_data['extreme_pct'], width,
               label='Over-attr', color='#2ca02c', alpha=0.7)
    
    ax3.set_xlabel('Frequency Bin')
    ax3.set_ylabel('% Features with |log ratio| > 2')
    ax3.set_title('Extreme Misalignment by Frequency')
    ax3.set_xticks(x)
    ax3.set_xticklabels(under_data['freq_bin'].values, rotation=45, ha='right')
    ax3.legend()
    
    # 4. Focus on 10-100 occurrence range
    ax4 = plt.subplot(2, 3, 4)
    
    under_attr = results['results_by_type']['under_attributed']
    over_attr = results['results_by_type']['over_attributed']
    
    # Filter to 10-100 range
    under_mid = {fid: f for fid, f in under_attr.items() if 10 <= f['n_occurrences'] <= 100}
    over_mid = {fid: f for fid, f in over_attr.items() if 10 <= f['n_occurrences'] <= 100}
    
    # Create 2D histogram
    if under_mid:
        occs = [f['n_occurrences'] for f in under_mid.values()]
        ratios = [f['mean_log_ratio'] for f in under_mid.values()]
        ax4.scatter(occs, ratios, alpha=0.3, s=5, color='#ff7f0e', label='Under-attr')
    
    if over_mid:
        occs = [f['n_occurrences'] for f in over_mid.values()]
        ratios = [f['mean_log_ratio'] for f in over_mid.values()]
        ax4.scatter(occs, ratios, alpha=0.3, s=5, color='#2ca02c', label='Over-attr')
    
    ax4.set_xlabel('Occurrences')
    ax4.set_ylabel('Log Ratio')
    ax4.set_title('10-100 Occurrence Features Detail')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.axhline(y=2, color='gray', linestyle=':', alpha=0.3)
    ax4.axhline(y=-2, color='gray', linestyle=':', alpha=0.3)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Ratio distribution for different frequency bins
    ax5 = plt.subplot(2, 3, 5)
    
    freq_ranges = [(1, 10), (11, 50), (51, 100), (101, 500), (501, 10000)]
    all_features = {**under_attr, **over_attr}
    
    violin_data = []
    labels = []
    
    for min_occ, max_occ in freq_ranges:
        bin_ratios = [
            abs(f['mean_log_ratio']) for f in all_features.values()
            if min_occ <= f['n_occurrences'] <= max_occ
        ]
        if bin_ratios:
            violin_data.append(bin_ratios)
            labels.append(f"{min_occ}-{max_occ}")
    
    if violin_data:
        parts = ax5.violinplot(violin_data, showmeans=True, showmedians=True)
        ax5.set_xlabel('Occurrence Range')
        ax5.set_ylabel('|Log Ratio|')
        ax5.set_title('Ratio Distribution by Frequency')
        ax5.set_xticks(range(1, len(labels) + 1))
        ax5.set_xticklabels(labels, rotation=45)
        ax5.set_ylim(0, 4)
    
    # 6. Cumulative percentage of features and impact
    ax6 = plt.subplot(2, 3, 6)
    
    # Sort all features by occurrence
    all_features_sorted = sorted(all_features.items(), key=lambda x: x[1]['n_occurrences'])
    
    cumsum_features = np.arange(1, len(all_features_sorted) + 1) / len(all_features_sorted) * 100
    cumsum_occurrences = np.cumsum([f[1]['n_occurrences'] for f in all_features_sorted])
    cumsum_occurrences = cumsum_occurrences / cumsum_occurrences[-1] * 100
    
    cumsum_impact = np.cumsum([
        f[1]['n_occurrences'] * abs(f[1]['mean_log_ratio']) 
        for f in all_features_sorted
    ])
    cumsum_impact = cumsum_impact / cumsum_impact[-1] * 100
    
    occurrences = [f[1]['n_occurrences'] for f in all_features_sorted]
    
    ax6.plot(occurrences, cumsum_features, label='% of Features', linewidth=2)
    ax6.plot(occurrences, cumsum_occurrences, label='% of Total Occurrences', linewidth=2)
    ax6.plot(occurrences, cumsum_impact, label='% of Weighted Impact', linewidth=2)
    
    ax6.set_xlabel('Feature Occurrence Threshold')
    ax6.set_ylabel('Cumulative Percentage')
    ax6.set_title('Cumulative Distribution Analysis')
    ax6.set_xscale('log')
    ax6.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='10 occ')
    ax6.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='100 occ')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Rare Feature Impact Analysis', fontsize=16)
    plt.tight_layout()
    
    save_file = save_path / "rare_feature_analysis.png"
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_file}")
    
    plt.show()
    
    return impact_df


def print_summary_insights(impact_df: pd.DataFrame):
    """Print key insights about rare features."""
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS: ARE RARE FEATURES NOISE OR SIGNAL?")
    print('='*80)
    
    # Calculate total impacts
    under_df = impact_df[impact_df['category'] == 'under_attributed']
    over_df = impact_df[impact_df['category'] == 'over_attributed']
    
    for category, df in [('Under-attributed', under_df), ('Over-attributed', over_df)]:
        if df.empty:
            continue
            
        print(f"\n{category.upper()}:")
        print("-"*40)
        
        total_impact = df['total_weighted_impact'].sum()
        
        # Calculate percentage contribution of each bin
        print("\nContribution to total weighted impact:")
        for _, row in df.iterrows():
            pct = 100 * row['total_weighted_impact'] / total_impact
            print(f"  {row['freq_bin']:25s}: {pct:6.2f}% "
                  f"({row['n_features']:5d} features, "
                  f"{row['total_occurrences']:6d} occurrences)")
        
        # Specific analysis for 10-100 range
        mid_range = df[(df['min_occ'] >= 10) & (df['max_occ'] <= 100)]
        if not mid_range.empty:
            mid_impact_pct = 100 * mid_range['total_weighted_impact'].sum() / total_impact
            mid_features_pct = 100 * mid_range['n_features'].sum() / df['n_features'].sum()
            
            print(f"\n10-100 occurrence features specifically:")
            print(f"  - Represent {mid_features_pct:.1f}% of all features")
            print(f"  - Contribute {mid_impact_pct:.1f}% of total weighted impact")
            print(f"  - Average |log ratio|: {mid_range['mean_log_ratio'].abs().mean():.3f}")
            print(f"  - {mid_range['extreme_pct'].mean():.1f}% have |ratio| > 2")


def main():
    """Main analysis function."""
    
    file_path = "results/saco_features_direct_l6.pt"
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"Analyzing rare features in: {file_path}")
    
    # Load results
    results = load_full_results(file_path)
    
    # Analyze cumulative impact
    impact_df = analyze_cumulative_impact(results)
    
    # Print summary insights
    print_summary_insights(impact_df)
    
    # Deep dive into middle frequency features
    analyze_middle_frequency_features(results, min_occ=10, max_occ=100)
    
    # Analyze clustering patterns
    analyze_feature_clustering(results)
    
    # Create visualizations
    impact_df = create_rare_feature_visualizations(results)
    
    # Save detailed analysis
    impact_df.to_csv("rare_feature_impact_analysis.csv", index=False)
    print(f"\nDetailed impact analysis saved to: rare_feature_impact_analysis.csv")


if __name__ == "__main__":
    main()