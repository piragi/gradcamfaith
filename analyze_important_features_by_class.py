"""
Detailed Analysis of High-Impact Features (10-100 occurrences, |ratio| > 1.0)
Focus on understanding class distribution and patterns.
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


def load_and_filter_features(file_path: str, min_occ: int = 10, max_occ: int = 100, 
                             min_ratio: float = 1.0) -> Tuple[Dict, Dict]:
    """Load results and filter to high-impact features in target range."""
    
    results = torch.load(file_path, map_location='cpu', weights_only=False)
    
    under_attr = results['results_by_type']['under_attributed']
    over_attr = results['results_by_type']['over_attributed']
    
    # Filter to target features
    filtered_under = {
        fid: fdata for fid, fdata in under_attr.items()
        if min_occ <= fdata['n_occurrences'] <= max_occ and fdata['mean_log_ratio'] > min_ratio
    }
    
    filtered_over = {
        fid: fdata for fid, fdata in over_attr.items()
        if min_occ <= fdata['n_occurrences'] <= max_occ and fdata['mean_log_ratio'] < -min_ratio
    }
    
    return filtered_under, filtered_over


def analyze_class_distribution(features: Dict, category: str) -> pd.DataFrame:
    """Analyze how features are distributed across classes."""
    
    data = []
    
    for fid, fdata in features.items():
        class_dist = fdata.get('class_distribution', {})
        dominant_class = fdata.get('dominant_class', 'unknown')
        total_occ = fdata['n_occurrences']
        
        # Calculate class percentages
        class_percentages = {}
        for cls in ['COVID-19', 'Non-COVID', 'Normal']:
            count = class_dist.get(cls, 0)
            class_percentages[cls] = 100 * count / total_occ if total_occ > 0 else 0
        
        # Determine if feature is class-specific
        max_pct = max(class_percentages.values())
        is_specific = max_pct > 70  # Consider specific if >70% in one class
        
        data.append({
            'feature_id': fid,
            'category': category,
            'log_ratio': fdata['mean_log_ratio'],
            'abs_log_ratio': abs(fdata['mean_log_ratio']),
            'n_occurrences': total_occ,
            'dominant_class': dominant_class,
            'dominant_pct': max_pct,
            'is_class_specific': is_specific,
            'covid_pct': class_percentages.get('COVID-19', 0),
            'noncovid_pct': class_percentages.get('Non-COVID', 0),
            'normal_pct': class_percentages.get('Normal', 0),
            'covid_count': class_dist.get('COVID-19', 0),
            'noncovid_count': class_dist.get('Non-COVID', 0),
            'normal_count': class_dist.get('Normal', 0),
            'std_log_ratio': fdata.get('std_log_ratio', 0),
            'avg_n_patches': fdata.get('avg_n_patches', 0)
        })
    
    return pd.DataFrame(data)


def print_detailed_class_analysis(under_df: pd.DataFrame, over_df: pd.DataFrame):
    """Print comprehensive analysis of class distributions."""
    
    print("\n" + "="*80)
    print("HIGH-IMPACT FEATURES ANALYSIS (10-100 occurrences, |ratio| > 1.0)")
    print("="*80)
    
    # Overall statistics
    print("\nOVERALL STATISTICS:")
    print("-"*40)
    print(f"Under-attributed features: {len(under_df)}")
    print(f"Over-attributed features: {len(over_df)}")
    print(f"Total high-impact features: {len(under_df) + len(over_df)}")
    
    # Analyze each category
    for category, df in [('UNDER-ATTRIBUTED', under_df), ('OVER-ATTRIBUTED', over_df)]:
        if df.empty:
            continue
            
        print(f"\n\n{category} FEATURES:")
        print("="*60)
        
        # Basic stats
        print(f"\nTotal features: {len(df)}")
        print(f"Mean |log ratio|: {df['abs_log_ratio'].mean():.3f}")
        print(f"Mean occurrences: {df['n_occurrences'].mean():.1f}")
        
        # Class dominance analysis
        print("\n1. DOMINANT CLASS DISTRIBUTION:")
        print("-"*40)
        class_counts = df['dominant_class'].value_counts()
        for cls, count in class_counts.items():
            pct = 100 * count / len(df)
            mean_ratio = df[df['dominant_class'] == cls]['abs_log_ratio'].mean()
            mean_occ = df[df['dominant_class'] == cls]['n_occurrences'].mean()
            print(f"  {cls:10s}: {count:4d} features ({pct:5.1f}%), "
                  f"avg |ratio|={mean_ratio:.3f}, avg occ={mean_occ:.1f}")
        
        # Class specificity analysis
        print("\n2. CLASS SPECIFICITY (>70% in one class):")
        print("-"*40)
        specific_features = df[df['is_class_specific']]
        print(f"Class-specific features: {len(specific_features)} ({100*len(specific_features)/len(df):.1f}%)")
        
        if not specific_features.empty:
            for cls in ['COVID-19', 'Non-COVID', 'Normal']:
                cls_specific = specific_features[specific_features['dominant_class'] == cls]
                if not cls_specific.empty:
                    print(f"\n  {cls}-specific features: {len(cls_specific)}")
                    print(f"    Mean dominance: {cls_specific['dominant_pct'].mean():.1f}%")
                    print(f"    Mean |ratio|: {cls_specific['abs_log_ratio'].mean():.3f}")
                    
                    # Top examples
                    top_examples = cls_specific.nlargest(3, 'abs_log_ratio')
                    if not top_examples.empty:
                        print(f"    Top examples:")
                        for _, row in top_examples.iterrows():
                            print(f"      Feature {int(row['feature_id']):5d}: "
                                  f"ratio={row['log_ratio']:+.3f}, "
                                  f"dominance={row['dominant_pct']:.1f}%, "
                                  f"occ={int(row['n_occurrences'])}")
        
        # Mixed features analysis
        print("\n3. MIXED FEATURES (no class >70%):")
        print("-"*40)
        mixed_features = df[~df['is_class_specific']]
        print(f"Mixed features: {len(mixed_features)} ({100*len(mixed_features)/len(df):.1f}%)")
        
        if not mixed_features.empty:
            print(f"  Mean |ratio|: {mixed_features['abs_log_ratio'].mean():.3f}")
            print(f"  Distribution across classes:")
            print(f"    COVID-19: {mixed_features['covid_pct'].mean():.1f}% avg presence")
            print(f"    Non-COVID: {mixed_features['noncovid_pct'].mean():.1f}% avg presence")
            print(f"    Normal: {mixed_features['normal_pct'].mean():.1f}% avg presence")
        
        # Ratio distribution by class
        print("\n4. LOG RATIO DISTRIBUTION BY DOMINANT CLASS:")
        print("-"*40)
        ratio_bins = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, float('inf'))]
        
        for cls in ['COVID-19', 'Non-COVID', 'Normal']:
            cls_features = df[df['dominant_class'] == cls]
            if cls_features.empty:
                continue
                
            print(f"\n  {cls}:")
            for min_r, max_r in ratio_bins:
                count = len(cls_features[(cls_features['abs_log_ratio'] >= min_r) & 
                                        (cls_features['abs_log_ratio'] < max_r)])
                if count > 0:
                    pct = 100 * count / len(cls_features)
                    print(f"    |ratio| {min_r:.1f}-{max_r:.1f}: {count:3d} ({pct:5.1f}%)")
        
        # Cross-class appearance analysis
        print("\n5. CROSS-CLASS APPEARANCE PATTERNS:")
        print("-"*40)
        
        # Features appearing in all three classes
        all_classes = df[(df['covid_count'] > 0) & 
                        (df['noncovid_count'] > 0) & 
                        (df['normal_count'] > 0)]
        print(f"Features in all 3 classes: {len(all_classes)} ({100*len(all_classes)/len(df):.1f}%)")
        
        # Features in exactly two classes
        two_classes = df[((df['covid_count'] > 0).astype(int) + 
                         (df['noncovid_count'] > 0).astype(int) + 
                         (df['normal_count'] > 0).astype(int)) == 2]
        print(f"Features in exactly 2 classes: {len(two_classes)} ({100*len(two_classes)/len(df):.1f}%)")
        
        # Features in only one class
        one_class = df[((df['covid_count'] > 0).astype(int) + 
                       (df['noncovid_count'] > 0).astype(int) + 
                       (df['normal_count'] > 0).astype(int)) == 1]
        print(f"Features in only 1 class: {len(one_class)} ({100*len(one_class)/len(df):.1f}%)")


def create_class_distribution_visualizations(under_df: pd.DataFrame, over_df: pd.DataFrame, 
                                            save_dir: str = "analysis_plots"):
    """Create comprehensive visualizations of class distributions."""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Class distribution comparison
    ax1 = plt.subplot(3, 3, 1)
    
    classes = ['COVID-19', 'Non-COVID', 'Normal']
    x = np.arange(len(classes))
    width = 0.35
    
    under_counts = [len(under_df[under_df['dominant_class'] == cls]) for cls in classes]
    over_counts = [len(over_df[over_df['dominant_class'] == cls]) for cls in classes]
    
    bars1 = ax1.bar(x - width/2, under_counts, width, label='Under-attr', 
                    color='#ff7f0e', alpha=0.7)
    bars2 = ax1.bar(x + width/2, over_counts, width, label='Over-attr', 
                    color='#2ca02c', alpha=0.7)
    
    ax1.set_xlabel('Dominant Class')
    ax1.set_ylabel('Number of Features')
    ax1.set_title('Feature Count by Dominant Class\n(10-100 occ, |ratio| > 1.0)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 2. Class specificity pie charts
    for idx, (category, df, color) in enumerate([
        ('Under-attributed', under_df, '#ff7f0e'),
        ('Over-attributed', over_df, '#2ca02c')
    ]):
        ax = plt.subplot(3, 3, 2 + idx)
        
        specific = len(df[df['is_class_specific']])
        mixed = len(df[~df['is_class_specific']])
        
        sizes = [specific, mixed]
        labels = [f'Class-specific\n({specific})', f'Mixed\n({mixed})']
        colors = [color, 'lightgray']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{category}\nClass Specificity')
    
    # 3. Ratio distribution by class (violin plot)
    ax3 = plt.subplot(3, 3, 4)
    
    df_combined = pd.concat([under_df, over_df])
    
    if not df_combined.empty:
        sns.violinplot(data=df_combined, x='dominant_class', y='abs_log_ratio', 
                      hue='category', split=True, ax=ax3, palette=['#ff7f0e', '#2ca02c'])
        ax3.set_xlabel('Dominant Class')
        ax3.set_ylabel('|Log Ratio|')
        ax3.set_title('Ratio Distribution by Class')
        ax3.set_ylim(1, 4)
    
    # 4. Heatmap of class percentages
    ax4 = plt.subplot(3, 3, 5)
    
    # Calculate mean percentages for each category and class
    heatmap_data = []
    for category, df in [('Under-attr', under_df), ('Over-attr', over_df)]:
        row = []
        for cls in ['COVID-19', 'Non-COVID', 'Normal']:
            cls_features = df[df['dominant_class'] == cls]
            if not cls_features.empty:
                # Mean percentage of occurrences in each class
                row.append([
                    cls_features['covid_pct'].mean(),
                    cls_features['noncovid_pct'].mean(),
                    cls_features['normal_pct'].mean()
                ])
            else:
                row.append([0, 0, 0])
        heatmap_data.append(row)
    
    # Reshape for heatmap
    heatmap_array = np.array(heatmap_data).reshape(2, 9)
    
    im = ax4.imshow(heatmap_array, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(9))
    ax4.set_xticklabels(['COVID', 'Non-COVID', 'Normal'] * 3, rotation=45, ha='right')
    ax4.set_yticks(range(2))
    ax4.set_yticklabels(['Under-attr', 'Over-attr'])
    ax4.set_title('Mean Class Distribution %\n(columns grouped by dominant class)')
    
    # Add text annotations
    for i in range(2):
        for j in range(9):
            text = ax4.text(j, i, f'{heatmap_array[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax4, label='Mean %')
    
    # 5. Scatter plot: occurrences vs ratio, colored by class
    ax5 = plt.subplot(3, 3, 6)
    
    for cls, marker in zip(['COVID-19', 'Non-COVID', 'Normal'], ['o', 's', '^']):
        under_cls = under_df[under_df['dominant_class'] == cls]
        over_cls = over_df[over_df['dominant_class'] == cls]
        
        if not under_cls.empty:
            ax5.scatter(under_cls['n_occurrences'], under_cls['log_ratio'],
                       alpha=0.5, marker=marker, s=30, label=f'Under-{cls}')
        if not over_cls.empty:
            ax5.scatter(over_cls['n_occurrences'], over_cls['log_ratio'],
                       alpha=0.5, marker=marker, s=30, label=f'Over-{cls}')
    
    ax5.set_xlabel('Number of Occurrences')
    ax5.set_ylabel('Log Ratio')
    ax5.set_title('Features by Class and Occurrences')
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax5.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    ax5.axhline(y=-1, color='gray', linestyle=':', alpha=0.3)
    ax5.legend(fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)
    
    # 6. Class dominance distribution
    ax6 = plt.subplot(3, 3, 7)
    
    dominance_bins = [70, 80, 90, 100]
    bin_labels = ['70-80%', '80-90%', '90-100%']
    
    under_binned = pd.cut(under_df['dominant_pct'], bins=[70, 80, 90, 100], 
                          labels=bin_labels, include_lowest=True)
    over_binned = pd.cut(over_df['dominant_pct'], bins=[70, 80, 90, 100], 
                        labels=bin_labels, include_lowest=True)
    
    under_counts = under_binned.value_counts().sort_index()
    over_counts = over_binned.value_counts().sort_index()
    
    x = np.arange(len(bin_labels))
    width = 0.35
    
    ax6.bar(x - width/2, under_counts, width, label='Under-attr', 
           color='#ff7f0e', alpha=0.7)
    ax6.bar(x + width/2, over_counts, width, label='Over-attr', 
           color='#2ca02c', alpha=0.7)
    
    ax6.set_xlabel('Dominance Level')
    ax6.set_ylabel('Number of Features')
    ax6.set_title('Class Dominance Distribution\n(for class-specific features)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(bin_labels)
    ax6.legend()
    
    # 7. Cross-class appearance patterns
    ax7 = plt.subplot(3, 3, 8)
    
    appearance_data = []
    for category, df in [('Under-attr', under_df), ('Over-attr', over_df)]:
        n_classes = ((df['covid_count'] > 0).astype(int) + 
                    (df['noncovid_count'] > 0).astype(int) + 
                    (df['normal_count'] > 0).astype(int))
        
        one_class = len(df[n_classes == 1])
        two_classes = len(df[n_classes == 2])
        three_classes = len(df[n_classes == 3])
        
        appearance_data.append([one_class, two_classes, three_classes])
    
    x = np.arange(3)
    width = 0.35
    labels = ['1 class only', '2 classes', 'All 3 classes']
    
    ax7.bar(x - width/2, appearance_data[0], width, label='Under-attr', 
           color='#ff7f0e', alpha=0.7)
    ax7.bar(x + width/2, appearance_data[1], width, label='Over-attr', 
           color='#2ca02c', alpha=0.7)
    
    ax7.set_xlabel('Cross-class Appearance')
    ax7.set_ylabel('Number of Features')
    ax7.set_title('Features by Cross-class Appearance')
    ax7.set_xticks(x)
    ax7.set_xticklabels(labels)
    ax7.legend()
    
    # 8. Top features table
    ax8 = plt.subplot(3, 3, 9)
    ax8.axis('tight')
    ax8.axis('off')
    
    # Get top 5 features from each category
    top_under = under_df.nlargest(5, 'abs_log_ratio')[['feature_id', 'log_ratio', 
                                                        'n_occurrences', 'dominant_class']]
    top_over = over_df.nlargest(5, 'abs_log_ratio')[['feature_id', 'log_ratio', 
                                                      'n_occurrences', 'dominant_class']]
    
    # Create table data
    table_data = []
    table_data.append(['Under-attributed', '', 'Over-attributed', ''])
    table_data.append(['Feature', 'Ratio/Occ/Class', 'Feature', 'Ratio/Occ/Class'])
    
    for i in range(5):
        row = []
        if i < len(top_under):
            r = top_under.iloc[i]
            row.extend([f"{int(r['feature_id'])}", 
                       f"{r['log_ratio']:.2f}/{int(r['n_occurrences'])}/{r['dominant_class'][:3]}"])
        else:
            row.extend(['', ''])
        
        if i < len(top_over):
            r = top_over.iloc[i]
            row.extend([f"{int(r['feature_id'])}", 
                       f"{r['log_ratio']:.2f}/{int(r['n_occurrences'])}/{r['dominant_class'][:3]}"])
        else:
            row.extend(['', ''])
        
        table_data.append(row)
    
    table = ax8.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax8.set_title('Top 5 Features by |Log Ratio|', pad=20)
    
    plt.suptitle('High-Impact Features Class Distribution Analysis\n(10-100 occurrences, |ratio| > 1.0)', 
                fontsize=16)
    plt.tight_layout()
    
    save_file = save_path / "class_distribution_analysis.png"
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_file}")
    
    plt.show()


def analyze_class_imbalance(under_df: pd.DataFrame, over_df: pd.DataFrame):
    """Analyze and quantify class imbalance in features."""
    
    print("\n" + "="*80)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*80)
    
    # Calculate class representation indices
    total_under = len(under_df)
    total_over = len(over_df)
    
    print("\nCLASS REPRESENTATION INDEX (actual/expected ratio):")
    print("Expected: 33.3% per class if balanced")
    print("-"*40)
    
    for category, df, total in [('Under-attributed', under_df, total_under), 
                                ('Over-attributed', over_df, total_over)]:
        if total == 0:
            continue
            
        print(f"\n{category}:")
        class_counts = df['dominant_class'].value_counts()
        
        for cls in ['COVID-19', 'Non-COVID', 'Normal']:
            count = class_counts.get(cls, 0)
            actual_pct = 100 * count / total
            expected_pct = 33.3
            ratio = actual_pct / expected_pct
            
            status = "over-represented" if ratio > 1.2 else "under-represented" if ratio < 0.8 else "balanced"
            print(f"  {cls:10s}: {actual_pct:5.1f}% (ratio: {ratio:.2f}) - {status}")
    
    # Chi-square test for uniformity
    from scipy.stats import chisquare
    
    print("\nSTATISTICAL SIGNIFICANCE (Chi-square test):")
    print("-"*40)
    
    for category, df in [('Under-attributed', under_df), ('Over-attributed', over_df)]:
        if df.empty:
            continue
            
        class_counts = df['dominant_class'].value_counts()
        observed = [class_counts.get(cls, 0) for cls in ['COVID-19', 'Non-COVID', 'Normal']]
        expected = [len(df) / 3] * 3
        
        chi2, p_value = chisquare(observed, expected)
        
        print(f"\n{category}:")
        print(f"  Chi-square statistic: {chi2:.2f}")
        print(f"  P-value: {p_value:.4e}")
        print(f"  Significant imbalance: {'Yes' if p_value < 0.001 else 'No'}")


def main():
    """Main analysis function."""
    
    file_path = "results/saco_features_direct_l5.pt"
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"Analyzing high-impact features from: {file_path}")
    
    # Load and filter features
    filtered_under, filtered_over = load_and_filter_features(
        file_path, min_occ=10, max_occ=100, min_ratio=1.0
    )
    
    print(f"\nFiltered to 10-100 occurrences with |ratio| > 1.0:")
    print(f"  Under-attributed: {len(filtered_under)} features")
    print(f"  Over-attributed: {len(filtered_over)} features")
    
    # Create dataframes for analysis
    under_df = analyze_class_distribution(filtered_under, 'under_attributed')
    over_df = analyze_class_distribution(filtered_over, 'over_attributed')
    
    # Print detailed analysis
    print_detailed_class_analysis(under_df, over_df)
    
    # Analyze class imbalance
    analyze_class_imbalance(under_df, over_df)
    
    # Create visualizations
    create_class_distribution_visualizations(under_df, over_df)
    
    # Save detailed data
    combined_df = pd.concat([under_df, over_df])
    combined_df.to_csv("high_impact_features_class_analysis.csv", index=False)
    print(f"\nDetailed data saved to: high_impact_features_class_analysis.csv")


if __name__ == "__main__":
    main()