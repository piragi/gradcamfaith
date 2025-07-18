import logging
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats

warnings.filterwarnings('ignore')
# Suppress matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

from transmm_sfaf import SAE_CONFIG
# Import IDX2CLS for class names
from vit.model import IDX2CLS

# SAE Configuration from transmm_sfaf.py


class CorrelationDictAnalyzer:
    """Comprehensive analyzer for correlation dictionaries from corr_dict.py"""

    def __init__(self, save_plots: bool = True, output_dir: str = "./analysis_results"):
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Colors for consistent plotting
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

    def load_correlation_dict(self, dict_path: str) -> Dict[str, Any]:
        """Load correlation dictionary from file"""
        if not Path(dict_path).exists():
            raise FileNotFoundError(f"Dictionary not found at {dict_path}")

        dict_data = torch.load(dict_path, weights_only=False)
        print(f"Loaded dictionary with {len(dict_data['feature_stats'])} features")
        return dict_data

    def extract_class_data(self, dict_data: Dict[str, Any]) -> Dict[str, List]:
        """Extract class-specific data from dictionary"""
        class_correlations = defaultdict(list)
        class_steerabilities = defaultdict(list)
        class_occurrences = defaultdict(list)
        feature_class_counts = defaultdict(dict)

        for fid, stats in dict_data['feature_stats'].items():
            class_mean_pfac = stats['class_mean_pfac']
            class_mean_steer = stats.get('class_mean_steerability', {})
            class_count_map = stats['class_count_map']

            for cls_id, corr_val in class_mean_pfac.items():
                class_correlations[cls_id].append(corr_val)
                class_occurrences[cls_id].append(class_count_map.get(cls_id, 0))
                feature_class_counts[fid][cls_id] = class_count_map.get(cls_id, 0)

                # Add steerability if available
                steer_val = class_mean_steer.get(cls_id, 0.0)
                class_steerabilities[cls_id].append(steer_val)

        return {
            'class_correlations': dict(class_correlations),
            'class_steerabilities': dict(class_steerabilities),
            'class_occurrences': dict(class_occurrences),
            'feature_class_counts': dict(feature_class_counts)
        }

    def analyze_correlation_distributions(self, class_data: Dict[str, Any], layer_idx: int = None) -> Dict[str, Any]:
        """Analyze correlation distributions per class"""
        class_correlations = class_data['class_correlations']
        analysis_results = {}

        for cls_id, correlations in class_correlations.items():
            corr_array = np.array(correlations)

            analysis_results[cls_id] = {
                'mean': np.mean(corr_array),
                'std': np.std(corr_array),
                'median': np.median(corr_array),
                'min': np.min(corr_array),
                'max': np.max(corr_array),
                'q25': np.percentile(corr_array, 25),
                'q75': np.percentile(corr_array, 75),
                'count': len(corr_array),
                'skewness': stats.skew(corr_array),
                'kurtosis': stats.kurtosis(corr_array)
            }

        # Create summary DataFrame
        df = pd.DataFrame(analysis_results).T
        df['class_name'] = [IDX2CLS.get(cls_id, f"Class_{cls_id}") for cls_id in df.index]

        print(f"\nCorrelation Distribution Analysis {'for Layer ' + str(layer_idx) if layer_idx else ''}:")
        print("=" * 70)
        print(df.round(4))

        return analysis_results

    def analyze_occurrence_distributions(self, class_data: Dict[str, Any], layer_idx: int = None) -> Dict[str, Any]:
        """Analyze occurrence distributions per class"""
        class_occurrences = class_data['class_occurrences']
        analysis_results = {}

        for cls_id, occurrences in class_occurrences.items():
            occ_array = np.array(occurrences)

            analysis_results[cls_id] = {
                'mean': np.mean(occ_array),
                'std': np.std(occ_array),
                'median': np.median(occ_array),
                'min': np.min(occ_array),
                'max': np.max(occ_array),
                'total': np.sum(occ_array),
                'count': len(occ_array),
                'skewness': stats.skew(occ_array),
                'kurtosis': stats.kurtosis(occ_array)
            }

        # Create summary DataFrame
        df = pd.DataFrame(analysis_results).T
        df['class_name'] = [IDX2CLS.get(cls_id, f"Class_{cls_id}") for cls_id in df.index]

        print(f"\nOccurrence Distribution Analysis {'for Layer ' + str(layer_idx) if layer_idx else ''}:")
        print("=" * 70)
        print(df.round(4))

        return analysis_results

    def create_binning_analysis(self, class_data: Dict[str, Any], n_bins: int = 20) -> Dict[str, Any]:
        """Create binning analysis for correlations and occurrences"""
        class_correlations = class_data['class_correlations']
        class_occurrences = class_data['class_occurrences']

        # Find global ranges for consistent binning
        all_correlations = []
        all_occurrences = []

        for correlations in class_correlations.values():
            all_correlations.extend(correlations)
        for occurrences in class_occurrences.values():
            all_occurrences.extend(occurrences)

        corr_bins = np.linspace(np.min(all_correlations), np.max(all_correlations), n_bins + 1)
        occ_bins = np.linspace(np.min(all_occurrences), np.max(all_occurrences), n_bins + 1)

        binning_results = {
            'correlation_bins': corr_bins,
            'occurrence_bins': occ_bins,
            'class_correlation_histograms': {},
            'class_occurrence_histograms': {}
        }

        # Create histograms for each class
        for cls_id, correlations in class_correlations.items():
            hist, _ = np.histogram(correlations, bins=corr_bins)
            binning_results['class_correlation_histograms'][cls_id] = hist

        for cls_id, occurrences in class_occurrences.items():
            hist, _ = np.histogram(occurrences, bins=occ_bins)
            binning_results['class_occurrence_histograms'][cls_id] = hist

        return binning_results

    def plot_correlation_distributions(self, class_data: Dict[str, Any], layer_idx: int = None):
        """Plot correlation distributions per class"""
        class_correlations = class_data['class_correlations']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f'Correlation Distribution Analysis {"- Layer " + str(layer_idx) if layer_idx else ""}', fontsize=16
        )

        # Box plot
        ax1 = axes[0, 0]
        data_to_plot = []
        labels = []
        for cls_id, correlations in class_correlations.items():
            data_to_plot.append(correlations)
            labels.append(IDX2CLS.get(cls_id, f"Class_{cls_id}"))

        bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
        ax1.set_title('Correlation Distribution by Class')
        ax1.set_ylabel('PFAC Correlation')
        ax1.tick_params(axis='x', rotation=45)

        # Histogram overlay
        ax2 = axes[0, 1]
        for i, (cls_id, correlations) in enumerate(class_correlations.items()):
            ax2.hist(
                correlations, alpha=0.7, bins=20, color=self.colors[i], label=IDX2CLS.get(cls_id, f"Class_{cls_id}")
            )
        ax2.set_title('Correlation Histograms')
        ax2.set_xlabel('PFAC Correlation')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # Violin plot
        ax3 = axes[1, 0]
        parts = ax3.violinplot(data_to_plot, showmeans=True, showmedians=True)
        for pc, color in zip(parts['bodies'], self.colors):
            pc.set_facecolor(color)
        ax3.set_title('Correlation Density by Class')
        ax3.set_ylabel('PFAC Correlation')
        ax3.set_xticks(range(1, len(labels) + 1))
        ax3.set_xticklabels(labels, rotation=45)

        # Cumulative distributions
        ax4 = axes[1, 1]
        for i, (cls_id, correlations) in enumerate(class_correlations.items()):
            sorted_corr = np.sort(correlations)
            y = np.arange(1, len(sorted_corr) + 1) / len(sorted_corr)
            ax4.plot(sorted_corr, y, color=self.colors[i], linewidth=2, label=IDX2CLS.get(cls_id, f"Class_{cls_id}"))
        ax4.set_title('Cumulative Distribution Functions')
        ax4.set_xlabel('PFAC Correlation')
        ax4.set_ylabel('Cumulative Probability')
        ax4.legend()

        plt.tight_layout()

        if self.save_plots:
            suffix = f"_layer{layer_idx}" if layer_idx is not None else ""
            plt.savefig(self.output_dir / f'correlation_distributions{suffix}.png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_occurrence_distributions(self, class_data: Dict[str, Any], layer_idx: int = None):
        """Plot occurrence distributions per class"""
        class_occurrences = class_data['class_occurrences']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f'Occurrence Distribution Analysis {"- Layer " + str(layer_idx) if layer_idx else ""}', fontsize=16
        )

        # Box plot
        ax1 = axes[0, 0]
        data_to_plot = []
        labels = []
        for cls_id, occurrences in class_occurrences.items():
            data_to_plot.append(occurrences)
            labels.append(IDX2CLS.get(cls_id, f"Class_{cls_id}"))

        bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
        ax1.set_title('Occurrence Distribution by Class')
        ax1.set_ylabel('Number of Occurrences')
        ax1.tick_params(axis='x', rotation=45)

        # Histogram overlay
        ax2 = axes[0, 1]
        for i, (cls_id, occurrences) in enumerate(class_occurrences.items()):
            ax2.hist(
                occurrences, alpha=0.7, bins=20, color=self.colors[i], label=IDX2CLS.get(cls_id, f"Class_{cls_id}")
            )
        ax2.set_title('Occurrence Histograms')
        ax2.set_xlabel('Number of Occurrences')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # Bar plot of total occurrences
        ax3 = axes[1, 0]
        total_occurrences = {cls_id: sum(occurrences) for cls_id, occurrences in class_occurrences.items()}
        cls_ids = list(total_occurrences.keys())
        totals = list(total_occurrences.values())
        class_names = [IDX2CLS.get(cls_id, f"Class_{cls_id}") for cls_id in cls_ids]

        bars = ax3.bar(class_names, totals, color=self.colors[:len(cls_ids)])
        ax3.set_title('Total Occurrences by Class')
        ax3.set_ylabel('Total Occurrences')
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, total in zip(bars, totals):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01, f'{total}', ha='center', va='bottom')

        # Log-scale histogram for better visualization
        ax4 = axes[1, 1]
        for i, (cls_id, occurrences) in enumerate(class_occurrences.items()):
            ax4.hist(
                occurrences, alpha=0.7, bins=20, color=self.colors[i], label=IDX2CLS.get(cls_id, f"Class_{cls_id}")
            )
        ax4.set_title('Occurrence Histograms (Log Scale)')
        ax4.set_xlabel('Number of Occurrences')
        ax4.set_ylabel('Frequency')
        ax4.set_yscale('log')
        ax4.legend()

        plt.tight_layout()

        if self.save_plots:
            suffix = f"_layer{layer_idx}" if layer_idx is not None else ""
            plt.savefig(self.output_dir / f'occurrence_distributions{suffix}.png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_binning_analysis(self, binning_results: Dict[str, Any], layer_idx: int = None):
        """Plot binning analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Binning Analysis {"- Layer " + str(layer_idx) if layer_idx else ""}', fontsize=16)

        # Correlation bins heatmap
        ax1 = axes[0, 0]
        corr_hists = binning_results['class_correlation_histograms']
        corr_bins = binning_results['correlation_bins']

        data_matrix = []
        class_names = []
        for cls_id, hist in corr_hists.items():
            data_matrix.append(hist)
            class_names.append(IDX2CLS.get(cls_id, f"Class_{cls_id}"))

        im1 = ax1.imshow(data_matrix, aspect='auto', cmap='viridis')
        ax1.set_title('Correlation Binning Heatmap')
        ax1.set_xlabel('Correlation Bins')
        ax1.set_ylabel('Classes')
        ax1.set_yticks(range(len(class_names)))
        ax1.set_yticklabels(class_names)

        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Count')

        # Occurrence bins heatmap
        ax2 = axes[0, 1]
        occ_hists = binning_results['class_occurrence_histograms']
        occ_bins = binning_results['occurrence_bins']

        data_matrix = []
        for cls_id, hist in occ_hists.items():
            data_matrix.append(hist)

        im2 = ax2.imshow(data_matrix, aspect='auto', cmap='plasma')
        ax2.set_title('Occurrence Binning Heatmap')
        ax2.set_xlabel('Occurrence Bins')
        ax2.set_ylabel('Classes')
        ax2.set_yticks(range(len(class_names)))
        ax2.set_yticklabels(class_names)

        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Count')

        # Bin edge analysis
        ax3 = axes[1, 0]
        bin_centers = (corr_bins[:-1] + corr_bins[1:]) / 2
        for i, (cls_id, hist) in enumerate(corr_hists.items()):
            ax3.plot(bin_centers, hist, 'o-', color=self.colors[i], label=IDX2CLS.get(cls_id, f"Class_{cls_id}"))
        ax3.set_title('Correlation Bin Profiles')
        ax3.set_xlabel('Correlation Bin Centers')
        ax3.set_ylabel('Count')
        ax3.legend()

        # Occurrence bin analysis
        ax4 = axes[1, 1]
        bin_centers = (occ_bins[:-1] + occ_bins[1:]) / 2
        for i, (cls_id, hist) in enumerate(occ_hists.items()):
            ax4.plot(bin_centers, hist, 'o-', color=self.colors[i], label=IDX2CLS.get(cls_id, f"Class_{cls_id}"))
        ax4.set_title('Occurrence Bin Profiles')
        ax4.set_xlabel('Occurrence Bin Centers')
        ax4.set_ylabel('Count')
        ax4.legend()

        plt.tight_layout()

        if self.save_plots:
            suffix = f"_layer{layer_idx}" if layer_idx is not None else ""
            plt.savefig(self.output_dir / f'binning_analysis{suffix}.png', dpi=300, bbox_inches='tight')

        plt.show()

    def analyze_feature_importance(self, dict_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature importance across classes"""
        feature_importance = {}

        for fid, stats in dict_data['feature_stats'].items():
            # Calculate importance metrics
            total_occurrences = stats['occurrences']
            mean_correlation = abs(stats['mean_pfac_corr'])
            class_diversity = len(stats['class_count_map'])

            # Combined importance score
            importance_score = mean_correlation * np.log(total_occurrences + 1) * class_diversity

            feature_importance[fid] = {
                'importance_score': importance_score,
                'mean_correlation': mean_correlation,
                'total_occurrences': total_occurrences,
                'class_diversity': class_diversity,
                'class_specificity': max(stats['class_count_map'].values()) / total_occurrences
            }

        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1]['importance_score'], reverse=True)

        print("\nTop 20 Most Important Features:")
        print("=" * 80)
        print(f"{'Feature ID':<12} {'Importance':<12} {'Mean Corr':<12} {'Occurrences':<12} {'Classes':<8}")
        print("-" * 80)

        for fid, stats in sorted_features[:20]:
            print(
                f"{fid:<12} {stats['importance_score']:<12.4f} {stats['mean_correlation']:<12.4f} "
                f"{stats['total_occurrences']:<12} {stats['class_diversity']:<8}"
            )

        return dict(sorted_features)

    def cross_class_analysis(self, class_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-class analysis"""
        class_correlations = class_data['class_correlations']

        # Compute pairwise correlations between class correlation distributions
        class_pairs = {}
        class_list = list(class_correlations.keys())

        for i, cls1 in enumerate(class_list):
            for j, cls2 in enumerate(class_list[i + 1:], i + 1):
                corr1 = np.array(class_correlations[cls1])
                corr2 = np.array(class_correlations[cls2])

                # Compute various similarity metrics
                if len(corr1) > 1 and len(corr2) > 1:
                    # Statistical tests
                    t_stat, p_value = stats.ttest_ind(corr1, corr2)
                    ks_stat, ks_p_value = stats.ks_2samp(corr1, corr2)

                    pair_key = f"{cls1}_{cls2}"
                    class_pairs[pair_key] = {
                        'mean_diff': np.mean(corr1) - np.mean(corr2),
                        'std_diff': np.std(corr1) - np.std(corr2),
                        't_test_p': p_value,
                        'ks_test_p': ks_p_value,
                        'significant_diff': p_value < 0.05
                    }

        print("\nCross-Class Analysis:")
        print("=" * 70)
        print(f"{'Class Pair':<15} {'Mean Diff':<12} {'T-test p':<12} {'KS-test p':<12} {'Significant'}")
        print("-" * 70)

        for pair, pair_stats in class_pairs.items():
            cls1, cls2 = pair.split('_')
            class_name1 = IDX2CLS.get(int(cls1), f"Class_{cls1}")
            class_name2 = IDX2CLS.get(int(cls2), f"Class_{cls2}")
            pair_name = f"{class_name1[:5]}-{class_name2[:5]}"

            print(
                f"{pair_name:<15} {pair_stats['mean_diff']:<12.4f} {pair_stats['t_test_p']:<12.4f} "
                f"{pair_stats['ks_test_p']:<12.4f} {'Yes' if pair_stats['significant_diff'] else 'No'}"
            )

        return class_pairs

    def analyze_high_occurrence_high_correlation_features(
        self,
        dict_data: Dict[str, Any],
        occurrence_threshold: int = 50,
        correlation_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Analyze features with both high occurrence and high correlation"""
        special_features = {}

        for fid, stats in dict_data['feature_stats'].items():
            total_occurrences = stats['occurrences']
            mean_correlation = abs(stats['mean_pfac_corr'])

            # Check if feature meets both criteria
            if total_occurrences >= occurrence_threshold and mean_correlation >= correlation_threshold:
                class_correlations = stats['class_mean_pfac']
                class_counts = stats['class_count_map']

                # Find dominant class (highest correlation)
                dominant_class = max(class_correlations.items(), key=lambda x: abs(x[1]))

                # Find most frequent class
                most_frequent_class = max(class_counts.items(), key=lambda x: x[1])

                special_features[fid] = {
                    'total_occurrences': total_occurrences,
                    'mean_correlation': mean_correlation,
                    'dominant_class': dominant_class[0],
                    'dominant_class_correlation': dominant_class[1],
                    'most_frequent_class': most_frequent_class[0],
                    'most_frequent_class_count': most_frequent_class[1],
                    'class_correlations': class_correlations,
                    'class_counts': class_counts,
                    'class_diversity': len(class_counts),
                    'class_specificity': most_frequent_class[1] / total_occurrences
                }

        # Sort by combined score (occurrence * correlation)
        sorted_features = sorted(
            special_features.items(), key=lambda x: x[1]['total_occurrences'] * x[1]['mean_correlation'], reverse=True
        )

        print(f"\nHigh Occurrence & High Correlation Features:")
        print(f"(Occurrence >= {occurrence_threshold}, |Correlation| >= {correlation_threshold})")
        print("=" * 120)
        print(
            f"{'Feature':<8} {'Occurrences':<12} {'Mean Corr':<12} {'Dominant Class':<15} {'Dom Corr':<12} {'Most Freq Class':<15} {'Freq Count':<12} {'Specificity':<12}"
        )
        print("-" * 120)

        for fid, feature_stats in sorted_features:
            dom_class_name = IDX2CLS.get(feature_stats['dominant_class'], f"Class_{feature_stats['dominant_class']}")
            freq_class_name = IDX2CLS.get(
                feature_stats['most_frequent_class'], f"Class_{feature_stats['most_frequent_class']}"
            )

            print(
                f"{fid:<8} {feature_stats['total_occurrences']:<12} {feature_stats['mean_correlation']:<12.4f} "
                f"{dom_class_name:<15} {feature_stats['dominant_class_correlation']:<12.4f} "
                f"{freq_class_name:<15} {feature_stats['most_frequent_class_count']:<12} "
                f"{feature_stats['class_specificity']:<12.4f}"
            )

        return dict(sorted_features)

    def analyze_steerability_distributions(self, class_data: Dict[str, Any], layer_idx: int = None) -> Dict[str, Any]:
        """Analyze steerability distributions per class"""
        class_steerabilities = class_data['class_steerabilities']
        analysis_results = {}

        for cls_id, steerabilities in class_steerabilities.items():
            steer_array = np.array(steerabilities)

            analysis_results[cls_id] = {
                'mean': np.mean(steer_array),
                'std': np.std(steer_array),
                'median': np.median(steer_array),
                'min': np.min(steer_array),
                'max': np.max(steer_array),
                'q25': np.percentile(steer_array, 25),
                'q75': np.percentile(steer_array, 75),
                'count': len(steer_array),
                'skewness': stats.skew(steer_array),
                'kurtosis': stats.kurtosis(steer_array)
            }

        # Create summary DataFrame
        df = pd.DataFrame(analysis_results).T
        df['class_name'] = [IDX2CLS.get(cls_id, f"Class_{cls_id}") for cls_id in df.index]

        print(f"\nSteerability Distribution Analysis {'for Layer ' + str(layer_idx) if layer_idx else ''}:")
        print("=" * 70)
        print(df.round(4))

        return analysis_results

    def analyze_top_steerable_features(self, dict_data: Dict[str, Any], top_k: int = 20) -> Dict[str, Any]:
        """Analyze features ranked by steerability"""
        steerable_features = {}

        for fid, stats in dict_data['feature_stats'].items():
            mean_steerability = stats.get('mean_steerability', 0.0)
            total_occurrences = stats['occurrences']
            mean_correlation = abs(stats['mean_pfac_corr'])

            steerable_features[fid] = {
                'mean_steerability': mean_steerability,
                'mean_correlation': mean_correlation,
                'total_occurrences': total_occurrences,
                'class_mean_steerability': stats.get('class_mean_steerability', {}),
                'class_mean_pfac': stats['class_mean_pfac'],
                'class_count_map': stats['class_count_map']
            }

        # Sort by steerability
        sorted_steerable = sorted(steerable_features.items(), key=lambda x: x[1]['mean_steerability'], reverse=True)

        print(f"\nTop {top_k} Most Steerable Features:")
        print("=" * 120)
        print(
            f"{'Feature ID':<12} {'Steerability':<15} {'Correlation':<15} {'Occurrences':<12} {'Dominant Class':<20} {'Dom Steer':<12} {'Dom Corr':<12}"
        )
        print("-" * 120)

        for fid, feature_stats in sorted_steerable[:top_k]:
            # Find dominant class by steerability
            class_steers = feature_stats['class_mean_steerability']
            class_corrs = feature_stats['class_mean_pfac']

            if class_steers:
                dom_steer_class = max(class_steers.items(), key=lambda x: x[1])
                dom_steer_corr = class_corrs.get(dom_steer_class[0], 0.0)
                dom_class_name = IDX2CLS.get(dom_steer_class[0], f"Class_{dom_steer_class[0]}")
            else:
                dom_steer_class = (0, 0.0)
                dom_steer_corr = 0.0
                dom_class_name = "N/A"

            print(
                f"{fid:<12} {feature_stats['mean_steerability']:<15.4f} "
                f"{feature_stats['mean_correlation']:<15.4f} {feature_stats['total_occurrences']:<12} "
                f"{dom_class_name:<20} {dom_steer_class[1]:<12.4f} {dom_steer_corr:<12.4f}"
            )

        return dict(sorted_steerable)

    def analyze_correlation_steerability_overlap(self, dict_data: Dict[str, Any], top_k: int = 50) -> Dict[str, Any]:
        """Analyze overlap between high correlation and high steerability features"""
        features_data = []

        for fid, stats in dict_data['feature_stats'].items():
            mean_steerability = stats.get('mean_steerability', 0.0)
            mean_correlation = abs(stats['mean_pfac_corr'])
            total_occurrences = stats['occurrences']

            features_data.append({
                'feature_id': fid,
                'steerability': mean_steerability,
                'correlation': mean_correlation,
                'occurrences': total_occurrences,
                'stats': stats
            })

        # Sort by different metrics
        top_steerable = sorted(features_data, key=lambda x: x['steerability'], reverse=True)[:top_k]
        top_correlated = sorted(features_data, key=lambda x: x['correlation'], reverse=True)[:top_k]

        # Find overlap
        steerable_ids = {f['feature_id'] for f in top_steerable}
        correlated_ids = {f['feature_id'] for f in top_correlated}

        overlap_ids = steerable_ids & correlated_ids
        steerable_only = steerable_ids - correlated_ids
        correlated_only = correlated_ids - steerable_ids

        # Compute correlation between steerability and correlation scores
        all_steers = [f['steerability'] for f in features_data]
        all_corrs = [f['correlation'] for f in features_data]
        from scipy import stats as scipy_stats
        pearson_corr, p_value = scipy_stats.pearsonr(all_steers, all_corrs)
        spearman_corr, spearman_p = scipy_stats.spearmanr(all_steers, all_corrs)

        results = {
            'top_steerable': top_steerable,
            'top_correlated': top_correlated,
            'overlap_features': [f for f in features_data if f['feature_id'] in overlap_ids],
            'steerable_only': [f for f in features_data if f['feature_id'] in steerable_only],
            'correlated_only': [f for f in features_data if f['feature_id'] in correlated_only],
            'overlap_stats': {
                'overlap_count': len(overlap_ids),
                'steerable_only_count': len(steerable_only),
                'correlated_only_count': len(correlated_only),
                'overlap_percentage': len(overlap_ids) / top_k * 100,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': p_value,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p
            }
        }

        # Print analysis
        print(f"\nCorrelation-Steerability Overlap Analysis (Top {top_k}):")
        print("=" * 80)
        print(f"Features in both top lists (overlap): {len(overlap_ids)} ({len(overlap_ids)/top_k*100:.1f}%)")
        print(f"Features only in top steerable: {len(steerable_only)}")
        print(f"Features only in top correlated: {len(correlated_only)}")
        print(f"Pearson correlation between metrics: {pearson_corr:.4f} (p={p_value:.4f})")
        print(f"Spearman correlation between metrics: {spearman_corr:.4f} (p={spearman_p:.4f})")

        # Print top overlap features
        overlap_features_sorted = sorted(
            results['overlap_features'], key=lambda x: x['steerability'] + x['correlation'], reverse=True
        )
        print(f"\nTop 10 Overlap Features (High in Both Metrics):")
        print(f"{'Feature':<8} {'Steerability':<12} {'Correlation':<12} {'Occurrences':<12} {'Combined':<12}")
        print("-" * 60)

        for feature in overlap_features_sorted[:10]:
            combined_score = feature['steerability'] + feature['correlation']
            print(
                f"{feature['feature_id']:<8} {feature['steerability']:<12.4f} "
                f"{feature['correlation']:<12.4f} {feature['occurrences']:<12} {combined_score:<12.4f}"
            )

        return results

    def plot_steerability_analysis(self, dict_data: Dict[str, Any], class_data: Dict[str, Any], layer_idx: int = None):
        """Create comprehensive steerability visualization"""
        # Get steerability data
        steerable_features = self.analyze_top_steerable_features(dict_data, top_k=10)
        overlap_analysis = self.analyze_correlation_steerability_overlap(dict_data, top_k=50)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f'Steerability Analysis {"- Layer " + str(layer_idx) if layer_idx else ""}', fontsize=16, fontweight='bold'
        )

        # 1. Steerability vs Correlation scatter
        ax1 = axes[0, 0]
        all_steers = [stats.get('mean_steerability', 0.0) for stats in dict_data['feature_stats'].values()]
        all_corrs = [abs(stats['mean_pfac_corr']) for stats in dict_data['feature_stats'].values()]
        all_occs = [stats['occurrences'] for stats in dict_data['feature_stats'].values()]

        scatter = ax1.scatter(all_corrs, all_steers, c=all_occs, cmap='viridis', alpha=0.6, s=50)
        ax1.set_xlabel('Absolute Correlation')
        ax1.set_ylabel('Steerability')
        ax1.set_title('Steerability vs Correlation')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Occurrences')

        # 2. Top steerable features bar plot
        ax2 = axes[0, 1]
        top_15_steerable = list(steerable_features.items())[:15]
        feature_ids = [fid for fid, _ in top_15_steerable]
        steerabilities = [stats['mean_steerability'] for _, stats in top_15_steerable]

        bars = ax2.bar(range(len(feature_ids)), steerabilities, color='red', alpha=0.7)
        ax2.set_title('Top 15 Most Steerable Features')
        ax2.set_ylabel('Mean Steerability')
        ax2.set_xlabel('Feature Rank')
        ax2.set_xticks(range(len(feature_ids)))
        ax2.set_xticklabels([f'F{fid}' for fid in feature_ids], rotation=45)

        # 3. Overlap analysis pie chart
        ax3 = axes[0, 2]
        overlap_stats = overlap_analysis['overlap_stats']
        sizes = [
            overlap_stats['overlap_count'], overlap_stats['steerable_only_count'],
            overlap_stats['correlated_only_count']
        ]
        labels = ['Both High', 'Steerable Only', 'Correlated Only']
        colors = ['gold', 'lightcoral', 'lightblue']

        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Top 50 Features Overlap')

        # 4. Class-wise steerability distributions
        ax4 = axes[1, 0]
        class_steerabilities = class_data['class_steerabilities']
        steer_data = []
        class_names = []

        for cls_id, steers in class_steerabilities.items():
            if steers:  # Only include classes with steerability data
                steer_data.append(steers)
                class_names.append(IDX2CLS.get(cls_id, f"Class_{cls_id}"))

        if steer_data:
            bp = ax4.boxplot(steer_data, labels=class_names, patch_artist=True)
            for patch, color in zip(bp['boxes'], self.colors):
                patch.set_facecolor(color)
            ax4.set_title('Steerability Distribution by Class')
            ax4.set_ylabel('Steerability')
            ax4.tick_params(axis='x', rotation=45)

        # 5. Combined ranking scatter
        ax5 = axes[1, 1]
        combined_scores = [s + c for s, c in zip(all_steers, all_corrs)]
        scatter2 = ax5.scatter(combined_scores, all_occs, c=all_steers, cmap='plasma', alpha=0.6, s=50)
        ax5.set_xlabel('Combined Score (Steerability + Correlation)')
        ax5.set_ylabel('Occurrences')
        ax5.set_title('Combined Score vs Occurrences')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax5, label='Steerability')

        # 6. Correlation coefficient comparison
        ax6 = axes[1, 2]
        corr_types = ['Pearson', 'Spearman']
        corr_values = [
            overlap_analysis['overlap_stats']['pearson_correlation'],
            overlap_analysis['overlap_stats']['spearman_correlation']
        ]
        p_values = [
            overlap_analysis['overlap_stats']['pearson_p_value'], overlap_analysis['overlap_stats']['spearman_p_value']
        ]

        bars = ax6.bar(corr_types, corr_values, color=['blue', 'green'], alpha=0.7)
        ax6.set_title('Correlation Between Metrics')
        ax6.set_ylabel('Correlation Coefficient')
        ax6.set_ylim(0, 1)

        # Add p-values as text
        for bar, p_val in zip(bars, p_values):
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.02,
                f'p={p_val:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.tight_layout()

        if self.save_plots:
            suffix = f"_layer{layer_idx}" if layer_idx is not None else ""
            plt.savefig(self.output_dir / f'steerability_analysis{suffix}.png', dpi=300, bbox_inches='tight')

        plt.show()

    def analyze_top_features_per_class(self, dict_data: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """Analyze top K steerable and correlated features per class"""
        print(f"\n{'='*100}")
        print(f"TOP {top_k} FEATURES PER CLASS ANALYSIS")
        print(f"{'='*100}")

        class_features = defaultdict(list)

        # Collect all feature-class pairs with both metrics
        for fid, stats in dict_data['feature_stats'].items():
            class_mean_pfac = stats['class_mean_pfac']
            class_mean_steer = stats.get('class_mean_steerability', {})
            class_count_map = stats['class_count_map']
            total_occurrences = stats['occurrences']

            for cls_id, corr_val in class_mean_pfac.items():
                steer_val = class_mean_steer.get(cls_id, 0.0)
                class_occurrences = class_count_map.get(cls_id, 0)
                patch_span = stats.get('class_mean_patch_span', {}).get(cls_id, 0.0)
                patch_span_ratio = stats.get('class_mean_patch_span_ratio', {}).get(cls_id, 0.0)

                class_features[cls_id].append({
                    'feature_id': fid,
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val),
                    'steerability': steer_val,
                    'class_occurrences': class_occurrences,
                    'total_occurrences': total_occurrences,
                    'patch_span': patch_span,
                    'patch_span_ratio': patch_span_ratio
                })

        # Analyze each class
        results = {}

        for cls_id, features in class_features.items():
            class_name = IDX2CLS.get(cls_id, f"Class_{cls_id}")

            # Sort by steerability and correlation
            top_steerable = sorted(features, key=lambda x: x['steerability'], reverse=True)[:top_k]
            top_correlated = sorted(features, key=lambda x: x['abs_correlation'], reverse=True)[:top_k]

            # Find overlap
            steerable_ids = {f['feature_id'] for f in top_steerable}
            correlated_ids = {f['feature_id'] for f in top_correlated}
            overlap_ids = steerable_ids & correlated_ids

            results[cls_id] = {
                'class_name': class_name,
                'top_steerable': top_steerable,
                'top_correlated': top_correlated,
                'overlap_features': overlap_ids,
                'overlap_count': len(overlap_ids),
                'total_features': len(features)
            }

            # Print analysis for this class
            print(f"\n🎯 {class_name.upper()} (Class {cls_id}) - {len(features)} total features")
            print(f"   Overlap: {len(overlap_ids)}/{top_k} features appear in both top lists")
            print("-" * 90)

            # Top steerable features
            print(f"\n📈 TOP {top_k} STEERABLE FEATURES:")
            print(
                f"{'Rank':<4} {'Feature':<8} {'Steerability':<12} {'Correlation':<12} {'Class Occ':<10} {'Total Occ':<10}"
            )
            print("-" * 60)
            for i, feature in enumerate(top_steerable, 1):
                print(
                    f"{i:<4} {feature['feature_id']:<8} {feature['steerability']:<12.4f} "
                    f"{feature['correlation']:<12.4f} {feature['class_occurrences']:<10} {feature['total_occurrences']:<10}"
                )

            # Top correlated features
            print(f"\n🔗 TOP {top_k} CORRELATED FEATURES:")
            print(
                f"{'Rank':<4} {'Feature':<8} {'Correlation':<12} {'Steerability':<12} {'Class Occ':<10} {'Total Occ':<10}"
            )
            print("-" * 60)
            for i, feature in enumerate(top_correlated, 1):
                print(
                    f"{i:<4} {feature['feature_id']:<8} {feature['correlation']:<12.4f} "
                    f"{feature['steerability']:<12.4f} {feature['class_occurrences']:<10} {feature['total_occurrences']:<10}"
                )

            # Highlight overlap features
            if overlap_ids:
                print(f"\n⭐ OVERLAP FEATURES (appear in both lists):")
                overlap_features = [f for f in features if f['feature_id'] in overlap_ids]
                overlap_features.sort(key=lambda x: x['steerability'] + x['abs_correlation'], reverse=True)
                print(f"{'Feature':<8} {'Steerability':<12} {'Correlation':<12} {'Combined':<12}")
                print("-" * 48)
                for feature in overlap_features:
                    combined = feature['steerability'] + feature['abs_correlation']
                    print(
                        f"{feature['feature_id']:<8} {feature['steerability']:<12.4f} "
                        f"{feature['correlation']:<12.4f} {combined:<12.4f}"
                    )
            else:
                print(f"\n❌ NO OVERLAP: Different features are steerable vs correlated for {class_name}")

        # Add correlation analysis between steerability and correlation per class
        print(f"\n{'='*100}")
        print("CORRELATION ANALYSIS: STEERABILITY vs CORRELATION PER CLASS")
        print(f"{'='*100}")

        for cls_id, class_result in results.items():
            class_name = class_result['class_name']
            features = class_features[cls_id]

            if len(features) < 3:  # Need at least 3 features for meaningful correlation
                continue

            # Extract steerability and correlation values
            steerability_values = [f['steerability'] for f in features]
            correlation_values = [f['abs_correlation'] for f in features]

            # Calculate correlation between steerability and correlation
            if len(steerability_values) > 1 and len(correlation_values) > 1:
                from scipy import stats as scipy_stats

                # Pearson correlation
                pearson_corr, pearson_p = scipy_stats.pearsonr(steerability_values, correlation_values)

                # Spearman correlation
                spearman_corr, spearman_p = scipy_stats.spearmanr(steerability_values, correlation_values)

                print(f"\n📊 {class_name.upper()} (Class {cls_id}):")
                print(f"   Features analyzed: {len(features)}")
                print(f"   Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
                print(f"   Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")

                # Interpretation
                if abs(pearson_corr) > 0.5:
                    strength = "strong"
                elif abs(pearson_corr) > 0.3:
                    strength = "moderate"
                else:
                    strength = "weak"

                direction = "positive" if pearson_corr > 0 else "negative"
                significance = "significant" if pearson_p < 0.05 else "not significant"

                print(f"   Interpretation: {strength} {direction} correlation ({significance})")

                # Store correlation results in the results dict
                results[cls_id]['steerability_correlation_analysis'] = {
                    'pearson_corr': pearson_corr,
                    'pearson_p': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p': spearman_p,
                    'n_features': len(features),
                    'interpretation': {
                        'strength': strength,
                        'direction': direction,
                        'significance': significance
                    }
                }

        return results

    def plot_per_class_features_analysis(self, per_class_results: Dict[str, Any], layer_idx: int = None):
        """Create visualizations for per-class feature analysis"""
        n_classes = len(per_class_results)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f'Per-Class Feature Analysis {"- Layer " + str(layer_idx) if layer_idx else ""}',
            fontsize=16,
            fontweight='bold'
        )

        class_names = [data['class_name'] for data in per_class_results.values()]
        class_ids = list(per_class_results.keys())

        # 1. Overlap percentage per class
        ax1 = axes[0, 0]
        overlap_percentages = []
        top_k = len(per_class_results[class_ids[0]]['top_steerable'])  # Get top_k from first class

        for cls_id in class_ids:
            overlap_count = per_class_results[cls_id]['overlap_count']
            overlap_pct = (overlap_count / top_k) * 100
            overlap_percentages.append(overlap_pct)

        bars = ax1.bar(range(len(class_names)), overlap_percentages, color=self.colors[:len(class_names)], alpha=0.8)
        ax1.set_title('Feature Overlap Percentage per Class')
        ax1.set_ylabel('Overlap Percentage (%)')
        ax1.set_xlabel('Classes')
        ax1.set_xticks(range(len(class_names)))
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.set_ylim(0, 100)

        # Add percentage labels on bars
        for bar, pct in zip(bars, overlap_percentages):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2., height + 1, f'{pct:.0f}%', ha='center', va='bottom', fontsize=10
            )

        # 2. Top steerability values per class
        ax2 = axes[0, 1]
        max_steers = []
        mean_steers = []

        for cls_id in class_ids:
            top_steerable = per_class_results[cls_id]['top_steerable']
            steers = [f['steerability'] for f in top_steerable]
            max_steers.append(max(steers) if steers else 0)
            mean_steers.append(np.mean(steers) if steers else 0)

        x = np.arange(len(class_names))
        width = 0.35
        ax2.bar(x - width / 2, max_steers, width, label='Max Steerability', alpha=0.8, color='red')
        ax2.bar(x + width / 2, mean_steers, width, label='Mean Steerability', alpha=0.8, color='orange')
        ax2.set_title('Steerability Range per Class')
        ax2.set_ylabel('Steerability')
        ax2.set_xlabel('Classes')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.legend()

        # 3. Top correlation values per class
        ax3 = axes[0, 2]
        max_corrs = []
        mean_corrs = []

        for cls_id in class_ids:
            top_correlated = per_class_results[cls_id]['top_correlated']
            corrs = [abs(f['correlation']) for f in top_correlated]
            max_corrs.append(max(corrs) if corrs else 0)
            mean_corrs.append(np.mean(corrs) if corrs else 0)

        ax3.bar(x - width / 2, max_corrs, width, label='Max |Correlation|', alpha=0.8, color='blue')
        ax3.bar(x + width / 2, mean_corrs, width, label='Mean |Correlation|', alpha=0.8, color='lightblue')
        ax3.set_title('Correlation Range per Class')
        ax3.set_ylabel('Absolute Correlation')
        ax3.set_xlabel('Classes')
        ax3.set_xticks(x)
        ax3.set_xticklabels(class_names, rotation=45, ha='right')
        ax3.legend()

        # 4. Feature count per class
        ax4 = axes[1, 0]
        feature_counts = [per_class_results[cls_id]['total_features'] for cls_id in class_ids]
        bars = ax4.bar(range(len(class_names)), feature_counts, color=self.colors[:len(class_names)], alpha=0.8)
        ax4.set_title('Total Features per Class')
        ax4.set_ylabel('Number of Features')
        ax4.set_xlabel('Classes')
        ax4.set_xticks(range(len(class_names)))
        ax4.set_xticklabels(class_names, rotation=45, ha='right')

        # 5. Scatter: Max Steerability vs Max Correlation per class
        ax5 = axes[1, 1]
        scatter = ax5.scatter(
            max_corrs, max_steers, c=overlap_percentages, cmap='viridis', s=100, alpha=0.7, edgecolors='black'
        )
        ax5.set_xlabel('Max Absolute Correlation')
        ax5.set_ylabel('Max Steerability')
        ax5.set_title('Max Steerability vs Max Correlation')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Overlap %')

        # Add class labels to scatter points
        for i, cls_name in enumerate(class_names):
            ax5.annotate(
                cls_name[:4], (max_corrs[i], max_steers[i]), xytext=(5, 5), textcoords='offset points', fontsize=9
            )

        # 6. Heatmap of overlap counts
        ax6 = axes[1, 2]
        overlap_counts = [per_class_results[cls_id]['overlap_count'] for cls_id in class_ids]
        bars = ax6.bar(range(len(class_names)), overlap_counts, color='gold', alpha=0.8)
        ax6.set_title('Feature Overlap Count per Class')
        ax6.set_ylabel('Number of Overlapping Features')
        ax6.set_xlabel('Classes')
        ax6.set_xticks(range(len(class_names)))
        ax6.set_xticklabels(class_names, rotation=45, ha='right')
        ax6.set_ylim(0, top_k)

        # Add count labels on bars
        for bar, count in zip(bars, overlap_counts):
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2., height + 0.05, f'{count}', ha='center', va='bottom', fontsize=10
            )

        plt.tight_layout()

        if self.save_plots:
            suffix = f"_layer{layer_idx}" if layer_idx is not None else ""
            plt.savefig(self.output_dir / f'per_class_features_analysis{suffix}.png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_special_features_analysis(self, special_features: Dict[str, Any], layer_idx: int = None):
        """Create detailed plots for special features"""
        if not special_features:
            print("No special features found to plot.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f'High Occurrence & High Correlation Features Analysis {"- Layer " + str(layer_idx) if layer_idx else ""}',
            fontsize=16
        )

        # Extract data for plotting
        feature_ids = list(special_features.keys())
        occurrences = [special_features[fid]['total_occurrences'] for fid in feature_ids]
        correlations = [special_features[fid]['mean_correlation'] for fid in feature_ids]
        specificities = [special_features[fid]['class_specificity'] for fid in feature_ids]
        diversities = [special_features[fid]['class_diversity'] for fid in feature_ids]

        # 1. Scatter plot: Occurrence vs Correlation
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            occurrences, correlations, c=specificities, cmap='viridis', s=100, alpha=0.7, edgecolors='black'
        )
        ax1.set_xlabel('Total Occurrences')
        ax1.set_ylabel('Mean |Correlation|')
        ax1.set_title('Occurrence vs Correlation\n(Color = Class Specificity)')
        plt.colorbar(scatter, ax=ax1, label='Class Specificity')

        # Add feature ID labels for top features
        top_features = sorted(zip(feature_ids, occurrences, correlations), key=lambda x: x[1] * x[2], reverse=True)[:10]
        for fid, occ, corr in top_features:
            ax1.annotate(str(fid), (occ, corr), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

        # 2. Class distribution for top features
        ax2 = axes[0, 1]
        top_5_features = list(special_features.keys())[:5]
        class_data = []
        feature_labels = []

        for fid in top_5_features:
            class_counts = special_features[fid]['class_counts']
            class_data.append(list(class_counts.values()))
            feature_labels.append(f'F{fid}')

        if class_data:
            class_names = [IDX2CLS.get(cls_id, f"Class_{cls_id}") for cls_id in class_counts.keys()]
            x = np.arange(len(feature_labels))
            width = 0.12

            for i, class_name in enumerate(class_names):
                counts = [data[i] if i < len(data) else 0 for data in class_data]
                ax2.bar(x + i * width, counts, width, label=class_name, alpha=0.8)

            ax2.set_xlabel('Top 5 Features')
            ax2.set_ylabel('Class Occurrence Count')
            ax2.set_title('Class Distribution in Top Features')
            ax2.set_xticks(x + width * (len(class_names) - 1) / 2)
            ax2.set_xticklabels(feature_labels)
            ax2.legend()

        # 3. Correlation heatmap for top features
        ax3 = axes[0, 2]
        top_10_features = list(special_features.keys())[:10]

        if top_10_features:
            # Get all unique classes across all features
            all_classes = set()
            for fid in top_10_features:
                all_classes.update(special_features[fid]['class_correlations'].keys())
            all_classes = sorted(all_classes)

            # Create matrix with consistent dimensions
            corr_matrix = []
            for fid in top_10_features:
                class_correlations = special_features[fid]['class_correlations']
                corr_row = []
                for cls_id in all_classes:
                    corr_row.append(class_correlations.get(cls_id, 0.0))  # Default to 0 if class not present
                corr_matrix.append(corr_row)

            corr_matrix = np.array(corr_matrix)
            im = ax3.imshow(corr_matrix, cmap='RdBu_r', aspect='auto')
            ax3.set_xlabel('Classes')
            ax3.set_ylabel('Top 10 Features')
            ax3.set_title('Correlation Heatmap')
            ax3.set_xticks(range(len(all_classes)))
            ax3.set_xticklabels([IDX2CLS.get(cls_id, f"C{cls_id}") for cls_id in all_classes], rotation=45)
            ax3.set_yticks(range(len(top_10_features)))
            ax3.set_yticklabels([f'F{fid}' for fid in top_10_features])
            plt.colorbar(im, ax=ax3, label='Correlation')
        else:
            ax3.text(0.5, 0.5, 'No features to display', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Correlation Heatmap')

        # 4. Feature specificity distribution
        ax4 = axes[1, 0]
        ax4.hist(specificities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Class Specificity')
        ax4.set_ylabel('Number of Features')
        ax4.set_title('Distribution of Class Specificity')
        ax4.axvline(np.mean(specificities), color='red', linestyle='--', label=f'Mean: {np.mean(specificities):.3f}')
        ax4.legend()

        # 5. Class diversity vs correlation
        ax5 = axes[1, 1]
        ax5.scatter(diversities, correlations, c=occurrences, cmap='plasma', s=100, alpha=0.7, edgecolors='black')
        ax5.set_xlabel('Class Diversity')
        ax5.set_ylabel('Mean |Correlation|')
        ax5.set_title('Class Diversity vs Correlation\n(Color = Occurrences)')
        scatter2 = ax5.scatter(
            diversities, correlations, c=occurrences, cmap='plasma', s=100, alpha=0.7, edgecolors='black'
        )
        plt.colorbar(scatter2, ax=ax5, label='Occurrences')

        # 6. Combined importance score
        ax6 = axes[1, 2]
        importance_scores = [occ * corr for occ, corr in zip(occurrences, correlations)]
        sorted_indices = np.argsort(importance_scores)[-15:]  # Top 15

        top_features_plot = [feature_ids[i] for i in sorted_indices]
        top_scores = [importance_scores[i] for i in sorted_indices]

        bars = ax6.barh(range(len(top_features_plot)), top_scores, color='lightcoral', alpha=0.8)
        ax6.set_xlabel('Importance Score (Occ × |Corr|)')
        ax6.set_ylabel('Feature ID')
        ax6.set_title('Top 15 Features by Importance')
        ax6.set_yticks(range(len(top_features_plot)))
        ax6.set_yticklabels([f'F{fid}' for fid in top_features_plot])

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax6.text(
                bar.get_width() + score * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{score:.1f}',
                va='center',
                fontsize=8
            )

        plt.tight_layout()

        if self.save_plots:
            suffix = f"_layer{layer_idx}" if layer_idx is not None else ""
            plt.savefig(self.output_dir / f'special_features_analysis{suffix}.png', dpi=300, bbox_inches='tight')

        plt.show()

    def comprehensive_analysis(self, layer_idx: int = None, dict_path: str = None) -> Dict[str, Any]:
        """Run comprehensive analysis on a correlation dictionary"""
        if dict_path is None:
            if layer_idx is None:
                raise ValueError("Must provide either layer_idx or dict_path")
            dict_path = SAE_CONFIG[layer_idx]["dict_path"]

        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE ANALYSIS {'FOR LAYER ' + str(layer_idx) if layer_idx else 'FOR CUSTOM DICT'}")
        print(f"{'='*80}")

        # Load dictionary
        dict_data = self.load_correlation_dict(dict_path)

        # Extract class data
        class_data = self.extract_class_data(dict_data)

        # Run all analyses
        results = {}

        # 1. Correlation distribution analysis
        results['correlation_analysis'] = self.analyze_correlation_distributions(class_data, layer_idx)

        # 2. Occurrence distribution analysis
        results['occurrence_analysis'] = self.analyze_occurrence_distributions(class_data, layer_idx)

        # 3. Binning analysis
        results['binning_analysis'] = self.create_binning_analysis(class_data)

        # 4. Feature importance analysis
        results['feature_importance'] = self.analyze_feature_importance(dict_data)

        # 5. Cross-class analysis
        results['cross_class_analysis'] = self.cross_class_analysis(class_data)

        # 6. Special features analysis (high occurrence + high correlation)
        results['special_features'] = self.analyze_high_occurrence_high_correlation_features(
            dict_data, occurrence_threshold=50, correlation_threshold=0.3
        )

        # Generate plots
        self.plot_correlation_distributions(class_data, layer_idx)
        self.plot_occurrence_distributions(class_data, layer_idx)
        self.plot_binning_analysis(results['binning_analysis'], layer_idx)
        self.plot_special_features_analysis(results['special_features'], layer_idx)

        return results

    def compare_layers(self, layer_list: List[int] = None) -> Dict[str, Any]:
        """Compare analysis across multiple layers"""
        if layer_list is None:
            layer_list = [6, 7, 8, 9, 10]

        print(f"\n{'='*80}")
        print("MULTI-LAYER COMPARISON ANALYSIS")
        print(f"{'='*80}")

        layer_results = {}

        for layer_idx in layer_list:
            if layer_idx not in SAE_CONFIG:
                print(f"Skipping layer {layer_idx} - no configuration found")
                continue

            try:
                dict_path = SAE_CONFIG[layer_idx]["dict_path"]
                if not Path(dict_path).exists():
                    print(f"Skipping layer {layer_idx} - dictionary not found")
                    continue

                print(f"\nAnalyzing Layer {layer_idx}...")
                results = self.comprehensive_analysis(layer_idx, dict_path)
                layer_results[layer_idx] = results

            except Exception as e:
                print(f"Error analyzing layer {layer_idx}: {e}")

        # Create cross-layer comparison plots
        self._plot_layer_comparison(layer_results)

        return layer_results

    def _plot_layer_comparison(self, layer_results: Dict[int, Dict[str, Any]]):
        """Create comparison plots across layers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Layer Comparison', fontsize=16)

        # Extract data for comparison
        layers = sorted(layer_results.keys())

        # Mean correlation by class across layers
        ax1 = axes[0, 0]
        class_means = defaultdict(list)

        for layer in layers:
            corr_analysis = layer_results[layer]['correlation_analysis']
            for cls_id, stats in corr_analysis.items():
                class_means[cls_id].append(stats['mean'])

        for cls_id, means in class_means.items():
            ax1.plot(layers, means, 'o-', label=IDX2CLS.get(cls_id, f"Class_{cls_id}"))

        ax1.set_title('Mean Correlation by Class Across Layers')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Mean Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Feature count across layers
        ax2 = axes[0, 1]
        feature_counts = [len(layer_results[layer]['feature_importance']) for layer in layers]
        ax2.bar(layers, feature_counts, color='skyblue')
        ax2.set_title('Number of Features Across Layers')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Feature Count')
        ax2.grid(True, alpha=0.3)

        # Total occurrences by class across layers
        ax3 = axes[1, 0]
        class_totals = defaultdict(list)

        for layer in layers:
            occ_analysis = layer_results[layer]['occurrence_analysis']
            for cls_id, stats in occ_analysis.items():
                class_totals[cls_id].append(stats['total'])

        for cls_id, totals in class_totals.items():
            ax3.plot(layers, totals, 'o-', label=IDX2CLS.get(cls_id, f"Class_{cls_id}"))

        ax3.set_title('Total Occurrences by Class Across Layers')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Total Occurrences')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Correlation std by class across layers
        ax4 = axes[1, 1]
        class_stds = defaultdict(list)

        for layer in layers:
            corr_analysis = layer_results[layer]['correlation_analysis']
            for cls_id, stats in corr_analysis.items():
                class_stds[cls_id].append(stats['std'])

        for cls_id, stds in class_stds.items():
            ax4.plot(layers, stds, 'o-', label=IDX2CLS.get(cls_id, f"Class_{cls_id}"))

        ax4.set_title('Correlation Std by Class Across Layers')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Std Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(self.output_dir / 'layer_comparison.png', dpi=300, bbox_inches='tight')

        plt.show()

    def analyze_special_features_only(
        self,
        layer_idx: int = None,
        dict_path: str = None,
        occurrence_threshold: int = 50,
        correlation_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Quick analysis focused only on special features"""
        if dict_path is None:
            if layer_idx is None:
                raise ValueError("Must provide either layer_idx or dict_path")
            dict_path = SAE_CONFIG[layer_idx]["dict_path"]

        print(f"\n{'='*80}")
        print(f"SPECIAL FEATURES ANALYSIS {'FOR LAYER ' + str(layer_idx) if layer_idx else 'FOR CUSTOM DICT'}")
        print(f"{'='*80}")

        # Load dictionary
        dict_data = self.load_correlation_dict(dict_path)

        # Analyze special features
        special_features = self.analyze_high_occurrence_high_correlation_features(
            dict_data, occurrence_threshold, correlation_threshold
        )

        # Generate plot
        self.plot_special_features_analysis(special_features, layer_idx)

        return special_features

    def analyze_cross_layer_feature_landscape(
        self,
        layer_list: List[int] = None,
        occurrence_threshold: int = 20,
        correlation_threshold: float = 0.15
    ) -> Dict[str, Any]:
        """Analyze feature landscape across all layers to identify interesting layers"""
        if layer_list is None:
            layer_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        print(f"\n{'='*100}")
        print("CROSS-LAYER FEATURE LANDSCAPE ANALYSIS")
        print(f"{'='*100}")

        layer_data = {}
        all_class_ids = set()

        # Load data for all available layers
        for layer_idx in layer_list:
            if layer_idx not in SAE_CONFIG:
                print(f"Skipping layer {layer_idx} - no configuration found")
                continue

            dict_path = SAE_CONFIG[layer_idx]["dict_path"]
            if not Path(dict_path).exists():
                print(f"Skipping layer {layer_idx} - dictionary not found at {dict_path}")
                continue

            try:
                print(f"Loading layer {layer_idx}...")
                dict_data = self.load_correlation_dict(dict_path)

                # Extract features meeting criteria
                high_quality_features = {}
                class_statistics = defaultdict(lambda: {'correlations': [], 'occurrences': [], 'feature_count': 0})

                for fid, stats in dict_data['feature_stats'].items():
                    total_occurrences = stats['occurrences']
                    mean_correlation = abs(stats['mean_pfac_corr'])

                    if total_occurrences >= occurrence_threshold and mean_correlation >= correlation_threshold:
                        high_quality_features[fid] = stats

                        # Aggregate by class
                        for cls_id, corr_val in stats['class_mean_pfac'].items():
                            class_statistics[cls_id]['correlations'].append(abs(corr_val))
                            class_statistics[cls_id]['occurrences'].append(stats['class_count_map'].get(cls_id, 0))
                            all_class_ids.add(cls_id)

                # Compute layer-level statistics
                layer_stats = {}
                for cls_id, data in class_statistics.items():
                    data['feature_count'] = len(data['correlations'])
                    layer_stats[cls_id] = {
                        'mean_correlation': np.mean(data['correlations']) if data['correlations'] else 0,
                        'std_correlation': np.std(data['correlations']) if data['correlations'] else 0,
                        'mean_occurrence': np.mean(data['occurrences']) if data['occurrences'] else 0,
                        'total_occurrences': sum(data['occurrences']) if data['occurrences'] else 0,
                        'feature_count': data['feature_count'],
                        'max_correlation': max(data['correlations']) if data['correlations'] else 0,
                        'max_occurrence': max(data['occurrences']) if data['occurrences'] else 0
                    }

                layer_data[layer_idx] = {
                    'total_high_quality_features': len(high_quality_features),
                    'class_statistics': layer_stats,
                    'raw_features': high_quality_features
                }

                print(f"  Layer {layer_idx}: {len(high_quality_features)} high-quality features")

            except Exception as e:
                print(f"Error processing layer {layer_idx}: {e}")

        # Create comprehensive analysis
        results = {
            'layer_data': layer_data,
            'all_classes': sorted(all_class_ids),
            'parameters': {
                'occurrence_threshold': occurrence_threshold,
                'correlation_threshold': correlation_threshold,
                'layers_analyzed': list(layer_data.keys())
            }
        }

        # Generate analysis plots
        self._plot_cross_layer_landscape(results)

        # Print summary
        self._print_cross_layer_summary(results)

        return results

    def _plot_cross_layer_landscape(self, results: Dict[str, Any]):
        """Create comprehensive cross-layer visualization"""
        layer_data = results['layer_data']
        all_classes = results['all_classes']
        layers = sorted(layer_data.keys())

        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Feature count heatmap by layer and class
        ax1 = fig.add_subplot(gs[0, :2])
        feature_count_matrix = []
        for layer in layers:
            row = []
            for cls_id in all_classes:
                count = layer_data[layer]['class_statistics'].get(cls_id, {}).get('feature_count', 0)
                row.append(count)
            feature_count_matrix.append(row)

        feature_count_matrix = np.array(feature_count_matrix)
        im1 = ax1.imshow(feature_count_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_title('High-Quality Feature Count by Layer and Class', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Layers')
        ax1.set_xticks(range(len(all_classes)))
        ax1.set_xticklabels([IDX2CLS.get(cls_id, f"C{cls_id}") for cls_id in all_classes], rotation=45)
        ax1.set_yticks(range(len(layers)))
        ax1.set_yticklabels([f"L{layer}" for layer in layers])
        plt.colorbar(im1, ax=ax1, label='Feature Count')

        # Add text annotations
        for i in range(len(layers)):
            for j in range(len(all_classes)):
                text = ax1.text(
                    j,
                    i,
                    f'{feature_count_matrix[i, j]}',
                    ha="center",
                    va="center",
                    color="white" if feature_count_matrix[i, j] > feature_count_matrix.max() / 2 else "black"
                )

        # 2. Mean correlation heatmap
        ax2 = fig.add_subplot(gs[0, 2:])
        correlation_matrix = []
        for layer in layers:
            row = []
            for cls_id in all_classes:
                corr = layer_data[layer]['class_statistics'].get(cls_id, {}).get('mean_correlation', 0)
                row.append(corr)
            correlation_matrix.append(row)

        correlation_matrix = np.array(correlation_matrix)
        im2 = ax2.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto')
        ax2.set_title('Mean Correlation by Layer and Class', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Layers')
        ax2.set_xticks(range(len(all_classes)))
        ax2.set_xticklabels([IDX2CLS.get(cls_id, f"C{cls_id}") for cls_id in all_classes], rotation=45)
        ax2.set_yticks(range(len(layers)))
        ax2.set_yticklabels([f"L{layer}" for layer in layers])
        plt.colorbar(im2, ax=ax2, label='Mean Correlation')

        # 3. Total high-quality features per layer
        ax3 = fig.add_subplot(gs[1, :2])
        total_features = [layer_data[layer]['total_high_quality_features'] for layer in layers]
        bars = ax3.bar(range(len(layers)), total_features, color='skyblue', alpha=0.8, edgecolor='navy')
        ax3.set_title('Total High-Quality Features per Layer', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Number of Features')
        ax3.set_xticks(range(len(layers)))
        ax3.set_xticklabels([f"L{layer}" for layer in layers])
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars, total_features):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.,
                height + height * 0.01,
                f'{count}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )

        # 4. Mean occurrence by layer and class
        ax4 = fig.add_subplot(gs[1, 2:])
        occurrence_matrix = []
        for layer in layers:
            row = []
            for cls_id in all_classes:
                occ = layer_data[layer]['class_statistics'].get(cls_id, {}).get('mean_occurrence', 0)
                row.append(occ)
            occurrence_matrix.append(row)

        occurrence_matrix = np.array(occurrence_matrix)
        im4 = ax4.imshow(occurrence_matrix, cmap='Purples', aspect='auto')
        ax4.set_title('Mean Occurrence by Layer and Class', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Classes')
        ax4.set_ylabel('Layers')
        ax4.set_xticks(range(len(all_classes)))
        ax4.set_xticklabels([IDX2CLS.get(cls_id, f"C{cls_id}") for cls_id in all_classes], rotation=45)
        ax4.set_yticks(range(len(layers)))
        ax4.set_yticklabels([f"L{layer}" for layer in layers])
        plt.colorbar(im4, ax=ax4, label='Mean Occurrence')

        # 5. Layer-wise feature quality score (correlation * occurrence)
        ax5 = fig.add_subplot(gs[2, :2])
        class_quality_scores = {}

        for cls_id in all_classes:
            scores = []
            for layer in layers:
                stats = layer_data[layer]['class_statistics'].get(cls_id, {})
                score = stats.get('mean_correlation', 0) * stats.get('mean_occurrence', 0)
                scores.append(score)
            class_quality_scores[cls_id] = scores

        for cls_id, scores in class_quality_scores.items():
            class_name = IDX2CLS.get(cls_id, f"Class_{cls_id}")
            ax5.plot(range(len(layers)), scores, 'o-', linewidth=2, markersize=6, label=class_name, alpha=0.8)

        ax5.set_title('Feature Quality Score by Layer\n(Correlation × Occurrence)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Layer')
        ax5.set_ylabel('Quality Score')
        ax5.set_xticks(range(len(layers)))
        ax5.set_xticklabels([f"L{layer}" for layer in layers])
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)

        # 6. Layer complexity analysis
        ax6 = fig.add_subplot(gs[2, 2:])
        complexity_metrics = {'Feature Diversity': [], 'Max Correlation': [], 'Feature Density': []}

        for layer in layers:
            # Feature diversity: number of classes with features
            diversity = len([
                cls for cls in all_classes
                if layer_data[layer]['class_statistics'].get(cls, {}).get('feature_count', 0) > 0
            ])
            complexity_metrics['Feature Diversity'].append(diversity)

            # Max correlation across all classes
            max_corr = max([
                layer_data[layer]['class_statistics'].get(cls, {}).get('max_correlation', 0) for cls in all_classes
            ],
                           default=0)
            complexity_metrics['Max Correlation'].append(max_corr)

            # Feature density: total features / number of classes
            total_feat = layer_data[layer]['total_high_quality_features']
            density = total_feat / len(all_classes) if all_classes else 0
            complexity_metrics['Feature Density'].append(density)

        x = np.arange(len(layers))
        width = 0.25

        for i, (metric, values) in enumerate(complexity_metrics.items()):
            # Normalize values for comparison
            norm_values = np.array(values) / max(values) if max(values) > 0 else np.array(values)
            ax6.bar(x + i * width, norm_values, width, label=metric, alpha=0.8)

        ax6.set_title('Layer Complexity Metrics (Normalized)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Layer')
        ax6.set_ylabel('Normalized Score')
        ax6.set_xticks(x + width)
        ax6.set_xticklabels([f"L{layer}" for layer in layers])
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.suptitle('Cross-Layer Feature Landscape Analysis', fontsize=18, fontweight='bold', y=0.98)

        if self.save_plots:
            plt.savefig(self.output_dir / 'cross_layer_feature_landscape.png', dpi=300, bbox_inches='tight')

        plt.show()

    def _print_cross_layer_summary(self, results: Dict[str, Any]):
        """Print comprehensive summary of cross-layer analysis"""
        layer_data = results['layer_data']
        all_classes = results['all_classes']
        layers = sorted(layer_data.keys())

        print(f"\n{'='*80}")
        print("CROSS-LAYER ANALYSIS SUMMARY")
        print(f"{'='*80}")

        # Find best layers for each metric
        layer_feature_counts = [(layer, layer_data[layer]['total_high_quality_features']) for layer in layers]
        best_layer_features = max(layer_feature_counts, key=lambda x: x[1])

        print(f"\n🏆 LAYER RANKINGS:")
        print(f"   Most High-Quality Features: Layer {best_layer_features[0]} ({best_layer_features[1]} features)")

        # Best layer per class
        print(f"\n📊 BEST LAYERS PER CLASS:")
        print(f"{'Class':<15} {'Best Layer':<12} {'Feature Count':<15} {'Mean Correlation':<18} {'Mean Occurrence'}")
        print("-" * 80)

        for cls_id in all_classes:
            class_name = IDX2CLS.get(cls_id, f"Class_{cls_id}")
            best_layer = None
            best_score = 0
            best_stats = {}

            for layer in layers:
                stats = layer_data[layer]['class_statistics'].get(cls_id, {})
                score = stats.get('feature_count', 0) * stats.get('mean_correlation', 0)
                if score > best_score:
                    best_score = score
                    best_layer = layer
                    best_stats = stats

            if best_layer:
                print(
                    f"{class_name:<15} {best_layer:<12} {best_stats.get('feature_count', 0):<15} "
                    f"{best_stats.get('mean_correlation', 0):<18.4f} {best_stats.get('mean_occurrence', 0):.2f}"
                )

        # Layer progression analysis
        print(f"\n📈 LAYER PROGRESSION ANALYSIS:")
        print(f"{'Layer':<8} {'Total Features':<15} {'Avg Correlation':<18} {'Avg Occurrence':<15} {'Quality Score'}")
        print("-" * 80)

        for layer in layers:
            total_features = layer_data[layer]['total_high_quality_features']

            # Calculate averages across all classes
            all_corrs = []
            all_occs = []
            for cls_id in all_classes:
                stats = layer_data[layer]['class_statistics'].get(cls_id, {})
                if stats.get('feature_count', 0) > 0:
                    all_corrs.append(stats.get('mean_correlation', 0))
                    all_occs.append(stats.get('mean_occurrence', 0))

            avg_corr = np.mean(all_corrs) if all_corrs else 0
            avg_occ = np.mean(all_occs) if all_occs else 0
            quality_score = avg_corr * avg_occ

            print(f"L{layer:<7} {total_features:<15} {avg_corr:<18.4f} {avg_occ:<15.2f} {quality_score:<.4f}")

        # Identify interesting layers
        feature_counts = [layer_data[layer]['total_high_quality_features'] for layer in layers]
        mean_features = np.mean(feature_counts)
        std_features = np.std(feature_counts)

        interesting_layers = []
        for layer in layers:
            if layer_data[layer]['total_high_quality_features'] > mean_features + 0.5 * std_features:
                interesting_layers.append(layer)

        print(f"\nMOST INTERESTING LAYERS:")
        print(f"   Layers with above-average feature quality: {interesting_layers}")
        print(
            f"   These layers have {len(interesting_layers)/len(layers)*100:.1f}% more high-quality features than average"
        )

    def analyze_correlation_outliers(
        self, dict_data: Dict[str, Any], min_correlation: float = 0.4, layer_idx: int = None
    ) -> Dict[str, Any]:
        """Analyze high-correlation features regardless of occurrence frequency"""
        print(f"\n{'='*80}")
        print(f"CORRELATION OUTLIERS ANALYSIS {'FOR LAYER ' + str(layer_idx) if layer_idx else ''}")
        print(f"Analyzing features with |correlation| >= {min_correlation}")
        print(f"{'='*80}")

        outlier_features = {}
        correlation_occurrence_pairs = []

        for fid, stats in dict_data['feature_stats'].items():
            max_abs_correlation = max([abs(corr) for corr in stats['class_mean_pfac'].values()])
            total_occurrences = stats['occurrences']

            # Collect all correlation-occurrence pairs for scatter analysis
            for cls_id, corr_val in stats['class_mean_pfac'].items():
                correlation_occurrence_pairs.append({
                    'feature_id': fid,
                    'class_id': cls_id,
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val),
                    'class_occurrences': stats['class_count_map'].get(cls_id, 0),
                    'total_occurrences': total_occurrences
                })

            # Check if feature has high correlation
            if max_abs_correlation >= min_correlation:
                # Find the class with maximum absolute correlation
                dominant_class_data = max(stats['class_mean_pfac'].items(), key=lambda x: abs(x[1]))

                outlier_features[fid] = {
                    'max_abs_correlation': max_abs_correlation,
                    'dominant_class': dominant_class_data[0],
                    'dominant_correlation': dominant_class_data[1],
                    'total_occurrences': total_occurrences,
                    'class_correlations': stats['class_mean_pfac'],
                    'class_counts': stats['class_count_map'],
                    'rarity_score': max_abs_correlation / (total_occurrences + 1),  # High correlation, low occurrence
                    'reliability_score':
                    max_abs_correlation * np.log(total_occurrences + 1)  # High correlation, high occurrence
                }

        # Sort by maximum absolute correlation
        sorted_outliers = sorted(outlier_features.items(), key=lambda x: x[1]['max_abs_correlation'], reverse=True)

        # Analyze correlation-occurrence distribution
        corr_occ_analysis = self._analyze_correlation_occurrence_distribution(correlation_occurrence_pairs)

        results = {
            'outlier_features': dict(sorted_outliers),
            'correlation_occurrence_data': correlation_occurrence_pairs,
            'distribution_analysis': corr_occ_analysis,
            'parameters': {
                'min_correlation': min_correlation,
                'total_outliers_found': len(outlier_features)
            }
        }

        # Print summary
        self._print_outlier_summary(results, layer_idx)

        # Generate visualizations
        self._plot_correlation_outliers(results, layer_idx)

        return results

    def _analyze_correlation_occurrence_distribution(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze the distribution of correlation vs occurrence patterns"""
        df = pd.DataFrame(data)

        # Define outlier categories
        high_corr_low_occ = df[(df['abs_correlation'] >= 0.5) & (df['class_occurrences'] <= 5)]
        high_corr_high_occ = df[(df['abs_correlation'] >= 0.5) & (df['class_occurrences'] > 20)]
        medium_corr_any_occ = df[(df['abs_correlation'] >= 0.3) & (df['abs_correlation'] < 0.5)]

        return {
            'high_corr_low_occ': len(high_corr_low_occ),
            'high_corr_high_occ': len(high_corr_high_occ),
            'medium_corr_any_occ': len(medium_corr_any_occ),
            'total_feature_class_pairs': len(df),
            'correlation_percentiles': {
                '99th': np.percentile(df['abs_correlation'], 99),
                '95th': np.percentile(df['abs_correlation'], 95),
                '90th': np.percentile(df['abs_correlation'], 90)
            },
            'occurrence_percentiles': {
                '99th': np.percentile(df['class_occurrences'], 99),
                '95th': np.percentile(df['class_occurrences'], 95),
                '90th': np.percentile(df['class_occurrences'], 90)
            }
        }

    def _print_outlier_summary(self, results: Dict[str, Any], layer_idx: int = None):
        """Print detailed summary of correlation outliers"""
        outliers = results['outlier_features']
        dist_analysis = results['distribution_analysis']

        print(f"\n🎯 HIGH CORRELATION FEATURES (Top 15):")
        print(
            f"{'Feature':<8} {'Max |Corr|':<12} {'Dom Class':<15} {'Dom Corr':<12} {'Total Occ':<12} {'Rarity':<10} {'Reliability':<12}"
        )
        print("-" * 90)

        for i, (fid, stats) in enumerate(list(outliers.items())[:15]):
            dom_class_name = IDX2CLS.get(stats['dominant_class'], f"C{stats['dominant_class']}")
            print(
                f"{fid:<8} {stats['max_abs_correlation']:<12.4f} {dom_class_name:<15} "
                f"{stats['dominant_correlation']:<12.4f} {stats['total_occurrences']:<12} "
                f"{stats['rarity_score']:<10.4f} {stats['reliability_score']:<12.4f}"
            )

        print(f"\n📊 CORRELATION-OCCURRENCE DISTRIBUTION:")
        print(
            f"   High Correlation (≥0.5) + Low Occurrence (≤5):  {dist_analysis['high_corr_low_occ']} feature-class pairs"
        )
        print(
            f"   High Correlation (≥0.5) + High Occurrence (>20): {dist_analysis['high_corr_high_occ']} feature-class pairs"
        )
        print(
            f"   Medium Correlation (0.3-0.5):                    {dist_analysis['medium_corr_any_occ']} feature-class pairs"
        )
        print(f"   Total feature-class pairs analyzed:              {dist_analysis['total_feature_class_pairs']}")

        print(f"\n📈 CORRELATION PERCENTILES:")
        for percentile, value in dist_analysis['correlation_percentiles'].items():
            print(f"   {percentile}: {value:.4f}")

        print(f"\n📈 OCCURRENCE PERCENTILES:")
        for percentile, value in dist_analysis['occurrence_percentiles'].items():
            print(f"   {percentile}: {value:.1f}")

        # Identify truly rare but highly correlated features
        rare_gems = [(fid, stats) for fid, stats in outliers.items()
                     if stats['max_abs_correlation'] >= 0.5 and stats['total_occurrences'] <= 3]

        if rare_gems:
            print(f"\n💎 RARE GEMS (High correlation ≥0.5, Total occurrences ≤3):")
            for fid, stats in rare_gems[:10]:
                dom_class_name = IDX2CLS.get(stats['dominant_class'], f"C{stats['dominant_class']}")
                print(
                    f"   Feature {fid}: {stats['max_abs_correlation']:.4f} correlation, "
                    f"{stats['total_occurrences']} occurrences, dominant in {dom_class_name}"
                )

    def _plot_correlation_outliers(self, results: Dict[str, Any], layer_idx: int = None):
        """Create comprehensive visualization of correlation outliers"""
        outliers = results['outlier_features']
        data = results['correlation_occurrence_data']
        df = pd.DataFrame(data)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f'Correlation Outliers Analysis {"- Layer " + str(layer_idx) if layer_idx else ""}',
            fontsize=16,
            fontweight='bold'
        )

        # 1. Correlation vs Occurrence Scatter Plot
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            df['class_occurrences'], df['abs_correlation'], c=df['feature_id'], cmap='viridis', alpha=0.6, s=50
        )
        ax1.set_xlabel('Class Occurrences')
        ax1.set_ylabel('Absolute Correlation')
        ax1.set_title('Correlation vs Occurrence Distribution')
        ax1.grid(True, alpha=0.3)

        # Add threshold lines
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='High Correlation (0.5)')
        ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Medium Correlation (0.3)')
        ax1.axvline(x=5, color='blue', linestyle='--', alpha=0.7, label='Low Occurrence (5)')
        ax1.legend()

        # 2. Log scale correlation vs occurrence
        ax2 = axes[0, 1]
        ax2.scatter(np.log1p(df['class_occurrences']), df['abs_correlation'], alpha=0.6, s=50, c='purple')
        ax2.set_xlabel('Log(Occurrences + 1)')
        ax2.set_ylabel('Absolute Correlation')
        ax2.set_title('Correlation vs Log(Occurrence)')
        ax2.grid(True, alpha=0.3)

        # 3. Top outliers by correlation
        ax3 = axes[0, 2]
        top_15_outliers = list(outliers.items())[:15]
        feature_ids = [fid for fid, _ in top_15_outliers]
        correlations = [stats['max_abs_correlation'] for _, stats in top_15_outliers]
        occurrences = [stats['total_occurrences'] for _, stats in top_15_outliers]

        bars = ax3.bar(
            range(len(feature_ids)),
            correlations,
            color=plt.cm.Reds(np.array(correlations) / max(correlations)),
            alpha=0.8
        )
        ax3.set_xlabel('Top Features (by correlation)')
        ax3.set_ylabel('Max Absolute Correlation')
        ax3.set_title('Top 15 High-Correlation Features')
        ax3.set_xticks(range(len(feature_ids)))
        ax3.set_xticklabels([f'F{fid}' for fid in feature_ids], rotation=45)

        # Add occurrence as text on bars
        for i, (bar, occ) in enumerate(zip(bars, occurrences)):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2., height + 0.01, f'n={occ}', ha='center', va='bottom', fontsize=8
            )

        # 4. Rarity vs Reliability scatter
        ax4 = axes[1, 0]
        rarity_scores = [stats['rarity_score'] for _, stats in outliers.items()]
        reliability_scores = [stats['reliability_score'] for _, stats in outliers.items()]

        scatter2 = ax4.scatter(
            reliability_scores,
            rarity_scores,
            c=[stats['max_abs_correlation'] for _, stats in outliers.items()],
            cmap='plasma',
            s=80,
            alpha=0.7
        )
        ax4.set_xlabel('Reliability Score (Corr × log(Occ))')
        ax4.set_ylabel('Rarity Score (Corr / Occ)')
        ax4.set_title('Feature Rarity vs Reliability')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax4, label='Max Correlation')

        # 5. Correlation distribution histogram
        ax5 = axes[1, 1]
        ax5.hist(df['abs_correlation'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='High Threshold (0.5)')
        ax5.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Medium Threshold (0.3)')
        ax5.set_xlabel('Absolute Correlation')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Correlation Distribution')
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Occurrence distribution histogram
        ax6 = axes[1, 2]
        ax6.hist(df['class_occurrences'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax6.axvline(x=5, color='blue', linestyle='--', alpha=0.7, label='Low Occurrence (5)')
        ax6.axvline(x=20, color='green', linestyle='--', alpha=0.7, label='High Occurrence (20)')
        ax6.set_xlabel('Class Occurrences')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Occurrence Distribution')
        ax6.set_yscale('log')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_plots:
            suffix = f"_layer{layer_idx}" if layer_idx is not None else ""
            plt.savefig(self.output_dir / f'correlation_outliers_analysis{suffix}.png', dpi=300, bbox_inches='tight')

        plt.show()

    def analyze_outliers_only(self,
                              layer_idx: int = None,
                              dict_path: str = None,
                              min_correlation: float = 0.4) -> Dict[str, Any]:
        """Quick analysis focused only on correlation outliers"""
        if dict_path is None:
            if layer_idx is None:
                raise ValueError("Must provide either layer_idx or dict_path")
            dict_path = SAE_CONFIG[layer_idx]["dict_path"]

        # Load dictionary
        dict_data = self.load_correlation_dict(dict_path)

        # Analyze outliers
        outlier_results = self.analyze_correlation_outliers(dict_data, min_correlation, layer_idx)

        return outlier_results

    def analyze_top_correlations_per_class(
        self,
        dict_data: Dict[str, Any] = None,
        layer_idx: int = None,
        dict_path: str = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze top K correlations per class showing mean, max, min, count and overall occurrence.
        This helps identify feature sparsity and class-specific contexts.
        
        Args:
            dict_data: Pre-loaded dictionary data
            layer_idx: Layer index to analyze
            dict_path: Path to dictionary file
            top_k: Number of top correlations to analyze per class
        
        Returns:
            Dictionary containing analysis results per class
        """
        if dict_data is None:
            if dict_path is None:
                if layer_idx is None:
                    raise ValueError("Must provide either dict_data, layer_idx, or dict_path")
                dict_path = SAE_CONFIG[layer_idx]["dict_path"]
            dict_data = self.load_correlation_dict(dict_path)

        print(f"\n{'='*80}")
        print(f"TOP {top_k} CORRELATIONS PER CLASS ANALYSIS {'FOR LAYER ' + str(layer_idx) if layer_idx else ''}")
        print(f"{'='*80}")

        # Extract all feature-class correlation pairs
        all_correlations_by_class = defaultdict(list)
        feature_overall_stats = {}

        for fid, stats in dict_data['feature_stats'].items():
            class_mean_pfac = stats['class_mean_pfac']
            class_count_map = stats['class_count_map']
            total_occurrences = stats['occurrences']

            # Store overall stats for this feature
            feature_overall_stats[fid] = {
                'total_occurrences': total_occurrences,
                'class_diversity': len(class_count_map),  # Number of classes this feature appears in
                'mean_pfac_corr': stats['mean_pfac_corr']
            }

            # Add to class-specific lists
            for cls_id, corr_val in class_mean_pfac.items():
                all_correlations_by_class[cls_id].append({
                    'feature_id':
                    fid,
                    'correlation':
                    corr_val,
                    'abs_correlation':
                    abs(corr_val),
                    'class_occurrences':
                    class_count_map.get(cls_id, 0),
                    'total_occurrences':
                    total_occurrences,
                    'class_diversity':
                    len(class_count_map),
                    'polysemantic_score':
                    len(class_count_map) / total_occurrences if total_occurrences > 0 else 0
                })

        # Analyze top correlations per class
        results = {}

        for cls_id, correlations in all_correlations_by_class.items():
            # Sort by absolute correlation (descending)
            sorted_correlations = sorted(correlations, key=lambda x: x['abs_correlation'], reverse=True)
            top_correlations = sorted_correlations[:top_k]

            # Calculate statistics for top correlations
            top_corr_values = [item['abs_correlation'] for item in top_correlations]
            top_occurrences = [item['class_occurrences'] for item in top_correlations]
            top_total_occurrences = [item['total_occurrences'] for item in top_correlations]
            top_polysemantic_scores = [item['polysemantic_score'] for item in top_correlations]

            results[cls_id] = {
                'top_features': top_correlations,
                'statistics': {
                    'mean_correlation':
                    np.mean(top_corr_values),
                    'max_correlation':
                    np.max(top_corr_values),
                    'min_correlation':
                    np.min(top_corr_values),
                    'std_correlation':
                    np.std(top_corr_values),
                    'count':
                    len(top_correlations),
                    'mean_class_occurrences':
                    np.mean(top_occurrences),
                    'mean_total_occurrences':
                    np.mean(top_total_occurrences),
                    'mean_polysemantic_score':
                    np.mean(top_polysemantic_scores),
                    'sparsity_indicator':
                    np.mean(top_total_occurrences) / max(top_total_occurrences) if top_total_occurrences else 0
                }
            }

        # Print comprehensive analysis
        self._print_top_correlations_analysis(results, top_k)

        # Generate visualizations
        self._plot_top_correlations_per_class(results, layer_idx, top_k)

        return results

    def _print_top_correlations_analysis(self, results: Dict[str, Any], top_k: int):
        """Print detailed analysis of top correlations per class"""
        print(f"\n🎯 TOP {top_k} CORRELATIONS STATISTICS PER CLASS:")
        print(
            f"{'Class':<15} {'Mean':<8} {'Max':<8} {'Min':<8} {'Count':<6} {'Avg Occ':<8} {'Avg Tot':<8} {'Polysem':<8} {'Sparsity':<8}"
        )
        print("-" * 90)

        for cls_id, analysis in results.items():
            class_name = IDX2CLS.get(cls_id, f"Class_{cls_id}")
            stats = analysis['statistics']

            print(
                f"{class_name:<15} {stats['mean_correlation']:<8.4f} {stats['max_correlation']:<8.4f} "
                f"{stats['min_correlation']:<8.4f} {stats['count']:<6} {stats['mean_class_occurrences']:<8.1f} "
                f"{stats['mean_total_occurrences']:<8.1f} {stats['mean_polysemantic_score']:<8.3f} "
                f"{stats['sparsity_indicator']:<8.3f}"
            )

        print(f"\n📊 FEATURE SPARSITY AND POLYSEMANTIC ANALYSIS:")
        print("Legend:")
        print("  - Avg Occ: Average occurrences in this class")
        print("  - Avg Tot: Average total occurrences across all classes")
        print("  - Polysem: Polysemantic score (class_diversity / total_occurrences)")
        print("  - Sparsity: Sparsity indicator (higher = more sparse)")

        # Identify most/least sparse and polysemantic classes
        sparsity_scores = [(cls_id, analysis['statistics']['sparsity_indicator'])
                           for cls_id, analysis in results.items()]
        polysemantic_scores = [(cls_id, analysis['statistics']['mean_polysemantic_score'])
                               for cls_id, analysis in results.items()]

        sparsity_scores.sort(key=lambda x: x[1], reverse=True)
        polysemantic_scores.sort(key=lambda x: x[1], reverse=True)

        print(f"\n🔍 INSIGHTS:")
        print(
            f"  Most sparse features (class-specific): {IDX2CLS.get(sparsity_scores[0][0], f'Class_{sparsity_scores[0][0]}')}"
        )
        print(
            f"  Least sparse features (broadly active): {IDX2CLS.get(sparsity_scores[-1][0], f'Class_{sparsity_scores[-1][0]}')}"
        )
        print(
            f"  Most polysemantic features: {IDX2CLS.get(polysemantic_scores[0][0], f'Class_{polysemantic_scores[0][0]}')}"
        )
        print(
            f"  Least polysemantic features: {IDX2CLS.get(polysemantic_scores[-1][0], f'Class_{polysemantic_scores[-1][0]}')}"
        )

        # Print top features for each class
        print(f"\n🏆 TOP 5 FEATURES PER CLASS:")
        for cls_id, analysis in results.items():
            class_name = IDX2CLS.get(cls_id, f"Class_{cls_id}")
            print(f"\n{class_name}:")
            for i, feature in enumerate(analysis['top_features'][:5]):
                print(
                    f"  {i+1}. Feature {feature['feature_id']}: "
                    f"corr={feature['abs_correlation']:.4f}, "
                    f"occ={feature['class_occurrences']}, "
                    f"tot={feature['total_occurrences']}, "
                    f"classes={feature['class_diversity']}"
                )

    def _plot_top_correlations_per_class(self, results: Dict[str, Any], layer_idx: int, top_k: int):
        """Create visualizations for top correlations per class analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f'Top {top_k} Correlations Per Class Analysis {"- Layer " + str(layer_idx) if layer_idx else ""}',
            fontsize=16,
            fontweight='bold'
        )

        # Extract data for plotting
        class_names = [IDX2CLS.get(cls_id, f"Class_{cls_id}") for cls_id in results.keys()]
        class_ids = list(results.keys())

        # 1. Mean correlation per class
        ax1 = axes[0, 0]
        mean_corrs = [results[cls_id]['statistics']['mean_correlation'] for cls_id in class_ids]
        bars1 = ax1.bar(range(len(class_names)), mean_corrs, color=self.colors[:len(class_names)], alpha=0.8)
        ax1.set_title(f'Mean of Top {top_k} Correlations per Class')
        ax1.set_ylabel('Mean Absolute Correlation')
        ax1.set_xticks(range(len(class_names)))
        ax1.set_xticklabels(class_names, rotation=45, ha='right')

        # Add value labels on bars
        for bar, val in zip(bars1, mean_corrs):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.005,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

        # 2. Max vs Min correlation per class
        ax2 = axes[0, 1]
        max_corrs = [results[cls_id]['statistics']['max_correlation'] for cls_id in class_ids]
        min_corrs = [results[cls_id]['statistics']['min_correlation'] for cls_id in class_ids]

        x = np.arange(len(class_names))
        width = 0.35
        ax2.bar(x - width / 2, max_corrs, width, label='Max', color='red', alpha=0.7)
        ax2.bar(x + width / 2, min_corrs, width, label='Min', color='blue', alpha=0.7)
        ax2.set_title('Max vs Min Correlations per Class')
        ax2.set_ylabel('Absolute Correlation')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.legend()

        # 3. Polysemantic score vs sparsity
        ax3 = axes[0, 2]
        polysemantic_scores = [results[cls_id]['statistics']['mean_polysemantic_score'] for cls_id in class_ids]
        sparsity_scores = [results[cls_id]['statistics']['sparsity_indicator'] for cls_id in class_ids]

        scatter = ax3.scatter(
            polysemantic_scores, sparsity_scores, c=mean_corrs, cmap='plasma', s=100, alpha=0.7, edgecolors='black'
        )
        ax3.set_xlabel('Mean Polysemantic Score')
        ax3.set_ylabel('Sparsity Indicator')
        ax3.set_title('Polysemantic vs Sparsity')
        plt.colorbar(scatter, ax=ax3, label='Mean Correlation')

        # Add class labels to scatter points
        for i, cls_name in enumerate(class_names):
            ax3.annotate(
                cls_name[:4], (polysemantic_scores[i], sparsity_scores[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

        # 4. Average occurrences comparison
        ax4 = axes[1, 0]
        class_occs = [results[cls_id]['statistics']['mean_class_occurrences'] for cls_id in class_ids]
        total_occs = [results[cls_id]['statistics']['mean_total_occurrences'] for cls_id in class_ids]

        x = np.arange(len(class_names))
        width = 0.35
        ax4.bar(x - width / 2, class_occs, width, label='Class Specific', color='green', alpha=0.7)
        ax4.bar(x + width / 2, total_occs, width, label='Total', color='orange', alpha=0.7)
        ax4.set_title('Mean Occurrences: Class vs Total')
        ax4.set_ylabel('Mean Occurrences')
        ax4.set_xticks(x)
        ax4.set_xticklabels(class_names, rotation=45, ha='right')
        ax4.legend()

        # 5. Correlation distribution violin plot
        ax5 = axes[1, 1]
        correlation_data = []
        for cls_id in class_ids:
            top_features = results[cls_id]['top_features']
            correlations = [f['abs_correlation'] for f in top_features]
            correlation_data.append(correlations)

        parts = ax5.violinplot(correlation_data, showmeans=True, showmedians=True)
        for pc, color in zip(parts['bodies'], self.colors):
            pc.set_facecolor(color)
        ax5.set_title('Correlation Distribution per Class')
        ax5.set_ylabel('Absolute Correlation')
        ax5.set_xticks(range(1, len(class_names) + 1))
        ax5.set_xticklabels(class_names, rotation=45, ha='right')

        # 6. Feature diversity heatmap
        ax6 = axes[1, 2]
        diversity_matrix = []
        for cls_id in class_ids:
            top_features = results[cls_id]['top_features']
            diversities = [f['class_diversity'] for f in top_features]
            diversity_matrix.append(diversities)

        # Pad with zeros to make rectangular matrix
        max_len = max(len(row) for row in diversity_matrix)
        diversity_matrix = [row + [0] * (max_len - len(row)) for row in diversity_matrix]

        im = ax6.imshow(diversity_matrix, cmap='YlOrRd', aspect='auto')
        ax6.set_title('Feature Class Diversity Heatmap')
        ax6.set_ylabel('Classes')
        ax6.set_xlabel(f'Top {top_k} Features (ranked by correlation)')
        ax6.set_yticks(range(len(class_names)))
        ax6.set_yticklabels(class_names)
        plt.colorbar(im, ax=ax6, label='Number of Classes')

        plt.tight_layout()

        if self.save_plots:
            suffix = f"_layer{layer_idx}" if layer_idx is not None else ""
            plt.savefig(self.output_dir / f'top_correlations_per_class{suffix}.png', dpi=300, bbox_inches='tight')

        plt.show()


def run_steerability_analysis_per_layer(analyzer, layer_idx, dict_path):
    """Run comprehensive steerability analysis for a single layer"""
    print(f"\n{'='*80}")
    print(f"STEERABILITY ANALYSIS FOR LAYER {layer_idx}")
    print(f"{'='*80}")

    # Load dictionary
    dict_data = analyzer.load_correlation_dict(dict_path)

    # Check if steerability data exists
    has_steerability = any('mean_steerability' in stats for stats in dict_data['feature_stats'].values())
    if not has_steerability:
        print(f"No steerability data found in layer {layer_idx} dictionary!")
        return

    # Extract class data (including steerability)
    class_data = analyzer.extract_class_data(dict_data)

    # Run steerability analyses
    print(f"\n1. Analyzing steerability distributions...")
    steer_distributions = analyzer.analyze_steerability_distributions(class_data, layer_idx)

    print(f"\n2. Analyzing top steerable features...")
    top_steerable = analyzer.analyze_top_steerable_features(dict_data, top_k=20)

    print(f"\n3. Analyzing correlation-steerability overlap...")
    overlap_analysis = analyzer.analyze_correlation_steerability_overlap(dict_data, top_k=50)

    print(f"\n4. Analyzing top features per class...")
    per_class_analysis = analyzer.analyze_top_features_per_class(dict_data, top_k=5)

    print(f"\n5. Generating steerability visualizations...")
    analyzer.plot_steerability_analysis(dict_data, class_data, layer_idx)

    print(f"\n6. Generating per-class visualizations...")
    analyzer.plot_per_class_features_analysis(per_class_analysis, layer_idx)

    print(f"\nSteerability analysis complete for layer {layer_idx}!")
    return {
        'steer_distributions': steer_distributions,
        'top_steerable': top_steerable,
        'overlap_analysis': overlap_analysis,
        'per_class_analysis': per_class_analysis
    }


def run_analysis_per_layer(analyzer, layer_idx, dict_path):
    # Correlation outliers analysis - THE NEW ANALYSIS YOU REQUESTED
    print(f"Running correlation outliers analysis for layer {layer_idx}...")
    outlier_results = analyzer.analyze_outliers_only(
        layer_idx=layer_idx,
        min_correlation=0.3,  # Adjust to find features with correlation >= 0.3
        dict_path=dict_path
    )

    # Optional: Special features analysis
    print(f"\nRunning special features analysis for layer {layer_idx}...")
    special_features = analyzer.analyze_special_features_only(
        layer_idx=layer_idx, occurrence_threshold=1, correlation_threshold=0.2, dict_path=dict_path
    )

    print(f"\nRunning top correlations per class analysis for layer {layer_idx}...")
    top_correlations_results = analyzer.analyze_top_correlations_per_class(
        layer_idx=layer_idx, dict_path=dict_path, top_k=10
    )

    print("\nAnalysis complete! Check the './analysis_results' directory for saved plots.")


def run_all():
    for layer_idx in range(1, 11):
        analyzer = CorrelationDictAnalyzer(save_plots=True)
        dict_path = f"./sae_dictionaries/sfaf_stealth_l{layer_idx}_alignment_min3_128k64.pt"
        run_analysis_per_layer(analyzer, layer_idx, dict_path)


def main():
    """Main execution function"""
    # Test steerability analysis on a single layer
    for layer_idx in range(2, 11):
        analyzer = CorrelationDictAnalyzer(save_plots=True)
        dict_path = f"./sae_dictionaries/steer_corr_l{layer_idx}_alignment_min1_128k64.pt"

        # Run steerability analysis
        run_steerability_analysis_per_layer(analyzer, layer_idx, dict_path)


if __name__ == "__main__":
    main()
