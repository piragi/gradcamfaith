"""
Head Analysis Visualization Module - Class Agnostic Version
Generates comprehensive visualizations for ViT head analysis research
"""

import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from vit.model import CLASSES


class ClassConfig:
    """Configuration for dataset classes."""

    def __init__(self, classes: List[str]):
        """
        Initialize class configuration.
        
        Args:
            classes: List of class names
        """
        self.CLASSES = classes
        self.NUM_CLASSES = len(classes)
        self.IDX2CLS = {i: cls for i, cls in enumerate(classes)}
        self.CLS2IDX = {cls: i for i, cls in enumerate(classes)}

        # Generate colors dynamically
        self.CLASS_COLORS = self._generate_colors()

    def _generate_colors(self) -> Dict[int, str]:
        """Generate distinct colors for each class."""
        if self.NUM_CLASSES <= 7:
            # Use predefined colors for small number of classes
            predefined = [
                '#FF6B6B',  # Red
                '#4ECDC4',  # Teal
                '#45B7D1',  # Blue
                '#96CEB4',  # Green
                '#FFEAA7',  # Yellow
                '#DDA0DD',  # Plum
                '#FFA07A'  # Light salmon
            ]
            return {i: predefined[i] for i in range(self.NUM_CLASSES)}
        else:
            # Use a colormap for larger number of classes
            cmap = get_cmap('tab20' if self.NUM_CLASSES <= 20 else 'hsv')
            colors = {}
            for i in range(self.NUM_CLASSES):
                rgba = cmap(i / (self.NUM_CLASSES - 1))
                colors[i] = '#{:02x}{:02x}{:02x}'.format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            return colors


class HeadAnalysisVisualizer:
    """Comprehensive visualization suite for head analysis results."""

    def __init__(self, save_dir: Path, class_config: ClassConfig):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
            class_config: ClassConfig instance with class information
        """
        self.save_dir = save_dir
        self.viz_dir = save_dir / "visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)

        self.config = class_config
        self.class_names = self.config.IDX2CLS
        self.class_colors = self.config.CLASS_COLORS
        self.num_classes = self.config.NUM_CLASSES

        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        self.descriptions = {}

    def _get_subplot_layout(self, n_items: int, max_cols: int = 4) -> Tuple[int, int]:
        """
        Calculate optimal subplot layout for n items.
        
        Args:
            n_items: Number of items to plot
            max_cols: Maximum number of columns
            
        Returns:
            (n_rows, n_cols) tuple
        """
        n_cols = min(n_items, max_cols)
        n_rows = (n_items + n_cols - 1) // n_cols
        return n_rows, n_cols

    def clear_memory(self):
        """Clear matplotlib and Python memory."""
        plt.clf()
        plt.close('all')
        gc.collect()

    def generate_all_visualizations(
        self,
        direction_similarities: Dict,
        head_importance: Dict,
        token_patterns: Dict,
        num_layers: int = 5,
        start_layer: int = 7
    ):
        """Generate all visualizations and descriptions."""

        print(f"Generating comprehensive visualizations for {self.num_classes}-class analysis...")

        # 1. Head Importance Heatmaps
        print("1/8 - Head Importance Heatmaps...")
        self._visualize_head_importance(head_importance, num_layers, start_layer)
        self.clear_memory()

        # 2. Similarity Distribution Analysis
        print("2/8 - Similarity Distributions...")
        # self._visualize_similarity_distributions(direction_similarities)
        self.clear_memory()

        # 3. Top Contributing Heads Comparison
        print("3/8 - Top Heads Comparison...")
        self._visualize_top_heads_comparison(head_importance, start_layer)
        self.clear_memory()

        # 4. Token Pattern Analysis (PCA/Clustering)
        print("4/8 - Token Pattern Clustering...")
        self._visualize_token_patterns(token_patterns, start_layer)
        self.clear_memory()

        # 5. Class-wise Head Activation Patterns
        print("5/8 - Class Activation Patterns...")
        self._visualize_class_activation_patterns(direction_similarities, num_layers)
        self.clear_memory()

        # 6. Statistical Summary Tables
        print("6/8 - Head Specialization...")
        self._generate_summary_tables(head_importance, token_patterns)
        self.clear_memory()

        # 7. Inter-class Head Specialization
        print("7/8 - Layer Progression...")
        self._visualize_head_specialization(head_importance, num_layers, start_layer)
        self.clear_memory()

        # 8. Layer-wise Importance Progression
        print("8/8 - Summary Tables...")
        self._visualize_layer_progression(head_importance, num_layers, start_layer)
        self.clear_memory()

        # Save descriptions
        self._save_descriptions()

        print(f"All visualizations saved to {self.viz_dir}")

    def _visualize_head_importance(self, head_importance: Dict, num_layers: int, start_layer: int):
        """Create heatmaps showing head importance for each class."""

        # Dynamic layout based on number of classes
        n_rows, n_cols = self._get_subplot_layout(self.num_classes)

        fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.25, hspace=0.3)

        importance_data = head_importance['class_head_importance']

        for class_idx in range(self.num_classes):
            row = class_idx // n_cols
            col = class_idx % n_cols

            ax = fig.add_subplot(gs[row, col])

            # Create heatmap data
            heatmap_data = importance_data[class_idx]

            # Create custom colormap
            colors = ['white', self.class_colors[class_idx]]
            n_bins = 100
            cmap = sns.blend_palette(colors, as_cmap=True)

            # Plot heatmap
            im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

            # Customize axes
            ax.set_xlabel('Head Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('Layer', fontsize=12, fontweight='bold')
            ax.set_title(
                f'{self.class_names[class_idx].upper()} Head Importance', fontsize=14, fontweight='bold', pad=10
            )

            # Set ticks
            ax.set_xticks(range(12))
            ax.set_yticks(range(num_layers))
            ax.set_yticklabels([f'L{start_layer + i}' for i in range(num_layers)])

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Importance Score', fontsize=10)

            # Add grid
            ax.set_xticks(np.arange(12) - 0.5, minor=True)
            ax.set_yticks(np.arange(num_layers) - 0.5, minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

            # Highlight most important heads
            threshold = 0.5
            for i in range(num_layers):
                for j in range(12):
                    if heatmap_data[i, j] > threshold:
                        rect = Rectangle((j - 0.45, i - 0.45),
                                         0.9,
                                         0.9,
                                         linewidth=2,
                                         edgecolor='black',
                                         facecolor='none',
                                         linestyle='--')
                        ax.add_patch(rect)

        plt.suptitle('Head Importance Analysis Across Classes', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'head_importance_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.descriptions['head_importance_heatmaps'] = f"""
**Head Importance Heatmaps**

These heatmaps visualize the importance of each attention head in the last {num_layers} layers of the ViT model for each class prediction. The importance score (0-1) indicates how frequently a head contributes significantly to the class direction vector.

Key insights:
- Darker colors indicate heads that are more important for the respective class
- Dashed boxes highlight heads with >50% importance score
- Different classes show distinct patterns of head specialization
- Later layers tend to show more class-specific specialization

This visualization helps identify which attention heads are most critical for each class, suggesting potential head pruning strategies or interpretation focal points.
"""

    def _visualize_similarity_distributions(self, direction_similarities: Dict):
        """Visualize distribution of similarity scores across classes."""

        # Dynamic layout
        n_rows, n_cols = self._get_subplot_layout(self.num_classes + 1)  # +1 for overall distribution

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        # Collect similarities by class
        class_similarities = {i: [] for i in range(self.num_classes)}
        all_similarities = []

        for img_data in direction_similarities.values():
            class_idx = img_data['predicted_class']
            if class_idx < self.num_classes:  # Safety check
                similarities = img_data['similarities']['all'].flatten()
                class_similarities[class_idx].extend(similarities)
                all_similarities.extend(similarities)

        # Plot 1: Overall distribution
        ax = axes[0]
        ax.hist(all_similarities, bins=50, alpha=0.7, color='gray', edgecolor='black')
        ax.axvline(
            np.mean(all_similarities), color='red', linestyle='--', label=f'Mean: {np.mean(all_similarities):.3f}'
        )
        ax.axvline(
            np.median(all_similarities),
            color='blue',
            linestyle='--',
            label=f'Median: {np.median(all_similarities):.3f}'
        )
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Overall Similarity Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Class-specific distributions
        for class_idx in range(self.num_classes):
            ax = axes[class_idx + 1]
            similarities = class_similarities[class_idx]

            if similarities:
                ax.hist(similarities, bins=50, alpha=0.7, color=self.class_colors[class_idx], edgecolor='black')
                ax.axvline(
                    np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}'
                )
                ax.set_xlabel('Cosine Similarity', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title(
                    f'{self.class_names[class_idx].upper()} Similarity Distribution', fontsize=14, fontweight='bold'
                )
                ax.legend()
                ax.grid(True, alpha=0.3)

        # Hide any unused subplots
        for idx in range(self.num_classes + 1, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Cosine Similarity Distributions by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'similarity_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_top_heads_comparison(self, head_importance: Dict, start_layer: int):
        """Create comparative visualization of top contributing heads."""

        fig, ax = plt.subplots(figsize=(18, 10))

        # Extract top heads for each class
        top_heads_data = []
        for class_idx in range(self.num_classes):
            ranked_heads = head_importance['class_ranked_heads'][class_idx][:10]
            for rank, (layer, head, score) in enumerate(ranked_heads):
                top_heads_data.append({
                    'Class': self.class_names[class_idx],
                    'Layer': start_layer + layer,
                    'Head': head,
                    'Score': score,
                    'Rank': rank + 1,
                    'Label': f'L{start_layer + layer}H{head}'
                })

        # Create grouped bar plot
        df = pd.DataFrame(top_heads_data)

        # Create position for each bar
        class_positions = {name: i for i, name in enumerate(self.class_names.values())}
        x_positions = []
        colors = []
        heights = []
        labels = []

        for idx, row in df.iterrows():
            class_pos = class_positions[row['Class']]
            x_pos = class_pos * 11 + row['Rank'] - 1
            x_positions.append(x_pos)
            colors.append(self.class_colors[list(self.class_names.values()).index(row['Class'])])
            heights.append(row['Score'])
            labels.append(row['Label'])

        bars = ax.bar(x_positions, heights, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar, label in zip(bars, labels):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                label,
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=45
            )

        # Customize plot
        ax.set_xlabel('Top 10 Heads per Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Top Contributing Heads Comparison Across Classes', fontsize=14, fontweight='bold')

        # Add class labels
        for class_name, pos in class_positions.items():
            ax.text(
                pos * 11 + 4.5,
                -0.05,
                class_name.upper(),
                ha='center',
                transform=ax.get_xaxis_transform(),
                fontsize=12,
                fontweight='bold'
            )

        # Remove x-ticks
        ax.set_xticks([])
        ax.set_ylim(0, max(heights) * 1.2)
        ax.grid(True, axis='y', alpha=0.3)

        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=self.class_colors[i], label=self.class_names[i].upper())
            for i in range(self.num_classes)
        ]
        ax.legend(handles=legend_elements, loc='upper right', ncol=min(3, self.num_classes))

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'top_heads_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_token_patterns(self, token_patterns: Dict, start_layer: int):
        """Visualize token activation patterns using PCA/clustering results."""

        # Dynamic layout
        n_rows, n_cols = self._get_subplot_layout(len(token_patterns))

        fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.25, hspace=0.3)

        for idx, (class_idx, patterns) in enumerate(token_patterns.items()):
            row = idx // n_cols
            col = idx % n_cols

            ax = fig.add_subplot(gs[row, col])

            # Plot PCA results with clusters
            pca_result = patterns['pca_result']
            cluster_labels = patterns['cluster_labels']

            # Create scatter plot
            scatter = ax.scatter(
                pca_result[:, 0],
                pca_result[:, 1],
                c=cluster_labels,
                cmap='viridis',
                s=50,
                alpha=0.7,
                edgecolors='black'
            )

            # Add cluster centers
            for cluster_info in patterns['clusters']:
                cluster_id = cluster_info['cluster_id']
                cluster_points = pca_result[cluster_labels == cluster_id]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    ax.scatter(
                        center[0],
                        center[1],
                        c='red',
                        s=200,
                        marker='*',
                        edgecolors='black',
                        linewidth=2,
                        label=f'Cluster {cluster_id+1} center'
                    )

            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_title(
                f'{self.class_names[class_idx].upper()} Token Activation Patterns', fontsize=14, fontweight='bold'
            )
            ax.grid(True, alpha=0.3)

            # Add variance explained
            var_explained = sum(patterns['explained_variance'][:2])
            ax.text(
                0.02,
                0.98,
                f'Variance explained: {var_explained:.2%}',
                transform=ax.transAxes,
                va='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        plt.suptitle('Token Activation Pattern Clustering (PCA Visualization)', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'token_pattern_clustering.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create detailed cluster analysis figure
        self._visualize_cluster_details(token_patterns, start_layer)

    def _visualize_cluster_details(self, token_patterns: Dict, start_layer: int):
        """Create detailed visualization of cluster characteristics."""

        n_classes_with_patterns = len(token_patterns)
        fig, axes = plt.subplots(n_classes_with_patterns, 3, figsize=(18, 4 * n_classes_with_patterns))

        if n_classes_with_patterns == 1:
            axes = axes.reshape(1, -1)

        for idx, (class_idx, patterns) in enumerate(token_patterns.items()):
            row_axes = axes[idx]

            # Plot 1: Cluster sizes
            ax = row_axes[0]
            cluster_sizes = [c['size'] for c in patterns['clusters']]
            cluster_names = [f'C{c["cluster_id"]+1}' for c in patterns['clusters']]
            bars = ax.bar(
                cluster_names, cluster_sizes, color=self.class_colors[class_idx], alpha=0.7, edgecolor='black'
            )
            ax.set_xlabel('Cluster', fontsize=10)
            ax.set_ylabel('Number of Images', fontsize=10)
            ax.set_title(f'{self.class_names[class_idx].upper()} - Cluster Sizes', fontsize=12)

            # Add percentage labels
            for bar, cluster in zip(bars, patterns['clusters']):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.5,
                    f'{cluster["percentage"]*100:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

            # Plot 2: Top activations heatmap
            ax = row_axes[1]

            # Create activation matrix for top tokens across clusters
            n_clusters = len(patterns['clusters'])
            n_top = 5  # Show top 5 activations per cluster
            activation_matrix = np.zeros((n_clusters, n_top))
            activation_labels = []

            for i, cluster in enumerate(patterns['clusters']):
                top_acts = sorted(cluster['top_activations'], key=lambda x: x[3], reverse=True)[:n_top]
                for j, (l, h, t, v) in enumerate(top_acts):
                    activation_matrix[i, j] = v
                    if i == 0:  # Only create labels once
                        activation_labels.append(f'L{start_layer+l}H{h}T{t}')

            im = ax.imshow(activation_matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(n_top))
            ax.set_xticklabels(activation_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(n_clusters))
            ax.set_yticklabels([f'C{i+1}' for i in range(n_clusters)])
            ax.set_xlabel('Top Activated Positions', fontsize=10)
            ax.set_ylabel('Cluster', fontsize=10)
            ax.set_title(f'{self.class_names[class_idx].upper()} - Top Activations', fontsize=12)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            # Plot 3: PCA variance explained
            ax = row_axes[2]
            var_explained = patterns['explained_variance'][:10] * 100
            ax.bar(
                range(1,
                      len(var_explained) + 1),
                var_explained,
                color=self.class_colors[class_idx],
                alpha=0.7,
                edgecolor='black'
            )
            ax.set_xlabel('Principal Component', fontsize=10)
            ax.set_ylabel('Variance Explained (%)', fontsize=10)
            ax.set_title(f'{self.class_names[class_idx].upper()} - PCA Variance', fontsize=12)
            ax.set_xticks(range(1, len(var_explained) + 1))

            # Add cumulative line
            cumsum = np.cumsum(var_explained)
            ax2 = ax.twinx()
            ax2.plot(range(1, len(var_explained) + 1), cumsum, 'r-o', markersize=4, linewidth=2, label='Cumulative')
            ax2.set_ylabel('Cumulative Variance (%)', fontsize=10, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, 100)

        plt.suptitle('Detailed Token Pattern Cluster Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'cluster_details.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_class_activation_patterns(self, direction_similarities: Dict, num_layers: int):
        """Visualize activation patterns across token positions."""

        # Dynamic layout
        n_rows, n_cols = self._get_subplot_layout(self.num_classes)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        # Aggregate activation patterns by class
        for class_idx in range(self.num_classes):
            ax = axes[class_idx]

            # Collect all similarity matrices for this class
            class_similarities = []
            for img_data in direction_similarities.values():
                if img_data['predicted_class'] == class_idx:
                    # Average across layers and heads
                    sim_matrix = img_data['similarities']['all']
                    avg_by_token = np.mean(sim_matrix, axis=(0, 1))  # Average over layers and heads
                    class_similarities.append(avg_by_token)

            if class_similarities:
                # Calculate mean and std
                similarities_array = np.array(class_similarities)
                mean_pattern = np.mean(similarities_array, axis=0)
                std_pattern = np.std(similarities_array, axis=0)

                # Limit to first 50 tokens for visibility
                n_tokens = min(50, len(mean_pattern))
                x = np.arange(n_tokens)

                # Plot with error bars
                ax.bar(
                    x,
                    mean_pattern[:n_tokens],
                    yerr=std_pattern[:n_tokens],
                    color=self.class_colors[class_idx],
                    alpha=0.7,
                    edgecolor='black',
                    error_kw={
                        'elinewidth': 1,
                        'alpha': 0.5
                    }
                )

                # Highlight CLS token
                ax.bar(0, mean_pattern[0], color='red', alpha=0.8, edgecolor='black', linewidth=2, label='CLS Token')

                ax.set_xlabel('Token Position', fontsize=12)
                ax.set_ylabel('Mean Similarity Score', fontsize=12)
                ax.set_title(
                    f'{self.class_names[class_idx].upper()} Token Activation Pattern', fontsize=14, fontweight='bold'
                )
                ax.legend()
                ax.grid(True, axis='y', alpha=0.3)

                # Add annotation for top tokens
                top_tokens = np.argsort(mean_pattern[:n_tokens])[-5:]
                for token_idx in top_tokens:
                    if token_idx != 0:  # Skip CLS token as it's already highlighted
                        ax.annotate(
                            f'T{token_idx}',
                            xy=(token_idx, mean_pattern[token_idx]),
                            xytext=(token_idx, mean_pattern[token_idx] + 0.05),
                            ha='center',
                            fontsize=8
                        )

        # Hide any unused subplots
        for idx in range(self.num_classes, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Average Token Activation Patterns by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'token_activation_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_head_specialization(self, head_importance: Dict, num_layers: int, start_layer: int):
        """Visualize head specialization across classes."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Calculate specialization scores
        importance_data = head_importance['class_head_importance']

        # Reshape data for analysis
        n_heads_total = num_layers * 12
        head_scores = np.zeros((n_heads_total, self.num_classes))

        for class_idx in range(self.num_classes):
            class_importance = importance_data[class_idx].flatten()
            head_scores[:, class_idx] = class_importance

        # Calculate specialization metrics
        # 1. Entropy-based specialization (lower = more specialized)
        epsilon = 1e-10
        head_probs = head_scores / (head_scores.sum(axis=1, keepdims=True) + epsilon)
        head_entropy = -np.sum(head_probs * np.log(head_probs + epsilon), axis=1)

        # 2. Max-class dominance
        max_class_score = np.max(head_scores, axis=1)
        dominant_class = np.argmax(head_scores, axis=1)

        # Plot 1: Specialization heatmap
        specialization_matrix = 1 - (head_entropy / np.log(self.num_classes))  # Normalize to [0,1]
        specialization_matrix = specialization_matrix.reshape(num_layers, 12)

        im1 = ax1.imshow(specialization_matrix, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
        ax1.set_xlabel('Head Index', fontsize=12)
        ax1.set_ylabel('Layer', fontsize=12)
        ax1.set_title('Head Specialization Score\n(Blue = Specialized, Red = General)', fontsize=14, fontweight='bold')
        ax1.set_yticks(range(num_layers))
        ax1.set_yticklabels([f'L{start_layer + i}' for i in range(num_layers)])
        ax1.set_xticks(range(12))

        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Specialization Score', fontsize=10)

        # Plot 2: Class assignment
        class_assignment = dominant_class.reshape(num_layers, 12)

        # Create custom colormap for classes
        colors_list = [self.class_colors[i] for i in range(self.num_classes)]
        from matplotlib.colors import ListedColormap
        cmap_classes = ListedColormap(colors_list)

        im2 = ax2.imshow(class_assignment, cmap=cmap_classes, aspect='auto', vmin=0, vmax=self.num_classes - 1)
        ax2.set_xlabel('Head Index', fontsize=12)
        ax2.set_ylabel('Layer', fontsize=12)
        ax2.set_title('Dominant Class per Head', fontsize=14, fontweight='bold')
        ax2.set_yticks(range(num_layers))
        ax2.set_yticklabels([f'L{start_layer + i}' for i in range(num_layers)])
        ax2.set_xticks(range(12))

        # Create discrete colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, ticks=list(range(self.num_classes)))
        cbar2.set_ticklabels([self.class_names[i].upper() for i in range(self.num_classes)])

        # Add significance markers for highly specialized heads
        threshold = 0.7
        for i in range(num_layers):
            for j in range(12):
                if specialization_matrix[i, j] > threshold:
                    ax1.plot(j, i, 'k*', markersize=10)
                    ax2.plot(j, i, 'k*', markersize=10)

        plt.suptitle('Head Specialization Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'head_specialization.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_layer_progression(self, head_importance: Dict, num_layers: int, start_layer: int):
        """Visualize how importance progresses through layers."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        importance_data = head_importance['class_head_importance']

        # Calculate layer-wise statistics
        layer_importance = {class_idx: [] for class_idx in range(self.num_classes)}
        layer_diversity = []

        for layer in range(num_layers):
            # Average importance per layer for each class
            for class_idx in range(self.num_classes):
                avg_importance = np.mean(importance_data[class_idx][layer])
                layer_importance[class_idx].append(avg_importance)

            # Calculate diversity (std across heads)
            all_heads = []
            for class_idx in range(self.num_classes):
                all_heads.extend(importance_data[class_idx][layer])
            layer_diversity.append(np.std(all_heads))

        # Plot 1: Layer-wise importance progression
        x = np.arange(num_layers)
        width = 0.8 / self.num_classes  # Dynamic width based on number of classes

        for class_idx in range(self.num_classes):
            offset = (class_idx - self.num_classes / 2 + 0.5) * width
            bars = ax1.bar(
                x + offset,
                layer_importance[class_idx],
                width,
                label=self.class_names[class_idx].upper(),
                color=self.class_colors[class_idx],
                alpha=0.8,
                edgecolor='black'
            )

        ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Head Importance', fontsize=12, fontweight='bold')
        ax1.set_title('Layer-wise Importance Progression', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'L{start_layer + i}' for i in range(num_layers)])
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, axis='y', alpha=0.3)

        # Plot 2: Cumulative importance and diversity
        ax2_twin = ax2.twinx()

        # Cumulative importance
        for class_idx in range(self.num_classes):
            cumulative = np.cumsum(layer_importance[class_idx])
            ax2.plot(
                x,
                cumulative,
                'o-',
                color=self.class_colors[class_idx],
                linewidth=2,
                markersize=8,
                label=self.class_names[class_idx].upper()
            )

        # Diversity
        ax2_twin.plot(x, layer_diversity, 's--', color='purple', linewidth=2, markersize=8, label='Head Diversity')

        ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Importance', fontsize=12, fontweight='bold')
        ax2_twin.set_ylabel('Head Diversity (Std)', fontsize=12, fontweight='bold', color='purple')
        ax2.set_title('Cumulative Importance and Head Diversity', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'L{start_layer + i}' for i in range(num_layers)])
        ax2_twin.tick_params(axis='y', labelcolor='purple')

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Layer-wise Analysis of Head Contributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'layer_progression.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_summary_tables(self, head_importance: Dict, token_patterns: Dict):
        """Generate summary tables for key findings."""

        # Dynamic figure size based on number of classes
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Statistical Summary Tables', fontsize=16, fontweight='bold')

        # Table 1: Top heads per class
        ax1 = plt.subplot(2, 2, 1)
        ax1.axis('tight')
        ax1.axis('off')

        table_data = []
        for class_idx in range(self.num_classes):
            if class_idx in head_importance['class_ranked_heads']:
                top_heads = head_importance['class_ranked_heads'][class_idx][:5]
                for rank, (layer, head, score) in enumerate(top_heads, 1):
                    table_data.append([
                        self.class_names[class_idx].upper(), rank, f'L{layer + 7}H{head}', f'{score:.3f}',
                        f'{score*100:.1f}%'
                    ])

        table1 = ax1.table(
            cellText=table_data,
            colLabels=['Class', 'Rank', 'Head', 'Score', 'Importance %'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table1.auto_set_font_size(False)
        table1.set_fontsize(8)
        table1.scale(1.2, 1.5)
        ax1.set_title('Top 5 Important Heads per Class', fontsize=12, fontweight='bold', pad=20)

        # Table 2: Class statistics
        ax2 = plt.subplot(2, 2, 2)
        ax2.axis('tight')
        ax2.axis('off')

        stats_data = []
        for class_idx in range(self.num_classes):
            n_images = head_importance['class_image_counts'].get(class_idx, 0)
            importance_scores = head_importance['class_head_importance'][class_idx].flatten()
            stats_data.append([
                self.class_names[class_idx].upper(), n_images, f'{np.mean(importance_scores):.3f}',
                f'{np.std(importance_scores):.3f}', f'{np.sum(importance_scores > 0.5)}'
            ])

        table2 = ax2.table(
            cellText=stats_data,
            colLabels=['Class', 'Images', 'Mean Import.', 'Std Import.', 'High Import. Heads'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(8)
        table2.scale(1.2, 1.5)
        ax2.set_title('Class-wise Statistics', fontsize=12, fontweight='bold', pad=20)

        # Table 3: Token pattern clusters
        ax3 = plt.subplot(2, 2, 3)
        ax3.axis('tight')
        ax3.axis('off')

        cluster_data = []
        for class_idx, patterns in token_patterns.items():
            for cluster in patterns['clusters']:
                cluster_data.append([
                    self.class_names[class_idx].upper(), f"Cluster {cluster['cluster_id']+1}", cluster['size'],
                    f"{cluster['percentage']*100:.1f}%",
                    len(cluster['top_activations'])
                ])

        table3 = ax3.table(
            cellText=cluster_data,
            colLabels=['Class', 'Cluster', 'Size', 'Percentage', 'Key Activations'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table3.auto_set_font_size(False)
        table3.set_fontsize(8)
        table3.scale(1.2, 1.5)
        ax3.set_title('Token Pattern Cluster Summary', fontsize=12, fontweight='bold', pad=20)

        # Table 4: Similarity statistics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('tight')
        ax4.axis('off')

        sim_stats = head_importance['similarity_stats']
        stats_rows = [['Mean', f"{sim_stats['mean']:.4f}"], ['Median', f"{sim_stats['median']:.4f}"],
                      ['Std Dev', f"{sim_stats['std']:.4f}"], ['Min', f"{sim_stats['min']:.4f}"],
                      ['Max', f"{sim_stats['max']:.4f}"], ['25th Percentile', f"{sim_stats['percentiles']['25']:.4f}"],
                      ['75th Percentile', f"{sim_stats['percentiles']['75']:.4f}"],
                      ['90th Percentile', f"{sim_stats['percentiles']['90']:.4f}"]]

        table4 = ax4.table(
            cellText=stats_rows, colLabels=['Metric', 'Value'], cellLoc='center', loc='center', bbox=[0, 0, 1, 1]
        )
        table4.auto_set_font_size(False)
        table4.set_fontsize(9)
        table4.scale(1.2, 1.5)
        ax4.set_title('Similarity Score Statistics', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'summary_tables.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _save_descriptions(self):
        """Save all descriptions to a markdown file."""

        with open(self.viz_dir / 'visualization_descriptions.md', 'w') as f:
            f.write(f"# Head Analysis Visualization Descriptions ({self.num_classes} Classes)\n\n")
            f.write("This document provides detailed descriptions of all generated visualizations.\n\n")

            for viz_name, description in self.descriptions.items():
                f.write(f"## {viz_name.replace('_', ' ').title()}\n\n")
                f.write(description)
                f.write("\n\n---\n\n")

        print(f"Visualization descriptions saved to {self.viz_dir / 'visualization_descriptions.md'}")


def create_comprehensive_visualizations(
    data_dir: Path, classes: List[str] = None, num_layers: int = 5, start_layer: int = 7
):
    """
    Main function to load data and create all visualizations.
    
    Args:
        data_dir: Directory containing the analysis output files
        classes: List of class names. If None, will try to infer from data
        num_layers: Number of layers analyzed
        start_layer: Starting layer index
    """

    # Load the analysis results
    print("Loading analysis data...")

    # Load head direction similarities
    similarities_path = data_dir / "head_direction_similarities.npy"
    direction_similarities = np.load(similarities_path, allow_pickle=True).item()

    # Load head importance analysis
    importance_path = data_dir / "head_importance_analysis.npy"
    head_importance = np.load(importance_path, allow_pickle=True).item()

    # If classes not provided, try to infer from data
    if classes is None:
        # Try to get number of classes from the data
        if 'class_head_importance' in head_importance:
            num_classes = len(head_importance['class_head_importance'])
            classes = [f"class_{i}" for i in range(num_classes)]
            print(f"Inferred {num_classes} classes from data")
        else:
            raise ValueError("Cannot infer number of classes. Please provide class names.")

    # Create class configuration
    class_config = ClassConfig(classes)

    # Load token patterns for each class
    token_patterns = {}
    for class_idx in range(class_config.NUM_CLASSES):
        pattern_path = data_dir / f"token_patterns_class_{class_idx}.npy"
        if pattern_path.exists():
            token_patterns[class_idx] = np.load(pattern_path, allow_pickle=True).item()

    # Create visualizer and generate all visualizations
    visualizer = HeadAnalysisVisualizer(data_dir, class_config)
    visualizer.generate_all_visualizations(
        direction_similarities, head_importance, token_patterns, num_layers=num_layers, start_layer=start_layer
    )

    print("\nVisualization generation complete!")
    print(f"All visualizations saved to: {data_dir / 'visualizations'}")
    print(f"Descriptions saved to: {data_dir / 'visualizations' / 'visualization_descriptions.md'}")


# Example usage for different datasets
if __name__ == "__main__":
    # For your new 6-class dataset:
    data_directory = Path("./results/train/head_analysis")
    create_comprehensive_visualizations(data_directory, classes=CLASSES, num_layers=4, start_layer=8)
