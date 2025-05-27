"""
Head Analysis Visualization Module
Generates comprehensive visualizations for ViT head analysis research
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd


class HeadAnalysisVisualizer:
    """Comprehensive visualization suite for head analysis results."""
    
    def __init__(self, save_dir: Path, class_names: Dict[int, str] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
            class_names: Mapping of class indices to names (default: {0: 'Covid', 1: 'Non-Covid', 2: 'Normal'})
        """
        self.save_dir = save_dir
        self.viz_dir = save_dir / "visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.class_names = class_names or {
            0: 'Covid',
            1: 'Non-Covid', 
            2: 'Normal'
        }
        self.class_colors = {
            0: '#FF6B6B',  # Red for Covid
            1: '#4ECDC4',  # Teal for Non-Covid
            2: '#45B7D1'   # Blue for Normal
        }
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.descriptions = {}
        
    def generate_all_visualizations(self, 
                                  direction_similarities: Dict,
                                  head_importance: Dict,
                                  token_patterns: Dict,
                                  num_layers: int = 5,
                                  start_layer: int = 7):
        """Generate all visualizations and descriptions."""
        
        print("Generating comprehensive visualizations...")
        
        # 1. Head Importance Heatmaps
        self._visualize_head_importance(head_importance, num_layers, start_layer)
        
        # 2. Similarity Distribution Analysis
        self._visualize_similarity_distributions(direction_similarities)
        
        # 3. Top Contributing Heads Comparison
        self._visualize_top_heads_comparison(head_importance, start_layer)
        
        # 4. Token Pattern Analysis (PCA/Clustering)
        self._visualize_token_patterns(token_patterns, start_layer)
        
        # 5. Class-wise Head Activation Patterns
        self._visualize_class_activation_patterns(direction_similarities, num_layers)
        
        # 6. Statistical Summary Tables
        self._generate_summary_tables(head_importance, token_patterns)
        
        # 7. Inter-class Head Specialization
        self._visualize_head_specialization(head_importance, num_layers, start_layer)
        
        # 8. Layer-wise Importance Progression
        self._visualize_layer_progression(head_importance, num_layers, start_layer)
        
        # Save descriptions
        self._save_descriptions()
        
        print(f"All visualizations saved to {self.viz_dir}")
        
    def _visualize_head_importance(self, head_importance: Dict, num_layers: int, start_layer: int):
        """Create heatmaps showing head importance for each class."""
        
        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
        
        importance_data = head_importance['class_head_importance']
        
        for class_idx in range(3):
            ax = fig.add_subplot(gs[0, class_idx])
            
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
            ax.set_title(f'{self.class_names[class_idx]} Head Importance', 
                        fontsize=14, fontweight='bold', pad=10)
            
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
                        rect = Rectangle((j-0.45, i-0.45), 0.9, 0.9, 
                                       linewidth=2, edgecolor='black', 
                                       facecolor='none', linestyle='--')
                        ax.add_patch(rect)
        
        plt.suptitle('Head Importance Analysis Across Classes', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'head_importance_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.descriptions['head_importance_heatmaps'] = """
**Head Importance Heatmaps**

These heatmaps visualize the importance of each attention head in the last 5 layers of the ViT model for each class prediction. The importance score (0-1) indicates how frequently a head contributes significantly to the class direction vector.

Key insights:
- Darker colors indicate heads that are more important for the respective class
- Dashed boxes highlight heads with >50% importance score
- Different classes show distinct patterns of head specialization
- Later layers tend to show more class-specific specialization

This visualization helps identify which attention heads are most critical for each class prediction, suggesting potential head pruning strategies or interpretation focal points.
"""

    def _visualize_similarity_distributions(self, direction_similarities: Dict):
        """Visualize distribution of similarity scores across classes."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Collect similarities by class
        class_similarities = {0: [], 1: [], 2: []}
        all_similarities = []
        
        for img_data in direction_similarities.values():
            class_idx = img_data['predicted_class']
            similarities = img_data['similarities']['all'].flatten()
            class_similarities[class_idx].extend(similarities)
            all_similarities.extend(similarities)
        
        # Plot 1: Overall distribution
        ax = axes[0]
        ax.hist(all_similarities, bins=50, alpha=0.7, color='gray', edgecolor='black')
        ax.axvline(np.mean(all_similarities), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(all_similarities):.3f}')
        ax.axvline(np.median(all_similarities), color='blue', linestyle='--', 
                  label=f'Median: {np.median(all_similarities):.3f}')
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Overall Similarity Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plots 2-4: Class-specific distributions
        for idx, class_idx in enumerate([0, 1, 2]):
            ax = axes[idx + 1]
            similarities = class_similarities[class_idx]
            
            if similarities:
                ax.hist(similarities, bins=50, alpha=0.7, 
                       color=self.class_colors[class_idx], edgecolor='black')
                ax.axvline(np.mean(similarities), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(similarities):.3f}')
                ax.set_xlabel('Cosine Similarity', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title(f'{self.class_names[class_idx]} Similarity Distribution', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Cosine Similarity Distributions by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'similarity_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.descriptions['similarity_distributions'] = """
**Similarity Distribution Analysis**

These histograms show the distribution of cosine similarity scores between head outputs and class direction vectors. The analysis includes both overall distribution and class-specific breakdowns.

Key observations:
- The overall distribution shows the general alignment pattern across all classes
- Class-specific distributions reveal unique activation patterns for each diagnosis
- Mean and median lines indicate central tendencies
- Distribution shape indicates consistency of head contributions

This analysis helps understand:
1. How consistently heads align with class directions
2. Whether certain classes have more focused or dispersed attention patterns
3. The threshold selection for determining "important" heads
"""

    def _visualize_top_heads_comparison(self, head_importance: Dict, start_layer: int):
        """Create comparative visualization of top contributing heads."""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Extract top heads for each class
        top_heads_data = []
        for class_idx in range(3):
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
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   label, ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Customize plot
        ax.set_xlabel('Top 10 Heads per Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Top Contributing Heads Comparison Across Classes', 
                    fontsize=14, fontweight='bold')
        
        # Add class labels
        for class_name, pos in class_positions.items():
            ax.text(pos * 11 + 4.5, -0.05, class_name, ha='center', 
                   transform=ax.get_xaxis_transform(), fontsize=12, fontweight='bold')
        
        # Remove x-ticks
        ax.set_xticks([])
        ax.set_ylim(0, max(heights) * 1.2)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, fc=self.class_colors[i], 
                                       label=self.class_names[i]) for i in range(3)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'top_heads_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.descriptions['top_heads_comparison'] = """
**Top Contributing Heads Comparison**

This bar chart compares the top 10 most important attention heads for each class. Each bar represents a specific layer-head combination (e.g., L11H7 = Layer 11, Head 7) with its importance score.

Key findings:
- Each class relies on a distinct set of attention heads
- Some heads may appear in multiple classes but with different importance levels
- Later layers (L10-L11) tend to dominate the top positions
- The importance scores indicate how consistently these heads contribute to correct predictions

This comparison helps identify:
1. Class-specific attention mechanisms
2. Shared vs. unique processing pathways
3. Potential targets for interpretability studies
"""

    def _visualize_token_patterns(self, token_patterns: Dict, start_layer: int):
        """Visualize token activation patterns using PCA/clustering results."""
        
        # Create figure with subplots for each class
        fig = plt.figure(figsize=(20, 6))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
        
        for idx, (class_idx, patterns) in enumerate(token_patterns.items()):
            ax = fig.add_subplot(gs[0, idx])
            
            # Plot PCA results with clusters
            pca_result = patterns['pca_result']
            cluster_labels = patterns['cluster_labels']
            
            # Create scatter plot
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                               c=cluster_labels, cmap='viridis', 
                               s=50, alpha=0.7, edgecolors='black')
            
            # Add cluster centers
            for cluster_info in patterns['clusters']:
                cluster_id = cluster_info['cluster_id']
                cluster_points = pca_result[cluster_labels == cluster_id]
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    ax.scatter(center[0], center[1], c='red', s=200, 
                             marker='*', edgecolors='black', linewidth=2,
                             label=f'Cluster {cluster_id+1} center')
            
            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_title(f'{self.class_names[class_idx]} Token Activation Patterns', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add variance explained
            var_explained = sum(patterns['explained_variance'][:2])
            ax.text(0.02, 0.98, f'Variance explained: {var_explained:.2%}',
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Token Activation Pattern Clustering (PCA Visualization)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'token_pattern_clustering.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed cluster analysis figure
        self._visualize_cluster_details(token_patterns, start_layer)
        
        self.descriptions['token_pattern_clustering'] = """
**Token Activation Pattern Analysis**

These visualizations show the results of PCA and k-means clustering applied to token activation patterns for each class. Each point represents an image, positioned based on its activation pattern similarity.

Key insights:
- Distinct clusters indicate different activation pattern "modes" within each class
- Cluster centers (red stars) represent prototypical activation patterns
- Variance explained shows how well 2D PCA captures the pattern diversity
- Cluster separation indicates heterogeneity in processing strategies

This analysis reveals:
1. Whether images within a class are processed uniformly or through multiple pathways
2. Potential sub-types within each diagnostic category
3. The complexity of the learned representations
"""

    def _visualize_cluster_details(self, token_patterns: Dict, start_layer: int):
        """Create detailed visualization of cluster characteristics."""
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        for class_idx, patterns in token_patterns.items():
            row_axes = axes[class_idx]
            
            # Plot 1: Cluster sizes
            ax = row_axes[0]
            cluster_sizes = [c['size'] for c in patterns['clusters']]
            cluster_names = [f'C{c["cluster_id"]+1}' for c in patterns['clusters']]
            bars = ax.bar(cluster_names, cluster_sizes, color=self.class_colors[class_idx], 
                          alpha=0.7, edgecolor='black')
            ax.set_xlabel('Cluster', fontsize=10)
            ax.set_ylabel('Number of Images', fontsize=10)
            ax.set_title(f'{self.class_names[class_idx]} - Cluster Sizes', fontsize=12)
            
            # Add percentage labels
            for bar, cluster in zip(bars, patterns['clusters']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{cluster["percentage"]*100:.1f}%', 
                       ha='center', va='bottom', fontsize=9)
            
            # Plot 2: Top activations heatmap
            ax = row_axes[1]
            
            # Create activation matrix for top tokens across clusters
            n_clusters = len(patterns['clusters'])
            n_top = 5  # Show top 5 activations per cluster
            activation_matrix = np.zeros((n_clusters, n_top))
            activation_labels = []
            
            for i, cluster in enumerate(patterns['clusters']):
                top_acts = sorted(cluster['top_activations'], 
                                key=lambda x: x[3], reverse=True)[:n_top]
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
            ax.set_title(f'{self.class_names[class_idx]} - Top Activations', fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            
            # Plot 3: PCA variance explained
            ax = row_axes[2]
            var_explained = patterns['explained_variance'][:10] * 100
            ax.bar(range(1, len(var_explained)+1), var_explained, 
                  color=self.class_colors[class_idx], alpha=0.7, edgecolor='black')
            ax.set_xlabel('Principal Component', fontsize=10)
            ax.set_ylabel('Variance Explained (%)', fontsize=10)
            ax.set_title(f'{self.class_names[class_idx]} - PCA Variance', fontsize=12)
            ax.set_xticks(range(1, len(var_explained)+1))
            
            # Add cumulative line
            cumsum = np.cumsum(var_explained)
            ax2 = ax.twinx()
            ax2.plot(range(1, len(var_explained)+1), cumsum, 'r-o', 
                    markersize=4, linewidth=2, label='Cumulative')
            ax2.set_ylabel('Cumulative Variance (%)', fontsize=10, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, 100)
        
        plt.suptitle('Detailed Token Pattern Cluster Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'cluster_details.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.descriptions['cluster_details'] = """
**Detailed Cluster Analysis**

This multi-panel visualization provides in-depth analysis of the token activation clusters for each class:

Left panels - Cluster Sizes:
- Shows the distribution of images across clusters
- Percentages indicate the prevalence of each activation pattern

Middle panels - Top Activations:
- Heatmaps show the strongest activated layer-head-token combinations per cluster
- Darker colors indicate stronger activations
- Labels show specific positions (e.g., L11H7T0 = Layer 11, Head 7, Token 0)

Right panels - PCA Variance:
- Bar charts show variance explained by each principal component
- Red line shows cumulative variance
- Indicates the dimensionality of the activation patterns

This detailed view helps understand:
1. The diversity of processing strategies within each class
2. Which specific attention positions are most discriminative
3. The complexity of the learned feature space
"""

    def _visualize_class_activation_patterns(self, direction_similarities: Dict, num_layers: int):
        """Visualize activation patterns across token positions."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Aggregate activation patterns by class
        for class_idx in range(3):
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
                ax.bar(x, mean_pattern[:n_tokens], 
                      yerr=std_pattern[:n_tokens],
                      color=self.class_colors[class_idx], 
                      alpha=0.7, edgecolor='black',
                      error_kw={'elinewidth': 1, 'alpha': 0.5})
                
                # Highlight CLS token
                ax.bar(0, mean_pattern[0], color='red', alpha=0.8, 
                      edgecolor='black', linewidth=2, label='CLS Token')
                
                ax.set_xlabel('Token Position', fontsize=12)
                ax.set_ylabel('Mean Similarity Score', fontsize=12)
                ax.set_title(f'{self.class_names[class_idx]} Token Activation Pattern', 
                           fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, axis='y', alpha=0.3)
                
                # Add annotation for top tokens
                top_tokens = np.argsort(mean_pattern[:n_tokens])[-5:]
                for token_idx in top_tokens:
                    if token_idx != 0:  # Skip CLS token as it's already highlighted
                        ax.annotate(f'T{token_idx}', 
                                  xy=(token_idx, mean_pattern[token_idx]),
                                  xytext=(token_idx, mean_pattern[token_idx] + 0.05),
                                  ha='center', fontsize=8)
        
        plt.suptitle('Average Token Activation Patterns by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'token_activation_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.descriptions['token_activation_patterns'] = """
**Token Activation Patterns**

These bar charts show the average activation strength across token positions for each class. Error bars indicate standard deviation across images.

Key observations:
- CLS token (position 0, highlighted in red) typically shows strong activation
- Different classes exhibit distinct spatial attention patterns
- Some tokens consistently show high activation across images (annotated)
- Variance (error bars) indicates consistency of the pattern

This visualization reveals:
1. How different classes attend to different spatial regions
2. The importance of the CLS token for classification
3. Whether certain image regions are consistently important for diagnosis
"""

    def _visualize_head_specialization(self, head_importance: Dict, num_layers: int, start_layer: int):
        """Visualize head specialization across classes."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate specialization scores
        importance_data = head_importance['class_head_importance']
        
        # Reshape data for analysis
        n_heads_total = num_layers * 12
        head_scores = np.zeros((n_heads_total, 3))
        
        for class_idx in range(3):
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
        specialization_matrix = 1 - (head_entropy / np.log(3))  # Normalize to [0,1]
        specialization_matrix = specialization_matrix.reshape(num_layers, 12)
        
        im1 = ax1.imshow(specialization_matrix, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
        ax1.set_xlabel('Head Index', fontsize=12)
        ax1.set_ylabel('Layer', fontsize=12)
        ax1.set_title('Head Specialization Score\n(Blue = Specialized, Red = General)', 
                     fontsize=14, fontweight='bold')
        ax1.set_yticks(range(num_layers))
        ax1.set_yticklabels([f'L{start_layer + i}' for i in range(num_layers)])
        ax1.set_xticks(range(12))
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Specialization Score', fontsize=10)
        
        # Plot 2: Class assignment
        class_assignment = dominant_class.reshape(num_layers, 12)
        
        # Create custom colormap for classes
        colors_list = [self.class_colors[i] for i in range(3)]
        from matplotlib.colors import ListedColormap
        cmap_classes = ListedColormap(colors_list)
        
        im2 = ax2.imshow(class_assignment, cmap=cmap_classes, aspect='auto', vmin=0, vmax=2)
        ax2.set_xlabel('Head Index', fontsize=12)
        ax2.set_ylabel('Layer', fontsize=12)
        ax2.set_title('Dominant Class per Head', fontsize=14, fontweight='bold')
        ax2.set_yticks(range(num_layers))
        ax2.set_yticklabels([f'L{start_layer + i}' for i in range(num_layers)])
        ax2.set_xticks(range(12))
        
        # Create discrete colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
        cbar2.set_ticklabels([self.class_names[i] for i in range(3)])
        
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
        
        self.descriptions['head_specialization'] = """
**Head Specialization Analysis**

This visualization analyzes how specialized each attention head is for specific classes:

Left panel - Specialization Score:
- Blue indicates heads specialized for one class
- Red indicates heads that contribute equally to all classes
- Stars mark highly specialized heads (>70% specialization)

Right panel - Dominant Class:
- Shows which class each head contributes to most strongly
- Color coding matches the class colors (Red=Covid, Teal=Non-Covid, Blue=Normal)
- Stars mark the same highly specialized heads

Key insights:
- Later layers tend to show more specialization
- Some heads are class-agnostic (general feature extractors)
- Highly specialized heads are prime candidates for interpretation
- Class distribution reveals the model's internal organization
"""

    def _visualize_layer_progression(self, head_importance: Dict, num_layers: int, start_layer: int):
        """Visualize how importance progresses through layers."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        importance_data = head_importance['class_head_importance']
        
        # Calculate layer-wise statistics
        layer_importance = {class_idx: [] for class_idx in range(3)}
        layer_diversity = []
        
        for layer in range(num_layers):
            # Average importance per layer for each class
            for class_idx in range(3):
                avg_importance = np.mean(importance_data[class_idx][layer])
                layer_importance[class_idx].append(avg_importance)
            
            # Calculate diversity (std across heads)
            all_heads = []
            for class_idx in range(3):
                all_heads.extend(importance_data[class_idx][layer])
            layer_diversity.append(np.std(all_heads))
        
        # Plot 1: Layer-wise importance progression
        x = np.arange(num_layers)
        width = 0.25
        
        for class_idx in range(3):
            offset = (class_idx - 1) * width
            bars = ax1.bar(x + offset, layer_importance[class_idx], width,
                          label=self.class_names[class_idx],
                          color=self.class_colors[class_idx],
                          alpha=0.8, edgecolor='black')
        
        ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Head Importance', fontsize=12, fontweight='bold')
        ax1.set_title('Layer-wise Importance Progression', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'L{start_layer + i}' for i in range(num_layers)])
        ax1.legend()
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Plot 2: Cumulative importance and diversity
        ax2_twin = ax2.twinx()
        
        # Cumulative importance
        cumulative_importance = {}
        for class_idx in range(3):
            cumulative = np.cumsum(layer_importance[class_idx])
            ax2.plot(x, cumulative, 'o-', color=self.class_colors[class_idx],
                    linewidth=2, markersize=8, label=self.class_names[class_idx])
        
        # Diversity
        ax2_twin.plot(x, layer_diversity, 's--', color='purple', 
                     linewidth=2, markersize=8, label='Head Diversity')
        
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
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Layer-wise Analysis of Head Contributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'layer_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.descriptions['layer_progression'] = """
**Layer-wise Progression Analysis**

This analysis shows how attention head importance evolves through the network layers:

Left panel - Average Importance by Layer:
- Bar heights show the mean importance of all heads in each layer
- Different colors represent different classes
- Higher values indicate layers with more discriminative heads

Right panel - Cumulative Importance and Diversity:
- Solid lines show cumulative importance (increasing = more heads contribute)
- Dashed purple line shows head diversity (variation in importance scores)
- Higher diversity indicates more specialized heads within a layer

Key findings:
- Later layers generally show higher importance for classification
- Diversity patterns reveal where specialization occurs
- Class-specific patterns emerge more strongly in final layers
- The progression indicates hierarchical feature learning
"""

    def _generate_summary_tables(self, head_importance: Dict, token_patterns: Dict):
        """Generate summary tables for key findings."""
        
        # Create figure for tables
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Statistical Summary Tables', fontsize=16, fontweight='bold')
        
        # Table 1: Top heads per class
        ax1 = plt.subplot(2, 2, 1)
        ax1.axis('tight')
        ax1.axis('off')
        
        table_data = []
        for class_idx in range(3):
            top_heads = head_importance['class_ranked_heads'][class_idx][:5]
            for rank, (layer, head, score) in enumerate(top_heads, 1):
                table_data.append([
                    self.class_names[class_idx],
                    rank,
                    f'L{layer + 7}H{head}',
                    f'{score:.3f}',
                    f'{score*100:.1f}%'
                ])
        
        table1 = ax1.table(cellText=table_data,
                          colLabels=['Class', 'Rank', 'Head', 'Score', 'Importance %'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
        table1.auto_set_font_size(False)
        table1.set_fontsize(9)
        table1.scale(1.2, 1.5)
        ax1.set_title('Top 5 Important Heads per Class', fontsize=12, fontweight='bold', pad=20)
        
        # Table 2: Class statistics
        ax2 = plt.subplot(2, 2, 2)
        ax2.axis('tight')
        ax2.axis('off')
        
        stats_data = []
        for class_idx in range(3):
            n_images = head_importance['class_image_counts'][class_idx]
            importance_scores = head_importance['class_head_importance'][class_idx].flatten()
            stats_data.append([
                self.class_names[class_idx],
                n_images,
                f'{np.mean(importance_scores):.3f}',
                f'{np.std(importance_scores):.3f}',
                f'{np.sum(importance_scores > 0.5)}'
            ])
        
        table2 = ax2.table(cellText=stats_data,
                          colLabels=['Class', 'Images', 'Mean Import.', 'Std Import.', 'High Import. Heads'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
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
                    self.class_names[class_idx],
                    f"Cluster {cluster['cluster_id']+1}",
                    cluster['size'],
                    f"{cluster['percentage']*100:.1f}%",
                    len(cluster['top_activations'])
                ])
        
        table3 = ax3.table(cellText=cluster_data,
                          colLabels=['Class', 'Cluster', 'Size', 'Percentage', 'Key Activations'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1.2, 1.5)
        ax3.set_title('Token Pattern Cluster Summary', fontsize=12, fontweight='bold', pad=20)
        
        # Table 4: Similarity statistics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('tight')
        ax4.axis('off')
        
        sim_stats = head_importance['similarity_stats']
        stats_rows = [
            ['Mean', f"{sim_stats['mean']:.4f}"],
            ['Median', f"{sim_stats['median']:.4f}"],
            ['Std Dev', f"{sim_stats['std']:.4f}"],
            ['Min', f"{sim_stats['min']:.4f}"],
            ['Max', f"{sim_stats['max']:.4f}"],
            ['25th Percentile', f"{sim_stats['percentiles']['25']:.4f}"],
            ['75th Percentile', f"{sim_stats['percentiles']['75']:.4f}"],
            ['90th Percentile', f"{sim_stats['percentiles']['90']:.4f}"]
        ]
        
        table4 = ax4.table(cellText=stats_rows,
                          colLabels=['Metric', 'Value'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
        table4.auto_set_font_size(False)
        table4.set_fontsize(9)
        table4.scale(1.2, 1.5)
        ax4.set_title('Similarity Score Statistics', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'summary_tables.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.descriptions['summary_tables'] = """
**Statistical Summary Tables**

These tables provide quantitative summaries of the analysis:

Table 1 - Top Important Heads:
- Lists the 5 most important heads for each class
- Shows both raw scores and percentage importance
- Reveals which specific heads drive each classification

Table 2 - Class Statistics:
- Number of images analyzed per class
- Mean and standard deviation of importance scores
- Count of highly important heads (>50% importance)

Table 3 - Token Pattern Clusters:
- Summary of clustering results for each class
- Shows cluster sizes and percentages
- Indicates the number of key activation positions

Table 4 - Similarity Statistics:
- Overall distribution metrics for cosine similarities
- Helps understand the range and central tendency
- Guides threshold selection for importance determination

These tables provide quick reference for key quantitative findings.
"""

    def _save_descriptions(self):
        """Save all descriptions to a markdown file."""
        
        with open(self.viz_dir / 'visualization_descriptions.md', 'w') as f:
            f.write("# Head Analysis Visualization Descriptions\n\n")
            f.write("This document provides detailed descriptions of all generated visualizations.\n\n")
            
            for viz_name, description in self.descriptions.items():
                f.write(f"## {viz_name.replace('_', ' ').title()}\n\n")
                f.write(description)
                f.write("\n\n---\n\n")
        
        print(f"Visualization descriptions saved to {self.viz_dir / 'visualization_descriptions.md'}")


def create_comprehensive_visualizations(data_dir: Path):
    """
    Main function to load data and create all visualizations.
    
    Args:
        data_dir: Directory containing the analysis output files
    """
    
    # Load the analysis results
    print("Loading analysis data...")
    
    # Load head direction similarities
    similarities_path = data_dir / "head_direction_similarities.npy"
    direction_similarities = np.load(similarities_path, allow_pickle=True).item()
    
    # Load head importance analysis
    importance_path = data_dir / "head_importance_analysis.npy"
    head_importance = np.load(importance_path, allow_pickle=True).item()
    
    # Load token patterns for each class
    token_patterns = {}
    for class_idx in range(3):
        pattern_path = data_dir / f"token_patterns_class_{class_idx}.npy"
        if pattern_path.exists():
            token_patterns[class_idx] = np.load(pattern_path, allow_pickle=True).item()
    
    # Create visualizer and generate all visualizations
    visualizer = HeadAnalysisVisualizer(data_dir)
    visualizer.generate_all_visualizations(
        direction_similarities,
        head_importance,
        token_patterns,
        num_layers=5,
        start_layer=7
    )
    
    print("\nVisualization generation complete!")
    print(f"All visualizations saved to: {data_dir / 'visualizations'}")
    print(f"Descriptions saved to: {data_dir / 'visualizations' / 'visualization_descriptions.md'}")


# Example usage
if __name__ == "__main__":
    # Replace with your actual data directory
    data_directory = Path("./results/train/head_analysis")
    create_comprehensive_visualizations(data_directory)