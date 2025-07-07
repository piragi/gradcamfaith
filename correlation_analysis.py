# feature_attribution_correlation.py
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def compute_feature_attribution_correlation(
    sae_codes: torch.Tensor,  # Shape: (1, T, F)
    attribution_map: np.ndarray,  # Shape: (H, W), e.g., (224, 224)
    patch_size: int = 16,
    min_activation_threshold: float = 0.1,
    device: torch.device = torch.device("cpu")
) -> Dict[str, np.ndarray]:
    """
    Compute correlation between SAE feature activations and attribution values
    for each feature in a single image.
    
    Returns:
        Dictionary containing:
        - 'correlations': (F,) correlation coefficient for each feature
        - 'p_values': (F,) p-values for correlation significance
        - 'active_features': (F,) boolean mask of features that were active
        - 'spatial_concentration': (F,) measure of how localized each feature is
    """
    # Remove batch dimension and separate CLS from spatial tokens
    codes = sae_codes[0]  # Shape: (T, F)
    codes_spatial = codes[1:]  # Shape: (T-1, F), excluding CLS token

    n_patches = codes_spatial.shape[0]
    n_features = codes_spatial.shape[1]

    # Convert attribution map to patch space
    # Assuming square image and patches
    n_patches_per_side = int(np.sqrt(n_patches))

    # Resize attribution map to patch resolution
    attribution_tensor = torch.tensor(attribution_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    attribution_patches = F.adaptive_avg_pool2d(attribution_tensor, (n_patches_per_side, n_patches_per_side))
    attribution_patches = attribution_patches.squeeze().flatten().numpy()  # Shape: (T-1,)

    # Initialize results
    correlations = np.zeros(n_features)
    p_values = np.ones(n_features)  # Default to 1 (non-significant)
    active_features = np.zeros(n_features, dtype=bool)
    spatial_concentration = np.zeros(n_features)

    # Compute correlation for each feature
    for f in range(n_features):
        feature_activations = codes_spatial[:, f].cpu().numpy()  # Shape: (T-1,)

        # Check if feature is active (at least some patches above threshold)
        active_patches = feature_activations > min_activation_threshold
        n_active = active_patches.sum()

        if n_active < 2:  # Need at least 2 points for correlation
            continue

        active_features[f] = True

        # Compute Pearson correlation
        # We correlate activation strength with attribution values
        if n_active == n_patches:  # Feature active everywhere
            # Use activation strength directly
            corr, p_val = stats.pearsonr(feature_activations, attribution_patches)
        else:
            # Create binary mask and correlate with attribution
            # This captures whether high attribution areas coincide with feature activation
            activation_mask = active_patches.astype(float)
            corr, p_val = stats.pearsonr(activation_mask, attribution_patches)

        correlations[f] = corr
        p_values[f] = p_val

        # Compute spatial concentration (how localized is the feature?)
        if n_active > 0:
            # Reshape to 2D grid
            activation_grid = feature_activations.reshape(n_patches_per_side, n_patches_per_side)
            active_grid = active_patches.reshape(n_patches_per_side, n_patches_per_side)

            # Compute center of mass of activations
            y_indices, x_indices = np.ogrid[:n_patches_per_side, :n_patches_per_side]
            if activation_grid.sum() > 0:
                center_y = (y_indices * activation_grid).sum() / activation_grid.sum()
                center_x = (x_indices * activation_grid).sum() / activation_grid.sum()

                # Compute average distance from center (normalized by grid size)
                distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
                avg_distance = (distances * activation_grid).sum() / activation_grid.sum()
                max_distance = np.sqrt(2) * n_patches_per_side / 2

                # Concentration: 1 = highly concentrated, 0 = spread out
                spatial_concentration[f] = 1 - (avg_distance / max_distance)

    return {
        'correlations': correlations,
        'p_values': p_values,
        'active_features': active_features,
        'spatial_concentration': spatial_concentration
    }


def aggregate_feature_correlations(
    correlation_results: List[Dict[str, np.ndarray]],
    sf_af_dict: Dict[str, torch.Tensor],
    predicted_classes: List[int],
    output_dir: Path,
    significance_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Aggregate correlation results across multiple images and combine with S_f/A_f values.
    
    Returns:
        DataFrame with columns:
        - feature_id
        - mean_correlation
        - std_correlation
        - n_active_images
        - mean_spatial_concentration
        - mean_s_f (averaged across relevant classes)
        - mean_a_f
        - is_spatially_aligned (boolean based on correlation significance)
    """
    n_features = correlation_results[0]['correlations'].shape[0] if correlation_results else 0

    # Initialize aggregation arrays
    correlation_sum = np.zeros(n_features)
    correlation_sq_sum = np.zeros(n_features)
    active_count = np.zeros(n_features)
    significant_positive_count = np.zeros(n_features)
    significant_negative_count = np.zeros(n_features)
    concentration_sum = np.zeros(n_features)

    # Aggregate across images
    for i, result in enumerate(correlation_results):
        active_mask = result['active_features']
        correlations = result['correlations']
        p_values = result['p_values']

        # Update sums for active features
        correlation_sum[active_mask] += correlations[active_mask]
        correlation_sq_sum[active_mask] += correlations[active_mask]**2
        active_count[active_mask] += 1
        concentration_sum[active_mask] += result['spatial_concentration'][active_mask]

        # Count significant correlations
        sig_mask = p_values < significance_threshold
        significant_positive_count[(correlations > 0) & sig_mask & active_mask] += 1
        significant_negative_count[(correlations < 0) & sig_mask & active_mask] += 1

    # Compute statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_correlation = correlation_sum / active_count
        variance = (correlation_sq_sum / active_count) - mean_correlation**2
        std_correlation = np.sqrt(np.maximum(variance, 0))
        mean_concentration = concentration_sum / active_count

    # Replace NaN with 0 for features that were never active
    mean_correlation = np.nan_to_num(mean_correlation)
    std_correlation = np.nan_to_num(std_correlation)
    mean_concentration = np.nan_to_num(mean_concentration)

    # Aggregate S_f and A_f across predicted classes
    S_f = sf_af_dict['S_f']  # Shape: (F, C)
    A_f = sf_af_dict['A_f']  # Shape: (F, C)

    # Weight S_f and A_f by class frequency
    class_counts = np.bincount(predicted_classes)
    class_weights = class_counts / class_counts.sum()

    mean_s_f = np.zeros(n_features)
    mean_a_f = np.zeros(n_features)

    for c, weight in enumerate(class_weights):
        if weight > 0:
            mean_s_f += weight * S_f[:, c].cpu().numpy()
            mean_a_f += weight * A_f[:, c].cpu().numpy()

    # Determine spatially aligned features
    # A feature is spatially aligned if it has significant positive correlation in many images
    alignment_ratio = significant_positive_count / (active_count + 1e-9)
    is_spatially_aligned = (alignment_ratio > 0.1) & (mean_correlation > 0.1)

    # Create DataFrame
    results_df = pd.DataFrame({
        'feature_id': np.arange(n_features),
        'mean_correlation': mean_correlation,
        'std_correlation': std_correlation,
        'n_active_images': active_count.astype(int),
        'n_significant_positive': significant_positive_count.astype(int),
        'n_significant_negative': significant_negative_count.astype(int),
        'alignment_ratio': alignment_ratio,
        'mean_spatial_concentration': mean_concentration,
        'mean_s_f': mean_s_f,
        'abs_mean_s_f': np.abs(mean_s_f),
        'mean_a_f': mean_a_f,
        'is_spatially_aligned': is_spatially_aligned
    })

    # Save detailed results
    results_df.to_csv(output_dir / "feature_spatial_correlation_analysis.csv", index=False)

    return results_df


def identify_attribution_compatible_features(
    correlation_df: pd.DataFrame,
    s_f_percentile_threshold: float = 70.0,  # Top 30% by |S_f|
    correlation_threshold: float = 0.2,  # Minimum correlation
    min_active_images: int = 10,  # Minimum images where feature is active
    output_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    Identify different categories of features based on their S_f and spatial alignment.
    
    Returns dictionary with DataFrames for each category:
    - 'attribution_heroes': High |S_f| + spatially aligned
    - 'hidden_circuits': High |S_f| + NOT spatially aligned  
    - 'spurious_attention': Low |S_f| + spatially aligned
    - 'irrelevant': Low |S_f| + NOT spatially aligned
    """
    # Filter for features that appear in enough images
    active_df = correlation_df[correlation_df['n_active_images'] >= min_active_images].copy()

    # Compute thresholds
    s_f_threshold = np.percentile(active_df['abs_mean_s_f'], s_f_percentile_threshold)

    # Categorize features
    high_impact = active_df['abs_mean_s_f'] > s_f_threshold
    spatially_aligned = (active_df['mean_correlation'] > correlation_threshold) & active_df['is_spatially_aligned']

    categories = {
        'attribution_heroes': active_df[high_impact & spatially_aligned],
        'hidden_circuits': active_df[high_impact & ~spatially_aligned],
        'spurious_attention': active_df[~high_impact & spatially_aligned],
        'irrelevant': active_df[~high_impact & ~spatially_aligned]
    }

    # Add category column to each DataFrame
    for category_name, df in categories.items():
        df['category'] = category_name

    # Print summary
    print("\n=== Feature Category Summary ===")
    for category, df in categories.items():
        print(f"{category}: {len(df)} features ({len(df)/len(active_df)*100:.1f}%)")

    # Save category assignments
    if output_dir:
        all_categorized = pd.concat(categories.values(), ignore_index=True)
        all_categorized.to_csv(output_dir / "feature_categories.csv", index=False)

        # Save top attribution heroes for easy access
        top_heroes = categories['attribution_heroes'].nlargest(100, 'abs_mean_s_f')
        top_heroes.to_csv(output_dir / "top_attribution_heroes.csv", index=False)

    return categories


def visualize_correlation_analysis(correlation_df: pd.DataFrame, categories: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate visualizations for the correlation analysis."""

    # 1. Scatter plot: S_f vs Correlation
    plt.figure(figsize=(12, 8))

    # Plot each category with different colors
    colors = {
        'attribution_heroes': '#2ecc71',  # Green
        'hidden_circuits': '#e74c3c',  # Red
        'spurious_attention': '#f39c12',  # Orange
        'irrelevant': '#95a5a6'  # Gray
    }

    for category, df in categories.items():
        if len(df) > 0:
            plt.scatter(
                df['mean_correlation'],
                df['mean_s_f'],
                c=colors[category],
                alpha=0.6,
                s=30,
                label=f"{category} (n={len(df)})"
            )

    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)

    plt.xlabel('Mean Spatial Correlation with Attribution')
    plt.ylabel('Mean S_f (Steerability)')
    plt.title('Feature Categories: Steerability vs Spatial Alignment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_categories_scatter.png", dpi=150)
    plt.close()

    # 2. Distribution of correlations by category
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (category, df) in enumerate(categories.items()):
        if len(df) > 0 and i < 4:
            ax = axes[i]
            ax.hist(df['mean_correlation'], bins=30, color=colors[category], alpha=0.7, edgecolor='black')
            ax.axvline(
                df['mean_correlation'].mean(),
                color='red',
                linestyle='--',
                label=f"Mean: {df['mean_correlation'].mean():.3f}"
            )
            ax.set_title(f"{category} (n={len(df)})")
            ax.set_xlabel('Mean Correlation')
            ax.set_ylabel('Count')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_distributions_by_category.png", dpi=150)
    plt.close()

    # 3. Heatmap of top attribution heroes
    if len(categories['attribution_heroes']) > 0:
        top_heroes = categories['attribution_heroes'].nlargest(20, 'abs_mean_s_f')

        plt.figure(figsize=(10, 8))
        hero_data = top_heroes[[
            'mean_correlation', 'mean_s_f', 'mean_a_f', 'mean_spatial_concentration', 'alignment_ratio'
        ]]

        # Normalize for better visualization
        hero_data_norm = (hero_data - hero_data.mean()) / hero_data.std()

        sns.heatmap(
            hero_data_norm.T,
            xticklabels=[f"F{int(fid)}" for fid in top_heroes['feature_id']],
            yticklabels=['Correlation', 'S_f', 'A_f', 'Concentration', 'Alignment'],
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Normalized Value'}
        )
        plt.title('Top 20 Attribution Heroes - Feature Properties')
        plt.tight_layout()
        plt.savefig(output_dir / "top_attribution_heroes_heatmap.png", dpi=150)
        plt.close()


# Integration function for your pipeline
def run_correlation_analysis_in_pipeline(
    classification_results: List,  # Your ClassificationResult objects
    sae: Any,  # Your SAE model
    sf_af_dict: Dict[str, torch.Tensor],
    config: Any,  # Your PipelineConfig
    layer_idx: int = 6
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run the complete correlation analysis as part of your pipeline.
    
    This should be called after classification and attribution generation.
    """
    correlation_results = []
    predicted_classes = []

    print("\n=== Running Feature-Attribution Correlation Analysis ===")

    # Process each image
    for result in tqdm(classification_results, desc="Computing correlations"):
        try:
            # Load attribution map
            attribution_path = result.attribution_paths.attribution_path
            attribution_map = np.load(attribution_path)

            # Load the original image to get SAE codes
            image_path = result.image_path
            _, input_tensor = preprocessing.preprocess_image(str(image_path), img_size=config.classify.target_size[0])
            input_tensor = input_tensor.to(next(sae.parameters()).device)

            # Get SAE codes for this image
            with torch.no_grad():
                # You'll need to get the residual activations at the target layer
                # This depends on your model architecture
                # For now, assuming you can get them:
                resid = get_residual_at_layer(vit_model, input_tensor, layer_idx)
                _, codes = sae.encode(resid)

            # Compute correlations
            corr_result = compute_feature_attribution_correlation(
                codes,
                attribution_map,
                patch_size=config.classify.target_size[0] // 14  # Assuming 14x14 patches
            )

            correlation_results.append(corr_result)
            predicted_classes.append(result.prediction.predicted_class_idx)

        except Exception as e:
            print(f"Error processing {result.image_path.name}: {e}")
            continue

    # Aggregate results
    correlation_df = aggregate_feature_correlations(
        correlation_results, sf_af_dict, predicted_classes, config.file.output_dir
    )

    # Identify feature categories
    categories = identify_attribution_compatible_features(correlation_df, output_dir=config.file.output_dir)

    # Generate visualizations
    visualize_correlation_analysis(correlation_df, categories, config.file.output_dir)

    print(f"\nAnalysis complete. Results saved to {config.file.output_dir}")

    return correlation_df, categories


# Helper function - you'll need to implement this based on your model
def get_residual_at_layer(model, input_tensor, layer_idx):
    """
    Extract residual stream activation at specified layer.
    This needs to be implemented based on your specific model architecture.
    """
    # Placeholder - implement based on your HookedViT
    raise NotImplementedError("Implement residual extraction for your model")
