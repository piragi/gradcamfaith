# analysis.py
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.special import softmax
from sklearn.metrics import mutual_info_score

import perturbation
import visualization


def compare_attributions(original_results_df: pd.DataFrame,
                         perturbed_results_df: pd.DataFrame,
                         output_dir: str = "./results",
                         generate_visualizations: bool = True) -> pd.DataFrame:
    """
    Compare attributions between original and perturbed images.
    
    Args:
        original_results_df: DataFrame with original classification results
        perturbed_results_df: DataFrame with perturbed classification results
        output_dir: Directory to save comparison results
        generate_visualizations: Whether to generate and save visualizations
        compute_statistics: Whether to compute detailed attribution statistics
        
    Returns:
        DataFrame with attribution comparison results
    """
    output_path = Path(output_dir)
    comparison_dir = output_path / "comparisons"
    patch_mask_dir = output_path / "patch_masks"

    # Only create visualization directory if needed
    if generate_visualizations:
        comparison_dir.mkdir(exist_ok=True, parents=True)

    comparison_results = []

    for _, perturbed_row in perturbed_results_df.iterrows():
        perturbed_path = Path(perturbed_row["image_path"])
        perturbed_filename = perturbed_path.stem

        # Extract the original filename part (before "_patch")
        if "_patch" not in perturbed_filename:
            print(f"Skipping {perturbed_filename}: not a patch-perturbed file")
            continue

        original_part = perturbed_filename.split("_patch")[0]
        original_filenames = original_results_df["image_path"].apply(
            lambda path: Path(path).stem)

        # Find exact matches (preserving whitespace)
        matching_indices = original_filenames[original_filenames ==
                                              original_part].index

        if len(matching_indices) == 0:
            continue

        original_row = original_results_df.loc[matching_indices[0]]

        # Get mask path
        mask_path = Path(f"{patch_mask_dir}/{perturbed_filename}_mask.npy")

        if not mask_path.exists():
            print(f"Mask file not found: {mask_path}")
            continue

        try:
            # Load attributions
            original_attribution = np.load(original_row["attribution_path"])
            perturbed_attribution = np.load(perturbed_row["attribution_path"])
            np_mask = np.load(mask_path)

            # Initialize comparison path and diff_stats
            comparison_path = None
            diff_stats = {}

            # Generate comparison visualization if requested
            if generate_visualizations:
                comparison_path = comparison_dir / f"{perturbed_filename}_comparison.png"
                diff_stats = visualization.visualize_attribution_diff(
                    original_attribution,
                    perturbed_attribution,
                    np_mask,
                    base_name=perturbed_filename,
                    save_dir=str(comparison_dir))
            # Just compute statistics without visualization if needed
            diff_stats = visualization.calculate_attribution_statistics(
                original_attribution, perturbed_attribution,
                perturbed_attribution - original_attribution, np_mask)

            # Extract patch information
            patch_info = extract_patch_info_from_filename(perturbed_filename)
            patch_id, x, y = patch_info["patch_id"], patch_info[
                "x"], patch_info["y"]

            # Calculate mean attribution in the patch if coordinates are found
            mean_attribution = calculate_patch_mean_attribution(
                original_attribution, x, y, patch_size=16)

            # Calculate SSIM between the actual ViT inputs
            ssim_score = calculate_ssim_if_available(original_row,
                                                     perturbed_row)

            # Prepare comparison result
            result = {
                "original_image":
                original_row["image_path"],
                "perturbed_image":
                str(perturbed_path),
                "patch_id":
                patch_id,
                "x":
                x,
                "y":
                y,
                "mean_attribution":
                mean_attribution,
                "original_class":
                original_row["predicted_class"],
                "perturbed_class":
                perturbed_row["predicted_class"],
                "class_changed":
                original_row["predicted_class_idx"]
                != perturbed_row["predicted_class_idx"],
                "original_confidence":
                original_row["confidence"],
                "perturbed_confidence":
                perturbed_row["confidence"],
                "confidence_delta":
                perturbed_row["confidence"] - original_row["confidence"],
                "confidence_delta_abs":
                abs(perturbed_row["confidence"] - original_row["confidence"]),
                "vit_input_ssim":
                ssim_score
            }

            # Add comparison path if visualization was generated
            if comparison_path:
                result["comparison_path"] = str(comparison_path)

            # Add key metrics from diff_stats if available
            for category in [
                    "original_stats", "perturbed_stats", "difference_stats"
            ]:
                if category in diff_stats:
                    for key, value in diff_stats[category].items():
                        result[f"{category}_{key}"] = value

            comparison_results.append(result)

        except Exception as e:
            print(f"Error processing {perturbed_filename}: {e}")
            continue

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_results)

    if not comparison_df.empty:
        # Add rankings for impact and attribution
        comparison_df = add_rankings_to_comparison_df(comparison_df)
        comparison_df.to_csv(output_path / "patch_attribution_comparisons.csv",
                             index=False)
    else:
        print("Warning: No comparison results were generated.")

    return comparison_df


def extract_patch_info_from_filename(filename: str) -> Dict[str, int]:
    """
    Extract patch coordinates and ID from a perturbed image filename.
    
    Args:
        filename: Perturbed image filename
        
    Returns:
        Dictionary with patch_id, x, and y coordinates
    """
    patch_id, x, y = -1, -1, -1

    # Try to extract patch info from filename
    filename_parts = filename.split('_')
    for part in filename_parts:
        if part.startswith('patch'):
            try:
                patch_id = int(part[5:])
            except ValueError:
                pass
        elif part.startswith('x'):
            try:
                x = int(part[1:])
            except ValueError:
                pass
        elif part.startswith('y'):
            try:
                y = int(part[1:])
            except ValueError:
                pass

    return {"patch_id": patch_id, "x": x, "y": y}


def calculate_patch_mean_attribution(attribution: np.ndarray,
                                     x: int,
                                     y: int,
                                     patch_size: int = 16) -> Optional[float]:
    """
    Calculate the mean attribution value within a patch.
    
    Args:
        attribution: Attribution map
        x: X-coordinate of patch
        y: Y-coordinate of patch
        patch_size: Size of the patch
        
    Returns:
        Mean attribution in the patch or None if coordinates are invalid
    """
    if x < 0 or y < 0:
        return None

    patch_end_x = min(x + patch_size, attribution.shape[1])
    patch_end_y = min(y + patch_size, attribution.shape[0])

    if x < attribution.shape[1] and y < attribution.shape[0]:
        patch_attribution = attribution[y:patch_end_y, x:patch_end_x]
        return np.mean(patch_attribution)

    return None


def calculate_ssim_if_available(original_row: pd.Series,
                                perturbed_row: pd.Series) -> Optional[float]:
    """
    Calculate SSIM between original and perturbed images if paths are available.
    
    Args:
        original_row: Row with original image data
        perturbed_row: Row with perturbed image data
        
    Returns:
        SSIM score or None if paths are not available
    """
    if "image_path" in original_row and "image_path" in perturbed_row:
        try:
            original_vit_img = Image.open(
                original_row["image_path"]).convert('RGB')
            perturbed_vit_img = Image.open(
                perturbed_row["image_path"]).convert('RGB')
            return perturbation.patch_similarity(original_vit_img,
                                                 perturbed_vit_img)
        except Exception as e:
            print(f"Error calculating SSIM: {e}")

    return None


def add_rankings_to_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rankings for impact and attribution to the comparison DataFrame.
    
    Args:
        df: DataFrame with comparison results
        
    Returns:
        DataFrame with added ranking columns
    """
    if "confidence_delta_abs" in df.columns:
        df["impact_rank"] = df["confidence_delta_abs"].rank(ascending=False)

    if "mean_attribution" in df.columns:
        df["attribution_rank"] = df["mean_attribution"].rank(ascending=False)

    if "impact_rank" in df.columns and "attribution_rank" in df.columns:
        df["rank_difference"] = df["attribution_rank"] - df["impact_rank"]

    return df


# SaCo Calculation Functions
def calculate_saco_with_details(
        data_path: str = "./results/patch_attribution_comparisons.csv",
        method: str = "mean"
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:
    """
    Calculate SaCo (Saliency Correlation) scores from comparison data.
    
    Args:
        data_path: Path to patch attribution comparisons CSV
        method: Perturbation method filter
        
    Returns:
        Tuple of (saco_scores, pair_data) where:
            - saco_scores: Dict mapping image names to SaCo scores
            - pair_data: Dict mapping image names to pair-wise comparison DataFrames
    """
    data_df = pd.read_csv(data_path)

    if method:
        data_df = data_df[data_df['perturbed_image'].str.contains(
            f"_{method}.png")]

    results = {}
    pair_data = {}

    for image_name, image_data in data_df.groupby('original_image'):
        image_data = image_data.sort_values(
            'mean_attribution', ascending=False).reset_index(drop=True)
        attributions = image_data['mean_attribution'].values
        confidence_impacts = image_data['confidence_delta_abs'].values
        patch_ids = image_data['patch_id'].values

        saco_score, image_pair_data = calculate_image_saco_with_details(
            attributions, confidence_impacts, patch_ids)

        results[image_name] = saco_score
        pair_data[image_name] = image_pair_data

    print(f"Average SaCo score: {np.mean(list(results.values())):.4f}")
    return results, pair_data


def calculate_image_saco_with_details(
        attributions: np.ndarray, confidence_impacts: np.ndarray,
        patch_ids: np.ndarray) -> Tuple[float, pd.DataFrame]:
    """
    Calculate SaCo for a single image and return detailed pair-wise comparison data.
    
    Args:
        attributions: Array of attribution values per patch
        confidence_impacts: Array of confidence impact values per patch
        patch_ids: Array of patch IDs
        
    Returns:
        Tuple of (saco_score, pair_data_df) where:
            - saco_score: Float SaCo score for the image
            - pair_data_df: DataFrame with pair-wise comparison data
    """
    F = 0
    total_weight = 0
    pairs_data = []

    for i in range(len(attributions) - 1):
        for j in range(i + 1, len(attributions)):
            attr_diff = attributions[i] - attributions[j]
            impact_i, impact_j = confidence_impacts[i], confidence_impacts[j]
            patch_i, patch_j = patch_ids[i], patch_ids[j]

            # Calculate weight for SaCo
            is_faithful = impact_i >= impact_j
            weight = attr_diff if is_faithful else -attr_diff
            F += weight
            total_weight += abs(weight)

            # Store pair data
            pair_info = {
                'patch_i': patch_i,
                'patch_j': patch_j,
                'is_faithful': is_faithful,
                'weight': weight
            }
            pairs_data.append(pair_info)

    # Avoid division by zero
    if total_weight > 0:
        F /= total_weight
    else:
        F = 0.0

    return F, pd.DataFrame(pairs_data)


def analyze_patch_metrics(pair_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Analyze patch-wise comparison data and return a flat DataFrame with patch-specific metrics.
    
    Args:
        pair_data: Dictionary mapping image names to pair-wise comparison DataFrames
        
    Returns:
        DataFrame with patch-specific metrics
    """
    rows = []

    for image_name, image_pairs in pair_data.items():
        unique_patches = get_unique_patches_from_pairs(image_pairs)

        for patch_id in unique_patches:
            # Get pairs involving this patch
            pairs_with_i = image_pairs[image_pairs['patch_i'] == patch_id]
            pairs_with_j = image_pairs[image_pairs['patch_j'] == patch_id]

            # Calculate metrics for this patch
            patch_metrics = calculate_patch_specific_metrics(
                patch_id, pairs_with_i, pairs_with_j, image_name)

            rows.append(patch_metrics)

    return pd.DataFrame(rows)


def get_unique_patches_from_pairs(pairs_df: pd.DataFrame) -> Set[int]:
    """
    Extract unique patch IDs from a pair-wise comparison DataFrame.
    
    Args:
        pairs_df: DataFrame with pair-wise comparison data
        
    Returns:
        Set of unique patch IDs
    """
    unique_i = set(pairs_df['patch_i'].tolist())
    unique_j = set(pairs_df['patch_j'].tolist())
    return unique_i.union(unique_j)


def calculate_patch_specific_metrics(patch_id: int, pairs_with_i: pd.DataFrame,
                                     pairs_with_j: pd.DataFrame,
                                     image_name: str) -> Dict[str, Any]:
    """
    Calculate metrics specific to a patch based on its pair-wise comparisons.
    
    Args:
        patch_id: ID of the patch
        pairs_with_i: DataFrame with pairs where this patch is the first one
        pairs_with_j: DataFrame with pairs where this patch is the second one
        image_name: Name of the original image
        
    Returns:
        Dictionary with patch-specific metrics
    """
    # For pairs where patch is the first one (i)
    weights_i = pairs_with_i['weight'].tolist()
    faithful_i = pairs_with_i['is_faithful'].sum()

    # For pairs where patch is the second one (j)
    weights_j = pairs_with_j['weight'].tolist()
    faithful_j = pairs_with_j['is_faithful'].sum()

    # Combine data from both directions
    all_weights = weights_i + weights_j
    total_faithful = faithful_i + faithful_j
    total_pairs = len(pairs_with_i) + len(pairs_with_j)

    # Calculate patch-specific SaCo
    patch_saco = 0.0
    if all_weights:
        sum_weights = sum(all_weights)
        sum_abs_weights = sum(abs(w) for w in all_weights)
        if sum_abs_weights > 0:
            patch_saco = sum_weights / sum_abs_weights

    # Create a row for this patch
    return {
        'image_name':
        image_name,
        'patch_id':
        patch_id,
        'faithful_pairs_count':
        total_faithful,
        'unfaithful_pairs_count':
        total_pairs - total_faithful,
        'faithful_pairs_pct':
        (total_faithful / total_pairs * 100) if total_pairs else 0,
        'patch_saco':
        patch_saco
    }


# Correlation Analysis Functions
def analyze_faithfulness_vs_correctness(
    saco_scores: Dict[str, float],
    classification_results: str = "./results/classification_results.csv"
) -> pd.DataFrame:
    """
    Analyze the relationship between attribution faithfulness and prediction correctness.
    
    Args:
        saco_scores: Dictionary mapping image names to SaCo scores
        classification_results: Path to classification results CSV
        
    Returns:
        DataFrame with SaCo scores, correctness, and confidence information
    """
    df = pd.read_csv(classification_results)
    results = []

    for _, row in df.iterrows():
        filename = row['image_path']
        saco_score = saco_scores.get(filename)
        if saco_score is None:
            continue

        # Get true class from filename
        true_class = extract_true_class_from_filename(filename)
        if true_class is None:
            continue

        # Store all relevant information
        results.append({
            'filename': filename,
            'saco_score': saco_score,
            'predicted_class': row['predicted_class'],
            'true_class': true_class,
            'is_correct': row['predicted_class'] == true_class,
            'confidence': row['confidence'],
            'attribution_path': row['attribution_path'],
            'attribution_neg_path': row['attribution_neg_path'],
            'ffn_activity_path': row['ffn_activity_path'],
            'class_embedding_path': row['class_embedding_path']
        })

    return pd.DataFrame(results)


def extract_true_class_from_filename(filename: str) -> Optional[str]:
    """
    Extract the true class from a filename based on path patterns.
    
    Args:
        filename: Image filepath
        
    Returns:
        True class name or None if pattern doesn't match
    """
    if filename.startswith("images/Normal"):
        return "Normal"
    elif filename.startswith("images/covid"):
        return "COVID-19"
    elif filename.startswith("images/non_COVID"):
        return "Non-COVID"
    return None


def analyze_key_attribution_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive analysis of attribution patterns with correlation calculations.
    
    Args:
        df: DataFrame with SaCo scores and attribution paths
        
    Returns:
        DataFrame with added attribution metrics and correlation analysis
    """
    # Process attribution metrics
    df = add_basic_attribution_metrics(df)
    df = add_entropy_metrics(df)
    df = add_concentration_metrics(df)
    df = add_information_theory_metrics(df)
    df = add_gradient_based_metrics(df)
    df = add_sparsity_metrics(df)
    df = add_attribution_consistency_metrics(df)
    df = add_robustness_metrics(df)
    df = add_ffn_activity_metrics(df)
    df = add_class_embedding_metrics(df)

    # Clean data
    df_clean = df.dropna(
        subset=['saco_score', 'neg_magnitude', 'pos_magnitude'])

    # Define key metrics to analyze
    key_metrics = get_key_attribution_metrics()

    # Add class column
    df_clean = df_clean[df_clean['is_correct']]
    if 'filename' in df_clean.columns:
        df_clean['class'] = df_clean['true_class']

    # Calculate correlations
    if len(df_clean) > 0:
        calculate_overall_correlations(df_clean, key_metrics)
        calculate_per_class_correlations(df_clean, key_metrics)

        calculate_class_correlations_by_prediction(df_clean)

    return df_clean


def get_key_attribution_metrics() -> List[str]:
    """
    Get the list of key attribution metrics for correlation analysis.
    
    Returns:
        List of metric names
    """
    return [
        'neg_pos_ratio',
        'neg_entropy',
        'pos_entropy',
        'entropy_ratio',
        'neg_gini',
        'pos_gini',
        'neg_top10_conc',
        'pos_top10_conc',
        'mutual_information',
        'neg_pos_contingency',
        # Gradient metrics
        'gradient_magnitude',
        'gradient_variance',
        'gradient_entropy',
        'gradient_sparsity',
        # Sparsity metrics
        'pos_l0_sparsity',
        'neg_l0_sparsity',
        'pos_effective_sparsity',
        'neg_effective_sparsity',
        'pos_kurtosis',
        'neg_kurtosis',
        # Attribution consistency metrics
        'pos_neg_coherence',
        'attribution_evenness',
        'feature_consensus',
        # Robustness metrics
        'smoothing_stability',
        'noise_stability',
        'peak_persistence',
        # FFN activity metrics
        'ffn_mean_activity',
        'ffn_cls_activity',
        'ffn_last_layer_activity',
        'ffn_max_layer_activity',
        'ffn_early_layers_activity',
        'ffn_middle_layers_activity',
        'ffn_late_layers_activity',
        'ffn_layer_activity_variance',
        'ffn_token_activity_variance'
    ]


def calculate_correlation_significance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate p-value for a correlation between two arrays.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        P-value of the correlation
    """
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) < 3:  # Need at least 3 points for correlation
        return 1.0

    corr, p_value = stats.pearsonr(x[valid_mask], y[valid_mask])
    return p_value


def calculate_overall_correlations(df: pd.DataFrame,
                                   metrics: List[str]) -> None:
    """
    Calculate and print correlations between SaCo scores and all metrics.
    
    Args:
        df: DataFrame with attribution metrics
        metrics: List of metric names to analyze
    """
    print("\n" + "=" * 60)
    print("OVERALL CORRELATIONS WITH SACO SCORE:")
    print("=" * 60)

    correlations_overall = {}
    for metric in metrics:
        if metric in df.columns:
            corr = df['saco_score'].corr(df[metric])
            p_value = calculate_correlation_significance(
                df['saco_score'].values, df[metric].values)
            correlations_overall[metric] = (corr, p_value)
            print(f"{metric}: r={corr:.3f}, p={p_value:.5f}")


def calculate_per_class_correlations(df: pd.DataFrame,
                                     metrics: List[str]) -> None:
    """
    Calculate and print correlations between SaCo scores and all metrics by class.
    
    Args:
        df: DataFrame with attribution metrics and class information
        metrics: List of metric names to analyze
    """
    if 'class' in df.columns:
        classes = df['class'].unique()

        for cls in classes:
            class_df = df[df['class'] == cls]

            if len(class_df
                   ) > 5:  # Only calculate if we have enough data points
                print("\n" + "-" * 60)
                print(f"CORRELATIONS FOR CLASS: {cls}")
                print("-" * 60)

                for metric in metrics:
                    if metric in class_df.columns:
                        corr = class_df['saco_score'].corr(class_df[metric])
                        p_value = calculate_correlation_significance(
                            class_df['saco_score'].values,
                            class_df[metric].values)
                        print(f"{metric}: r={corr:.3f}, p={p_value:.5f}")


def add_class_embedding_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add class embedding metrics for correlation analysis.
    
    Args:
        df: DataFrame with class embedding paths
        
    Returns:
        DataFrame with added class embedding metrics
    """
    # First check if we have the class_embedding_path column
    if 'class_embedding_path' not in df.columns:
        print("No class_embedding_path column found in DataFrame")
        return df

    for idx, row in df.iterrows():
        try:
            # Load class embedding data
            embeddings = np.load(row['class_embedding_path'],
                                 allow_pickle=True)

            # Last layer analysis (more refined)
            last_layer = embeddings[-1]
            attn_logits = last_layer['attention_class_representation']
            mlp_logits = last_layer['mlp_class_representation']

            # Convert to probabilities
            attn_probs = softmax(attn_logits, axis=-1)
            mlp_probs = softmax(mlp_logits, axis=-1)

            # For each class, compute image-level metrics
            num_classes = attn_probs.shape[-1]
            for cls in range(num_classes):
                # Create column names
                mean_attn_col = f'class_{cls}_mean_attn_prob'
                max_attn_col = f'class_{cls}_max_attn_prob'
                mean_mlp_col = f'class_{cls}_mean_mlp_prob'
                max_mlp_col = f'class_{cls}_max_mlp_prob'
                cls_attn_col = f'class_{cls}_cls_token_attn_prob'
                cls_mlp_col = f'class_{cls}_cls_token_mlp_prob'

                # Initialize columns if they don't exist
                if mean_attn_col not in df.columns:
                    df[mean_attn_col] = np.nan
                    df[max_attn_col] = np.nan
                    df[mean_mlp_col] = np.nan
                    df[max_mlp_col] = np.nan
                    df[cls_attn_col] = np.nan
                    df[cls_mlp_col] = np.nan

                # Add values directly to the DataFrame
                df.at[idx, mean_attn_col] = np.mean(attn_probs[:, cls])
                df.at[idx, max_attn_col] = np.max(attn_probs[:, cls])
                df.at[idx, mean_mlp_col] = np.mean(mlp_probs[:, cls])
                df.at[idx, max_mlp_col] = np.max(mlp_probs[:, cls])
                df.at[idx, cls_attn_col] = attn_probs[0, cls]
                df.at[idx, cls_mlp_col] = mlp_probs[0, cls]

        except Exception as e:
            print(
                f"Error processing class embeddings for {row.get('filename', 'unknown')}: {e}"
            )
            # Values already initialized as NaN, so we can continue
            continue

    return df


def calculate_class_specific_correlations(
        df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate correlations between SaCo scores and class-specific embeddings.
    
    Args:
        df: DataFrame with SaCo scores and class embedding metrics
        
    Returns:
        Dictionary of correlations by class and metric
    """
    class_correlations = {}

    # Get all class embedding metric columns
    class_columns = [
        col for col in df.columns
        if col.startswith('class_') and not col.endswith('_path')
    ]  # Exclude path columns

    for col in class_columns:
        if col in df.columns:
            # Only calculate correlation if we have valid values
            valid_mask = ~(pd.isna(df['saco_score']) | pd.isna(df[col]))
            if valid_mask.sum() > 2:  # Need at least 3 valid points
                corr = df.loc[valid_mask, 'saco_score'].corr(df.loc[valid_mask,
                                                                    col])
                p_value = calculate_correlation_significance(
                    df.loc[valid_mask, 'saco_score'].values,
                    df.loc[valid_mask, col].values)

                class_correlations[col] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'n_valid': valid_mask.sum()
                }

    # Print results organized by class
    print("\n" + "=" * 60)
    print("CLASS-SPECIFIC EMBEDDING CORRELATIONS WITH SACO SCORE:")
    print("=" * 60)

    # Group by class
    class_groups = {}
    for key, stats in class_correlations.items():
        parts = key.split('_')
        if len(parts) >= 2:
            class_num = parts[1]
            if class_num not in class_groups:
                class_groups[class_num] = {}
            class_groups[class_num][key] = stats

    for cls, metrics in class_groups.items():
        print(f"\nClass {cls}:")
        for metric, stats in metrics.items():
            if not np.isnan(stats['correlation']):
                print(
                    f"  {metric}: r={stats['correlation']:.3f}, p={stats['p_value']:.5f}, n={stats['n_valid']}"
                )

    return class_correlations


def analyze_patch_level_correlations(df: pd.DataFrame) -> None:
    """
    Analyze patch-level correlations between class embeddings and attributions.
    
    Args:
        df: DataFrame with attribution and class embedding paths
    """
    patch_correlations = []

    for _, row in df.iterrows():
        try:
            # Load data
            pos_attr = np.load(row['attribution_path'])
            neg_attr = np.load(row['attribution_neg_path'])
            embeddings = np.load(row['class_embedding_path'],
                                 allow_pickle=True)

            last_layer = embeddings[-1]
            attn_logits = last_layer['attention_class_representation']
            attn_probs = softmax(attn_logits, axis=-1)

            # Flatten attribution maps if they're 2D
            if pos_attr.ndim > 1:
                pos_attr_flat = pos_attr.flatten()
                neg_attr_flat = neg_attr.flatten()
            else:
                pos_attr_flat = pos_attr
                neg_attr_flat = neg_attr

            # Handle dimension matching
            # Typically, we need to exclude the CLS token
            if attn_probs.shape[0] > len(pos_attr_flat):
                # Exclude CLS token (usually the first token)
                token_probs = attn_probs[1:, :]
            else:
                token_probs = attn_probs

            # Check if dimensions match after adjustment
            if len(token_probs) != len(pos_attr_flat):
                print(
                    f"Dimension mismatch for {row['filename']}: tokens={len(token_probs)}, attributions={len(pos_attr_flat)}"
                )
                continue

            # For each class, calculate patch-level correlations
            for cls in range(token_probs.shape[1]):
                class_probs = token_probs[:, cls]

                # Calculate correlations
                pos_corr = np.corrcoef(class_probs, pos_attr_flat)[0, 1]
                neg_corr = np.corrcoef(class_probs, neg_attr_flat)[0, 1]

                # Only add if correlations are valid (not NaN)
                if not np.isnan(pos_corr) and not np.isnan(neg_corr):
                    patch_correlations.append({
                        'filename': row['filename'],
                        'true_class': row['true_class'],
                        'class_idx': cls,
                        'pos_attribution_corr': pos_corr,
                        'neg_attribution_corr': neg_corr
                    })

        except Exception as e:
            print(
                f"Error in patch-level analysis for {row.get('filename', 'unknown')}: {e}"
            )

    # Check if we have any correlations
    if not patch_correlations:
        print("\nNo patch-level correlations could be calculated.")
        return

    # Convert to DataFrame and analyze
    patch_df = pd.DataFrame(patch_correlations)

    # Aggregate results
    print("\n" + "=" * 60)
    print("PATCH-LEVEL CLASS EMBEDDING CORRELATIONS:")
    print("=" * 60)

    for cls in sorted(patch_df['class_idx'].unique()):
        cls_data = patch_df[patch_df['class_idx'] == cls]
        print(f"\nClass {cls}:")
        print(
            f"  Mean correlation with positive attributions: {cls_data['pos_attribution_corr'].mean():.3f}"
        )
        print(
            f"  Mean correlation with negative attributions: {cls_data['neg_attribution_corr'].mean():.3f}"
        )
        print(f"  Number of samples: {len(cls_data)}")

        # Also show breakdown by true class
        print(f"  Breakdown by true class:")
        for true_cls in sorted(cls_data['true_class'].unique()):
            true_cls_data = cls_data[cls_data['true_class'] == true_cls]
            if len(true_cls_data) > 0:
                print(
                    f"    {true_cls}: pos={true_cls_data['pos_attribution_corr'].mean():.3f}, "
                    f"neg={true_cls_data['neg_attribution_corr'].mean():.3f}, n={len(true_cls_data)}"
                )


# Metric Calculation Functions
def add_basic_attribution_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic attribution magnitude metrics to the DataFrame.
    
    Args:
        df: DataFrame with attribution paths
        
    Returns:
        DataFrame with added basic attribution metrics
    """
    metrics = {
        'neg_magnitude': [],  # Total magnitude of negative attributions
        'pos_magnitude': [],  # Total magnitude of positive attributions
        'neg_max': [],  # Maximum negative attribution value
        'pos_max': [],  # Maximum positive attribution value
        'neg_pos_ratio': []  # Ratio of negative to positive magnitudes
    }

    for _, row in df.iterrows():
        try:
            pos_attr = np.load(row['attribution_path'])
            neg_attr = np.load(
                row['attribution_neg_path'])  # Already absolute values

            # Print diagnostics for first few files
            if len(metrics['neg_magnitude']) < 3:
                print(f"File: {row['filename']}")
                print(
                    f"Positive attr shape: {pos_attr.shape}, min: {pos_attr.min():.5f}, max: {pos_attr.max():.5f}"
                )
                print(
                    f"Negative attr shape: {neg_attr.shape}, min: {neg_attr.min():.5f}, max: {neg_attr.max():.5f}"
                )

            neg_sum = np.sum(neg_attr)
            pos_sum = np.sum(pos_attr)

            metrics['neg_magnitude'].append(neg_sum)
            metrics['pos_magnitude'].append(pos_sum)
            metrics['neg_max'].append(np.max(neg_attr))
            metrics['pos_max'].append(np.max(pos_attr))

            # Calculate ratio safely
            if pos_sum > 1e-10:
                metrics['neg_pos_ratio'].append(neg_sum / pos_sum)
            else:
                metrics['neg_pos_ratio'].append(np.nan)

        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def add_entropy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add entropy-based metrics for attribution maps.
    
    Args:
        df: DataFrame with attribution paths
        
    Returns:
        DataFrame with added entropy metrics
    """
    metrics = {
        'neg_entropy': [],  # Shannon entropy of negative attributions
        'pos_entropy': [],  # Shannon entropy of positive attributions
        'entropy_ratio': []  # Ratio of negative to positive entropy
    }

    for _, row in df.iterrows():
        try:
            pos_attr = np.load(row['attribution_path'])
            neg_attr = np.load(row['attribution_neg_path'])

            # Normalize distributions for entropy calculation
            pos_norm = pos_attr / (np.sum(pos_attr) + 1e-10)
            neg_norm = neg_attr / (np.sum(neg_attr) + 1e-10)

            # Calculate Shannon entropy (using only non-zero values)
            pos_entropy = -np.sum(pos_norm[pos_norm > 0] *
                                  np.log2(pos_norm[pos_norm > 0] + 1e-10))
            neg_entropy = -np.sum(neg_norm[neg_norm > 0] *
                                  np.log2(neg_norm[neg_norm > 0] + 1e-10))

            metrics['neg_entropy'].append(neg_entropy)
            metrics['pos_entropy'].append(pos_entropy)
            metrics['entropy_ratio'].append(neg_entropy /
                                            (pos_entropy + 1e-10))

        except Exception as e:
            print(f"Error calculating entropy for {row['filename']}: {e}")
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def add_concentration_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add concentration metrics for attribution maps.
    
    Args:
        df: DataFrame with attribution paths
        
    Returns:
        DataFrame with added concentration metrics
    """
    metrics = {
        'neg_gini': [],  # Gini coefficient for negative attributions
        'pos_gini': [],  # Gini coefficient for positive attributions
        'neg_top10_conc': [],  # Concentration of top 10% negative attributions
        'pos_top10_conc': []  # Concentration of top 10% positive attributions
    }

    for _, row in df.iterrows():
        try:
            pos_attr = np.load(row['attribution_path']).flatten()
            neg_attr = np.load(row['attribution_neg_path']).flatten()

            # Calculate Gini coefficients
            metrics['neg_gini'].append(calculate_gini(neg_attr))
            metrics['pos_gini'].append(calculate_gini(pos_attr))

            # Top 10% concentration
            neg_threshold = np.percentile(neg_attr, 90)
            pos_threshold = np.percentile(pos_attr, 90)
            metrics['neg_top10_conc'].append(
                np.sum(neg_attr[neg_attr >= neg_threshold]) /
                (np.sum(neg_attr) + 1e-10))
            metrics['pos_top10_conc'].append(
                np.sum(pos_attr[pos_attr >= pos_threshold]) /
                (np.sum(pos_attr) + 1e-10))

        except Exception as e:
            print(
                f"Error calculating concentration for {row['filename']}: {e}")
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def calculate_gini(x: np.ndarray) -> float:
    """
    Calculate the Gini coefficient for an array of values.
    
    Args:
        x: Array of values
        
    Returns:
        Gini coefficient
    """
    # Sort values
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    # Return Gini coefficient
    return (np.sum((2 * np.arange(1, n + 1) - n - 1) *
                   sorted_x)) / (n * np.sum(sorted_x) + 1e-10)


def add_information_theory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add information theory metrics for attribution maps.
    
    Args:
        df: DataFrame with attribution paths
        
    Returns:
        DataFrame with added information theory metrics
    """
    metrics = {
        'mutual_information':
        [],  # Mutual information between positive and negative attributions
        'neg_pos_contingency':
        []  # Contingency between negative and positive attributions
    }

    for _, row in df.iterrows():
        try:
            pos_attr = np.load(row['attribution_path'])
            neg_attr = np.load(row['attribution_neg_path'])

            if pos_attr.shape != neg_attr.shape:
                for key in metrics:
                    metrics[key].append(np.nan)
                continue

            # Discretize for information theory metrics (into 20 bins)
            pos_bins = np.linspace(pos_attr.min(), pos_attr.max(), 20)
            neg_bins = np.linspace(neg_attr.min(), neg_attr.max(), 20)

            pos_discrete = np.digitize(pos_attr.flatten(), pos_bins)
            neg_discrete = np.digitize(neg_attr.flatten(), neg_bins)

            # Calculate mutual information
            mi = mutual_info_score(pos_discrete, neg_discrete)
            metrics['mutual_information'].append(mi)

            # Calculate contingency (how often high/low values co-occur)
            pos_median = np.median(pos_attr)
            neg_median = np.median(neg_attr)

            high_pos = pos_attr > pos_median
            high_neg = neg_attr > neg_median

            # Calculate contingency ratio
            contingency = (np.sum(np.logical_and(high_pos, high_neg)) + np.sum(
                np.logical_and(~high_pos, ~high_neg))) / pos_attr.size
            metrics['neg_pos_contingency'].append(contingency)

        except Exception as e:
            print(
                f"Error calculating information metrics for {row['filename']}: {e}"
            )
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def add_gradient_based_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add gradient-based metrics for attribution maps.
    
    Args:
        df: DataFrame with attribution paths
        
    Returns:
        DataFrame with added gradient-based metrics
    """
    metrics = {
        'gradient_magnitude': [],  # Average magnitude of gradients
        'gradient_variance': [],  # Variance of gradient values
        'gradient_entropy': [],  # Entropy of gradient distribution
        'gradient_sparsity': []  # Percentage of near-zero gradients
    }

    for _, row in df.iterrows():
        try:
            # For gradient-based metrics, we'll use the positive and negative
            # attribution maps as proxies for gradient information
            pos_attr = np.load(row['attribution_path'])
            neg_attr = np.load(row['attribution_neg_path'])

            # Combined attribution (absolute values)
            combined_attr = np.abs(pos_attr) + np.abs(neg_attr)

            # Calculate metrics
            metrics['gradient_magnitude'].append(np.mean(combined_attr))
            metrics['gradient_variance'].append(np.var(combined_attr))

            # Calculate entropy of normalized gradients
            grad_norm = combined_attr / (np.sum(combined_attr) + 1e-10)
            entropy = -np.sum(grad_norm * np.log2(grad_norm + 1e-10))
            metrics['gradient_entropy'].append(entropy)

            # Calculate sparsity (% of gradients close to zero)
            threshold = 0.01 * np.max(combined_attr)
            sparsity = np.mean(combined_attr < threshold)
            metrics['gradient_sparsity'].append(sparsity)

        except Exception as e:
            print(
                f"Error processing gradients for {row.get('filename', 'unknown')}: {e}"
            )
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def add_sparsity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sparsity metrics for attribution maps.
    
    Args:
        df: DataFrame with attribution paths
        
    Returns:
        DataFrame with added sparsity metrics
    """
    metrics = {
        'pos_l0_sparsity': [],  # Percentage of near-zero positive attributions
        'neg_l0_sparsity': [],  # Percentage of near-zero negative attributions
        'pos_effective_sparsity':
        [],  # Effective number of non-zero positive elements
        'neg_effective_sparsity':
        [],  # Effective number of non-zero negative elements
        'pos_kurtosis': [],  # Measure of "peakedness" of positive distribution
        'neg_kurtosis': []  # Measure of "peakedness" of negative distribution
    }

    for _, row in df.iterrows():
        try:
            pos_attr = np.load(row['attribution_path'])
            neg_attr = np.load(row['attribution_neg_path'])

            flat_pos = pos_attr.flatten()
            flat_neg = neg_attr.flatten()

            # L0 sparsity (percentage of near-zero elements)
            pos_threshold = 0.01 * np.max(pos_attr)
            neg_threshold = 0.01 * np.max(neg_attr)
            metrics['pos_l0_sparsity'].append(
                np.mean(pos_attr < pos_threshold))
            metrics['neg_l0_sparsity'].append(
                np.mean(neg_attr < neg_threshold))

            # Effective sparsity for positive attributions
            pos_normalized = flat_pos / (np.sum(flat_pos) + 1e-10)
            pos_effective_nonzero = 1.0 / (np.sum(pos_normalized**2) + 1e-10)
            metrics['pos_effective_sparsity'].append(1.0 -
                                                     (pos_effective_nonzero /
                                                      len(flat_pos)))

            # Effective sparsity for negative attributions
            neg_normalized = flat_neg / (np.sum(flat_neg) + 1e-10)
            neg_effective_nonzero = 1.0 / (np.sum(neg_normalized**2) + 1e-10)
            metrics['neg_effective_sparsity'].append(1.0 -
                                                     (neg_effective_nonzero /
                                                      len(flat_neg)))

            # Kurtosis (measure of "peakedness")
            metrics['pos_kurtosis'].append(stats.kurtosis(flat_pos))
            metrics['neg_kurtosis'].append(stats.kurtosis(flat_neg))

        except Exception as e:
            print(
                f"Error calculating sparsity for {row.get('filename', 'unknown')}: {e}"
            )
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def add_attribution_consistency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add consistency metrics for attribution maps.
    
    Args:
        df: DataFrame with attribution paths
        
    Returns:
        DataFrame with added consistency metrics
    """
    metrics = {
        'pos_neg_coherence':
        [],  # Correlation between positive and negative attribution patterns
        'attribution_evenness':
        [],  # How evenly distributed the attribution intensity is across regions
        'feature_consensus': []  # Agreement between different attention heads
    }

    for _, row in df.iterrows():
        try:
            pos_attr = np.load(row['attribution_path'])
            neg_attr = np.load(row['attribution_neg_path'])

            # Correlation between positive and negative maps
            flat_pos = pos_attr.flatten()
            flat_neg = neg_attr.flatten()
            pos_neg_corr = np.corrcoef(flat_pos, flat_neg)[0, 1]
            metrics['pos_neg_coherence'].append(
                abs(pos_neg_corr))  # Use absolute value to measure alignment

            # Measure attribution evenness (ratio of mean to max)
            combined_attr = pos_attr + neg_attr
            evenness = np.mean(combined_attr) / (np.max(combined_attr) + 1e-10)
            metrics['attribution_evenness'].append(evenness)

            # If head-specific attributions are available, measure agreement between heads
            if 'attribution_heads_path' in row and pd.notnull(
                    row['attribution_heads_path']):
                heads_attr = np.load(row['attribution_heads_path'])
                feature_consensus = calculate_feature_consensus(heads_attr)
                metrics['feature_consensus'].append(feature_consensus)
            else:
                metrics['feature_consensus'].append(np.nan)

        except Exception as e:
            print(
                f"Error calculating consistency for {row.get('filename', 'unknown')}: {e}"
            )
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def calculate_feature_consensus(heads_attr: np.ndarray) -> float:
    """
    Calculate consensus among different attention heads.
    
    Args:
        heads_attr: Array of attribution values for each head
        
    Returns:
        Mean correlation between heads
    """
    num_heads = heads_attr.shape[0]
    correlations = []

    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            corr = np.corrcoef(heads_attr[i].flatten(),
                               heads_attr[j].flatten())[0, 1]
            correlations.append(corr)

    return np.mean(correlations) if correlations else np.nan


def add_robustness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add robustness metrics for attribution maps.
    
    Args:
        df: DataFrame with attribution paths
        
    Returns:
        DataFrame with added robustness metrics
    """
    metrics = {
        'smoothing_stability': [],  # Stability under Gaussian smoothing
        'noise_stability': [],  # Stability under small noise addition
        'peak_persistence':
        []  # How persistent the highest attribution regions are
    }

    for _, row in df.iterrows():
        try:
            pos_attr = np.load(row['attribution_path'])

            # Smoothing stability
            smoothed = gaussian_filter(pos_attr, sigma=1.0)
            stability = np.corrcoef(pos_attr.flatten(), smoothed.flatten())[0,
                                                                            1]
            metrics['smoothing_stability'].append(stability)

            # Noise stability (adding small random noise)
            noise_level = 0.05 * np.std(pos_attr)
            noise = np.random.normal(0, noise_level, pos_attr.shape)
            noisy = pos_attr + noise
            noise_stability = np.corrcoef(pos_attr.flatten(),
                                          noisy.flatten())[0, 1]
            metrics['noise_stability'].append(noise_stability)

            # Peak persistence
            # How much the top 10% attribution region overlaps after smoothing
            threshold = np.percentile(pos_attr, 90)
            smoothed_threshold = np.percentile(smoothed, 90)

            top_orig = pos_attr >= threshold
            top_smoothed = smoothed >= smoothed_threshold

            overlap = np.logical_and(top_orig, top_smoothed)
            persistence = np.sum(overlap) / (np.sum(top_orig) + 1e-10)
            metrics['peak_persistence'].append(persistence)

        except Exception as e:
            print(
                f"Error calculating robustness for {row.get('filename', 'unknown')}: {e}"
            )
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def add_ffn_activity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add FFN activity metrics to the dataframe for correlation analysis.
    
    Args:
        df: DataFrame with FFN activity paths
        
    Returns:
        DataFrame with added FFN activity metrics
    """
    metrics = {
        'ffn_mean_activity': [],  # Mean activity across all layers and tokens
        'ffn_cls_activity': [],  # Mean activity of CLS token across layers
        'ffn_last_layer_activity': [],  # Activity in the last layer
        'ffn_max_layer_activity': [],  # Maximum layer-wise activity
        'ffn_early_layers_activity': [],  # Activity in first third of layers
        'ffn_middle_layers_activity': [],  # Activity in middle third of layers
        'ffn_late_layers_activity': [],  # Activity in last third of layers
        'ffn_layer_activity_variance':
        [],  # Variance of activity across layers
        'ffn_token_activity_variance':
        []  # Mean variance of activity across tokens
    }

    for _, row in df.iterrows():
        if 'ffn_activity_path' not in row or pd.isna(row['ffn_activity_path']):
            for key in metrics:
                metrics[key].append(np.nan)
            continue

        try:
            # Load FFN activity data
            ffn_data = np.load(row['ffn_activity_path'], allow_pickle=True)

            # Calculate various aggregations
            layer_means = [layer['mean_activity'] for layer in ffn_data]
            metrics['ffn_mean_activity'].append(np.mean(layer_means))

            cls_activities = [
                layer['cls_activity'] for layer in ffn_data
                if 'cls_activity' in layer
            ]
            metrics['ffn_cls_activity'].append(
                np.mean(cls_activities) if cls_activities else np.nan)

            metrics['ffn_last_layer_activity'].append(
                ffn_data[-1]['mean_activity'])
            metrics['ffn_max_layer_activity'].append(np.max(layer_means))

            # Activity by layer groups (early, middle, late)
            num_layers = len(ffn_data)
            third = max(1, num_layers // 3)
            metrics['ffn_early_layers_activity'].append(
                np.mean(layer_means[:third]))
            metrics['ffn_middle_layers_activity'].append(
                np.mean(layer_means[third:2 * third]))
            metrics['ffn_late_layers_activity'].append(
                np.mean(layer_means[2 * third:]))

            metrics['ffn_layer_activity_variance'].append(np.var(layer_means))

            # Mean variance across tokens
            token_variances = calculate_token_variances(ffn_data)
            metrics['ffn_token_activity_variance'].append(
                np.mean(token_variances) if token_variances else np.nan)

        except Exception as e:
            print(
                f"Error processing FFN activity for {row.get('filename', 'unknown')}: {e}"
            )
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def calculate_token_variances(ffn_data: np.ndarray) -> List[float]:
    """
    Calculate variance of token activities across layers.
    
    Args:
        ffn_data: Array of FFN activities
        
    Returns:
        List of token variance values
    """
    token_variances = []

    for i in range(len(ffn_data)):
        if 'activity' not in ffn_data[i] or not isinstance(
                ffn_data[i]['activity'], np.ndarray):
            continue

        token_activity = ffn_data[i]['activity']
        if token_activity.ndim > 0 and token_activity.shape[0] > 1:
            token_variances.append(np.var(
                token_activity[1:]))  # Exclude CLS token

    return token_variances


# Visualization Functions
def compare_concentration_distributions(
        df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compare the distribution of concentration metrics across classes.
    
    Args:
        df: DataFrame with attribution metrics and class information
        
    Returns:
        Dictionary with statistics by class and metric
    """
    # Skip if there's no class column or not enough data
    if 'class' not in df.columns or len(df) < 5:
        return {}

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['neg_gini', 'pos_gini', 'neg_entropy', 'pos_entropy']

    # Calculate statistics
    stats = {}
    for cls in df['class'].unique():
        stats[cls] = {}
        for metric in metrics:
            if metric not in df.columns:
                continue

            class_values = df[df['class'] == cls][metric]
            stats[cls][metric] = {
                'mean': class_values.mean(),
                'std': class_values.std(),
                'range': class_values.max() - class_values.min(),
                'percentiles': np.percentile(class_values,
                                             [10, 25, 50, 75, 90])
            }

    # Print the statistics
    print("Concentration Distribution By Class")
    print("===================================")

    for metric in metrics:
        if metric not in df.columns:
            continue

        print(f"\n{metric}")
        for cls in stats:
            if metric not in stats[cls]:
                continue

            pct = stats[cls][metric]['percentiles']
            print(f"{cls}:")
            print(f"  Mean: {stats[cls][metric]['mean']:.3f}")
            print(f"  Std Dev: {stats[cls][metric]['std']:.3f}")
            print(f"  Range (10th-90th): {pct[4]-pct[0]:.3f}")
            print(f"  Distribution: 10%={pct[0]:.3f}, 25%={pct[1]:.3f}, " +
                  f"50%={pct[2]:.3f}, 75%={pct[3]:.3f}, 90%={pct[4]:.3f}")

    # Create the plots
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue

        ax = axes[i // 2, i % 2]
        for cls in df['class'].unique():
            sns.kdeplot(df[df['class'] == cls][metric], ax=ax, label=cls)

        ax.set_title(f'Distribution of {metric}')
        ax.legend()

    plt.tight_layout()
    plt.savefig('concentration_distributions.png')

    return stats


def analyze_patch_saco_class_correlations(
        faithfulness_df: pd.DataFrame, patch_metrics_df: pd.DataFrame) -> None:
    """
    Analyze correlations between patch-level SaCo scores and class embedding probabilities,
    grouped by the model's predicted class.
    
    Args:
        faithfulness_df: DataFrame with class embedding paths and predicted classes
        patch_metrics_df: DataFrame with patch-level SaCo scores
    """
    results_by_prediction = {}

    # Group by image and predicted class
    image_predictions = {}
    for _, row in faithfulness_df.iterrows():
        image_name = row['filename']
        predicted_class = row['predicted_class']
        class_embedding_path = row['class_embedding_path']

        if image_name not in image_predictions:
            image_predictions[image_name] = {
                'predicted_class': predicted_class,
                'class_embedding_path': class_embedding_path,
                'attribution_path': row['attribution_path']
            }

    # Process each image in patch metrics
    for image_name, image_data in patch_metrics_df.groupby('image_name'):
        if image_name not in image_predictions:
            continue

        predicted_class = image_predictions[image_name]['predicted_class']

        try:
            # Load class embeddings
            embeddings = np.load(
                image_predictions[image_name]['class_embedding_path'],
                allow_pickle=True)
            last_layer = embeddings[-1]
            attn_logits = last_layer['attention_class_representation']
            attn_probs = softmax(attn_logits, axis=-1)

            # Remove CLS token
            token_probs = attn_probs[1:, :]

            # Load attribution map to get patch ordering
            attr_map = np.load(
                image_predictions[image_name]['attribution_path'])

            # Convert to patch grid indices
            img_size = int(np.sqrt(
                attr_map.size)) if attr_map.ndim == 1 else attr_map.shape[0]
            patch_size = 16
            num_patches = img_size // patch_size

            # Map patch IDs to grid positions
            patch_id_to_index = {}
            index = 0
            for i in range(num_patches):
                for j in range(num_patches):
                    patch_id_to_index[index] = (i, j)
                    index += 1

            # Initialize results structure
            if predicted_class not in results_by_prediction:
                results_by_prediction[predicted_class] = {
                    'saco_scores': [],
                    'class_probs': {
                        0: [],
                        1: [],
                        2: []
                    }  # Assuming 3 classes
                }

            # Process each patch in this image
            for _, patch_row in image_data.iterrows():
                patch_id = patch_row['patch_id']
                patch_saco = patch_row['patch_saco']

                # Map patch ID to token index
                # This depends on how your patch IDs correspond to positions
                # Assuming patch_id is a sequential index
                if patch_id < len(token_probs):
                    results_by_prediction[predicted_class][
                        'saco_scores'].append(patch_saco)

                    for cls in range(token_probs.shape[1]):
                        results_by_prediction[predicted_class]['class_probs'][
                            cls].append(token_probs[patch_id, cls])

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue

    # Calculate and display correlations
    print("\n" + "=" * 60)
    print("PATCH-LEVEL SACO CORRELATIONS BY PREDICTED CLASS:")
    print("=" * 60)

    class_names = {0: 'COVID-19', 1: 'Non-COVID', 2: 'Normal'}

    for pred_class, data in results_by_prediction.items():
        print(f"\nPredicted Class: {pred_class}")
        saco_scores = np.array(data['saco_scores'])

        if len(saco_scores) == 0:
            print("  No data available for this class")
            continue

        for cls in data['class_probs']:
            class_probs = np.array(data['class_probs'][cls])
            if len(class_probs) > 0:
                corr = np.corrcoef(saco_scores, class_probs)[0, 1]
                p_value = calculate_correlation_significance(
                    saco_scores, class_probs)
                print(
                    f"  Correlation with class {class_names.get(cls, cls)}: r={corr:.3f}, p={p_value:.5f}"
                )

        # Additional statistics
        print(f"  Number of patches analyzed: {len(saco_scores)}")
        print(f"  Mean patch SaCo score: {np.mean(saco_scores):.3f}")

        # Show distribution by class probability
        for cls in data['class_probs']:
            class_probs = np.array(data['class_probs'][cls])
            if len(class_probs) > 0:
                high_prob_mask = class_probs > 0.5
                if np.any(high_prob_mask):
                    high_prob_saco = saco_scores[high_prob_mask]
                    low_prob_saco = saco_scores[~high_prob_mask]
                    print(
                        f"  Class {class_names.get(cls, cls)} - High prob (>0.5) mean SaCo: {np.mean(high_prob_saco):.3f}"
                    )
                    print(
                        f"  Class {class_names.get(cls, cls)} - Low prob (<=0.5) mean SaCo: {np.mean(low_prob_saco):.3f}"
                    )


def calculate_class_correlations_by_prediction(df: pd.DataFrame) -> None:
    """
    Calculate correlations between SaCo scores and class-specific embeddings,
    grouped by predicted class.
    
    Args:
        df: DataFrame with SaCo scores, class embedding metrics, and predicted classes
    """
    # Group by predicted class
    predicted_classes = df['predicted_class'].unique()

    print("\n" + "=" * 60)
    print("CLASS-SPECIFIC EMBEDDING CORRELATIONS BY PREDICTED CLASS:")
    print("=" * 60)

    for pred_class in sorted(predicted_classes):
        pred_df = df[df['predicted_class'] == pred_class]

        print(f"\nPREDICTED CLASS: {pred_class}")
        print("-" * 40)

        # Get all class embedding metric columns
        class_columns = [
            col for col in df.columns
            if col.startswith('class_') and not col.endswith('_path')
        ]

        # Group columns by class
        class_groups = {}
        for col in class_columns:
            parts = col.split('_')
            if len(parts) >= 2:
                class_num = parts[1]
                if class_num not in class_groups:
                    class_groups[class_num] = []
                class_groups[class_num].append(col)

        # Calculate correlations for each class
        for cls in sorted(class_groups.keys()):
            print(f"\nClass {cls}:")

            for metric in sorted(class_groups[cls]):
                if metric in pred_df.columns:
                    # Only calculate correlation if we have valid values
                    valid_mask = ~(pd.isna(pred_df['saco_score'])
                                   | pd.isna(pred_df[metric]))

                    if valid_mask.sum() > 2:  # Need at least 3 valid points
                        corr = pred_df.loc[valid_mask, 'saco_score'].corr(
                            pred_df.loc[valid_mask, metric])
                        p_value = calculate_correlation_significance(
                            pred_df.loc[valid_mask, 'saco_score'].values,
                            pred_df.loc[valid_mask, metric].values)

                        print(
                            f"  {metric}: r={corr:.3f}, p={p_value:.5f}, n={valid_mask.sum()}"
                        )

        # Add summary statistics for this predicted class
        print(f"\nSummary for {pred_class}:")
        print(f"  Total samples: {len(pred_df)}")
        print(f"  Mean SaCo score: {pred_df['saco_score'].mean():.3f}")
        print(f"  Median SaCo score: {pred_df['saco_score'].median():.3f}")
        print(f"  Std SaCo score: {pred_df['saco_score'].std():.3f}")
