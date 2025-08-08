# analysis.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

from data_types import (AnalysisContext, ClassificationResult, PerturbationPatchInfo)
from vit.model import CLASSES, CLS2IDX, IDX2CLS, VisionTransformer


def generate_perturbation_comparison_dataframe(
    context: AnalysisContext, generate_visualizations: bool = True
) -> pd.DataFrame:
    """
    Generates the comparison DataFrame needed for SaCo scores, 
    mimicking the initial part of the old `compare_attributions`.
    This DataFrame will contain paths and basic computed info.
    """
    comparison_data_list = []
    analysis_results_dir = context.config.file.output_dir
    viz_comparison_subdir = analysis_results_dir / "saco_perturbation_comparisons_viz"

    if generate_visualizations:
        viz_comparison_subdir.mkdir(parents=True, exist_ok=True)

    # Efficiently map original results by their image path for quick lookup
    original_results_map: Dict[Path, ClassificationResult] = {
        orig_res.image_path: orig_res
        for orig_res in context.original_results
    }

    print("Generating perturbation comparison data for SaCo...")
    for p_record in tqdm(context.all_perturbed_records, desc="Processing Perturbations"):
        original_image_path = p_record.original_image_path
        perturbed_image_path = p_record.perturbed_image_path

        if original_image_path not in original_results_map:
            print(
                f"Warning: Original result for {original_image_path} not found. Skipping perturbation {perturbed_image_path.name}."
            )
            continue

        original_class_res = original_results_map[original_image_path]
        perturbed_class_res = context.perturbed_classification_results_map.get(perturbed_image_path)

        if perturbed_class_res is None:
            continue

        original_attr_np = None
        if original_class_res.attribution_paths and original_class_res.attribution_paths.attribution_path.exists():
            try:
                original_attr_np = np.load(original_class_res.attribution_paths.attribution_path)
            except Exception as e:
                print(f"Error loading original attribution for {original_class_res.image_path.name}: {e}")
                continue  # Skip if essential data is missing
        else:
            continue

        # Perturbed attribution (positive) - only if visualizations are needed or if SaCo uses it directly
        perturbed_attr_np = None
        if generate_visualizations:  # Or if perturbed attribution itself is a metric
            if perturbed_class_res.attribution_paths and perturbed_class_res.attribution_paths.attribution_path.exists(
            ):
                try:
                    perturbed_attr_np = np.load(perturbed_class_res.attribution_paths.attribution_path)
                except Exception as e:
                    print(
                        f"Error loading perturbed positive attribution for {perturbed_class_res.image_path.name}: {e}"
                    )

        # Patch info from PerturbedImageRecord
        patch_info_obj: PerturbationPatchInfo = p_record.patch_info
        patch_id = patch_info_obj.patch_id
        x_coord = patch_info_obj.x
        y_coord = patch_info_obj.y

        # Calculate mean attribution in the patch on the original image
        mean_original_patch_attr = calculate_patch_mean_attribution(
            original_attr_np, x_coord, y_coord, patch_size=context.config.perturb.patch_size
        )

        row_data = {
            "original_image":
            str(original_class_res.image_path),  # Old: "original_image"
            "perturbed_image":
            str(perturbed_class_res.image_path),  # Old: "perturbed_image"
            "patch_id":
            patch_id,
            "x":
            x_coord,  # Assuming these were used by SaCo or for context
            "y":
            y_coord,  # Assuming these were used by SaCo or for context
            "original_patch_mean_attribution":
            mean_original_patch_attr,  # Critical for SaCo - old: "mean_attribution"
            "original_predicted_class":
            original_class_res.prediction.predicted_class_label,  # Old: "original_class"
            "original_predicted_idx":
            original_class_res.prediction.predicted_class_idx,
            "original_confidence":
            original_class_res.prediction.confidence,  # Old: "original_confidence"
            "perturbed_predicted_class":
            perturbed_class_res.prediction.predicted_class_label,  # Old: "perturbed_class"
            "perturbed_predicted_idx":
            perturbed_class_res.prediction.predicted_class_idx,
            "perturbed_confidence":
            perturbed_class_res.prediction.confidence,  # Old: "perturbed_confidence"
            "class_changed":
            original_class_res.prediction.predicted_class_idx != perturbed_class_res.prediction.predicted_class_idx,
            "confidence_delta":
            perturbed_class_res.prediction.confidence - original_class_res.prediction.confidence,
            "confidence_delta_abs":
            abs(perturbed_class_res.prediction.confidence -
                original_class_res.prediction.confidence),  # Critical for SaCo

            # Paths for potential later detailed loading if other analyses need them
            "original_attribution_path":
            str(original_class_res.attribution_paths.attribution_path)
            if original_class_res.attribution_paths else None,
            "perturbed_attribution_path":
            str(perturbed_class_res.attribution_paths.attribution_path)
            if perturbed_class_res.attribution_paths else None,
            "mask_path":
            str(p_record.mask_path)
        }

        comparison_data_list.append(row_data)

    comparison_df = pd.DataFrame(comparison_data_list)

    if 'original_patch_mean_attribution' in comparison_df.columns:
        comparison_df = comparison_df.rename(columns={'original_patch_mean_attribution': 'mean_attribution'})

    # Save this intermediate DataFrame
    output_csv_path = analysis_results_dir / f"perturbation_comparison_for_saco{context.config.file.output_suffix}.csv"
    comparison_df.to_csv(output_csv_path, index=False)
    print(f"Base comparison DataFrame for SaCo saved to {output_csv_path}")

    return comparison_df


def run_saco_analysis(
    context: AnalysisContext,
    perturbation_comparison_df: pd.DataFrame,
    perturb_method_filter: str = "mean"
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Orchestrates SaCo calculation using the provided comparison DataFrame.
    """
    if perturbation_comparison_df.empty:
        print("SaCo Analysis: Input comparison DataFrame is empty. Skipping.")
        return {}, {}, None

    saco_scores, pair_data = calculate_saco_with_details_from_df(
        perturbation_comparison_df,  # Pass the DataFrame directly
        method_filter_str=perturb_method_filter  # For any internal filtering if still needed
    )

    avg_saco = np.mean(list(saco_scores.values())) if saco_scores else 0
    print(f"Average SaCo score for method '{perturb_method_filter}': {avg_saco:.4f} (over {len(saco_scores)} images)")

    patch_analysis_df = analyze_patch_metrics(pair_data)  # This should be fine

    # Save patch_analysis_df
    patch_analysis_csv_path = context.config.file.output_dir / f"saco_patch_analysis_{perturb_method_filter}{context.config.file.output_suffix}.csv"
    patch_analysis_df.to_csv(patch_analysis_csv_path, index=False)
    print(f"SaCo patch analysis saved to {patch_analysis_csv_path}")

    return saco_scores, pair_data, patch_analysis_df


def calculate_saco_with_details_from_df(  # NEW: Takes DF instead of CSV path
    comparison_df: pd.DataFrame,
    method_filter_str: Optional[str] = None
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:
    """
    Calculate SaCo (Saliency Correlation) scores from comparison data DataFrame.
    Modified from original to take DataFrame.
    """
    data_df = comparison_df.copy()  # Work on a copy

    results = {}
    pair_data_map = {}  # Renamed from pair_data to avoid conflict

    if 'mean_attribution' not in data_df.columns or \
       'confidence_delta_abs' not in data_df.columns or \
       'patch_id' not in data_df.columns:
        print(
            "Error: SaCo requires 'mean_attribution', 'confidence_delta_abs', and 'patch_id' columns in the DataFrame."
        )
        return {}, {}

    for image_name, image_data_group in data_df.groupby('original_image'):
        # Original code sorts by 'mean_attribution'
        image_data_sorted = image_data_group.sort_values('mean_attribution', ascending=False).reset_index(drop=True)

        attributions_np = image_data_sorted['mean_attribution'].values
        confidence_impacts_np = image_data_sorted['confidence_delta_abs'].values
        patch_ids_np = image_data_sorted['patch_id'].values

        saco_score, image_pair_df = calculate_image_saco_with_details(
            attributions_np, confidence_impacts_np, patch_ids_np
        )

        results[image_name] = saco_score
        pair_data_map[image_name] = image_pair_df

    return results, pair_data_map


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


def calculate_patch_mean_attribution(attribution: np.ndarray, x: int, y: int, patch_size: int = 16) -> Optional[float]:
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


# SaCo Calculation Functions
def calculate_saco_with_details(data_path: Path,
                                method: str = "mean") -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:
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
        data_df = data_df[data_df['perturbed_image'].str.contains(f"_{method}.png")]

    results = {}
    pair_data = {}

    for image_name, image_data in data_df.groupby('original_image'):
        image_data = image_data.sort_values('mean_attribution', ascending=False).reset_index(drop=True)
        attributions = image_data['mean_attribution'].values
        confidence_impacts = image_data['confidence_delta_abs'].values
        patch_ids = image_data['patch_id'].values

        saco_score, image_pair_data = calculate_image_saco_with_details(attributions, confidence_impacts, patch_ids)

        results[image_name] = saco_score
        pair_data[image_name] = image_pair_data

    print(f"Average SaCo score: {np.mean(list(results.values())):.4f}")
    print(len(results))
    return results, pair_data


def calculate_image_saco_with_details(attributions: np.ndarray, confidence_impacts: np.ndarray,
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
            pair_info = {'patch_i': patch_i, 'patch_j': patch_j, 'is_faithful': is_faithful, 'weight': weight}
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
            patch_metrics = calculate_patch_specific_metrics(patch_id, pairs_with_i, pairs_with_j, image_name)

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


def calculate_patch_specific_metrics(
    patch_id: int, pairs_with_i: pd.DataFrame, pairs_with_j: pd.DataFrame, image_name: str
) -> Dict[str, Any]:
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
        'image_name': image_name,
        'patch_id': patch_id,
        'faithful_pairs_count': total_faithful,
        'unfaithful_pairs_count': total_pairs - total_faithful,
        'faithful_pairs_pct': (total_faithful / total_pairs * 100) if total_pairs else 0,
        'patch_saco': patch_saco
    }


# Correlation Analysis Functions
def analyze_faithfulness_vs_correctness_from_objects(  # Renamed for clarity
    saco_scores: Dict[
        str, float],  # Key is string representation of original image path
    original_classification_results: List[ClassificationResult]
) -> pd.DataFrame:
    """
    Analyze the relationship between attribution faithfulness (SaCo) and prediction correctness.
    
    Args:
        saco_scores: Dictionary mapping string original image paths to SaCo scores.
        original_classification_results: List of ClassificationResult objects for original images.
        
    Returns:
        DataFrame with SaCo scores, correctness, confidence, and paths for further analysis.
    """
    analysis_data_list = []

    for original_res in original_classification_results:
        image_path_str = str(original_res.image_path)
        saco_score = saco_scores.get(image_path_str)

        # Use true_label from ClassificationResult if available, otherwise fall back to extraction
        true_class_label = original_res.true_label if original_res.true_label else extract_true_class_from_filename(original_res.image_path)

        prediction_info = original_res.prediction
        attribution_paths_info = original_res.attribution_paths  # This is an AttributionOutputPaths object or None

        row_data = {
            'filename':
            image_path_str,  # Keep 'filename' for consistency with old DF structure
            'saco_score':
            saco_score,
            'predicted_class':
            prediction_info.predicted_class_label,
            'predicted_idx':
            prediction_info.predicted_class_idx,  # Good to have
            'true_class':
            true_class_label,
            'is_correct':
            prediction_info.predicted_class_label == true_class_label,
            'confidence':
            prediction_info.confidence,
            # These come from the AttributionOutputPaths dataclass associated with ClassificationResult
            'attribution_path':
            str(attribution_paths_info.attribution_path) if attribution_paths_info else None,
            'logits':
            str(attribution_paths_info.logits) if attribution_paths_info and attribution_paths_info.logits else None,
            'ffn_activity_path':
            str(attribution_paths_info.ffn_activity_path)
            if attribution_paths_info and attribution_paths_info.ffn_activity_path else None,
            'class_embedding_path':
            str(attribution_paths_info.class_embedding_path)
            if attribution_paths_info and attribution_paths_info.class_embedding_path else None,
            'probabilities':
            prediction_info.probabilities
        }
        analysis_data_list.append(row_data)

    return pd.DataFrame(analysis_data_list)


def extract_true_class_from_filename(filename: Union[str, Path]) -> Optional[str]:
    """
    Extract the true class from a filename based on directory structure or filename patterns.
    Handles both new directory structure (lung/Test/COVID-19/images/) and old flat structure,
    as well as unified format (class_0, class_1, class_2).
    """
    filepath_str = str(filename).lower()
    
    # Check for unified format first (class_0, class_1, class_2)
    if '/class_0/' in filepath_str or '\\class_0\\' in filepath_str:
        return CLASSES[0]  # COVID-19
    elif '/class_1/' in filepath_str or '\\class_1\\' in filepath_str:
        return CLASSES[1]  # Non-COVID
    elif '/class_2/' in filepath_str or '\\class_2\\' in filepath_str:
        return CLASSES[2]  # Normal
    
    # First, try to extract from directory structure (preferred method)
    for cls in CLASSES:
        cls_lower = cls.lower()
        # Look for class name in the directory path (e.g., "/covid-19/images/")
        if f"/{cls_lower}/" in filepath_str or f"\\{cls_lower}\\" in filepath_str:
            return cls
    
    # Fallback to filename-based extraction for backward compatibility
    for cls in CLASSES:
        # Check exact match first
        if cls in filepath_str:
            return cls
        
        # Check with underscore instead of hyphen for flexibility
        cls_underscore = cls.replace('-', '_').lower()
        if cls_underscore in filepath_str:
            return cls
            
        # Check with hyphen instead of underscore
        cls_hyphen = cls.replace('_', '-').lower()
        if cls_hyphen in filepath_str:
            return cls

    return None


def analyze_key_attribution_patterns(df: pd.DataFrame, model, config) -> pd.DataFrame:
    """
    Comprehensive analysis of attribution patterns with correlation calculations.
    
    Args:
        df: DataFrame with SaCo scores and attribution paths
        
    Returns:
        DataFrame with added attribution metrics and correlation analysis
    """
    # df = add_basic_attribution_metrics(df)
    # df = add_entropy_metrics(df)
    # df = add_concentration_metrics(df)
    # df = add_information_theory_metrics(df)
    # df = add_gradient_based_metrics(df)
    # df = add_sparsity_metrics(df)
    # df = add_attribution_consistency_metrics(df)
    # df = add_robustness_metrics(df)
    # df = add_ffn_activity_metrics(df)
    # if config.classify.data_collection:
    # df = add_class_embedding_metrics(df)
    # df = add_embedding_space_metrics(df, model)

    # Clean data
    df_clean = df.dropna(subset=['saco_score'])

    # Define key metrics to analyze
    key_metrics = get_key_attribution_metrics()

    # Add class column
    df_clean = df_clean[df_clean['is_correct']]
    if 'filename' in df_clean.columns:
        df_clean['class'] = df_clean['true_class']

    print(df_clean.head())

    # Calculate correlations
    if len(df_clean) > 0:
        calculate_overall_correlations(df_clean, key_metrics)
        calculate_per_class_correlations(df_clean, key_metrics)

        #TODO: this is not entirely correct
        # calculate_class_correlations_by_prediction(df_clean)
        # analyze_layer_wise_correlations(df_clean)

    return df_clean


def get_key_attribution_metrics() -> List[str]:
    """
    Get the list of key attribution metrics for correlation analysis.
    
    Returns:
        List of metric names
    """

    # Define layer and class ranges based on your data
    layers = range(0, 12)  # Adjust the range as needed
    classes = range(0, 3)  # Based on your classes 0, 1, 2 mentioned in the metrics

    # Generate column names for all embedding metrics
    embedding_columns = []
    embedding_column_endings = [
        'mean_attn_logit',
        'mean_mlp_logit',
        'cls_token_attn_logit',
        'cls_token_mlp_logit',
        'var_attn_logit',
        'var_mlp_logit',
        'mean_attn_alignment_change',
        'cls_attn_alignment_change',
    ]

    # Class token MLP probabilities and logits
    for layer_idx in layers:
        for cls in classes:
            for ending in embedding_column_endings:
                embedding_columns.extend([
                    f'layer_{layer_idx}_class_{cls}_{ending}',
                ])

    # Add per-head metrics for each class
    num_heads = 12  # Adjust based on your model
    head_stats = ['mean_attn_logit', 'cls_token_attn_logit']

    # Per-head and per-class metrics
    head_class_columns = [
        f'layer_{l}_head_{h}_class_{c}_{stat}' for l in layers for h in range(num_heads) for c in classes
        for stat in head_stats
    ]

    # embedding_columns.extend(head_class_columns)

    base_metrics = [
        'mlp_mean_dist_to_pred_class',
        'attn_mean_dist_to_pred_class',
    ]

    # Add class-specific distance metrics
    for c in classes:
        base_metrics.extend([f'mlp_mean_dist_to_class_{c}', f'attn_mean_dist_to_class_{c}'])

    # Generate all column names with layer prefixes
    geometry_metrics = [f'layer_{l}_{metric}' for l in layers for metric in base_metrics]

    return [
        'neg_pos_ratio',
        'neg_entropy',
        'pos_entropy',
        'entropy_ratio',
        'neg_gini',
        'pos_gini',
        # 'neg_top10_conc',
        # 'pos_top10_conc',
        # 'mutual_information',
        # 'neg_pos_contingency',
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
        # 'pos_kurtosis',
        # 'neg_kurtosis',
        # Attribution consistency metrics
        'pos_neg_coherence',
        'attribution_evenness',
        'feature_consensus',
        # Robustness metrics
        # 'smoothing_stability',
        # 'noise_stability',
        # 'peak_persistence',
        # FFN activity metrics
        'ffn_mean_activity',
        'ffn_cls_activity',
        'ffn_last_layer_activity',
        'ffn_max_layer_activity',
        'ffn_early_layers_activity',
        'ffn_middle_layers_activity',
        'ffn_late_layers_activity',
        'ffn_layer_activity_variance',
        'ffn_token_activity_variance',
        # Class embedding distance metrics
        'mlp_mean_dist_to_pred_class',
        'mlp_mean_decision_margin',
        'mlp_embedding_dist_variance',
        'mlp_mean_dist_to_class_0',
        'mlp_mean_dist_to_class_1',
        'mlp_mean_dist_to_class_2',
        'attn_mean_dist_to_pred_class',
        'attn_mean_decision_margin',
        'attn_embedding_dist_variance',
        'attn_mean_dist_to_class_0',
        'attn_mean_dist_to_class_1',
        'attn_mean_dist_to_class_2',
        # Add all embedding columns
        *embedding_columns,
        # *geometry_metrics
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


def calculate_overall_correlations(df: pd.DataFrame, metrics: List[str]) -> None:
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
            p_value = calculate_correlation_significance(df['saco_score'].values, df[metric].values)
            correlations_overall[metric] = (corr, p_value)
            print(f"{metric}: r={corr:.3f}, p={p_value:.5f}")


def calculate_per_class_correlations(
    df: pd.DataFrame, metrics: List[str], percentile_low: float = 0.4, percentile_high: float = 0.6
) -> None:
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

            if len(class_df) > 5:  # Only calculate if we have enough data points
                print("\n" + "-" * 60)
                print(f"CORRELATIONS FOR CLASS: {cls}")
                print("-" * 60)

                for metric in metrics:
                    if metric in class_df.columns:
                        corr = class_df['saco_score'].corr(class_df[metric])
                        p_value = calculate_correlation_significance(
                            class_df['saco_score'].values, class_df[metric].values
                        )
                        print(f"{metric}: r={corr:.3f}, p={p_value:.5f}")

            print(f"\nSummary for {cls}:")
            print(f"  Total samples: {len(class_df)}")
            print(f"  Mean SaCo score: {class_df['saco_score'].mean():.3f}")
            print(f"  Median SaCo score: {class_df['saco_score'].median():.3f}")
            print(f"  Std SaCo score: {class_df['saco_score'].std():.3f}")


def add_class_embedding_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add class embedding metrics for correlation analysis across all layers.
    
    Args:
        df: DataFrame with class embedding paths
        
    Returns:
        DataFrame with added class embedding metrics for all layers
    """
    if 'class_embedding_path' not in df.columns:
        return df

    # STATIC LAYERS/CLASSES/STATS - assumed constant across all rows
    layers = range(0, 12)  # Process all layers 0-11
    classes = range(0, 3)  # 0-2
    stats = [
        'mean_attn_logit',
        'mean_mlp_logit',
        'cls_token_attn_logit',
        'cls_token_mlp_logit',
        'var_attn_logit',
        'var_mlp_logit',
        'mean_attn_alignment_change',
        'cls_attn_alignment_change',
    ]

    # Generate all column names for layer-level metrics
    new_columns = [f'layer_{l}_class_{c}_{stat}' for l in layers for c in classes for stat in stats]

    # Add per-head metrics for each class
    num_heads = 12  # Adjust based on your model

    # Generate column names for head-specific metrics for each class
    head_class_columns = []
    for l in layers:
        for h in range(num_heads):
            for c in classes:
                # Mean attention logit per class for each head
                head_class_columns.append(f'layer_{l}_head_{h}_class_{c}_mean_attn_logit')
                # CLS token attention logit per class for each head
                head_class_columns.append(f'layer_{l}_head_{h}_class_{c}_cls_token_attn_logit')

    # Combine all columns
    new_columns.extend(head_class_columns)

    # Pre-allocate results dictionary with NaN values
    results_dict = {col: np.full(len(df), np.nan) for col in new_columns}

    # Process each row
    for idx, row in df.iterrows():
        try:
            embeddings = np.load(row['class_embedding_path'], allow_pickle=True)

            # Process each layer
            for layer_idx in layers:
                if layer_idx >= len(embeddings):
                    continue

                layer_data = embeddings[layer_idx]
                attn_logits_input = layer_data['attention_class_representation_input']
                attn_logits = layer_data['attention_class_representation_output']
                mlp_logits_input = layer_data['mlp_class_representation_input']
                mlp_logits = layer_data['mlp_class_representation_output']

                attention_map = layer_data['attention_map']

                # Process layer-level metrics for each class
                for cls in classes:
                    # Calculate all metrics at once
                    mean_attn = np.mean(attn_logits[:, cls])
                    cls_token_attn = attn_logits[0, cls]
                    var_attn = np.var(attn_logits[:, cls])

                    mean_mlp = np.mean(mlp_logits[:, cls])
                    cls_token_mlp = mlp_logits[0, cls]
                    var_mlp = np.var(mlp_logits[:, cls])

                    alignment_change_attn_cls = mlp_logits[0, cls] - mlp_logits_input[0, cls]
                    alignment_change_attn = attn_logits[:, cls] - attn_logits_input[:, cls]

                    # Store in results dictionary for layer-level metrics
                    results_dict[f'layer_{layer_idx}_class_{cls}_mean_attn_logit'][idx] = mean_attn
                    results_dict[f'layer_{layer_idx}_class_{cls}_mean_attn_alignment_change'][idx] = np.mean(
                        alignment_change_attn
                    )
                    results_dict[f'layer_{layer_idx}_class_{cls}_cls_attn_alignment_change'][
                        idx] = alignment_change_attn_cls
                    results_dict[f'layer_{layer_idx}_class_{cls}_cls_token_attn_logit'][idx] = cls_token_attn
                    results_dict[f'layer_{layer_idx}_class_{cls}_var_attn_logit'][idx] = var_attn

                    results_dict[f'layer_{layer_idx}_class_{cls}_mean_mlp_logit'][idx] = mean_mlp
                    results_dict[f'layer_{layer_idx}_class_{cls}_cls_token_mlp_logit'][idx] = cls_token_mlp
                    results_dict[f'layer_{layer_idx}_class_{cls}_var_mlp_logit'][idx] = var_mlp

                # Check if attention maps are available
                if 'attention_map' in layer_data and len(layer_data['attention_map']) > 0 and False:
                    attention_map = layer_data['attention_map']

                    # Attention map should be [heads, seq, seq]
                    num_heads = attention_map.shape[0]
                    actual_heads = min(num_heads, 12)  # Cap at expected head count

                    for h in range(actual_heads):
                        attention_head = attention_map[h, :, :]  # [seq, seq]

                        # For each class, calculate the head's contribution to class representation
                        for cls in classes:
                            # Following the paper's approach of analyzing how attention transforms
                            # representations in the class embedding space

                            # 1. Calculate the change in class representation due to this head
                            # The attention output is a weighted sum of input values based on attention weights
                            head_class_contributions = np.zeros(attn_logits_input.shape[0])

                            for i in range(attn_logits_input.shape[0]):
                                # weighted sum of input class logits based on attention weights
                                weighted_inputs = attention_head[i, :] * attn_logits_input[:, cls]
                                head_class_contributions[i] = np.sum(weighted_inputs)

                            # 2. Store the mean attention logit per class for this head
                            results_dict[f'layer_{layer_idx}_head_{h}_class_{cls}_mean_attn_logit'][idx] = np.mean(
                                head_class_contributions
                            )

                            # 3. Store the CLS token attention logit per class for this head
                            results_dict[f'layer_{layer_idx}_head_{h}_class_{cls}_cls_token_attn_logit'][
                                idx] = head_class_contributions[0]

        except Exception as e:
            print(f"Error processing class embeddings for {row.get('filename', 'unknown')}: {e}")
            # NaN values are already pre-allocated for error cases

    # Create a DataFrame from results_dict and join with original df
    results_df = pd.DataFrame(results_dict, index=df.index)
    return pd.concat([df, results_df], axis=1)


def track_cross_class_information_flow(blk, model, target_class=0):
    """Track how information flows from other classes to build COVID representations"""

    # Get attention patterns
    attention_map = blk.attn.get_attention_map()  # [batch, heads, seq, seq]

    # Get class identifiability before and after
    input_tokens = blk.attn.input_tokens
    output_tokens = blk.attn.output_tokens

    input_logits = model.get_class_embedding_space_representation(input_tokens)
    output_logits = model.get_class_embedding_space_representation(output_tokens)

    # Identify source tokens (high class 1 or 2)
    class1_sources = (input_logits[:, :, 1] > input_logits[:, :, 1].mean()).float()
    class2_sources = (input_logits[:, :, 2] > input_logits[:, :, 2].mean()).float()

    # Identify tokens that GAINED COVID identifiability
    covid_gain = output_logits[:, :, 0] - input_logits[:, :, 0]
    covid_receivers = (covid_gain > 0).float()

    # Track attention flow from sources to receivers
    attn_from_class1 = torch.einsum('bhij,bi->bhj', attention_map, class1_sources).mean(1)
    attn_from_class2 = torch.einsum('bhij,bi->bhj', attention_map, class2_sources).mean(1)

    # Weight by COVID gain
    class1_contribution = attn_from_class1 * covid_gain
    class2_contribution = attn_from_class2 * covid_gain

    # Important tokens are those that:
    # 1. Receive info from class 1 (pathological) and gain COVID
    # 2. Don't receive much from class 2 (healthy) and gain COVID
    importance = class1_contribution - 0.5 * class2_contribution

    return importance


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
            neg_attr = np.load(row['logits'])  # Already absolute values

            # Print diagnostics for first few files
            if len(metrics['neg_magnitude']) < 3:
                print(f"File: {row['filename']}")
                print(f"Positive attr shape: {pos_attr.shape}, min: {pos_attr.min():.5f}, max: {pos_attr.max():.5f}")
                print(f"Negative attr shape: {neg_attr.shape}, min: {neg_attr.min():.5f}, max: {neg_attr.max():.5f}")

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
            neg_attr = np.load(row['logits_path'])

            # Normalize distributions for entropy calculation
            pos_norm = pos_attr / (np.sum(pos_attr) + 1e-10)
            neg_norm = neg_attr / (np.sum(neg_attr) + 1e-10)

            # Calculate Shannon entropy (using only non-zero values)
            pos_entropy = -np.sum(pos_norm[pos_norm > 0] * np.log2(pos_norm[pos_norm > 0] + 1e-10))
            neg_entropy = -np.sum(neg_norm[neg_norm > 0] * np.log2(neg_norm[neg_norm > 0] + 1e-10))

            metrics['neg_entropy'].append(neg_entropy)
            metrics['pos_entropy'].append(pos_entropy)
            metrics['entropy_ratio'].append(neg_entropy / (pos_entropy + 1e-10))

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
            neg_attr = np.load(row['logits_path']).flatten()

            # Calculate Gini coefficients
            metrics['neg_gini'].append(calculate_gini(neg_attr))
            metrics['pos_gini'].append(calculate_gini(pos_attr))

            # Top 10% concentration
            neg_threshold = np.percentile(neg_attr, 90)
            pos_threshold = np.percentile(pos_attr, 90)
            metrics['neg_top10_conc'].append(np.sum(neg_attr[neg_attr >= neg_threshold]) / (np.sum(neg_attr) + 1e-10))
            metrics['pos_top10_conc'].append(np.sum(pos_attr[pos_attr >= pos_threshold]) / (np.sum(pos_attr) + 1e-10))

        except Exception as e:
            print(f"Error calculating concentration for {row['filename']}: {e}")
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
    return (np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_x)) / (n * np.sum(sorted_x) + 1e-10)


def add_information_theory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add information theory metrics for attribution maps.
    
    Args:
        df: DataFrame with attribution paths
        
    Returns:
        DataFrame with added information theory metrics
    """
    metrics = {
        'mutual_information': [],  # Mutual information between positive and negative attributions
        'neg_pos_contingency': []  # Contingency between negative and positive attributions
    }

    for _, row in df.iterrows():
        try:
            pos_attr = np.load(row['attribution_path'])
            neg_attr = np.load(row['logits_path'])

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
            contingency = (
                np.sum(np.logical_and(high_pos, high_neg)) + np.sum(np.logical_and(~high_pos, ~high_neg))
            ) / pos_attr.size
            metrics['neg_pos_contingency'].append(contingency)

        except Exception as e:
            print(f"Error calculating information metrics for {row['filename']}: {e}")
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
            neg_attr = np.load(row['logits_path'])

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
            print(f"Error processing gradients for {row.get('filename', 'unknown')}: {e}")
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
        'pos_effective_sparsity': [],  # Effective number of non-zero positive elements
        'neg_effective_sparsity': [],  # Effective number of non-zero negative elements
        'pos_kurtosis': [],  # Measure of "peakedness" of positive distribution
        'neg_kurtosis': []  # Measure of "peakedness" of negative distribution
    }

    for _, row in df.iterrows():
        try:
            pos_attr = np.load(row['attribution_path'])
            neg_attr = np.load(row['logits_path'])

            flat_pos = pos_attr.flatten()
            flat_neg = neg_attr.flatten()

            # L0 sparsity (percentage of near-zero elements)
            pos_threshold = 0.01 * np.max(pos_attr)
            neg_threshold = 0.01 * np.max(neg_attr)
            metrics['pos_l0_sparsity'].append(np.mean(pos_attr < pos_threshold))
            metrics['neg_l0_sparsity'].append(np.mean(neg_attr < neg_threshold))

            # Effective sparsity for positive attributions
            pos_normalized = flat_pos / (np.sum(flat_pos) + 1e-10)
            pos_effective_nonzero = 1.0 / (np.sum(pos_normalized**2) + 1e-10)
            metrics['pos_effective_sparsity'].append(1.0 - (pos_effective_nonzero / len(flat_pos)))

            # Effective sparsity for negative attributions
            neg_normalized = flat_neg / (np.sum(flat_neg) + 1e-10)
            neg_effective_nonzero = 1.0 / (np.sum(neg_normalized**2) + 1e-10)
            metrics['neg_effective_sparsity'].append(1.0 - (neg_effective_nonzero / len(flat_neg)))

            # Kurtosis (measure of "peakedness")
            metrics['pos_kurtosis'].append(stats.kurtosis(flat_pos))
            metrics['neg_kurtosis'].append(stats.kurtosis(flat_neg))

        except Exception as e:
            print(f"Error calculating sparsity for {row.get('filename', 'unknown')}: {e}")
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
        'pos_neg_coherence': [],  # Correlation between positive and negative attribution patterns
        'attribution_evenness': [],  # How evenly distributed the attribution intensity is across regions
        'feature_consensus': []  # Agreement between different attention heads
    }

    for _, row in df.iterrows():
        try:
            pos_attr = np.load(row['attribution_path'])
            neg_attr = np.load(row['logits_path'])

            # Correlation between positive and negative maps
            flat_pos = pos_attr.flatten()
            flat_neg = neg_attr.flatten()
            pos_neg_corr = np.corrcoef(flat_pos, flat_neg)[0, 1]
            metrics['pos_neg_coherence'].append(abs(pos_neg_corr))  # Use absolute value to measure alignment

            # Measure attribution evenness (ratio of mean to max)
            combined_attr = pos_attr + neg_attr
            evenness = np.mean(combined_attr) / (np.max(combined_attr) + 1e-10)
            metrics['attribution_evenness'].append(evenness)

            # If head-specific attributions are available, measure agreement between heads
            if 'attribution_heads_path' in row and pd.notnull(row['attribution_heads_path']):
                heads_attr = np.load(row['attribution_heads_path'])
                feature_consensus = calculate_feature_consensus(heads_attr)
                metrics['feature_consensus'].append(feature_consensus)
            else:
                metrics['feature_consensus'].append(np.nan)

        except Exception as e:
            print(f"Error calculating consistency for {row.get('filename', 'unknown')}: {e}")
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
            corr = np.corrcoef(heads_attr[i].flatten(), heads_attr[j].flatten())[0, 1]
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
        'peak_persistence': []  # How persistent the highest attribution regions are
    }

    for _, row in df.iterrows():
        try:
            pos_attr = np.load(row['attribution_path'])

            # Smoothing stability
            smoothed = gaussian_filter(pos_attr, sigma=1.0)
            stability = np.corrcoef(pos_attr.flatten(), smoothed.flatten())[0, 1]
            metrics['smoothing_stability'].append(stability)

            # Noise stability (adding small random noise)
            noise_level = 0.05 * np.std(pos_attr)
            noise = np.random.normal(0, noise_level, pos_attr.shape)
            noisy = pos_attr + noise
            noise_stability = np.corrcoef(pos_attr.flatten(), noisy.flatten())[0, 1]
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
            print(f"Error calculating robustness for {row.get('filename', 'unknown')}: {e}")
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
        'ffn_layer_activity_variance': [],  # Variance of activity across layers
        'ffn_token_activity_variance': []  # Mean variance of activity across tokens
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

            cls_activities = [layer['cls_activity'] for layer in ffn_data if 'cls_activity' in layer]
            metrics['ffn_cls_activity'].append(np.mean(cls_activities) if cls_activities else np.nan)

            metrics['ffn_last_layer_activity'].append(ffn_data[-1]['mean_activity'])
            metrics['ffn_max_layer_activity'].append(np.max(layer_means))

            # Activity by layer groups (early, middle, late)
            num_layers = len(ffn_data)
            third = max(1, num_layers // 3)
            metrics['ffn_early_layers_activity'].append(np.mean(layer_means[:third]))
            metrics['ffn_middle_layers_activity'].append(np.mean(layer_means[third:2 * third]))
            metrics['ffn_late_layers_activity'].append(np.mean(layer_means[2 * third:]))

            metrics['ffn_layer_activity_variance'].append(np.var(layer_means))

            # Mean variance across tokens
            token_variances = calculate_token_variances(ffn_data)
            metrics['ffn_token_activity_variance'].append(np.mean(token_variances) if token_variances else np.nan)

        except Exception as e:
            print(f"Error processing FFN activity for {row.get('filename', 'unknown')}: {e}")
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
        if 'activity' not in ffn_data[i] or not isinstance(ffn_data[i]['activity'], np.ndarray):
            continue

        token_activity = ffn_data[i]['activity']
        if token_activity.ndim > 0 and token_activity.shape[0] > 1:
            token_variances.append(np.var(token_activity[1:]))  # Exclude CLS token

    return token_variances


def add_embedding_space_metrics(df: pd.DataFrame, model: VisionTransformer) -> pd.DataFrame:
    """
    Add distance-based metrics from the class embedding space for all layers.
    
    Args:
        df: DataFrame with class embedding paths
        model: The trained Vision Transformer model
        
    Returns:
        DataFrame with added embedding space metrics for all layers
    """
    # Extract class prototypes from the model's classification head
    class_prototypes = model.head.weight.detach().cpu().numpy()  # Shape: (num_classes, embed_dim)

    # First, analyze the prototype geometry
    print_prototype_geometry(class_prototypes)

    class_to_idx = IDX2CLS
    num_classes = len(class_to_idx)

    layers = range(0, 12)

    # Define base metrics
    base_metrics = [
        'mlp_mean_dist_to_pred_class', 'mlp_mean_decision_margin', 'mlp_embedding_dist_variance',
        'attn_mean_dist_to_pred_class', 'attn_mean_decision_margin', 'attn_embedding_dist_variance'
    ]

    # Add class-specific distance metrics
    for c in range(num_classes):
        base_metrics.extend([f'mlp_mean_dist_to_class_{c}', f'attn_mean_dist_to_class_{c}'])

    # Generate all column names with layer prefixes
    result_cols = [f'layer_{l}_{metric}' for l in layers for metric in base_metrics]

    # Pre-allocate result columns with NaN values
    results_dict = {col: np.full(len(df), np.nan) for col in result_cols}

    # Process each row
    for idx, row in df.iterrows():
        try:
            embeddings = np.load(row['class_embedding_path'], allow_pickle=True)

            # Get predicted class index
            pred_class = class_to_idx[row['predicted_class']]

            # Process each layer
            for layer_idx in layers:
                if layer_idx >= len(embeddings):
                    continue

                layer_data = embeddings[layer_idx]

                # Get logits from both MLP and attention
                attn_logits = layer_data['attention_class_representation_output']
                mlp_logits = layer_data['mlp_class_representation_output']

                # Process both MLP and Attention representations
                for rep_type, logits in [('mlp', mlp_logits), ('attn', attn_logits)]:
                    # For each token, calculate distances (negative logits as proxy for distances)
                    token_distances = -logits  # Higher logit = closer

                    # Mean distance to predicted class across all tokens
                    mean_dist_to_pred = np.mean(token_distances[:, pred_class])

                    # Decision margins for each token (distance between closest and 2nd closest class)
                    sorted_dists = np.sort(token_distances, axis=1)
                    decision_margins = sorted_dists[:, 1] - sorted_dists[:, 0]  # Gap between closest and 2nd closest
                    mean_decision_margin = np.mean(decision_margins)

                    # Distance variance (spread of tokens from prototypes)
                    dist_variance = np.var(token_distances)

                    # Store computed metrics in results_dict with layer prefix
                    results_dict[f'layer_{layer_idx}_{rep_type}_mean_dist_to_pred_class'][idx] = mean_dist_to_pred
                    results_dict[f'layer_{layer_idx}_{rep_type}_mean_decision_margin'][idx] = mean_decision_margin
                    results_dict[f'layer_{layer_idx}_{rep_type}_embedding_dist_variance'][idx] = dist_variance

                    # Store distances to each class
                    for c in range(num_classes):
                        results_dict[f'layer_{layer_idx}_{rep_type}_mean_dist_to_class_{c}'][idx] = np.mean(
                            token_distances[:, c]
                        )

        except Exception as e:
            print(f"Error processing {row.get('filename', 'unknown')}: {e}")
            # NaN values are already pre-allocated for error cases

    # Create a DataFrame from results_dict and join with original df
    results_df = pd.DataFrame(results_dict, index=df.index)
    return pd.concat([df, results_df], axis=1)


def print_prototype_geometry(class_prototypes):
    """Print the geometric relationships between class prototypes."""
    from scipy.spatial.distance import pdist, squareform

    prototype_distances = squareform(pdist(class_prototypes, metric='euclidean'))
    prototype_cosine_sim = squareform(pdist(class_prototypes, metric='cosine'))

    print("\n" + "=" * 60)
    print("CLASS PROTOTYPE GEOMETRY:")
    print("=" * 60)

    class_names = CLS2IDX
    for i in range(7):
        for j in range(i + 1, 7):
            print(f"Distance {class_names[i]} - {class_names[j]}: {prototype_distances[i,j]:.3f}")
            print(f"Cosine similarity {class_names[i]} - {class_names[j]}: {1-prototype_cosine_sim[i,j]:.3f}")
