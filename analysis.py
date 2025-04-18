import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

import perturbation
import transformer as trans

OUTPUT_DIR = "./results"


def compare_attributions(original_results_df: pd.DataFrame,
                         perturbed_results_df: pd.DataFrame,
                         output_dir: str = "./results") -> pd.DataFrame:
    """
    Compare attributions between original and perturbed images.
    
    Args:
        original_results_df: DataFrame with original classification results
        perturbed_results_df: DataFrame with perturbed classification results (patch-perturbed)
        
    Returns:
        DataFrame with attribution comparison results
    """
    output_path = Path(output_dir)
    comparison_dir = output_path / "comparisons"
    patch_mask_dir = output_path / "patch_masks"
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
            # print(f"No matching original image found for {perturbed_filename}")
            continue

        original_row = original_results_df.loc[matching_indices[0]]

        # Get mask path TODO: fix this part to be generally applicable
        mask_path = Path(
            f"./results/patch_masks/{perturbed_filename}_mask.npy")

        if not mask_path.exists():
            print(f"Mask file not found: {mask_path}")
            continue

        try:
            # Load attributions
            original_attribution = np.load(original_row["attribution_path"])
            perturbed_attribution = np.load(perturbed_row["attribution_path"])
            np_mask = np.load(mask_path)

            # Generate comparison visualization
            comparison_path = comparison_dir / f"{perturbed_filename}_comparison.png"
            diff_stats = trans.explain_attribution_diff(
                original_attribution,
                perturbed_attribution,
                np_mask,
                base_name=perturbed_filename,
                save_dir=str(comparison_dir))

            # Calculate SSIM between the actual 224x224 ViT inputs
            ssim_score = None
            if "attribution_vis_path" in original_row and "attribution_vis_path" in perturbed_row:
                original_vit_img = Image.open(
                    original_row["attribution_vis_path"]).convert('RGB')
                perturbed_vit_img = Image.open(
                    perturbed_row["attribution_vis_path"]).convert('RGB')
                ssim_score = perturbation.patch_similarity(
                    original_vit_img, perturbed_vit_img)

            # Extract patch coordinates from filename
            x, y = -1, -1
            patch_id = -1

            # Try to extract patch info from filename
            filename_parts = perturbed_filename.split('_')
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

            # Calculate mean attribution in the patch if coordinates are found
            mean_attribution = None
            if x >= 0 and y >= 0:
                patch_size = 16  # Default patch size
                patch_end_x = min(x + patch_size,
                                  original_attribution.shape[1])
                patch_end_y = min(y + patch_size,
                                  original_attribution.shape[0])

                if x < original_attribution.shape[
                        1] and y < original_attribution.shape[0]:
                    patch_attribution = original_attribution[y:patch_end_y,
                                                             x:patch_end_x]
                    mean_attribution = np.mean(patch_attribution)

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
                ssim_score,
                "comparison_path":
                str(comparison_path)
            }

            # Add key metrics from diff_stats
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
        if "confidence_delta_abs" in comparison_df.columns:
            comparison_df["impact_rank"] = comparison_df[
                "confidence_delta_abs"].rank(ascending=False)

        if "mean_attribution" in comparison_df.columns:
            comparison_df["attribution_rank"] = comparison_df[
                "mean_attribution"].rank(ascending=False)

        if "impact_rank" in comparison_df.columns and "attribution_rank" in comparison_df.columns:
            comparison_df["rank_difference"] = comparison_df[
                "attribution_rank"] - comparison_df["impact_rank"]

        comparison_df.to_csv(output_path / "patch_attribution_comparisons.csv",
                             index=False)
    else:
        print("Warning: No comparison results were generated.")

    return comparison_df


def calculate_saco_with_details(
        data_path: str = "./results/patch_attribution_comparisons.csv",
        method: str = "mean"):
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
        # print(f"Calculated SaCo for {image_name}: {saco_score:.4f}")

    print(f"Average SaCo score: {sum(list(results.values())) / len(results)}")
    return results, pair_data


def calculate_image_saco_with_details(attributions, confidence_impacts,
                                      patch_ids):
    """Calculate SaCo and return detailed pair-wise comparison data"""
    F = 0
    total_weight = 0
    pairs_data = []

    for i in range(len(attributions) - 1):
        for j in range(i + 1, len(attributions)):
            attr_diff = attributions[i] - attributions[j]
            impact_i, impact_j = confidence_impacts[i], confidence_impacts[j]
            patch_i, patch_j = patch_ids[i], patch_ids[j]

            # Calculate weight for SaCo
            weight = attr_diff if impact_i >= impact_j else -attr_diff
            F += weight
            total_weight += abs(weight)

            # Store pair data
            pair_info = {
                'patch_i': patch_i,
                'patch_j': patch_j,
                'is_faithful': impact_i >= impact_j,
                'weight': weight
            }
            pairs_data.append(pair_info)

    F /= total_weight
    return F, pd.DataFrame(pairs_data)


def analyze_patch_metrics(pair_data):
    """Analyze the pair-wise comparison data and return a flat DataFrame with patch-specific metrics"""
    rows = []

    for image_name, image_pairs in pair_data.items():
        unique_patches = set(image_pairs['patch_i'].tolist() +
                             image_pairs['patch_j'].tolist())

        for patch_id in unique_patches:
            pairs_with_i = image_pairs[image_pairs['patch_i'] == patch_id]
            pairs_with_j = image_pairs[image_pairs['patch_j'] == patch_id]

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
            patch_saco = sum(all_weights) / sum(
                abs(w) for w in all_weights) if all_weights else 0

            # Create a row for this patch
            row = {
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

            rows.append(row)

    return pd.DataFrame(rows)


def analyze_faithfulness_vs_correctness(
        saco_scores,
        classification_results="./results/classification_results.csv"):
    """
    Analyze the relationship between attribution faithfulness and prediction correctness.
    Returns a DataFrame with SaCo scores, correctness, and confidence information.
    """
    df = pd.read_csv(classification_results)
    results = []

    for _, row in df.iterrows():
        filename = row['image_path']
        saco_score = saco_scores.get(filename)
        if saco_score is None:
            continue

        # Get true class from filename
        if filename.startswith("images/Normal"):
            true_class = "Normal"
        elif filename.startswith("images/covid"):
            true_class = "COVID-19"
        elif filename.startswith("images/non_COVID"):
            true_class = "Non-COVID"
        else:
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
            'ffn_activity_path': row['ffn_activity_path']
        })

    return pd.DataFrame(results)


def analyze_attribution_patterns(df):
    """
    Calculate basic magnitude metrics for attribution maps without correlation analysis.
    """
    # Add metrics to understand attribution patterns
    metrics = {
        'neg_magnitude':
        [],  # Total magnitude of negative attributions (already absolute)
        'pos_magnitude': [],  # Total magnitude of positive attributions
        'neg_max': [],  # Maximum negative attribution value
        'pos_max': [],  # Maximum positive attribution value
        'neg_pos_ratio': []  # Ratio of negative to positive magnitudes
    }

    # Process each file
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

    # Clean dataframe
    df_clean = df.dropna(subset=['neg_magnitude', 'pos_magnitude'])

    # Print diagnostics
    print(f"\nProcessed {len(df_clean)} files successfully")
    print(
        f"Negative magnitude range: {df_clean['neg_magnitude'].min():.5f} to {df_clean['neg_magnitude'].max():.5f}"
    )
    print(
        f"Positive magnitude range: {df_clean['pos_magnitude'].min():.5f} to {df_clean['pos_magnitude'].max():.5f}"
    )
    print(
        f"Neg/Pos ratio range: {df_clean['neg_pos_ratio'].min():.5f} to {df_clean['neg_pos_ratio'].max():.5f}"
    )

    # Just create a basic scatter plot without correlation analysis
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(data=df_clean,
                              x='saco_score',
                              y='neg_pos_ratio',
                              hue='is_correct',
                              size='confidence',
                              sizes=(20, 200),
                              alpha=0.7)
    plt.title('SaCo Score vs. Negative/Positive Attribution Ratio')
    plt.xlabel('SaCo Score')
    plt.ylabel('Negative/Positive Attribution Ratio')
    plt.tight_layout()
    plt.savefig('saco_vs_neg_pos_ratio.png')

    # Return the dataframe with the added metrics
    return df


def analyze_key_attribution_patterns(df):
    """
    Analysis focusing on key metrics for understanding negative attribution patterns,
    with correlations calculated overall and per class
    """
    # First run basic analysis (without correlation calculation)
    df = analyze_attribution_patterns(df)

    # Add our prioritized new metrics
    df = add_entropy_metrics(df)
    df = add_concentration_metrics(df)
    df = add_information_theory_metrics(df)
    df = gradient_based_metrics(df)
    df = sparsity_metrics(df)
    df = attribution_consistency_metrics(df)
    df = robustness_metrics(df)
    df = add_ffn_activity_metrics(df)
    # Clean data
    df_clean = df.dropna(
        subset=['saco_score', 'neg_magnitude', 'pos_magnitude'])

    # Define key metrics to analyze
    key_metrics = [
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
        # Add FFN activity metrics if they exist in the dataframe
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

    # Extract class from image name (assuming 'image_name' column exists)
    def get_class(image_name):
        class_prefixes = ["images/Normal", "images/covid", "images/non_COVID"]
        for prefix in class_prefixes:
            if str(image_name).startswith(prefix):
                return prefix
        return "unknown"

    # Add class column if image_name exists
    df_clean = df_clean[df_clean['is_correct']]

    if 'filename' in df_clean.columns:
        df_clean['class'] = df_clean['filename'].apply(get_class)
    df_clean['class'] = df_clean['true_class']
    # Calculate overall correlations with SaCo score
    if len(df_clean) > 0:
        print("\n" + "=" * 60)
        print("OVERALL CORRELATIONS WITH SACO SCORE:")
        print("=" * 60)

        correlations_overall = {}
        for metric in key_metrics:
            if metric in df_clean.columns:
                corr = df_clean['saco_score'].corr(df_clean[metric])
                p_value = calculate_correlation_significance(
                    df_clean['saco_score'], df_clean[metric])
                correlations_overall[metric] = (corr, p_value)
                print(f"{metric}: r={corr:.3f}, p={p_value:.5f}")

        # Calculate per-class correlations if we have class information
        if 'class' in df_clean.columns:
            classes = df_clean['class'].unique()

            for cls in classes:
                class_df = df_clean[df_clean['class'] == cls]

                if len(class_df
                       ) > 5:  # Only calculate if we have enough data points
                    print("\n" + "-" * 60)
                    print(f"CORRELATIONS FOR CLASS: {cls}")
                    print("-" * 60)

                    for metric in key_metrics:
                        if metric in class_df.columns:
                            corr = class_df['saco_score'].corr(
                                class_df[metric])
                            p_value = calculate_correlation_significance(
                                class_df['saco_score'], class_df[metric])
                            print(f"{metric}: r={corr:.3f}, p={p_value:.5f}")
    compare_concentration_distributions(df_clean)

    return df_clean


def calculate_correlation_significance(x, y):
    """Calculate p-value for a correlation"""
    from scipy import stats
    corr, p_value = stats.pearsonr(x, y)
    return p_value


def add_entropy_metrics(df):
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


def add_concentration_metrics(df):
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

            # Calculate Gini coefficient (measure of inequality)
            def gini(x):
                # Sort values
                sorted_x = np.sort(x)
                n = len(x)
                cumsum = np.cumsum(sorted_x)
                # Return Gini coefficient
                return (np.sum((2 * np.arange(1, n + 1) - n - 1) *
                               sorted_x)) / (n * np.sum(sorted_x))

            metrics['neg_gini'].append(gini(neg_attr))
            metrics['pos_gini'].append(gini(pos_attr))

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


def add_information_theory_metrics(df):
    metrics = {
        'mutual_information':
        [],  # Mutual information between positive and negative attributions
        'neg_pos_contingency':
        []  # Contingency between negative and positive attributions
    }

    from sklearn.metrics import mutual_info_score

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


def compare_concentration_distributions(df):
    """Compare the distribution of concentration metrics across classes"""
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['neg_gini', 'pos_gini', 'neg_entropy', 'pos_entropy']

    # Calculate statistics
    stats = {}
    for cls in df['class'].unique():
        stats[cls] = {}
        for metric in metrics:
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
        print(f"\n{metric}")
        for cls in stats:
            pct = stats[cls][metric]['percentiles']
            print(f"{cls}:")
            print(f"  Mean: {stats[cls][metric]['mean']:.3f}")
            print(f"  Std Dev: {stats[cls][metric]['std']:.3f}")
            print(f"  Range (10th-90th): {pct[4]-pct[0]:.3f}")
            print(f"  Distribution: 10%={pct[0]:.3f}, 25%={pct[1]:.3f}, " +
                  f"50%={pct[2]:.3f}, 75%={pct[3]:.3f}, 90%={pct[4]:.3f}")

    # Create the plots
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        for cls in df['class'].unique():
            sns.kdeplot(df[df['class'] == cls][metric], ax=ax, label=cls)

        ax.set_title(f'Distribution of {metric}')
        ax.legend()

    plt.tight_layout()
    plt.savefig('concentration_distributions.png')
    return stats


def gradient_based_metrics(df):
    """Analyze properties of the gradients used in attribution"""
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
                f"Error processing gradients for {row['filename'] if 'filename' in row else 'unknown'}: {e}"
            )
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def sparsity_metrics(df):
    """Calculate various sparsity measures for attribution maps"""
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
            from scipy import stats
            metrics['pos_kurtosis'].append(stats.kurtosis(flat_pos))
            metrics['neg_kurtosis'].append(stats.kurtosis(flat_neg))

        except Exception as e:
            print(
                f"Error calculating sparsity for {row['filename'] if 'filename' in row else 'unknown'}: {e}"
            )
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def attribution_consistency_metrics(df):
    """Measure consistency between different aspects of the attribution"""
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

                # Calculate pairwise correlations between heads
                num_heads = heads_attr.shape[0]
                correlations = []
                for i in range(num_heads):
                    for j in range(i + 1, num_heads):
                        corr = np.corrcoef(heads_attr[i].flatten(),
                                           heads_attr[j].flatten())[0, 1]
                        correlations.append(corr)

                metrics['feature_consensus'].append(np.mean(correlations))
            else:
                metrics['feature_consensus'].append(np.nan)

        except Exception as e:
            print(f"Error calculating consistency for {row['filename']}: {e}")
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def robustness_metrics(df):
    """Calculate robustness of attribution to small changes"""
    from scipy.ndimage import gaussian_filter

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
            print(f"Error calculating robustness for {row['filename']}: {e}")
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df


def add_ffn_activity_metrics(df):
    """
    Add FFN activity metrics to the dataframe for correlation analysis.
    """
    # Initialize new columns for FFN activity metrics
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
        try:
            # Create filename based on image name - adjust pattern as needed
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
            third = num_layers // 3
            metrics['ffn_early_layers_activity'].append(
                np.mean(layer_means[:third]))
            metrics['ffn_middle_layers_activity'].append(
                np.mean(layer_means[third:2 * third]))
            metrics['ffn_late_layers_activity'].append(
                np.mean(layer_means[2 * third:]))

            metrics['ffn_layer_activity_variance'].append(np.var(layer_means))

            # Mean variance across tokens
            token_variances = []
            for i in range(len(ffn_data)):
                if 'activity' not in ffn_data[i] or not isinstance(
                        ffn_data[i]['activity'], np.ndarray):
                    continue
                token_activity = ffn_data[i]['activity']
                if token_activity.ndim > 0 and token_activity.shape[0] > 1:
                    token_variances.append(np.var(
                        token_activity[1:]))  # Exclude CLS token

            metrics['ffn_token_activity_variance'].append(
                np.mean(token_variances) if token_variances else np.nan)

        except Exception as e:
            print(f"Error processing FFN activity for {row['filename']}: {e}")
            for key in metrics:
                metrics[key].append(np.nan)

    # Add metrics to dataframe
    for key, values in metrics.items():
        df[key] = values

    return df
