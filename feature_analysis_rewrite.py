import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as scipy_stats


def load_faithfulness_results(path: Path) -> pd.DataFrame:
    saco_csv_file = list(path.glob("analysis_faithfulness_correctness_*.csv"))
    if not saco_csv_file:
        raise FileNotFoundError(f"No SaCo faithfulness CSV found in {experiment_path}")

    df = pd.read_csv(saco_csv_file[0])
    faithfulness_json = list(path.glob("faithfulness_stats_*.json"))
    if not faithfulness_json:
        raise FileNotFoundError(f"No faithfulness CSV found in {experiment_path}")

    with open(faithfulness_json[0], 'r') as f:
        faithfulness_stats = json.load(f)
    metrics = faithfulness_stats.get('metrics', {})

    for metric_name, metric_data in metrics.items():
        df[metric_name] = metric_data['mean_scores']

    # Extract actual image index from filename for proper attribution/image lookups
    # For ImageNet: "data/imagenet_unified/test/class_-1/img_-01_test_083810.jpeg" -> 83810
    # For CovidQuex: sequential indexing is fine
    if 'filename' in df.columns:
        # Try to extract image number from filename
        df['image_idx'] = df['filename'].str.extract(r'_(\d+)\.(?:jpeg|png)$')[0].astype(int)
    else:
        df['image_idx'] = range(len(df))

    return df


def load_debug_data(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    debug_dir = path / "debug_data"
    if not debug_dir.exists():
        raise FileNotFoundError(f"Debug data directory not found: {debug_dir}")

    debug_files = list(debug_dir.glob("layer_*_debug.npz"))
    if not debug_files:
        raise FileNotFoundError(f"No debug NPZ files found in {debug_dir}")

    debug_data = {}
    for debug_file in sorted(debug_files):
        layer_idx = int(debug_file.stem.split('_')[1])
        data = np.load(debug_file, allow_pickle=True)

        debug_data[layer_idx] = {
            'sparse_indices': data['sparse_indices'],
            'sparse_activations': data['sparse_activations'],
            'sparse_gradients': data['sparse_gradients'],
            'sparse_contributions': data.get('sparse_contributions', None),  # New field
            'gate_values': data['gate_values'],
            'patch_attribution_deltas': data.get('patch_attribution_deltas', None),  # May not exist in old caches
            'contribution_sum': data.get('contribution_sum', None),  # New field
            'total_contribution_magnitude': data.get('total_contribution_magnitude', None)  # New field
        }

        n_images = len(debug_data[layer_idx]['sparse_indices'])
        print(f"  Layer {layer_idx}: {n_images} images")

    return debug_data


def load_combined_results(path: Path) -> Dict[int, pd.DataFrame]:
    merged_dir = path / "merged_data"
    merged_data = {}
    for parquet_file in sorted(merged_dir.glob("layer_*.parquet")):
        layer_idx = int(parquet_file.stem.split('_')[1])
        merged_data[layer_idx] = pd.read_parquet(parquet_file)
        print(f"  Loaded merged_data for layer {layer_idx}: {len(merged_data[layer_idx])} rows")

    return merged_data


def combine_vanilla_gated(vanilla_faithfulness, gated_faithfulness, gated_debug) -> Dict[int, pd.DataFrame]:
    """
    Merge feature debug data with faithfulness results.

    Args:
        features_debug: Debug data per layer
        faithfulness_df: Faithfulness results with image_idx

    Returns:
        Dictionary mapping layer_idx -> DataFrame with columns:
        - image_idx
        - saco_score, predicted_class, true_class, is_correct
        - patch_idx (196 rows per image)
        - gate_value
        - active_features (list of feature indices for this patch)
        - feature_activations (list of activation values)
        - feature_gradients (list of gradient values)
    """
    merged_data = {}
    gated_records = gated_faithfulness.to_dict('records')
    vanilla_records = vanilla_faithfulness.to_dict('records')

    for layer_idx, layer_data in gated_debug.items():
        rows = []
        n_images = len(layer_data['sparse_indices'])
        for img_idx in range(n_images):
            if img_idx >= len(gated_records):
                print(f"Warning: image_idx {img_idx} not in faithfulness data, skipping")
                continue

            faith_row = gated_records[img_idx]
            vanilla_row = vanilla_records[img_idx]

            img_sparse_indices = layer_data['sparse_indices'][img_idx]
            img_sparse_activations = layer_data['sparse_activations'][img_idx]
            img_sparse_gradients = layer_data['sparse_gradients'][img_idx]
            img_gate_values = layer_data['gate_values'][img_idx]
            img_sparse_contributions = layer_data['sparse_contributions'][img_idx]
            img_contribution_sum = layer_data['contribution_sum'][img_idx]
            img_total_contribution_magnitude = layer_data['total_contribution_magnitude'][img_idx]
            img_attribution_deltas = layer_data['patch_attribution_deltas'][img_idx]

            for patch_idx in range(len(img_gate_values)):
                row = {
                    'image_idx':
                    faith_row['image_idx'],  # Use actual image index from faithfulness data, not loop counter
                    'patch_idx':
                    patch_idx,
                    'saco_score':
                    faith_row['saco_score'],
                    'delta_saco':
                    faith_row['saco_score'] - vanilla_row['saco_score'],
                    'predicted_class':
                    faith_row['predicted_class'],
                    'true_class':
                    faith_row['true_class'],
                    'is_correct':
                    faith_row['is_correct'],
                    'FaithfulnessCorrelation':
                    faith_row.get('FaithfulnessCorrelation', np.nan),
                    'delta_faith':
                    faith_row['FaithfulnessCorrelation'] - vanilla_row['FaithfulnessCorrelation'],
                    'PixelFlipping':
                    faith_row.get('PixelFlipping', np.nan),
                    'delta_pixel':
                    faith_row['PixelFlipping'] - vanilla_row['PixelFlipping'],
                    'gate_value':
                    img_gate_values[patch_idx],
                    'attribution_delta':
                    img_attribution_deltas[patch_idx] if img_attribution_deltas is not None else np.nan,
                    'contribution_sum':
                    img_contribution_sum[patch_idx] if img_contribution_sum is not None else np.nan,
                    'total_contribution_magnitude':
                    img_total_contribution_magnitude[patch_idx]
                    if img_total_contribution_magnitude is not None else np.nan,
                    'active_features':
                    img_sparse_indices[patch_idx],
                    'feature_activations':
                    img_sparse_activations[patch_idx],
                    'feature_gradients':
                    img_sparse_gradients[patch_idx],
                    'feature_contributions':
                    img_sparse_contributions[patch_idx] if img_sparse_contributions is not None else [],
                    'n_active_features':
                    len(img_sparse_indices[patch_idx])
                }
                rows.append(row)

        merged_df = pd.DataFrame(rows)
        merged_data[layer_idx] = merged_df

    return merged_data


def analyze_feature_composition_of_patches(
    data: Dict[int, pd.DataFrame],
    patch_impact_results: Dict[str, Any],
    layer_idx: int,
    min_occurrences: int = 10,
    top_n: int = 20
) -> pd.DataFrame:
    df = data[layer_idx]
    metrics = ["saco", "faith", "pixel"]
    patch_keys = {}
    patch_counts = {}

    for metric in metrics:
        high_patches = patch_impact_results[f"high_improvement_patches_{metric}"]
        low_patches = patch_impact_results[f"low_improvement_patches_{metric}"]

        patch_keys[f'high_{metric}'] = set((r['image_idx'], r['patch_idx']) for _, r in high_patches.iterrows())
        patch_keys[f'low_{metric}'] = set((r['image_idx'], r['patch_idx']) for _, r in low_patches.iterrows())
        patch_counts[f'high_{metric}'] = len(patch_keys[f'high_{metric}'])
        patch_counts[f'low_{metric}'] = len(patch_keys[f'low_{metric}'])

    for metric in metrics:
        print(f"  {metric}: {patch_counts[f'high_{metric}']} high, {patch_counts[f'low_{metric}']} low")

    feature_stats = defaultdict(
        lambda: {
            metric: {
                'high': {
                    'count': 0,
                    'activations': [],
                    'contributions': [],
                    'gates': []
                },
                'low': {
                    'count': 0,
                    'activations': [],
                    'contributions': [],
                    'gates': []
                }
            }
            for metric in metrics
        }
    )

    for row in df.itertuples(index=False):
        patch_key = (row.image_idx, row.patch_idx)

        group_membership = {
            metric: (
                'high' if patch_key in patch_keys[f'high_{metric}'] else
                'low' if patch_key in patch_keys[f'low_{metric}'] else None
            )
            for metric in metrics
        }

        if all(v is None for v in group_membership.values()):
            continue

        active_features = row.active_features
        feature_acts = row.feature_activations
        feature_contribs = getattr(row, 'feature_contributions', [])
        gate_value = row.gate_value

        for i, feat_idx in enumerate(active_features):
            for metric in metrics:
                group = group_membership[metric]
                if group is None:
                    continue

                stats = feature_stats[feat_idx][metric][group]
                stats['count'] += 1
                stats['activations'].append(feature_acts[i])
                stats['gates'].append(gate_value)
                if len(feature_contribs) > i:
                    stats['contributions'].append(feature_contribs[i])

    feature_results = []
    min_count_per_group = max(3, min_occurrences // 10)
    for feat_idx, feat_stat in feature_stats.items():
        total_counts = [feat_stat[m]['high']['count'] + feat_stat[m]['low']['count'] for m in metrics]
        if max(total_counts) < min_occurrences: continue

        result_row = {'feature_idx': feat_idx}
        enrichments = []

        for metric in metrics:
            high_stats = feat_stat[metric]['high']
            low_stats = feat_stat[metric]['low']

            count_h = high_stats['count']
            count_l = low_stats['count']
            total_h = patch_counts[f'high_{metric}']
            total_l = patch_counts[f'low_{metric}']
            freq_h = count_h / total_h if total_h > 0 else 0
            freq_l = count_l / total_l if total_l > 0 else 0
            enrichment = freq_h / freq_l if freq_l > 0 else (float('inf') if freq_h > 0 else 1.0)

            # TODO: There is one thing to consider, if a feature appears in lots and lots of patches but only in a few selected images this makes it not super realiable
            # This breaks the assumption of independence (we can show the average amount of occurences in an image or something?)
            # Fisher's exact test
            p_val = 1.0
            if count_h >= min_count_per_group and count_l >= min_count_per_group:
                contingency = [[count_h, total_h - count_h], [count_l, total_l - count_l]]
                _, p_val = scipy_stats.fisher_exact(contingency)

            # Store results
            result_row[f'count_high_{metric}'] = count_h
            result_row[f'count_low_{metric}'] = count_l
            result_row[f'enrichment_{metric}'] = enrichment
            result_row[f'p_value_{metric}'] = p_val
            result_row[f'mean_activation_high_{metric}'] = np.mean(high_stats['activations']
                                                                   ) if high_stats['activations'] else 0
            result_row[f'mean_activation_low_{metric}'] = np.mean(low_stats['activations']
                                                                  ) if low_stats['activations'] else 0
            result_row[f'mean_contribution_high_{metric}'] = np.mean(high_stats['contributions']
                                                                     ) if high_stats['contributions'] else 0
            result_row[f'mean_contribution_low_{metric}'] = np.mean(low_stats['contributions']
                                                                    ) if low_stats['contributions'] else 0

            enrichments.append(enrichment)

        # Combined enrichment
        result_row['enrichment_combined'] = np.mean(enrichments)
        feature_results.append(result_row)

    results_df = pd.DataFrame(feature_results)
    results_df = results_df.sort_values('enrichment_combined', ascending=False)

    print(f"Found {len(results_df)} features with sufficient occurrences\n")
    print(f"Top {top_n} ENRICHED features:")
    for _, row in results_df.head(top_n).iterrows():
        feat_idx = int(row['feature_idx'])
        combined_enrichment = row['enrichment_combined']

        print(f"Feature {feat_idx}: enrichment={combined_enrichment:.2f}x")
        print(
            f"  Per-metric: SaCo={row['enrichment_saco']:.2f}x, "
            f"Faith={row['enrichment_faith']:.2f}x, Pixel={row['enrichment_pixel']:.2f}x"
        )
        print(
            f"  Counts: SaCo ({int(row['count_high_saco'])}/{int(row['count_low_saco'])}), "
            f"Faith ({int(row['count_high_faith'])}/{int(row['count_low_faith'])}), "
            f"Pixel ({int(row['count_high_pixel'])}/{int(row['count_low_pixel'])})"
        )
        print(
            f"  Mean activation (high/low): SaCo ({row['mean_activation_high_saco']:.3f}/{row['mean_activation_low_saco']:.3f}), "
            f"Faith ({row['mean_activation_high_faith']:.3f}/{row['mean_activation_low_faith']:.3f}), "
            f"Pixel ({row['mean_activation_high_pixel']:.3f}/{row['mean_activation_low_pixel']:.3f})"
        )
        print(
            f"  Mean contribution (high/low): SaCo ({row['mean_contribution_high_saco']:.2e}/{row['mean_contribution_low_saco']:.2e}), "
            f"Faith ({row['mean_contribution_high_faith']:.2e}/{row['mean_contribution_low_faith']:.2e}), "
            f"Pixel ({row['mean_contribution_high_pixel']:.2e}/{row['mean_contribution_low_pixel']:.2e})"
        )

    return results_df


def analyze_feature_distributions(
    data: Dict[int, pd.DataFrame],
    layer_idx: int,
    feature_composition_results: pd.DataFrame,
    output_dir: Path,
    top_n_enriched: int = 100
) -> Dict[str, Any]:
    """
    Analyze feature and contribution distributions to justify enrichment approach.

    Generates three key analyses:
    1. Feature ubiquity: How many images does each feature appear in?
    2. Contribution distributions: Do enriched features contribute more?
    3. Selectivity vs contribution: Is there a relationship?

    Args:
        data: Merged data from combine_vanilla_gated
        layer_idx: Which layer to analyze
        feature_composition_results: Results from enrichment analysis
        output_dir: Where to save plots
        top_n_enriched: Number of top enriched features to compare

    Returns:
        Dictionary with summary statistics
    """
    import matplotlib.pyplot as plt

    df = data[layer_idx]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")

    # ========== Analysis 1: Feature Ubiquity ==========
    print("\n1. Feature Ubiquity Distribution")

    feature_image_counts = {}
    feature_contributions = defaultdict(list)

    for row in df.itertuples(index=False):
        image_idx = row.image_idx
        active_features = row.active_features
        feature_contribs = getattr(row, 'feature_contributions', [])

        for i, feat_idx in enumerate(active_features):
            if feat_idx not in feature_image_counts:
                feature_image_counts[feat_idx] = set()
            feature_image_counts[feat_idx].add(image_idx)

            if i < len(feature_contribs):
                feature_contributions[feat_idx].append(abs(feature_contribs[i]))

    # Convert to counts
    n_images_per_feature = {feat: len(imgs) for feat, imgs in feature_image_counts.items()}
    ubiquity_values = list(n_images_per_feature.values())
    total_images = df['image_idx'].nunique()

    # Statistics
    ubiquity_stats = {
        'n_features': len(ubiquity_values),
        'mean_images': np.mean(ubiquity_values),
        'median_images': np.median(ubiquity_values),
        'min_images': np.min(ubiquity_values),
        'max_images': np.max(ubiquity_values),
        'total_images': total_images,
        'pct_ubiquitous': sum(1 for v in ubiquity_values if v > 0.9 * total_images) / len(ubiquity_values) * 100
    }

    print(f"  Total features: {ubiquity_stats['n_features']}")
    print(f"  Mean images per feature: {ubiquity_stats['mean_images']:.1f} / {total_images}")
    print(f"  Median: {ubiquity_stats['median_images']:.1f}")
    print(f"  Range: [{ubiquity_stats['min_images']}, {ubiquity_stats['max_images']}]")
    print(f"  Ubiquitous features (>90% images): {ubiquity_stats['pct_ubiquitous']:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ubiquity_values, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(total_images * 0.9, color='red', linestyle='--', label='90% of images')
    ax.set_xlabel('Number of Images Feature Appears In')
    ax.set_ylabel('Number of Features')
    ax.set_title(f'Layer {layer_idx}: Feature Ubiquity Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'layer_{layer_idx}_feature_ubiquity.png', dpi=150)
    plt.close()

    # ========== Analysis 2: Contribution Distribution ==========
    print("\n2. Contribution Magnitude Distribution")

    # Get mean contribution per feature
    mean_contributions = {
        feat: np.mean(contribs) if contribs else 0
        for feat, contribs in feature_contributions.items()
    }

    # Split into enriched vs non-enriched
    enriched_features = set(feature_composition_results.head(top_n_enriched)['feature_idx'].astype(int))

    enriched_contribs = [mean_contributions.get(f, 0) for f in enriched_features if f in mean_contributions]
    non_enriched_features = [f for f in mean_contributions.keys() if f not in enriched_features]
    non_enriched_contribs = [mean_contributions[f] for f in non_enriched_features]

    # Statistics
    contrib_stats = {
        'enriched_mean': np.mean(enriched_contribs),
        'enriched_median': np.median(enriched_contribs),
        'non_enriched_mean': np.mean(non_enriched_contribs),
        'non_enriched_median': np.median(non_enriched_contribs),
        'enriched_std': np.std(enriched_contribs),
        'non_enriched_std': np.std(non_enriched_contribs)
    }

    print(f"  Enriched features (top {top_n_enriched}):")
    print(f"    Mean |contribution|: {contrib_stats['enriched_mean']:.2e}")
    print(f"    Median |contribution|: {contrib_stats['enriched_median']:.2e}")
    print(f"  Non-enriched features:")
    print(f"    Mean |contribution|: {contrib_stats['non_enriched_mean']:.2e}")
    print(f"    Median |contribution|: {contrib_stats['non_enriched_median']:.2e}")
    print(f"  Ratio (enriched/non-enriched): {contrib_stats['enriched_mean']/contrib_stats['non_enriched_mean']:.2f}x")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(enriched_contribs, bins=50, alpha=0.5, label=f'Enriched (top {top_n_enriched})', edgecolor='blue')
    ax.hist(non_enriched_contribs, bins=50, alpha=0.5, label='Non-enriched', edgecolor='gray')
    ax.set_xlabel('Mean |Contribution|')
    ax.set_ylabel('Number of Features')
    ax.set_title(f'Layer {layer_idx}: Contribution Distribution (Enriched vs Non-enriched)')
    ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'layer_{layer_idx}_contribution_distribution.png', dpi=150)
    plt.close()

    # ========== Analysis 3: Selectivity vs Contribution ==========
    print("\n3. Selectivity vs Contribution Relationship")

    # Prepare data for scatter
    scatter_data = []
    for feat_idx, n_images in n_images_per_feature.items():
        if feat_idx not in mean_contributions:
            continue

        # Get enrichment if available
        enrichment_row = feature_composition_results[feature_composition_results['feature_idx'] == feat_idx]
        enrichment = enrichment_row['enrichment_combined'].iloc[0] if len(enrichment_row) > 0 else 0

        scatter_data.append({
            'feature_idx': feat_idx,
            'n_images': n_images,
            'selectivity': total_images - n_images,  # Fewer images = more selective
            'mean_contribution': mean_contributions[feat_idx],
            'enrichment': enrichment
        })

    scatter_df = pd.DataFrame(scatter_data)

    # Correlation
    from scipy.stats import pearsonr, spearmanr
    corr_pearson, p_pearson = pearsonr(scatter_df['selectivity'], scatter_df['mean_contribution'])
    corr_spearman, p_spearman = spearmanr(scatter_df['selectivity'], scatter_df['mean_contribution'])

    print(f"  Pearson correlation (selectivity vs contribution): r={corr_pearson:.3f}, p={p_pearson:.2e}")
    print(f"  Spearman correlation: ρ={corr_spearman:.3f}, p={p_spearman:.2e}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        scatter_df['n_images'],
        scatter_df['mean_contribution'],
        c=scatter_df['enrichment'],
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Enrichment Score')
    ax.set_xlabel('Number of Images Feature Appears In')
    ax.set_ylabel('Mean |Contribution|')
    ax.set_title(f'Layer {layer_idx}: Selectivity vs Contribution (colored by enrichment)')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / f'layer_{layer_idx}_selectivity_vs_contribution.png', dpi=150)
    plt.close()

    # ========== Analysis 4: Faithfulness Improvement Distribution ==========
    print("\n4. Faithfulness Improvement Distribution")

    # Get unique images and their deltas
    image_deltas = df.groupby('image_idx').first()[['delta_saco', 'delta_faith', 'delta_pixel']].reset_index()

    improvement_stats = {
        'saco_mean': image_deltas['delta_saco'].mean(),
        'saco_std': image_deltas['delta_saco'].std(),
        'faith_mean': image_deltas['delta_faith'].mean(),
        'faith_std': image_deltas['delta_faith'].std(),
        'pixel_mean': image_deltas['delta_pixel'].mean(),
        'pixel_std': image_deltas['delta_pixel'].std(),
    }

    print(f"  ΔSaCo: μ={improvement_stats['saco_mean']:.4f}, σ={improvement_stats['saco_std']:.4f}")
    print(f"  ΔFaith: μ={improvement_stats['faith_mean']:.4f}, σ={improvement_stats['faith_std']:.4f}")
    print(f"  ΔPixel: μ={improvement_stats['pixel_mean']:.4f}, σ={improvement_stats['pixel_std']:.4f}")

    # Plot side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(image_deltas['delta_saco'], bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    axes[0].set_xlabel('ΔSaCo Score')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('SaCo Improvement Distribution')
    axes[0].legend()

    axes[1].hist(image_deltas['delta_faith'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    axes[1].set_xlabel('ΔFaithfulness Correlation')
    axes[1].set_ylabel('Number of Images')
    axes[1].set_title('FaithCorr Improvement Distribution')
    axes[1].legend()

    axes[2].hist(image_deltas['delta_pixel'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[2].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    axes[2].set_xlabel('ΔPixel Flipping (lower is better)')
    axes[2].set_ylabel('Number of Images')
    axes[2].set_title('Pixel Flipping Improvement Distribution')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'layer_{layer_idx}_improvement_distribution.png', dpi=150)
    plt.close()

    # ========== Analysis 5: Enrichment Concept Visualization ==========
    print("\n5. Enrichment Concept Visualization")

    # Get top 15 enriched features
    top_features = feature_composition_results.head(15)

    # Get quartile thresholds for stratification
    q_high = image_deltas['delta_saco'].quantile(0.75)
    q_low = image_deltas['delta_saco'].quantile(0.25)

    # Count feature occurrences in high vs low quartile
    enrichment_viz_data = []

    for _, feat_row in top_features.iterrows():
        feat_idx = int(feat_row['feature_idx'])

        # Count patches where this feature appears
        count_high = 0
        count_low = 0

        for row in df.itertuples(index=False):
            delta_saco = row.delta_saco
            active_features = row.active_features

            if feat_idx in active_features:
                if delta_saco >= q_high:
                    count_high += 1
                elif delta_saco <= q_low:
                    count_low += 1

        enrichment_viz_data.append({
            'feature_idx': feat_idx,
            'high_quartile': count_high,
            'low_quartile': count_low,
            'enrichment': feat_row['enrichment_combined']
        })

    enrichment_viz_df = pd.DataFrame(enrichment_viz_data)

    print(f"  Showing top 15 features by enrichment")
    print(f"  High quartile: ΔSaCo >= {q_high:.4f}")
    print(f"  Low quartile: ΔSaCo <= {q_low:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(enrichment_viz_df))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        enrichment_viz_df['high_quartile'],
        width,
        label='High Improvement (Top 25%)',
        color='green',
        alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2,
        enrichment_viz_df['low_quartile'],
        width,
        label='Low Improvement (Bottom 25%)',
        color='red',
        alpha=0.8
    )

    ax.set_xlabel('Feature ID')
    ax.set_ylabel('Number of Patches')
    ax.set_title(f'Layer {layer_idx}: Enrichment Concept - Feature Presence in High vs Low Improvement Images')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f)}" for f in enrichment_viz_df['feature_idx']], rotation=45, ha='right')
    ax.legend()

    # Add enrichment values as text on top
    for i, (_, row) in enumerate(enrichment_viz_df.iterrows()):
        ax.text(
            i,
            max(row['high_quartile'], row['low_quartile']) + 50,
            f"{row['enrichment']:.1f}x",
            ha='center',
            fontsize=8,
            color='blue'
        )

    plt.tight_layout()
    plt.savefig(output_dir / f'layer_{layer_idx}_enrichment_concept.png', dpi=150)
    plt.close()

    # ========== Analysis 6: Enrichment Ratio Distribution ==========
    print("\n6. Enrichment Ratio Distribution")

    # Get enrichment ratios for all features
    enrichment_ratios = feature_composition_results['enrichment_combined'].values

    # Calculate statistics
    enrichment_stats = {
        'median': np.median(enrichment_ratios),
        'mean': np.mean(enrichment_ratios),
        'std': np.std(enrichment_ratios),
        'q25': np.percentile(enrichment_ratios, 25),
        'q75': np.percentile(enrichment_ratios, 75),
        'q90': np.percentile(enrichment_ratios, 90),
        'q95': np.percentile(enrichment_ratios, 95),
        'max': np.max(enrichment_ratios),
        'n_above_2': np.sum(enrichment_ratios > 2.0),
        'n_above_1_5': np.sum(enrichment_ratios > 1.5),
    }

    print(f"  Total features: {len(enrichment_ratios)}")
    print(f"  Enrichment statistics:")
    print(f"    Median: {enrichment_stats['median']:.2f}")
    print(f"    Mean: {enrichment_stats['mean']:.2f} ± {enrichment_stats['std']:.2f}")
    print(f"    Q25/Q75: {enrichment_stats['q25']:.2f} / {enrichment_stats['q75']:.2f}")
    print(f"    Q90/Q95: {enrichment_stats['q90']:.2f} / {enrichment_stats['q95']:.2f}")
    print(f"    Max: {enrichment_stats['max']:.2f}")
    print(
        f"    Features with enrichment > 1.5: {enrichment_stats['n_above_1_5']} ({100*enrichment_stats['n_above_1_5']/len(enrichment_ratios):.1f}%)"
    )
    print(
        f"    Features with enrichment > 2.0: {enrichment_stats['n_above_2']} ({100*enrichment_stats['n_above_2']/len(enrichment_ratios):.1f}%)"
    )

    # Plot histogram
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use bins that capture the distribution well
    # Most features around 1.0, but want to see the tail
    bins = np.linspace(enrichment_ratios.min(), min(enrichment_ratios.max(), 10), 50)

    counts, bin_edges, patches = ax.hist(enrichment_ratios, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')

    # Highlight the tail (enrichment > 2.0)
    for patch, left_edge, right_edge in zip(patches, bin_edges[:-1], bin_edges[1:]):
        if left_edge >= 2.0:
            patch.set_facecolor('orange')
            patch.set_alpha(0.8)

    # Add vertical lines for thresholds
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=2, label='Null (enrichment = 1.0)', alpha=0.7)
    ax.axvline(2.0, color='red', linestyle='--', linewidth=2, label='High enrichment (2.0x)', alpha=0.7)
    ax.axvline(
        enrichment_stats['median'],
        color='green',
        linestyle='-',
        linewidth=2,
        label=f'Median ({enrichment_stats["median"]:.2f})',
        alpha=0.7
    )

    ax.set_xlabel('Enrichment Ratio (High/Low Frequency)')
    ax.set_ylabel('Number of Features')
    ax.set_title(f'Layer {layer_idx}: Distribution of Feature Enrichment Ratios')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add text box with key statistics
    textstr = f"N = {len(enrichment_ratios)}\nMedian = {enrichment_stats['median']:.2f}\n"
    textstr += f"Q95 = {enrichment_stats['q95']:.2f}\n"
    textstr += f"Enrichment > 2.0: {enrichment_stats['n_above_2']} features"
    ax.text(
        0.65,
        0.97,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_dir / f'layer_{layer_idx}_enrichment_distribution.png', dpi=150)
    plt.close()

    print(f"\nPlots saved to: {output_dir}/")

    return {
        'ubiquity_stats': ubiquity_stats,
        'contribution_stats': contrib_stats,
        'improvement_stats': improvement_stats,
        'enrichment_stats': enrichment_stats,
        'correlation_pearson': corr_pearson,
        'correlation_spearman': corr_spearman,
        'p_value_pearson': p_pearson,
        'p_value_spearman': p_spearman
    }


def save_combined_to_cache(results: Dict[int, pd.DataFrame], output_path: Path):
    """
    Save merged data and feature statistics to disk for later use.

    Args:
        results: Output from combine_vanilla_gated()
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_dir = output_path / "merged_data"
    merged_dir.mkdir(exist_ok=True)
    for layer_idx, df in results.items():
        df.to_parquet(merged_dir / f"layer_{layer_idx}.parquet")
        print(f"  Saved merged_data for layer {layer_idx}: {len(df)} rows")


def analyze_patch_impact_on_faithfulness(
    data: Dict[int, pd.DataFrame],
    layer_idx: int,
    improvement_quantiles: Tuple[float, float] = (0.75, 0.25),
    patch_quantile: float = 0.80,
) -> Dict[str, Any]:
    """
    Stage 1: Identify patch-level characteristics that correlate with faithfulness improvements.

    Uses quantile-based stratification and effect size analysis (no arbitrary thresholds).

    Args:
        merged_data: Output from merge_features_faithfulness
        faithfulness_vanilla: Vanilla faithfulness results
        faithfulness_gated: Gated faithfulness results
        layer_idx: Which layer to analyze
        improvement_quantiles: (high_threshold, low_threshold) for stratifying images
        patch_quantile: Within-image quantile for identifying high-impact patches
        min_images: Minimum images needed for each group

    Returns:
        Dictionary with:
            - 'image_stratification': DataFrame with image groups and statistics
            - 'patch_statistics': DataFrame with patch-level effect sizes
            - 'high_impact_patches': DataFrame with patches from high-improvement images
            - 'low_impact_patches': DataFrame with patches from low-improvement images
    """

    df = data[layer_idx]
    improvements = df.groupby('image_idx').first()[['delta_saco', 'delta_pixel', 'delta_faith']].reset_index()
    z_score = lambda series: (series - series.mean()) / series.std()
    improvements['delta_saco_z'] = z_score(improvements['delta_saco'])
    improvements['delta_faith_z'] = z_score(improvements['delta_faith'])
    improvements['delta_pixel_z'] = -z_score(improvements['delta_pixel'])
    improvements['composite_improvement'
                 ] = (improvements['delta_saco_z'] + improvements['delta_faith_z'] + improvements['delta_pixel_z']) / 3

    # Print distribution statistics
    print("Faithfulness Improvement Distribution:")
    for metric in ['delta_saco', 'delta_faith', 'delta_pixel', 'composite_improvement']:
        values = improvements[metric].values
        print(f"  {metric}:")
        print(f"    Mean: {values.mean():.6f}, Std: {values.std():.6f}")
        print(
            f"    Quartiles: 25%={np.percentile(values, 25):.4f}, 50%={np.percentile(values, 50):.4f}, 75%={np.percentile(values, 75):.4f}"
        )

    metrics = ['saco', 'faith', 'pixel']
    for metric in metrics:
        high_q, low_q = (
            improvement_quantiles[1], improvement_quantiles[0]
        ) if metric == "pixel" else improvement_quantiles
        high_thresh = improvements[f'delta_{metric}'].quantile(high_q)
        low_thresh = improvements[f'delta_{metric}'].quantile(low_q)

        if metric == 'pixel':
            improvements[f'group_{metric}'] = improvements[f'delta_{metric}'].apply(
                lambda x: 'high' if x <= high_thresh else ('low' if x >= low_thresh else 'medium')
            )
            direction = f"<={high_thresh:.4f}", f">={low_thresh:.4f}"
        else:
            improvements[f'group_{metric}'] = improvements[f'delta_{metric}'].apply(
                lambda x: 'high' if x >= high_thresh else ('low' if x <= low_thresh else 'medium')
            )
            direction = f">={high_thresh:.4f}", f"<={low_thresh:.4f}"

        counts = improvements[f'group_{metric}'].value_counts()
        print(
            f"  {metric.capitalize()}: High ({direction[0]}): {counts.get('high', 0)}, "
            f"Low ({direction[1]}): {counts.get('low', 0)}, Medium: {counts.get('medium', 0)}"
        )

    patch_df = df.merge(
        improvements[['image_idx', 'group_saco', 'group_faith', 'group_pixel', 'composite_improvement']],
        on='image_idx',
        how='left'
    )

    high_impact_patches = []
    for img_idx in patch_df['image_idx'].unique():
        img_patches = patch_df[patch_df['image_idx'] == img_idx].copy()
        img_patches['abs_attr_delta'] = img_patches['attribution_delta'].abs()
        threshold = img_patches['abs_attr_delta'].quantile(patch_quantile)
        high_patches = img_patches[img_patches['abs_attr_delta'] >= threshold]
        high_impact_patches.append(high_patches)

    high_impact_df = pd.concat(high_impact_patches, ignore_index=True) if high_impact_patches else pd.DataFrame()

    patch_groups = {}
    for metric in ['saco', 'faith', 'pixel']:
        high_patches = patch_df[patch_df[f'group_{metric}'] == 'high']
        low_patches = patch_df[patch_df[f'group_{metric}'] == 'low']

        # Store for Stage 2
        patch_groups[f'high_improvement_patches_{metric}'] = high_patches
        patch_groups[f'low_improvement_patches_{metric}'] = low_patches

    return {
        'image_stratification': improvements,
        'patch_statistics': patch_df,
        'high_impact_patches': high_impact_df,
        **patch_groups  # Unpack the patch groups
    }


def save_analysis_results(patch_impact_results, feature_composition_results, output_dir):
    """Save all analysis results to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for layer_idx, patch_impact in patch_impact_results.items():
        # Save patch impact results
        if 'image_stratification' in patch_impact:
            patch_impact['image_stratification'].to_csv(
                output_dir / f"layer_{layer_idx}_image_stratification.csv", index=False
            )

        # Save feature composition
        if layer_idx in feature_composition_results:
            feature_composition_results[layer_idx].to_csv(
                output_dir / f"layer_{layer_idx}_feature_composition.csv", index=False
            )

    print(f"Results saved to: {output_dir}/")


def visualize_feature_detailed(
    feature_idx: int,
    layer_idx: int,
    merged_data: Dict[int, pd.DataFrame],
    image_dir: Path,
    output_dir: Path,
    vanilla_attribution_dir: Optional[Path] = None,
    gated_attribution_dir: Optional[Path] = None,
    patch_impact_results: Optional[Dict[str, Any]] = None,
    n_examples: int = 50,
    sort_by: str = 'composite_improvement',
    dataset: str = 'covidquex'
):
    """Save visualizations for a single feature showing top activating images."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib import cm

    df = merged_data[layer_idx]
    feature_patches = []
    for _, row in df.iterrows():
        if feature_idx not in row['active_features']:
            continue

        feat_pos = list(row['active_features']).index(feature_idx)
        feature_patches.append({
            'image_idx':
            row['image_idx'],
            'patch_idx':
            row['patch_idx'],
            'activation':
            row['feature_activations'][feat_pos],
            'contribution':
            row.get('feature_contributions', [[]])[feat_pos] if 'feature_contributions' in row else None,
            'gate_value':
            row['gate_value'],
        })

    if not feature_patches:
        print(f"Feature {feature_idx} not found in layer {layer_idx}")
        return

    # Group by image
    feature_df = pd.DataFrame(feature_patches)
    image_groups = feature_df.groupby('image_idx').agg({
        'patch_idx': list,
        'activation': list,
        'contribution': list,
        'gate_value': 'mean',
    }).reset_index()

    # Merge improvement scores if available
    if patch_impact_results is not None and 'image_stratification' in patch_impact_results:
        improvement_df = patch_impact_results['image_stratification'][[
            'image_idx', 'delta_saco', 'delta_faith', 'delta_pixel', 'composite_improvement'
        ]]
        image_groups = image_groups.merge(improvement_df, on='image_idx', how='left')

    # Sort by requested metric
    if sort_by == 'composite_improvement':
        if 'composite_improvement' not in image_groups.columns:
            print(f"Warning: composite_improvement not available, falling back to activation")
            image_groups['sort_key'] = image_groups['activation'].apply(max)
        else:
            image_groups['sort_key'] = image_groups['composite_improvement']
    elif sort_by == 'contribution':
        image_groups['sort_key'] = image_groups['contribution'].apply(
            lambda x: np.mean(np.abs([c for c in x if c is not None]))
        )
    else:  # activation
        image_groups['sort_key'] = image_groups['activation'].apply(max)

    image_groups = image_groups.sort_values('sort_key', ascending=False)

    # Create output directory
    feature_dir = output_dir / f"feature_{feature_idx}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Visualize top examples
    summary_stats = []
    patch_size = 16  # 224/14 for ViT
    patches_per_side = 14

    for rank, (_, row) in enumerate(image_groups.head(n_examples).iterrows(), 1):
        img_idx = row['image_idx']

        # Get image path based on dataset
        if dataset == 'imagenet':
            image_path = get_image_path_imagenet(img_idx, image_dir)
        else:  # covidquex
            image_path = get_image_path_covidquex(img_idx, image_dir)

        if not image_path or not image_path.exists():
            continue

        # Load attribution maps based on dataset
        vanilla_attr = None
        gated_attr = None
        if vanilla_attribution_dir is not None:
            if dataset == 'imagenet':
                vanilla_attr = load_attribution_imagenet(img_idx, vanilla_attribution_dir)
            else:
                vanilla_attr = load_attribution(img_idx, vanilla_attribution_dir)
        if gated_attribution_dir is not None:
            if dataset == 'imagenet':
                gated_attr = load_attribution_imagenet(img_idx, gated_attribution_dir)
            else:
                gated_attr = load_attribution(img_idx, gated_attribution_dir)

        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        # Check if we have attributions
        has_attributions = vanilla_attr is not None and gated_attr is not None
        if not has_attributions:
            print(f"Warning: Missing attributions for image {img_idx}, skipping")
            continue

        # Create 2-panel figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Panel 1: Vanilla attribution
        axes[0].imshow(img_array)
        attr_normalized = (vanilla_attr - vanilla_attr.min()) / (vanilla_attr.max() - vanilla_attr.min() + 1e-8)
        axes[0].imshow(attr_normalized, cmap='jet', alpha=0.5, interpolation='bilinear')
        axes[0].set_title("Vanilla Attribution", fontsize=10)
        axes[0].axis('off')

        # Panel 2: Gated attribution with feature patches overlayed
        axes[1].imshow(img_array)
        attr_normalized = (gated_attr - gated_attr.min()) / (gated_attr.max() - gated_attr.min() + 1e-8)
        axes[1].imshow(attr_normalized, cmap='jet', alpha=0.5, interpolation='bilinear')

        # Overlay feature patches on top (frame only)
        n_boost = n_deboost = 0
        padding = 1  # pixels of padding to prevent overlap
        for patch_idx, activation, contrib in zip(row['patch_idx'], row['activation'], row['contribution']):
            row_idx = patch_idx // patches_per_side
            col_idx = patch_idx % patches_per_side
            x, y = col_idx * patch_size, row_idx * patch_size

            # Color based on contribution
            if contrib and contrib < 0:
                color, n_deboost = 'red', n_deboost + 1
            else:
                color, n_boost = 'lime', n_boost + 1

            # Opacity based on activation strength (0.3 to 1.0)
            alpha = 0.3 + min(activation * 0.35, 0.7)
            rect = patches.Rectangle((x + padding, y + padding),
                                     patch_size - 2 * padding,
                                     patch_size - 2 * padding,
                                     linewidth=1,
                                     edgecolor=color,
                                     facecolor='none',
                                     alpha=alpha)
            axes[1].add_patch(rect)

        axes[1].set_title(
            f"Gated Attribution + Feature {feature_idx} Patches\nBoost: {n_boost}, Deboost: {n_deboost}", fontsize=10
        )
        axes[1].axis('off')

        # Add title with improvement metrics
        title_parts = [f"Rank {rank} | Image {img_idx} | Feature {feature_idx}"]

        if 'composite_improvement' in row.index:
            title_parts.append(
                f"Composite Δ: {row['composite_improvement']:.3f} | "
                f"ΔSaCo: {row['delta_saco']:.4f} | ΔFaith: {row['delta_faith']:.4f} | ΔPixel: {row['delta_pixel']:.4f}"
            )

        fig.suptitle("\n".join(title_parts), fontsize=9)

        # Save
        plt.tight_layout()
        plt.savefig(feature_dir / f"rank_{rank:02d}_img_{img_idx}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Collect stats
        stats = {
            'rank': rank,
            'image_idx': img_idx,
            'n_patches': len(row['patch_idx']),
            'max_activation': max(row['activation']),
            'mean_activation': np.mean(row['activation']),
            'gate_value': row['gate_value'],
        }

        # Add improvement metrics if available
        if 'composite_improvement' in row.index:
            stats['composite_improvement'] = row['composite_improvement']
            stats['delta_saco'] = row['delta_saco']
            stats['delta_faith'] = row['delta_faith']
            stats['delta_pixel'] = row['delta_pixel']

        summary_stats.append(stats)

    # Save summary
    pd.DataFrame(summary_stats).to_csv(feature_dir / "summary_statistics.csv", index=False)

    print(f"Saved {len(summary_stats)} examples to {feature_dir}/")


def visualize_top_features(
    feature_composition_results,
    merged_data,
    patch_impact_results,
    image_dir,
    output_dir,
    vanilla_attribution_dir=None,
    gated_attribution_dir=None,
    n_features=30,
    n_examples=50,
    dataset='covidquex'
):
    """Visualize top enriched features for each layer."""
    output_dir = Path(output_dir)

    for layer_idx, feature_df in feature_composition_results.items():
        if feature_df.empty:
            continue

        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx} - Visualizing top {n_features} features")
        print(f"{'='*60}")

        for rank, (_, row) in enumerate(feature_df.head(n_features).iterrows(), 1):
            feat_idx = int(row['feature_idx'])
            enrichment = row['enrichment_combined']

            print(f"[{rank}/{n_features}] Feature {feat_idx} (enrichment={enrichment:.2f}x)")

            visualize_feature_detailed(
                feature_idx=feat_idx,
                layer_idx=layer_idx,
                merged_data=merged_data,
                image_dir=image_dir,
                output_dir=output_dir / f"layer_{layer_idx}",
                vanilla_attribution_dir=vanilla_attribution_dir,
                gated_attribution_dir=gated_attribution_dir,
                patch_impact_results=patch_impact_results.get(layer_idx),
                n_examples=n_examples,
                sort_by='activation',
                dataset=dataset
            )

    print(f"\nVisualization complete. Saved to: {output_dir}/")


def generate_summary_report(feature_composition_results, output_path):
    """Generate a concise summary report of enrichment analysis."""
    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        f.write("FEATURE ENRICHMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        for layer_idx, df in feature_composition_results.items():
            if df.empty:
                continue

            f.write(f"\nLAYER {layer_idx}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total features: {len(df)}\n")
            f.write(f"Mean enrichment: {df['enrichment_combined'].mean():.2f}x\n")
            f.write(f"Median enrichment: {df['enrichment_combined'].median():.2f}x\n\n")

            # Top 5 features
            f.write("Top 5 features:\n")
            for rank, (_, row) in enumerate(df.head(5).iterrows(), 1):
                f.write(
                    f"  {rank}. Feature {int(row['feature_idx'])}: "
                    f"enrichment={row['enrichment_combined']:.2f}x, "
                    f"saco_high/low={int(row['count_high_saco'])}/{int(row['count_low_saco'])}\n"
                )
            f.write("\n")

    print(f"Summary report saved to: {output_path}")


def get_image_path_covidquex(image_idx: int, image_dir: Path) -> Optional[Path]:
    """
    Map image_idx to actual file path for CovidQuex dataset.

    Dataset structure:
    - class_0/: 1903 images (idx 0-1902)
    - class_1/: 1802 images (idx 1903-3704)
    - class_2/: 1712 images (idx 3705-5416)
    """
    if image_idx < 1903:
        class_idx = 0
        local_idx = image_idx
    elif image_idx < 3705:  # 1903 + 1802
        class_idx = 1
        local_idx = image_idx - 1903
    else:
        class_idx = 2
        local_idx = image_idx - 3705

    image_path = image_dir / f"class_{class_idx}" / f"img_{class_idx:02d}_val_{local_idx:05d}.png"

    if image_path.exists():
        return image_path
    else:
        return None


def get_image_path_imagenet(image_idx: int, image_dir: Path) -> Optional[Path]:
    """
    Map image_idx to actual file path for ImageNet dataset.

    Dataset structure:
    - class_-1/: 100000 images (idx 0-99999)
    - Naming: img_-01_test_{idx:06d}.jpeg
    """
    image_path = image_dir / "class_-1" / f"img_-01_test_{image_idx:06d}.jpeg"

    if image_path.exists():
        return image_path
    else:
        return None


def load_attribution(image_idx: int, attribution_dir: Path) -> Optional[np.ndarray]:
    """
    Load full pixel-wise attribution for a given image.

    Args:
        image_idx: Image index (0-based)
        attribution_dir: Directory containing attribution files

    Returns:
        Array of shape (224, 224) with pixel-wise attributions, or None if not found
    """
    if image_idx < 1903:
        class_idx = 0
        local_idx = image_idx
        split = 'test'  # Adjust based on your needs
    elif image_idx < 3705:
        class_idx = 1
        local_idx = image_idx - 1903
        split = 'test'
    else:
        class_idx = 2
        local_idx = image_idx - 3705
        split = 'test'

    attr_path = attribution_dir / f"img_{class_idx:02d}_{split}_{local_idx:05d}_attribution.npy"

    if attr_path.exists():
        return np.load(attr_path)
    else:
        return None


def load_attribution_imagenet(image_idx: int, attribution_dir: Path) -> Optional[np.ndarray]:
    """
    Load full pixel-wise attribution for a given image (ImageNet dataset).

    Args:
        image_idx: Image index (0-based)
        attribution_dir: Directory containing attribution files

    Returns:
        Array of shape (224, 224) with pixel-wise attributions, or None if not found
    """
    attr_path = attribution_dir / f"img_-01_test_{image_idx:06d}_attribution.npy"

    if attr_path.exists():
        return np.load(attr_path)
    else:
        return None


dataset = "imagenet"
experiment_path = f"./experiments/feature_gradient_sweep_20251118_203648"
experiment_config = "layers_6_9_10_kappa_0.5_topk_None_combined_clamp_10.0"
experiment_split = "test"

vanilla_experiment_path = Path(f'{experiment_path}/{dataset}/vanilla/{experiment_split}/')
gated_experiment_path = Path(f'{experiment_path}/{dataset}/{experiment_config}/{experiment_split}')
cache_experiment_path = Path(f'{experiment_path}/analysis_cache')

if (cache_experiment_path / "merged_data").exists():
    combined_results = load_combined_results(cache_experiment_path)
else:
    vanilla_faithfulness = load_faithfulness_results(vanilla_experiment_path)
    gated_faithfulness = load_faithfulness_results(gated_experiment_path)
    gated_debug = load_debug_data(gated_experiment_path)
    combined_results = combine_vanilla_gated(vanilla_faithfulness, gated_faithfulness, gated_debug)
    save_combined_to_cache(combined_results, output_path=cache_experiment_path)

# Stage 1: Feature composition
patch_impact_results_per_layer = {}
feature_composition_results_per_layer = {}

for layer_idx in combined_results.keys():
    print(f"\n{'='*80}")
    print(f"LAYER {layer_idx}")
    print(f"{'='*80}")

    # Stage 1: Patch impact analysis
    patch_impact = analyze_patch_impact_on_faithfulness(
        combined_results,
        layer_idx=layer_idx,
        improvement_quantiles=(0.75, 0.25),  # Top 25% vs bottom 25%
        patch_quantile=0.80,  # Top 20% of patches within each image
    )
    patch_impact_results_per_layer[layer_idx] = patch_impact

    # Stage 2: Feature composition analysis
    feature_composition = analyze_feature_composition_of_patches(
        combined_results,
        patch_impact,
        layer_idx=layer_idx,
        min_occurrences=100,  # Require at least 100 total occurrences
        top_n=20
    )
    feature_composition_results_per_layer[layer_idx] = feature_composition

    # Stage 3: Distribution analysis to justify enrichment approach
    distribution_analysis = analyze_feature_distributions(
        combined_results,
        layer_idx=layer_idx,
        feature_composition_results=feature_composition,
        output_dir=Path(f"{experiment_path}/two_stage_analysis/distributions"),
        top_n_enriched=100
    )

output_base = Path(f"{experiment_path}/two_stage_analysis")
save_analysis_results(patch_impact_results_per_layer, feature_composition_results_per_layer, output_dir=output_base)

# Determine image directory based on dataset
if dataset == "imagenet":
    image_dir = Path("./data/imagenet_unified/test")
else:  # covidquex
    image_dir = Path("./data/covidquex_unified/val")

visualize_top_features(
    feature_composition_results_per_layer,
    merged_data=combined_results,
    patch_impact_results=patch_impact_results_per_layer,
    image_dir=image_dir,
    output_dir=output_base / "visualizations",
    vanilla_attribution_dir=vanilla_experiment_path / "attributions",
    gated_attribution_dir=gated_experiment_path / "attributions",
    n_features=30,
    n_examples=50,
    dataset=dataset
)
generate_summary_report(feature_composition_results_per_layer, output_path=output_base / "summary_report.txt")
