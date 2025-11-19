import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_faithfulness_results(experiment_path: Path) -> pd.DataFrame:
    csv_files = list(experiment_path.glob("analysis_faithfulness_correctness_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No faithfulness CSV found in {experiment_path}")

    csv_path = sorted(csv_files)[-1]
    print(f"Loading faithfulness data from: {csv_path}")
    df = pd.read_csv(csv_path)

    json_files = list(experiment_path.glob("faithfulness_stats_*.json"))
    if json_files:
        json_path = sorted(json_files)[-1]
        print(f"Loading additional metrics from: {json_path}")
        with open(json_path, 'r') as f:
            stats = json.load(f)

        metrics = stats.get('metrics', {})
        for metric_name, metric_data in metrics.items():
            if 'mean_scores' in metric_data:
                df[metric_name] = metric_data['mean_scores']

    df['image_idx'] = range(len(df))
    return df


def load_features_debug(experiment_path: Path, layers: Optional[List[int]] = None) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load debug feature data for specified layers.

    Args:
        experiment_path: Path to experiment folder containing debug_data/
        layers: List of layer indices to load (if None, loads all available)

    Returns:
        Dictionary mapping layer_idx -> {
            'sparse_indices': object array [n_images], each element is list of arrays per patch
            'sparse_activations': object array [n_images]
            'sparse_gradients': object array [n_images]
            'sparse_contributions': object array [n_images]
            'gate_values': array [n_images, 196]
            'contribution_sum': array [n_images, 196]
            'total_contribution_magnitude': array [n_images, 196]
        }
    """
    debug_dir = experiment_path / "debug_data"

    if not debug_dir.exists():
        raise FileNotFoundError(f"Debug data directory not found: {debug_dir}")

    # Find all layer debug files
    debug_files = list(debug_dir.glob("layer_*_debug.npz"))

    if not debug_files:
        raise FileNotFoundError(f"No debug NPZ files found in {debug_dir}")

    debug_data = {}
    for debug_file in sorted(debug_files):
        layer_idx = int(debug_file.stem.split('_')[1])

        print(f"Loading debug data for layer {layer_idx} from: {debug_file}")
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


def merge_features_faithfulness(features_debug: Dict[int, Dict[str, np.ndarray]],
                                faithfulness_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
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

    for layer_idx, layer_data in features_debug.items():
        print(f"\nMerging layer {layer_idx}...")

        rows = []

        n_images = len(layer_data['sparse_indices'])
        print(f"  Processing {n_images} images, faithfulness_df has {len(faithfulness_df)} rows")

        for img_idx in range(n_images):
            if img_idx >= len(faithfulness_df):
                print(f"Warning: image_idx {img_idx} not in faithfulness data, skipping")
                continue

            faith_row = faithfulness_df.iloc[img_idx]

            img_sparse_indices = layer_data['sparse_indices'][img_idx]
            img_sparse_activations = layer_data['sparse_activations'][img_idx]
            img_sparse_gradients = layer_data['sparse_gradients'][img_idx]
            img_gate_values = layer_data['gate_values'][img_idx]

            # Get new contribution fields (may be None for old caches)
            img_sparse_contributions = None
            if layer_data['sparse_contributions'] is not None:
                img_sparse_contributions = layer_data['sparse_contributions'][img_idx]

            img_contribution_sum = None
            if layer_data['contribution_sum'] is not None:
                img_contribution_sum = layer_data['contribution_sum'][img_idx]

            img_total_contribution_magnitude = None
            if layer_data['total_contribution_magnitude'] is not None:
                img_total_contribution_magnitude = layer_data['total_contribution_magnitude'][img_idx]

            # Get attribution deltas if available (may be None for old caches)
            img_attribution_deltas = None
            if layer_data['patch_attribution_deltas'] is not None:
                img_attribution_deltas = layer_data['patch_attribution_deltas'][img_idx]

            if img_idx == 0:
                print(
                    f"  Image 0: gate_values shape: {img_gate_values.shape if hasattr(img_gate_values, 'shape') else type(img_gate_values)}"
                )
                print(
                    f"  Image 0: sparse_indices type: {type(img_sparse_indices)}, len: {len(img_sparse_indices) if hasattr(img_sparse_indices, '__len__') else 'N/A'}"
                )
                if img_attribution_deltas is not None:
                    print(
                        f"  Image 0: attribution_deltas shape: {img_attribution_deltas.shape if hasattr(img_attribution_deltas, 'shape') else type(img_attribution_deltas)}"
                    )

            for patch_idx in range(len(img_gate_values)):
                row = {
                    'image_idx': img_idx,
                    'patch_idx': patch_idx,
                    'saco_score': faith_row['saco_score'],
                    'predicted_class': faith_row['predicted_class'],
                    'true_class': faith_row['true_class'],
                    'is_correct': faith_row['is_correct'],
                    'FaithfulnessCorrelation': faith_row.get('FaithfulnessCorrelation', np.nan),
                    'PixelFlipping': faith_row.get('PixelFlipping', np.nan),
                    'gate_value': img_gate_values[patch_idx],
                    'attribution_delta': img_attribution_deltas[patch_idx] if img_attribution_deltas is not None else np.nan,
                    'contribution_sum': img_contribution_sum[patch_idx] if img_contribution_sum is not None else np.nan,
                    'total_contribution_magnitude': img_total_contribution_magnitude[patch_idx] if img_total_contribution_magnitude is not None else np.nan,
                    'active_features': img_sparse_indices[patch_idx],
                    'feature_activations': img_sparse_activations[patch_idx],
                    'feature_gradients': img_sparse_gradients[patch_idx],
                    'feature_contributions': img_sparse_contributions[patch_idx] if img_sparse_contributions is not None else [],
                    'n_active_features': len(img_sparse_indices[patch_idx])
                }
                rows.append(row)

        merged_df = pd.DataFrame(rows)
        merged_data[layer_idx] = merged_df

        print(f"  Layer {layer_idx}: {len(merged_df)} patch-level rows ({len(merged_df) // 196} images)")

    return merged_data


def compute_feature_statistics(
    merged_data: Dict[int, pd.DataFrame], layer_idx: int, chunk_size: int = 50000
) -> pd.DataFrame:
    df = merged_data[layer_idx]

    if df.empty:
        print(f"Warning: No data for layer {layer_idx}, returning empty DataFrame")
        return pd.DataFrame()

    print(f"Computing feature statistics for layer {layer_idx}...")
    print(f"  Processing {len(df)} rows in chunks of {chunk_size}")

    # Process in chunks to avoid OOM
    # Accumulate statistics using dictionaries (memory-efficient)
    from collections import defaultdict

    feature_stats = defaultdict(
        lambda: {
            'count': 0,
            'sum_activation': 0.0,
            'sum_activation_sq': 0.0,
            'sum_gradient': 0.0,
            'sum_gradient_sq': 0.0,
            'sum_gate': 0.0,
            'sum_gate_sq': 0.0,
            'sum_saco': 0.0,
            'sum_saco_sq': 0.0,
            'sum_saco_correct': 0.0,
            'count_correct': 0,
            'sum_saco_incorrect': 0.0,
            'count_incorrect': 0,
            'sum_faith_corr': 0.0,
            'count_faith_corr': 0,
            'sum_pixel_flip': 0.0,
            'count_pixel_flip': 0,
        }
    )

    # Process in chunks
    n_rows = len(df)
    for chunk_start in range(0, n_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_rows)
        print(f"  Processing rows {chunk_start} to {chunk_end}")

        chunk_df = df.iloc[chunk_start:chunk_end]

        # Process each row in chunk (still faster than iterrows)
        for idx in chunk_df.index:
            row = chunk_df.loc[idx]
            features = row['active_features']
            activations = row['feature_activations']
            gradients = row['feature_gradients']
            gate = row['gate_value']
            saco = row['saco_score']
            is_correct = row['is_correct']
            faith_corr = row.get('FaithfulnessCorrelation', np.nan)
            pixel_flip = row.get('PixelFlipping', np.nan)

            # Accumulate statistics for each feature
            for feat_idx, feat_act, feat_grad in zip(features, activations, gradients):
                stats = feature_stats[feat_idx]
                stats['count'] += 1
                stats['sum_activation'] += feat_act
                stats['sum_activation_sq'] += feat_act**2
                stats['sum_gradient'] += feat_grad
                stats['sum_gradient_sq'] += feat_grad**2
                stats['sum_gate'] += gate
                stats['sum_gate_sq'] += gate**2
                stats['sum_saco'] += saco
                stats['sum_saco_sq'] += saco**2

                if is_correct:
                    stats['sum_saco_correct'] += saco
                    stats['count_correct'] += 1
                else:
                    stats['sum_saco_incorrect'] += saco
                    stats['count_incorrect'] += 1

                if not np.isnan(faith_corr):
                    stats['sum_faith_corr'] += faith_corr
                    stats['count_faith_corr'] += 1

                if not np.isnan(pixel_flip):
                    stats['sum_pixel_flip'] += pixel_flip
                    stats['count_pixel_flip'] += 1

    # Convert accumulated statistics to DataFrame
    print(f"  Computing final statistics...")
    stats_rows = []
    for feat_idx, stats in feature_stats.items():
        n = stats['count']

        # Compute means and standard deviations
        mean_act = stats['sum_activation'] / n
        std_act = np.sqrt(stats['sum_activation_sq'] / n - mean_act**2)

        mean_grad = stats['sum_gradient'] / n
        std_grad = np.sqrt(stats['sum_gradient_sq'] / n - mean_grad**2)

        mean_gate = stats['sum_gate'] / n
        std_gate = np.sqrt(stats['sum_gate_sq'] / n - mean_gate**2)

        mean_saco = stats['sum_saco'] / n
        std_saco = np.sqrt(stats['sum_saco_sq'] / n - mean_saco**2)

        mean_saco_correct = stats['sum_saco_correct'] / stats['count_correct'] if stats['count_correct'] > 0 else np.nan
        mean_saco_incorrect = stats['sum_saco_incorrect'] / stats['count_incorrect'] if stats['count_incorrect'
                                                                                              ] > 0 else np.nan
        mean_faith_corr = stats['sum_faith_corr'] / stats['count_faith_corr'] if stats['count_faith_corr'
                                                                                       ] > 0 else np.nan
        mean_pixel_flip = stats['sum_pixel_flip'] / stats['count_pixel_flip'] if stats['count_pixel_flip'
                                                                                       ] > 0 else np.nan

        stats_rows.append({
            'feature_idx': feat_idx,
            'n_occurrences': n,
            'mean_activation': mean_act,
            'std_activation': std_act,
            'mean_gradient': mean_grad,
            'std_gradient': std_grad,
            'mean_gate': mean_gate,
            'std_gate': std_gate,
            'mean_saco_score': mean_saco,
            'std_saco_score': std_saco,
            'mean_saco_correct': mean_saco_correct,
            'mean_saco_incorrect': mean_saco_incorrect,
            'mean_faith_corr': mean_faith_corr,
            'mean_pixel_flip': mean_pixel_flip,
        })

    feature_stats_df = pd.DataFrame(stats_rows)
    feature_stats_df = feature_stats_df.sort_values('n_occurrences', ascending=False)

    print(f"  Found {len(feature_stats_df)} unique features")
    print(f"  Top 10 most common features: {feature_stats_df.head(10)['feature_idx'].tolist()}")

    return feature_stats_df


def save_analysis_results(
    merged_data: Dict[int, pd.DataFrame],
    feature_stats: Dict[int, pd.DataFrame],
    output_path: Path,
    feature_impacts: Optional[Dict[int, pd.DataFrame]] = None,
):
    """
    Save merged data and feature statistics to disk for later use.

    Args:
        merged_data: Output from merge_features_faithfulness
        feature_stats: Output from compute_feature_statistics
        output_path: Directory to save results
        feature_impacts: Optional output from compute_feature_impact_discovery
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving analysis results to {output_path}...")

    # Save merged_data (one parquet file per layer)
    merged_dir = output_path / "merged_data"
    merged_dir.mkdir(exist_ok=True)
    for layer_idx, df in merged_data.items():
        df.to_parquet(merged_dir / f"layer_{layer_idx}.parquet")
        print(f"  Saved merged_data for layer {layer_idx}: {len(df)} rows")

    # Save feature_stats (one parquet file per layer)
    stats_dir = output_path / "feature_stats"
    stats_dir.mkdir(exist_ok=True)
    for layer_idx, df in feature_stats.items():
        df.to_parquet(stats_dir / f"layer_{layer_idx}.parquet")
        print(f"  Saved feature_stats for layer {layer_idx}: {len(df)} features")

    # Save feature_impacts (one parquet file per layer)
    if feature_impacts is not None:
        impacts_dir = output_path / "feature_impacts"
        impacts_dir.mkdir(exist_ok=True)
        for layer_idx, df in feature_impacts.items():
            df.to_parquet(impacts_dir / f"layer_{layer_idx}.parquet")
            print(f"  Saved feature_impacts for layer {layer_idx}: {len(df)} features")

    print("Done saving!")


def load_analysis_results(
    input_path: Path
) -> tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Load previously saved analysis results.

    Args:
        input_path: Directory where results were saved

    Returns:
        Tuple of (merged_data, feature_stats, feature_impacts)
        feature_impacts will be empty dict if not found
    """
    input_path = Path(input_path)

    print(f"Loading analysis results from {input_path}...")

    # Load merged_data
    merged_dir = input_path / "merged_data"
    merged_data = {}
    for parquet_file in sorted(merged_dir.glob("layer_*.parquet")):
        layer_idx = int(parquet_file.stem.split('_')[1])
        merged_data[layer_idx] = pd.read_parquet(parquet_file)
        print(f"  Loaded merged_data for layer {layer_idx}: {len(merged_data[layer_idx])} rows")

    # Load feature_stats
    stats_dir = input_path / "feature_stats"
    feature_stats = {}
    for parquet_file in sorted(stats_dir.glob("layer_*.parquet")):
        layer_idx = int(parquet_file.stem.split('_')[1])
        feature_stats[layer_idx] = pd.read_parquet(parquet_file)
        print(f"  Loaded feature_stats for layer {layer_idx}: {len(feature_stats[layer_idx])} features")

    # Load feature_impacts (may not exist for older caches)
    impacts_dir = input_path / "feature_impacts"
    feature_impacts = {}
    if impacts_dir.exists():
        for parquet_file in sorted(impacts_dir.glob("layer_*.parquet")):
            layer_idx = int(parquet_file.stem.split('_')[1])
            feature_impacts[layer_idx] = pd.read_parquet(parquet_file)
            print(f"  Loaded feature_impacts for layer {layer_idx}: {len(feature_impacts[layer_idx])} features")
    else:
        print("  No feature_impacts found (older cache format)")

    print("Done loading!")
    return merged_data, feature_stats, feature_impacts


def compare_vanilla_gated(faithfulness_vanilla: pd.DataFrame, faithfulness_gated: pd.DataFrame) -> pd.DataFrame:
    metrics_to_compare = ['saco_score', 'FaithfulnessCorrelation', 'PixelFlipping']

    comparison = pd.DataFrame()

    for metric in metrics_to_compare:
        if metric in faithfulness_vanilla.columns and metric in faithfulness_gated.columns:
            vanilla_mean = faithfulness_vanilla[metric].mean()
            gated_mean = faithfulness_gated[metric].mean()
            improvement = gated_mean - vanilla_mean

            comparison = pd.concat([
                comparison,
                pd.DataFrame({
                    'metric': [metric],
                    'vanilla_mean': [vanilla_mean],
                    'gated_mean': [gated_mean],
                    'improvement': [improvement]
                })
            ],
                                   ignore_index=True)

    print("\n=== Faithfulness Comparison ===")
    print(comparison.to_string(index=False))

    return comparison


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


def get_attribution_path_covidquex(image_idx: int, attribution_dir: Path) -> Optional[Path]:
    """
    Map image_idx to attribution file path for CovidQuex dataset.
    """
    if image_idx < 1903:
        class_idx = 0
        local_idx = image_idx
    elif image_idx < 3705:
        class_idx = 1
        local_idx = image_idx - 1903
    else:
        class_idx = 2
        local_idx = image_idx - 3705

    attr_path = attribution_dir / f"img_{class_idx:02d}_val_{local_idx:05d}_attribution.npy"

    if attr_path.exists():
        return attr_path
    else:
        return None


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Overlay a heatmap on an image using a colormap.

    Args:
        image: PIL Image (RGB)
        heatmap: 2D numpy array [H, W] with values in [0, 1]
        alpha: Transparency of overlay

    Returns:
        PIL Image with heatmap overlay
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # Apply colormap (hot: black -> red -> yellow -> white)
    colormap = cm.get_cmap('hot')
    heatmap_colored = colormap(heatmap)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Convert to PIL
    heatmap_img = Image.fromarray(heatmap_colored, mode='RGB')

    # Blend with original image
    blended = Image.blend(image, heatmap_img, alpha=alpha)

    return blended


def visualize_feature_activations(
    feature_idx: int,
    layer_idx: int,
    merged_data: Dict[int, pd.DataFrame],
    faithfulness_vanilla: pd.DataFrame,
    faithfulness_gated: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    vanilla_attr_dir: Optional[Path] = None,
    gated_attr_dir: Optional[Path] = None,
    top_k: int = 50,
    sort_by: str = 'activation',
    sort_ascending: bool = False,
    patch_size: int = 16,
    image_size: int = 224,
):
    """
    Visualize where a specific feature activates across images with attribution heatmap overlays.

    Creates 3x3 grid visualizations:
    - Row 1: Original | Vanilla Attribution Heatmap | Gated Attribution Heatmap
    - Row 2: Difference Map | Feature Active Patches | Gate Strength Heatmap
    - Row 3: Original | Vanilla Attribution Values | Gated Attribution Values

    Args:
        feature_idx: The feature to visualize
        layer_idx: Which layer this feature is from
        merged_data: Output from merge_features_faithfulness
        faithfulness_vanilla: Vanilla faithfulness results for comparison
        faithfulness_gated: Gated faithfulness results
        image_dir: Directory containing original images
        output_dir: Where to save visualizations
        vanilla_attr_dir: Directory containing vanilla attribution .npy files
        gated_attr_dir: Directory containing gated attribution .npy files
        top_k: How many top images to visualize
        sort_by: Sort images by 'activation', 'gate', or 'attr_diff' (attribution difference)
        sort_ascending: If True, sort ascending (lowest first); if False, descending (highest first)
        patch_size: Size of each patch in pixels (14 for ViT-Base/16)
        image_size: Total image size (224 for ViT-Base)
    """
    df = merged_data[layer_idx]

    # Filter to rows where this feature is active
    feature_rows = []
    for idx in df.index:
        row = df.loc[idx]
        if feature_idx in row['active_features']:
            # Find the index of this feature in the active_features list
            feat_position = list(row['active_features']).index(feature_idx)
            feature_rows.append({
                'image_idx': row['image_idx'],
                'patch_idx': row['patch_idx'],
                'activation': row['feature_activations'][feat_position],
                'gradient': row['feature_gradients'][feat_position],
                'gate_value': row['gate_value'],
                'saco_score': row['saco_score'],
                'is_correct': row['is_correct'],
                'true_class': row['true_class'],
                'FaithfulnessCorrelation': row.get('FaithfulnessCorrelation', np.nan),
                'PixelFlipping': row.get('PixelFlipping', np.nan),
            })

    if not feature_rows:
        print(f"Feature {feature_idx} not found in layer {layer_idx}")
        return

    feature_df = pd.DataFrame(feature_rows)
    print(f"Feature {feature_idx} appears in {len(feature_df)} patch-level occurrences")

    # Group by image to get all patches per image
    images_with_feature = feature_df.groupby('image_idx').agg({
        'patch_idx': list,
        'activation': list,
        'gate_value': list,
        'saco_score': 'first',
        'is_correct': 'first',
        'true_class': 'first',
        'FaithfulnessCorrelation': 'first',
        'PixelFlipping': 'first',
    }).reset_index()

    patches_per_side = image_size // patch_size

    # Sort by the requested metric
    if sort_by == 'activation':
        images_with_feature['sort_value'] = images_with_feature['activation'].apply(np.mean)
    elif sort_by == 'gate':
        images_with_feature['sort_value'] = images_with_feature['gate_value'].apply(np.mean)
    elif sort_by == 'attr_diff':
        # Sort by mean attribution difference in patches where feature is active
        if vanilla_attr_dir is None or gated_attr_dir is None:
            print("Warning: attr_diff sorting requires attribution directories, falling back to activation")
            images_with_feature['sort_value'] = images_with_feature['activation'].apply(np.mean)
        else:
            attr_diffs = []
            for _, row in images_with_feature.iterrows():
                img_idx = row['image_idx']
                patch_indices = row['patch_idx']

                # Load attributions
                vanilla_attr_path = get_attribution_path_covidquex(img_idx, vanilla_attr_dir)
                gated_attr_path = get_attribution_path_covidquex(img_idx, gated_attr_dir)

                if vanilla_attr_path and gated_attr_path:
                    vanilla_attr = np.load(vanilla_attr_path)
                    gated_attr = np.load(gated_attr_path)

                    # Compute mean attribution difference in patches where feature is active
                    patch_diffs = []
                    for patch_idx in patch_indices:
                        patch_row = patch_idx // patches_per_side
                        patch_col = patch_idx % patches_per_side
                        x1 = patch_col * patch_size
                        y1 = patch_row * patch_size
                        x2 = x1 + patch_size
                        y2 = y1 + patch_size

                        vanilla_val = vanilla_attr[y1:y2, x1:x2].mean()
                        gated_val = gated_attr[y1:y2, x1:x2].mean()
                        patch_diffs.append(gated_val - vanilla_val)

                    attr_diffs.append(np.mean(patch_diffs))
                else:
                    attr_diffs.append(0.0)

            images_with_feature['sort_value'] = attr_diffs
    else:
        raise ValueError(f"sort_by must be 'activation', 'gate', or 'attr_diff', got '{sort_by}'")

    images_with_feature = images_with_feature.sort_values('sort_value', ascending=sort_ascending).head(top_k)

    print(f"Visualizing top {len(images_with_feature)} images sorted by {sort_by}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
        small_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 16)
        tiny_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
        micro_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 7)  # For patch values
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        tiny_font = ImageFont.load_default()
        micro_font = ImageFont.load_default()

    for _, row in images_with_feature.iterrows():
        img_idx = row['image_idx']
        patch_indices = row['patch_idx']
        activations = row['activation']
        gate_values = row['gate_value']

        # Get vanilla metrics for comparison
        vanilla_row = faithfulness_vanilla.iloc[img_idx]
        gated_row = faithfulness_gated.iloc[img_idx]

        # Find the image file using dataset-specific mapping
        image_path = get_image_path_covidquex(img_idx, image_dir)

        if image_path is None:
            print(f"Warning: Could not find image for idx {img_idx}")
            continue

        # Load image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((image_size, image_size))

        # Load attribution maps if directories provided
        vanilla_attr = None
        gated_attr = None
        if vanilla_attr_dir is not None:
            vanilla_attr_path = get_attribution_path_covidquex(img_idx, vanilla_attr_dir)
            if vanilla_attr_path:
                vanilla_attr = np.load(vanilla_attr_path)

        if gated_attr_dir is not None:
            gated_attr_path = get_attribution_path_covidquex(img_idx, gated_attr_dir)
            if gated_attr_path:
                gated_attr = np.load(gated_attr_path)

        # Create 2x3 grid images
        # Top row: Original, Vanilla Attribution, Gated Attribution
        img_original = img.copy()

        if vanilla_attr is not None:
            img_vanilla_attr = overlay_heatmap_on_image(img.copy(), vanilla_attr, alpha=0.5)
        else:
            img_vanilla_attr = img.copy()

        if gated_attr is not None:
            img_gated_attr = overlay_heatmap_on_image(img.copy(), gated_attr, alpha=0.5)
        else:
            img_gated_attr = img.copy()

        # Bottom row: Difference map, Feature boxes, Gate heatmap
        if vanilla_attr is not None and gated_attr is not None:
            diff_attr = gated_attr - vanilla_attr
            # Normalize diff to [-1, 1] -> [0, 1]
            diff_attr_norm = (diff_attr + 1) / 2
            diff_attr_norm = np.clip(diff_attr_norm, 0, 1)
            img_diff = overlay_heatmap_on_image(img.copy(), diff_attr_norm, alpha=0.6)
        else:
            img_diff = img.copy()

        # Feature boxes
        img_boxes = img.copy()
        draw_boxes = ImageDraw.Draw(img_boxes)
        for patch_idx, activation, gate_val in zip(patch_indices, activations, gate_values):
            patch_row = patch_idx // patches_per_side
            patch_col = patch_idx % patches_per_side
            x1 = patch_col * patch_size
            y1 = patch_row * patch_size
            x2 = x1 + patch_size
            y2 = y1 + patch_size
            box_color = (0, 255, 0) if gate_val > 1.0 else (255, 0, 0)
            draw_boxes.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

        # Gate heatmap with diverging colormap (blue = deboost, white = neutral, red = boost)
        import matplotlib.pyplot as plt
        from matplotlib import cm

        gate_heatmap = np.zeros((patches_per_side, patches_per_side))
        gate_mask = np.zeros((patches_per_side, patches_per_side), dtype=bool)  # Track where feature is active

        for patch_idx, gate_val in zip(patch_indices, gate_values):
            patch_row = patch_idx // patches_per_side
            patch_col = patch_idx % patches_per_side
            gate_heatmap[patch_row, patch_col] = gate_val
            gate_mask[patch_row, patch_col] = True

        # Use diverging colormap centered at gate=1.0 (neutral)
        # Map gates so that: gate=0 -> 0, gate=1 -> 0.5 (white), gate=3+ -> 1.0 (red)
        gate_heatmap_norm = np.where(
            gate_mask,
            (gate_heatmap - 1.0) / 2.0 + 0.5,  # Center at gate=1.0
            0.5  # Gray will be applied later anyway
        )
        gate_heatmap_norm = np.clip(gate_heatmap_norm, 0, 1)

        # Apply RdBu_r colormap (red = high, blue = low, white = middle)
        colormap = cm.get_cmap('RdBu_r')
        gate_colored = colormap(gate_heatmap_norm)[:, :, :3]

        # Make inactive patches gray
        for i in range(patches_per_side):
            for j in range(patches_per_side):
                if not gate_mask[i, j]:
                    gate_colored[i, j] = [0.9, 0.9, 0.9]  # Light gray for inactive

        gate_colored = (gate_colored * 255).astype(np.uint8)
        gate_heatmap_img = Image.fromarray(gate_colored, mode='RGB')
        gate_heatmap_img = gate_heatmap_img.resize((image_size, image_size), Image.NEAREST)

        # Blend with original image
        img_gate = Image.blend(img.copy(), gate_heatmap_img, alpha=0.6)

        # Row 3: Original with attribution values as text
        img_original_copy = img.copy()

        # Create images with attribution values overlaid
        img_vanilla_values = img.copy()
        img_gated_values = img.copy()

        draw_vanilla_values = ImageDraw.Draw(img_vanilla_values)
        draw_gated_values = ImageDraw.Draw(img_gated_values)

        # For each patch where feature is active, extract attribution values
        for patch_idx in patch_indices:
            patch_row = patch_idx // patches_per_side
            patch_col = patch_idx % patches_per_side

            # Pixel coordinates
            x1 = patch_col * patch_size
            y1 = patch_row * patch_size
            x2 = x1 + patch_size
            y2 = y1 + patch_size

            # Extract attribution values from that patch region
            if vanilla_attr is not None:
                patch_vanilla_attr = vanilla_attr[y1:y2, x1:x2].mean()
                # Draw text centered in patch
                text = f"{patch_vanilla_attr:.2f}"
                # Use white text with black outline for visibility
                text_bbox = draw_vanilla_values.textbbox((0, 0), text, font=micro_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x1 + (patch_size - text_width) // 2
                text_y = y1 + (patch_size - text_height) // 2
                # Draw outline
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw_vanilla_values.text((text_x + dx, text_y + dy), text, fill=(0, 0, 0), font=micro_font)
                # Draw text
                draw_vanilla_values.text((text_x, text_y), text, fill=(255, 255, 255), font=micro_font)

            if gated_attr is not None:
                patch_gated_attr = gated_attr[y1:y2, x1:x2].mean()
                text = f"{patch_gated_attr:.2f}"
                text_bbox = draw_gated_values.textbbox((0, 0), text, font=micro_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x1 + (patch_size - text_width) // 2
                text_y = y1 + (patch_size - text_height) // 2
                # Draw outline
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw_gated_values.text((text_x + dx, text_y + dy), text, fill=(0, 0, 0), font=micro_font)
                # Draw text
                draw_gated_values.text((text_x, text_y), text, fill=(255, 255, 255), font=micro_font)

        # Create 3x3 grid
        top_padding = 100
        bottom_padding = 160
        spacing = 15
        grid_width = image_size * 3 + spacing * 2
        grid_height = image_size * 3 + spacing * 2 + top_padding + bottom_padding

        combined = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))

        # Paste images in 3x3 grid
        # Row 1
        combined.paste(img_original, (0, top_padding))
        combined.paste(img_vanilla_attr, (image_size + spacing, top_padding))
        combined.paste(img_gated_attr, (image_size * 2 + spacing * 2, top_padding))
        # Row 2
        row2_y = image_size + spacing + top_padding
        combined.paste(img_diff, (0, row2_y))
        combined.paste(img_boxes, (image_size + spacing, row2_y))
        combined.paste(img_gate, (image_size * 2 + spacing * 2, row2_y))
        # Row 3
        row3_y = image_size * 2 + spacing * 2 + top_padding
        combined.paste(img_original_copy, (0, row3_y))
        combined.paste(img_vanilla_values, (image_size + spacing, row3_y))
        combined.paste(img_gated_values, (image_size * 2 + spacing * 2, row3_y))

        # Add labels and metadata
        draw_combined = ImageDraw.Draw(combined)

        # Title
        true_class = row['true_class']
        title = f"Feature {feature_idx} (Layer {layer_idx}) - Image {img_idx} - Class: {true_class}"
        draw_combined.text((15, 15), title, fill=(0, 0, 0), font=font)

        # Subtitle
        draw_combined.text(
            (15, 45),
            f"Patches: {len(patch_indices)} | Avg Act: {np.mean(activations):.3f} | Avg Gate: {np.mean(gate_values):.3f}",
            fill=(0, 0, 0),
            font=small_font
        )

        # Row 1 labels
        label_y = top_padding - 18
        draw_combined.text((10, label_y), "Original", fill=(0, 0, 0), font=tiny_font)
        draw_combined.text((image_size + spacing + 10, label_y), "Vanilla Attr.", fill=(0, 0, 0), font=tiny_font)
        draw_combined.text((image_size * 2 + spacing * 2 + 10, label_y), "Gated Attr.", fill=(0, 0, 0), font=tiny_font)

        # Row 2 labels
        label_y = image_size + spacing + top_padding - 18
        draw_combined.text((10, label_y), "Difference", fill=(0, 0, 0), font=tiny_font)
        draw_combined.text((image_size + spacing + 10, label_y), "Feature Active", fill=(0, 0, 0), font=tiny_font)
        draw_combined.text((image_size * 2 + spacing * 2 + 10, label_y), "Gate Strength", fill=(0, 0, 0), font=tiny_font)

        # Row 3 labels
        label_y = image_size * 2 + spacing * 2 + top_padding - 18
        draw_combined.text((10, label_y), "Original", fill=(0, 0, 0), font=tiny_font)
        draw_combined.text((image_size + spacing + 10, label_y), "Vanilla Values", fill=(0, 0, 0), font=tiny_font)
        draw_combined.text((image_size * 2 + spacing * 2 + 10, label_y), "Gated Values", fill=(0, 0, 0), font=tiny_font)

        # Bottom metadata (after row 3)
        y_offset = image_size * 3 + spacing * 2 + top_padding + 15
        saco_change = gated_row['saco_score'] - vanilla_row['saco_score']
        faith_change = gated_row['FaithfulnessCorrelation'] - vanilla_row['FaithfulnessCorrelation']
        pixel_change = gated_row['PixelFlipping'] - vanilla_row['PixelFlipping']

        draw_combined.text((15, y_offset),
                           f"ΔSaCo: {saco_change:+.3f} | ΔFaith: {faith_change:+.3f} | ΔPixel: {pixel_change:+.3f}",
                           fill=(0, 0, 0),
                           font=small_font)
        draw_combined.text((15, y_offset + 22),
                           f"Vanilla: SaCo={vanilla_row['saco_score']:.3f} | Gated: SaCo={gated_row['saco_score']:.3f}",
                           fill=(0, 0, 0),
                           font=tiny_font)

        # Gate heatmap legend
        draw_combined.text((15, y_offset + 44),
                           f"Gate colors: Blue=deboosting (<1.0) | White=neutral (1.0) | Red=boosting (>1.0) | Gray=inactive",
                           fill=(0, 0, 0),
                           font=tiny_font)

        # Save
        output_path = output_dir / f"feature_{feature_idx}_layer_{layer_idx}_image_{img_idx:05d}.png"
        combined.save(output_path)

    print(f"Saved {len(images_with_feature)} visualizations to {output_dir}")



def compute_per_class_statistics(deltas_subset: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    """
    Compute per-class impact statistics from a subset of deltas.

    Args:
        deltas_subset: DataFrame with columns [delta_saco, delta_faith, delta_pixel, true_class]
        prefix: Prefix for column names (e.g., 'boost' or 'deboost')

    Returns:
        Dictionary with per-class statistics and class counts
    """
    stats = {}

    # Get unique classes
    classes = deltas_subset['true_class'].unique()

    for class_name in classes:
        class_deltas = deltas_subset[deltas_subset['true_class'] == class_name]

        if len(class_deltas) == 0:
            continue

        # Count occurrences
        stats[f'n_{class_name}'] = len(class_deltas)

        # Mean deltas per class
        stats[f'{prefix}_delta_saco_mean_{class_name}'] = class_deltas['delta_saco'].mean()
        stats[f'{prefix}_delta_faith_mean_{class_name}'] = class_deltas['delta_faith'].mean()
        stats[f'{prefix}_delta_pixel_mean_{class_name}'] = class_deltas['delta_pixel'].mean()

        # Improvement rates per class
        stats[f'{prefix}_saco_improvement_rate_{class_name}'] = (class_deltas['delta_saco'] > 0).mean()
        stats[f'{prefix}_faith_improvement_rate_{class_name}'] = (class_deltas['delta_faith'] > 0).mean()
        stats[f'{prefix}_pixel_improvement_rate_{class_name}'] = (class_deltas['delta_pixel'] > 0).mean()

    return stats


def compute_feature_impact_discovery(
    merged_data: Dict[int, pd.DataFrame],
    faithfulness_vanilla: pd.DataFrame,
    faithfulness_gated: pd.DataFrame,
    layer_idx: int,
    min_occurrences: int = 50,
    compute_spatial: bool = True
) -> pd.DataFrame:
    """
    Find features whose presence correlates with faithfulness changes.
    Goal: Scientific discovery of what image patterns matter for explanations.

    Args:
        merged_data: Output from merge_features_faithfulness
        faithfulness_vanilla: Vanilla faithfulness results
        faithfulness_gated: Gated faithfulness results
        layer_idx: Which layer to analyze
        min_occurrences: Minimum number of times feature must appear
        compute_spatial: Whether to compute spatial entropy (can be slow)

    Returns:
        DataFrame with impact statistics per feature, sorted by total impact
    """
    from collections import defaultdict

    df = merged_data[layer_idx]

    if df.empty:
        print(f"Warning: No data for layer {layer_idx}")
        return pd.DataFrame()

    print(f"Computing feature impact for layer {layer_idx}...")

    # Compute per-image deltas once
    deltas = pd.DataFrame({
        'image_idx':
        range(len(faithfulness_vanilla)),
        'delta_saco':
        faithfulness_gated['saco_score'].values - faithfulness_vanilla['saco_score'].values,
        'delta_faith':
        faithfulness_gated['FaithfulnessCorrelation'].values - faithfulness_vanilla['FaithfulnessCorrelation'].values,
        'delta_pixel':
        faithfulness_gated['PixelFlipping'].values - faithfulness_vanilla['PixelFlipping'].values,
        'true_class':
        faithfulness_vanilla['true_class'].values,
    })

    # Convert to dict for O(1) lookups instead of O(n) filtering
    deltas_dict = deltas.set_index('image_idx').to_dict('index')

    # Compute quantile thresholds for differential enrichment analysis
    top_percentile = 0.75
    bottom_percentile = 0.25

    # Define image groups based on delta quantiles
    high_imgs_saco = set(deltas[deltas['delta_saco'] > deltas['delta_saco'].quantile(top_percentile)]['image_idx'])
    decrease_imgs_saco = set(deltas[deltas['delta_saco'] < deltas['delta_saco'].quantile(bottom_percentile)]['image_idx'])

    high_imgs_faith = set(deltas[deltas['delta_faith'] > deltas['delta_faith'].quantile(top_percentile)]['image_idx'])
    decrease_imgs_faith = set(deltas[deltas['delta_faith'] < deltas['delta_faith'].quantile(bottom_percentile)]['image_idx'])

    high_imgs_pixel = set(deltas[deltas['delta_pixel'] > deltas['delta_pixel'].quantile(top_percentile)]['image_idx'])
    decrease_imgs_pixel = set(deltas[deltas['delta_pixel'] < deltas['delta_pixel'].quantile(bottom_percentile)]['image_idx'])

    # For each image, find which features are present and their average gate
    # Also track spatial locations and activations for spatial moment computation
    image_features = defaultdict(lambda: defaultdict(list))
    feature_spatial_data = defaultdict(lambda: {'patch_indices': [], 'activations': []})

    # Determine grid size (assuming 14x14 patches for ViT-Base/16 on 224x224 images)
    patches_per_side = 14

    # Also track attribution deltas per feature per image
    feature_attribution_deltas = defaultdict(lambda: defaultdict(list))

    for idx in df.index:
        row = df.loc[idx]
        img_idx = row['image_idx']
        patch_idx = row['patch_idx']
        attribution_delta = row.get('attribution_delta', np.nan)

        for i, feat_idx in enumerate(row['active_features']):
            image_features[img_idx][feat_idx].append(row['gate_value'])

            # Store patch location and activation for spatial moments
            activation = row['feature_activations'][i]
            feature_spatial_data[feat_idx]['patch_indices'].append(patch_idx)
            feature_spatial_data[feat_idx]['activations'].append(activation)

            # Store attribution delta for this feature at this patch
            if not np.isnan(attribution_delta):
                feature_attribution_deltas[img_idx][feat_idx].append(attribution_delta)

    # Convert to per-image, per-feature summary
    feature_presence = defaultdict(
        lambda: {
            'images_with_boosting': [],
            'images_with_deboosting': [],
            'boost_magnitudes': [],
            'deboost_magnitudes': [],
            'boost_deltas_saco': [],  # For correlation computation
            'boost_deltas_faith': [],  # For correlation computation
            'boost_deltas_pixel': [],  # For correlation computation
            'boost_gates': [],  # For correlation computation
            'boost_attribution_deltas': [],  # Sum of abs attribution deltas where feature is active
            'deboost_deltas_saco': [],  # For correlation computation
            'deboost_deltas_faith': [],  # For correlation computation
            'deboost_deltas_pixel': [],  # For correlation computation
            'deboost_gates': [],  # For correlation computation
            'deboost_attribution_deltas': [],  # Sum of abs attribution deltas where feature is active
            # Differential enrichment tracking
            'images_in_high_saco': [],
            'images_in_decrease_saco': [],
            'images_in_high_faith': [],
            'images_in_decrease_faith': [],
            'images_in_high_pixel': [],
            'images_in_decrease_pixel': [],
        }
    )

    for img_idx, features in image_features.items():
        for feat_idx, gates in features.items():
            avg_gate = np.mean(gates)

            # Get delta for correlation computation
            if img_idx in deltas_dict:
                delta_saco = deltas_dict[img_idx]['delta_saco']
                delta_faith = deltas_dict[img_idx]['delta_faith']
                delta_pixel = deltas_dict[img_idx]['delta_pixel']

                # Get total attribution delta for this feature in this image
                attr_delta_sum = 0.0
                if img_idx in feature_attribution_deltas and feat_idx in feature_attribution_deltas[img_idx]:
                    attr_delta_sum = np.sum(np.abs(feature_attribution_deltas[img_idx][feat_idx]))

                # Track differential enrichment - which group does this image belong to?
                if img_idx in high_imgs_saco:
                    feature_presence[feat_idx]['images_in_high_saco'].append(img_idx)
                if img_idx in decrease_imgs_saco:
                    feature_presence[feat_idx]['images_in_decrease_saco'].append(img_idx)

                if img_idx in high_imgs_faith:
                    feature_presence[feat_idx]['images_in_high_faith'].append(img_idx)
                if img_idx in decrease_imgs_faith:
                    feature_presence[feat_idx]['images_in_decrease_faith'].append(img_idx)

                if img_idx in high_imgs_pixel:
                    feature_presence[feat_idx]['images_in_high_pixel'].append(img_idx)
                if img_idx in decrease_imgs_pixel:
                    feature_presence[feat_idx]['images_in_decrease_pixel'].append(img_idx)

                if avg_gate > 1.0:  # Boosted
                    feature_presence[feat_idx]['images_with_boosting'].append(img_idx)
                    feature_presence[feat_idx]['boost_magnitudes'].append(avg_gate)
                    feature_presence[feat_idx]['boost_deltas_saco'].append(delta_saco)
                    feature_presence[feat_idx]['boost_deltas_faith'].append(delta_faith)
                    feature_presence[feat_idx]['boost_deltas_pixel'].append(delta_pixel)
                    feature_presence[feat_idx]['boost_gates'].append(avg_gate)
                    feature_presence[feat_idx]['boost_attribution_deltas'].append(attr_delta_sum)
                else:  # Deboosted
                    feature_presence[feat_idx]['images_with_deboosting'].append(img_idx)
                    feature_presence[feat_idx]['deboost_magnitudes'].append(avg_gate)
                    feature_presence[feat_idx]['deboost_deltas_saco'].append(delta_saco)
                    feature_presence[feat_idx]['deboost_deltas_faith'].append(delta_faith)
                    feature_presence[feat_idx]['deboost_deltas_pixel'].append(delta_pixel)
                    feature_presence[feat_idx]['deboost_gates'].append(avg_gate)
                    feature_presence[feat_idx]['deboost_attribution_deltas'].append(attr_delta_sum)

    # Compute impact statistics per feature
    impact_rows = []

    for feat_idx, presence in feature_presence.items():
        n_boosted = len(presence['images_with_boosting'])
        n_deboosted = len(presence['images_with_deboosting'])
        n_total = n_boosted + n_deboosted

        if n_total < min_occurrences:
            continue

        row = {
            'feature_idx': feat_idx,
            'n_total_occurrences': n_total,
            'n_boosted': n_boosted,
            'n_deboosted': n_deboosted,
        }

        # Compute class prevalence across all images where feature appears
        all_imgs = presence['images_with_boosting'] + presence['images_with_deboosting']
        class_counts = defaultdict(int)
        for img_idx in all_imgs:
            if img_idx in deltas_dict:
                class_counts[deltas_dict[img_idx]['true_class']] += 1
        for class_name, count in class_counts.items():
            row[f'n_total_{class_name}'] = count

        # Compute spatial moments (weighted by activation magnitude)
        if feat_idx in feature_spatial_data:
            patch_indices = np.array(feature_spatial_data[feat_idx]['patch_indices'])
            activations = np.array(feature_spatial_data[feat_idx]['activations'])

            # Convert patch indices to x,y coordinates (row-major order)
            y_coords = patch_indices // patches_per_side
            x_coords = patch_indices % patches_per_side

            # Normalize coordinates to [0, 1]
            y_coords_norm = y_coords / (patches_per_side - 1)
            x_coords_norm = x_coords / (patches_per_side - 1)

            # Weighted moments
            total_activation = activations.sum()
            if total_activation > 0:
                mean_x = np.sum(x_coords_norm * activations) / total_activation
                mean_y = np.sum(y_coords_norm * activations) / total_activation

                # Compute standard deviations
                std_x = np.sqrt(np.sum(((x_coords_norm - mean_x)**2) * activations) / total_activation)
                std_y = np.sqrt(np.sum(((y_coords_norm - mean_y)**2) * activations) / total_activation)
            else:
                mean_x = mean_y = std_x = std_y = 0.5  # Default to center if no activation

            row['spatial_mean_x'] = mean_x
            row['spatial_mean_y'] = mean_y
            row['spatial_std_x'] = std_x
            row['spatial_std_y'] = std_y
        else:
            row['spatial_mean_x'] = 0.5
            row['spatial_mean_y'] = 0.5
            row['spatial_std_x'] = 0.0
            row['spatial_std_y'] = 0.0

        # Compute differential enrichment metrics
        # SaCo enrichment
        n_in_high_saco = len(presence['images_in_high_saco'])
        n_in_decrease_saco = len(presence['images_in_decrease_saco'])
        n_high_total = len(high_imgs_saco)
        n_decrease_total = len(decrease_imgs_saco)

        enrichment_high_saco = n_in_high_saco / n_high_total if n_high_total > 0 else 0.0
        enrichment_decrease_saco = n_in_decrease_saco / n_decrease_total if n_decrease_total > 0 else 0.0
        specificity_saco = enrichment_high_saco - enrichment_decrease_saco

        row['n_in_high_saco'] = n_in_high_saco
        row['n_in_decrease_saco'] = n_in_decrease_saco
        row['enrichment_high_saco'] = enrichment_high_saco
        row['enrichment_decrease_saco'] = enrichment_decrease_saco
        row['specificity_saco'] = specificity_saco

        # Faithfulness enrichment
        n_in_high_faith = len(presence['images_in_high_faith'])
        n_in_decrease_faith = len(presence['images_in_decrease_faith'])
        n_high_faith_total = len(high_imgs_faith)
        n_decrease_faith_total = len(decrease_imgs_faith)

        enrichment_high_faith = n_in_high_faith / n_high_faith_total if n_high_faith_total > 0 else 0.0
        enrichment_decrease_faith = n_in_decrease_faith / n_decrease_faith_total if n_decrease_faith_total > 0 else 0.0
        specificity_faith = enrichment_high_faith - enrichment_decrease_faith

        row['n_in_high_faith'] = n_in_high_faith
        row['n_in_decrease_faith'] = n_in_decrease_faith
        row['enrichment_high_faith'] = enrichment_high_faith
        row['enrichment_decrease_faith'] = enrichment_decrease_faith
        row['specificity_faith'] = specificity_faith

        # Pixel flipping enrichment
        n_in_high_pixel = len(presence['images_in_high_pixel'])
        n_in_decrease_pixel = len(presence['images_in_decrease_pixel'])
        n_high_pixel_total = len(high_imgs_pixel)
        n_decrease_pixel_total = len(decrease_imgs_pixel)

        enrichment_high_pixel = n_in_high_pixel / n_high_pixel_total if n_high_pixel_total > 0 else 0.0
        enrichment_decrease_pixel = n_in_decrease_pixel / n_decrease_pixel_total if n_decrease_pixel_total > 0 else 0.0
        specificity_pixel = enrichment_high_pixel - enrichment_decrease_pixel

        row['n_in_high_pixel'] = n_in_high_pixel
        row['n_in_decrease_pixel'] = n_in_decrease_pixel
        row['enrichment_high_pixel'] = enrichment_high_pixel
        row['enrichment_decrease_pixel'] = enrichment_decrease_pixel
        row['specificity_pixel'] = specificity_pixel

        # Combined specificity score (average across all 3 metrics)
        row['specificity_combined'] = (specificity_saco + specificity_faith + specificity_pixel) / 3.0

        # Consistency: how many metrics show positive specificity (beneficial)?
        row['n_metrics_beneficial'] = sum([
            specificity_saco > 0,
            specificity_faith > 0,
            specificity_pixel > 0
        ])

        # How many metrics show negative specificity (harmful)?
        row['n_metrics_harmful'] = sum([
            specificity_saco < 0,
            specificity_faith < 0,
            specificity_pixel < 0
        ])

        # Compute impact-gate correlations for SaCo
        if len(presence['boost_gates']) > 1:
            boost_corr_saco = np.corrcoef(presence['boost_gates'], presence['boost_deltas_saco'])[0, 1]
            row['boost_impact_gate_correlation'] = boost_corr_saco if not np.isnan(boost_corr_saco) else 0.0
        else:
            row['boost_impact_gate_correlation'] = 0.0

        if len(presence['deboost_gates']) > 1:
            deboost_corr_saco = np.corrcoef(presence['deboost_gates'], presence['deboost_deltas_saco'])[0, 1]
            row['deboost_impact_gate_correlation'] = deboost_corr_saco if not np.isnan(deboost_corr_saco) else 0.0
        else:
            row['deboost_impact_gate_correlation'] = 0.0

        # Compute impact-gate correlations for Faithfulness
        if len(presence['boost_gates']) > 1:
            boost_corr_faith = np.corrcoef(presence['boost_gates'], presence['boost_deltas_faith'])[0, 1]
            row['boost_impact_gate_correlation_faith'] = boost_corr_faith if not np.isnan(boost_corr_faith) else 0.0
        else:
            row['boost_impact_gate_correlation_faith'] = 0.0

        if len(presence['deboost_gates']) > 1:
            deboost_corr_faith = np.corrcoef(presence['deboost_gates'], presence['deboost_deltas_faith'])[0, 1]
            row['deboost_impact_gate_correlation_faith'] = deboost_corr_faith if not np.isnan(deboost_corr_faith) else 0.0
        else:
            row['deboost_impact_gate_correlation_faith'] = 0.0

        # Compute impact-gate correlations for Pixel Flipping
        if len(presence['boost_gates']) > 1:
            boost_corr_pixel = np.corrcoef(presence['boost_gates'], presence['boost_deltas_pixel'])[0, 1]
            row['boost_impact_gate_correlation_pixel'] = boost_corr_pixel if not np.isnan(boost_corr_pixel) else 0.0
        else:
            row['boost_impact_gate_correlation_pixel'] = 0.0

        # Compute attribution-delta correlations (how much actual CAM change predicts improvement)
        if len(presence['boost_attribution_deltas']) > 1:
            # Mean attribution delta magnitude
            row['boost_mean_attribution_delta'] = np.mean(presence['boost_attribution_deltas'])

            # Correlation with SaCo improvement
            boost_attr_corr_saco = np.corrcoef(presence['boost_attribution_deltas'], presence['boost_deltas_saco'])[0, 1]
            row['boost_attribution_delta_correlation_saco'] = boost_attr_corr_saco if not np.isnan(boost_attr_corr_saco) else 0.0

            # Correlation with Faith improvement
            boost_attr_corr_faith = np.corrcoef(presence['boost_attribution_deltas'], presence['boost_deltas_faith'])[0, 1]
            row['boost_attribution_delta_correlation_faith'] = boost_attr_corr_faith if not np.isnan(boost_attr_corr_faith) else 0.0

            # Correlation with Pixel improvement
            boost_attr_corr_pixel = np.corrcoef(presence['boost_attribution_deltas'], presence['boost_deltas_pixel'])[0, 1]
            row['boost_attribution_delta_correlation_pixel'] = boost_attr_corr_pixel if not np.isnan(boost_attr_corr_pixel) else 0.0
        else:
            row['boost_mean_attribution_delta'] = 0.0
            row['boost_attribution_delta_correlation_saco'] = 0.0
            row['boost_attribution_delta_correlation_faith'] = 0.0
            row['boost_attribution_delta_correlation_pixel'] = 0.0

        if len(presence['deboost_attribution_deltas']) > 1:
            row['deboost_mean_attribution_delta'] = np.mean(presence['deboost_attribution_deltas'])

            deboost_attr_corr_saco = np.corrcoef(presence['deboost_attribution_deltas'], presence['deboost_deltas_saco'])[0, 1]
            row['deboost_attribution_delta_correlation_saco'] = deboost_attr_corr_saco if not np.isnan(deboost_attr_corr_saco) else 0.0

            deboost_attr_corr_faith = np.corrcoef(presence['deboost_attribution_deltas'], presence['deboost_deltas_faith'])[0, 1]
            row['deboost_attribution_delta_correlation_faith'] = deboost_attr_corr_faith if not np.isnan(deboost_attr_corr_faith) else 0.0

            deboost_attr_corr_pixel = np.corrcoef(presence['deboost_attribution_deltas'], presence['deboost_deltas_pixel'])[0, 1]
            row['deboost_attribution_delta_correlation_pixel'] = deboost_attr_corr_pixel if not np.isnan(deboost_attr_corr_pixel) else 0.0
        else:
            row['deboost_mean_attribution_delta'] = 0.0
            row['deboost_attribution_delta_correlation_saco'] = 0.0
            row['deboost_attribution_delta_correlation_faith'] = 0.0
            row['deboost_attribution_delta_correlation_pixel'] = 0.0

        if len(presence['deboost_gates']) > 1:
            deboost_corr_pixel = np.corrcoef(presence['deboost_gates'], presence['deboost_deltas_pixel'])[0, 1]
            row['deboost_impact_gate_correlation_pixel'] = deboost_corr_pixel if not np.isnan(deboost_corr_pixel) else 0.0
        else:
            row['deboost_impact_gate_correlation_pixel'] = 0.0

        # When boosted: what was the improvement?
        if n_boosted > 0:
            boosted_imgs = presence['images_with_boosting']
            boosted_deltas_list = [deltas_dict[img_idx] for img_idx in boosted_imgs if img_idx in deltas_dict]
            boosted_deltas = pd.DataFrame(boosted_deltas_list)

            row['boost_avg_gate'] = np.mean(presence['boost_magnitudes'])
            row['boost_delta_saco_mean'] = boosted_deltas['delta_saco'].mean()
            row['boost_delta_saco_std'] = boosted_deltas['delta_saco'].std()
            row['boost_delta_faith_mean'] = boosted_deltas['delta_faith'].mean()
            row['boost_delta_faith_std'] = boosted_deltas['delta_faith'].std()
            row['boost_delta_pixel_mean'] = boosted_deltas['delta_pixel'].mean()
            row['boost_delta_pixel_std'] = boosted_deltas['delta_pixel'].std()

            # How consistent is the improvement? (% positive deltas)
            row['boost_saco_improvement_rate'] = (boosted_deltas['delta_saco'] > 0).mean()
            row['boost_faith_improvement_rate'] = (boosted_deltas['delta_faith'] > 0).mean()
            row['boost_pixel_improvement_rate'] = (boosted_deltas['delta_pixel'] > 0).mean()

            # Combined consistency: did all 3 metrics improve on average?
            metrics_improved = sum([
                row['boost_delta_saco_mean'] > 0, row['boost_delta_faith_mean'] > 0, row['boost_delta_pixel_mean'] > 0
            ])
            row['boost_metrics_improved'] = metrics_improved

            # Per-class statistics
            per_class_stats = compute_per_class_statistics(boosted_deltas, 'boost')
            row.update(per_class_stats)
        else:
            row.update({
                'boost_avg_gate': np.nan,
                'boost_delta_saco_mean': np.nan,
                'boost_delta_saco_std': np.nan,
                'boost_delta_faith_mean': np.nan,
                'boost_delta_faith_std': np.nan,
                'boost_delta_pixel_mean': np.nan,
                'boost_delta_pixel_std': np.nan,
                'boost_saco_improvement_rate': np.nan,
                'boost_faith_improvement_rate': np.nan,
                'boost_pixel_improvement_rate': np.nan,
                'boost_metrics_improved': 0,
            })

        # When deboosted: what was the improvement?
        if n_deboosted > 0:
            deboosted_imgs = presence['images_with_deboosting']
            deboosted_deltas_list = [deltas_dict[img_idx] for img_idx in deboosted_imgs if img_idx in deltas_dict]
            deboosted_deltas = pd.DataFrame(deboosted_deltas_list)

            row['deboost_avg_gate'] = np.mean(presence['deboost_magnitudes'])
            row['deboost_delta_saco_mean'] = deboosted_deltas['delta_saco'].mean()
            row['deboost_delta_saco_std'] = deboosted_deltas['delta_saco'].std()
            row['deboost_delta_faith_mean'] = deboosted_deltas['delta_faith'].mean()
            row['deboost_delta_faith_std'] = deboosted_deltas['delta_faith'].std()
            row['deboost_delta_pixel_mean'] = deboosted_deltas['delta_pixel'].mean()
            row['deboost_delta_pixel_std'] = deboosted_deltas['delta_pixel'].std()

            row['deboost_saco_improvement_rate'] = (deboosted_deltas['delta_saco'] > 0).mean()
            row['deboost_faith_improvement_rate'] = (deboosted_deltas['delta_faith'] > 0).mean()
            row['deboost_pixel_improvement_rate'] = (deboosted_deltas['delta_pixel'] > 0).mean()

            metrics_improved = sum([
                row['deboost_delta_saco_mean'] > 0, row['deboost_delta_faith_mean'] > 0, row['deboost_delta_pixel_mean']
                > 0
            ])
            row['deboost_metrics_improved'] = metrics_improved

            # Per-class statistics
            per_class_stats = compute_per_class_statistics(deboosted_deltas, 'deboost')
            row.update(per_class_stats)
        else:
            row.update({
                'deboost_avg_gate': np.nan,
                'deboost_delta_saco_mean': np.nan,
                'deboost_delta_saco_std': np.nan,
                'deboost_delta_faith_mean': np.nan,
                'deboost_delta_faith_std': np.nan,
                'deboost_delta_pixel_mean': np.nan,
                'deboost_delta_pixel_std': np.nan,
                'deboost_saco_improvement_rate': np.nan,
                'deboost_faith_improvement_rate': np.nan,
                'deboost_pixel_improvement_rate': np.nan,
                'deboost_metrics_improved': 0,
            })

        impact_rows.append(row)

    impact_df = pd.DataFrame(impact_rows)

    if impact_df.empty:
        return impact_df

    # Add a combined "impact score" for sorting
    # Features that improve all 3 metrics consistently and frequently
    impact_df['boost_total_impact'] = (
        impact_df['boost_delta_saco_mean'].fillna(0) * (impact_df['boost_metrics_improved'] / 3.0) *
        np.log1p(impact_df['n_boosted'])  # Log-scale frequency bonus
    )

    impact_df['deboost_total_impact'] = (
        impact_df['deboost_delta_saco_mean'].fillna(0) * (impact_df['deboost_metrics_improved'] / 3.0) *
        np.log1p(impact_df['n_deboosted'])
    )

    print(f"  Found {len(impact_df)} features with >= {min_occurrences} occurrences")

    # Compute spatial entropy for each feature
    if compute_spatial and not impact_df.empty:
        print(f"  Computing spatial entropy for {len(impact_df)} features...")

        # Build reverse index: feature_idx -> list of patch_idx (much faster!)
        from collections import defaultdict
        feature_to_patches = defaultdict(list)

        print(f"    Building feature->patch index...")
        for idx in df.index:
            row = df.loc[idx]
            patch_idx = row['patch_idx']
            for feat_idx in row['active_features']:
                feature_to_patches[feat_idx].append(patch_idx)

        print(f"    Computing entropy for {len(impact_df)} features...")
        spatial_results = []
        for feat_idx in impact_df['feature_idx']:
            patch_activations = feature_to_patches[feat_idx]

            if not patch_activations:
                spatial_results.append({
                    'feature_idx': feat_idx,
                    'spatial_entropy': np.nan,
                    'spatial_entropy_raw': np.nan,
                    'n_unique_patches': 0,
                    'top_patches': []
                })
                continue

            # Compute spatial distribution
            patch_counts = np.bincount(patch_activations, minlength=196)
            patch_probs = patch_counts / patch_counts.sum()

            # Spatial entropy
            entropy = -np.sum(patch_probs[patch_probs > 0] * np.log(patch_probs[patch_probs > 0] + 1e-10))
            max_entropy = np.log(196)
            normalized_entropy = entropy / max_entropy

            # Top patches
            top_patches = np.argsort(patch_counts)[-10:][::-1]

            spatial_results.append({
                'feature_idx': feat_idx,
                'spatial_entropy': normalized_entropy,
                'spatial_entropy_raw': entropy,
                'n_unique_patches': np.sum(patch_counts > 0),
                'top_patches': top_patches[:5].tolist()
            })

        spatial_df = pd.DataFrame(spatial_results)
        impact_df = impact_df.merge(spatial_df, on='feature_idx', how='left')

        # Add interpretability score: high impact + low entropy + consistency
        # Use 1/(entropy+0.1) to avoid division by zero, reward low entropy
        impact_df['boost_interpretability_score'] = (
            impact_df['boost_total_impact'] * (1.0 / (impact_df['spatial_entropy'] + 0.1))  # Reward spatial consistency
        )

        impact_df['deboost_interpretability_score'] = (
            impact_df['deboost_total_impact'] * (1.0 / (impact_df['spatial_entropy'] + 0.1))
        )

        print(
            f"  Spatial entropy computed. Range: [{impact_df['spatial_entropy'].min():.3f}, {impact_df['spatial_entropy'].max():.3f}]"
        )

    return impact_df




def find_spatial_outliers(
    impact_df: pd.DataFrame,
    layer_idx: int,
    classes: List[str],
    top_n: int = 10,
    merged_data: Optional[Dict[int, pd.DataFrame]] = None,
    gated_attr_dir: Optional[Path] = None
) -> None:
    """
    Find features with extreme spatial patterns (corners, edges, highly localized, etc.)

    Args:
        impact_df: Feature impact DataFrame
        layer_idx: Layer number
        classes: List of class names
        top_n: Number of features to show per outlier type
        merged_data: Merged feature data (for loading attribution values)
        gated_attr_dir: Directory with gated attribution .npy files (for sorting by attribution)
    """
    # Filter features with spatial data
    spatial_df = impact_df[impact_df['spatial_mean_x'].notna()].copy()

    # Compute mean attribution values for sorting if attribution directory provided
    sort_key = 'boost_mean_attribution_delta' if 'boost_mean_attribution_delta' in spatial_df.columns else 'n_total_occurrences'  # Sort by CAM delta impact

    if len(spatial_df) == 0:
        print("\nNo spatial data available")
        return

    print("\n=== SPATIAL OUTLIERS ===")
    print(f"(Sorting by: {sort_key})")

    # 1. Corner features (all 4 corners)
    print(f"\n--- Top-Left Corner Features (artifact candidates) ---")
    corner_tl = spatial_df[(spatial_df['spatial_mean_x'] < 0.15) & (spatial_df['spatial_mean_y'] < 0.15)]
    corner_tl = corner_tl.sort_values(sort_key, ascending=False).head(top_n)
    for _, row in corner_tl.iterrows():
        print(f"  Feature {int(row['feature_idx'])}: pos=({row['spatial_mean_x']:.2f}, {row['spatial_mean_y']:.2f}), "
              f"freq={int(row['n_total_occurrences'])}")
        if 'mean_gated_attribution' in row:
            print(f"    Mean Attr: {row['mean_gated_attribution']:.4f}")
        print(f"    ΔSaCo: {row['boost_delta_saco_mean']:+.3f}, "
              f"ΔFaith: {row['boost_delta_faith_mean']:+.3f}, "
              f"ΔPixel: {row['boost_delta_pixel_mean']:+.3f}")

    print(f"\n--- Top-Right Corner Features ---")
    corner_tr = spatial_df[(spatial_df['spatial_mean_x'] > 0.85) & (spatial_df['spatial_mean_y'] < 0.15)]
    corner_tr = corner_tr.sort_values(sort_key, ascending=False).head(top_n)
    for _, row in corner_tr.iterrows():
        print(f"  Feature {int(row['feature_idx'])}: pos=({row['spatial_mean_x']:.2f}, {row['spatial_mean_y']:.2f}), "
              f"freq={int(row['n_total_occurrences'])}")
        if 'mean_gated_attribution' in row:
            print(f"    Mean Attr: {row['mean_gated_attribution']:.4f}")
        print(f"    ΔSaCo: {row['boost_delta_saco_mean']:+.3f}, "
              f"ΔFaith: {row['boost_delta_faith_mean']:+.3f}, "
              f"ΔPixel: {row['boost_delta_pixel_mean']:+.3f}")

    print(f"\n--- Bottom-Left Corner Features ---")
    corner_bl = spatial_df[(spatial_df['spatial_mean_x'] < 0.15) & (spatial_df['spatial_mean_y'] > 0.85)]
    corner_bl = corner_bl.sort_values(sort_key, ascending=False).head(top_n)
    for _, row in corner_bl.iterrows():
        print(f"  Feature {int(row['feature_idx'])}: pos=({row['spatial_mean_x']:.2f}, {row['spatial_mean_y']:.2f}), "
              f"freq={int(row['n_total_occurrences'])}")
        if 'mean_gated_attribution' in row:
            print(f"    Mean Attr: {row['mean_gated_attribution']:.4f}")
        print(f"    ΔSaCo: {row['boost_delta_saco_mean']:+.3f}, "
              f"ΔFaith: {row['boost_delta_faith_mean']:+.3f}, "
              f"ΔPixel: {row['boost_delta_pixel_mean']:+.3f}")

    print(f"\n--- Bottom-Right Corner Features ---")
    corner_br = spatial_df[(spatial_df['spatial_mean_x'] > 0.85) & (spatial_df['spatial_mean_y'] > 0.85)]
    corner_br = corner_br.sort_values(sort_key, ascending=False).head(top_n)
    for _, row in corner_br.iterrows():
        print(f"  Feature {int(row['feature_idx'])}: pos=({row['spatial_mean_x']:.2f}, {row['spatial_mean_y']:.2f}), "
              f"freq={int(row['n_total_occurrences'])}")
        if 'mean_gated_attribution' in row:
            print(f"    Mean Attr: {row['mean_gated_attribution']:.4f}")
        print(f"    ΔSaCo: {row['boost_delta_saco_mean']:+.3f}, "
              f"ΔFaith: {row['boost_delta_faith_mean']:+.3f}, "
              f"ΔPixel: {row['boost_delta_pixel_mean']:+.3f}")

    # 2. Highly localized features (low spatial std)
    print(f"\n--- Most Localized Features (low spatial_std) ---")
    spatial_df['spatial_spread'] = np.sqrt(spatial_df['spatial_std_x']**2 + spatial_df['spatial_std_y']**2)
    localized = spatial_df.nsmallest(top_n, 'spatial_spread')
    for _, row in localized.iterrows():
        print(f"  Feature {int(row['feature_idx'])}: pos=({row['spatial_mean_x']:.2f}, {row['spatial_mean_y']:.2f}), "
              f"spread={row['spatial_spread']:.3f}, freq={int(row['n_total_occurrences'])}")
        print(f"    ΔSaCo: {row['boost_delta_saco_mean']:+.3f}, "
              f"ΔFaith: {row['boost_delta_faith_mean']:+.3f}, "
              f"ΔPixel: {row['boost_delta_pixel_mean']:+.3f}")

    # 3. Highly diffuse features (high spatial std)
    print(f"\n--- Most Diffuse Features (high spatial_std) ---")
    diffuse = spatial_df.nlargest(top_n, 'spatial_spread')
    for _, row in diffuse.iterrows():
        print(f"  Feature {int(row['feature_idx'])}: pos=({row['spatial_mean_x']:.2f}, {row['spatial_mean_y']:.2f}), "
              f"spread={row['spatial_spread']:.3f}, freq={int(row['n_total_occurrences'])}")
        print(f"    ΔSaCo: {row['boost_delta_saco_mean']:+.3f}, "
              f"ΔFaith: {row['boost_delta_faith_mean']:+.3f}, "
              f"ΔPixel: {row['boost_delta_pixel_mean']:+.3f}")


def find_functional_outliers(impact_df: pd.DataFrame, layer_idx: int, classes: List[str], top_n: int = 10) -> None:
    """
    Find features with extreme functional properties (asymmetry, negative impact, class-reversal)

    Args:
        impact_df: Feature impact DataFrame
        layer_idx: Layer number
        classes: List of class names
        top_n: Number of features to show per outlier type
    """
    print("\n=== FUNCTIONAL OUTLIERS ===")

    # Compute derived metrics
    impact_df = impact_df.copy()
    impact_df['saco_asymmetry'] = impact_df['boost_delta_saco_mean'].fillna(0) - impact_df['deboost_delta_saco_mean'].fillna(0)
    impact_df['faith_asymmetry'] = impact_df['boost_delta_faith_mean'].fillna(0) - impact_df['deboost_delta_faith_mean'].fillna(0)

    # 1. High positive asymmetry (boost >> deboost)
    print(f"\n--- Highest Positive Asymmetry (presence helps more than absence) ---")
    pos_asym = impact_df.nlargest(top_n, 'saco_asymmetry')
    for _, row in pos_asym.iterrows():
        print(f"  Feature {int(row['feature_idx'])}: boost={row['boost_delta_saco_mean']:.3f}, "
              f"deboost={row['deboost_delta_saco_mean']:.3f}, "
              f"asym={row['saco_asymmetry']:.3f}, freq={int(row['n_total_occurrences'])}")

    # 2. High negative asymmetry (deboost >> boost)
    print(f"\n--- Highest Negative Asymmetry (absence helps more than presence) ---")
    neg_asym = impact_df.nsmallest(top_n, 'saco_asymmetry')
    for _, row in neg_asym.iterrows():
        print(f"  Feature {int(row['feature_idx'])}: boost={row['boost_delta_saco_mean']:.3f}, "
              f"deboost={row['deboost_delta_saco_mean']:.3f}, "
              f"asym={row['saco_asymmetry']:.3f}, freq={int(row['n_total_occurrences'])}")

    # 3. Negative impact features (hurt when boosted)
    print(f"\n--- Negative Impact Features (hurt when boosted) ---")
    negative = impact_df[impact_df['boost_delta_saco_mean'] < -0.01].copy()
    if len(negative) > 0:
        negative = negative.nsmallest(top_n, 'boost_delta_saco_mean')
        for _, row in negative.iterrows():
            print(f"  Feature {int(row['feature_idx'])}: boost_saco={row['boost_delta_saco_mean']:.3f}, "
                  f"boost_faith={row['boost_delta_faith_mean']:.3f}, freq={int(row['n_total_occurrences'])}")
    else:
        print("  (None found)")

    # 4. Class-reversal features (help Normal but hurt COVID)
    print(f"\n--- Class-Reversal Features (help Normal, hurt COVID) ---")
    covid_col = f'boost_delta_saco_mean_{classes[0]}'  # COVID-19
    normal_col = f'boost_delta_saco_mean_{classes[2]}'  # Normal

    if covid_col in impact_df.columns and normal_col in impact_df.columns:
        reversal = impact_df[(impact_df[normal_col] > 0.05) & (impact_df[covid_col] < 0)].copy()
        if len(reversal) > 0:
            reversal['reversal_strength'] = reversal[normal_col] - reversal[covid_col]
            reversal = reversal.nlargest(top_n, 'reversal_strength')
            for _, row in reversal.iterrows():
                print(f"  Feature {int(row['feature_idx'])}: COVID={row[covid_col]:.3f}, "
                      f"Normal={row[normal_col]:.3f}, freq={int(row['n_total_occurrences'])}")
        else:
            print("  (None found)")
    else:
        print("  (Per-class data not available)")

    # 5. High class-specificity (large variance across classes)
    print(f"\n--- Most Class-Specific Features (high variance) ---")
    var_col = 'boost_saco_variance_across_classes'
    if var_col in impact_df.columns:
        specific = impact_df.nlargest(top_n, var_col)
        for _, row in specific.iterrows():
            print(f"  Feature {int(row['feature_idx'])}: variance={row[var_col]:.4f}, "
                  f"COVID={row[covid_col]:.3f}, Normal={row[normal_col]:.3f}, "
                  f"freq={int(row['n_total_occurrences'])}")
    else:
        print("  (Variance data not available)")

    # 6. Strong gate correlation (synergistic features)
    print(f"\n--- Strongest Gate Correlation - SaCo (synergistic features) ---")
    if 'boost_impact_gate_correlation' in impact_df.columns:
        # Take features with high |correlation| and sufficient frequency
        freq_features = impact_df[impact_df['n_total_occurrences'] >= 100].copy()
        freq_features['abs_corr'] = freq_features['boost_impact_gate_correlation'].abs()
        correlated = freq_features.nlargest(top_n, 'abs_corr')
        for _, row in correlated.iterrows():
            print(f"  Feature {int(row['feature_idx'])}: corr={row['boost_impact_gate_correlation']:.3f}, "
                  f"boost_saco={row['boost_delta_saco_mean']:.3f}, freq={int(row['n_total_occurrences'])}")
    else:
        print("  (Correlation data not available)")

    # 7. Strong gate correlation - Faithfulness
    print(f"\n--- Strongest Gate Correlation - Faithfulness (synergistic features) ---")
    if 'boost_impact_gate_correlation_faith' in impact_df.columns:
        freq_features = impact_df[impact_df['n_total_occurrences'] >= 100].copy()
        freq_features['abs_corr_faith'] = freq_features['boost_impact_gate_correlation_faith'].abs()
        correlated_faith = freq_features.nlargest(top_n, 'abs_corr_faith')
        for _, row in correlated_faith.iterrows():
            print(f"  Feature {int(row['feature_idx'])}: corr={row['boost_impact_gate_correlation_faith']:.3f}, "
                  f"boost_faith={row['boost_delta_faith_mean']:.3f}, freq={int(row['n_total_occurrences'])}")
    else:
        print("  (Correlation data not available)")

    # 8. Strong gate correlation - Pixel Flipping
    print(f"\n--- Strongest Gate Correlation - Pixel Flipping (synergistic features) ---")
    if 'boost_impact_gate_correlation_pixel' in impact_df.columns:
        freq_features = impact_df[impact_df['n_total_occurrences'] >= 100].copy()
        freq_features['abs_corr_pixel'] = freq_features['boost_impact_gate_correlation_pixel'].abs()
        correlated_pixel = freq_features.nlargest(top_n, 'abs_corr_pixel')
        for _, row in correlated_pixel.iterrows():
            print(f"  Feature {int(row['feature_idx'])}: corr={row['boost_impact_gate_correlation_pixel']:.3f}, "
                  f"boost_pixel={row['boost_delta_pixel_mean']:.3f}, freq={int(row['n_total_occurrences'])}")
    else:
        print("  (Correlation data not available)")

    # 9. Largest attribution delta magnitude (features that actually change CAM values)
    print(f"\n--- Largest Attribution Delta Magnitude (features that move CAM the most) ---")
    if 'boost_mean_attribution_delta' in impact_df.columns:
        freq_features = impact_df[impact_df['n_total_occurrences'] >= 50].copy()
        largest_delta = freq_features.nlargest(top_n, 'boost_mean_attribution_delta')
        for _, row in largest_delta.iterrows():
            print(f"  Feature {int(row['feature_idx'])}: attr_delta={row['boost_mean_attribution_delta']:.4f}, "
                  f"gate={row['boost_avg_gate']:.2f}, boost_saco={row['boost_delta_saco_mean']:.3f}, "
                  f"freq={int(row['n_total_occurrences'])}")
    else:
        print("  (Attribution delta data not available)")

    # 10. Strong attribution-delta correlation - SaCo
    print(f"\n--- Strongest Attribution-Delta Correlation - SaCo (CAM change predicts improvement) ---")
    if 'boost_attribution_delta_correlation_saco' in impact_df.columns:
        freq_features = impact_df[impact_df['n_total_occurrences'] >= 100].copy()
        freq_features['abs_attr_corr'] = freq_features['boost_attribution_delta_correlation_saco'].abs()
        attr_correlated = freq_features.nlargest(top_n, 'abs_attr_corr')
        for _, row in attr_correlated.iterrows():
            print(f"  Feature {int(row['feature_idx'])}: corr={row['boost_attribution_delta_correlation_saco']:.3f}, "
                  f"mean_delta={row['boost_mean_attribution_delta']:.4f}, "
                  f"boost_saco={row['boost_delta_saco_mean']:.3f}, freq={int(row['n_total_occurrences'])}")
    else:
        print("  (Attribution delta correlation data not available)")

    # 11. Strong attribution-delta correlation - Faithfulness
    print(f"\n--- Strongest Attribution-Delta Correlation - Faithfulness ---")
    if 'boost_attribution_delta_correlation_faith' in impact_df.columns:
        freq_features = impact_df[impact_df['n_total_occurrences'] >= 100].copy()
        freq_features['abs_attr_corr_faith'] = freq_features['boost_attribution_delta_correlation_faith'].abs()
        attr_correlated_faith = freq_features.nlargest(top_n, 'abs_attr_corr_faith')
        for _, row in attr_correlated_faith.iterrows():
            print(f"  Feature {int(row['feature_idx'])}: corr={row['boost_attribution_delta_correlation_faith']:.3f}, "
                  f"mean_delta={row['boost_mean_attribution_delta']:.4f}, "
                  f"boost_faith={row['boost_delta_faith_mean']:.3f}, freq={int(row['n_total_occurrences'])}")
    else:
        print("  (Attribution delta correlation data not available)")

    # 12. Strong attribution-delta correlation - Pixel
    print(f"\n--- Strongest Attribution-Delta Correlation - Pixel Flipping ---")
    if 'boost_attribution_delta_correlation_pixel' in impact_df.columns:
        freq_features = impact_df[impact_df['n_total_occurrences'] >= 100].copy()
        freq_features['abs_attr_corr_pixel'] = freq_features['boost_attribution_delta_correlation_pixel'].abs()
        attr_correlated_pixel = freq_features.nlargest(top_n, 'abs_attr_corr_pixel')
        for _, row in attr_correlated_pixel.iterrows():
            print(f"  Feature {int(row['feature_idx'])}: corr={row['boost_attribution_delta_correlation_pixel']:.3f}, "
                  f"mean_delta={row['boost_mean_attribution_delta']:.4f}, "
                  f"boost_pixel={row['boost_delta_pixel_mean']:.3f}, freq={int(row['n_total_occurrences'])}")
    else:
        print("  (Attribution delta correlation data not available)")


def visualize_spatial_outliers(
    impact_df: pd.DataFrame,
    layer_idx: int,
    merged_data: Dict[int, pd.DataFrame],
    faithfulness_vanilla: pd.DataFrame,
    faithfulness_gated: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    vanilla_attr_dir: Optional[Path] = None,
    gated_attr_dir: Optional[Path] = None,
    top_n_per_category: int = 3,
    top_k_images: int = 50
) -> None:
    """
    Generate visualizations for top features in each spatial category.

    Args:
        impact_df: Feature impact DataFrame for this layer
        layer_idx: Layer number
        merged_data: Merged feature data (all layers)
        faithfulness_vanilla: Vanilla faithfulness results
        faithfulness_gated: Gated faithfulness results
        image_dir: Directory containing original images
        vanilla_attr_dir: Directory containing vanilla attribution .npy files
        gated_attr_dir: Directory containing gated attribution .npy files
        output_dir: Base output directory
        top_n_per_category: Number of features to visualize per category
        top_k_images: Number of images to visualize per feature
    """
    # Filter features with spatial data
    spatial_df = impact_df[impact_df['spatial_mean_x'].notna()].copy()

    if len(spatial_df) == 0:
        print("\nNo spatial data available for visualization")
        return

    # Create layer output directory
    layer_dir = output_dir / f"layer_{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== GENERATING VISUALIZATIONS FOR LAYER {layer_idx} ===")

    # Define categories and extract top features
    categories = {}

    # Localized and diffuse
    spatial_df['spatial_spread'] = np.sqrt(spatial_df['spatial_std_x']**2 + spatial_df['spatial_std_y']**2)
    categories['most_localized'] = spatial_df.nsmallest(top_n_per_category, 'spatial_spread')
    # categories['most_diffuse'] = spatial_df.nlargest(top_n_per_category, 'spatial_spread')

    # Corners
    categories['top_left_corner'] = spatial_df[
        (spatial_df['spatial_mean_x'] < 0.15) & (spatial_df['spatial_mean_y'] < 0.15)
    ].sort_values('n_total_occurrences', ascending=False).head(top_n_per_category)

    categories['top_right_corner'] = spatial_df[
        (spatial_df['spatial_mean_x'] > 0.85) & (spatial_df['spatial_mean_y'] < 0.15)
    ].sort_values('n_total_occurrences', ascending=False).head(top_n_per_category)

    categories['bottom_left_corner'] = spatial_df[
        (spatial_df['spatial_mean_x'] < 0.15) & (spatial_df['spatial_mean_y'] > 0.85)
    ].sort_values('n_total_occurrences', ascending=False).head(top_n_per_category)

    categories['bottom_right_corner'] = spatial_df[
        (spatial_df['spatial_mean_x'] > 0.85) & (spatial_df['spatial_mean_y'] > 0.85)
    ].sort_values('n_total_occurrences', ascending=False).head(top_n_per_category)


    # Visualize each category
    for category_name, category_df in categories.items():
        if len(category_df) == 0:
            continue

        print(f"\n{category_name.replace('_', ' ').title()}: {len(category_df)} features")
        category_dir = layer_dir / category_name
        category_dir.mkdir(exist_ok=True)

        for _, row in category_df.iterrows():
            feature_idx = int(row['feature_idx'])

            # Get class distribution for this feature
            df = merged_data[layer_idx]
            feature_images = []
            for idx in df.index:
                df_row = df.loc[idx]
                if feature_idx in df_row['active_features']:
                    feature_images.append({
                        'image_idx': df_row['image_idx'],
                        'true_class': df_row['true_class']
                    })

            if feature_images:
                feature_images_df = pd.DataFrame(feature_images).drop_duplicates('image_idx')
                class_counts = feature_images_df['true_class'].value_counts()
                class_dist_str = ', '.join([f"{cls}: {count}" for cls, count in class_counts.items()])
                print(f"  Feature {feature_idx} - Class distribution: {class_dist_str}")
            else:
                print(f"  Feature {feature_idx} - No class data")

            feature_dir = category_dir / f"feature_{feature_idx}"
            feature_dir.mkdir(exist_ok=True)

            try:
                visualize_feature_activations(
                    feature_idx=feature_idx,
                    layer_idx=layer_idx,
                    merged_data=merged_data,
                    faithfulness_vanilla=faithfulness_vanilla,
                    faithfulness_gated=faithfulness_gated,
                    image_dir=image_dir,
                    output_dir=feature_dir,
                    vanilla_attr_dir=vanilla_attr_dir,
                    gated_attr_dir=gated_attr_dir,
                    top_k=top_k_images,
                    sort_by='attr_diff',
                    sort_ascending=False
                )
                print(f"    ✓ Saved {top_k_images} images")
            except Exception as e:
                print(f"    ✗ Error: {e}")

    print(f"\nVisualizations saved to: {layer_dir}")


def visualize_top_correlation_features(
    impact_df: pd.DataFrame,
    layer_idx: int,
    merged_data: Dict[int, pd.DataFrame],
    faithfulness_vanilla: pd.DataFrame,
    faithfulness_gated: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    vanilla_attr_dir: Optional[Path] = None,
    gated_attr_dir: Optional[Path] = None,
    correlation_type: str = 'boost_impact_gate_correlation',
    metric_name: str = 'SaCo',
    top_n: int = 3,
    top_k_images: int = 10,
) -> None:
    """
    Visualize the top N features with strongest gate-impact correlation.

    This helps understand which features show the strongest relationship between
    gate strength and faithfulness improvement.

    Args:
        impact_df: Feature impact DataFrame from compute_feature_impact_discovery
        layer_idx: Layer number
        merged_data: Merged feature data
        faithfulness_vanilla: Vanilla faithfulness results
        faithfulness_gated: Gated faithfulness results
        image_dir: Directory containing original images
        output_dir: Where to save visualizations
        vanilla_attr_dir: Directory with vanilla attribution .npy files
        gated_attr_dir: Directory with gated attribution .npy files
        correlation_type: Which correlation column to use (e.g., 'boost_impact_gate_correlation',
                         'boost_impact_gate_correlation_faith', 'boost_impact_gate_correlation_pixel',
                         'deboost_impact_gate_correlation', etc.)
        metric_name: Name for output directory (e.g., 'SaCo', 'Faith', 'Pixel')
        top_n: Number of top features to visualize
        top_k_images: Number of images to show per feature
    """
    output_dir = Path(output_dir)

    # Filter out features without correlation data
    valid_df = impact_df[impact_df[correlation_type].notna()].copy()

    if len(valid_df) == 0:
        print(f"\nNo features with {correlation_type} data found")
        return

    # Sort by absolute correlation strength (we care about strength, not just direction)
    valid_df['abs_correlation'] = valid_df[correlation_type].abs()
    top_features = valid_df.nlargest(top_n, 'abs_correlation')

    print(f"\n{'='*80}")
    print(f"TOP {top_n} FEATURES BY {correlation_type.upper()} (Layer {layer_idx})")
    print(f"{'='*80}\n")

    # Determine if this is boost or deboost
    is_boost = 'boost' in correlation_type
    boost_str = 'Boost' if is_boost else 'Deboost'

    for rank, (_, row) in enumerate(top_features.iterrows(), 1):
        feature_idx = int(row['feature_idx'])
        correlation = row[correlation_type]

        print(f"\n{rank}. Feature {feature_idx}")
        print(f"   Correlation: {correlation:+.4f}")
        print(f"   Occurrences: {int(row['n_total_occurrences'])}")

        # Show relevant impact metrics based on boost/deboost
        if is_boost:
            print(f"   {boost_str} avg gate: {row['boost_avg_gate']:.3f}")
            print(f"   ΔSaCo: {row['boost_delta_saco_mean']:+.4f}")
            print(f"   ΔFaith: {row['boost_delta_faith_mean']:+.4f}")
            print(f"   ΔPixel: {row['boost_delta_pixel_mean']:+.4f}")
        else:
            print(f"   {boost_str} avg gate: {row['deboost_avg_gate']:.3f}")
            print(f"   ΔSaCo: {row['deboost_delta_saco_mean']:+.4f}")
            print(f"   ΔFaith: {row['deboost_delta_faith_mean']:+.4f}")
            print(f"   ΔPixel: {row['deboost_delta_pixel_mean']:+.4f}")

        # Spatial information if available
        if 'spatial_mean_x' in row and not pd.isna(row['spatial_mean_x']):
            print(f"   Spatial location: ({row['spatial_mean_x']:.2f}, {row['spatial_mean_y']:.2f})")
            print(f"   Spatial spread: ({row['spatial_std_x']:.2f}, {row['spatial_std_y']:.2f})")

        # Create output directory for this feature
        feature_dir = output_dir / f"layer_{layer_idx}_{metric_name}_correlation" / f"rank_{rank}_feature_{feature_idx}"
        feature_dir.mkdir(parents=True, exist_ok=True)

        # Visualize this feature
        try:
            visualize_feature_activations(
                feature_idx=feature_idx,
                layer_idx=layer_idx,
                merged_data=merged_data,
                faithfulness_vanilla=faithfulness_vanilla,
                faithfulness_gated=faithfulness_gated,
                image_dir=image_dir,
                output_dir=feature_dir,
                vanilla_attr_dir=vanilla_attr_dir,
                gated_attr_dir=gated_attr_dir,
                top_k=top_k_images,
                sort_by='gate',  # Sort by gate strength to see the correlation pattern
                sort_ascending=not is_boost  # For boost: high gates first; for deboost: low gates first
            )
            print(f"   ✓ Saved {top_k_images} visualizations to {feature_dir}")
        except Exception as e:
            print(f"   ✗ Error visualizing feature {feature_idx}: {e}")

    print(f"\n{'='*80}")
    print(f"Visualizations saved to: {output_dir / f'layer_{layer_idx}_{metric_name}_correlation'}")
    print(f"{'='*80}\n")



# =========================================================================
# FOCUSED ATTRIBUTION CHANGE ANALYSIS
# =========================================================================

def analyze_attribution_change_per_layer(merged_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Q1: How much do features change attribution on average per layer?

    Measures the mean absolute attribution delta across all patches per layer.

    Args:
        merged_data: Output from merge_features_faithfulness

    Returns:
        DataFrame with columns: layer_idx, mean_abs_delta, std_abs_delta, n_patches
    """
    results = []

    for layer_idx, df in merged_data.items():
        if 'attribution_delta' not in df.columns:
            print(f"Warning: Layer {layer_idx} missing attribution_delta column")
            continue

        # Filter out NaN values
        deltas = df['attribution_delta'].dropna()

        if len(deltas) == 0:
            continue

        results.append({
            'layer_idx': layer_idx,
            'mean_abs_delta': deltas.abs().mean(),
            'std_abs_delta': deltas.abs().std(),
            'mean_delta': deltas.mean(),
            'std_delta': deltas.std(),
            'n_patches': len(deltas),
            'n_images': df['image_idx'].nunique()
        })

    return pd.DataFrame(results).sort_values('layer_idx')


def analyze_boost_deboost_magnitude(merged_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Q2: How much are features boosted vs deboosted on average?

    Separates patches by gate_value > 1 (boost) vs gate_value < 1 (deboost)
    and computes average attribution delta for each.

    Args:
        merged_data: Output from merge_features_faithfulness

    Returns:
        DataFrame with columns per layer:
            - boost_mean_delta, boost_std_delta, n_boosted_patches
            - deboost_mean_delta, deboost_std_delta, n_deboosted_patches
    """
    results = []

    for layer_idx, df in merged_data.items():
        if 'attribution_delta' not in df.columns or 'gate_value' not in df.columns:
            continue

        # Filter out NaN attribution deltas
        valid_df = df[df['attribution_delta'].notna()].copy()

        if len(valid_df) == 0:
            continue

        # Split by boost/deboost
        boosted = valid_df[valid_df['gate_value'] > 1.0]
        deboosted = valid_df[valid_df['gate_value'] < 1.0]
        neutral = valid_df[valid_df['gate_value'] == 1.0]

        result = {
            'layer_idx': layer_idx,
            'n_total_patches': len(valid_df),
        }

        # Boosted statistics
        if len(boosted) > 0:
            result['boost_mean_abs_delta'] = boosted['attribution_delta'].abs().mean()
            result['boost_std_abs_delta'] = boosted['attribution_delta'].abs().std()
            result['boost_mean_delta'] = boosted['attribution_delta'].mean()
            result['boost_std_delta'] = boosted['attribution_delta'].std()
            result['n_boosted_patches'] = len(boosted)
            result['pct_boosted'] = len(boosted) / len(valid_df) * 100
        else:
            result.update({
                'boost_mean_abs_delta': 0,
                'boost_std_abs_delta': 0,
                'boost_mean_delta': 0,
                'boost_std_delta': 0,
                'n_boosted_patches': 0,
                'pct_boosted': 0
            })

        # Deboosted statistics
        if len(deboosted) > 0:
            result['deboost_mean_abs_delta'] = deboosted['attribution_delta'].abs().mean()
            result['deboost_std_abs_delta'] = deboosted['attribution_delta'].abs().std()
            result['deboost_mean_delta'] = deboosted['attribution_delta'].mean()
            result['deboost_std_delta'] = deboosted['attribution_delta'].std()
            result['n_deboosted_patches'] = len(deboosted)
            result['pct_deboosted'] = len(deboosted) / len(valid_df) * 100
        else:
            result.update({
                'deboost_mean_abs_delta': 0,
                'deboost_std_abs_delta': 0,
                'deboost_mean_delta': 0,
                'deboost_std_delta': 0,
                'n_deboosted_patches': 0,
                'pct_deboosted': 0
            })

        # Neutral statistics
        if len(neutral) > 0:
            result['n_neutral_patches'] = len(neutral)
            result['pct_neutral'] = len(neutral) / len(valid_df) * 100
        else:
            result['n_neutral_patches'] = 0
            result['pct_neutral'] = 0

        results.append(result)

    return pd.DataFrame(results).sort_values('layer_idx')


def analyze_delta_distribution(
    merged_data: Dict[int, pd.DataFrame],
    layer_idx: int,
    n_bins: int = 50,
    plot: bool = True,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Q3: How is boosting/deboosting spread out?

    Creates frequency analysis of attribution delta values.

    Args:
        merged_data: Output from merge_features_faithfulness
        layer_idx: Which layer to analyze
        n_bins: Number of histogram bins
        plot: Whether to create a plot
        output_path: Where to save plot (if plot=True)

    Returns:
        Dictionary with histogram data and statistics
    """
    df = merged_data[layer_idx]

    if 'attribution_delta' not in df.columns:
        raise ValueError(f"Layer {layer_idx} missing attribution_delta column")

    deltas = df['attribution_delta'].dropna()

    if len(deltas) == 0:
        raise ValueError(f"No valid attribution deltas for layer {layer_idx}")

    # Compute histogram
    counts, bin_edges = np.histogram(deltas, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Additional statistics about tails
    pct_near_zero = ((deltas.abs() < 0.01).sum() / len(deltas)) * 100
    pct_large_pos = ((deltas > 0.1).sum() / len(deltas)) * 100
    pct_large_neg = ((deltas < -0.1).sum() / len(deltas)) * 100

    # Statistics
    stats = {
        'layer_idx': layer_idx,
        'mean': deltas.mean(),
        'median': deltas.median(),
        'std': deltas.std(),
        'min': deltas.min(),
        'max': deltas.max(),
        'q25': deltas.quantile(0.25),
        'q75': deltas.quantile(0.75),
        'q95': deltas.quantile(0.95),
        'q05': deltas.quantile(0.05),
        'n_patches': len(deltas),
        'pct_near_zero': pct_near_zero,
        'pct_large_positive': pct_large_pos,
        'pct_large_negative': pct_large_neg,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'counts': counts,
    }

    if plot:
        import matplotlib.pyplot as plt

        # Separate positive and negative deltas
        deltas_pos = deltas[deltas > 0]
        deltas_neg = deltas[deltas < 0]
        deltas_zero = deltas[deltas == 0]

        # Create figure with 2 subplots: linear and log scale
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: Linear scale with separate colors
        ax1.hist([deltas_neg, deltas_pos], bins=n_bins, alpha=0.7,
                color=['crimson', 'forestgreen'],
                label=['Deboosting (negative)', 'Boosting (positive)'],
                edgecolor='black', linewidth=0.5)
        ax1.axvline(0, color='black', linestyle='-', linewidth=2, label='No change', zorder=10)
        ax1.axvline(deltas.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {deltas.mean():.4f}')
        ax1.axvline(deltas.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {deltas.median():.4f}')

        ax1.set_xlabel('Attribution Delta (Signed)', fontsize=12)
        ax1.set_ylabel('Frequency (number of patches)', fontsize=12)
        ax1.set_title(f'Linear Scale - Layer {layer_idx}', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # More x-axis ticks
        from matplotlib.ticker import MaxNLocator
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=15))

        # Right plot: Log scale with separate colors
        ax2.hist([deltas_neg, deltas_pos], bins=n_bins, alpha=0.7,
                color=['crimson', 'forestgreen'],
                label=['Deboosting (negative)', 'Boosting (positive)'],
                edgecolor='black', linewidth=0.5)
        ax2.axvline(0, color='black', linestyle='-', linewidth=2, label='No change', zorder=10)
        ax2.axvline(deltas.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {deltas.mean():.4f}')
        ax2.axvline(deltas.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {deltas.median():.4f}')

        ax2.set_xlabel('Attribution Delta (Signed)', fontsize=12)
        ax2.set_ylabel('Frequency (log scale)', fontsize=12)
        ax2.set_title(f'Log Scale - Layer {layer_idx}', fontsize=14)
        ax2.set_yscale('log')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, which='both')

        # More x-axis ticks
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=15))

        # Add text box with tail statistics
        textstr = '\n'.join([
            f'Near zero (|Δ|<0.01): {pct_near_zero:.1f}%',
            f'Boosting (Δ>0.1): {pct_large_pos:.1f}%',
            f'Deboosting (Δ<-0.1): {pct_large_neg:.1f}%',
            f'',
            f'Total boosting: {len(deltas_pos)}/{len(deltas)} ({100*len(deltas_pos)/len(deltas):.1f}%)',
            f'Total deboosting: {len(deltas_neg)}/{len(deltas)} ({100*len(deltas_neg)/len(deltas):.1f}%)',
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved distribution plot to {output_path}")

        plt.close()

    return stats


def analyze_feature_contribution_correlation(
    merged_data: Dict[int, pd.DataFrame],
    layer_idx: int,
    min_occurrences: int = 100,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Q4 (NEW): Which features' contributions are most correlated with attribution impact?

    This is the KEY metric: for each feature, compute the correlation between:
    - Feature contribution (activation × gradient) when that feature is active
    - Patch attribution delta (how much that patch's attribution changed)

    High correlation = feature's presence/strength reliably predicts attribution change.

    Args:
        merged_data: Output from merge_features_faithfulness
        layer_idx: Which layer to analyze
        min_occurrences: Minimum occurrences to be considered
        top_n: How many top features to return (None = all)

    Returns:
        DataFrame with columns:
            - feature_idx
            - correlation: Pearson correlation between contribution and attribution_delta
            - p_value: Statistical significance of correlation
            - mean_contribution: Average contribution magnitude
            - std_contribution: Std of contribution
            - mean_attribution_delta: Average attribution_delta when feature is active
            - n_occurrences: Number of patches where feature appears
            - effect_size: correlation × sqrt(n_occurrences)  # Weighted by frequency
    """
    from collections import defaultdict

    from scipy import stats as scipy_stats

    df = merged_data[layer_idx]

    if 'attribution_delta' not in df.columns or 'feature_contributions' not in df.columns:
        raise ValueError(f"Layer {layer_idx} missing required columns")

    # Check if contribution data is available
    has_contributions = df['feature_contributions'].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False).any()
    if not has_contributions:
        print(f"Warning: No contribution data available for layer {layer_idx}")
        return pd.DataFrame()

    # Accumulate contributions and attribution deltas per feature
    feature_data = defaultdict(lambda: {
        'contributions': [],
        'attribution_deltas': [],
    })

    for idx in df.index:
        row = df.loc[idx]
        attr_delta = row.get('attribution_delta', np.nan)

        if np.isnan(attr_delta):
            continue

        active_features = row['active_features']
        feature_contribs = row['feature_contributions']

        if not isinstance(active_features, (list, np.ndarray)) or not isinstance(feature_contribs, (list, np.ndarray)):
            continue

        if len(active_features) == 0 or len(feature_contribs) == 0:
            continue

        # For each active feature in this patch
        for feat_idx, contrib in zip(active_features, feature_contribs):
            feature_data[feat_idx]['contributions'].append(contrib)
            feature_data[feat_idx]['attribution_deltas'].append(attr_delta)

    # Compute correlation for each feature
    results = []
    for feat_idx, data in feature_data.items():
        n_occur = len(data['contributions'])

        if n_occur < min_occurrences:
            continue

        contribs = np.array(data['contributions'])
        attr_deltas = np.array(data['attribution_deltas'])

        # Compute Pearson correlation
        if len(contribs) >= 3:  # Need at least 3 points for correlation
            corr, p_value = scipy_stats.pearsonr(contribs, attr_deltas)
        else:
            corr, p_value = np.nan, np.nan

        # Effect size: correlation weighted by frequency
        effect_size = corr * np.sqrt(n_occur) if not np.isnan(corr) else np.nan

        results.append({
            'feature_idx': feat_idx,
            'correlation': corr,
            'p_value': p_value,
            'mean_contribution': contribs.mean(),
            'std_contribution': contribs.std(),
            'mean_abs_contribution': np.abs(contribs).mean(),
            'mean_attribution_delta': attr_deltas.mean(),
            'std_attribution_delta': attr_deltas.std(),
            'n_occurrences': n_occur,
            'effect_size': effect_size,
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print(f"Warning: No features found with >= {min_occurrences} occurrences")
        return pd.DataFrame()

    # Sort by absolute effect size (highest correlation × frequency)
    results_df = results_df.sort_values('effect_size', ascending=False, key=abs)

    if top_n is not None:
        results_df = results_df.head(top_n)

    return results_df




# REMOVED: analyze_feature_faithfulness_correlation, analyze_variance_explained,
# and analyze_variance_explained_pooled - these used invalid correlation-based approaches
# that don't respect the causal mechanism. See new two-stage analysis below.


# ============================================================================
# TWO-STAGE ANALYSIS: Patch/Gate Impact → Feature Composition
# ============================================================================


def analyze_patch_impact_on_faithfulness(
    merged_data: Dict[int, pd.DataFrame],
    faithfulness_vanilla: pd.DataFrame,
    faithfulness_gated: pd.DataFrame,
    layer_idx: int,
    improvement_quantiles: Tuple[float, float] = (0.75, 0.25),
    patch_quantile: float = 0.80,
    min_images: int = 10
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
    from scipy import stats as scipy_stats

    df = merged_data[layer_idx]

    if df.empty:
        print(f"Warning: No data for layer {layer_idx}")
        return {}

    print(f"\n{'='*60}")
    print(f"STAGE 1: PATCH IMPACT ANALYSIS - Layer {layer_idx}")
    print(f"{'='*60}\n")

    # Compute faithfulness improvements per metric
    improvements = pd.DataFrame({
        'image_idx': range(len(faithfulness_vanilla)),
        'delta_saco': faithfulness_gated['saco_score'].values - faithfulness_vanilla['saco_score'].values,
        'delta_faith': faithfulness_gated['FaithfulnessCorrelation'].values - faithfulness_vanilla['FaithfulnessCorrelation'].values,
        'delta_pixel': faithfulness_gated['PixelFlipping'].values - faithfulness_vanilla['PixelFlipping'].values,
    })

    # Compute composite improvement score (average of standardized metrics)
    # Note: PixelFlipping - lower is better, so we negate it
    improvements['delta_saco_z'] = (improvements['delta_saco'] - improvements['delta_saco'].mean()) / improvements['delta_saco'].std()
    improvements['delta_faith_z'] = (improvements['delta_faith'] - improvements['delta_faith'].mean()) / improvements['delta_faith'].std()
    improvements['delta_pixel_z'] = -(improvements['delta_pixel'] - improvements['delta_pixel'].mean()) / improvements['delta_pixel'].std()
    improvements['composite_improvement'] = (improvements['delta_saco_z'] + improvements['delta_faith_z'] + improvements['delta_pixel_z']) / 3

    # Print distribution statistics
    print("Faithfulness Improvement Distribution:")
    for metric in ['delta_saco', 'delta_faith', 'delta_pixel', 'composite_improvement']:
        values = improvements[metric].values
        print(f"  {metric}:")
        print(f"    Mean: {values.mean():.6f}, Std: {values.std():.6f}")
        print(f"    Quartiles: 25%={np.percentile(values, 25):.4f}, 50%={np.percentile(values, 50):.4f}, 75%={np.percentile(values, 75):.4f}")

    # Stratify images into improvement groups PER METRIC
    print(f"\nImage Stratification (per-metric):")

    # SaCo stratification
    high_threshold_saco = improvements['delta_saco'].quantile(improvement_quantiles[0])
    low_threshold_saco = improvements['delta_saco'].quantile(improvement_quantiles[1])
    improvements['group_saco'] = 'medium'
    improvements.loc[improvements['delta_saco'] >= high_threshold_saco, 'group_saco'] = 'high'
    improvements.loc[improvements['delta_saco'] <= low_threshold_saco, 'group_saco'] = 'low'
    group_counts_saco = improvements['group_saco'].value_counts()
    print(f"  SaCo: High (>={high_threshold_saco:.4f}): {group_counts_saco.get('high', 0)}, "
          f"Low (<={low_threshold_saco:.4f}): {group_counts_saco.get('low', 0)}, "
          f"Medium: {group_counts_saco.get('medium', 0)}")

    # Faithfulness stratification
    high_threshold_faith = improvements['delta_faith'].quantile(improvement_quantiles[0])
    low_threshold_faith = improvements['delta_faith'].quantile(improvement_quantiles[1])
    improvements['group_faith'] = 'medium'
    improvements.loc[improvements['delta_faith'] >= high_threshold_faith, 'group_faith'] = 'high'
    improvements.loc[improvements['delta_faith'] <= low_threshold_faith, 'group_faith'] = 'low'
    group_counts_faith = improvements['group_faith'].value_counts()
    print(f"  Faith: High (>={high_threshold_faith:.4f}): {group_counts_faith.get('high', 0)}, "
          f"Low (<={low_threshold_faith:.4f}): {group_counts_faith.get('low', 0)}, "
          f"Medium: {group_counts_faith.get('medium', 0)}")

    # PixelFlipping stratification (note: lower is better, so we flip the quantiles)
    high_threshold_pixel = improvements['delta_pixel'].quantile(improvement_quantiles[1])  # Lower values = better
    low_threshold_pixel = improvements['delta_pixel'].quantile(improvement_quantiles[0])   # Higher values = worse
    improvements['group_pixel'] = 'medium'
    improvements.loc[improvements['delta_pixel'] <= high_threshold_pixel, 'group_pixel'] = 'high'  # Low delta = good
    improvements.loc[improvements['delta_pixel'] >= low_threshold_pixel, 'group_pixel'] = 'low'     # High delta = bad
    group_counts_pixel = improvements['group_pixel'].value_counts()
    print(f"  Pixel: High (<={high_threshold_pixel:.4f}): {group_counts_pixel.get('high', 0)}, "
          f"Low (>={low_threshold_pixel:.4f}): {group_counts_pixel.get('low', 0)}, "
          f"Medium: {group_counts_pixel.get('medium', 0)}")

    # Check if we have enough images in each group for at least one metric
    has_sufficient_data = False
    for metric, counts in [('saco', group_counts_saco), ('faith', group_counts_faith), ('pixel', group_counts_pixel)]:
        if counts.get('high', 0) >= min_images and counts.get('low', 0) >= min_images:
            has_sufficient_data = True
            break

    if not has_sufficient_data:
        print(f"  Warning: Not enough images in high/low groups for any metric (need {min_images})")
        return {'image_stratification': improvements}

    # Collect patch-level data for each group
    patch_data = []

    for idx in df.index:
        row = df.loc[idx]
        img_idx = row['image_idx']

        if img_idx >= len(improvements):
            continue

        img_info = improvements.iloc[img_idx]

        patch_data.append({
            'image_idx': img_idx,
            'patch_idx': row['patch_idx'],
            'group_saco': img_info['group_saco'],
            'group_faith': img_info['group_faith'],
            'group_pixel': img_info['group_pixel'],
            'composite_improvement': img_info['composite_improvement'],
            'delta_saco': img_info['delta_saco'],
            'delta_faith': img_info['delta_faith'],
            'delta_pixel': img_info['delta_pixel'],
            'gate_value': row['gate_value'],
            'attribution_delta': row.get('attribution_delta', np.nan),
            'contribution_sum': row.get('contribution_sum', np.nan),
            'n_active_features': row['n_active_features'],
        })

    patch_df = pd.DataFrame(patch_data)

    # Within each image, identify high-impact patches
    high_impact_patches = []

    for img_idx in patch_df['image_idx'].unique():
        img_patches = patch_df[patch_df['image_idx'] == img_idx]

        # Find top patches by |attribution_delta|
        img_patches_sorted = img_patches.copy()
        img_patches_sorted['abs_attr_delta'] = img_patches_sorted['attribution_delta'].abs()

        # Get patches above the specified quantile
        threshold = img_patches_sorted['abs_attr_delta'].quantile(patch_quantile)
        high_patches = img_patches_sorted[img_patches_sorted['abs_attr_delta'] >= threshold]

        for _, patch in high_patches.iterrows():
            high_impact_patches.append(patch.to_dict())

    high_impact_df = pd.DataFrame(high_impact_patches)

    # Compute effect sizes: Compare patch characteristics between high vs low improvement groups (per metric)
    print(f"\n{'='*60}")
    print("EFFECT SIZE ANALYSIS: High vs Low Improvement Images (Per-Metric)")
    print(f"{'='*60}\n")

    # Helper function for effect size computation
    def compute_effect_sizes_for_metric(high_patches, low_patches, metric_name):
        effect_sizes = []

        print(f"--- {metric_name} ---")
        for feature in ['gate_value', 'attribution_delta', 'contribution_sum', 'n_active_features']:
            high_vals = high_patches[feature].dropna().values
            low_vals = low_patches[feature].dropna().values

            if len(high_vals) < 3 or len(low_vals) < 3:
                continue

            # Compute Cohen's d
            pooled_std = np.sqrt(((len(high_vals) - 1) * high_vals.std()**2 + (len(low_vals) - 1) * low_vals.std()**2) / (len(high_vals) + len(low_vals) - 2))
            cohens_d = (high_vals.mean() - low_vals.mean()) / pooled_std if pooled_std > 0 else 0

            # Permutation test for significance
            def permutation_test(group1, group2, n_permutations=1000):
                observed_diff = group1.mean() - group2.mean()
                combined = np.concatenate([group1, group2])
                count = 0
                for _ in range(n_permutations):
                    np.random.shuffle(combined)
                    perm_diff = combined[:len(group1)].mean() - combined[len(group1):].mean()
                    if abs(perm_diff) >= abs(observed_diff):
                        count += 1
                return count / n_permutations

            p_value = permutation_test(high_vals, low_vals)

            effect_sizes.append({
                'metric': metric_name,
                'feature': feature,
                'high_mean': high_vals.mean(),
                'high_std': high_vals.std(),
                'low_mean': low_vals.mean(),
                'low_std': low_vals.std(),
                'cohens_d': cohens_d,
                'p_value': p_value,
                'n_high': len(high_vals),
                'n_low': len(low_vals),
            })

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"  {feature}:")
            print(f"    High: μ={high_vals.mean():.4f}, σ={high_vals.std():.4f} (n={len(high_vals)})")
            print(f"    Low:  μ={low_vals.mean():.4f}, σ={low_vals.std():.4f} (n={len(low_vals)})")
            print(f"    Cohen's d={cohens_d:.4f}, p={p_value:.4f}{sig}")

        print()
        return effect_sizes

    all_effect_sizes = []

    # SaCo effect sizes
    high_patches_saco = patch_df[patch_df['group_saco'] == 'high']
    low_patches_saco = patch_df[patch_df['group_saco'] == 'low']
    all_effect_sizes.extend(compute_effect_sizes_for_metric(high_patches_saco, low_patches_saco, 'saco'))

    # Faithfulness effect sizes
    high_patches_faith = patch_df[patch_df['group_faith'] == 'high']
    low_patches_faith = patch_df[patch_df['group_faith'] == 'low']
    all_effect_sizes.extend(compute_effect_sizes_for_metric(high_patches_faith, low_patches_faith, 'faith'))

    # PixelFlipping effect sizes
    high_patches_pixel = patch_df[patch_df['group_pixel'] == 'high']
    low_patches_pixel = patch_df[patch_df['group_pixel'] == 'low']
    all_effect_sizes.extend(compute_effect_sizes_for_metric(high_patches_pixel, low_patches_pixel, 'pixel'))

    effect_size_df = pd.DataFrame(all_effect_sizes)

    return {
        'image_stratification': improvements,
        'patch_statistics': patch_df,
        'effect_sizes': effect_size_df,
        'high_impact_patches': high_impact_df,
        # Per-metric patch groups for Stage 2
        'high_improvement_patches_saco': high_patches_saco,
        'low_improvement_patches_saco': low_patches_saco,
        'high_improvement_patches_faith': high_patches_faith,
        'low_improvement_patches_faith': low_patches_faith,
        'high_improvement_patches_pixel': high_patches_pixel,
        'low_improvement_patches_pixel': low_patches_pixel,
    }


def analyze_feature_composition_of_patches(
    merged_data: Dict[int, pd.DataFrame],
    patch_impact_results: Dict[str, Any],
    layer_idx: int,
    min_occurrences: int = 10,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Stage 2: Analyze which features are present in high-impact patches.

    Takes the high-impact patches identified in Stage 1 and analyzes their feature composition.

    Args:
        merged_data: Output from merge_features_faithfulness
        patch_impact_results: Output from analyze_patch_impact_on_faithfulness
        layer_idx: Which layer to analyze
        min_occurrences: Minimum times feature must appear
        top_n: Top N features to return

    Returns:
        DataFrame with feature composition statistics
    """
    from collections import defaultdict

    from scipy import stats as scipy_stats

    df = merged_data[layer_idx]

    if 'high_improvement_patches_saco' not in patch_impact_results:
        print("Error: Need Stage 1 results first (with per-metric stratification)")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"STAGE 2: FEATURE COMPOSITION ANALYSIS - Layer {layer_idx} (Per-Metric)")
    print(f"{'='*60}\n")

    # Get per-metric patch groups
    high_patches_saco = patch_impact_results['high_improvement_patches_saco']
    low_patches_saco = patch_impact_results['low_improvement_patches_saco']
    high_patches_faith = patch_impact_results['high_improvement_patches_faith']
    low_patches_faith = patch_impact_results['low_improvement_patches_faith']
    high_patches_pixel = patch_impact_results['high_improvement_patches_pixel']
    low_patches_pixel = patch_impact_results['low_improvement_patches_pixel']

    # Create lookup sets for fast membership testing (per metric)
    high_patch_keys_saco = set((row['image_idx'], row['patch_idx']) for _, row in high_patches_saco.iterrows())
    low_patch_keys_saco = set((row['image_idx'], row['patch_idx']) for _, row in low_patches_saco.iterrows())

    high_patch_keys_faith = set((row['image_idx'], row['patch_idx']) for _, row in high_patches_faith.iterrows())
    low_patch_keys_faith = set((row['image_idx'], row['patch_idx']) for _, row in low_patches_faith.iterrows())

    high_patch_keys_pixel = set((row['image_idx'], row['patch_idx']) for _, row in high_patches_pixel.iterrows())
    low_patch_keys_pixel = set((row['image_idx'], row['patch_idx']) for _, row in low_patches_pixel.iterrows())

    # Collect feature statistics (per metric)
    feature_stats = defaultdict(lambda: {
        # SaCo
        'count_high_saco': 0,
        'count_low_saco': 0,
        'contributions_high_saco': [],
        'contributions_low_saco': [],
        'activations_high_saco': [],
        'activations_low_saco': [],
        'gates_high_saco': [],
        'gates_low_saco': [],
        # Faithfulness
        'count_high_faith': 0,
        'count_low_faith': 0,
        'contributions_high_faith': [],
        'contributions_low_faith': [],
        'activations_high_faith': [],
        'activations_low_faith': [],
        'gates_high_faith': [],
        'gates_low_faith': [],
        # PixelFlipping
        'count_high_pixel': 0,
        'count_low_pixel': 0,
        'contributions_high_pixel': [],
        'contributions_low_pixel': [],
        'activations_high_pixel': [],
        'activations_low_pixel': [],
        'gates_high_pixel': [],
        'gates_low_pixel': [],
    })

    for idx in df.index:
        row = df.loc[idx]
        patch_key = (row['image_idx'], row['patch_idx'])

        # Check membership in each metric's groups
        is_high_saco = patch_key in high_patch_keys_saco
        is_low_saco = patch_key in low_patch_keys_saco
        is_high_faith = patch_key in high_patch_keys_faith
        is_low_faith = patch_key in low_patch_keys_faith
        is_high_pixel = patch_key in high_patch_keys_pixel
        is_low_pixel = patch_key in low_patch_keys_pixel

        # Skip if not in any group
        if not (is_high_saco or is_low_saco or is_high_faith or is_low_faith or is_high_pixel or is_low_pixel):
            continue

        active_features = row['active_features']
        feature_acts = row['feature_activations']
        feature_contribs = row.get('feature_contributions', [])
        gate_value = row['gate_value']

        if not isinstance(active_features, (list, np.ndarray)):
            continue

        for i, feat_idx in enumerate(active_features):
            # SaCo
            if is_high_saco:
                feature_stats[feat_idx]['count_high_saco'] += 1
                feature_stats[feat_idx]['activations_high_saco'].append(feature_acts[i])
                feature_stats[feat_idx]['gates_high_saco'].append(gate_value)
                if len(feature_contribs) > i:
                    feature_stats[feat_idx]['contributions_high_saco'].append(feature_contribs[i])

            if is_low_saco:
                feature_stats[feat_idx]['count_low_saco'] += 1
                feature_stats[feat_idx]['activations_low_saco'].append(feature_acts[i])
                feature_stats[feat_idx]['gates_low_saco'].append(gate_value)
                if len(feature_contribs) > i:
                    feature_stats[feat_idx]['contributions_low_saco'].append(feature_contribs[i])

            # Faithfulness
            if is_high_faith:
                feature_stats[feat_idx]['count_high_faith'] += 1
                feature_stats[feat_idx]['activations_high_faith'].append(feature_acts[i])
                feature_stats[feat_idx]['gates_high_faith'].append(gate_value)
                if len(feature_contribs) > i:
                    feature_stats[feat_idx]['contributions_high_faith'].append(feature_contribs[i])

            if is_low_faith:
                feature_stats[feat_idx]['count_low_faith'] += 1
                feature_stats[feat_idx]['activations_low_faith'].append(feature_acts[i])
                feature_stats[feat_idx]['gates_low_faith'].append(gate_value)
                if len(feature_contribs) > i:
                    feature_stats[feat_idx]['contributions_low_faith'].append(feature_contribs[i])

            # PixelFlipping
            if is_high_pixel:
                feature_stats[feat_idx]['count_high_pixel'] += 1
                feature_stats[feat_idx]['activations_high_pixel'].append(feature_acts[i])
                feature_stats[feat_idx]['gates_high_pixel'].append(gate_value)
                if len(feature_contribs) > i:
                    feature_stats[feat_idx]['contributions_high_pixel'].append(feature_contribs[i])

            if is_low_pixel:
                feature_stats[feat_idx]['count_low_pixel'] += 1
                feature_stats[feat_idx]['activations_low_pixel'].append(feature_acts[i])
                feature_stats[feat_idx]['gates_low_pixel'].append(gate_value)
                if len(feature_contribs) > i:
                    feature_stats[feat_idx]['contributions_low_pixel'].append(feature_contribs[i])

    # Compute enrichment and statistics (per metric)
    feature_results = []

    total_high_patches_saco = len(high_patch_keys_saco)
    total_low_patches_saco = len(low_patch_keys_saco)
    total_high_patches_faith = len(high_patch_keys_faith)
    total_low_patches_faith = len(low_patch_keys_faith)
    total_high_patches_pixel = len(high_patch_keys_pixel)
    total_low_patches_pixel = len(low_patch_keys_pixel)

    print(f"Patch counts per metric:")
    print(f"  SaCo: {total_high_patches_saco} high, {total_low_patches_saco} low")
    print(f"  Faith: {total_high_patches_faith} high, {total_low_patches_faith} low")
    print(f"  Pixel: {total_high_patches_pixel} high, {total_low_patches_pixel} low\n")

    for feat_idx, stats in feature_stats.items():
        # Get counts for each metric
        count_high_saco = stats['count_high_saco']
        count_low_saco = stats['count_low_saco']
        count_high_faith = stats['count_high_faith']
        count_low_faith = stats['count_low_faith']
        count_high_pixel = stats['count_high_pixel']
        count_low_pixel = stats['count_low_pixel']

        # Stricter filtering to avoid ultra-rare features
        # Require minimum occurrences in at least one metric
        total_saco = count_high_saco + count_low_saco
        total_faith = count_high_faith + count_low_faith
        total_pixel = count_high_pixel + count_low_pixel

        max_total = max(total_saco, total_faith, total_pixel)
        if max_total < min_occurrences:
            continue

        # Minimum count per group (for features that appear in specific metrics)
        min_count_per_group = max(3, min_occurrences // 10)  # At least 3, or 10% of min_occurrences

        # Compute per-metric enrichments
        result_row = {'feature_idx': feat_idx}

        # Helper function to compute enrichment for a metric
        def compute_metric_enrichment(count_h, count_l, total_h, total_l, metric_suffix):
            freq_h = count_h / total_h if total_h > 0 else 0
            freq_l = count_l / total_l if total_l > 0 else 0

            # Enrichment ratio
            enrichment = freq_h / freq_l if freq_l > 0 else (float('inf') if freq_h > 0 else 1.0)

            # Specificity (differential enrichment)
            specificity = freq_h - freq_l

            # Fisher's exact test
            p_val = 1.0
            if count_h >= min_count_per_group and count_l >= min_count_per_group:
                contingency_table = [
                    [count_h, total_h - count_h],
                    [count_l, total_l - count_l]
                ]
                _, p_val = scipy_stats.fisher_exact(contingency_table)

            # Mean statistics
            mean_act_h = np.mean(stats[f'activations_high_{metric_suffix}']) if stats[f'activations_high_{metric_suffix}'] else 0
            mean_act_l = np.mean(stats[f'activations_low_{metric_suffix}']) if stats[f'activations_low_{metric_suffix}'] else 0
            mean_contrib_h = np.mean(stats[f'contributions_high_{metric_suffix}']) if stats[f'contributions_high_{metric_suffix}'] else 0
            mean_contrib_l = np.mean(stats[f'contributions_low_{metric_suffix}']) if stats[f'contributions_low_{metric_suffix}'] else 0
            mean_gate_h = np.mean(stats[f'gates_high_{metric_suffix}']) if stats[f'gates_high_{metric_suffix}'] else 0
            mean_gate_l = np.mean(stats[f'gates_low_{metric_suffix}']) if stats[f'gates_low_{metric_suffix}'] else 0

            return {
                f'count_high_{metric_suffix}': count_h,
                f'count_low_{metric_suffix}': count_l,
                f'freq_high_{metric_suffix}': freq_h,
                f'freq_low_{metric_suffix}': freq_l,
                f'enrichment_{metric_suffix}': enrichment,
                f'specificity_{metric_suffix}': specificity,
                f'p_value_{metric_suffix}': p_val,
                f'mean_activation_high_{metric_suffix}': mean_act_h,
                f'mean_activation_low_{metric_suffix}': mean_act_l,
                f'mean_contribution_high_{metric_suffix}': mean_contrib_h,
                f'mean_contribution_low_{metric_suffix}': mean_contrib_l,
                f'mean_gate_high_{metric_suffix}': mean_gate_h,
                f'mean_gate_low_{metric_suffix}': mean_gate_l,
            }

        # Compute for each metric
        result_row.update(compute_metric_enrichment(
            count_high_saco, count_low_saco, total_high_patches_saco, total_low_patches_saco, 'saco'
        ))
        result_row.update(compute_metric_enrichment(
            count_high_faith, count_low_faith, total_high_patches_faith, total_low_patches_faith, 'faith'
        ))
        result_row.update(compute_metric_enrichment(
            count_high_pixel, count_low_pixel, total_high_patches_pixel, total_low_patches_pixel, 'pixel'
        ))

        # Cross-metric consistency
        specificity_saco = result_row['specificity_saco']
        specificity_faith = result_row['specificity_faith']
        specificity_pixel = result_row['specificity_pixel']

        result_row['specificity_combined'] = (specificity_saco + specificity_faith + specificity_pixel) / 3.0
        result_row['n_metrics_beneficial'] = sum([specificity_saco > 0, specificity_faith > 0, specificity_pixel > 0])
        result_row['n_metrics_harmful'] = sum([specificity_saco < 0, specificity_faith < 0, specificity_pixel < 0])

        # Average enrichment across metrics (for sorting)
        result_row['enrichment_combined'] = (result_row['enrichment_saco'] + result_row['enrichment_faith'] + result_row['enrichment_pixel']) / 3.0

        feature_results.append(result_row)

    results_df = pd.DataFrame(feature_results)

    if results_df.empty:
        print("No features found meeting criteria")
        return results_df

    # Sort by enrichment (like before), using average enrichment across metrics
    # Optionally filter to only beneficial features (positive specificity)
    # results_df = results_df[results_df['specificity_combined'] > 0]  # Uncomment to filter
    results_df = results_df.sort_values('enrichment_combined', ascending=False)

    print(f"Found {len(results_df)} features with sufficient occurrences\n")

    # Print statistics about ALL features found
    if len(results_df) > 0:
        print("Cross-Metric Consistency:")
        all_beneficial = results_df[results_df['n_metrics_beneficial'] == 3]
        all_harmful = results_df[results_df['n_metrics_harmful'] == 3]
        mixed = results_df[(results_df['n_metrics_beneficial'] > 0) & (results_df['n_metrics_harmful'] > 0)]
        neutral = results_df[(results_df['n_metrics_beneficial'] == 0) & (results_df['n_metrics_harmful'] == 0)]

        print(f"  All 3 metrics beneficial: {len(all_beneficial)} features ({len(all_beneficial)/len(results_df)*100:.1f}%)")
        print(f"  All 3 metrics harmful: {len(all_harmful)} features ({len(all_harmful)/len(results_df)*100:.1f}%)")
        print(f"  Mixed (some beneficial, some harmful): {len(mixed)} features ({len(mixed)/len(results_df)*100:.1f}%)")
        print(f"  Neutral (no strong effect): {len(neutral)} features ({len(neutral)/len(results_df)*100:.1f}%)")

        print(f"\nPer-Metric Enrichment Statistics:")
        for metric in ['saco', 'faith', 'pixel']:
            enrichment_col = f'enrichment_{metric}'
            specificity_col = f'specificity_{metric}'
            print(f"  {metric.upper()}:")
            print(f"    Mean enrichment: {results_df[enrichment_col].mean():.2f}x")
            print(f"    Mean specificity: {results_df[specificity_col].mean():+.3f}")
            beneficial = results_df[results_df[specificity_col] > 0]
            harmful = results_df[results_df[specificity_col] < 0]
            print(f"    Beneficial: {len(beneficial)} ({len(beneficial)/len(results_df)*100:.1f}%), Harmful: {len(harmful)} ({len(harmful)/len(results_df)*100:.1f}%)")

        print(f"\nCombined Specificity:")
        print(f"  Mean: {results_df['specificity_combined'].mean():+.3f}")
        print(f"  Median: {results_df['specificity_combined'].median():+.3f}")
        print(f"  Max: {results_df['specificity_combined'].max():+.3f}")
        print(f"  Min: {results_df['specificity_combined'].min():+.3f}")

    print(f"\n{'='*60}")
    print(f"Top {top_n} ENRICHED features (highest combined enrichment):")
    print(f"{'='*60}\n")
    for _, row in results_df.head(top_n).iterrows():
        feat_idx = int(row['feature_idx'])
        n_beneficial = int(row['n_metrics_beneficial'])
        n_harmful = int(row['n_metrics_harmful'])
        combined_enrichment = row['enrichment_combined']
        combined_spec = row['specificity_combined']

        # Determine consistency tag
        consistency_tag = ""
        if n_beneficial == 3:
            consistency_tag = " [ALL 3 BENEFICIAL]"
        elif n_beneficial == 2:
            consistency_tag = " [2/3 BENEFICIAL]"
        elif n_harmful >= 2:
            consistency_tag = " [HARMFUL]"

        # Determine booster/debooster from SaCo contribution
        contrib_type = ""
        if row['mean_contribution_high_saco'] > 0:
            contrib_type = " [BOOSTER]"
        elif row['mean_contribution_high_saco'] < 0:
            contrib_type = " [DEBOOSTER]"

        print(f"  Feature {feat_idx}{consistency_tag}{contrib_type}: enrichment={combined_enrichment:.2f}x, specificity={combined_spec:+.3f}")
        print(f"    Per-metric enrichment: SaCo={row['enrichment_saco']:.2f}x, Faith={row['enrichment_faith']:.2f}x, Pixel={row['enrichment_pixel']:.2f}x")
        print(f"    Per-metric specificity: SaCo={row['specificity_saco']:+.3f}, Faith={row['specificity_faith']:+.3f}, Pixel={row['specificity_pixel']:+.3f}")
        print(f"    Counts - SaCo: high={int(row['count_high_saco'])}, low={int(row['count_low_saco'])} | Faith: high={int(row['count_high_faith'])}, low={int(row['count_low_faith'])} | Pixel: high={int(row['count_high_pixel'])}, low={int(row['count_low_pixel'])}")
        print(f"    Activation - SaCo: high={row['mean_activation_high_saco']:.3f}, low={row['mean_activation_low_saco']:.3f} | Faith: high={row['mean_activation_high_faith']:.3f}, low={row['mean_activation_low_faith']:.3f} | Pixel: high={row['mean_activation_high_pixel']:.3f}, low={row['mean_activation_low_pixel']:.3f}")
        print(f"    Contribution (SaCo): high={row['mean_contribution_high_saco']:+.2e}, low={row['mean_contribution_low_saco']:+.2e}")
        print()

    # Show top harmful features (enriched in low-improvement images)
    harmful_features = results_df[results_df['n_metrics_harmful'] >= 2].head(10)
    if len(harmful_features) > 0:
        print(f"\n{'='*60}")
        print(f"Top {len(harmful_features)} HARMFUL features (enriched in low-improvement):")
        print(f"{'='*60}\n")
        for _, row in harmful_features.iterrows():
            feat_idx = int(row['feature_idx'])
            n_harmful = int(row['n_metrics_harmful'])
            combined_enrichment = row['enrichment_combined']
            combined_spec = row['specificity_combined']

            consistency_tag = ""
            if n_harmful == 3:
                consistency_tag = " [ALL 3 HARMFUL]"
            elif n_harmful == 2:
                consistency_tag = " [2/3 HARMFUL]"

            # Determine booster/debooster from SaCo contribution
            contrib_type = ""
            if row['mean_contribution_high_saco'] > 0:
                contrib_type = " [BOOSTER]"
            elif row['mean_contribution_high_saco'] < 0:
                contrib_type = " [DEBOOSTER]"

            print(f"  Feature {feat_idx}{consistency_tag}{contrib_type}: enrichment={combined_enrichment:.2f}x, specificity={combined_spec:+.3f}")
            print(f"    Per-metric enrichment: SaCo={row['enrichment_saco']:.2f}x, Faith={row['enrichment_faith']:.2f}x, Pixel={row['enrichment_pixel']:.2f}x")
            print(f"    Per-metric specificity: SaCo={row['specificity_saco']:+.3f}, Faith={row['specificity_faith']:+.3f}, Pixel={row['specificity_pixel']:+.3f}")
            print(f"    Counts - SaCo: high={int(row['count_high_saco'])}, low={int(row['count_low_saco'])} | Faith: high={int(row['count_high_faith'])}, low={int(row['count_low_faith'])} | Pixel: high={int(row['count_high_pixel'])}, low={int(row['count_low_pixel'])}")
            print(f"    Activation - SaCo: high={row['mean_activation_high_saco']:.3f}, low={row['mean_activation_low_saco']:.3f} | Faith: high={row['mean_activation_high_faith']:.3f}, low={row['mean_activation_low_faith']:.3f} | Pixel: high={row['mean_activation_high_pixel']:.3f}, low={row['mean_activation_low_pixel']:.3f}")
            print(f"    Contribution (SaCo): high={row['mean_contribution_high_saco']:+.2e}, low={row['mean_contribution_low_saco']:+.2e}")
            print()

    print("\n" + "=" * 60 + "\n")

    return results_df


def analyze_feature_contribution_by_class(
    merged_data: Dict[int, pd.DataFrame],
    layer_idx: int,
    stratify_by: str = 'predicted_class',
    min_occurrences: int = 50,
    top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Stratify feature-contribution correlation analysis by class.

    This reveals whether features are class-specific (only helpful for certain classes)
    or universal (helpful across all classes).

    Args:
        merged_data: Output from merge_features_faithfulness
        layer_idx: Which layer to analyze
        stratify_by: 'predicted_class' or 'true_class'
        min_occurrences: Minimum occurrences per class
        top_n: Top features to return per class

    Returns:
        Dictionary mapping class_name -> DataFrame of top features for that class
    """
    from collections import defaultdict

    from scipy import stats as scipy_stats

    df = merged_data[layer_idx]

    if stratify_by not in df.columns:
        raise ValueError(f"Column {stratify_by} not in dataframe")

    # Get unique classes
    classes = df[stratify_by].unique()
    results_per_class = {}

    for class_name in classes:
        if pd.isna(class_name):
            continue

        # Filter to this class only
        class_df = df[df[stratify_by] == class_name]

        # Accumulate contributions per feature for this class
        feature_data = defaultdict(lambda: {
            'contributions': [],
            'attribution_deltas': [],
        })

        for idx in class_df.index:
            row = class_df.loc[idx]
            attr_delta = row.get('attribution_delta', np.nan)

            if np.isnan(attr_delta):
                continue

            active_features = row['active_features']
            feature_contribs = row['feature_contributions']

            if not isinstance(active_features, (list, np.ndarray)) or not isinstance(feature_contribs, (list, np.ndarray)):
                continue

            if len(active_features) == 0 or len(feature_contribs) == 0:
                continue

            for feat_idx, contrib in zip(active_features, feature_contribs):
                feature_data[feat_idx]['contributions'].append(contrib)
                feature_data[feat_idx]['attribution_deltas'].append(attr_delta)

        # Compute correlation for each feature
        results = []
        for feat_idx, data in feature_data.items():
            n_occur = len(data['contributions'])

            if n_occur < min_occurrences:
                continue

            contribs = np.array(data['contributions'])
            attr_deltas = np.array(data['attribution_deltas'])

            if len(contribs) >= 3:
                corr, p_value = scipy_stats.pearsonr(contribs, attr_deltas)
            else:
                corr, p_value = np.nan, np.nan

            effect_size = corr * np.sqrt(n_occur) if not np.isnan(corr) else np.nan

            results.append({
                'feature_idx': feat_idx,
                'class': class_name,
                'correlation': corr,
                'p_value': p_value,
                'mean_contribution': contribs.mean(),
                'n_occurrences': n_occur,
                'effect_size': effect_size,
            })

        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('effect_size', ascending=False, key=abs).head(top_n)
            results_per_class[class_name] = results_df

    return results_per_class


def analyze_failure_cases(
    merged_data: Dict[int, pd.DataFrame],
    faithfulness_vanilla: pd.DataFrame,
    faithfulness_gated: pd.DataFrame,
    layer_idx: int,
    metric: str = 'saco_score',
    min_occurrences: int = 50
) -> pd.DataFrame:
    """
    Analyze cases where gating made faithfulness WORSE, not better.

    This identifies "anti-features" - features that correlate with degraded performance.

    Args:
        merged_data: Output from merge_features_faithfulness
        faithfulness_vanilla: Vanilla faithfulness results
        faithfulness_gated: Gated faithfulness results
        layer_idx: Which layer to analyze
        metric: 'saco_score', 'FaithfulnessCorrelation', or 'PixelFlipping'
        min_occurrences: Minimum occurrences to be considered

    Returns:
        DataFrame with features that appear most in failure cases
    """
    from collections import defaultdict

    df = merged_data[layer_idx]

    # Compute delta in faithfulness (gated - vanilla)
    faithfulness_delta = faithfulness_gated[metric] - faithfulness_vanilla[metric]

    # Identify failure cases (any decrease in faithfulness)
    # For saco and FaithfulnessCorrelation: lower is worse (negative delta = failure)
    # For PixelFlipping: higher is worse (positive delta = failure)
    if metric in ['saco_score', 'FaithfulnessCorrelation']:
        failure_mask = faithfulness_delta < 0
    else:  # PixelFlipping
        failure_mask = faithfulness_delta > 0

    failure_image_indices = faithfulness_vanilla[failure_mask].index.tolist()

    print(f"  Found {len(failure_image_indices)} failure cases (out of {len(faithfulness_vanilla)} total)")

    if len(failure_image_indices) == 0:
        return pd.DataFrame()

    # For each feature, track occurrence in failure vs success cases
    feature_stats = defaultdict(lambda: {
        'failure_count': 0,
        'success_count': 0,
        'failure_contributions': [],
        'success_contributions': [],
        'failure_deltas': [],
        'success_deltas': [],
    })

    # Iterate through patches
    for idx in df.index:
        row = df.loc[idx]
        img_idx = row['image_idx']

        is_failure = img_idx in failure_image_indices

        active_features = row['active_features']
        feature_contribs = row['feature_contributions']

        if not isinstance(active_features, (list, np.ndarray)) or not isinstance(feature_contribs, (list, np.ndarray)):
            continue

        if len(active_features) == 0 or len(feature_contribs) == 0:
            continue

        attr_delta = row.get('attribution_delta', np.nan)

        for feat_idx, contrib in zip(active_features, feature_contribs):
            if is_failure:
                feature_stats[feat_idx]['failure_count'] += 1
                feature_stats[feat_idx]['failure_contributions'].append(contrib)
                if not np.isnan(attr_delta):
                    feature_stats[feat_idx]['failure_deltas'].append(attr_delta)
            else:
                feature_stats[feat_idx]['success_count'] += 1
                feature_stats[feat_idx]['success_contributions'].append(contrib)
                if not np.isnan(attr_delta):
                    feature_stats[feat_idx]['success_deltas'].append(attr_delta)

    # Compute statistics for each feature
    results = []
    for feat_idx, stats in feature_stats.items():
        total_count = stats['failure_count'] + stats['success_count']

        if total_count < min_occurrences:
            continue

        failure_rate = stats['failure_count'] / total_count if total_count > 0 else 0

        # Average contribution in failure vs success cases
        mean_failure_contrib = np.mean(stats['failure_contributions']) if stats['failure_contributions'] else np.nan
        mean_success_contrib = np.mean(stats['success_contributions']) if stats['success_contributions'] else np.nan

        # Contribution difference (how different is this feature in failures?)
        contrib_diff = mean_failure_contrib - mean_success_contrib if not (np.isnan(mean_failure_contrib) or np.isnan(mean_success_contrib)) else np.nan

        # Attribution delta in failure vs success
        mean_failure_delta = np.mean(stats['failure_deltas']) if stats['failure_deltas'] else np.nan
        mean_success_delta = np.mean(stats['success_deltas']) if stats['success_deltas'] else np.nan

        results.append({
            'feature_idx': feat_idx,
            'failure_rate': failure_rate,
            'failure_count': stats['failure_count'],
            'success_count': stats['success_count'],
            'total_count': total_count,
            'mean_failure_contrib': mean_failure_contrib,
            'mean_success_contrib': mean_success_contrib,
            'contrib_diff': contrib_diff,
            'mean_failure_delta': mean_failure_delta,
            'mean_success_delta': mean_success_delta,
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return pd.DataFrame()

    # Sort by failure rate (features that appear most often in failures)
    results_df = results_df.sort_values('failure_rate', ascending=False)

    return results_df


def find_top_attribution_changing_features(
    merged_data: Dict[int, pd.DataFrame],
    layer_idx: int,
    top_n: int = 10,
    mode: str = 'boost',
    min_occurrences: int = 100
) -> pd.DataFrame:
    """
    Q4: What features cause the most boosting/deboosting?

    Groups by feature and computes weighted impact: mean_delta * sqrt(n_occurrences)
    This balances high per-occurrence impact with reasonable frequency.

    Args:
        merged_data: Output from merge_features_faithfulness
        layer_idx: Which layer to analyze
        top_n: How many top features to return
        mode: 'boost' (positive deltas) or 'deboost' (negative deltas)
        min_occurrences: Minimum occurrences to be considered

    Returns:
        DataFrame with top features sorted by weighted impact
    """
    from collections import defaultdict

    df = merged_data[layer_idx]

    if 'attribution_delta' not in df.columns:
        raise ValueError(f"Layer {layer_idx} missing attribution_delta column")

    # Accumulate attribution deltas per feature
    feature_impacts = defaultdict(lambda: {
        'total_abs_delta': 0.0,
        'total_delta': 0.0,
        'n_occurrences': 0,
        'deltas': []
    })

    for idx in df.index:
        row = df.loc[idx]
        attr_delta = row.get('attribution_delta', np.nan)

        if np.isnan(attr_delta):
            continue

        for feat_idx in row['active_features']:
            feature_impacts[feat_idx]['total_abs_delta'] += abs(attr_delta)
            feature_impacts[feat_idx]['total_delta'] += attr_delta
            feature_impacts[feat_idx]['n_occurrences'] += 1
            feature_impacts[feat_idx]['deltas'].append(attr_delta)

    # Convert to DataFrame
    results = []
    for feat_idx, impact in feature_impacts.items():
        if impact['n_occurrences'] < min_occurrences:
            continue

        mean_abs_delta = impact['total_abs_delta'] / impact['n_occurrences']
        mean_delta = impact['total_delta'] / impact['n_occurrences']
        std_delta = np.std(impact['deltas'])

        # Weighted impact: mean * sqrt(frequency)
        # Balances high impact with reasonable frequency
        weighted_impact_abs = mean_abs_delta * np.sqrt(impact['n_occurrences'])
        weighted_impact_signed = mean_delta * np.sqrt(impact['n_occurrences'])

        results.append({
            'feature_idx': feat_idx,
            'weighted_impact_abs': weighted_impact_abs,
            'weighted_impact_signed': weighted_impact_signed,
            'mean_abs_delta': mean_abs_delta,
            'mean_delta': mean_delta,
            'std_delta': std_delta,
            'n_occurrences': impact['n_occurrences'],
            'total_abs_delta': impact['total_abs_delta'],
            'total_delta': impact['total_delta'],
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print(f"Warning: No features found with >= {min_occurrences} occurrences")
        return pd.DataFrame()

    # Sort by requested mode
    if mode == 'boost':
        # Features with most positive weighted impact
        results_df = results_df.sort_values('weighted_impact_signed', ascending=False)
    elif mode == 'deboost':
        # Features with most negative weighted impact
        results_df = results_df.sort_values('weighted_impact_signed', ascending=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return results_df.head(top_n)


def visualize_top_feature_examples(
    feature_idx: int,
    layer_idx: int,
    merged_data: Dict[int, pd.DataFrame],
    image_dir: Path,
    output_dir: Path,
    n_examples: int = 5,
    sort_by: str = 'attribution_delta'
):
    """
    Visualize representative images where a feature is active.

    Args:
        feature_idx: The feature to visualize
        layer_idx: Which layer this feature is from
        merged_data: Output from merge_features_faithfulness
        image_dir: Directory containing original images
        output_dir: Where to save visualizations
        n_examples: Number of example images to show
        sort_by: How to select examples ('attribution_delta' for highest impact, 'activation' for highest activation)
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    df = merged_data[layer_idx]

    # Find all patches where this feature is active
    feature_rows = []
    for idx in df.index:
        row = df.loc[idx]
        if feature_idx in row['active_features']:
            feat_position = list(row['active_features']).index(feature_idx)
            feature_rows.append({
                'image_idx': row['image_idx'],
                'patch_idx': row['patch_idx'],
                'activation': row['feature_activations'][feat_position],
                'gradient': row['feature_gradients'][feat_position],
                'gate_value': row['gate_value'],
                'attribution_delta': row.get('attribution_delta', 0),
            })

    if not feature_rows:
        print(f"Feature {feature_idx} not found in layer {layer_idx}")
        return

    feature_df = pd.DataFrame(feature_rows)

    # Group by image and aggregate
    image_groups = feature_df.groupby('image_idx').agg({
        'patch_idx': list,
        'activation': lambda x: list(x),
        'attribution_delta': 'mean',
        'gate_value': 'mean',
    }).reset_index()

    # Sort by requested metric
    if sort_by == 'attribution_delta':
        image_groups = image_groups.sort_values('attribution_delta', ascending=False, key=abs)
    elif sort_by == 'activation':
        image_groups['max_activation'] = image_groups['activation'].apply(max)
        image_groups = image_groups.sort_values('max_activation', ascending=False)

    # Take top N examples
    top_examples = image_groups.head(n_examples)

    # Create visualization
    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
    if n_examples == 1:
        axes = [axes]

    for ax_idx, (_, example) in enumerate(top_examples.iterrows()):
        img_idx = example['image_idx']
        patch_indices = example['patch_idx']
        activations = example['activation']

        # Load image using dataset-specific function
        image_path = get_image_path_covidquex(img_idx, image_dir)

        if image_path is None or not image_path.exists():
            print(f"Warning: Image not found for idx {img_idx}")
            continue

        img = Image.open(image_path).convert('RGB')

        # Display image
        axes[ax_idx].imshow(img)

        # Overlay patches where feature is active
        patch_size = 16  # 224/14 for ViT
        patches_per_side = 14

        for patch_idx, activation in zip(patch_indices, activations):
            row = patch_idx // patches_per_side
            col = patch_idx % patches_per_side

            x = col * patch_size
            y = row * patch_size

            # Color intensity based on activation
            alpha = min(activation / 2.0, 0.8)  # Scale alpha
            rect = patches.Rectangle((x, y), patch_size, patch_size,
                                     linewidth=2, edgecolor='lime', facecolor='lime',
                                     alpha=alpha)
            axes[ax_idx].add_patch(rect)

        axes[ax_idx].set_title(
            f"Img {img_idx}\nΔ={example['attribution_delta']:.4f}\nGate={example['gate_value']:.2f}",
            fontsize=10
        )
        axes[ax_idx].axis('off')

    plt.suptitle(f"Feature {feature_idx} (Layer {layer_idx}) - Top {n_examples} Examples", fontsize=14)
    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"feature_{feature_idx}_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def visualize_feature_detailed(
    feature_idx: int,
    layer_idx: int,
    merged_data: Dict[int, pd.DataFrame],
    image_dir: Path,
    output_dir: Path,
    n_examples: int = 50,
    sort_by: str = 'contribution'
):
    """
    Save detailed visualizations for a single feature (50 individual images in subfolder).

    Args:
        feature_idx: The feature to visualize
        layer_idx: Which layer this feature is from
        merged_data: Output from merge_features_faithfulness
        image_dir: Directory containing original images
        output_dir: Base directory (subfolder will be created per feature)
        n_examples: Number of example images to save
        sort_by: 'contribution' (by feature contribution) or 'activation' (by activation strength)
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    df = merged_data[layer_idx]

    # Find all patches where this feature is active
    feature_rows = []
    for idx in df.index:
        row = df.loc[idx]
        if feature_idx in row['active_features']:
            feat_position = list(row['active_features']).index(feature_idx)

            # Get contribution if available
            contribution = None
            if 'feature_contributions' in row and len(row['feature_contributions']) > feat_position:
                contribution = row['feature_contributions'][feat_position]

            feature_rows.append({
                'image_idx': row['image_idx'],
                'patch_idx': row['patch_idx'],
                'activation': row['feature_activations'][feat_position],
                'gradient': row['feature_gradients'][feat_position],
                'contribution': contribution,
                'gate_value': row['gate_value'],
                'attribution_delta': row.get('attribution_delta', 0),
            })

    if not feature_rows:
        print(f"Feature {feature_idx} not found in layer {layer_idx}")
        return

    feature_df = pd.DataFrame(feature_rows)

    # Group by image and aggregate
    agg_dict = {
        'patch_idx': list,
        'activation': lambda x: list(x),
        'attribution_delta': 'mean',
        'gate_value': 'mean',
    }

    if feature_df['contribution'].notna().any():
        agg_dict['contribution'] = lambda x: list(x)

    image_groups = feature_df.groupby('image_idx').agg(agg_dict).reset_index()

    # Sort by requested metric
    if sort_by == 'contribution' and 'contribution' in image_groups.columns:
        image_groups['mean_abs_contribution'] = image_groups['contribution'].apply(lambda x: np.mean(np.abs(x)))
        image_groups = image_groups.sort_values('mean_abs_contribution', ascending=False)
    elif sort_by == 'activation':
        image_groups['max_activation'] = image_groups['activation'].apply(max)
        image_groups = image_groups.sort_values('max_activation', ascending=False)
    else:  # Default to attribution_delta
        image_groups = image_groups.sort_values('attribution_delta', ascending=False, key=abs)

    # Take top N examples
    top_examples = image_groups.head(n_examples)

    # Create subfolder for this feature
    feature_dir = output_dir / f"feature_{feature_idx}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Collect statistics for CSV
    summary_stats = []

    # Save each image individually
    for rank, (_, example) in enumerate(top_examples.iterrows(), 1):
        img_idx = example['image_idx']
        patch_indices = example['patch_idx']
        activations = example['activation']
        contributions = example.get('contribution', [None] * len(patch_indices))

        # Get ALL patches for this image to compute within-image delta distribution
        all_image_patches = df[df['image_idx'] == img_idx]
        all_deltas = all_image_patches['attribution_delta'].values
        all_deltas = all_deltas[~np.isnan(all_deltas)]  # Remove NaN values

        # Compute within-image statistics
        if len(all_deltas) > 0:
            delta_mean = all_deltas.mean()
            delta_median = np.median(all_deltas)
            delta_std = all_deltas.std()
            # Compute percentile of the feature's mean delta in this image
            feature_delta = example['attribution_delta']
            percentile = scipy_stats.percentileofscore(all_deltas, feature_delta)
        else:
            delta_mean = delta_median = delta_std = percentile = np.nan

        # Collect stats for this image
        stats_row = {
            'rank': rank,
            'image_idx': img_idx,
            'n_patches': len(patch_indices),
            'feature_mean_delta': example['attribution_delta'],
            'percentile_in_image': percentile,
            'image_delta_mean': delta_mean,
            'image_delta_median': delta_median,
            'image_delta_std': delta_std,
            'gate_value': example['gate_value'],
        }
        if 'mean_abs_contribution' in example:
            stats_row['mean_abs_contribution'] = example['mean_abs_contribution']
        summary_stats.append(stats_row)

        # Load image using dataset-specific function
        image_path = get_image_path_covidquex(img_idx, image_dir)

        if image_path is None or not image_path.exists():
            continue

        img = Image.open(image_path).convert('RGB')

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img)

        # Overlay patches where feature is active
        patch_size = 16  # 224/14 for ViT
        patches_per_side = 14

        for i, (patch_idx, activation) in enumerate(zip(patch_indices, activations)):
            row_idx = patch_idx // patches_per_side
            col_idx = patch_idx % patches_per_side

            x = col_idx * patch_size
            y = row_idx * patch_size

            # Determine color based on contribution sign
            contrib = contributions[i] if i < len(contributions) else None
            if contrib is not None and contrib < 0:
                # Deboosting: red color
                color = 'red'
            else:
                # Boosting or unknown: green color
                color = 'lime'

            # Color intensity based on activation
            alpha = min(activation / 2.0, 0.8)  # Scale alpha
            rect = patches.Rectangle((x, y), patch_size, patch_size,
                                     linewidth=2, edgecolor=color, facecolor=color,
                                     alpha=alpha)
            ax.add_patch(rect)

        # Count boosting vs deboosting patches for legend
        n_boosters = sum(1 for c in contributions if c is not None and c > 0)
        n_deboosters = sum(1 for c in contributions if c is not None and c < 0)
        n_unknown = len(contributions) - n_boosters - n_deboosters

        # Title with metrics
        title_parts = [f"Rank {rank} | Image {img_idx} | Feature {feature_idx}"]
        title_parts.append(f"Green=Boost({n_boosters}), Red=Deboost({n_deboosters}), Unknown({n_unknown})")
        if 'mean_abs_contribution' in example:
            title_parts.append(f"Contrib: {example['mean_abs_contribution']:.6f}")
        title_parts.append(f"Δ: {example['attribution_delta']:.4f} ({percentile:.0f}%ile in image)")
        title_parts.append(f"Gate: {example['gate_value']:.2f}")
        title_parts.append(f"Image Δ dist: μ={delta_mean:.4f}, med={delta_median:.4f}")

        ax.set_title('\n'.join(title_parts), fontsize=9)
        ax.axis('off')

        plt.tight_layout()

        # Save individual image
        output_path = feature_dir / f"rank_{rank:02d}_img_{img_idx}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Save summary statistics to CSV
    summary_df = pd.DataFrame(summary_stats)
    summary_csv_path = feature_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"Saved {len(top_examples)} examples to {feature_dir}/")


if __name__ == "__main__":
    experiment_path = "./experiments/feature_gradient_sweep_20251110_210044"
    path_vanilla = Path(f"{experiment_path}/covidquex/vanilla/val/")
    path_gated = Path(f"{experiment_path}/covidquex/layers_2_3_4_kappa_0.5_topk_None_combined_clamp_10.0/val/")

    # Cache directory for saving/loading processed results
    cache_dir = Path(f"{experiment_path}/analysis_cache")

    faithfulness_vanilla = load_faithfulness_results(path_vanilla)
    faithfulness_gated = load_faithfulness_results(path_gated)

    # Check if cached results exist
    if (cache_dir / "merged_data").exists() and (cache_dir / "feature_stats").exists():
        print("\n=== Loading cached analysis results ===")
        merged_gated, feature_stats_per_layer, feature_impacts_per_layer = load_analysis_results(cache_dir)
    else:
        print("\n=== Computing analysis (no cache found) ===")
        features_debug_gated = load_features_debug(path_gated)
        merged_gated = merge_features_faithfulness(features_debug_gated, faithfulness_gated)

        feature_stats_per_layer = {}
        for layer_idx in merged_gated.keys():
            print(f"\n=== Computing Feature Statistics for Layer {layer_idx} ===")
            feature_stats_per_layer[layer_idx] = compute_feature_statistics(merged_gated, layer_idx=layer_idx)

        feature_impacts_per_layer = {}  # Will be computed below

    comparison = compare_vanilla_gated(faithfulness_vanilla, faithfulness_gated)

    # =========================================================================
    # TWO-STAGE ANALYSIS: Patch Impact → Feature Composition
    # =========================================================================
    print("\n" + "=" * 80)
    print("TWO-STAGE CAUSAL ANALYSIS")
    print("Stage 1: Which patches/gates drove faithfulness improvements?")
    print("Stage 2: Which features were present in those patches?")
    print("=" * 80)

    # Store results for later use and saving
    patch_impact_results_per_layer = {}
    feature_composition_results_per_layer = {}

    for layer_idx in merged_gated.keys():
        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*80}")

        # Stage 1: Patch impact analysis
        patch_impact = analyze_patch_impact_on_faithfulness(
            merged_gated,
            faithfulness_vanilla,
            faithfulness_gated,
            layer_idx=layer_idx,
            improvement_quantiles=(0.75, 0.25),  # Top 25% vs bottom 25%
            patch_quantile=0.80,  # Top 20% of patches within each image
            min_images=10
        )
        patch_impact_results_per_layer[layer_idx] = patch_impact

        # Stage 2: Feature composition analysis
        if patch_impact:  # Only if Stage 1 succeeded
            feature_composition = analyze_feature_composition_of_patches(
                merged_gated,
                patch_impact,
                layer_idx=layer_idx,
                min_occurrences=100,  # Require at least 100 total occurrences
                top_n=20
            )
            feature_composition_results_per_layer[layer_idx] = feature_composition

            # Save results
            output_dir = Path(f"{experiment_path}/two_stage_analysis")
            output_dir.mkdir(exist_ok=True, parents=True)

            # Save patch impact results
            if 'image_stratification' in patch_impact:
                patch_impact['image_stratification'].to_csv(
                    output_dir / f"layer_{layer_idx}_image_stratification.csv",
                    index=False
                )
            if 'effect_sizes' in patch_impact:
                patch_impact['effect_sizes'].to_csv(
                    output_dir / f"layer_{layer_idx}_effect_sizes.csv",
                    index=False
                )

            # Save feature composition results
            if not feature_composition.empty:
                feature_composition.to_csv(
                    output_dir / f"layer_{layer_idx}_feature_composition.csv",
                    index=False
                )

                # Print top 10 features for this layer (using new per-metric columns)
                print(f"\n{'='*60}")
                print(f"TOP 10 FEATURES FOR LAYER {layer_idx} (by enrichment)")
                print(f"{'='*60}\n")
                for _, row in feature_composition.head(10).iterrows():
                    feat_idx = int(row['feature_idx'])
                    n_beneficial = int(row['n_metrics_beneficial'])
                    n_harmful = int(row['n_metrics_harmful'])
                    combined_enrichment = row['enrichment_combined']
                    combined_spec = row['specificity_combined']

                    consistency_tag = ""
                    if n_beneficial == 3:
                        consistency_tag = " [ALL 3 BENEFICIAL]"
                    elif n_beneficial == 2:
                        consistency_tag = " [2/3 BENEFICIAL]"
                    elif n_harmful >= 2:
                        consistency_tag = " [HARMFUL]"

                    # Determine booster/debooster from SaCo contribution
                    contrib_type = ""
                    if row['mean_contribution_high_saco'] > 0:
                        contrib_type = " [BOOSTER]"
                    elif row['mean_contribution_high_saco'] < 0:
                        contrib_type = " [DEBOOSTER]"

                    print(f"Feature {feat_idx}{consistency_tag}{contrib_type}: enrichment={combined_enrichment:.2f}x, specificity={combined_spec:+.3f}")
                    print(f"  Per-metric enrichment: SaCo={row['enrichment_saco']:.2f}x, Faith={row['enrichment_faith']:.2f}x, Pixel={row['enrichment_pixel']:.2f}x")
                    print(f"  Per-metric specificity: SaCo={row['specificity_saco']:+.3f}, Faith={row['specificity_faith']:+.3f}, Pixel={row['specificity_pixel']:+.3f}")
                    print(f"  Counts - SaCo: high={int(row['count_high_saco'])}, low={int(row['count_low_saco'])} | Faith: high={int(row['count_high_faith'])}, low={int(row['count_low_faith'])} | Pixel: high={int(row['count_high_pixel'])}, low={int(row['count_low_pixel'])}")
                    print(f"  Activation - SaCo: high={row['mean_activation_high_saco']:.3f}, low={row['mean_activation_low_saco']:.3f} | Faith: high={row['mean_activation_high_faith']:.3f}, low={row['mean_activation_low_faith']:.3f} | Pixel: high={row['mean_activation_high_pixel']:.3f}, low={row['mean_activation_low_pixel']:.3f}")
                    print(f"  Contribution (SaCo): high={row['mean_contribution_high_saco']:+.2e}, low={row['mean_contribution_low_saco']:+.2e}")
                    print()

    print("\n" + "=" * 80)
    print("TWO-STAGE ANALYSIS COMPLETE")
    print(f"Results saved to: {experiment_path}/two_stage_analysis/")
    print("=" * 80 + "\n")

    # =========================================================================
    # VISUALIZE TOP ENRICHED FEATURES
    # =========================================================================
    print("\n" + "=" * 80)
    print("VISUALIZING TOP ENRICHED FEATURES")
    print("=" * 80 + "\n")

    image_dir = Path("./data/covidquex_unified/val")
    viz_output_dir = Path(f"{experiment_path}/enriched_feature_visualizations")

    # Number of features to visualize per layer
    n_features_to_viz = 30  # Extended to capture more features

    for layer_idx, feature_composition in feature_composition_results_per_layer.items():
        if feature_composition.empty:
            continue

        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx} - VISUALIZING TOP {n_features_to_viz} ENRICHED FEATURES")
        print(f"{'='*80}\n")

        # Visualize top N most enriched features
        top_features = feature_composition.head(n_features_to_viz)

        for rank, (_, row) in enumerate(top_features.iterrows(), 1):
            feat_idx = int(row['feature_idx'])
            # Use SaCo as primary metric for visualization
            enrichment = row['enrichment_saco']
            count_high = int(row['count_high_saco'])
            count_low = int(row['count_low_saco'])
            p_value = row['p_value_saco']
            specificity = row['specificity_combined']

            # Activation statistics (SaCo)
            act_high = row['mean_activation_high_saco']
            act_low = row['mean_activation_low_saco']
            act_diff = act_high - act_low
            act_diff_pct = (act_diff / act_low * 100) if act_low > 0 else 0

            # Contribution statistics (SaCo)
            contrib_high = row['mean_contribution_high_saco']
            contrib_low = row['mean_contribution_low_saco']
            contrib_diff = contrib_high - contrib_low
            contrib_diff_pct = (contrib_diff / contrib_low * 100) if contrib_low > 0 else 0

            # Significance marker
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

            print(f"\n[{rank}/{n_features_to_viz}] Feature {feat_idx}")
            print(f"{'='*60}")
            print(f"  Enrichment (SaCo): {enrichment:.2f}x{sig} (p={p_value:.4f})")
            print(f"  Specificity (combined): {specificity:+.3f}")
            print(f"  Occurrence (SaCo): high={count_high}, low={count_low}")
            print(f"  Frequencies (SaCo): high={row['freq_high_saco']:.6f}, low={row['freq_low_saco']:.6f}")
            print(f"  Activation: high={act_high:.3f}, low={act_low:.3f} "
                  f"(Δ={act_diff:+.3f}, {act_diff_pct:+.1f}%)")
            print(f"  Contribution: high={contrib_high:.2e}, low={contrib_low:.2e} "
                  f"(Δ={contrib_diff:+.2e}, {contrib_diff_pct:+.1f}%)")

            # Boosting vs Deboosting
            if contrib_high > 0:
                print(f"  → BOOSTER: Pushes toward predicted class (positive contribution)")
            elif contrib_high < 0:
                print(f"  → DEBOOSTER: Pushes away from predicted class (negative contribution)")

            # Interpretation hints
            if enrichment >= 3.0:
                print(f"  → HIGHLY SPECIFIC marker (appears 3x+ more in high-improvement images)")
            elif enrichment >= 2.0:
                print(f"  → STRONG marker (appears 2x+ more in high-improvement images)")
            elif enrichment >= 1.5:
                print(f"  → MODERATE marker")

            if abs(act_diff_pct) > 20:
                print(f"  → Activation differs substantially between groups ({act_diff_pct:+.1f}%)")

            if abs(contrib_diff_pct) > 50:
                print(f"  → Contribution differs substantially between groups ({contrib_diff_pct:+.1f}%)")

            print(f"\n  Generating 50-image visualization...")

            # Visualize this feature
            visualize_feature_detailed(
                feature_idx=feat_idx,
                layer_idx=layer_idx,
                merged_data=merged_gated,
                image_dir=image_dir,
                output_dir=viz_output_dir / f"layer_{layer_idx}",
                n_examples=50,
                sort_by='activation'  # Sort by activation strength
            )

            print(f"  ✓ Saved to: {viz_output_dir}/layer_{layer_idx}/feature_{feat_idx}/")

        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx} COMPLETE: {n_features_to_viz} features visualized")
        print(f"{'='*60}\n")

        # Also visualize top deboosting features
        deboosters = feature_composition[feature_composition['mean_contribution_high'] < 0]
        if len(deboosters) > 0:
            print(f"\n{'='*80}")
            print(f"LAYER {layer_idx} - VISUALIZING TOP 10 DEBOOSTING FEATURES")
            print(f"{'='*80}\n")

            top_deboosters = deboosters.head(10)

            for rank, (_, row) in enumerate(top_deboosters.iterrows(), 1):
                feat_idx = int(row['feature_idx'])
                # Use SaCo metric for deboosters
                enrichment = row['enrichment_saco']
                count_high = int(row['count_high_saco'])
                count_low = int(row['count_low_saco'])
                contrib_high = row['mean_contribution_high_saco']
                specificity = row['specificity_combined']

                print(f"\n[{rank}/10] DEBOOSTER Feature {feat_idx}")
                print(f"{'='*60}")
                print(f"  Enrichment (SaCo): {enrichment:.2f}x")
                print(f"  Specificity (combined): {specificity:+.3f}")
                print(f"  Counts (SaCo): high={count_high}, low={count_low}")
                print(f"  Contribution (SaCo): {contrib_high:.2e} (negative = suppresses class prediction)")
                print(f"\n  Generating 50-image visualization...")

                visualize_feature_detailed(
                    feature_idx=feat_idx,
                    layer_idx=layer_idx,
                    merged_data=merged_gated,
                    image_dir=image_dir,
                    output_dir=viz_output_dir / f"layer_{layer_idx}_deboosters",
                    n_examples=50,
                    sort_by='activation'
                )

                print(f"  ✓ Saved to: {viz_output_dir}/layer_{layer_idx}_deboosters/feature_{feat_idx}/")

            print(f"\n{'='*60}")
            print(f"LAYER {layer_idx} DEBOOSTERS COMPLETE: {len(top_deboosters)} features visualized")
            print(f"{'='*60}\n")

    # Generate summary report
    print("\n" + "=" * 80)
    print("FEATURE ENRICHMENT SUMMARY REPORT")
    print("=" * 80 + "\n")

    summary_report_path = viz_output_dir / "enrichment_summary_report.txt"
    with open(summary_report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE ENRICHMENT SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        for layer_idx, feature_composition in feature_composition_results_per_layer.items():
            if feature_composition.empty:
                continue

            f.write(f"\n{'='*80}\n")
            f.write(f"LAYER {layer_idx}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Total features meeting criteria: {len(feature_composition)}\n")
            f.write(f"Features visualized: {n_features_to_viz}\n\n")

            top_features = feature_composition.head(n_features_to_viz)

            # Categorize features by combined specificity
            all_beneficial = top_features[top_features['n_metrics_beneficial'] == 3]
            mostly_beneficial = top_features[top_features['n_metrics_beneficial'] == 2]
            mixed = top_features[(top_features['n_metrics_beneficial'] > 0) & (top_features['n_metrics_harmful'] > 0)]
            all_harmful = top_features[top_features['n_metrics_harmful'] == 3]

            f.write(f"Cross-Metric Consistency:\n")
            f.write(f"  All 3 metrics beneficial: {len(all_beneficial)} features\n")
            f.write(f"  2/3 metrics beneficial: {len(mostly_beneficial)} features\n")
            f.write(f"  Mixed effects: {len(mixed)} features\n")
            f.write(f"  All 3 metrics harmful: {len(all_harmful)} features\n\n")

            # Specificity statistics
            mean_specificity = top_features['specificity_combined'].mean()
            median_specificity = top_features['specificity_combined'].median()
            max_specificity = top_features['specificity_combined'].max()
            min_specificity = top_features['specificity_combined'].min()

            f.write(f"Specificity Statistics (Combined):\n")
            f.write(f"  Mean: {mean_specificity:+.3f}\n")
            f.write(f"  Median: {median_specificity:+.3f}\n")
            f.write(f"  Max: {max_specificity:+.3f}\n")
            f.write(f"  Min: {min_specificity:+.3f}\n\n")

            # Per-metric enrichment (using SaCo as example)
            mean_enrichment_saco = top_features['enrichment_saco'].mean()
            median_enrichment_saco = top_features['enrichment_saco'].median()

            f.write(f"Enrichment Statistics (SaCo):\n")
            f.write(f"  Mean: {mean_enrichment_saco:.2f}x\n")
            f.write(f"  Median: {median_enrichment_saco:.2f}x\n\n")

            # Occurrence patterns (SaCo)
            mean_count_high = top_features['count_high_saco'].mean()
            mean_count_low = top_features['count_low_saco'].mean()

            f.write(f"Occurrence Patterns (SaCo):\n")
            f.write(f"  Mean occurrences in high-improvement: {mean_count_high:.0f}\n")
            f.write(f"  Mean occurrences in low-improvement: {mean_count_low:.0f}\n")
            f.write(f"  Ratio: {mean_count_high/mean_count_low:.2f}x\n\n")

            # Activation patterns (SaCo)
            features_higher_act = top_features[top_features['mean_activation_high_saco'] >
                                              top_features['mean_activation_low_saco']]
            features_lower_act = top_features[top_features['mean_activation_high_saco'] <
                                             top_features['mean_activation_low_saco']]

            f.write(f"Activation Patterns (SaCo):\n")
            f.write(f"  Higher activation in high-improvement: {len(features_higher_act)} features\n")
            f.write(f"  Lower activation in high-improvement: {len(features_lower_act)} features\n")
            f.write(f"  Same activation: {len(top_features) - len(features_higher_act) - len(features_lower_act)} features\n\n")

            # Top 5 features detailed
            f.write(f"Top 5 Most Beneficial Features:\n")
            f.write("=" * 60 + "\n")
            for rank, (_, row) in enumerate(top_features.head(5).iterrows(), 1):
                f.write(f"\n{rank}. Feature {int(row['feature_idx'])}\n")
                f.write(f"   Combined Specificity: {row['specificity_combined']:+.3f}\n")
                f.write(f"   Per-metric specificity: SaCo={row['specificity_saco']:+.3f}, Faith={row['specificity_faith']:+.3f}, Pixel={row['specificity_pixel']:+.3f}\n")
                f.write(f"   Enrichment (SaCo): {row['enrichment_saco']:.2f}x\n")
                f.write(f"   Counts (SaCo): high={int(row['count_high_saco'])}, low={int(row['count_low_saco'])}\n")
                f.write(f"   Activation (SaCo): high={row['mean_activation_high_saco']:.3f}, low={row['mean_activation_low_saco']:.3f}\n")
                f.write(f"   Contribution (SaCo): high={row['mean_contribution_high_saco']:.2e}, low={row['mean_contribution_low_saco']:.2e}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Summary report saved to: {summary_report_path}")

    print("\n" + "=" * 80)
    print("FEATURE VISUALIZATION COMPLETE")
    print(f"Total features visualized: {n_features_to_viz} per layer")
    print(f"Results saved to: {viz_output_dir}/")
    print("=" * 80 + "\n")

    # =========================================================================
    # FOCUSED ATTRIBUTION CHANGE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("FOCUSED ATTRIBUTION CHANGE ANALYSIS")
    print("=" * 80)

    # Q1: Average attribution change per layer
    print("\n=== Q1: Average Attribution Change Per Layer ===")
    layer_changes = analyze_attribution_change_per_layer(merged_gated)
    print(layer_changes.to_string(index=False))

    # Q2: Boost vs Deboost magnitude
    print("\n=== Q2: Boost vs Deboost Magnitude ===")
    boost_deboost = analyze_boost_deboost_magnitude(merged_gated)
    print("\nBoost/Deboost Statistics:")
    for _, row in boost_deboost.iterrows():
        print(f"\nLayer {int(row['layer_idx'])}:")
        print(f"  Total patches: {row['n_total_patches']}")
        print(f"  Boosted ({row['pct_boosted']:.1f}%): mean_abs_delta={row['boost_mean_abs_delta']:.6f}, mean_delta={row['boost_mean_delta']:+.6f}")
        print(f"  Deboosted ({row['pct_deboosted']:.1f}%): mean_abs_delta={row['deboost_mean_abs_delta']:.6f}, mean_delta={row['deboost_mean_delta']:+.6f}")

    # # Q3: Distribution analysis per layer
    # print("\n=== Q3: Distribution of Attribution Deltas ===")
    # dist_output_dir = Path(f"{experiment_path}/focused_analysis/distributions")
    # for layer_idx in merged_gated.keys():
    #     print(f"\nLayer {layer_idx} distribution:")
    #     dist_stats = analyze_delta_distribution(
    #         merged_gated,
    #         layer_idx=layer_idx,
    #         n_bins=50,
    #         plot=True,
    #         output_path=dist_output_dir / f"layer_{layer_idx}_delta_distribution.png"
    #     )
    #     print(f"  Mean: {dist_stats['mean']:.6f}, Median: {dist_stats['median']:.6f}, Std: {dist_stats['std']:.6f}")
    #     print(f"  Range: [{dist_stats['min']:.6f}, {dist_stats['max']:.6f}]")
    #     print(f"  Q25: {dist_stats['q25']:.6f}, Q75: {dist_stats['q75']:.6f}")
    #     print(f"  Tail percentages: Near-zero={dist_stats['pct_near_zero']:.1f}%, Large-pos={dist_stats['pct_large_positive']:.1f}%, Large-neg={dist_stats['pct_large_negative']:.1f}%")

    # # Q3b: Feature-Faithfulness Correlation (IMAGE LEVEL)
    # print("\n=== Q3b: Feature-Faithfulness Correlation Analysis ===")
    # print("Image-level correlation: feature contribution → faithfulness improvement")
    # print("Bins: 100-1K (rare), 1K-10K (common), 10K-50K (frequent), >50K (ubiquitous)")
    # print("Metrics: SaCo, FaithfulnessCorr, PixelFlip\n")

    # image_dir = Path("./data/covidquex_unified/val")

    # for layer_idx in merged_gated.keys():
    #     print(f"\n{'='*60}")
    #     print(f"LAYER {layer_idx} - FAITHFULNESS CORRELATION")
    #     print(f"{'='*60}")

    #     stratified_results = analyze_feature_faithfulness_correlation(
    #         merged_gated,
    #         faithfulness_vanilla,
    #         faithfulness_gated,
    #         layer_idx=layer_idx,
    #         frequency_bins=[(100, 500), (500, 1000), (1000, 3000), (3000, float('inf'))],
    #         top_n_per_bin=10
    #     )

    #     if not stratified_results:
    #         print("  No features with sufficient data")
    #         continue

    #     for bin_name, bin_df in stratified_results.items():
    #         print(f"\n--- Frequency bin: {bin_name} images ---")

    #         # Three separate rankings by each metric
    #         metrics = [
    #             ('SaCo', 'corr_saco', 'p_saco', 'mean_delta_saco', 'percentile_saco'),
    #             ('FaithfulnessCorr', 'corr_faith', 'p_faith', 'mean_delta_faith', 'percentile_faith'),
    #             ('PixelFlipping', 'corr_pixel', 'p_pixel', 'mean_delta_pixel', 'percentile_pixel')
    #         ]

    #         top_features_all_metrics = set()

    #         for metric_name, corr_col, p_col, delta_col, percentile_col in metrics:
    #             print(f"\nTop features by {metric_name} correlation:")

    #             # Sort by this metric's correlation (absolute value for ranking)
    #             metric_sorted = bin_df.sort_values(corr_col, ascending=False, key=abs).head(5)

    #             for idx, row in metric_sorted.iterrows():
    #                 # Get significance markers for each metric
    #                 sig_saco = "***" if row['p_saco'] < 0.001 else "**" if row['p_saco'] < 0.01 else "*" if row['p_saco'] < 0.05 else ""
    #                 sig_faith = "***" if row['p_faith'] < 0.001 else "**" if row['p_faith'] < 0.01 else "*" if row['p_faith'] < 0.05 else ""
    #                 sig_pixel = "***" if row['p_pixel'] < 0.001 else "**" if row['p_pixel'] < 0.01 else "*" if row['p_pixel'] < 0.05 else ""

    #                 print(f"  Feature {int(row['feature_idx'])}: "
    #                       f"corr={row[corr_col]:+.3f}{sig_saco if metric_name=='SaCo' else sig_faith if metric_name=='FaithfulnessCorr' else sig_pixel} | "
    #                       f"SaCo={row['corr_saco']:+.3f}, "
    #                       f"Faith={row['corr_faith']:+.3f}, "
    #                       f"Pixel={row['corr_pixel']:+.3f} | "
    #                       f"n_img={int(row['n_images'])}")
    #                 print(f"    Delta: {row[delta_col]:+.6f} ({row[percentile_col]:.1f}%)")
    #                 print(f"    Within-image: "
    #                       f"med={row['percentile_median']:.0f}%ile, "
    #                       f"High={row['pct_high_impact']:.0f}% | "
    #                       f"patches={row['median_patches_per_image']:.0f}")

    #                 top_features_all_metrics.add(int(row['feature_idx']))

    #         # Save results
    #         focused_output_dir = Path(f"{experiment_path}/focused_analysis")
    #         stratified_output_dir = focused_output_dir / "faithfulness_correlation"
    #         stratified_output_dir.mkdir(exist_ok=True, parents=True)
    #         bin_df.to_csv(
    #             stratified_output_dir / f"layer_{layer_idx}_bin_{bin_name}.csv",
    #             index=False
    #         )

    #         # Visualize top 2 features from each metric (up to 6 total, or fewer if overlap)
    #         print(f"\n  Generating 50-image visualizations for top features from each metric...")
    #         features_to_viz = []
    #         for metric_name, corr_col, p_col, delta_col, percentile_col in metrics:
    #             metric_sorted = bin_df.sort_values(corr_col, ascending=False, key=abs).head(2)
    #             for _, row in metric_sorted.iterrows():
    #                 feat_idx = int(row['feature_idx'])
    #                 if feat_idx not in [f[0] for f in features_to_viz]:  # Avoid duplicates
    #                     features_to_viz.append((feat_idx, metric_name, row[corr_col]))

    #         for feat_idx, metric_name, corr_val in features_to_viz[:6]:  # Limit to 6 visualizations
    #             viz_dir = stratified_output_dir / f"layer_{layer_idx}" / f"bin_{bin_name}"
    #             print(f"    Feature {feat_idx} (top for {metric_name}, corr={corr_val:+.3f})...")
    #             visualize_feature_detailed(
    #                 feature_idx=feat_idx,
    #                 layer_idx=layer_idx,
    #                 merged_data=merged_gated,
    #                 image_dir=image_dir,
    #                 output_dir=viz_dir,
    #                 n_examples=50,
    #                 sort_by='contribution'
    #             )

    # COMMENTED OUT - Old variance explained analysis (invalid approach)
    # Q3b2: Pooled Cross-Layer Variance Explained
    # print("\n=== Q3b2: Pooled Cross-Layer Variance Explained ===")
    # print("Multiple regression across ALL layers: How much variance do features explain?\n")

    # pooled_variance_results = analyze_variance_explained_pooled(
    #         merged_gated,
    #         faithfulness_vanilla,
    #         faithfulness_gated,
    #         layer_indices=list(merged_gated.keys()),
    #         top_k_values= [1, 5, 10, 20, 50, 100, 500, 1000, 2000,3000,4000, 5000]
    #     )
    # 
    #     print(f"\n{'='*60}")
    #     print(f"POOLED RESULTS SUMMARY")
    #     print(f"{'='*60}")
    # 
    #     for metric_name, results_df in pooled_variance_results.items():
    #         if len(results_df) == 0:
    #             continue
    # 
    #         print(f"\n{metric_name}:")
    #         for _, row in results_df.iterrows():
    #             print(f"  Top {int(row['k']):3d} features: R²={row['r2_pct']:5.1f}%")
    # 
    #         # Find k for ~80% variance
    #         if len(results_df[results_df['r2'] >= 0.80]) > 0:
    #             k_80 = results_df[results_df['r2'] >= 0.80].iloc[0]['k']
    #             r2_80 = results_df[results_df['r2'] >= 0.80].iloc[0]['r2'] * 100
    #             print(f"  → {int(k_80)} features needed for ~80% variance (actual: {r2_80:.1f}%)")
    #         else:
    #             max_r2 = results_df['r2'].max() * 100
    #             max_k = results_df.loc[results_df['r2'].idxmax(), 'k']
    #             print(f"  → Maximum R²={max_r2:.1f}% with {int(max_k)} features (did not reach 80%)")
    # 
    #     # Save pooled results
    #     focused_output_dir = Path(f"{experiment_path}/focused_analysis")
    #     variance_output_dir = focused_output_dir / "variance_explained"
    #     variance_output_dir.mkdir(exist_ok=True, parents=True)
    # 
    #     for metric_name, results_df in pooled_variance_results.items():
    #         results_df.to_csv(
    #             variance_output_dir / f"pooled_all_layers_{metric_name}.csv",
    #             index=False
    #         )
    # 
    #     # Q3b3: Per-Layer Variance Explained (for comparison)
    #     print("\n=== Q3b3: Per-Layer Variance Explained (for comparison) ===")
    #     print("Multiple regression: How many features explain 80% of faithfulness variance?\n")
    # 
    #     for layer_idx in merged_gated.keys():
    #         print(f"\n{'='*60}")
    #         print(f"LAYER {layer_idx} - VARIANCE EXPLAINED")
    #         print(f"{'='*60}")
    # 
    #         variance_results = analyze_variance_explained(
    #             merged_gated,
    #             faithfulness_vanilla,
    #             faithfulness_gated,
    #             layer_idx=layer_idx,
    #             top_k_values=[1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
    #         )
    # 
    #         for metric_name, results_df in variance_results.items():
    #             if len(results_df) == 0:
    #                 continue
    # 
    #             print(f"\n{metric_name}:")
    #             for _, row in results_df.iterrows():
    #                 r2_pct = row['r2'] * 100
    #                 print(f"  Top {row['k']:3d} features: R²={r2_pct:5.1f}% variance explained")
    # 
    #             # Find k for ~80% variance
    #             if len(results_df[results_df['r2'] >= 0.80]) > 0:
    #                 k_80 = results_df[results_df['r2'] >= 0.80].iloc[0]['k']
    #                 print(f"  → {k_80} features needed for 80% variance")
    #             else:
    #                 max_r2 = results_df['r2'].max() * 100
    #                 max_k = results_df.loc[results_df['r2'].idxmax(), 'k']
    #                 print(f"  → Maximum R²={max_r2:.1f}% with {int(max_k)} features (< 80%)")
    # 
    #         # Save results
    #         focused_output_dir = Path(f"{experiment_path}/focused_analysis")
    #         variance_output_dir = focused_output_dir / "variance_explained"
    #         variance_output_dir.mkdir(exist_ok=True, parents=True)
    # 
    #         for metric_name, results_df in variance_results.items():
    #             results_df.to_csv(
    #                 variance_output_dir / f"layer_{layer_idx}_{metric_name}_variance.csv",
    #                 index=False)

    # Q3c: Class-stratified analysis
    print("\n=== Q3c: Class-Stratified Feature Analysis ===")
    print("Are features class-specific or universal?\n")

    for layer_idx in merged_gated.keys():
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx} - BY PREDICTED CLASS")
        print(f"{'='*60}")

        class_analysis = analyze_feature_contribution_by_class(
            merged_gated, layer_idx=layer_idx, stratify_by='predicted_class',
            min_occurrences=50, top_n=5
        )

        for class_name, class_df in class_analysis.items():
            print(f"\n--- Class: {class_name} ---")
            print(f"Top 5 features for this class:")
            for idx, row in class_df.iterrows():
                sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                print(f"  Feature {int(row['feature_idx'])}: "
                      f"corr={row['correlation']:+.4f}{sig}, "
                      f"effect={row['effect_size']:+.3f}, "
                      f"n={row['n_occurrences']}")

            # Save per-class results
            focused_output_dir = Path(f"{experiment_path}/focused_analysis")
            class_output_dir = focused_output_dir / "by_class"
            class_output_dir.mkdir(exist_ok=True, parents=True)
            class_df.to_csv(
                class_output_dir / f"layer_{layer_idx}_class_{class_name}_features.csv",
                index=False
            )

    # Q3d: Failure analysis - when does gating make things worse?
    print("\n=== Q3d: Failure Analysis ===")
    print("Analyzing features that appear when gating degrades faithfulness\n")

    for layer_idx in merged_gated.keys():
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx} - FAILURE ANALYSIS")
        print(f"{'='*60}")

        # Analyze failures for SaCo score
        print("\n--- Based on SaCo Score ---")
        failure_analysis_saco = analyze_failure_cases(
            merged_gated, faithfulness_vanilla, faithfulness_gated,
            layer_idx=layer_idx, metric='saco_score', min_occurrences=50
        )

        if not failure_analysis_saco.empty:
            print("\nTop 10 features appearing in SaCo failures:")
            top_failures = failure_analysis_saco.head(10)
            for idx, row in top_failures.iterrows():
                print(f"  Feature {int(row['feature_idx'])}: "
                      f"failure_rate={row['failure_rate']:.2%}, "
                      f"contrib_diff={row['contrib_diff']:+.6f}, "
                      f"failures={row['failure_count']}/{row['total_count']}")

            # Save results
            focused_output_dir = Path(f"{experiment_path}/focused_analysis")
            failure_output_dir = focused_output_dir / "failures"
            failure_output_dir.mkdir(exist_ok=True, parents=True)
            failure_analysis_saco.to_csv(
                failure_output_dir / f"layer_{layer_idx}_saco_failure_features.csv",
                index=False
            )

        # Analyze failures for FaithfulnessCorrelation
        if 'FaithfulnessCorrelation' in faithfulness_vanilla.columns:
            print("\n--- Based on FaithfulnessCorrelation ---")
            failure_analysis_faith = analyze_failure_cases(
                merged_gated, faithfulness_vanilla, faithfulness_gated,
                layer_idx=layer_idx, metric='FaithfulnessCorrelation', min_occurrences=50
            )

            if not failure_analysis_faith.empty:
                print("\nTop 10 features appearing in FaithfulnessCorrelation failures:")
                top_failures = failure_analysis_faith.head(10)
                for idx, row in top_failures.iterrows():
                    print(f"  Feature {int(row['feature_idx'])}: "
                          f"failure_rate={row['failure_rate']:.2%}, "
                          f"contrib_diff={row['contrib_diff']:+.6f}, "
                          f"failures={row['failure_count']}/{row['total_count']}")

                # Save results
                failure_analysis_faith.to_csv(
                    failure_output_dir / f"layer_{layer_idx}_faithcorr_failure_features.csv",
                    index=False
                )

    # Q4: Top attribution-changing features (weighted impact)
    print("\n=== Q4: Top Attribution-Changing Features (Weighted Impact) ===")
    print("Using weighted impact = mean_delta × sqrt(n_occurrences)")
    print("This balances high per-occurrence impact with reasonable frequency\n")

    image_dir = Path("./data/covidquex_unified/val")

    for layer_idx in merged_gated.keys():
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*60}")

        # Top boosting features
        print("\nTop 10 BOOSTING features (weighted impact):")
        top_boost = find_top_attribution_changing_features(
            merged_gated, layer_idx, top_n=10, mode='boost', min_occurrences=100
        )

        if len(top_boost) == 0:
            print("  No features found with >= 100 occurrences")
        else:
            for idx, row in top_boost.iterrows():
                print(f"  Feature {int(row['feature_idx'])}: "
                      f"weighted_impact={row['weighted_impact_signed']:+.4f}, "
                      f"mean_delta={row['mean_delta']:+.6f} ± {row['std_delta']:.6f}, "
                      f"n_occur={row['n_occurrences']}")

        # Top deboosting features
        print("\nTop 10 DEBOOSTING features (weighted impact):")
        top_deboost = find_top_attribution_changing_features(
            merged_gated, layer_idx, top_n=10, mode='deboost', min_occurrences=100
        )

        if len(top_deboost) == 0:
            print("  No features found with >= 100 occurrences")
        else:
            for idx, row in top_deboost.iterrows():
                print(f"  Feature {int(row['feature_idx'])}: "
                      f"weighted_impact={row['weighted_impact_signed']:+.4f}, "
                      f"mean_delta={row['mean_delta']:+.6f} ± {row['std_delta']:.6f}, "
                      f"n_occur={row['n_occurrences']}")

        # Visualize top 3 boosting and top 3 deboosting features
        print(f"\nGenerating visualizations for layer {layer_idx}...")
        viz_output_dir = Path(f"{experiment_path}/focused_analysis/feature_examples")

        # Top 3 boosting
        for i, (_, row) in enumerate(top_boost.head(3).iterrows()):
            feat_idx = int(row['feature_idx'])
            print(f"  Visualizing boosting feature {feat_idx}...")
            visualize_top_feature_examples(
                feature_idx=feat_idx,
                layer_idx=layer_idx,
                merged_data=merged_gated,
                image_dir=image_dir,
                output_dir=viz_output_dir / f"layer_{layer_idx}" / "boosting",
                n_examples=5,
                sort_by='attribution_delta'
            )

        # Top 3 deboosting
        for i, (_, row) in enumerate(top_deboost.head(3).iterrows()):
            feat_idx = int(row['feature_idx'])
            print(f"  Visualizing deboosting feature {feat_idx}...")
            visualize_top_feature_examples(
                feature_idx=feat_idx,
                layer_idx=layer_idx,
                merged_data=merged_gated,
                image_dir=image_dir,
                output_dir=viz_output_dir / f"layer_{layer_idx}" / "deboosting",
                n_examples=5,
                sort_by='attribution_delta'
            )

    # Save focused analysis results
    focused_output_dir = Path(f"{experiment_path}/focused_analysis")
    focused_output_dir.mkdir(parents=True, exist_ok=True)
    layer_changes.to_csv(focused_output_dir / "q1_layer_changes.csv", index=False)
    boost_deboost.to_csv(focused_output_dir / "q2_boost_deboost.csv", index=False)

    # Save top features per layer (Q4 - weighted impact)
    for layer_idx in merged_gated.keys():
        top_boost = find_top_attribution_changing_features(
            merged_gated, layer_idx, top_n=50, mode='boost', min_occurrences=100
        )
        top_deboost = find_top_attribution_changing_features(
            merged_gated, layer_idx, top_n=50, mode='deboost', min_occurrences=100
        )
        top_boost.to_csv(focused_output_dir / f"q4_layer_{layer_idx}_top_boost_features.csv", index=False)
        top_deboost.to_csv(focused_output_dir / f"q4_layer_{layer_idx}_top_deboost_features.csv", index=False)

    print(f"\n✓ Focused analysis results saved to {focused_output_dir}/")

    # # Compute feature impact analysis for each layer (if not already cached)
    # if not feature_impacts_per_layer:
    # print("\n" + "=" * 80)
    # print("COMPUTING FEATURE IMPACT ANALYSIS")
    # print("=" * 80)
    # 
    # for layer_idx in merged_gated.keys():
    # feature_impacts_per_layer[layer_idx] = compute_feature_impact_discovery(
    # merged_data=merged_gated,
    # faithfulness_vanilla=faithfulness_vanilla,
    # faithfulness_gated=faithfulness_gated,
    # layer_idx=layer_idx,
    # min_occurrences=50
    # )
    # 
    # # Save feature impacts to cache
    # print("\n=== Saving feature impacts to cache ===")
    # save_analysis_results(
    # merged_gated, feature_stats_per_layer, cache_dir, feature_impacts=feature_impacts_per_layer
    # )
    # else:
    # print("\n=== Using cached feature impacts ===")
    # print(f"Loaded {len(feature_impacts_per_layer)} layers")
    # 
    # # Analyze and visualize impactful features
    # for layer_idx, impact_df in feature_impacts_per_layer.items():
    # if impact_df.empty:
    # print(f"\n=== Skipping Layer {layer_idx} (no data) ===")
    # continue
    # 
    # print(f"\n{'='*80}")
    # print(f"LAYER {layer_idx} IMPACT ANALYSIS")
    # print(f"{'='*80}")
    # 
    # # Sort by interpretability score (impact + spatial consistency)
    # has_spatial = 'boost_interpretability_score' in impact_df.columns
    # 
    # if has_spatial:
    # boost_interpretable = impact_df.sort_values('boost_interpretability_score', ascending=False)
    # deboost_interpretable = impact_df.sort_values('deboost_interpretability_score', ascending=False)
    # 
    # print(f"\n=== Top 10 Most INTERPRETABLE Boosted Features ===")
    # print("(High impact + spatially consistent + frequent)")
    # top_boosted = boost_interpretable.head(10)
    # else:
    # boost_interpretable = impact_df.sort_values('boost_total_impact', ascending=False)
    # deboost_interpretable = impact_df.sort_values('deboost_total_impact', ascending=False)
    # 
    # print(f"\n=== Top 10 Most Impactful BOOSTED Features ===")
    # print("(Features that when boosted, improved faithfulness across all 3 metrics)")
    # top_boosted = boost_interpretable.head(10)
    # 
    # for idx, row in top_boosted.iterrows():
    # print(f"\nFeature {int(row['feature_idx'])}:")
    # print(f"  Appears in {row['n_boosted']} images when boosted (avg gate: {row['boost_avg_gate']:.3f})")
    # print(
    # f"  ΔSaCo:  {row['boost_delta_saco_mean']:+.4f} ± {row['boost_delta_saco_std']:.4f} " +
    # f"({row['boost_saco_improvement_rate']*100:.0f}% improved)"
    # )
    # print(
    # f"  ΔFaith: {row['boost_delta_faith_mean']:+.4f} ± {row['boost_delta_faith_std']:.4f} " +
    # f"({row['boost_faith_improvement_rate']*100:.0f}% improved)"
    # )
    # print(
    # f"  ΔPixel: {row['boost_delta_pixel_mean']:+.4f} ± {row['boost_delta_pixel_std']:.4f} " +
    # f"({row['boost_pixel_improvement_rate']*100:.0f}% improved)"
    # )
    # print(
    # f"  Metrics improved: {int(row['boost_metrics_improved'])}/3 | Impact score: {row['boost_total_impact']:.4f}"
    # )
    # if has_spatial:
    # # Convert patch indices to (row, col) for 14x14 grid
    # top_patches = row['top_patches'][:3] if isinstance(row['top_patches'], list) else []
    # patch_coords = [f"({p//14},{p%14})" for p in top_patches]
    # print(
    # f"  Spatial entropy: {row['spatial_entropy']:.3f} | " +
    # f"Active in {row['n_unique_patches']}/196 patches | " + f"Top locations: {', '.join(patch_coords)}"
    # )
    # print(f"  Interpretability score: {row['boost_interpretability_score']:.4f}")
    # 
    # if has_spatial:
    # print(f"\n=== Top 10 Most INTERPRETABLE Deboosted Features ===")
    # print("(High impact + spatially consistent + frequent)")
    # top_deboosted = deboost_interpretable.head(10)
    # else:
    # print(f"\n=== Top 10 Most Impactful DEBOOSTED Features ===")
    # print("(Features that when deboosted, improved faithfulness across all 3 metrics)")
    # top_deboosted = deboost_interpretable.head(10)
    # 
    # for idx, row in top_deboosted.iterrows():
    # print(f"\nFeature {int(row['feature_idx'])}:")
    # print(f"  Appears in {row['n_deboosted']} images when deboosted (avg gate: {row['deboost_avg_gate']:.3f})")
    # print(
    # f"  ΔSaCo:  {row['deboost_delta_saco_mean']:+.4f} ± {row['deboost_delta_saco_std']:.4f} " +
    # f"({row['deboost_saco_improvement_rate']*100:.0f}% improved)"
    # )
    # print(
    # f"  ΔFaith: {row['deboost_delta_faith_mean']:+.4f} ± {row['deboost_delta_faith_std']:.4f} " +
    # f"({row['deboost_faith_improvement_rate']*100:.0f}% improved)"
    # )
    # print(
    # f"  ΔPixel: {row['deboost_delta_pixel_mean']:+.4f} ± {row['deboost_delta_pixel_std']:.4f} " +
    # f"({row['deboost_pixel_improvement_rate']*100:.0f}% improved)"
    # )
    # print(
    # f"  Metrics improved: {int(row['deboost_metrics_improved'])}/3 | Impact score: {row['deboost_total_impact']:.4f}"
    # )
    # if has_spatial:
    # top_patches = row['top_patches'][:3] if isinstance(row['top_patches'], list) else []
    # patch_coords = [f"({p//14},{p%14})" for p in top_patches]
    # print(
    # f"  Spatial entropy: {row['spatial_entropy']:.3f} | " +
    # f"Active in {row['n_unique_patches']}/196 patches | " + f"Top locations: {', '.join(patch_coords)}"
    # )
    # print(f"  Interpretability score: {row['deboost_interpretability_score']:.4f}")

        # # Visualize top impactful boosted features (COMMENTED OUT - already generated)
        # print(f"\n=== Visualizing Top 10 Impactful Boosted Features ===")
        # image_dir = Path("data/covidquex_unified/val")
        # for idx, row in top_boosted.iterrows():
        #     feat_idx = int(row['feature_idx'])
        #     print(f"  Visualizing feature {feat_idx} (impact: {row['boost_total_impact']:.4f})")
        #     output_dir = Path(f"{experiment_path}/visualizations_impact/layer_{layer_idx}/boosted/feature_{feat_idx}")
        #     visualize_feature_activations(
        #         feature_idx=feat_idx,
        #         layer_idx=layer_idx,
        #         merged_data=merged_gated,
        #         faithfulness_vanilla=faithfulness_vanilla,
        #         faithfulness_gated=faithfulness_gated,
        #         image_dir=image_dir,
        #         output_dir=output_dir,
        #         top_k=50,
        #         sort_by='gate'
        #     )

        # # Visualize top impactful deboosted features (COMMENTED OUT - already generated)
        # print(f"\n=== Visualizing Top 10 Impactful Deboosted Features ===")
        # for idx, row in top_deboosted.iterrows():
        #     feat_idx = int(row['feature_idx'])
        #     print(f"  Visualizing feature {feat_idx} (impact: {row['deboost_total_impact']:.4f})")
        #     output_dir = Path(f"{experiment_path}/visualizations_impact/layer_{layer_idx}/deboosted/feature_{feat_idx}")
        #     visualize_feature_activations(
        #         feature_idx=feat_idx,
        #         layer_idx=layer_idx,
        #         merged_data=merged_gated,
        #         faithfulness_vanilla=faithfulness_vanilla,
        #         faithfulness_gated=faithfulness_gated,
        #         image_dir=image_dir,
        #         output_dir=output_dir,
        #         top_k=50,
        #         sort_by='gate',
        #         sort_ascending=True
        #     )

    # =========================================================================
    # OUTLIER DETECTION (replaces clustering analysis)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SPATIAL OUTLIER ANALYSIS")
    print("=" * 80)

    classes = ['COVID-19', 'Non-COVID', 'Normal']

    # Set paths for attribution loading (needed for spatial outlier sorting)
    gated_attr_dir = path_gated / "attributions"

    for layer_idx, impact_df in feature_impacts_per_layer.items():
        if impact_df.empty:
            print(f"\nSkipping layer {layer_idx} (no features)")
            continue

        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx} OUTLIERS")
        print(f"{'='*80}")

        find_spatial_outliers(impact_df, layer_idx, classes, merged_data=merged_gated, gated_attr_dir=gated_attr_dir)
        find_functional_outliers(impact_df, layer_idx, classes)

    # =========================================================================
    # VISUALIZE SPATIAL OUTLIERS
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING SPATIAL OUTLIER VISUALIZATIONS")
    print("=" * 80)

    # Set paths for visualization
    image_dir = Path("./data/covidquex_unified/val")
    visualization_output_dir = Path(f"{experiment_path}/spatial_visualizations")

    # Attribution directories
    vanilla_attr_dir = path_vanilla / "attributions"

    for layer_idx, impact_df in feature_impacts_per_layer.items():
        if impact_df.empty:
            continue

        # TODO: Remove this filter - temporarily only visualizing layers 3 and 4
        if layer_idx not in [3, 4]:
            print(f"\nSkipping layer {layer_idx} visualization (temporarily only doing layers 3 and 4)")
            continue

        # visualize_spatial_outliers(
            # impact_df=impact_df,
            # layer_idx=layer_idx,
            # merged_data=merged_gated,
            # faithfulness_vanilla=faithfulness_vanilla,
            # faithfulness_gated=faithfulness_gated,
            # image_dir=image_dir,
            # output_dir=visualization_output_dir,
            # vanilla_attr_dir=vanilla_attr_dir,
            # gated_attr_dir=gated_attr_dir,
            # top_n_per_category=10,
            # top_k_images=100
        # )

    # =========================================================================
    # VISUALIZE TOP CORRELATION FEATURES
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING TOP CORRELATION FEATURE VISUALIZATIONS")
    print("=" * 80)

    correlation_output_dir = Path(f"{experiment_path}/correlation_visualizations")

    for layer_idx, impact_df in feature_impacts_per_layer.items():
        if impact_df.empty:
            continue

        # TODO: Remove this filter - temporarily only visualizing layers 3 and 4
        if layer_idx not in [3, 4]:
            print(f"\nSkipping layer {layer_idx} correlation visualization (temporarily only doing layers 3 and 4)")
            continue

        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx} CORRELATION VISUALIZATIONS")
        print(f"{'='*80}")

        # Visualize top 3 features for each metric's correlation
        # SaCo correlation (boost)
        # visualize_top_correlation_features(
        # impact_df=impact_df,
        # layer_idx=layer_idx,
        # merged_data=merged_gated,
        # faithfulness_vanilla=faithfulness_vanilla,
        # faithfulness_gated=faithfulness_gated,
        # image_dir=image_dir,
        # output_dir=correlation_output_dir,
        # vanilla_attr_dir=vanilla_attr_dir,
        # gated_attr_dir=gated_attr_dir,
        # correlation_type='boost_impact_gate_correlation',
        # metric_name='SaCo_boost',
        # top_n=10,
        # top_k_images=100
        # )
        # 
        # # Faith correlation (boost)
        # visualize_top_correlation_features(
        # impact_df=impact_df,
        # layer_idx=layer_idx,
        # merged_data=merged_gated,
        # faithfulness_vanilla=faithfulness_vanilla,
        # faithfulness_gated=faithfulness_gated,
        # image_dir=image_dir,
        # output_dir=correlation_output_dir,
        # vanilla_attr_dir=vanilla_attr_dir,
        # gated_attr_dir=gated_attr_dir,
        # correlation_type='boost_impact_gate_correlation_faith',
        # metric_name='Faith_boost',
        # top_n=10,
        # top_k_images=100
        # )
        # 
        # # Pixel correlation (boost)
        # visualize_top_correlation_features(
        # impact_df=impact_df,
        # layer_idx=layer_idx,
        # merged_data=merged_gated,
        # faithfulness_vanilla=faithfulness_vanilla,
        # faithfulness_gated=faithfulness_gated,
        # image_dir=image_dir,
        # output_dir=correlation_output_dir,
        # vanilla_attr_dir=vanilla_attr_dir,
        # gated_attr_dir=gated_attr_dir,
        # correlation_type='boost_impact_gate_correlation_pixel',
        # metric_name='Pixel_boost',
        # top_n=10,
        # top_k_images=100
        # )


