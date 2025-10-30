import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

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
            'gate_values': array [n_images, 196]
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
            'gate_values': data['gate_values']
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

            if img_idx == 0:
                print(
                    f"  Image 0: gate_values shape: {img_gate_values.shape if hasattr(img_gate_values, 'shape') else type(img_gate_values)}"
                )
                print(
                    f"  Image 0: sparse_indices type: {type(img_sparse_indices)}, len: {len(img_sparse_indices) if hasattr(img_sparse_indices, '__len__') else 'N/A'}"
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
                    'active_features': img_sparse_indices[patch_idx],
                    'feature_activations': img_sparse_activations[patch_idx],
                    'feature_gradients': img_sparse_gradients[patch_idx],
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
):
    """
    Save merged data and feature statistics to disk for later use.

    Args:
        merged_data: Output from merge_features_faithfulness
        feature_stats: Output from compute_feature_statistics
        output_path: Directory to save results
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

    print("Done saving!")


def load_analysis_results(input_path: Path) -> tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Load previously saved analysis results.

    Args:
        input_path: Directory where results were saved

    Returns:
        Tuple of (merged_data, feature_stats)
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

    print("Done loading!")
    return merged_data, feature_stats


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


def visualize_feature_activations(
    feature_idx: int,
    layer_idx: int,
    merged_data: Dict[int, pd.DataFrame],
    faithfulness_vanilla: pd.DataFrame,
    faithfulness_gated: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    top_k: int = 50,
    sort_by: str = 'activation',
    patch_size: int = 16,
    image_size: int = 224,
):
    """
    Visualize where a specific feature activates across images.

    Creates side-by-side visualizations:
    - Left: Full image with bounding boxes around patches where feature is active
    - Right: Same image with gate values written on those patches

    Args:
        feature_idx: The feature to visualize
        layer_idx: Which layer this feature is from
        merged_data: Output from merge_features_faithfulness
        faithfulness_vanilla: Vanilla faithfulness results for comparison
        faithfulness_gated: Gated faithfulness results
        image_dir: Directory containing original images
        output_dir: Where to save visualizations
        top_k: How many top images to visualize
        sort_by: Sort images by 'activation' or 'gate'
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
        'FaithfulnessCorrelation': 'first',
        'PixelFlipping': 'first',
    }).reset_index()

    # Sort by the requested metric
    if sort_by == 'activation':
        images_with_feature['sort_value'] = images_with_feature['activation'].apply(np.mean)
    elif sort_by == 'gate':
        images_with_feature['sort_value'] = images_with_feature['gate_value'].apply(np.mean)
    else:
        raise ValueError(f"sort_by must be 'activation' or 'gate', got '{sort_by}'")

    images_with_feature = images_with_feature.sort_values('sort_value', ascending=False).head(top_k)

    print(f"Visualizing top {len(images_with_feature)} images sorted by {sort_by}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
        small_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 16)
        tiny_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        tiny_font = ImageFont.load_default()

    patches_per_side = image_size // patch_size

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

        # Create two copies
        img_boxes = img.copy()
        img_values = img.copy()

        draw_boxes = ImageDraw.Draw(img_boxes)
        draw_values = ImageDraw.Draw(img_values)

        # Draw patches
        for patch_idx, activation, gate_val in zip(patch_indices, activations, gate_values):
            # Convert patch index to (row, col) coordinates
            patch_row = patch_idx // patches_per_side
            patch_col = patch_idx % patches_per_side

            # Calculate pixel coordinates
            x1 = patch_col * patch_size
            y1 = patch_row * patch_size
            x2 = x1 + patch_size
            y2 = y1 + patch_size

            # Draw bounding box (left image)
            box_color = (0, 255, 0) if gate_val > 1.0 else (255, 0, 0)
            draw_boxes.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

            # Draw gate value text (right image)
            text = f"{gate_val:.2f}"
            # Create semi-transparent background for text
            draw_values.rectangle([x1, y1, x2, y2], outline=box_color, width=1)
            draw_values.text((x1 + 2, y1 + 2), text, fill=box_color, font=small_font)

        # Create side-by-side image with more padding for text
        top_padding = 80
        bottom_padding = 120
        side_spacing = 30
        combined = Image.new(
            'RGB', (image_size * 2 + side_spacing, image_size + top_padding + bottom_padding), color=(255, 255, 255)
        )
        combined.paste(img_boxes, (0, top_padding))
        combined.paste(img_values, (image_size + side_spacing, top_padding))

        # Add metadata text
        draw_combined = ImageDraw.Draw(combined)

        # Top metadata
        title = f"Feature {feature_idx} (Layer {layer_idx}) - Image {img_idx}"
        draw_combined.text((15, 15), title, fill=(0, 0, 0), font=font)
        draw_combined.text(
            (15, 45),
            f"Patches: {len(patch_indices)} | Avg Act: {np.mean(activations):.3f} | Avg Gate: {np.mean(gate_values):.3f}",
            fill=(0, 0, 0),
            font=small_font
        )

        # Bottom metadata
        y_offset = image_size + top_padding + 15
        saco_change = gated_row['saco_score'] - vanilla_row['saco_score']
        faith_change = gated_row['FaithfulnessCorrelation'] - vanilla_row['FaithfulnessCorrelation']
        pixel_change = gated_row['PixelFlipping'] - vanilla_row['PixelFlipping']

        draw_combined.text((15, y_offset),
                           f"SaCo: {gated_row['saco_score']:.3f} (Δ{saco_change:+.3f})",
                           fill=(0, 0, 0),
                           font=small_font)
        draw_combined.text((15, y_offset + 22),
                           f"Faith.Corr: {gated_row['FaithfulnessCorrelation']:.3f} (Δ{faith_change:+.3f})",
                           fill=(0, 0, 0),
                           font=small_font)
        draw_combined.text((15, y_offset + 44),
                           f"PixelFlip: {gated_row['PixelFlipping']:.3f} (Δ{pixel_change:+.3f})",
                           fill=(0, 0, 0),
                           font=small_font)

        # Add legend
        draw_combined.text((image_size + side_spacing + 15, y_offset),
                           "Green: Boosted (gate > 1.0)",
                           fill=(0, 200, 0),
                           font=tiny_font)
        draw_combined.text((image_size + side_spacing + 15, y_offset + 18),
                           "Red: Deboosted (gate < 1.0)",
                           fill=(200, 0, 0),
                           font=tiny_font)

        # Save
        output_path = output_dir / f"feature_{feature_idx}_layer_{layer_idx}_image_{img_idx:05d}.png"
        combined.save(output_path)

    print(f"Saved {len(images_with_feature)} visualizations to {output_dir}")


if __name__ == "__main__":
    experiment_path = "./experiments/feature_gradient_sweep_20251029_224838"
    path_vanilla = Path(f"{experiment_path}/covidquex/vanilla/val/")
    path_gated = Path(f"{experiment_path}/covidquex/layers_2_3_4_kappa_0.5_topk_None_combined_clamp_10.0/val/")

    # Cache directory for saving/loading processed results
    cache_dir = Path(f"{experiment_path}/analysis_cache")

    faithfulness_vanilla = load_faithfulness_results(path_vanilla)
    faithfulness_gated = load_faithfulness_results(path_gated)

    # Check if cached results exist
    if (cache_dir / "merged_data").exists() and (cache_dir / "feature_stats").exists():
        print("\n=== Loading cached analysis results ===")
        merged_gated, feature_stats_per_layer = load_analysis_results(cache_dir)
    else:
        print("\n=== Computing analysis (no cache found) ===")
        features_debug_gated = load_features_debug(path_gated)
        merged_gated = merge_features_faithfulness(features_debug_gated, faithfulness_gated)

        feature_stats_per_layer = {}
        for layer_idx in merged_gated.keys():
            print(f"\n=== Computing Feature Statistics for Layer {layer_idx} ===")
            feature_stats_per_layer[layer_idx] = compute_feature_statistics(merged_gated, layer_idx=layer_idx)

        # Save results for future use
        print("\n=== Saving results to cache ===")
        save_analysis_results(merged_gated, feature_stats_per_layer, cache_dir)

    comparison = compare_vanilla_gated(faithfulness_vanilla, faithfulness_gated)

    for layer_idx, feature_stats in feature_stats_per_layer.items():
        if feature_stats.empty:
            print(f"\n=== Skipping Layer {layer_idx} (no data) ===")
            continue

        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx} ANALYSIS")
        print(f"{'='*80}")

        print(f"\n=== Top 20 Most Common Features ===")
        print(feature_stats.head(20)[['feature_idx', 'n_occurrences', 'mean_gate', 'mean_saco_score']])

        print(f"\n=== Features with Highest Average SaCo ===")
        high_saco_features = feature_stats.nlargest(10, 'mean_saco_score')
        print(high_saco_features[['feature_idx', 'n_occurrences', 'mean_gate', 'mean_saco_score']])

        print(f"\n=== Features with Lowest Average SaCo ===")
        low_saco_features = feature_stats.nsmallest(10, 'mean_saco_score')
        print(low_saco_features[['feature_idx', 'n_occurrences', 'mean_gate', 'mean_saco_score']])

        if 'mean_faith_corr' in feature_stats.columns:
            valid_faith_corr = feature_stats.dropna(subset=['mean_faith_corr'])
            if not valid_faith_corr.empty:
                print(f"\n=== Features with Highest FaithfulnessCorrelation ===")
                high_faith = valid_faith_corr.nlargest(10, 'mean_faith_corr')
                print(high_faith[['feature_idx', 'n_occurrences', 'mean_gate', 'mean_faith_corr']])

                print(f"\n=== Features with Lowest FaithfulnessCorrelation ===")
                low_faith = valid_faith_corr.nsmallest(10, 'mean_faith_corr')
                print(low_faith[['feature_idx', 'n_occurrences', 'mean_gate', 'mean_faith_corr']])

        if 'mean_pixel_flip' in feature_stats.columns:
            valid_pixel_flip = feature_stats.dropna(subset=['mean_pixel_flip'])
            if not valid_pixel_flip.empty:
                print(f"\n=== Features with Highest PixelFlipping ===")
                high_pixel = valid_pixel_flip.nlargest(10, 'mean_pixel_flip')
                print(high_pixel[['feature_idx', 'n_occurrences', 'mean_gate', 'mean_pixel_flip']])

                print(f"\n=== Features with Lowest PixelFlipping ===")
                low_pixel = valid_pixel_flip.nsmallest(10, 'mean_pixel_flip')
                print(low_pixel[['feature_idx', 'n_occurrences', 'mean_gate', 'mean_pixel_flip']])

        print(f"\n=== Most Boosted Features (Highest mean_gate) ===")
        most_boosted = feature_stats.nlargest(10, 'mean_gate')
        print(most_boosted.head(10)[['feature_idx', 'n_occurrences', 'mean_gate', 'mean_saco_score']])

        print(f"\n=== Most Deboosted Features (Lowest mean_gate) ===")
        most_deboosted = feature_stats.nsmallest(10, 'mean_gate')
        print(most_deboosted.head(10)[['feature_idx', 'n_occurrences', 'mean_gate', 'mean_saco_score']])

        # Visualize top boosted and deboosted features
        print(f"\n=== Visualizing Top 10 Boosted Features ===")
        image_dir = Path("data/covidquex_unified/val")
        for idx, row in most_boosted.iterrows():
            feat_idx = row['feature_idx']
            print(f"  Visualizing feature {feat_idx} (mean_gate: {row['mean_gate']:.3f})")
            output_dir = Path(f"{experiment_path}/visualizations/layer_{layer_idx}/boosted/feature_{feat_idx}")
            visualize_feature_activations(
                feature_idx=feat_idx,
                layer_idx=layer_idx,
                merged_data=merged_gated,
                faithfulness_vanilla=faithfulness_vanilla,
                faithfulness_gated=faithfulness_gated,
                image_dir=image_dir,
                output_dir=output_dir,
                top_k=50,
                sort_by='gate'
            )

        print(f"\n=== Visualizing Top 10 Deboosted Features ===")
        for idx, row in most_deboosted.iterrows():
            feat_idx = row['feature_idx']
            print(f"  Visualizing feature {feat_idx} (mean_gate: {row['mean_gate']:.3f})")
            output_dir = Path(f"{experiment_path}/visualizations/layer_{layer_idx}/deboosted/feature_{feat_idx}")
            visualize_feature_activations(
                feature_idx=feat_idx,
                layer_idx=layer_idx,
                merged_data=merged_gated,
                faithfulness_vanilla=faithfulness_vanilla,
                faithfulness_gated=faithfulness_gated,
                image_dir=image_dir,
                output_dir=output_dir,
                top_k=50,
                sort_by='gate'
            )
