"""
Qualitative Case Study Analysis
Traces improvements in top-performing images back to specific features and their semantics.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image


def load_and_preprocess_image(image_path: Path, dataset_config) -> np.ndarray:
    """
    Load and preprocess image using the EXACT same pipeline as the model.
    Uses dataset_config.get_transforms() - the single source of truth.

    Returns:
        Preprocessed image as numpy array [224, 224, 3] for visualization.
        Note: Will be normalized (not in [0,1] range), but suitable for visualization.
    """
    if not image_path.exists():
        return None

    img = Image.open(image_path).convert('RGB')

    # Use the EXACT same transforms as the model
    transform = dataset_config.get_transforms('test')
    img_tensor = transform(img)  # [3, 224, 224], normalized

    # Convert to numpy for visualization [224, 224, 3]
    img_array = img_tensor.permute(1, 2, 0).numpy()

    # Denormalize for visualization (reverse the normalization)
    # For CLIP: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    # We'll handle this by clipping to reasonable range for display
    img_array = np.clip(img_array, -3, 3)  # Clip extreme values
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)

    return img_array


def load_faithfulness_results(path: Path) -> pd.DataFrame:
    """Load faithfulness results from experiment directory."""
    saco_csv_file = list(path.glob("analysis_faithfulness_correctness_*.csv"))
    if not saco_csv_file:
        raise FileNotFoundError(f"No SaCo faithfulness CSV found in {path}")

    df = pd.read_csv(saco_csv_file[0])

    # Load additional metrics
    faithfulness_json = list(path.glob("faithfulness_stats_*.json"))
    if faithfulness_json:
        with open(faithfulness_json[0], 'r') as f:
            faithfulness_stats = json.load(f)
        metrics = faithfulness_stats.get('metrics', {})

        for metric_name, metric_data in metrics.items():
            df[metric_name] = metric_data['mean_scores']

    # Extract image index from filename
    if 'filename' in df.columns:
        df['image_idx'] = df['filename'].str.extract(r'_(\d+)\.(?:jpeg|png)$')[0].astype(int)
    else:
        df['image_idx'] = range(len(df))

    return df


def load_debug_data(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """Load debug data containing feature activations and contributions."""
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
            'sparse_contributions': data.get('sparse_contributions', None),
            'gate_values': data['gate_values'],
            'patch_attribution_deltas': data.get('patch_attribution_deltas', None),
        }

        print(f"  Layer {layer_idx}: {len(debug_data[layer_idx]['sparse_indices'])} images")

    return debug_data


def load_activation_data(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """Load lightweight activation data (from extract_sae_activations.py)."""
    debug_dir = path / "debug_data"
    if not debug_dir.exists():
        raise FileNotFoundError(f"Activation data directory not found: {debug_dir}")

    activation_files = list(debug_dir.glob("layer_*_activations.npz"))
    if not activation_files:
        raise FileNotFoundError(f"No activation NPZ files found in {debug_dir}")

    activation_data = {}
    for activation_file in sorted(activation_files):
        layer_idx = int(activation_file.stem.split('_')[1])
        data = np.load(activation_file, allow_pickle=True)

        activation_data[layer_idx] = {
            'sparse_indices': data['sparse_indices'],
            'sparse_activations': data['sparse_activations'],
        }

        print(f"  Layer {layer_idx}: {len(activation_data[layer_idx]['sparse_indices'])} images")

    return activation_data


def build_feature_activation_index(
    debug_data: Dict[int, Dict[str, np.ndarray]],
    layer_idx: int,
    source_name: str = "test",
    top_k_per_feature: int = 100,
    batch_size: int = 10000
) -> Dict[int, List[Tuple[int, int, float]]]:
    """
    Build reverse index: feature_idx -> list of (debug_idx, patch_idx, activation).
    Memory-efficient version that processes in batches and only keeps top-K per feature.

    Note: debug_idx is the position in debug data arrays (0-N), not the actual image_idx.

    Args:
        debug_data: Debug data containing activations
        layer_idx: Layer to build index for
        source_name: Name of the source (e.g., "test", "validation") for logging
        top_k_per_feature: Only keep top K activations per feature (reduces memory)
        batch_size: Process images in batches to reduce memory usage
    """
    print(f"Building activation index for layer {layer_idx} from {source_name} set...")
    print(f"  Using batched processing (batch_size={batch_size}, keeping top-{top_k_per_feature} per feature)")

    feature_index = defaultdict(list)

    sparse_indices = debug_data[layer_idx]['sparse_indices']
    sparse_activations = debug_data[layer_idx]['sparse_activations']
    n_images = len(sparse_indices)

    # Process in batches
    for batch_start in range(0, n_images, batch_size):
        batch_end = min(batch_start + batch_size, n_images)

        if batch_start % (batch_size * 2) == 0:
            print(f"  Processing images {batch_start}-{batch_end}/{n_images}...")

        # Process this batch
        for debug_idx in range(batch_start, batch_end):
            for patch_idx in range(len(sparse_indices[debug_idx])):
                patch_features = sparse_indices[debug_idx][patch_idx]
                patch_activations = sparse_activations[debug_idx][patch_idx]

                for feat_idx, activation in zip(patch_features, patch_activations):
                    feature_index[int(feat_idx)].append((debug_idx, patch_idx, float(activation)))

        # After each batch, prune each feature to top-K to save memory
        for feat_idx in feature_index:
            if len(feature_index[feat_idx]) > top_k_per_feature * 2:
                # Only prune if we have significantly more than top_k
                feature_index[feat_idx] = sorted(feature_index[feat_idx], key=lambda x: x[2],
                                                 reverse=True)[:top_k_per_feature]

    # Final sort and prune for all features
    print(f"  Finalizing index (keeping top-{top_k_per_feature} activations per feature)...")
    for feat_idx in feature_index:
        feature_index[feat_idx] = sorted(feature_index[feat_idx], key=lambda x: x[2], reverse=True)[:top_k_per_feature]

    print(f"  Indexed {len(feature_index)} unique features from {n_images} images")
    return dict(feature_index)


def compute_composite_improvement(vanilla_df: pd.DataFrame, gated_df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite improvement score from multiple metrics."""
    improvements = gated_df[['image_idx']].copy()

    # Calculate deltas
    improvements['delta_saco'] = gated_df['saco_score'] - vanilla_df['saco_score']
    improvements['delta_faith'] = gated_df['FaithfulnessCorrelation'] - vanilla_df['FaithfulnessCorrelation']
    improvements['delta_pixel'] = gated_df['PixelFlipping'] - vanilla_df['PixelFlipping']

    # Z-score normalization
    z_score = lambda series: (series - series.mean()) / (series.std() + 1e-9)

    improvements['composite_improvement'] = (
        z_score(improvements['delta_saco']) + z_score(improvements['delta_faith']) +
        -z_score(improvements['delta_pixel'])  # Lower is better for pixel flipping
    ) / 3

    return improvements


def find_dominant_features_in_image(
    debug_idx: int,
    image_idx: int,
    layer_idx: int,
    debug_data: Dict[int, Dict[str, np.ndarray]],
    vanilla_attr: np.ndarray,
    gated_attr: np.ndarray,
    n_top_patches: int = 5
) -> List[Dict[str, Any]]:
    """
    Find features that contributed most to attribution changes in an image.

    For each of the top-N patches (by |final_attribution_delta|), extract the
    feature with the strongest contribution in the matching direction.

    Args:
        debug_idx: Index in debug data arrays (not the actual image_idx)
        image_idx: Actual image index (for file lookup)
        vanilla_attr: Final vanilla attribution map [224, 224]
        gated_attr: Final gated attribution map [224, 224]
    """
    sparse_indices = debug_data[layer_idx]['sparse_indices'][debug_idx]
    sparse_contributions = debug_data[layer_idx]['sparse_contributions'][debug_idx]
    layer_patch_deltas = debug_data[layer_idx]['patch_attribution_deltas'][debug_idx]

    # Detect patch configuration from debug data
    n_patches = len(layer_patch_deltas)
    patches_per_side = int(np.sqrt(n_patches))
    patch_size = 224 // patches_per_side  # 224/14=16 for B/16, 224/7=32 for B/32

    final_patch_deltas = []
    for patch_idx in range(n_patches):
        row_idx = patch_idx // patches_per_side
        col_idx = patch_idx % patches_per_side
        y, x = row_idx * patch_size, col_idx * patch_size

        vanilla_patch_val = vanilla_attr[y:y + patch_size, x:x + patch_size].mean()
        gated_patch_val = gated_attr[y:y + patch_size, x:x + patch_size].mean()
        final_delta = gated_patch_val - vanilla_patch_val
        final_patch_deltas.append(final_delta)

    final_patch_deltas = np.array(final_patch_deltas)

    # Find patches with largest absolute FINAL attribution changes
    patch_rankings = [(i, abs(delta)) for i, delta in enumerate(final_patch_deltas)]
    patch_rankings.sort(key=lambda x: x[1], reverse=True)
    top_patch_indices = [idx for idx, _ in patch_rankings[:n_top_patches]]

    dominant_features = []
    skipped_reasons = {'negligible_delta': 0, 'no_features': 0, 'no_matching_direction': 0}

    for patch_idx in top_patch_indices:
        final_delta = final_patch_deltas[patch_idx]
        layer_delta = layer_patch_deltas[patch_idx]

        if abs(final_delta) < 1e-6:
            skipped_reasons['negligible_delta'] += 1
            continue

        patch_features = sparse_indices[patch_idx]
        patch_contributions = sparse_contributions[patch_idx]

        if len(patch_features) == 0:
            skipped_reasons['no_features'] += 1
            continue

        # Find feature with strongest contribution in matching direction
        # Use FINAL delta to determine boost vs suppress
        if final_delta > 0:
            # Boosted patch - find max positive contributor
            pos_contribs = [(i, c) for i, c in enumerate(patch_contributions) if c > 0]
            if not pos_contribs:
                skipped_reasons['no_matching_direction'] += 1
                continue
            best_idx, best_contrib = max(pos_contribs, key=lambda x: x[1])
            role = "BOOST"
        else:
            # Suppressed patch - find max negative (most negative) contributor
            neg_contribs = [(i, c) for i, c in enumerate(patch_contributions) if c < 0]
            if not neg_contribs:
                skipped_reasons['no_matching_direction'] += 1
                continue
            best_idx, best_contrib = min(neg_contribs, key=lambda x: x[1])
            role = "SUPPRESS"

        feature_idx = int(patch_features[best_idx])

        dominant_features.append({
            'patch_idx': patch_idx,
            'final_patch_delta': float(final_delta),
            'layer_patch_delta': float(layer_delta),
            'feature_idx': feature_idx,
            'contribution': float(best_contrib),
            'role': role
        })

    # Debug: log if we filtered out all patches
    if sum(skipped_reasons.values()) > 0:
        print(f"    Warning: Image {image_idx} - all {n_top_patches} patches filtered. Reasons: {skipped_reasons}")

    return dominant_features


def extract_case_studies(
    vanilla_faithfulness: pd.DataFrame,
    gated_faithfulness: pd.DataFrame,
    debug_data: Dict[int, Dict[str, np.ndarray]],
    layer_idx: int,
    vanilla_attribution_dir: Path,
    gated_attribution_dir: Path,
    dataset: str,
    n_top_images: int = 100,
    n_patches_per_image: int = 5
) -> pd.DataFrame:
    """
    Extract case studies from top improved images.

    Returns a DataFrame with one row per (image, patch, feature) combination.
    """
    # Compute improvements
    improvements = compute_composite_improvement(vanilla_faithfulness, gated_faithfulness)
    improvements = improvements.merge(
        gated_faithfulness[['image_idx', 'predicted_class', 'true_class', 'is_correct']], on='image_idx'
    )

    # IMPORTANT: Debug data is indexed by processing order (0-N), not by image_idx
    # We need to add a processing order index to match with debug data
    improvements['debug_idx'] = range(len(improvements))

    # Get top improved images
    top_images = improvements.nlargest(n_top_images, 'composite_improvement')

    print(f"\nAnalyzing top {n_top_images} improved images:")
    print(
        f"  Composite improvement range: [{top_images['composite_improvement'].min():.3f}, "
        f"{top_images['composite_improvement'].max():.3f}]"
    )

    # Extract dominant features from each image
    case_studies = []

    for _, img_row in top_images.iterrows():
        img_idx = img_row['image_idx']  # Actual image index (for file lookup)
        debug_idx = img_row['debug_idx']  # Index in debug data arrays

        # Load attribution maps
        if dataset == 'imagenet':
            vanilla_attr = load_attribution_imagenet(img_idx, vanilla_attribution_dir)
            gated_attr = load_attribution_imagenet(img_idx, gated_attribution_dir)
        else:
            vanilla_attr = load_attribution_covidquex(img_idx, vanilla_attribution_dir)
            gated_attr = load_attribution_covidquex(img_idx, gated_attribution_dir)

        if vanilla_attr is None or gated_attr is None:
            print(f"  Warning: Skipping image {img_idx} - attribution maps not found")
            continue

        dominant_features = find_dominant_features_in_image(
            debug_idx, img_idx, layer_idx, debug_data, vanilla_attr, gated_attr, n_top_patches=n_patches_per_image
        )

        for feat_info in dominant_features:
            case_studies.append({
                'image_idx': img_idx,
                'debug_idx': debug_idx,
                'composite_improvement': img_row['composite_improvement'],
                'delta_saco': img_row['delta_saco'],
                'delta_faith': img_row['delta_faith'],
                'delta_pixel': img_row['delta_pixel'],
                'predicted_class': img_row['predicted_class'],
                'true_class': img_row['true_class'],
                'is_correct': img_row['is_correct'],
                'layer_idx': layer_idx,
                **feat_info
            })

    case_studies_df = pd.DataFrame(case_studies)

    print(f"  Extracted {len(case_studies_df)} feature contributions")
    print(f"  Roles: {case_studies_df['role'].value_counts().to_dict()}")

    return case_studies_df


def visualize_case_study(
    case: pd.Series,
    feature_index: Dict[int, List[Tuple[int, int, float]]],
    debug_to_image_idx: Dict[int, int],
    debug_data: Dict[int, Dict[str, np.ndarray]],
    dataset_config,
    case_study_image_dir: Path,
    prototype_image_dir: Path,
    vanilla_attribution_dir: Path,
    gated_attribution_dir: Path,
    output_dir: Path,
    n_prototypes: int = 10,
    dataset: str = 'imagenet',
    prototype_path_mapping: Optional[Dict[int, Path]] = None
):
    """
    Visualize a single case study with feature prototypes.

    Creates a figure showing:
    1. The improved image with vanilla/gated attributions (from case study set)
    2. Top-K activating patches for the dominant feature (from prototype set)

    Args:
        debug_to_image_idx: Mapping from debug array index to actual image_idx for prototypes
        case_study_image_dir: Directory containing case study images (test set)
        prototype_image_dir: Directory containing prototype images (test or validation set)
    """
    img_idx = case['image_idx']
    feature_idx = case['feature_idx']
    layer_idx = case['layer_idx']

    # Get case study image path (always from test set)
    if dataset == 'imagenet':
        image_path = case_study_image_dir / "class_-1" / f"img_-01_test_{img_idx:06d}.jpeg"
    else:  # covidquex
        image_path = get_image_path_covidquex(img_idx, case_study_image_dir)

    if not image_path.exists():
        print(f"Warning: Image {img_idx} not found at {image_path}")
        return

    # Load attributions
    if dataset == 'imagenet':
        vanilla_attr = load_attribution_imagenet(img_idx, vanilla_attribution_dir)
        gated_attr = load_attribution_imagenet(img_idx, gated_attribution_dir)
    else:
        vanilla_attr = load_attribution_covidquex(img_idx, vanilla_attribution_dir)
        gated_attr = load_attribution_covidquex(img_idx, gated_attribution_dir)

    if vanilla_attr is None or gated_attr is None:
        print(f"Warning: Attributions not found for image {img_idx}")
        return

    # Get top activating examples for this feature (debug_idx from index)
    if feature_idx not in feature_index:
        print(f"Warning: Feature {feature_idx} not in index")
        return

    top_activations = feature_index[feature_idx][:n_prototypes]

    # Create figure
    n_cols = 4
    n_rows = 2 + (len(top_activations) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

    # Load and preprocess main image (same as model input)
    main_img_array = load_and_preprocess_image(image_path, dataset_config)
    if main_img_array is None:
        print(f"Warning: Could not load main image {image_path}")
        return

    # Row 1: Vanilla attribution
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(main_img_array)
    vanilla_norm = (vanilla_attr - vanilla_attr.min()) / (vanilla_attr.max() - vanilla_attr.min() + 1e-8)
    ax1.imshow(vanilla_norm, cmap='jet', alpha=0.3, interpolation='bilinear')
    ax1.set_title("Vanilla Attribution", fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Row 1: Gated attribution with patch highlight
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.imshow(main_img_array)
    gated_norm = (gated_attr - gated_attr.min()) / (gated_attr.max() - gated_attr.min() + 1e-8)
    ax2.imshow(gated_norm, cmap='jet', alpha=0.3, interpolation='bilinear')

    # Highlight the patch
    # Detect patch configuration from the data (need to load one debug sample)
    # We can infer from the attribution map size
    patch_idx = case['patch_idx']
    layer_idx = case['layer_idx']

    # Infer patch grid from layer_idx in debug data
    # Quick hack: load a sample to detect size
    sample_deltas = debug_data[layer_idx]['patch_attribution_deltas'][0]
    n_patches = len(sample_deltas)
    patches_per_side = int(np.sqrt(n_patches))
    patch_size = 224 // patches_per_side

    row_idx = patch_idx // patches_per_side
    col_idx = patch_idx % patches_per_side
    x, y = col_idx * patch_size, row_idx * patch_size

    # Color based on role
    color = 'lime' if case['role'] == 'BOOST' else 'red'
    rect = mpatches.Rectangle((x, y), patch_size, patch_size, linewidth=3, edgecolor=color, facecolor='none')
    ax2.add_patch(rect)

    # Add text showing the attribution change
    final_delta = case['final_patch_delta']
    change_sign = "↑" if final_delta > 0 else "↓"
    ax2.set_title(
        f"Gated Attribution ({case['role']} patch)\n"
        f"Final Δ: {final_delta:.3f} {change_sign}",
        fontsize=11,
        fontweight='bold'
    )
    ax2.axis('off')

    # Row 2: Info text
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')
    info_text = (
        f"Image {img_idx} | Layer {layer_idx} | Feature {feature_idx}\n"
        f"Composite Improvement: {case['composite_improvement']:.3f} | "
        f"ΔSaCo: {case['delta_saco']:.4f} | ΔFaith: {case['delta_faith']:.4f}\n"
        f"Final Attribution Δ: {case['final_patch_delta']:.4f} ({case['role']}) | "
        f"Layer {layer_idx} CAM Δ: {case['layer_patch_delta']:.4f}\n"
        f"Feature Contribution: {case['contribution']:.4f}"
    )
    ax_info.text(
        0.5,
        0.5,
        info_text,
        ha='center',
        va='center',
        fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Remaining rows: Feature prototypes
    for proto_idx, (proto_debug_idx, proto_patch_idx, proto_activation) in enumerate(top_activations):
        row = 2 + proto_idx // n_cols
        col = proto_idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Convert debug_idx to actual image_idx for file lookup
        proto_img_idx = debug_to_image_idx.get(proto_debug_idx)
        if proto_img_idx is None:
            ax.text(0.5, 0.5, f"Debug idx {proto_debug_idx}\nmapping missing", ha='center', va='center')
            ax.axis('off')
            continue

        # Load prototype image (from validation or test set depending on configuration)
        if prototype_path_mapping is not None:
            # Use direct path mapping (for validation set with complex structure)
            proto_img_path = prototype_path_mapping.get(proto_debug_idx)
            if proto_img_path is None:
                ax.text(0.5, 0.5, f"Path not found\nfor idx {proto_debug_idx}", ha='center', va='center')
                ax.axis('off')
                continue
        elif dataset == 'imagenet':
            # Test set has simple structure
            proto_img_path = prototype_image_dir / "class_-1" / f"img_-01_test_{proto_img_idx:06d}.jpeg"
        else:
            proto_img_path = get_image_path_covidquex(proto_img_idx, prototype_image_dir)

        # Load and preprocess prototype image (same as model input)
        proto_img_array = load_and_preprocess_image(proto_img_path, dataset_config)

        if proto_img_array is not None:
            ax.imshow(proto_img_array)

            # Highlight the activating patch (use same patch grid as main image)
            proto_row = proto_patch_idx // patches_per_side
            proto_col = proto_patch_idx % patches_per_side
            proto_x = proto_col * patch_size
            proto_y = proto_row * patch_size

            rect = mpatches.Rectangle((proto_x, proto_y),
                                      patch_size,
                                      patch_size,
                                      linewidth=2,
                                      edgecolor='yellow',
                                      facecolor='none')
            ax.add_patch(rect)

            ax.set_title(f"#{proto_idx+1}: Act={proto_activation:.2f}", fontsize=9)
        else:
            ax.text(0.5, 0.5, f"Image {proto_img_idx}\nnot found", ha='center', va='center')

        ax.axis('off')

    # Main title
    fig.suptitle(f"Case Study: Feature {feature_idx} {case['role']}ing Attribution", fontsize=14, fontweight='bold')

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"case_img{img_idx}_layer{layer_idx}_feat{feature_idx}_{case['role'].lower()}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")


def save_case_study_individual_images(
    case: pd.Series,
    feature_index: Dict[int, List[Tuple[int, int, float]]],
    debug_to_image_idx: Dict[int, int],
    debug_data: Dict[int, Dict[str, np.ndarray]],
    dataset_config,
    case_study_image_dir: Path,
    prototype_image_dir: Path,
    vanilla_attribution_dir: Path,
    gated_attribution_dir: Path,
    output_dir: Path,
    n_prototypes: int = 10,
    dataset: str = 'imagenet',
    prototype_path_mapping: Optional[Dict[int, Path]] = None
):
    """
    Save case study as individual images in a folder with metadata JSON.

    Folder structure:
        case_img{img_idx}_layer{layer_idx}_feat{feature_idx}_{role}/
            - vanilla_attribution.png
            - gated_attribution.png  (with patch highlight)
            - prototype_0.png, prototype_1.png, ...
            - metadata.json
    """
    import json

    img_idx = case['image_idx']
    feature_idx = case['feature_idx']
    layer_idx = case['layer_idx']
    patch_idx = case['patch_idx']

    # Create case study folder
    folder_name = f"case_img{img_idx}_layer{layer_idx}_feat{feature_idx}_patch{patch_idx}_{case['role'].lower()}"
    case_dir = output_dir / folder_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # Get case study image path
    if dataset == 'imagenet':
        image_path = case_study_image_dir / "class_-1" / f"img_-01_test_{img_idx:06d}.jpeg"
    else:
        image_path = get_image_path_covidquex(img_idx, case_study_image_dir)

    if not image_path.exists():
        print(f"Warning: Image {img_idx} not found at {image_path}")
        return

    # Load attributions
    if dataset == 'imagenet':
        vanilla_attr = load_attribution_imagenet(img_idx, vanilla_attribution_dir)
        gated_attr = load_attribution_imagenet(img_idx, gated_attribution_dir)
    else:
        vanilla_attr = load_attribution_covidquex(img_idx, vanilla_attribution_dir)
        gated_attr = load_attribution_covidquex(img_idx, gated_attribution_dir)

    if vanilla_attr is None or gated_attr is None:
        print(f"Warning: Attributions not found for image {img_idx}")
        return

    # Load and preprocess main image
    main_img_array = load_and_preprocess_image(image_path, dataset_config)
    if main_img_array is None:
        print(f"Warning: Could not load main image {image_path}")
        return

    # Detect patch configuration
    sample_deltas = debug_data[layer_idx]['patch_attribution_deltas'][0]
    n_patches = len(sample_deltas)
    patches_per_side = int(np.sqrt(n_patches))
    patch_size = 224 // patches_per_side

    # Calculate patch coordinates
    row_idx = patch_idx // patches_per_side
    col_idx = patch_idx % patches_per_side
    x, y = col_idx * patch_size, row_idx * patch_size

    # Normalize attributions for visualization
    vanilla_norm = (vanilla_attr - vanilla_attr.min()) / (vanilla_attr.max() - vanilla_attr.min() + 1e-8)
    gated_norm = (gated_attr - gated_attr.min()) / (gated_attr.max() - gated_attr.min() + 1e-8)

    # Save vanilla attribution
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.imshow(main_img_array)
    ax.imshow(vanilla_norm, cmap='jet', alpha=0.3, interpolation='bilinear')
    ax.set_title("Vanilla Attribution", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(case_dir / "vanilla_attribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save gated attribution with patch highlight
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.imshow(main_img_array)
    ax.imshow(gated_norm, cmap='jet', alpha=0.3, interpolation='bilinear')

    # Highlight the patch
    color = 'lime' if case['role'] == 'BOOST' else 'red'
    rect = mpatches.Rectangle((x, y), patch_size, patch_size, linewidth=3, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    final_delta = case['final_patch_delta']
    change_sign = "↑" if final_delta > 0 else "↓"
    ax.set_title(
        f"Gated Attribution ({case['role']} patch)\nΔ: {final_delta:.3f} {change_sign}", fontsize=14, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(case_dir / "gated_attribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Get and save prototypes
    if feature_idx not in feature_index:
        print(f"Warning: Feature {feature_idx} not in index")
        top_activations = []
    else:
        top_activations = feature_index[feature_idx][:n_prototypes]

    prototype_metadata = []
    for proto_idx, (proto_debug_idx, proto_patch_idx, proto_activation) in enumerate(top_activations):
        # Convert debug_idx to actual image_idx for file lookup
        proto_img_idx = debug_to_image_idx.get(proto_debug_idx)
        if proto_img_idx is None:
            continue

        # Load prototype image
        if prototype_path_mapping is not None:
            proto_img_path = prototype_path_mapping.get(proto_debug_idx)
            if proto_img_path is None:
                continue
        elif dataset == 'imagenet':
            proto_img_path = prototype_image_dir / "class_-1" / f"img_-01_test_{proto_img_idx:06d}.jpeg"
        else:
            proto_img_path = get_image_path_covidquex(proto_img_idx, prototype_image_dir)

        proto_img_array = load_and_preprocess_image(proto_img_path, dataset_config)
        if proto_img_array is None:
            continue

        # Save prototype image with highlighted patch
        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.imshow(proto_img_array)

        # Highlight the activating patch
        proto_row = proto_patch_idx // patches_per_side
        proto_col = proto_patch_idx % patches_per_side
        proto_x = proto_col * patch_size
        proto_y = proto_row * patch_size

        rect = mpatches.Rectangle((proto_x, proto_y),
                                  patch_size,
                                  patch_size,
                                  linewidth=2,
                                  edgecolor='yellow',
                                  facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"Prototype {proto_idx+1}: Activation={proto_activation:.2f}", fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(case_dir / f"prototype_{proto_idx}.png", dpi=150, bbox_inches='tight')
        plt.close()

        prototype_metadata.append({
            'prototype_idx': proto_idx,
            'image_idx': int(proto_img_idx),
            'debug_idx': int(proto_debug_idx),
            'patch_idx': int(proto_patch_idx),
            'activation': float(proto_activation),
            'image_path': str(proto_img_path)
        })

    # Save metadata JSON
    metadata = {
        'image_idx': int(img_idx),
        'layer_idx': int(layer_idx),
        'feature_idx': int(feature_idx),
        'patch_idx': int(patch_idx),
        'role': case['role'],
        'patch_coordinates': {
            'x': int(x),
            'y': int(y),
            'width': int(patch_size),
            'height': int(patch_size)
        },
        'metrics': {
            'composite_improvement': float(case['composite_improvement']),
            'delta_saco': float(case['delta_saco']),
            'delta_faithfulness': float(case['delta_faith']),
            'delta_pixel_flipping': float(case['delta_pixel']),
            'final_patch_delta': float(case['final_patch_delta']),
            'layer_patch_delta': float(case['layer_patch_delta']),
            'feature_contribution': float(case['contribution'])
        },
        'classification': {
            'predicted_class': str(case['predicted_class']) if pd.notna(case['predicted_class']) else None,
            'true_class': str(case['true_class']) if pd.notna(case['true_class']) else None,
            'is_correct': bool(case['is_correct'])
        },
        'prototypes': prototype_metadata,
        'patch_config': {
            'patches_per_side': int(patches_per_side),
            'patch_size': int(patch_size),
            'total_patches': int(n_patches)
        }
    }

    with open(case_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: {folder_name}/")


def get_image_path_covidquex(image_idx: int, image_dir: Path) -> Optional[Path]:
    """Map image_idx to file path for CovidQuex dataset."""
    if image_idx < 1903:
        class_idx = 0
        local_idx = image_idx
    elif image_idx < 3705:
        class_idx = 1
        local_idx = image_idx - 1903
    else:
        class_idx = 2
        local_idx = image_idx - 3705

    image_path = image_dir / f"class_{class_idx}" / f"img_{class_idx:02d}_val_{local_idx:05d}.png"
    return image_path if image_path.exists() else None


def load_attribution_covidquex(image_idx: int, attribution_dir: Path) -> Optional[np.ndarray]:
    """Load attribution for CovidQuex dataset."""
    if image_idx < 1903:
        class_idx = 0
        local_idx = image_idx
    elif image_idx < 3705:
        class_idx = 1
        local_idx = image_idx - 1903
    else:
        class_idx = 2
        local_idx = image_idx - 3705

    attr_path = attribution_dir / f"img_{class_idx:02d}_test_{local_idx:05d}_attribution.npy"
    return np.load(attr_path) if attr_path.exists() else None


def load_attribution_imagenet(image_idx: int, attribution_dir: Path) -> Optional[np.ndarray]:
    """Load attribution for ImageNet dataset."""
    attr_path = attribution_dir / f"img_-01_test_{image_idx:06d}_attribution.npy"
    return np.load(attr_path) if attr_path.exists() else None


def run_case_study_analysis(
    experiment_path: Path,
    dataset: str,
    experiment_config: str,
    layers: List[int],
    n_top_images: int = 20,
    n_patches_per_image: int = 5,
    n_case_visualizations: int = 10,
    n_prototypes: int = 10,
    validation_activations_path: Optional[Path] = None
):
    """
    Main entry point for case study analysis.

    Args:
        validation_activations_path: Optional path to validation set activations for prototypes.
                                     If provided, prototypes will be from validation set,
                                     otherwise they'll be from test set.
    """
    print(f"\n{'='*80}")
    print(f"CASE STUDY ANALYSIS: {dataset} / {experiment_config}")
    print(f"{'='*80}\n")

    # Set paths
    vanilla_path = experiment_path / dataset / "vanilla" / "test"
    gated_path = experiment_path / dataset / experiment_config / "test"
    output_dir = experiment_path / "case_studies" / experiment_config

    # Load data
    print("Loading faithfulness results...")
    vanilla_faithfulness = load_faithfulness_results(vanilla_path)
    gated_faithfulness = load_faithfulness_results(gated_path)

    print("Loading debug data (test set for case studies)...")
    debug_data = load_debug_data(gated_path)

    # Load validation activations if provided
    if validation_activations_path is not None:
        print(f"\nLoading validation activations for prototypes from {validation_activations_path}...")
        validation_data = load_activation_data(validation_activations_path)
        use_validation_prototypes = True
    else:
        print("\nUsing test set activations for prototypes...")
        validation_data = None
        use_validation_prototypes = False

    # Load dataset config for preprocessing
    from dataset_config import get_dataset_config
    dataset_config = get_dataset_config(dataset)

    # Determine image directories
    if dataset == "imagenet":
        test_image_dir = Path("./data/imagenet_unified/test")
        val_image_dir = Path("./data/imagenet_unified/val")
    else:
        test_image_dir = Path(f"./data/{dataset}_unified/test")
        val_image_dir = Path(f"./data/{dataset}_unified/val")

    # Create mapping from debug_idx to image_idx for test set
    # The debug data is in processing order, same as faithfulness CSV rows
    test_debug_to_image_idx = dict(enumerate(gated_faithfulness['image_idx'].tolist()))

    # Create mapping for validation set if using validation prototypes
    if use_validation_prototypes:
        # Load the image path mapping from extraction metadata
        # This ensures we use the EXACT same order as when activations were extracted
        import json
        metadata_file = validation_activations_path / "extraction_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}\n"
                f"Please re-run extract_sae_activations.py to generate the metadata with image path mapping."
            )

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        if 'image_paths' not in metadata:
            raise ValueError(
                f"Metadata file is missing 'image_paths' field.\n"
                f"Please re-run extract_sae_activations.py to generate the correct metadata."
            )

        val_debug_to_path = {int(k): Path(v) for k, v in metadata['image_paths'].items()}
        val_debug_to_image_idx = {idx: idx for idx in val_debug_to_path.keys()}  # Simple mapping
        print(f"  Loaded mapping for {len(val_debug_to_path)} validation images from metadata")
    else:
        val_debug_to_image_idx = None
        val_debug_to_path = None

    # Process each layer
    all_case_studies = []

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*60}")

        # Build feature activation index
        # Use validation data for prototypes if available, otherwise use test data
        if use_validation_prototypes:
            feature_index = build_feature_activation_index(validation_data, layer_idx, source_name="validation")
            prototype_debug_to_image_idx = val_debug_to_image_idx
            prototype_image_dir = val_image_dir
            prototype_paths = val_debug_to_path  # Use path mapping for validation
        else:
            feature_index = build_feature_activation_index(debug_data, layer_idx, source_name="test")
            prototype_debug_to_image_idx = test_debug_to_image_idx
            prototype_image_dir = test_image_dir
            prototype_paths = None  # Test set uses simple path construction

        # Extract case studies (always from test set)
        case_studies = extract_case_studies(
            vanilla_faithfulness,
            gated_faithfulness,
            debug_data,
            layer_idx,
            vanilla_path / "attributions",
            gated_path / "attributions",
            dataset,
            n_top_images=n_top_images,
            n_patches_per_image=n_patches_per_image
        )

        all_case_studies.append(case_studies)

        # Save case studies table
        layer_output_dir = output_dir / f"layer_{layer_idx}"
        layer_output_dir.mkdir(parents=True, exist_ok=True)
        case_studies.to_csv(layer_output_dir / "case_studies.csv", index=False)

        # Visualize top cases
        print(f"\nGenerating visualizations (saving individual images)...")
        for case_idx, (_, case) in enumerate(case_studies.head(n_case_visualizations).iterrows()):
            save_case_study_individual_images(
                case,
                feature_index,
                prototype_debug_to_image_idx,
                debug_data,
                dataset_config,
                test_image_dir,  # Case study images always from test
                prototype_image_dir,  # Prototypes from validation or test
                vanilla_path / "attributions",
                gated_path / "attributions",
                layer_output_dir,
                n_prototypes=n_prototypes,
                dataset=dataset,
                prototype_path_mapping=prototype_paths
            )

        # Summary statistics
        print(f"\nLayer {layer_idx} Summary:")
        print(f"  Total case studies: {len(case_studies)}")
        print(f"  Unique features: {case_studies['feature_idx'].nunique()}")
        print(f"  Role distribution: {case_studies['role'].value_counts().to_dict()}")

        # Top features by frequency
        top_features = case_studies['feature_idx'].value_counts().head(10)
        print(f"\n  Most common features:")
        for feat_idx, count in top_features.items():
            avg_contrib = case_studies[case_studies['feature_idx'] == feat_idx]['contribution'].mean()
            print(f"    Feature {feat_idx}: appears {count}x, avg contribution {avg_contrib:.4f}")

    # Save combined results
    combined_df = pd.concat(all_case_studies, ignore_index=True)
    combined_df.to_csv(output_dir / "all_case_studies.csv", index=False)

    print(f"\n{'='*80}")
    print(f"Case study analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    return combined_df


if __name__ == "__main__":
    # Configuration
    dataset = "imagenet"
    experiment_path = Path("./experiments/feature_gradient_sweep_20251118_203648")
    experiment_config = "layers_6_9_10_kappa_0.5_topk_None_combined_clamp_10.0"
    layers = [6, 9, 10]

    # Optional: Use validation set activations for prototypes
    # Set to None to use test set activations instead
    validation_activations_path = Path("./sae_activations/imagenet_val")

    # Run analysis
    case_studies = run_case_study_analysis(
        experiment_path=experiment_path,
        dataset=dataset,
        experiment_config=experiment_config,
        layers=layers,
        n_top_images=100,
        n_patches_per_image=5,
        n_case_visualizations=500,  # Visualize all extracted feature contributions (20 images × 5 patches = ~100)
        n_prototypes=20,
        validation_activations_path=validation_activations_path
    )
