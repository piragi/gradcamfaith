# visualization.py
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def visualize_attribution_diff(
        original_attribution: np.ndarray,
        perturbed_attribution: np.ndarray,
        mask: np.ndarray,
        base_name: str = "attribution_diff",
        save_dir: str = "./results/comparisons"
) -> Dict[str, Dict[str, float]]:
    """
    Visualize the difference between original and perturbed attribution maps.
    
    Args:
        original_attribution: Attribution map for original image
        perturbed_attribution: Attribution map for perturbed image
        mask: Binary mask showing perturbed region
        base_name: Base filename for saving
        save_dir: Directory to save visualization
        
    Returns:
        Dictionary with statistics for original, perturbed, and difference maps
    """
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Calculate difference map
    diff_map = perturbed_attribution - original_attribution

    # Normalize each map for visualization
    orig_norm = normalize_for_display(original_attribution)
    pert_norm = normalize_for_display(perturbed_attribution)
    diff_norm = normalize_for_display(diff_map, use_abs=True)

    # Prepare mask overlay
    mask_overlay = prepare_mask_overlay(mask, orig_norm.shape)

    # Calculate statistics
    stats = calculate_attribution_statistics(original_attribution,
                                             perturbed_attribution, diff_map,
                                             mask)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original attribution with mask outline
    axes[0, 0].imshow(orig_norm, cmap='viridis')
    if mask_overlay is not None:
        axes[0, 0].imshow(mask_overlay, alpha=0.3)
    axes[0, 0].set_title("Original Attribution")
    axes[0, 0].axis('off')

    # Perturbed attribution
    axes[0, 1].imshow(pert_norm, cmap='viridis')
    axes[0, 1].set_title("Perturbed Attribution")
    axes[0, 1].axis('off')

    # Difference map
    im = axes[1, 0].imshow(diff_norm, cmap='coolwarm')
    axes[1, 0].set_title("Difference Map")
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Histogram of differences
    axes[1, 1].hist(diff_map.flatten(), bins=50)
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_title("Difference Histogram")

    # Add statistics text
    text_str = f"Change in perturbed area: {stats['difference_stats']['mean_in_mask']:.3f}\n"
    text_str += f"Change outside perturbed area: {stats['difference_stats']['mean_outside_mask']:.3f}\n"
    text_str += f"Max difference: {stats['difference_stats']['max']:.3f}"
    axes[1, 1].text(0.05,
                    0.95,
                    text_str,
                    transform=axes[1, 1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_comparison.png", dpi=150)
    plt.close()

    return stats


def normalize_for_display(attribution_map: np.ndarray,
                          use_abs: bool = False) -> np.ndarray:
    """
    Normalize an attribution map for visualization.
    
    Args:
        attribution_map: Raw attribution map
        use_abs: Whether to use absolute values before normalization
        
    Returns:
        Normalized attribution map for display
    """
    if use_abs:
        norm_map = np.abs(attribution_map)
    else:
        norm_map = attribution_map.copy()

    # Avoid division by zero
    if np.max(norm_map) == np.min(norm_map):
        return np.zeros_like(norm_map)

    # Normalize to [0, 1]
    return (norm_map - np.min(norm_map)) / (np.max(norm_map) -
                                            np.min(norm_map))


def prepare_mask_overlay(
        mask: np.ndarray, target_shape: Tuple[int,
                                              int]) -> Optional[np.ndarray]:
    """
    Prepare a mask overlay for visualization.
    
    Args:
        mask: Binary mask
        target_shape: Shape to resize mask to
        
    Returns:
        Colored mask overlay or None if mask is invalid
    """
    if mask is None or np.max(mask) == 0:
        return None

    # Ensure mask is the right shape
    if mask.shape != target_shape:
        # Resize mask using nearest neighbor
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((target_shape[1], target_shape[0]),
                                   Image.NEAREST)
        mask = np.array(mask_img) / 255.0

    # Create colored overlay (red)
    overlay = np.zeros((*target_shape, 4))  # RGBA
    overlay[..., 0] = 1.0  # Red channel
    overlay[..., 3] = mask  # Alpha channel

    return overlay


def calculate_attribution_statistics(
        original_map: np.ndarray, perturbed_map: np.ndarray,
        diff_map: np.ndarray, mask: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for attribution maps.
    
    Args:
        original_map: Original attribution map
        perturbed_map: Perturbed attribution map
        diff_map: Difference map (perturbed - original)
        mask: Binary mask showing perturbed region
        
    Returns:
        Dictionary with statistics for original, perturbed, and difference maps
    """
    # Ensure mask is boolean and properly shaped
    if mask.shape != original_map.shape:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize(
            (original_map.shape[1], original_map.shape[0]), Image.NEAREST)
        mask = np.array(mask_img) / 255.0 > 0.5
    else:
        mask = mask > 0.5

    # Calculate statistics for original map
    original_stats = {
        "mean": np.mean(original_map),
        "max": np.max(original_map),
        "min": np.min(original_map),
        "std": np.std(original_map),
        "mean_in_mask": np.mean(original_map[mask]) if np.any(mask) else 0,
        "mean_outside_mask":
        np.mean(original_map[~mask]) if np.any(~mask) else 0
    }

    # Calculate statistics for perturbed map
    perturbed_stats = {
        "mean": np.mean(perturbed_map),
        "max": np.max(perturbed_map),
        "min": np.min(perturbed_map),
        "std": np.std(perturbed_map),
        "mean_in_mask": np.mean(perturbed_map[mask]) if np.any(mask) else 0,
        "mean_outside_mask":
        np.mean(perturbed_map[~mask]) if np.any(~mask) else 0
    }

    # Calculate statistics for difference map
    difference_stats = {
        "mean":
        np.mean(diff_map),
        "max":
        np.max(diff_map),
        "min":
        np.min(diff_map),
        "std":
        np.std(diff_map),
        "abs_mean":
        np.mean(np.abs(diff_map)),
        "mean_in_mask":
        np.mean(diff_map[mask]) if np.any(mask) else 0,
        "mean_outside_mask":
        np.mean(diff_map[~mask]) if np.any(~mask) else 0,
        "abs_mean_in_mask":
        np.mean(np.abs(diff_map[mask])) if np.any(mask) else 0,
        "abs_mean_outside_mask":
        np.mean(np.abs(diff_map[~mask])) if np.any(~mask) else 0,
        "percent_area_perturbed":
        np.sum(mask) / mask.size * 100
    }

    return {
        "original_stats": original_stats,
        "perturbed_stats": perturbed_stats,
        "difference_stats": difference_stats
    }


def visualize_attribution_map(attribution_map: np.ndarray,
                              original_image: Optional[Union[
                                  np.ndarray, Image.Image]] = None,
                              overlay_alpha: float = 0.6,
                              colormap: str = 'viridis',
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize an attribution map, optionally overlaid on the original image.
    
    Args:
        attribution_map: Attribution map to visualize
        original_image: Original image to overlay attribution on
        overlay_alpha: Alpha value for overlay
        colormap: Matplotlib colormap name
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure object
    """
    # Normalize attribution map for visualization
    norm_attr = normalize_for_display(attribution_map)

    fig, ax = plt.subplots(figsize=(8, 8))

    if original_image is not None:
        # Convert PIL Image to numpy if needed
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        # Display original image
        ax.imshow(original_image)

        # Overlay attribution map
        attribution_overlay = plt.cm.get_cmap(colormap)(norm_attr)
        attribution_overlay[..., 3] = overlay_alpha  # Set alpha
        ax.imshow(attribution_overlay)
    else:
        # Display only the attribution map
        ax.imshow(norm_attr, cmap=colormap)

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_attribution_comparison(
        original_attribution: np.ndarray,
        negative_attribution: np.ndarray,
        original_image: Optional[Union[np.ndarray, Image.Image]] = None,
        save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a side-by-side comparison of positive and negative attribution maps.
    
    Args:
        original_attribution: Positive attribution map
        negative_attribution: Negative attribution map
        original_image: Original image for context
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure object
    """
    # Normalize attribution maps
    pos_norm = normalize_for_display(original_attribution)
    neg_norm = normalize_for_display(negative_attribution)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image if provided
    if original_image is not None:
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
    else:
        axes[0].axis('off')
        axes[0].set_title("Original Image (Not Provided)")

    # Positive attribution
    pos_img = axes[1].imshow(pos_norm, cmap='viridis')
    axes[1].set_title("Positive Attribution")
    plt.colorbar(pos_img, ax=axes[1], fraction=0.046, pad=0.04)

    # Negative attribution
    neg_img = axes[2].imshow(neg_norm, cmap='plasma')
    axes[2].set_title("Negative Attribution")
    plt.colorbar(neg_img, ax=axes[2], fraction=0.046, pad=0.04)

    # Turn off axis ticks
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_ffn_activity(ffn_activity: np.ndarray,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize FFN activity across layers.
    
    Args:
        ffn_activity: FFN activity data (loaded from .npy file)
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure object
    """
    # Extract activity metrics
    layers = range(len(ffn_activity))
    mean_activities = [layer['mean_activity'] for layer in ffn_activity]

    cls_activities = []
    for layer in ffn_activity:
        if 'cls_activity' in layer:
            cls_activities.append(layer['cls_activity'])
        else:
            cls_activities.append(np.nan)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean activity
    ax.plot(layers, mean_activities, 'o-', label='Mean Token Activity')

    # Plot CLS activity if available
    if not all(np.isnan(cls_activities)):
        ax.plot(layers, cls_activities, 's-', label='CLS Token Activity')

    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('FFN Activity')
    ax.set_title('Feed-Forward Network Activity Across Layers')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add token activity heatmap if available
    has_token_activity = False
    token_activities = []

    for layer in ffn_activity:
        if 'activity' in layer and isinstance(layer['activity'], np.ndarray):
            has_token_activity = True
            token_activities.append(layer['activity'])

    if has_token_activity:
        # Create a second figure for the heatmap
        fig2, ax2 = plt.subplots(figsize=(12, 8))

        # Stack activities into a 2D array
        activity_array = np.vstack(
            [act.reshape(1, -1) for act in token_activities])

        # Plot heatmap
        im = ax2.imshow(activity_array, aspect='auto', cmap='viridis')

        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Layer')
        ax2.set_title('FFN Activity by Token and Layer')

        plt.colorbar(im, ax=ax2, label='Activity')

        # Save if requested
        if save_path:
            token_save_path = f"{Path(save_path).stem}_tokens{Path(save_path).suffix}"
            plt.figure(fig2.number)
            plt.savefig(token_save_path, dpi=150, bbox_inches='tight')

    # Save main figure if requested
    if save_path:
        plt.figure(fig.number)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
