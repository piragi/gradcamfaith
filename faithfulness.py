"""
Faithfulness evaluation for the unified pipeline.
Adapts the original faithfulness.py to work with HookedSAEViT and CLIP models.
"""

import gc
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import quantus
import torch
import torch.nn.functional as F
from PIL import Image

from config import PipelineConfig
from data_types import ClassificationResult
from dataset_config import get_dataset_config


@dataclass
class FaithfulnessEstimatorConfig:
    """Configuration for a faithfulness estimator."""
    name: str
    n_trials: int
    estimator_fn: Callable
    kwargs: Dict[str, Any] = None

    def create_estimator(self):
        """Create the estimator instance using the function and kwargs."""
        if self.kwargs:
            return self.estimator_fn(**self.kwargs)
        return self.estimator_fn()


def faithfulness_correlation(subset_size, nr_runs):
    return quantus.FaithfulnessCorrelation(
        perturb_func=batch_patch_level_perturbation,  # Use batched version for better performance
        perturb_baseline="black",
        similarity_func=quantus.similarity_func.correlation_spearman,
        subset_size=subset_size,
        return_aggregate=False,
        normalise=False,
        nr_runs=nr_runs
    )


class PatchLevelPixelFlipping(quantus.PixelFlipping):
    """
    Custom PixelFlipping that works directly with patch attributions.
    Avoids expensive sorting of 150k duplicate pixel values by sorting patches directly.
    
    Inherits all quantus infrastructure while optimizing for patch-based models.
    """

    def __init__(
        self,
        features_in_step: int = 49,
        patch_size: int = 16,
        perturb_baseline: str = "black",
        n_patches: int = 196,
        **kwargs
    ):
        # Initialize parent with perturbation function
        super().__init__(
            features_in_step=features_in_step,
            perturb_func=patch_level_perturbation,
            perturb_baseline=perturb_baseline,
            **kwargs
        )
        self.patch_size = patch_size
        self.baseline_value = perturb_baseline
        self.n_patches = n_patches

    def evaluate_batch(self, model, x_batch: np.ndarray, y_batch: np.ndarray, a_batch: np.ndarray, **kwargs):
        """
        Fast patch-level PixelFlipping: Uses same sorting as standard but patch-level perturbation.
        """
        # Extract patch attributions (FAST!)
        batch_size = a_batch.shape[0]
        n_patches = self.n_patches
        patch_size = 32 if n_patches == 49 else 16

        if a_batch.shape[-1] > n_patches:
            a_batch_patches = self._extract_patch_attributions_exact(a_batch, n_patches)
        else:
            a_batch_patches = a_batch.reshape(batch_size, -1)

        # Sort patches by attribution importance - THE SPEED BOOST
        patch_indices_sorted = np.argsort(-a_batch_patches, axis=1)

        # Calculate number of perturbation steps
        n_steps = math.ceil(n_patches / self.features_in_step)

        # Store predictions for each step
        predictions = []
        x_perturbed = x_batch.copy()

        # Get patch-to-pixel mapping
        patch_to_pixels = _get_patch_to_pixels_mapping(patch_size=patch_size, n_patches=n_patches)

        for step in range(n_steps):
            # Get patches to remove in this step
            start_idx = step * self.features_in_step
            end_idx = min((step + 1) * self.features_in_step, n_patches)

            patches_to_remove = patch_indices_sorted[:, start_idx:end_idx]

            # Apply patch-level perturbation (fast!)
            for i in range(batch_size):
                valid_patches = patches_to_remove[i][patches_to_remove[i] < n_patches]
                if len(valid_patches) == 0:
                    continue

                # Get all pixel indices for this batch sample's patches
                all_pixel_indices = np.concatenate([
                    patch_to_pixels[patch_idx] for patch_idx in valid_patches if patch_idx in patch_to_pixels
                ])

                # Apply baseline to all pixels at once
                x_perturbed_flat = x_perturbed[i].reshape(-1)
                if self.baseline_value == "black":
                    x_perturbed_flat[all_pixel_indices] = 0.0
                elif self.baseline_value == "white":
                    x_perturbed_flat[all_pixel_indices] = 1.0
                elif self.baseline_value == "mean":
                    mean_val = np.mean(x_batch[i])
                    x_perturbed_flat[all_pixel_indices] = mean_val
                else:
                    try:
                        baseline_val = float(self.baseline_value)
                        x_perturbed_flat[all_pixel_indices] = baseline_val
                    except:
                        x_perturbed_flat[all_pixel_indices] = 0.0

                x_perturbed[i] = x_perturbed_flat.reshape(x_batch.shape[1:])

            # Predict on perturbed input
            x_input = model.shape_input(x_perturbed, x_batch.shape, channel_first=True, batched=True)
            y_pred_perturb = model.predict(x_input)[np.arange(batch_size), y_batch]
            predictions.append(y_pred_perturb)

        # Return in the same format as quantus PixelFlipping
        if self.return_auc_per_sample:
            import quantus.helpers.utils as utils
            return utils.calculate_auc(np.stack(predictions, axis=1), batched=True).tolist()

        return np.stack(predictions, axis=1).tolist()

    def _extract_patch_attributions_exact(self, a_batch, n_patches):
        """Extract patch attributions by reversing the exact upsampling process."""
        batch_size = a_batch.shape[0]
        patch_size = 32 if n_patches == 49 else 16
        grid_size = int(np.sqrt(n_patches))

        if len(a_batch.shape) == 4:  # (N, 3, 224, 224)
            # Take first channel since all channels have identical values due to upsampling
            a_spatial = a_batch[:, 0, :, :]  # (N, 224, 224)
        else:  # (N, 224, 224)
            a_spatial = a_batch

        # Reverse the upsampling
        # Sample every patch_size-th pixel to get back the original grid
        patch_attributions = a_spatial[:, ::patch_size, ::patch_size]  # (N, grid_size, grid_size)

        return patch_attributions.reshape(batch_size, n_patches)


def faithfulness_pixel_flipping_optimized(n_patches=196):
    """Create optimized patch-level PixelFlipping that avoids sorting bottleneck."""
    features_in_step = 8 if n_patches == 196 else 4  # Fewer steps for B-32
    return PatchLevelPixelFlipping(
        features_in_step=features_in_step, perturb_baseline="black", normalise=False, n_patches=n_patches
    )


def _get_patch_to_pixels_mapping(height=224, width=224, patch_size=16, channels=3, n_patches=None):
    """Compute patch-to-pixel mappings for perturbation."""
    if n_patches is not None:
        grid_size = int(np.sqrt(n_patches))
    else:
        grid_size = width // patch_size

    patch_to_pixels = {}

    for patch_idx in range(grid_size * grid_size):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        start_row, end_row = row * patch_size, (row + 1) * patch_size
        start_col, end_col = col * patch_size, (col + 1) * patch_size

        # Vectorized pixel index computation
        c_indices = np.arange(channels)
        r_indices = np.arange(start_row, end_row)
        col_indices = np.arange(start_col, end_col)

        # Create meshgrid and flatten
        c_grid, r_grid, col_grid = np.meshgrid(c_indices, r_indices, col_indices, indexing='ij')
        flat_indices = c_grid.flatten() * height * width + r_grid.flatten() * width + col_grid.flatten()

        patch_to_pixels[patch_idx] = flat_indices

    return patch_to_pixels


def batch_patch_level_perturbation(arr, indices, perturb_baseline="black", patch_size=16, **kwargs):
    """
    Batched custom quantus perturbation function for patch-based models.
    Converts pixel indices to patch indices and perturbs entire patches.
    
    Args:
        arr: np.ndarray of shape (batch_size, n_features) - flattened images
        indices: np.ndarray of shape (batch_size, n_indices) - pixel indices to perturb
        perturb_baseline: baseline value for perturbation
        patch_size: size of patches (16 for B-16, 32 for B-32)
        
    Returns:
        Perturbed array with entire patches replaced by baseline
    """
    from quantus.helpers.utils import get_baseline_value
    
    batch_size, n_features = arr.shape
    
    # Infer dimensions
    channels = 3
    spatial_size = int(np.sqrt(n_features // channels))  # Should be 224
    height = width = spatial_size
    grid_size = width // patch_size  # patches per row/col
    
    # Get baseline value
    baseline_value = get_baseline_value(
        value=perturb_baseline,
        arr=arr,
        return_shape=(batch_size, n_features),
        batched=True,
        **kwargs
    )
    
    perturbed_arr = arr.copy()
    
    # Get patch-to-pixel mapping once
    patch_to_pixels = _get_patch_to_pixels_mapping(height, width, patch_size, channels)
    
    # Vectorized processing for all samples
    for batch_idx in range(batch_size):
        # Get valid indices for this sample (non-NaN)
        valid_mask = ~np.isnan(indices[batch_idx])
        if not np.any(valid_mask):
            continue
            
        valid_indices = indices[batch_idx][valid_mask].astype(int)
        
        # Convert pixel indices to patch indices (vectorized)
        spatial_indices = valid_indices % (height * width)
        row_indices = (spatial_indices // width) // patch_size
        col_indices = (spatial_indices % width) // patch_size
        patch_indices = row_indices * grid_size + col_indices
        
        # Get unique patches to perturb
        unique_patches = np.unique(patch_indices)
        unique_patches = unique_patches[unique_patches < grid_size * grid_size]
        
        # Collect all pixel indices from the patches
        if len(unique_patches) > 0:
            all_pixel_indices = np.concatenate([patch_to_pixels[p] for p in unique_patches])
            all_pixel_indices = all_pixel_indices[all_pixel_indices < n_features]
            
            # Apply baseline
            if baseline_value.ndim == 2:
                perturbed_arr[batch_idx, all_pixel_indices] = baseline_value[batch_idx, all_pixel_indices]
            else:
                perturbed_arr[batch_idx, all_pixel_indices] = baseline_value
    
    return perturbed_arr


def patch_level_perturbation(arr, indices, perturb_baseline="black", patch_size=16, **kwargs):
    """
    Custom quantus perturbation function for patch-based models.
    """
    from quantus.helpers.utils import get_baseline_value

    batch_size, n_features = arr.shape

    # Infer original image dimensions (assuming 3 channels, square images)
    channels = 3
    spatial_size = int(np.sqrt(n_features // channels))  # Should be 224
    height = width = spatial_size
    grid_w = width // patch_size  # patches per row

    # Get baseline value
    baseline_value = get_baseline_value(
        value=perturb_baseline, arr=arr, return_shape=(batch_size, n_features), batched=True, **kwargs
    )

    # Copy input array
    perturbed_arr = arr.copy()

    # Get patch-to-pixel mappings
    patch_to_pixels = _get_patch_to_pixels_mapping(height, width, patch_size, channels)

    # Process each sample in the batch
    for i in range(batch_size):
        # Find unique patches that contain the selected pixels
        valid_mask = ~np.isnan(indices[i])
        if not np.any(valid_mask):
            continue

        valid_indices = indices[i][valid_mask].astype(int)

        # Convert pixel indices to patch indices
        spatial_indices = valid_indices % (height * width)
        patch_indices = (spatial_indices // width // patch_size) * grid_w + (spatial_indices % width // patch_size)

        # Get unique patches
        unique_patches = np.unique(patch_indices)
        unique_patches = unique_patches[unique_patches < grid_w * grid_w]  # Safety check

        # Perturb all pixels in the identified patches
        if len(unique_patches) > 0:
            all_pixel_indices = np.concatenate([patch_to_pixels[patch_idx] for patch_idx in unique_patches])
            all_pixel_indices = all_pixel_indices[all_pixel_indices < n_features]  # Safety check

            if baseline_value.ndim == 2:  # batched baseline
                perturbed_arr[i, all_pixel_indices] = baseline_value[i, all_pixel_indices]
            else:  # single baseline value
                perturbed_arr[i, all_pixel_indices] = baseline_value

    return perturbed_arr


def calc_faithfulness(
    model,  # Can be any model type (HookedSAEViT, CLIP, etc.)
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch_expl: np.ndarray,
    device: torch.device,
    n_trials: int = 3,
    nr_runs: int = 50,
    subset_size: int = 98,
    n_patches: int = 196
) -> Dict[str, Any]:
    """
    Calculate faithfulness scores with statistical robustness through multiple trials.
    
    Args:
        model: The model (any type that quantus can wrap)
        x_batch: Batch of input images (numpy array)
        y_batch: Batch of target classes (numpy array)
        a_batch_expl: Batch of attributions (numpy array)
        device: Device to run calculations on
        n_trials: Number of trials to run for statistical robustness
        nr_runs: Number of random perturbations per image
        subset_size: Size of feature subset to perturb
        n_patches: Number of patches (49 for B-32, 196 for B-16)
        
    Returns:
        Dictionary with faithfulness statistics for each estimator
    """
    print(f'Settings - n_trials: {n_trials}, nr_runs: {nr_runs}, subset_size: {subset_size}, patches: {n_patches}')

    # Adjust subset size for B-32 models
    if n_patches == 49:
        subset_size = min(subset_size, 25)  # Use fewer patches for B-32

    # Define estimators
    estimator_configs = [
        FaithfulnessEstimatorConfig(
            name="FaithfulnessCorrelation",
            n_trials=n_trials,
            estimator_fn=faithfulness_correlation,
            kwargs={
                "subset_size": min(20, n_patches // 2),
                "nr_runs": nr_runs
            }
        ),
        FaithfulnessEstimatorConfig(
            name="PixelFlipping_PatchLevel",
            n_trials=1,  # Fast patch-level version
            estimator_fn=lambda: faithfulness_pixel_flipping_optimized(n_patches),
        ),
    ]

    results_by_estimator = {}

    # Suppress quantus logging
    quantus_logger = logging.getLogger('quantus')
    original_level = quantus_logger.level
    quantus_logger.setLevel(logging.ERROR)

    for estimator_config in estimator_configs:
        print(f"Running estimator: {estimator_config.name}")

        estimator_results = _run_estimator_trials(
            estimator_config, model, x_batch, y_batch, a_batch_expl, device, n_trials, nr_runs, subset_size
        )

        if estimator_results:
            results_by_estimator[estimator_config.name] = estimator_results

    quantus_logger.setLevel(original_level)
    return results_by_estimator


def _run_estimator_trials(
    estimator_config: FaithfulnessEstimatorConfig,
    model,  # Can be any model type
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch_expl: np.ndarray,
    device: torch.device,
    n_trials: int,
    nr_runs: int,
    subset_size: int
) -> Dict[str, Any]:
    """Run multiple trials for a single estimator."""
    all_results = []

    for trial in range(estimator_config.n_trials):
        # Set reproducible seed for this trial
        original_state = np.random.get_state()
        np.random.seed(42 + trial)

        try:
            estimator = estimator_config.create_estimator()

            # Run estimator
            # Larger batch size for better GPU utilization
            faithfulness_estimate = estimator(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch_expl,
                device=str(device),
                batch_size=2048
            )

            # Process results
            scores = _process_estimator_output(faithfulness_estimate, len(y_batch))
            all_results.append(scores)

        except Exception as e:
            print(f"Error in trial {trial} for {estimator_config.name}: {e}")
            all_results.append(np.full(len(y_batch), np.nan))
        finally:
            np.random.set_state(original_state)

    if not all_results:
        return {}

    # Calculate statistics across trials
    all_results = np.stack(all_results, axis=0)
    return {
        "mean_scores": np.nanmean(all_results, axis=0),
        "std_scores": np.nanstd(all_results, axis=0),
        "all_trials": all_results,
        "n_trials": n_trials,
        "nr_runs": nr_runs,
        "subset_size": subset_size
    }


def _process_estimator_output(output: Any, expected_length: int) -> np.ndarray:
    """Process estimator output into a consistent numpy array format."""
    if isinstance(output, dict):
        scores = np.array(output.get('auc', [0]))
    else:
        scores = np.array(output)

    # Handle multi-dimensional results
    if len(scores.shape) > 1 and scores.shape[0] != expected_length:
        scores = np.mean(scores, axis=tuple(range(1, len(scores.shape))))

    return scores


def convert_patch_attribution_to_image(attribution, n_patches=196):
    """
    Convert patch attribution to 224x224 image format for quantus compatibility.
    Uses efficient array operations to avoid explicit loops.
    """
    patch_size = 32 if n_patches == 49 else 16
    grid_size = int(np.sqrt(n_patches))

    # Ensure correct shape
    if attribution.ndim == 1:
        attribution = attribution[:n_patches]
        attr_grid = attribution.reshape(grid_size, grid_size)
    else:
        attr_grid = attribution

    # Upsample to 224x224 using repeat (more efficient than interpolation)
    attr_image = np.repeat(np.repeat(attr_grid, patch_size, axis=0), patch_size, axis=1)

    # Ensure correct size for B-32 models (7*32=224 works perfectly)
    if attr_image.shape[0] != 224:
        # This shouldn't happen but handle it just in case
        from scipy.ndimage import zoom
        zoom_factor = 224 / attr_image.shape[0]
        attr_image = zoom(attr_image, zoom_factor, order=1)

    # Expand to 3 channels to match input format (C, H, W)
    attr_image_3c = np.stack([attr_image] * 3, axis=0)

    return attr_image_3c


def evaluate_faithfulness_for_results(
    config: PipelineConfig,
    model,
    device: torch.device,
    classification_results: List[ClassificationResult],
    batch_size: int = 2048,
    clip_classifier=None
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluate faithfulness scores with patch-level attributions.
    Processes data in batches to avoid memory issues.
    Supports both B-16 (196 patches) and B-32 (49 patches) models.
    """
    import gc

    # Detect patch configuration
    is_patch32 = False
    if hasattr(config.classify, 'clip_model_name') and config.classify.clip_model_name:
        model_name = config.classify.clip_model_name.lower()
        is_patch32 = "patch32" in model_name or "b-32" in model_name or "b32" in model_name

    n_patches = 49 if is_patch32 else 196

    total_samples = len(classification_results)
    num_batches = (total_samples + batch_size - 1) // batch_size

    print(f"Processing {total_samples} samples in {num_batches} batches")

    all_faithfulness_scores = {}
    all_y_labels = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_results = classification_results[start_idx:end_idx]

        print(f"\nBatch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx-1})")

        # Process batch data
        batch_data = _prepare_batch_data(config, batch_results, n_patches)
        if not batch_data:
            print(f"No valid samples in batch {batch_idx + 1}")
            continue

        x_batch, y_batch, a_batch = batch_data

        # Calculate faithfulness for this batch
        # Reduce parameters for faster evaluation:
        # - n_trials: 1 trial is often sufficient
        # - nr_runs: 10-20 runs gives reasonable estimates
        # - subset_size: smaller subsets are faster
        batch_faithfulness = calc_faithfulness(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch_expl=a_batch,
            device=device,
            n_trials=3,  # Reduced from 3 for speed
            nr_runs=20,  # Reduced from 50 for speed
            subset_size=98 if n_patches == 196 else 10,  # Reduced for speed
            n_patches=n_patches
        )

        # Store labels
        all_y_labels.extend(y_batch.tolist())

        # Aggregate scores from this batch
        _aggregate_batch_scores(all_faithfulness_scores, batch_faithfulness)

        # Clean up memory
        del x_batch, y_batch, a_batch
        gc.collect()
        torch.cuda.empty_cache()

    # Compute final statistics
    final_results = _compute_final_statistics(all_faithfulness_scores)

    print(f"\nProcessed {len(all_y_labels)} total samples")

    return final_results, np.array(all_y_labels)


def _prepare_batch_data(config: PipelineConfig,
                        batch_results: List[ClassificationResult],
                        n_patches: int = 196) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Prepare batch data for faithfulness evaluation."""
    x_list = []
    y_list = []
    a_list = []

    for result in batch_results:
        try:
            # Load image directly as numpy array
            img = Image.open(result.image_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Convert to CHW format
            img_array = np.transpose(img_array, (2, 0, 1))

            # Apply normalization (ImageNet stats)
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_array = (img_array - mean) / std

            # Load attribution
            attr_map = np.load(result.attribution_paths.raw_attribution_path)
            attr_map = _normalize_attribution_format(attr_map, n_patches)

            if attr_map is None:
                continue

            x_list.append(img_array)
            y_list.append(result.prediction.predicted_class_idx)
            a_list.append(convert_patch_attribution_to_image(attr_map, n_patches))

        except Exception as e:
            print(f"Warning: Could not process {result.image_path.name}: {e}")
            continue

    if not x_list:
        return None

    return np.stack(x_list), np.array(y_list), np.stack(a_list)


def _normalize_attribution_format(attr_map: np.ndarray, n_patches: int = 196) -> Optional[np.ndarray]:
    """Normalize attribution to patch format."""
    grid_size = int(np.sqrt(n_patches))
    patch_size = 32 if n_patches == 49 else 16

    # Handle different input formats
    if attr_map.shape == (224, 224):
        # Downsample to patch level
        attr_map = attr_map.reshape(grid_size, patch_size, grid_size, patch_size).mean(axis=(1, 3))

    if attr_map.ndim == 2:
        attr_map = attr_map.flatten()

    if attr_map.shape[0] != n_patches:
        print(f"Warning: Expected {n_patches} features, got {attr_map.shape[0]}")
        return None

    return attr_map


def _aggregate_batch_scores(all_scores: Dict[str, Dict], batch_scores: Dict[str, Dict]) -> None:
    """Aggregate scores from a batch into the overall scores dictionary."""
    for estimator_name, estimator_data in batch_scores.items():
        if estimator_name not in all_scores:
            all_scores[estimator_name] = {
                'mean_scores': [],
                'std_scores': [],
                'metadata': {
                    'n_trials': estimator_data.get('n_trials', 3),
                    'nr_runs': estimator_data.get('nr_runs', 50),
                    'subset_size': estimator_data.get('subset_size', 98)
                }
            }

        # Add scores from this batch
        mean_scores = estimator_data.get('mean_scores', [])
        std_scores = estimator_data.get('std_scores', [])

        if isinstance(mean_scores, np.ndarray):
            mean_scores = mean_scores.tolist()
        if isinstance(std_scores, np.ndarray):
            std_scores = std_scores.tolist()

        all_scores[estimator_name]['mean_scores'].extend(mean_scores)
        all_scores[estimator_name]['std_scores'].extend(std_scores)


def _compute_final_statistics(all_scores: Dict[str, Dict]) -> Dict[str, Dict]:
    """Compute final statistics from aggregated scores."""
    final_results = {}

    for estimator_name, data in all_scores.items():
        mean_scores = np.array(data['mean_scores'])
        std_scores = np.array(data['std_scores'])
        metadata = data['metadata']

        if len(mean_scores) == 0:
            final_results[estimator_name] = {
                'mean_scores': [],
                'std_scores': [],
                'overall': {
                    'mean': 0,
                    'std': 0,
                    'median': 0,
                    'min': 0,
                    'max': 0,
                    'count': 0
                },
                'error': 'No valid scores collected'
            }
        else:
            final_results[estimator_name] = {
                'mean_scores': mean_scores.tolist(),
                'std_scores': std_scores.tolist(),
                **metadata, 'overall': {
                    'mean': float(np.mean(mean_scores)),
                    'std': float(np.std(mean_scores)),
                    'median': float(np.median(mean_scores)),
                    'min': float(np.min(mean_scores)),
                    'max': float(np.max(mean_scores)),
                    'count': len(mean_scores)
                }
            }

    return final_results


def calculate_faithfulness_stats_by_class(faithfulness_results: Dict[str, Dict[str, Any]],
                                          class_labels: np.ndarray) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Calculate statistics for faithfulness scores grouped by class.
    
    Args:
        faithfulness_results: Dictionary with faithfulness results per estimator
        class_labels: Array of class labels
        
    Returns:
        Dictionary with per-class statistics for each estimator
    """
    stats_by_estimator = {}

    for estimator_name, estimator_results in faithfulness_results.items():
        scores = np.array(estimator_results["mean_scores"])
        stds = np.array(estimator_results["std_scores"])

        # Group scores by class
        class_stats = {}
        for class_idx in np.unique(class_labels):
            mask = class_labels == class_idx
            class_scores = scores[mask]
            class_stds = stds[mask]

            if len(class_scores) > 0:
                class_stats[int(class_idx)] = {
                    'count': len(class_scores),
                    'mean': float(np.mean(class_scores)),
                    'median': float(np.median(class_scores)),
                    'min': float(np.min(class_scores)),
                    'max': float(np.max(class_scores)),
                    'std': float(np.std(class_scores)),
                    'avg_trial_std': float(np.mean(class_stds))
                }

        stats_by_estimator[estimator_name] = class_stats

    return stats_by_estimator


def handle_array_values(arr):
    """Helper function to process array-like values and make them JSON serializable."""
    if hasattr(arr, 'tolist'):
        arr = arr.tolist()

    if isinstance(arr, list):
        # If the array has nested arrays, convert them too
        return [
            handle_array_values(item) if hasattr(item, 'tolist') or isinstance(item, list) else item for item in arr
        ]

    return arr


def evaluate_and_report_faithfulness(
    config: PipelineConfig,
    model,  # Can be any model type
    device: torch.device,
    classification_results: List[ClassificationResult],
    clip_classifier=None
) -> Dict[str, Any]:
    """
    Evaluate faithfulness and report statistics.
    
    Args:
        config: Pipeline configuration
        model: The model (can be HookedSAEViT, CLIP, etc.)
        device: Device to run calculations on
        classification_results: List of classification results
        clip_classifier: Optional CLIP classifier if using CLIP
        
    Returns:
        Dictionary with overall and per-class statistics
    """
    # Evaluate faithfulness
    faithfulness_results, class_labels = evaluate_faithfulness_for_results(
        config, model, device, classification_results, clip_classifier=clip_classifier
    )

    # Get dataset config for class names
    dataset_config = get_dataset_config(config.file.dataset_name)

    # Build results structure
    results = _build_results_structure(config, faithfulness_results, class_labels, dataset_config)

    # Print summary
    _print_faithfulness_summary(results['metrics'], dataset_config)

    # Save results
    _save_faithfulness_results(config, faithfulness_results, class_labels, results)

    return results


def _build_results_structure(
    config: PipelineConfig,
    faithfulness_results: Dict[str, Dict],
    class_labels: np.ndarray,
    dataset_config=None
) -> Dict[str, Any]:
    """Build the results structure with statistics."""
    results = {'dataset': config.file.dataset_name, 'metrics': {}, 'class_labels': class_labels.tolist()}

    # Calculate statistics for each estimator
    class_stats_all = calculate_faithfulness_stats_by_class(faithfulness_results, class_labels)

    for estimator_name, estimator_results in faithfulness_results.items():
        if "mean_scores" not in estimator_results:
            continue

        scores = np.array(estimator_results["mean_scores"])
        stds = np.array(estimator_results.get("std_scores", np.zeros_like(scores)))

        # Overall statistics
        overall_stats = {
            'count': len(scores),
            'mean': float(np.nanmean(scores)),
            'median': float(np.nanmedian(scores)),
            'min': float(np.nanmin(scores)),
            'max': float(np.nanmax(scores)),
            'std': float(np.nanstd(scores)),
            'avg_trial_std': float(np.nanmean(stds)),
            'method_params': {
                'n_trials': estimator_results.get("n_trials", 3),
                'nr_runs': estimator_results.get("nr_runs", 50),
                'subset_size': estimator_results.get("subset_size", 98)
            }
        }

        # Store results
        results['metrics'][estimator_name] = {
            'overall': overall_stats,
            'by_class': class_stats_all[estimator_name],
            'mean_scores': handle_array_values(scores),
            'std_scores': handle_array_values(stds)
        }

    return results


def _print_faithfulness_summary(metrics: Dict[str, Dict], dataset_config=None):
    """Print a summary of faithfulness metrics."""
    for estimator_name, estimator_data in metrics.items():
        overall = estimator_data['overall']
        print(f"\n{estimator_name} faithfulness statistics:")
        print(f"  Mean: {overall['mean']:.4f}")
        print(f"  Median: {overall['median']:.4f}")
        print(f"  Count: {overall['count']}")
        print(f"  Avg trial std: {overall['avg_trial_std']:.4f}")

        print(f"\n{estimator_name} per-class statistics:")
        for class_idx, stats in estimator_data['by_class'].items():
            class_name = dataset_config.idx_to_class.get(
                class_idx, f"Class {class_idx}"
            ) if dataset_config else f"Class {class_idx}"
            print(f"  {class_name}: mean={stats['mean']:.4f}, count={stats['count']}")


def _save_faithfulness_results(
    config: PipelineConfig, faithfulness_results: Dict[str, Dict], class_labels: np.ndarray, results: Dict[str, Any]
):
    """Save faithfulness results to files."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Save JSON statistics
    json_path = config.file.output_dir / f"faithfulness_stats{config.file.output_suffix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFaithfulness statistics saved to {json_path}")

    # Save raw scores for each estimator
    for estimator_name, estimator_results in faithfulness_results.items():
        scores_path = config.file.output_dir / f"faithfulness_scores_{estimator_name}{config.file.output_suffix}"

        save_dict = {
            'mean_scores': estimator_results["mean_scores"],
            'std_scores': estimator_results.get("std_scores", []),
            'class_labels': class_labels
        }

        np.savez(scores_path, **save_dict)
        print(f"Raw scores for {estimator_name} saved to {scores_path}.npz")
