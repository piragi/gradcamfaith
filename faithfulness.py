import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path  # Already in your other code, just a reminder
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import quantus
import torch
import torch.nn.functional as F

import vit.model as model
import vit.preprocessing as preprocessing
from config import PipelineConfig
from data_types import ClassificationResult


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
        perturb_func=patch_level_perturbation,  # Use patch-level perturbation
        perturb_baseline="black",
        similarity_func=quantus.similarity_func.correlation_spearman,
        subset_size=subset_size,
        return_aggregate=False,
        normalise=False,
        nr_runs=nr_runs
    )


def faithfulness_pixel_flipping():
    return quantus.PixelFlipping(
        features_in_step=49,  # Process 49 patches at a time (196/4 = 4 model calls)
        perturb_baseline="black",
        normalise=False,
        # Use quantus default perturbation (no custom perturb_func)
    )


class PatchLevelPixelFlipping(quantus.PixelFlipping):
    """
    Custom PixelFlipping that works directly with 196-dimensional patch attributions.
    Avoids expensive sorting of 150k duplicate pixel values by sorting patches directly.
    
    Inherits all quantus infrastructure while optimizing for patch-based models.
    """

    def __init__(self, features_in_step: int = 49, patch_size: int = 16, perturb_baseline: str = "black", **kwargs):
        # Initialize parent with dummy perturbation function (we'll override evaluation)
        super().__init__(
            features_in_step=features_in_step,
            perturb_func=patch_level_perturbation,  # We'll use this for actual perturbation
            perturb_baseline=perturb_baseline,
            **kwargs
        )
        self.patch_size = patch_size
        self.baseline_value = perturb_baseline

    def evaluate_batch(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        **kwargs,
    ):
        """
        Fast patch-level PixelFlipping: Uses same sorting as standard but patch-level perturbation.
        """
        # Extract 196-dimensional patch attributions (FAST!)
        batch_size = a_batch.shape[0]
        if a_batch.shape[-1] > 196:
            a_batch_patches = self._extract_patch_attributions_exact(a_batch)
        else:
            a_batch_patches = a_batch.reshape(batch_size, -1)

        # Sort patches by attribution importance (196 values, not 150k!) - THE SPEED BOOST
        patch_indices_sorted = np.argsort(-a_batch_patches, axis=1)

        # Calculate number of perturbation steps
        n_patches = 196
        n_steps = math.ceil(n_patches / self.features_in_step)

        # Store predictions for each step
        predictions = []
        x_perturbed = x_batch.copy()

        # Get patch-to-pixel mapping
        patch_to_pixels = _get_patch_to_pixels_mapping()

        for step in range(n_steps):
            # Get patches to remove in this step
            start_idx = step * self.features_in_step
            end_idx = min((step + 1) * self.features_in_step, n_patches)

            patches_to_remove = patch_indices_sorted[:, start_idx:end_idx]

            # Apply patch-level perturbation (fast!)
            for i in range(batch_size):
                valid_patches = patches_to_remove[i][patches_to_remove[i] < 196]
                if len(valid_patches) == 0:
                    continue

                # Get all pixel indices for this batch sample's patches
                all_pixel_indices = np.concatenate([patch_to_pixels[patch_idx] for patch_idx in valid_patches])

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

    def _extract_patch_attributions_exact(self, a_batch):
        """Extract 196-dimensional patch attributions by reversing the exact upsampling process."""
        batch_size = a_batch.shape[0]

        if len(a_batch.shape) == 4:  # (N, 3, 224, 224)
            # Take first channel since all channels have identical values due to upsampling
            a_spatial = a_batch[:, 0, :, :]  # (N, 224, 224)
        else:  # (N, 224, 224)
            a_spatial = a_batch

        # Reverse the upsampling: np.repeat(np.repeat(attr_grid, 16, axis=0), 16, axis=1)
        # Sample every 16th pixel to get back the original 14x14 grid
        patch_attributions = a_spatial[:, ::16, ::16]  # (N, 14, 14)

        return patch_attributions.reshape(batch_size, 196)


def faithfulness_pixel_flipping_optimized():
    """Create optimized patch-level PixelFlipping that avoids sorting bottleneck."""
    return PatchLevelPixelFlipping(
        features_in_step=8,  # Process 8 patches at a time
        perturb_baseline="black",
        normalise=False,
    )


def _get_patch_to_pixels_mapping(height=224, width=224, patch_size=16, channels=3):
    """Compute patch-to-pixel mappings for perturbation."""
    grid_w = width // patch_size  # 14 patches per row
    patch_to_pixels = {}

    for patch_idx in range(196):  # 14x14 patches
        row = patch_idx // grid_w
        col = patch_idx % grid_w
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
    grid_w = width // patch_size  # 14 patches per row

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
        unique_patches = unique_patches[unique_patches < 196]  # Safety check

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
    vit_model: model.VisionTransformer,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch_expl: np.ndarray,
    device: torch.device,
    n_trials: int = 3,  # Reduced from 5 for faster computation
    nr_runs: int = 50,  # Reduced from 100 - biggest performance impact!
    subset_size: int = 98,  # Reduced from 196 for faster sampling
    patch_size: int = 16,
) -> Dict[str, Any]:
    """
    Calculate faithfulness scores with statistical robustness through multiple trials.
    
    Args:
        vit_model: The ViT model
        x_batch: Batch of input images (numpy array)
        y_batch: Batch of target classes (numpy array)
        a_batch_expl: Batch of attributions (numpy array)
        device: Device to run calculations on
        n_trials: Number of trials to run for statistical robustness
        nr_runs: Number of random perturbations per image
        subset_size: Size of feature subset to perturb (should be less than total features)
        
    Returns:
        Dictionary with faithfulness statistics for each estimator
    """
    print(f'Settings - n_trials: {n_trials}, nr_runs: {nr_runs}, subset_size: {subset_size}')

    # Define all the estimators to use
    estimator_configs = [
        FaithfulnessEstimatorConfig(
            name="FaithfulnessCorrelation",
            n_trials=n_trials,
            estimator_fn=faithfulness_correlation,
            kwargs={
                "subset_size": 20,
                "nr_runs": nr_runs
            }
        ),
        FaithfulnessEstimatorConfig(
            name="PixelFlipping_PatchLevel",
            n_trials=1,  # Fast patch-level version (~100x faster)
            estimator_fn=faithfulness_pixel_flipping_optimized,
        ),
    ]
    results_by_estimator = {}

    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.WARNING)

    for estimator_config in estimator_configs:
        print(f"Running estimator: {estimator_config.name}")
        all_results = []
        estimator_results = {
            "n_trials": n_trials,
            "nr_runs": nr_runs,
            "subset_size": subset_size,
        }

        for trial in range(estimator_config.n_trials):
            # Set seed for this trial
            trial_seed = 42 + trial

            # Save original numpy random state
            original_state = np.random.get_state()

            # Set specific seed for this evaluation
            np.random.seed(trial_seed)

            try:
                # Create the estimator
                faithfulness_estimator = estimator_config.create_estimator()
                print(faithfulness_estimator.get_params)

                # Run the estimator
                faithfulness_estimate = faithfulness_estimator(
                    model=vit_model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch_expl,
                    device=str(device),
                )

                # Process results - estimators can return different types of data
                if isinstance(faithfulness_estimate, dict):
                    # For estimators that return dictionaries (like ROAD)
                    faithfulness_score = np.array(faithfulness_estimate.get('auc', [0]))
                else:
                    faithfulness_score = np.array(faithfulness_estimate)

                # Handle multi-dimensional results - some estimators return arrays per sample
                # If results are multi-dimensional, compute mean across dimensions
                if len(faithfulness_score.shape) > 1 and faithfulness_score.shape[0] == len(y_batch):
                    # For 2D arrays where rows are samples, keep as is
                    pass
                elif len(faithfulness_score.shape) > 1:
                    # For arrays with multiple values per sample, take mean
                    faithfulness_score = np.mean(
                        faithfulness_score, axis=tuple(range(1, len(faithfulness_score.shape)))
                    )

                all_results.append(faithfulness_score)
            except Exception as e:
                print(f"Error running estimator {estimator_config.name}: {e}")
                # If the estimator fails, use NaN values
                dummy_result = np.full_like(y_batch, np.nan, dtype=float)
                all_results.append(dummy_result)
            finally:
                # Restore random state
                np.random.set_state(original_state)

        if all_results:
            # Stack results from all trials
            all_results = np.stack(all_results, axis=0)

            # Calculate mean and std across trials for each sample
            mean_scores = np.nanmean(all_results, axis=0)
            std_scores = np.nanstd(all_results, axis=0)

            # Add the evaluation metrics to the results
            estimator_results.update({"mean_scores": mean_scores, "std_scores": std_scores, "all_trials": all_results})

            results_by_estimator[estimator_config.name] = estimator_results

    root_logger.setLevel(original_level)

    return results_by_estimator


def convert_patch_attribution_to_image(attribution_196, patch_size=16):
    """
    Convert 196-dimensional patch attribution to 224x224 image format for quantus compatibility.
    Uses efficient array operations to avoid explicit loops.
    """
    # Reshape to 14x14 grid
    attr_grid = attribution_196.reshape(14, 14)

    # Upsample to 224x224 using repeat (more efficient than interpolation)
    attr_image = np.repeat(np.repeat(attr_grid, patch_size, axis=0), patch_size, axis=1)

    # Expand to 3 channels to match input format (C, H, W)
    attr_image_3c = np.stack([attr_image] * 3, axis=0)

    return attr_image_3c


def evaluate_faithfulness_for_results(
    config: PipelineConfig, vit_model: model.VisionTransformer, device: torch.device,
    classification_results: List[ClassificationResult]
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluate faithfulness scores with patch-level attributions converted to quantus-compatible format.
    """
    x_batch_list = []
    y_batch_list = []
    a_batch_converted_list = []  # Store converted attributions

    for result in classification_results:
        # Load high-res image and labels
        img_path = result.image_path
        _, input_tensor = preprocessing.preprocess_image(str(img_path), img_size=config.classify.target_size[0])
        x_batch_list.append(input_tensor.cpu().numpy())
        class_idx = result.prediction.predicted_class_idx
        y_batch_list.append(class_idx)

        try:
            # Load attribution file
            attribution_path = result.attribution_paths.raw_attribution_path
            attr_map = np.load(attribution_path)

            # Convert to 196-dimensional if needed
            if attr_map.shape == (224, 224):
                # Downsample back to 14x14 patches by average pooling
                attr_map = attr_map.reshape(14, 16, 14, 16).mean(axis=(1, 3))

            # Flatten to 196-dimensional vector
            if attr_map.ndim == 2:
                attr_map = attr_map.flatten()

            # Ensure we have exactly 196 features
            if attr_map.shape[0] != 196:
                print(f"Warning: Expected 196 features, got {attr_map.shape[0]}. Skipping.")
                x_batch_list.pop()
                y_batch_list.pop()
                continue

            # Convert to quantus-compatible format (3, 224, 224)
            attr_image = convert_patch_attribution_to_image(attr_map)
            a_batch_converted_list.append(attr_image)

        except Exception as e:
            print(f"Warning: Could not process attribution for {Path(img_path).name}. Skipping. Error: {e}")
            x_batch_list.pop()
            y_batch_list.pop()
            continue

    # Convert lists to final numpy arrays
    x_batch = np.stack(x_batch_list)
    y_batch = np.array(y_batch_list)
    a_batch_converted = np.stack(a_batch_converted_list)  # Shape: (N, 3, 224, 224)

    # Call faithfulness with converted attributions
    faithfulness_results = calc_faithfulness(
        vit_model=vit_model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch_expl=a_batch_converted,  # Pass quantus-compatible attributions
        device=device,
    )

    return faithfulness_results, y_batch


def calculate_faithfulness_stats_by_class(faithfulness_results: Dict[str, Dict[str, Any]],
                                          class_labels: np.ndarray) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Calculate statistics for faithfulness scores grouped by class for each estimator.
    
    Args:
        faithfulness_results: Dictionary with faithfulness results per estimator
        class_labels: Array of class labels corresponding to each score
        
    Returns:
        Dictionary with estimator names as keys and per-class statistics as values
    """
    stats_by_estimator_and_class = {}

    for estimator_name, estimator_results in faithfulness_results.items():
        # Use the mean scores for statistics
        faithfulness_scores = estimator_results["mean_scores"]
        faithfulness_stds = estimator_results["std_scores"]

        # Group scores by class
        scores_by_class = defaultdict(list)
        stds_by_class = defaultdict(list)

        for score, std, label in zip(faithfulness_scores, faithfulness_stds, class_labels):
            # Handle multi-dimensional scores - some estimators return arrays per sample
            if hasattr(score, 'shape') and len(score.shape) > 0:
                # For array scores, use mean as the representative value
                score_value = float(np.mean(score))
            else:
                score_value = float(score)

            if hasattr(std, 'shape') and len(std.shape) > 0:
                std_value = float(np.mean(std))
            else:
                std_value = float(std)

            scores_by_class[int(label)].append(score_value)
            stds_by_class[int(label)].append(std_value)

        # Calculate statistics for each class
        stats_by_class = {}
        for class_idx, scores in scores_by_class.items():
            scores_array = np.array(scores)
            stds_array = np.array(stds_by_class[class_idx])

            stats_by_class[class_idx] = {
                'count': len(scores),
                'mean': float(np.mean(scores_array)),
                'median': float(np.median(scores_array)),
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'std': float(np.std(scores_array)),
                'avg_trial_std': float(np.mean(stds_array)),  # Average std across trials
            }

        stats_by_estimator_and_class[estimator_name] = stats_by_class

    return stats_by_estimator_and_class


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
    vit_model: model.VisionTransformer,
    device: torch.device,
    classification_results: List[ClassificationResult],
) -> Dict[str, Any]:
    """
    Evaluate faithfulness with statistical robustness and report statistics by class.
    
    Args:
        config: Pipeline configuration
        vit_model: The ViT model
        device: Device to run calculations on
        classification_results: List of classification results
        
    Returns:
        Dictionary with overall and per-class statistics for each estimator
    """
    faithfulness_results, class_labels = evaluate_faithfulness_for_results(
        config, vit_model, device, classification_results
    )

    # Initialize results structure
    results = {
        'boosting_method': 'head_boosting',
        'boost_factor_per_head': config.classify.head_boost_value,
        'head_boost_factors': config.classify.head_boost_factor_per_head_per_class,
        'metrics': {},
        'class_labels': class_labels.tolist()
    }

    # Check if existing results file exists
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_path = config.file.output_dir / f"faithfulness_stats{config.file.output_suffix}_{timestamp}.json"
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
                # Preserve existing data
                for key, value in existing_results.items():
                    if key != 'metrics' and key != 'class_labels':
                        results[key] = value
        except Exception as e:
            print(f"Error reading existing results file: {e}")

    # Calculate and store results for each estimator
    for estimator_name, estimator_results in faithfulness_results.items():
        # Get the mean scores for overall statistics
        faithfulness_scores = estimator_results["mean_scores"]
        faithfulness_stds = estimator_results["std_scores"]

        # Calculate overall statistics
        overall_stats = {
            'count':
            len(faithfulness_scores),
            'mean':
            float(
                np.nanmean(
                    np.array([
                        np.mean(s) if hasattr(s, 'shape') and len(s.shape) > 0 else s for s in faithfulness_scores
                    ])
                )
            ),
            'median':
            float(
                np.nanmedian(
                    np.array([
                        np.mean(s) if hasattr(s, 'shape') and len(s.shape) > 0 else s for s in faithfulness_scores
                    ])
                )
            ),
            'min':
            float(
                np.nanmin(
                    np.array([
                        np.mean(s) if hasattr(s, 'shape') and len(s.shape) > 0 else s for s in faithfulness_scores
                    ])
                )
            ),
            'max':
            float(
                np.nanmax(
                    np.array([
                        np.mean(s) if hasattr(s, 'shape') and len(s.shape) > 0 else s for s in faithfulness_scores
                    ])
                )
            ),
            'std':
            float(
                np.nanstd(
                    np.array([
                        np.mean(s) if hasattr(s, 'shape') and len(s.shape) > 0 else s for s in faithfulness_scores
                    ])
                )
            ),
            'avg_trial_std':
            float(
                np.nanmean(
                    np.array([np.mean(s) if hasattr(s, 'shape') and len(s.shape) > 0 else s for s in faithfulness_stds])
                )
            ),  # Average std across trials
            'method_params': {
                'n_trials': estimator_results["n_trials"],
                'nr_runs': estimator_results["nr_runs"],
                'subset_size': estimator_results["subset_size"]
            }
        }

        # Calculate per-class statistics for this estimator
        class_stats = calculate_faithfulness_stats_by_class({estimator_name: estimator_results},
                                                            class_labels)[estimator_name]

        # Store results for this estimator
        results['metrics'][estimator_name] = {
            'overall': overall_stats,
            'by_class': class_stats,
            'mean_scores': handle_array_values(faithfulness_scores),
            'std_scores': handle_array_values(faithfulness_stds)
        }

        # Print summary with added stability information
        print(f"\n{estimator_name} faithfulness statistics (across {estimator_results['n_trials']} trials):")
        print(f"  Mean: {overall_stats['mean']:.4f}")
        print(f"  Median: {overall_stats['median']:.4f}")
        print(f"  Count: {overall_stats['count']}")
        print(f"  Avg trial std: {overall_stats['avg_trial_std']:.4f} (lower is more stable)")

        print(f"\n{estimator_name} per-class faithfulness statistics:")
        for class_idx, stats in class_stats.items():
            print(f"  Class {class_idx}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Median: {stats['median']:.4f}")
            print(f"    Count: {stats['count']}")
            print(f"    Avg trial std: {stats['avg_trial_std']:.4f}")

    # Save scores for each estimator
    for estimator_name, estimator_results in faithfulness_results.items():
        scores_path = config.file.output_dir / f"faithfulness_scores_{estimator_name}{config.file.output_suffix}.npy"

        # Prepare data for saving
        save_dict = {
            'mean_scores': estimator_results["mean_scores"],
            'std_scores': estimator_results["std_scores"],
            'class_labels': class_labels,
            'all_trials': estimator_results["all_trials"]
        }

        # Add special metrics if they exist
        if 'road_curves' in estimator_results:
            save_dict['road_curves'] = estimator_results['road_curves']

        # Save to NPZ
        np.savez(scores_path, **save_dict)
        print(f"Raw scores for {estimator_name} saved to {scores_path}.npz")

    # Save JSON stats
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Faithfulness statistics saved to {results_path}")

    return results
