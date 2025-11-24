"""
Faithfulness evaluation for the unified pipeline.
Adapts the original faithfulness.py to work with HookedSAEViT and CLIP models.
"""

import gc
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
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


def calc_faithfulness(
    model,  # Can be any model type (HookedSAEViT, CLIP, etc.)
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    a_batch_expl: np.ndarray,
    device: torch.device,
    config: PipelineConfig,
    n_patches: int = 196
) -> Dict[str, Any]:
    """
    Calculate faithfulness scores with statistical robustness through multiple trials.

    Args:
        model: PyTorch model (HookedViT, CLIP, or CLIPModelWrapper)
        x_batch: Batch of input images (numpy array)
        y_batch: Batch of target classes (numpy array)
        a_batch_expl: Batch of attributions (numpy array)
        device: Device to run calculations on
        config: Pipeline configuration with faithfulness parameters
        n_patches: Number of patches (49 for B-32, 196 for B-16)

    Returns:
        Dictionary with faithfulness statistics for each estimator
    """
    # Get parameters from config
    n_trials = config.faithfulness.n_trials
    nr_runs = config.faithfulness.nr_runs
    gpu_batch_size = config.faithfulness.gpu_batch_size

    # Use appropriate subset size based on model architecture
    subset_size = config.faithfulness.subset_size_b32 if n_patches == 49 else config.faithfulness.subset_size

    print(
        f'Settings - n_trials: {n_trials}, nr_runs: {nr_runs}, subset_size: {subset_size}, patches: {n_patches}, gpu_batch_size: {gpu_batch_size}'
    )

    # Define estimators
    estimator_configs = [
        FaithfulnessEstimatorConfig(
            name="FaithfulnessCorrelation",
            n_trials=n_trials,
            estimator_fn=faithfulness_correlation,
            kwargs={
                "subset_size": subset_size,
                "nr_runs": nr_runs,
                "n_patches": n_patches,
                "perturb_baseline": config.faithfulness.perturb_baseline
            }
        ),
        FaithfulnessEstimatorConfig(
            name="PixelFlipping",
            n_trials=1,
            estimator_fn=lambda: faithfulness_pixel_flipping(
                n_patches=n_patches,
                features_in_step=config.faithfulness.features_in_step,
                perturb_baseline=config.faithfulness.perturb_baseline
            ),
        ),
    ]

    results_by_estimator = {}

    for estimator_config in estimator_configs:
        print(f"Running estimator: {estimator_config.name}")

        estimator_results = _run_estimator_trials(
            estimator_config, model, x_batch, y_batch, a_batch_expl, device, n_trials, nr_runs, subset_size,
            gpu_batch_size
        )

        if estimator_results:
            results_by_estimator[estimator_config.name] = estimator_results

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
    subset_size: int,
    gpu_batch_size: int
) -> Dict[str, Any]:
    """Run multiple trials for a single estimator."""
    all_results = []

    for trial in range(estimator_config.n_trials):
        # Set reproducible seed for this trial
        original_state = np.random.get_state()
        np.random.seed(42 + trial)

        try:
            estimator = estimator_config.create_estimator()

            # Run estimator with GPU-level batching
            faithfulness_estimate = estimator(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch_expl,
                device=str(device),
                batch_size=gpu_batch_size
            )

            # Process results
            scores = _process_estimator_output(faithfulness_estimate, len(y_batch))
            all_results.append(scores)

        except Exception as e:
            import traceback
            print(f"Error in trial {trial} for {estimator_config.name}: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")

            # Check for CUDA memory issues specifically
            if "CUDA out of memory" in str(e) or "RuntimeError" in str(type(e).__name__):
                print("CUDA memory issue detected. Consider reducing gpu_batch_size or nr_runs.")
                torch.cuda.empty_cache()
                gc.collect()

            all_results.append(np.full(len(y_batch), np.nan))
        finally:
            np.random.set_state(original_state)

    if not all_results:
        return {}

    # Calculate statistics across trials
    all_results = np.stack(all_results, axis=0)

    # Replace any remaining NaN values with 0 for stable statistics
    all_results = np.nan_to_num(all_results, nan=0.0)

    return {
        "mean_scores": np.mean(all_results, axis=0),
        "std_scores": np.std(all_results, axis=0),
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
    """Convert attribution to a flat patch vector of length ``n_patches``."""
    patch_size = 32 if n_patches == 49 else 16
    grid_size = int(np.sqrt(n_patches))

    attr = np.asarray(attribution)

    # Drop redundant channel dimension if present.
    if attr.ndim == 3 and attr.shape[0] == 3:
        attr = attr[0]

    # Downsample spatial attributions to patch level.
    if attr.shape == (224, 224):
        attr = attr.reshape(grid_size, patch_size, grid_size, patch_size).mean(axis=(1, 3))

    if attr.ndim == 2 and attr.shape != (grid_size, grid_size):
        if attr.shape[0] * attr.shape[1] == n_patches:
            attr = attr.reshape(grid_size, grid_size)

    attr = attr.reshape(-1)[:n_patches]

    if attr.shape[0] != n_patches:
        print(f"Warning: Expected {n_patches} features, got {attr.shape[0]}")
        return None

    return attr.astype(np.float32)


def evaluate_faithfulness_for_results(
    config: PipelineConfig,
    model,
    device: torch.device,
    classification_results: List[ClassificationResult]
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluate faithfulness scores with patch-level attributions.
    Uses cached tensors for efficient processing - no outer batching needed.
    Supports both B-16 (196 patches) and B-32 (49 patches) models.

    GPU batch size is configured via config.faithfulness.gpu_batch_size
    """

    # Detect patch configuration
    is_patch32 = False
    if hasattr(config.classify, 'clip_model_name') and config.classify.clip_model_name:
        model_name = config.classify.clip_model_name.lower()
        is_patch32 = "patch32" in model_name or "b-32" in model_name or "b32" in model_name

    n_patches = 49 if is_patch32 else 196
    total_samples = len(classification_results)

    print(f"Processing {total_samples} samples (batching at GPU level only)")

    # Prepare all data at once - fast since we're using cached tensors
    batch_data = _prepare_batch_data(config, classification_results, n_patches)
    if not batch_data:
        print("No valid samples found")
        return {}, np.array([])

    x_batch, y_batch, a_batch = batch_data

    # Calculate faithfulness with GPU-level batching only
    batch_faithfulness = calc_faithfulness(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch_expl=a_batch,
        device=device,
        config=config,
        n_patches=n_patches
    )

    # Build final results
    final_results = {}
    for estimator_name, data in batch_faithfulness.items():
        mean_scores = np.array(data['mean_scores'])
        std_scores = np.array(data['std_scores'])

        if len(mean_scores) == 0:
            continue

        final_results[estimator_name] = {
            'mean_scores': mean_scores.tolist(),
            'std_scores': std_scores.tolist(),
            'n_trials': data.get('n_trials', 3),
            'nr_runs': data.get('nr_runs', 20),
            'subset_size': data.get('subset_size', 98)
        }

    print(f"\nProcessed {len(y_batch)} total samples")

    return final_results, y_batch


def _prepare_batch_data(config: PipelineConfig, batch_results: List[ClassificationResult], n_patches: int = 196):
    """Prepare batch data using cached tensors - no disk I/O or retransforming."""
    x_list, y_list, a_list = [], [], []

    for result in batch_results:
        try:
            # Use cached tensor if available (already preprocessed)
            if result._cached_tensor is not None:
                img_array = result._cached_tensor
            else:
                # Fallback: load and transform (for cached results without tensors)
                from dataset_config import get_dataset_config
                ds_cfg = get_dataset_config(config.file.dataset_name)
                transform = ds_cfg.get_transforms('test')
                img = Image.open(result.image_path).convert('RGB')
                img_array = transform(img).cpu().numpy()

            # Use cached attribution if available
            if result._cached_raw_attribution is not None:
                attr_map = result._cached_raw_attribution
            else:
                # Fallback: load from disk
                attr_map = np.load(result.attribution_paths.raw_attribution_path)

            # Normalize attribution to patch format
            attr_map = _normalize_attribution_format(attr_map, n_patches)
            if attr_map is None:
                continue
            converted_attr = convert_patch_attribution_to_image(attr_map, n_patches)
            if converted_attr is None:
                continue

            x_list.append(img_array)
            y_list.append(result.prediction.predicted_class_idx)
            a_list.append(converted_attr)
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


def _compute_statistics_from_scores(scores: np.ndarray,
                                    stds: np.ndarray,
                                    class_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Compute statistics from scores array.

    Returns dict with 'overall' stats and optionally 'by_class' stats.
    """
    result = {
        'overall': {
            'count': len(scores),
            'mean': float(np.nanmean(scores)),
            'median': float(np.nanmedian(scores)),
            'min': float(np.nanmin(scores)) if len(scores) > 0 else 0,
            'max': float(np.nanmax(scores)) if len(scores) > 0 else 0,
            'std': float(np.nanstd(scores)),
            'avg_trial_std': float(np.nanmean(stds))
        }
    }

    # Compute per-class statistics if labels provided
    if class_labels is not None:
        result['by_class'] = {}
        for class_idx in np.unique(class_labels):
            mask = class_labels == class_idx
            class_scores = scores[mask]
            class_stds = stds[mask]

            if len(class_scores) > 0:
                result['by_class'][int(class_idx)] = {
                    'count': len(class_scores),
                    'mean': float(np.mean(class_scores)),
                    'median': float(np.median(class_scores)),
                    'min': float(np.min(class_scores)),
                    'max': float(np.max(class_scores)),
                    'std': float(np.std(class_scores)),
                    'avg_trial_std': float(np.mean(class_stds))
                }

    return result


def handle_array_values(arr):
    """Convert numpy arrays to JSON-serializable lists recursively."""
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    if isinstance(arr, list):
        return [handle_array_values(item) for item in arr]
    return arr


def create_patch_mask(patch_indices, image_shape, n_patches, patch_size):
    """
    Create binary mask for specified patches.

    Args:
        patch_indices: Indices of patches to mask
        image_shape: Shape of the image (C, H, W)
        n_patches: Total number of patches
        patch_size: Size of each patch in pixels

    Returns:
        Binary mask with True for patches to perturb
    """
    C, H, W = image_shape
    grid_size = int(np.sqrt(n_patches))

    # Initialize mask as False (don't perturb)
    mask = np.zeros(image_shape, dtype=bool)

    for patch_idx in patch_indices:
        if patch_idx >= n_patches:
            continue

        # Convert patch index to grid coordinates
        row = patch_idx // grid_size
        col = patch_idx % grid_size

        # Calculate pixel boundaries for this patch
        start_row = row * patch_size
        end_row = min(start_row + patch_size, H)
        start_col = col * patch_size
        end_col = min(start_col + patch_size, W)

        # Set mask to True for all channels in this patch
        mask[:, start_row:end_row, start_col:end_col] = True

    return mask


def apply_baseline_perturbation(image, mask, perturb_baseline):
    """
    Apply baseline perturbation to image using mask.

    Args:
        image: Image array to perturb
        mask: Binary mask indicating where to perturb
        perturb_baseline: Type of baseline ("black", "white", "mean", "uniform", or numeric)

    Returns:
        Perturbed image
    """
    arr = image.copy()

    # Get baseline value
    if perturb_baseline == "black":
        baseline_value = 0.0
    elif perturb_baseline == "white":
        baseline_value = arr.max()
    elif perturb_baseline == "mean":
        baseline_value = arr.mean()
    elif perturb_baseline == "uniform":
        baseline_value = np.random.uniform(0.0, 1.0, size=arr.shape)
    elif isinstance(perturb_baseline, (int, float)):
        baseline_value = float(perturb_baseline)
    else:
        baseline_value = 0.0  # Default fallback

    # Apply perturbation using np.where
    return np.where(mask, baseline_value, arr)


def evaluate_and_report_faithfulness(
    config: PipelineConfig,
    model,
    device: torch.device,
    classification_results: List[ClassificationResult],
    clip_classifier=None  # Kept for backward compatibility but unused
) -> Dict[str, Any]:
    """
    Evaluate faithfulness and report statistics.

    Args:
        config: Pipeline configuration
        model: PyTorch model (HookedViT, CLIP, or CLIPModelWrapper)
        device: Device to run calculations on
        classification_results: List of classification results

    Returns:
        Dictionary with overall and per-class statistics
    """
    # Evaluate faithfulness
    faithfulness_results, class_labels = evaluate_faithfulness_for_results(
        config, model, device, classification_results
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

    for estimator_name, estimator_results in faithfulness_results.items():
        if "mean_scores" not in estimator_results:
            continue

        scores = np.array(estimator_results["mean_scores"])
        stds = np.array(estimator_results.get("std_scores", np.zeros_like(scores)))

        # Compute all statistics at once
        stats = _compute_statistics_from_scores(scores, stds, class_labels)

        # Add method parameters
        stats['overall']['method_params'] = {
            'n_trials': estimator_results.get("n_trials", 3),
            'nr_runs': estimator_results.get("nr_runs", 50),
            'subset_size': estimator_results.get("subset_size", 98)
        }

        # Store results
        results['metrics'][estimator_name] = {
            **stats, 'mean_scores': handle_array_values(scores),
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


class PatchPixelFlipping:
    """
    Standalone patch-based pixel flipping implementation following Bach et al. (2015).

    Implements the pixel flipping experiment adapted for patch-level attributions:
    - Progressively perturbs the most important patches based on attribution scores
    - Measures prediction degradation as patches are removed
    - Returns AUC scores measuring explanation faithfulness

    This is a complete standalone implementation.
    """

    def __init__(self, n_patches=196, patch_size=16, features_in_step=1, perturb_baseline="mean"):
        """
        Initialize patch-based pixel flipping.

        Args:
            n_patches: Number of patches (196 for ViT-B/16, 49 for ViT-B/32)
            patch_size: Size of each patch in pixels (16 or 32)
            features_in_step: Number of patches to perturb at each step
            perturb_baseline: Baseline for perturbation ("black", "white", "mean", etc.)
        """
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.features_in_step = features_in_step
        self.perturb_baseline = perturb_baseline

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        device: str = None,
        batch_size: int = 256,
        **kwargs
    ):
        """Main evaluation method - standalone implementation."""
        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)
        a_batch = np.asarray(a_batch)

        scores = []
        for start in range(0, len(x_batch), batch_size):
            end = min(start + batch_size, len(x_batch))
            scores.extend(
                self.evaluate_batch(
                    model=model,
                    x_batch=x_batch[start:end],
                    y_batch=y_batch[start:end],
                    a_batch=a_batch[start:end],
                    device=device,
                    **kwargs
                )
            )

        return scores

    def evaluate_batch(
        self, model, x_batch: np.ndarray, y_batch: np.ndarray, a_batch: np.ndarray, device=None, **kwargs
    ):
        """
        Standalone implementation of patch-based pixel flipping following Bach et al. (2015).

        Progressively perturbs the most important patches and measures prediction degradation.
        This is a complete standalone implementation that works directly with PyTorch models.

        Args:
            model: PyTorch model (HookedViT, CLIP, or CLIPModelWrapper)
            x_batch: Input images (N, C, H, W)
            y_batch: True labels (N,)
            a_batch: Patch-level attributions (N, n_patches)
            device: Device string ("cuda" or "cpu")

        Returns:
            List of AUC scores measuring explanation faithfulness
        """
        batch_size = a_batch.shape[0]

        # Validate attribution shape
        if a_batch.shape[1] != self.n_patches:
            raise ValueError(f"Expected {self.n_patches} patches, got {a_batch.shape[1]}")

        # Sort patches by attribution importance (descending order)
        # Most important patches get indices 0, 1, 2, ... (will be perturbed first)
        patch_indices_sorted = np.argsort(-a_batch, axis=1)

        # Calculate number of perturbation steps
        n_perturbations = math.ceil(self.n_patches / self.features_in_step)
        predictions = []
        x_perturbed = x_batch.copy()

        # Get initial predictions on unperturbed input (baseline for the curve)
        # Use softmax for pixel flipping to get probability curves
        y_pred_initial = _predict_torch_model(model, x_batch, y_batch, device, use_softmax=True)
        predictions.append(y_pred_initial)

        # Progressive perturbation following Bach et al. methodology
        for step in range(n_perturbations):
            # Determine which patches to perturb in this step
            start_idx = step * self.features_in_step
            end_idx = min(start_idx + self.features_in_step, self.n_patches)

            # Get patch indices to perturb for each sample
            patches_to_perturb = patch_indices_sorted[:, start_idx:end_idx]

            # Apply perturbation to each sample in the batch
            for batch_idx in range(batch_size):
                # Create mask for patches to perturb
                mask = create_patch_mask(
                    patches_to_perturb[batch_idx], x_batch.shape[1:], self.n_patches, self.patch_size
                )

                # Apply baseline perturbation
                x_perturbed[batch_idx] = apply_baseline_perturbation(
                    x_perturbed[batch_idx], mask, self.perturb_baseline
                )

            # Get model predictions on perturbed input
            # Use softmax for pixel flipping to get probability curves
            y_pred = _predict_torch_model(model, x_perturbed, y_batch, device, use_softmax=True)
            predictions.append(y_pred)

        # Stack predictions: shape (batch_size, n_perturbations)
        predictions_array = np.stack(predictions, axis=1)

        # Calculate AUC for each sample's prediction curve
        # AUC measures faithfulness: larger drop in predictions = more faithful explanations
        # Use trapezoidal rule without normalization
        auc_scores = []
        for i in range(batch_size):
            curve = predictions_array[i]
            # Trapezoidal rule with dx=1
            auc = np.trapezoid(curve, dx=1)
            auc_scores.append(float(auc))

        return auc_scores


def _predict_torch_model(model, x_batch, y_batch, device=None, use_softmax=False):
    """
    Shared helper function for direct PyTorch prediction.

    Args:
        model: PyTorch model (HookedViT, CLIP, or CLIPModelWrapper)
        x_batch: Input images as numpy array (N, C, H, W)
        y_batch: True labels as numpy array (N,)
        device: Device string ("cuda" or "cpu")
        use_softmax: If True, return probabilities; if False, return logits

    Returns:
        Predictions as numpy array (N,) - probabilities or logits for target classes
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_batch).float()
        if device:
            x_tensor = x_tensor.to(device)

        outputs = model(x_tensor)

        # Apply softmax if requested (for pixel flipping probability curves)
        # Otherwise return raw logits (for correlation to avoid saturation)
        if use_softmax and outputs.shape[-1] > 1:
            outputs = torch.softmax(outputs, dim=-1)

        batch_size = len(y_batch)
        preds = outputs[torch.arange(batch_size), y_batch].cpu().numpy()

    return preds


def faithfulness_pixel_flipping(n_patches=196, features_in_step=1, perturb_baseline="mean"):
    """Create mask-based patch pixel flipping."""
    patch_size = 32 if n_patches == 49 else 16
    return PatchPixelFlipping(
        n_patches=n_patches,
        patch_size=patch_size,
        features_in_step=features_in_step,
        perturb_baseline=perturb_baseline
    )


class FaithfulnessCorrelation:
    """
    Standalone patch-based faithfulness correlation implementation.

    Measures correlation between attribution scores and prediction changes when
    random subsets of patches are perturbed. Higher correlation = more faithful explanations.

    Complete standalone implementation that works directly with PyTorch models.
    """

    def __init__(self, n_patches=196, patch_size=16, subset_size=20, nr_runs=50, perturb_baseline="mean"):
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.subset_size = min(subset_size, n_patches)
        self.nr_runs = nr_runs
        self.perturb_baseline = perturb_baseline

    def _compute_spearman_correlation(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute Spearman correlation for batched data.

        Args:
            a: Array of shape (batch_size, nr_runs) - attribution sums
            b: Array of shape (batch_size, nr_runs) - prediction deltas

        Returns:
            Array of shape (batch_size,) with correlation coefficients
        """
        from scipy.stats import spearmanr

        batch_size = a.shape[0]
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
        assert len(a.shape) == 2, "Arrays must be 2D (batch_size, nr_runs)"

        # Check for no variance cases and handle them before calling spearmanr
        correlations = np.zeros(batch_size)

        for i in range(batch_size):
            a_row = a[i]
            b_row = b[i]

            # Check if either array has no variance (all values identical)
            if np.std(a_row) < 1e-10 or np.std(b_row) < 1e-10:
                # No variance means correlation is undefined - use 0 instead of NaN
                correlations[i] = 0.0
            else:
                # Compute Spearman correlation for this sample
                corr, _ = spearmanr(a_row, b_row)
                correlations[i] = corr if not np.isnan(corr) else 0.0

        return correlations

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        device: str = None,
        batch_size: int = 256,
        **kwargs
    ):
        """Main evaluation method - standalone implementation."""
        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)
        a_batch = np.asarray(a_batch)

        scores = []
        for start in range(0, len(x_batch), batch_size):
            end = min(start + batch_size, len(x_batch))
            scores.extend(
                self.evaluate_batch(
                    model=model,
                    x_batch=x_batch[start:end],
                    y_batch=y_batch[start:end],
                    a_batch=a_batch[start:end],
                    device=device
                )
            )

        return scores

    def evaluate_batch(self, model, x_batch: np.ndarray, y_batch: np.ndarray, a_batch: np.ndarray, device=None):
        """
        Vectorized implementation of faithfulness correlation - no excessive copying.

        Measures correlation between attribution scores and prediction changes when
        random patch subsets are perturbed. Optimized to minimize memory copies.

        Args:
            model: PyTorch model (HookedViT, CLIP, or CLIPModelWrapper)
            x_batch: Input images (N, C, H, W)
            y_batch: True labels (N,)
            a_batch: Patch-level attributions (N, n_patches)
            device: Device string ("cuda" or "cpu")

        Returns:
            List of correlation scores measuring explanation faithfulness
        """
        batch_size = a_batch.shape[0]

        # Validate attribution shape
        if a_batch.shape[1] != self.n_patches:
            raise ValueError(f"Expected {self.n_patches} patches, got {a_batch.shape[1]}")

        # Get original predictions on unperturbed images
        # Use logits (not softmax) to avoid saturation at 0/1
        y_pred = _predict_torch_model(model, x_batch, y_batch, device, use_softmax=False)

        # Pre-generate all random patch choices for all runs at once
        all_patch_choices = np.stack([
            np.random.choice(self.n_patches, self.subset_size, replace=False) for _ in range(batch_size * self.nr_runs)
        ]).reshape(self.nr_runs, batch_size, self.subset_size)

        # Pre-compute attribution sums for all runs
        att_sums = np.array([
            a_batch[np.arange(batch_size)[:, None], all_patch_choices[run_idx]].sum(axis=1)
            for run_idx in range(self.nr_runs)
        ]).T  # (batch_size, nr_runs)

        # Process perturbations and predictions in batches
        pred_deltas = []

        # Process multiple runs at once to reduce forward pass overhead
        for run_idx in range(self.nr_runs):
            patch_choices = all_patch_choices[run_idx]

            # Create perturbed batch - single copy per run
            x_perturbed = x_batch.copy()

            # Vectorized mask creation and perturbation
            for batch_idx in range(batch_size):
                mask = create_patch_mask(patch_choices[batch_idx], x_batch.shape[1:], self.n_patches, self.patch_size)
                x_perturbed[batch_idx] = apply_baseline_perturbation(
                    x_perturbed[batch_idx], mask, self.perturb_baseline
                )

            # Get predictions on perturbed images
            y_pred_perturb = _predict_torch_model(model, x_perturbed, y_batch, device, use_softmax=False)

            # Store prediction deltas
            pred_deltas.append(y_pred - y_pred_perturb)

        # Stack results into arrays
        pred_deltas = np.stack(pred_deltas, axis=1)  # (batch_size, nr_runs)

        # Compute Spearman correlation between attribution sums and prediction deltas
        similarity = self._compute_spearman_correlation(a=att_sums, b=pred_deltas)

        return similarity.tolist()


def faithfulness_correlation(subset_size, nr_runs, n_patches, perturb_baseline="mean"):
    """Create mask-based faithfulness correlation."""
    patch_size = 32 if n_patches == 49 else 16
    return FaithfulnessCorrelation(
        n_patches=n_patches,
        patch_size=patch_size,
        subset_size=subset_size,
        nr_runs=nr_runs,
        perturb_baseline=perturb_baseline
    )
