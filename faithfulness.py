import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import quantus
import torch

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


def faithfulness_estimation(features):
    return quantus.FaithfulnessEstimate(
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        features_in_step=features,
        perturb_baseline="black",
        normalise=False,
        display_progressbar=True)


def faithfulness_correlation(subset_size, nr_runs):
    return quantus.FaithfulnessCorrelation(
        perturb_func=quantus.functions.perturb_func.
        baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_spearman,
        subset_size=subset_size,
        perturb_baseline="black",
        return_aggregate=False,
        normalise=False,
        nr_runs=nr_runs)


def faithfulness_monotonicity(features):
    return quantus.Monotonicity(
        features_in_step=features,
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        normalise=False,
    )


def faithfulness_monotonicity_correlation():
    return quantus.MonotonicityCorrelation(
        nr_samples=10,
        features_in_step=3136,
        perturb_baseline="uniform",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_spearman,
    )


def faithfulness_pixel_flipping():
    return quantus.PixelFlipping(
        features_in_step=224,
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    )


def faithfulness_road():
    return quantus.ROAD(
        noise=0.01,
        perturb_func=quantus.perturb_func.noisy_linear_imputation,
        percentages=list(range(1, 50, 10)),
        display_progressbar=False,
    )


def faithfulness_sufficiency():
    return quantus.Sufficiency(threshold=0.5,
                               return_aggregate=False,
                               abs=False,
                               normalise=False,
                               distance_func=quantus.similarity_func.cosine,
                               normalise_func=lambda x: x)


def calc_faithfulness(vit_model: model.VisionTransformer,
                      x_batch: np.ndarray,
                      y_batch: np.ndarray,
                      a_batch_expl: np.ndarray,
                      device: torch.device,
                      n_trials: int = 5,
                      nr_runs: int = 200,
                      subset_size: int = 98) -> Dict[str, Any]:
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
    print(
        f'Settings - n_trials: {n_trials}, nr_runs: {nr_runs}, subset_size: {subset_size}'
    )

    # Define all the estimators to use
    estimator_configs = [
        FaithfulnessEstimatorConfig(name="FaithfulnessCorrelation",
                                    n_trials=n_trials,
                                    estimator_fn=faithfulness_correlation,
                                    kwargs={
                                        "subset_size": subset_size,
                                        "nr_runs": nr_runs
                                    }),
        FaithfulnessEstimatorConfig(
            name="Sufficiency",
            n_trials=1,  # Deterministic metric - so one trial enough
            estimator_fn=faithfulness_sufficiency)
    ]
    results_by_estimator = {}

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
                    faithfulness_score = np.array(
                        faithfulness_estimate.get('auc', [0]))
                else:
                    faithfulness_score = np.array(faithfulness_estimate)

                # Handle multi-dimensional results - some estimators return arrays per sample
                # If results are multi-dimensional, compute mean across dimensions
                if len(faithfulness_score.shape
                       ) > 1 and faithfulness_score.shape[0] == len(y_batch):
                    # For 2D arrays where rows are samples, keep as is
                    pass
                elif len(faithfulness_score.shape) > 1:
                    # For arrays with multiple values per sample, take mean
                    faithfulness_score = np.mean(
                        faithfulness_score,
                        axis=tuple(range(1, len(faithfulness_score.shape))))

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
            estimator_results.update({
                "mean_scores": mean_scores,
                "std_scores": std_scores,
                "all_trials": all_results
            })

            results_by_estimator[estimator_config.name] = estimator_results

    return results_by_estimator


def evaluate_faithfulness_for_results(
    config: PipelineConfig, vit_model: model.VisionTransformer,
    device: torch.device, classification_results: List[ClassificationResult]
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluate faithfulness scores for classification results with attributions.
    
    Args:
        config: Pipeline configuration
        vit_model: The ViT model
        device: Device to run calculations on
        classification_results: List of classification results with attribution paths
        
    Returns:
        Tuple of (faithfulness statistics dict, class labels array)
    """
    x_batch_list = []
    y_batch_list = []
    a_batch_list = []

    for result in classification_results:
        # Load the image and preprocess
        img_path = result.image_path
        _, input_tensor = preprocessing.preprocess_image(
            str(img_path), img_size=config.classify.target_size[0])
        x_batch_list.append(input_tensor.cpu().numpy())

        # Get the class prediction
        class_idx = result.prediction.predicted_class_idx
        y_batch_list.append(class_idx)

        # Load the attribution
        attribution_path = result.attribution_paths.attribution_path
        attribution = np.load(attribution_path)
        a_batch_list.append(attribution)

    # Convert lists to numpy arrays
    x_batch = np.stack(x_batch_list)
    y_batch = np.array(y_batch_list)
    a_batch = np.stack(a_batch_list)

    # Calculate faithfulness with the robust method
    # Use a reasonable subset size for ViT (half of the 196 patches)
    faithfulness_results = calc_faithfulness(vit_model,
                                             x_batch,
                                             y_batch,
                                             a_batch,
                                             device,
                                             n_trials=3,
                                             nr_runs=20,
                                             subset_size=224)

    return faithfulness_results, y_batch


def calculate_faithfulness_stats_by_class(
        faithfulness_results: Dict[str, Dict[str, Any]],
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

        for score, std, label in zip(faithfulness_scores, faithfulness_stds,
                                     class_labels):
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
                'avg_trial_std':
                float(np.mean(stds_array)),  # Average std across trials
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
            handle_array_values(item)
            if hasattr(item, 'tolist') or isinstance(item, list) else item
            for item in arr
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
        config, vit_model, device, classification_results)

    # Initialize results structure
    results = {
        'boosting_method': 'head_boosting',
        'boost_factor_per_head': config.classify.adaptive_weighting_per_head,
        'head_boost_factors':
        config.classify.head_boost_factor_per_head_per_class,
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
                        np.mean(s)
                        if hasattr(s, 'shape') and len(s.shape) > 0 else s
                        for s in faithfulness_scores
                    ]))),
            'median':
            float(
                np.nanmedian(
                    np.array([
                        np.mean(s)
                        if hasattr(s, 'shape') and len(s.shape) > 0 else s
                        for s in faithfulness_scores
                    ]))),
            'min':
            float(
                np.nanmin(
                    np.array([
                        np.mean(s)
                        if hasattr(s, 'shape') and len(s.shape) > 0 else s
                        for s in faithfulness_scores
                    ]))),
            'max':
            float(
                np.nanmax(
                    np.array([
                        np.mean(s)
                        if hasattr(s, 'shape') and len(s.shape) > 0 else s
                        for s in faithfulness_scores
                    ]))),
            'std':
            float(
                np.nanstd(
                    np.array([
                        np.mean(s)
                        if hasattr(s, 'shape') and len(s.shape) > 0 else s
                        for s in faithfulness_scores
                    ]))),
            'avg_trial_std':
            float(
                np.nanmean(
                    np.array([
                        np.mean(s)
                        if hasattr(s, 'shape') and len(s.shape) > 0 else s
                        for s in faithfulness_stds
                    ]))),  # Average std across trials
            'method_params': {
                'n_trials': estimator_results["n_trials"],
                'nr_runs': estimator_results["nr_runs"],
                'subset_size': estimator_results["subset_size"]
            }
        }

        # Calculate per-class statistics for this estimator
        class_stats = calculate_faithfulness_stats_by_class(
            {estimator_name: estimator_results}, class_labels)[estimator_name]

        # Store results for this estimator
        results['metrics'][estimator_name] = {
            'overall': overall_stats,
            'by_class': class_stats,
            'mean_scores': handle_array_values(faithfulness_scores),
            'std_scores': handle_array_values(faithfulness_stds)
        }

        # Print summary with added stability information
        print(
            f"\n{estimator_name} faithfulness statistics (across {estimator_results['n_trials']} trials):"
        )
        print(f"  Mean: {overall_stats['mean']:.4f}")
        print(f"  Median: {overall_stats['median']:.4f}")
        print(f"  Count: {overall_stats['count']}")
        print(
            f"  Avg trial std: {overall_stats['avg_trial_std']:.4f} (lower is more stable)"
        )

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
