import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def compare_faithfulness_metrics(weighted_results_path: str, unweighted_results_path: str,
                                 output_path: str) -> Dict[str, Any]:
    """
    Compare faithfulness metrics between weighted and unweighted schemes.
    
    Args:
        weighted_results_path: Path to the weighted results JSON file
        unweighted_results_path: Path to the unweighted results JSON file
        output_path: Path to save the comparison results
        
    Returns:
        Dictionary with comparison results
    """
    # Load the result files
    with open(weighted_results_path, 'r') as f:
        weighted_results = json.load(f)

    with open(unweighted_results_path, 'r') as f:
        unweighted_results = json.load(f)

    # Extract metrics from both results
    weighted_metrics = weighted_results.get('metrics', {})
    unweighted_metrics = unweighted_results.get('metrics', {})

    # Find common estimators
    common_estimators = set(weighted_metrics.keys()).intersection(set(unweighted_metrics.keys()))

    if not common_estimators:
        raise ValueError("No common estimators found between weighted and unweighted results")

    # Prepare comparison results
    comparison_results = {"overall_comparison": {}, "per_estimator_comparison": {}, "statistical_tests": {}}

    # Perform comparison for each estimator
    for estimator_name in common_estimators:
        print(f"Comparing {estimator_name}...")

        weighted_estimator = weighted_metrics[estimator_name]
        unweighted_estimator = unweighted_metrics[estimator_name]

        # Compare overall metrics
        overall_comparison = compare_metric_sections(
            weighted_estimator.get('overall', {}), unweighted_estimator.get('overall', {}), section_name='overall'
        )

        # Compare per-class metrics
        weighted_by_class = weighted_estimator.get('by_class', {})
        unweighted_by_class = unweighted_estimator.get('by_class', {})

        # Find common classes
        common_classes = set(weighted_by_class.keys()).intersection(set(unweighted_by_class.keys()))

        class_comparisons = {}
        for class_id in common_classes:
            class_comparison = compare_metric_sections(
                weighted_by_class.get(class_id, {}),
                unweighted_by_class.get(class_id, {}),
                section_name=f'class_{class_id}'
            )
            class_comparisons[class_id] = class_comparison

        # Compare per-sample scores (for statistical significance)
        statistical_test = compare_sample_scores_by_class(
            weighted_estimator.get('mean_scores', []), unweighted_estimator.get('mean_scores', []),
            weighted_results.get('class_labels', None)
        )

        # Store results for this estimator
        comparison_results["per_estimator_comparison"][estimator_name] = {
            "overall": overall_comparison,
            "by_class": class_comparisons
        }

        comparison_results["statistical_tests"][estimator_name] = statistical_test

    # Create overall summary
    comparison_results["overall_comparison"] = {
        estimator_name: {
            "improvement_mean":
            comparison_results["per_estimator_comparison"][estimator_name]["overall"].get("mean", {}
                                                                                          ).get("improvement_percent"),
            "improvement_median":
            comparison_results["per_estimator_comparison"][estimator_name]["overall"].get("median", {}
                                                                                          ).get("improvement_percent"),
            "p_value":
            comparison_results["statistical_tests"][estimator_name].get("p_value")
        }
        for estimator_name in common_estimators
    }

    # Save comparison results
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    print(f"Comparison results saved to {output_path}")

    return comparison_results


def compare_metric_sections(weighted_section: Dict[str, Any], unweighted_section: Dict[str, Any],
                            section_name: str) -> Dict[str, Any]:
    """
    Compare two metric sections (overall or by-class).
    
    Args:
        weighted_section: Dictionary with weighted metrics
        unweighted_section: Dictionary with unweighted metrics
        section_name: Name of the section being compared
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}

    # Compare common metrics
    metrics_to_compare = ['mean', 'median', 'min', 'max', 'std']

    for metric in metrics_to_compare:
        if metric in weighted_section and metric in unweighted_section:
            w_value = weighted_section[metric]
            uw_value = unweighted_section[metric]

            # Calculate absolute and relative differences
            absolute_diff = w_value - uw_value

            # Avoid division by zero
            if uw_value != 0:
                improvement_percent = (w_value / uw_value - 1) * 100
            else:
                if w_value > 0:
                    improvement_percent = float('inf')
                elif w_value < 0:
                    improvement_percent = float('-inf')
                else:
                    improvement_percent = 0

            comparison[metric] = {
                "weighted": w_value,
                "unweighted": uw_value,
                "absolute_diff": absolute_diff,
                "improvement_percent": improvement_percent
            }

    return comparison


def compare_sample_scores(
    weighted_scores: List[float],
    unweighted_scores: List[float],
    class_labels: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Perform statistical tests on sample scores.
    
    Args:
        weighted_scores: List of scores from weighted scheme
        unweighted_scores: List of scores from unweighted scheme
        
    Returns:
        Dictionary with statistical test results
    """
    # Convert to numpy arrays if they aren't already
    w_scores = np.array(weighted_scores)
    uw_scores = np.array(unweighted_scores)

    # Make sure the arrays have the same length
    min_length = min(len(w_scores), len(uw_scores))
    w_scores = w_scores[:min_length]
    uw_scores = uw_scores[:min_length]

    if class_labels is not None:
        class_labels = np.array(class_labels)[:min_length]
        valid_classes = class_labels != 0
        w_scores = w_scores[valid_classes]
        uw_scores = uw_scores[valid_classes]

    # Remove samples with NaN values in either dataset
    valid_indices = ~(np.isnan(w_scores) | np.isnan(uw_scores))
    w_scores = w_scores[valid_indices]
    uw_scores = uw_scores[valid_indices]

    if len(w_scores) == 0:
        return {
            "sample_count": 0,
            "valid_sample_count": 0,
            "mean_difference": float('nan'),
            "median_difference": float('nan'),
            "p_value": float('nan'),
            "is_significant": False,
            "improved_samples_count": 0,
            "improved_samples_percent": 0
        }

    # Calculate differences (weighted - unweighted)
    differences = w_scores - uw_scores

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(w_scores, uw_scores)

    # Count improved samples
    improved_samples = np.sum(differences > 0)

    return {
        "sample_count": min_length,
        "valid_sample_count": len(w_scores),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "p_value": float(p_value),
        "is_significant": float(p_value) < 0.05,
        "t_statistic": float(t_stat),
        "improved_samples_count": int(improved_samples),
        "improved_samples_percent": float(improved_samples / len(w_scores) * 100)
    }


def compare_sample_scores_by_class(weighted_scores, unweighted_scores, class_labels):
    """Run separate statistical tests for each class."""
    # Make sure everything is a numpy array
    weighted_scores = np.array(weighted_scores)
    unweighted_scores = np.array(unweighted_scores)
    class_labels = np.array(class_labels)

    # Run overall test first
    results = {"overall": compare_sample_scores(weighted_scores, unweighted_scores)}

    # Get unique classes
    unique_classes = np.unique(class_labels)

    # For each class, run a separate test
    for cls in unique_classes:
        # Create mask and explicitly convert to numpy boolean array
        cls_mask = (class_labels == cls)

        # Debug information
        print(f"Class {cls}: {np.sum(cls_mask)} samples")

        # Make sure we have valid indices
        if np.sum(cls_mask) > 0:
            # Convert indices to positions where the condition is True
            indices = np.where(cls_mask)[0]

            # Use these indices for slicing
            cls_weighted = [weighted_scores[i] for i in indices]
            cls_unweighted = [unweighted_scores[i] for i in indices]

            results[f"class_{cls}"] = compare_sample_scores(cls_weighted, cls_unweighted)

    return results


def print_comparison_summary(comparison_results: Dict[str, Any]):
    """
    Print a human-readable summary of the comparison results.
    
    Args:
        comparison_results: Dictionary with comparison results
    """
    print("\n=== Faithfulness Metrics Comparison Summary ===\n")

    for estimator_name, tests in comparison_results["statistical_tests"].items():
        print(f"Estimator: {estimator_name}")

        # Print overall results
        overall_stats = tests.get("overall", {})
        print(
            f"  Samples: {overall_stats.get('valid_sample_count', 0)} (out of {overall_stats.get('sample_count', 0)})"
        )
        print(f"  Mean difference: {overall_stats.get('mean_difference', 0):.4f}")
        print(
            f"  Improved samples: {overall_stats.get('improved_samples_count', 0)} ({overall_stats.get('improved_samples_percent', 0):.1f}%)"
        )

        if overall_stats.get('is_significant', False):
            print(f"  Significant improvement (p={overall_stats.get('p_value', 1):.6f})")
        else:
            print(f"  No significant difference (p={overall_stats.get('p_value', 1):.6f})")

        # Get overall metrics
        overall = comparison_results["per_estimator_comparison"][estimator_name]["overall"]
        if "mean" in overall:
            w_mean = overall["mean"]["weighted"]
            uw_mean = overall["mean"]["unweighted"]
            diff_percent = overall["mean"]["improvement_percent"]
            print(f"  Mean: {w_mean:.4f} vs {uw_mean:.4f} ({diff_percent:+.1f}%)")

        if "median" in overall:
            w_median = overall["median"]["weighted"]
            uw_median = overall["median"]["unweighted"]
            diff_percent = overall["median"]["improvement_percent"]
            print(f"  Median: {w_median:.4f} vs {uw_median:.4f} ({diff_percent:+.1f}%)")

        # Print class-specific statistical results
        for key, stats in tests.items():
            if key != "overall" and key.startswith("class_"):
                class_id = key.replace("class_", "")
                print(f"\n  Class {class_id} Statistical Test:")
                print(f"    Samples: {stats.get('valid_sample_count', 0)} (out of {stats.get('sample_count', 0)})")
                print(f"    Mean difference: {stats.get('mean_difference', 0):.4f}")

                if stats.get('is_significant', False):
                    print(f"    Significant improvement (p={stats.get('p_value', 1):.6f})")
                else:
                    print(f"    No significant difference (p={stats.get('p_value', 1):.6f})")

        print("")

    print("=== Per-Class Comparison Summary ===\n")

    for estimator_name, estimator_data in comparison_results["per_estimator_comparison"].items():
        if "by_class" in estimator_data:
            print(f"Estimator: {estimator_name}")

            for class_id, class_data in estimator_data["by_class"].items():
                print(f"  Class {class_id}:")

                if "mean" in class_data:
                    w_mean = class_data["mean"]["weighted"]
                    uw_mean = class_data["mean"]["unweighted"]
                    diff_percent = class_data["mean"]["improvement_percent"]
                    print(f"    Mean: {w_mean:.4f} vs {uw_mean:.4f} ({diff_percent:+.1f}%)")

                if "median" in class_data:
                    w_median = class_data["median"]["weighted"]
                    uw_median = class_data["median"]["unweighted"]
                    diff_percent = class_data["median"]["improvement_percent"]
                    print(f"    Median: {w_median:.4f} vs {uw_median:.4f} ({diff_percent:+.1f}%)")

            print("")


comparison = compare_faithfulness_metrics(
    "./results/val_weighted/faithfulness_stats_2025-06-08_20-09.json",
    "./results/val/faithfulness_stats_2025-06-05_15-59.json", "results/val_weighted/faithfulness_comparison.json"
)
print_comparison_summary(comparison)
