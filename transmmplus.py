import glob
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd

import analysis
import pipeline as pipe
import transformer as trans


def classify_selected(image_paths: List[Path],
                      gini_params: Tuple[float, float,
                                         float] = (0.65, 8.0, 0.5),
                      use_cached: bool = False):
    output_dir = Path('./results_transmmplus')
    cache_dir = Path('./results_transmmplus/cache')
    attribution_dir = output_dir / f"attributions_mean"
    vit_inputs_dir = output_dir / "vit_inputs"

    dirs = {
        "output": output_dir,
        "cache": cache_dir,
        "attribution": attribution_dir,
        "vit_inputs": vit_inputs_dir
    }
    pipe.ensure_directories(list(dirs.values()))

    vit = trans.ViT(method="transmm")
    results = []

    for image_path in image_paths:
        result = pipe.classify_explain_single_image(image_path,
                                                    vit,
                                                    dirs,
                                                    output_suffix='_mean',
                                                    pretransform=True,
                                                    gini_params=gini_params,
                                                    use_cached=use_cached)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / f"classification_results_mean.csv", index=False)

    return df


def subset_images(n: int = 100):
    """
    Select n images from each class with lowest saco scores, based on image filename 
    prefixes (Normal, covid, non_COVID). Returns a list of Path objects for the selected images.
    """
    # Read the CSV file containing image paths and saco scores
    df = pd.read_csv('./results/saco_scores.csv')

    # Exclude Normal class for now due to very high SaCo scores
    prefixes = ["images/non_COVID"]
    result = []

    for prefix in prefixes:
        class_df = df[df['image_name'].str.startswith(prefix)]
        top_n = class_df.nsmallest(n, 'saco_score')['image_name']
        result.extend([Path(img) for img in top_n])

    return result


def calculate_saco_median(csv_path, n: int = 100, head: int = 100):
    df = pd.read_csv(csv_path)
    median_score = df.nsmallest(
        n, 'saco_score').head(head)['saco_score'].median()
    print(median_score)
    avg_score = df.nsmallest(n, 'saco_score').head(head)['saco_score'].mean()
    print(avg_score)
    return median_score


def grid_search_gini_params(image_subset, method: str = "gini"):
    """
    Perform grid search over gini-based normalization parameters and 
    save results with unique filenames.
    """
    # Define parameter ranges
    gini_thresholds = [0.7]
    steepness_values = [0.5]
    max_power_values = [0.7]

    # Get perturbed results (common to all runs)
    perturbed_df = pd.read_csv(
        './results/classification_results_perturbed.csv')

    # Track results for summary
    results = []

    for gini_threshold in gini_thresholds:
        for steepness in steepness_values:
            for max_power in max_power_values:
                # Create parameter string for filenames
                param_str = f"{method}{gini_threshold}_steep{steepness}_power{max_power}"
                print(f"\nRunning with parameters: {param_str}")

                # Run classification with current parameters
                classified_df = classify_selected(image_subset,
                                                  gini_params=(gini_threshold,
                                                               steepness,
                                                               max_power),
                                                  use_cached=False)

                # Run comparison analysis
                analysis.compare_attributions(
                    classified_df,
                    perturbed_df,
                    output_dir='./results_transmmplus/')

                # Calculate SaCo scores
                saco_scores, *_ = analysis.calculate_saco_with_details(
                    "./results_transmmplus/patch_attribution_comparisons.csv")

                # Create and save SaCo scores dataframe with parameter info
                saco_df = pd.DataFrame({
                    'image_name': list(saco_scores.keys()),
                    'saco_score': list(saco_scores.values()),
                    'gini_threshold': gini_threshold,
                    'steepness': steepness,
                    'max_power': max_power
                })

                # Save with unique filename
                saco_df.to_csv(
                    f"./results_transmmplus/saco_scores_{param_str}.csv",
                    index=False)

                # Calculate statistics for summary
                median_score = saco_df['saco_score'].median()
                mean_score = saco_df['saco_score'].mean()

                print(
                    f"Completed {param_str}: Median SaCo = {median_score:.4f}, Mean SaCo = {mean_score:.4f}"
                )

                # Track results
                results.append({
                    'gini_threshold': gini_threshold,
                    'steepness': steepness,
                    'max_power': max_power,
                    'params': param_str,
                    'median_saco': median_score,
                    'mean_saco': mean_score
                })

    # Save summary results
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values('median_saco', ascending=False)
    summary_df.to_csv('./results_transmmplus/gridsearch_summary.csv',
                      index=False)

    print("\nTop 5 parameter combinations:")
    print(summary_df.head(5)[['params', 'median_saco', 'mean_saco']])

    return summary_df


def analyze_saco_scores(n, head):
    file_pattern = os.path.join("./results_transmmplus/", "saco_scores_*")
    saco_files = glob.glob(file_pattern)

    # Process each file
    for file_path in saco_files:
        file_name = os.path.basename(file_path)
        print(f"Processed {file_name}:")
        calculate_saco_median(file_path, n, head)

    file_path = os.path.join("./results/", "saco_scores.csv")
    print(f"Processed original saco_scores:")
    calculate_saco_median(file_path, n, head)


subset = subset_images(n=200)
summary_df = grid_search_gini_params(subset, method='gini_noncov')

#analyze_saco_scores(400, 400)
