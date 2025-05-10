# main.py
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

import analysis
import config
import io_utils
import pipeline as pipe
import visualization
import vit.model as model


def main(test: bool = True):
    pipeline_config = config.PipelineConfig()
    results_df, perturbed_df = pipe.run_pipeline(
        pipeline_config, source_dir=Path("./COVID-QU-Ex/"))

    print("compare attributions")
    analysis.compare_attributions(results_df,
                                  perturbed_df,
                                  generate_visualizations=False)
    run_saco(output_dir="./results/")


def classify_original_only():
    pipeline_config = config.PipelineConfig()
    # pipeline_config.file.data_dir = Path("./images-test")
    # pipeline_config.file.output_dir = Path("./results-test")
    # pipeline_config.file.cache_dir = Path("./cache")
    # pipeline_config.file.__post_init__()
    pipeline_config.file.use_cached = False
    io_utils.ensure_directories(pipeline_config.directories)

    device = torch.device("cuda")

    results_df = pipe.run_classification(pipeline_config, device)
    perturbed_df = pd.read_csv("results/classification_results_perturbed.csv")
    analysis.compare_attributions(results_df,
                                  perturbed_df,
                                  output_dir="./results/",
                                  generate_visualizations=False)
    run_saco(output_dir="./results")


def run_saco(output_dir: str = "./results",
             comparison_path: Optional[str] = None,
             classification_path: Optional[str] = None,
             method: str = "mean",
             save_results: bool = True) -> Dict[str, Any]:
    """
    Run SaCo (Saliency Correlation) analysis on previously generated attribution comparisons.
    
    Args:
        output_dir: Directory to save and read results
        comparison_path: Path to comparison CSV (default: {output_dir}/patch_attribution_comparisons.csv)
        classification_path: Path to classification results (default: {output_dir}/classification_results.csv)
        method: Perturbation method to filter by
        save_results: Whether to save results to files
        
    Returns:
        Dictionary with analysis DataFrames
    """
    comparison_file = comparison_path or f"{output_dir}/patch_attribution_comparisons.csv"
    classification_file = classification_path or f"{output_dir}/classification_results.csv"

    print("calculate SaCo")
    saco_scores, pair_data = analysis.calculate_saco_with_details(
        data_path=comparison_file, method=method)

    saco_df = pd.DataFrame({
        'image_name': list(saco_scores.keys()),
        'saco_score': list(saco_scores.values())
    })
    output_path = Path(output_dir)
    saco_df.to_csv(output_path / "saco_scores_unchanged.csv", index=False)

    patch_metrics_df = analysis.analyze_patch_metrics(pair_data)

    # Analyze faithfulness vs. correctness
    faithfulness_df = analysis.analyze_faithfulness_vs_correctness(
        saco_scores, classification_results=classification_file)

    vit_model = model.load_vit_model(num_classes=3)

    # Analyze attribution patterns
    patterns_df = analysis.analyze_key_attribution_patterns(
        faithfulness_df, vit_model)

    # Save results if requested
    if save_results:
        output_path = Path(output_dir)
        saco_df.to_csv(output_path / "saco_scores.csv", index=False)
        patch_metrics_df.to_csv(output_path / "patch_analysis_results.csv",
                                index=False)
        faithfulness_df.to_csv(output_path / "faithfulness_correctness.csv",
                               index=False)
        if not patterns_df.empty:
            patterns_df.to_csv(output_path / "attribution_patterns.csv",
                               index=False)

    # Return all dataframes
    return {
        "saco_scores": saco_df,
        "patch_metrics": patch_metrics_df,
        "faithfulness_correctness": faithfulness_df,
        "attribution_patterns": patterns_df
    }


if __name__ == "__main__":
    main()
