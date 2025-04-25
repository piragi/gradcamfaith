# main.py
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

import analysis
import attribution_model.saco_refinement as saco_refinement
import config
import pipeline as pipe


def main():
    pipeline_config = config.PipelineConfig()
    results_df, perturbed_df = pipe.run_pipeline(
        pipeline_config, source_dir=Path("./COVID-QU-Ex/"))

    print("compare attributions")
    analysis.compare_attributions(results_df,
                                  perturbed_df,
                                  generate_visualizations=False)
    run_saco()


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

    patch_metrics_df = analysis.analyze_patch_metrics(pair_data)

    # Analyze faithfulness vs. correctness
    faithfulness_df = analysis.analyze_faithfulness_vs_correctness(
        saco_scores, classification_results=classification_file)

    # Analyze attribution patterns
    patterns_df = analysis.analyze_key_attribution_patterns(faithfulness_df)

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
    run_saco()
