# main.py
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

import analysis
import config
import io_utils
import pipeline as pipe
import vit.model as model


def main():
    pipeline_config = config.PipelineConfig()
    pipeline_config.file.use_cached_original = False
    pipeline_config.file.current_mode = "val"
    pipeline_config.classify.weigh_by_class_embedding = False
    pipeline_config.classify.data_collection = True

    original_classification, (
        perturbed_image_records, perturbed_classification) = pipe.run_pipeline(
            pipeline_config,
            source_dir_for_preprocessing=Path(
                f"./COVID-QU-Ex/{pipeline_config.file.current_mode}"))

    print("compare attributions")
    run_saco_from_pipeline_outputs(pipeline_config, original_classification,
                                   perturbed_image_records,
                                   perturbed_classification)


def run_saco_from_pipeline_outputs(
        pipeline_config: config.PipelineConfig,
        original_pipeline_results: List[pipe.ClassificationResult],
        all_perturbed_image_records: List[pipe.PerturbedImageRecord],
        perturbed_pipeline_results: List[pipe.ClassificationResult],
        generate_visualizations: bool = False,
        save_analysis_results: bool = True):
    """
    Run SaCo (Saliency Correlation) analysis using direct outputs from the pipeline.
    
    Args:
        pipeline_config: The configuration used for the pipeline run (for paths, params).
        original_pipeline_results: List of ClassificationResult for original images.
        all_perturbed_image_records: List of PerturbedImageRecord for all generated perturbations.
        perturbed_pipeline_results: List of ClassificationResult for perturbed images.
        model_instance: The loaded VisionTransformer model, if needed for pattern analysis.
        generate_visualizations: Whether to generate comparison visualizations.
        save_analysis_results: Whether to save all generated analysis DataFrames.
        
    Returns:
        Dictionary with analysis DataFrames.
    """
    saco_analysis_results: Dict[str, pd.DataFrame] = {}

    print("Building analysis context...")
    analysis_context = analysis.AnalysisContext.build(
        config=pipeline_config,
        original_results=original_pipeline_results,
        all_perturbed_records=all_perturbed_image_records,
        perturbed_classification_results=perturbed_pipeline_results)

    print("Generating perturbation comparison DataFrame for SaCo...")
    perturbation_comparison_df = analysis.generate_perturbation_comparison_dataframe(
        analysis_context, generate_visualizations=generate_visualizations)

    saco_analysis_results[
        "perturbation_comparison"] = perturbation_comparison_df

    print("Running core SaCo calculations...")
    saco_scores_dict, _, patch_analysis_df = analysis.run_saco_analysis(
        context=analysis_context,
        perturbation_comparison_df=perturbation_comparison_df,
        perturb_method_filter=pipeline_config.perturb.method)

    saco_df = pd.DataFrame({
        'image_name': list(saco_scores_dict.keys()),
        'saco_score': list(saco_scores_dict.values())
    })
    saco_analysis_results["saco_scores"] = saco_df
    saco_analysis_results["patch_metrics"] = patch_analysis_df

    saco_scores_map_for_analysis: Dict[str, float] = pd.Series(
        saco_analysis_results["saco_scores"].saco_score.values,
        index=saco_analysis_results["saco_scores"].image_name).to_dict()

    faithfulness_df = analysis.analyze_faithfulness_vs_correctness_from_objects(
        saco_scores_map_for_analysis,  # Pass the dictionary: {str(image_path): score}
        original_pipeline_results  # Pass the List[ClassificationResult]
    )
    saco_analysis_results["faithfulness_correctness"] = faithfulness_df

    print("Analyzing key attribution patterns...")
    model_instance = model.load_vit_model()
    patterns_df = analysis.analyze_key_attribution_patterns(
        saco_analysis_results["faithfulness_correctness"], model_instance)
    saco_analysis_results["attribution_patterns"] = patterns_df

    if save_analysis_results:
        print("Saving analysis results...")
        for name, df_to_save in saco_analysis_results.items():
            if isinstance(df_to_save, pd.DataFrame) and not df_to_save.empty:
                # Use a consistent naming convention, incorporating the mode and output_file_tag
                save_path = pipeline_config.file.output_dir / f"analysis_{name}{pipeline_config.file.output_suffix}.csv"
                df_to_save.to_csv(save_path, index=False)
                print(f"Saved {name} to {save_path}")


if __name__ == "__main__":
    main()
