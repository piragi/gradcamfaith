import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import analysis
import pipeline as pipe


def clear_gpu_memory():
    """Explicitly clear PyTorch CUDA cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    pipe.preprocess_dataset(
        source_dir="./chexpert",
        dest_dir="./images",
        target_size=(224, 224),
    )
    
    # Run original classification
    results_df = pipe.classify("./images")
    
    perturbed_paths = pipe.perturb_all_patches(
        results_df,
        sd_pipe=None,
        patch_size=16,  # Size of each patch
        strength=0.2,   # Perturbation strength
        max_images=1, # Process only one image for testing (remove or set to None for all)
        method = "mean"
    )
    print(f"Generated {len(perturbed_paths)} perturbed patch images")
    perturbed_results_df = pipe.classify("./results/patches", "_perturbed")
    analysis.compare_attributions(results_df, perturbed_results_df)

def run_saco():
    saco_scores, pair_data = analysis.calculate_saco_with_details()
    analysis_df = analysis.analyze_patch_metrics(pair_data)
    analysis_df.to_csv("./results/patch_analysis_results.csv", index=False)

    # Classify patches into categories
    analysis_df['faithfulness_category'] = pd.cut(
        analysis_df['patch_saco'], 
        bins=[-1, -0.5, 0, 0.5, 1],
        labels=['Very Unfaithful', 'Unfaithful', 'Faithful', 'Very Faithful']
    )

    # Count patches in each category
    category_counts = analysis_df['faithfulness_category'].value_counts().sort_index()
    print("Patches by faithfulness category:")
    print(category_counts)

    data_df = pd.read_csv("./results/patch_attribution_comparisons.csv")
    coordinates_df = data_df[['patch_id', 'x', 'y']].drop_duplicates()
    analysis_with_coords = analysis_df.merge(coordinates_df, on='patch_id')

    # Create heatmaps of patch SaCo scores for each image

    analysis.analyze_faithfulness_vs_correctness(saco_scores)
if __name__ == "__main__":
    main()
