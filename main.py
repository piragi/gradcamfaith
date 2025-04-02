import pipeline as pipe
import torch
import gc
import analysis
import sd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def clear_gpu_memory():
    """Explicitly clear PyTorch CUDA cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    processed_images = pipe.preprocess_dataset(
        source_dir="./chexpert",
        dest_dir="./images",
        target_size=(224, 224),
    )
    
    # Run original classification
    results_df = pipe.classify()
    
    # Load SD model
    sd_pipe = sd.load_model()
    
    perturbed_paths = pipe.perturb_all_patches(
        results_df,
        sd_pipe=sd_pipe,
        patch_size=16,  # Size of each patch
        strength=0.2,   # Perturbation strength
        max_images=100    # Process only one image for testing (remove or set to None for all)
    )
    print(f"Generated {len(perturbed_paths)} perturbed patch images")
    perturbed_results_df = pipe.classify("./results/patches", "_perturbed")
    comparison_df = analysis.compare_attributions(results_df, perturbed_results_df)

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
    for image_name, image_data in analysis_with_coords.groupby('image_name'):
        plt.figure(figsize=(10, 8))
        # Reshape data to create a 2D grid (adjust dimensions based on your patch grid)
        # Assuming 14x14 grid for a typical ViT
        heatmap_data = np.zeros((14, 14))
        for _, row in image_data.iterrows():
            x, y = int(row['x']/16), int(row['y']/16)  # Assuming 16x16 patches
            if 0 <= x < 14 and 0 <= y < 14:
                heatmap_data[y, x] = row['patch_saco']
        
        sns.heatmap(heatmap_data, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        plt.title(f'Patch SaCo Heatmap: {image_name}')
        plt.savefig(f'./results/saco_heatmap_{os.path.basename(image_name)}.png')
        plt.close()

if __name__ == "__main__":
    run_saco()
