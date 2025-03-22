import pipeline as pipe
import torch
import gc
import analysis
import sd

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
    
if __name__ == "__main__":
    main()
