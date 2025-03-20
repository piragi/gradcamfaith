import pipeline as pipe

def main():
    processed_images = pipe.preprocess_dataset(
        source_dir="./chexpert",  # Path to the cheXpert dataset
        dest_dir="./images",      # Destination directory
        target_size=(224, 224),   # Target size for resizing
    )
    results_df = pipe.classify()
    perturbed_paths = pipe.perturb_low_attribution_areas(results_df, percentile_threshold=10, strength=0.2)
    perturbed_results_df = pipe.classify("./results/perturbed", "_perturbed")
    pipe.compare_attributions(results_df, perturbed_results_df)


if __name__ == "__main__":
    main()
