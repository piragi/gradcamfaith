import pipeline as pipe

def main():
    pipe.preprocess_dataset(source_dir="./originals", dest_dir="./images")
    results_df = pipe.classify()
    perturbed_paths = pipe.perturb_low_attribution_areas(results_df, percentile_threshold=20, strength=0.2)
    perturbed_results_df = pipe.classify("./results/perturbed", "_perturbed")
    pipe.compare_attributions(results_df, perturbed_results_df)


if __name__ == "__main__":
    main()
