# main.py
import json
from datetime import datetime
from pathlib import Path

import config
from pipeline_unified import run_unified_pipeline


def main():
    datasets = [
        ("hyperkvasir", Path("./data/hyperkvasir/labeled-images/")),
        # ("covidquex", Path("./data/covidquex/data/lung/")),
        # ("waterbirds", Path("./data/waterbirds/waterbird_complete95_forest2water2")),
        # ("oxford_pets", Path("./data/oxford_pets"))
    ]

    # Layers to test
    layers_to_test = [10]

    # Feature gradient settings (NEW)
    USE_FEATURE_GRADIENTS = False  # Set to False to disable feature gradient gating
    FEATURE_GRADIENT_LAYERS = [4, 6]  # Which layers to apply feature gradients

    # Subset settings
    subset_size = None  # Set to None to use all images
    random_seed = 42  # For reproducibility

    # Output directory for all experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path(f"./experiments/layer_sweep_{timestamp}")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, source_path in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        for layer in layers_to_test:
            print(f"\nRunning with layer {layer}")

            # Setup config
            pipeline_config = config.PipelineConfig()
            pipeline_config.file.use_cached_original = False
            pipeline_config.file.use_cached_perturbed = ""
            pipeline_config.file.current_mode = "val"
            pipeline_config.file.weighted = False  # Disable SAE boosting - only use feature gradients
            pipeline_config.classify.analysis = False
            pipeline_config.classify.data_collection = False
            pipeline_config.file.set_dataset(dataset_name)

            # Enable CLIP for waterbirds and oxford_pets
            if dataset_name in ["waterbirds", "oxford_pets"]:
                pipeline_config.classify.use_clip = True
                # Use OpenCLIP DataComp.XL model to match the paper's setup
                pipeline_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"

                # Set appropriate prompts
                if dataset_name == "waterbirds":
                    pipeline_config.classify.clip_text_prompts = [
                        "a photo of a terrestrial bird", "a photo of an aquatic bird"
                    ]
                elif dataset_name == "oxford_pets":
                    pipeline_config.classify.clip_text_prompts = ["a photo of a cat", "a photo of a dog"]

            # Set boosting parameters - only changing the layer
            pipeline_config.classify.boosting.steering_layers = [layer]

            # Enable feature gradient gating (NEW)
            pipeline_config.classify.boosting.enable_feature_gradients = USE_FEATURE_GRADIENTS
            pipeline_config.classify.boosting.feature_gradient_layers = FEATURE_GRADIENT_LAYERS

            # Create output dir for this run
            exp_name = f"{dataset_name}_layer{layer}"
            exp_output_dir = output_base_dir / exp_name
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            pipeline_config.file.base_pipeline_dir = Path(f"./data/{dataset_name}_unified/results")  # exp_output_dir

            # Save config for this run
            config_dict = {
                'dataset': dataset_name,
                'layer': layer,
                'timestamp': datetime.now().isoformat(),
                'boosting_params': {
                    'boost_strength': pipeline_config.classify.boosting.boost_strength,
                    'max_boost': pipeline_config.classify.boosting.max_boost,
                    'selection_method': pipeline_config.classify.boosting.selection_method,
                    'min_log_ratio': pipeline_config.classify.boosting.min_log_ratio,
                    'steering_layers': pipeline_config.classify.boosting.steering_layers,
                },
                'feature_gradient_params': {
                    'enabled': USE_FEATURE_GRADIENTS,
                    'layers': FEATURE_GRADIENT_LAYERS
                }
            }

            with open(exp_output_dir / 'config.json', 'w') as f:
                json.dump(config_dict, f, indent=2)

            # Run pipeline
            try:
                results, saco_results = run_unified_pipeline(
                    config=pipeline_config,
                    dataset_name=dataset_name,
                    source_data_path=source_path,
                    prepared_data_path=Path(f"./data/{dataset_name}_unified/"),
                    force_prepare=False,
                    subset_size=subset_size,
                    random_seed=random_seed
                )
                print(f"Completed layer {layer} - processed {len(results)} images")

                # Update config with full SaCo results and save
                config_dict['saco_results'] = saco_results
                with open(exp_output_dir / 'results.json', 'w') as f:
                    json.dump(config_dict, f, indent=2)

            except Exception as e:
                print(f"Error with layer {layer}: {e}")

    print(f"\n{'='*60}")
    print(f"All experiments completed. Results saved to: {output_base_dir}")


if __name__ == "__main__":
    main()
