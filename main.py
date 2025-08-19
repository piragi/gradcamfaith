# main.py
import json
from datetime import datetime
from pathlib import Path

import config
from pipeline_unified import run_unified_pipeline


def main():
    datasets = [
        ("hyperkvasir", Path("./data/hyperkvasir/labeled-images/")),
        # ("covidquex", Path("./data/covidquex/data/"))
    ]

    # Layers to test
    layers_to_test = [6]

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
            pipeline_config.file.weighted = True
            pipeline_config.classify.analysis = False
            pipeline_config.classify.data_collection = False
            pipeline_config.file.set_dataset(dataset_name)

            # Set boosting parameters - only changing the layer
            pipeline_config.classify.boosting.steering_layers = [layer]

            # Create output dir for this run
            exp_name = f"{dataset_name}_layer{layer}"
            exp_output_dir = output_base_dir / exp_name
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            pipeline_config.file.base_pipeline_dir = exp_output_dir

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
                    force_prepare=False
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
