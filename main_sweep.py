"""
Feature Gradient Gating Sweep Experiments
Compares vanilla TransLRP vs feature gradient gating across different configurations
"""

import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import config
from pipeline import run_unified_pipeline


def run_single_experiment(
    dataset_name: str,
    source_path: Path,
    experiment_params: Dict[str, Any],
    output_dir: Path,
    subset_size: Optional[int] = None,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Run a single experiment with specified parameters.
    
    Args:
        dataset_name: Name of the dataset
        source_path: Path to dataset source
        experiment_params: Dictionary containing:
            - use_feature_gradients: bool
            - feature_gradient_layers: List[int] 
            - kappa: float (gating strength parameter)
            - topk_features: int (top-k features per patch)
        output_dir: Where to save results
        subset_size: Number of images to process (None for all)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with results and metadata
    """
    # Setup pipeline config
    pipeline_config = config.PipelineConfig()

    # Basic settings
    pipeline_config.file.use_cached_original = False
    pipeline_config.file.use_cached_perturbed = ""
    pipeline_config.file.current_mode = "val"
    pipeline_config.classify.analysis = True
    pipeline_config.file.set_dataset(dataset_name)
    pipeline_config.file.base_pipeline_dir = output_dir

    # CLIP settings for specific datasets
    if dataset_name in ["waterbirds", "oxford_pets"]:
        pipeline_config.classify.use_clip = True
        pipeline_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"

        if dataset_name == "waterbirds":
            pipeline_config.classify.clip_text_prompts = ["a photo of a terrestrial bird", "a photo of an aquatic bird"]
        elif dataset_name == "oxford_pets":
            pipeline_config.classify.clip_text_prompts = ["a photo of a cat", "a photo of a dog"]

    # Feature gradient gating settings
    pipeline_config.classify.boosting.enable_feature_gradients = experiment_params['use_feature_gradients']
    pipeline_config.classify.boosting.feature_gradient_layers = experiment_params.get('feature_gradient_layers', [])

    # Set kappa and topk_features in boosting config
    pipeline_config.classify.boosting.kappa = experiment_params.get('kappa', 50.0)
    pipeline_config.classify.boosting.top_k_features = experiment_params.get('topk_features', 5)

    # No steering layers since we're not using SAE boosting
    pipeline_config.classify.boosting.steering_layers = []

    # Save experiment config
    config_dict = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'experiment_params': experiment_params,
        'subset_size': subset_size,
        'random_seed': random_seed
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'experiment_config.json', 'w') as f:
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

        # Save results with SaCo data
        config_dict['status'] = 'success'
        config_dict['n_images'] = len(results)
        config_dict['saco_results'] = saco_results  # Add SaCo results back

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Aggressive cleanup after experiment
        import gc
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        return config_dict

    except Exception as e:
        print(f"Error in experiment: {e}")
        config_dict['status'] = 'error'
        config_dict['error'] = str(e)

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Cleanup even on error
        import gc
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return config_dict


def run_parameter_sweep(
    datasets: List[Tuple[str, Path]],
    layer_combinations: List[List[int]],
    kappa_values: List[float],
    topk_values: List[int],
    output_base_dir: Optional[Path] = None,
    subset_size: Optional[int] = None,
    random_seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Run a parameter sweep comparing vanilla TransLRP with feature gradient gating.
    
    Args:
        datasets: List of (dataset_name, source_path) tuples
        layer_combinations: List of layer combinations to test (e.g., [[4], [9], [4,9]])
        kappa_values: List of kappa values to test (gating strength)
        topk_values: List of top-k features per patch to test
        output_base_dir: Base directory for output (auto-generated if None)
        subset_size: Number of images per dataset (None for all)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping dataset names to lists of experiment results
    """
    # Create output directory
    if output_base_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = Path(f"./experiments/feature_gradient_sweep_{timestamp}")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep configuration
    sweep_config = {
        'datasets': [d[0] for d in datasets],
        'layer_combinations': layer_combinations,
        'kappa_values': kappa_values,
        'topk_values': topk_values,
        'subset_size': subset_size,
        'random_seed': random_seed,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_base_dir / 'sweep_config.json', 'w') as f:
        json.dump(sweep_config, f, indent=2)

    all_results = {}

    for dataset_name, source_path in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        # Print initial memory usage
        if torch.cuda.is_available():
            print(
                f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated, "
                f"{torch.cuda.memory_reserved()/1024**2:.1f} MB reserved"
            )

        dataset_results = []

        # First run vanilla TransLRP (baseline)
        print("\nRunning vanilla TransLRP (baseline)...")
        exp_params = {'use_feature_gradients': False, 'feature_gradient_layers': [], 'kappa': 0, 'topk_features': 0}
        
        exp_dir = output_base_dir / dataset_name / "vanilla"
        result = run_single_experiment(
        dataset_name=dataset_name,
        source_path=source_path,
        experiment_params=exp_params,
        output_dir=exp_dir,
        subset_size=subset_size,
        random_seed=random_seed
        )
        
        # Don't keep full results in memory - just save minimal info
        summary = {
        'name': 'vanilla',
        'status': result.get('status'),
        'n_images': result.get('n_images', 0),
        'error': result.get('error') if result.get('status') == 'error' else None
        }
        dataset_results.append(summary)
        
        # Explicitly delete the full result to free memory
        del result
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            print(
            f"GPU Memory after vanilla: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated, "
            f"{torch.cuda.memory_reserved()/1024**2:.1f} MB reserved"
            )

        # Run feature gradient gating experiments
        for layers, kappa, topk in product(layer_combinations, kappa_values, topk_values):
            layers_str = '_'.join(map(str, layers))
            exp_name = f"layers_{layers_str}_kappa_{kappa}_topk_{topk}"
            print(f"\nRunning {exp_name}...")

            exp_params = {
                'use_feature_gradients': True,
                'feature_gradient_layers': layers,
                'kappa': kappa,
                'topk_features': topk
            }

            exp_dir = output_base_dir / dataset_name / exp_name
            result = run_single_experiment(
                dataset_name=dataset_name,
                source_path=source_path,
                experiment_params=exp_params,
                output_dir=exp_dir,
                subset_size=subset_size,
                random_seed=random_seed
            )

            # Don't keep full results in memory - just save minimal info
            summary = {
                'name': exp_name,
                'status': result.get('status'),
                'n_images': result.get('n_images', 0),
                'error': result.get('error') if result.get('status') == 'error' else None
            }
            dataset_results.append(summary)

            # Explicitly delete the full result to free memory
            del result

            # Force garbage collection after each experiment
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # Print memory usage
            if torch.cuda.is_available():
                print(
                    f"GPU Memory after {exp_name}: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated, "
                    f"{torch.cuda.memory_reserved()/1024**2:.1f} MB reserved"
                )

        all_results[dataset_name] = dataset_results

    # Save summary of all results
    with open(output_base_dir / 'sweep_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sweep completed! Results saved to: {output_base_dir}")
    print(f"Total experiments: {sum(len(r) for r in all_results.values())}")

    return all_results


def main():
    """
    Main entry point for running feature gradient gating experiments.
    """
    # Define datasets to test
    datasets = [
        # ("hyperkvasir", Path("./data/hyperkvasir/labeled-images/")),
        # ("waterbirds", Path("./data/waterbirds/waterbird_complete95_forest2water2")),
        ("covidquex", Path("./data/covidquex/data/lung/")),
        # ("oxford_pets", Path("./data/oxford_pets"))
    ]

    # Define parameter grid
    layer_combinations = [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
        [1, 2, 3],  # Multiple early layers
        [4, 5, 6],
        [7, 8, 9],  # Multiple late layers
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ]

    kappa_values = [10.0, 30.0]  # Gating strength
    topk_values = [3, 10]  # Top-k features per patch

    # Run sweep
    results = run_parameter_sweep(
        datasets=datasets,
        layer_combinations=layer_combinations,
        kappa_values=kappa_values,
        topk_values=topk_values,
        subset_size=5,  # Use 100 images for quick testing, set to None for full dataset
        random_seed=42
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)

    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        successful = sum(1 for r in dataset_results if r.get('status') == 'success')
        failed = sum(1 for r in dataset_results if r.get('status') == 'error')
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
