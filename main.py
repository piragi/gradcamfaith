"""
Feature Gradient Gating Sweep Experiments
Compares vanilla TransLRP vs feature gradient gating across different configurations
"""

import json
import re
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import config
from dataset_config import get_dataset_config
from pipeline import (load_model_for_dataset, load_steering_resources, run_unified_pipeline)


def run_single_experiment(
    dataset_name: str,
    source_path: Path,
    experiment_params: Dict[str, Any],
    output_dir: Path,
    model: torch.nn.Module,
    steering_resources: Dict[int, Dict[str, Any]],
    clip_classifier: Optional[Any] = None,
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
            - gate_construction: str ("activation_only", "gradient_only", or "combined")
            - shuffle_decoder: bool (whether to shuffle decoder columns)
        output_dir: Where to save results
        model: Pre-loaded model to use
        steering_resources: Pre-loaded SAE resources
        clip_classifier: Pre-loaded CLIP classifier (None for non-CLIP models)
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
    if dataset_name in ["waterbirds", "imagenet"]:
        pipeline_config.classify.use_clip = True
        pipeline_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"

        if dataset_name == "waterbirds":
            pipeline_config.classify.clip_text_prompts = ["a photo of a terrestrial bird", "a photo of an aquatic bird"]
        elif dataset_name == "imagenet":
            # Create prompts for all 1000 ImageNet classes
            dataset_cfg = get_dataset_config(dataset_name)
            pipeline_config.classify.clip_text_prompts = [f"a photo of a {cls}" for cls in dataset_cfg.class_names]

    # Feature gradient gating settings
    pipeline_config.classify.boosting.enable_feature_gradients = experiment_params['use_feature_gradients']
    pipeline_config.classify.boosting.feature_gradient_layers = experiment_params.get('feature_gradient_layers', [])

    # Set kappa and topk_features in boosting config
    pipeline_config.classify.boosting.kappa = experiment_params.get('kappa', 50.0)
    pipeline_config.classify.boosting.top_k_features = experiment_params.get('topk_features', 5)
    pipeline_config.classify.boosting.gate_construction = experiment_params.get('gate_construction', 'combined')
    pipeline_config.classify.boosting.shuffle_decoder = experiment_params.get('shuffle_decoder', False)

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
            random_seed=random_seed,
            model=model,
            steering_resources=steering_resources,
            clip_classifier=clip_classifier
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
    gate_constructions: List[str] = ["combined"],
    shuffle_decoder_options: List[bool] = [False],
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
        gate_constructions: List of gate construction types to test
        shuffle_decoder_options: List of shuffle decoder options (True/False)
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
        'gate_constructions': gate_constructions,
        'shuffle_decoder_options': shuffle_decoder_options,
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

        # Load model and SAE resources once for this dataset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_config = get_dataset_config(dataset_name)
        print(dataset_config)

        # Create temporary pipeline config to get CLIP settings
        temp_config = config.PipelineConfig()
        temp_config.file.set_dataset(dataset_name)
        if dataset_name in ["waterbirds", "imagenet"]:
            temp_config.classify.use_clip = True
            temp_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
            if dataset_name == "waterbirds":
                temp_config.classify.clip_text_prompts = ["a photo of a terrestrial bird", "a photo of an aquatic bird"]
            elif dataset_name == "imagenet":

                def _first_synonym(name: str) -> str:
                    # "tench, Tinca tinca" -> "tench"
                    return name.split(",")[0].strip()

                def _needs_article(s: str) -> bool:
                    return not re.match(r"^(a|an|the)\b", s, flags=re.I)

                def _article(s: str) -> str:
                    return "an" if re.match(r"^[aeiou]", s, flags=re.I) else "a"

                imagenet_cfg = get_dataset_config("imagenet")
                names = imagenet_cfg.class_names  # length 1000

                cleaned = [_first_synonym(n) for n in names]
                temp_config.classify.clip_text_prompts = [
                    f"a photo of {(_article(n) + ' ') if _needs_article(n) else ''}{n}" for n in cleaned
                ]
        print(f"Loading model for {dataset_name}...")
        model, clip_classifier = load_model_for_dataset(dataset_config, device, temp_config)
        model = model.to(device)  # Ensure model is on correct device

        # Determine all layers that might need SAE resources
        all_layers_needed = set()
        for layers in layer_combinations:
            all_layers_needed.update(layers)

        print(f"Loading SAE resources for layers: {sorted(all_layers_needed)}")
        steering_resources = load_steering_resources(list(all_layers_needed), dataset_name=dataset_name)

        dataset_results = []

        # First run vanilla TransLRP (baseline)
        print("\nRunning vanilla TransLRP (baseline)...")
        exp_params = {
            'use_feature_gradients': False,
            'feature_gradient_layers': [],
            'kappa': 0,
            'topk_features': 0,
            'gate_construction': 'combined',
            'shuffle_decoder': False
        }

        exp_dir = output_base_dir / dataset_name / "vanilla"
        result = run_single_experiment(
            dataset_name=dataset_name,
            source_path=source_path,
            experiment_params=exp_params,
            output_dir=exp_dir,
            model=model,
            steering_resources=steering_resources,
            clip_classifier=clip_classifier,
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
        for layers, kappa, topk, gate_construction, shuffle_decoder in product(
            layer_combinations, kappa_values, topk_values, gate_constructions, shuffle_decoder_options
        ):
            layers_str = '_'.join(map(str, layers))
            shuffle_suffix = "_shuffled" if shuffle_decoder else ""
            exp_name = f"layers_{layers_str}_kappa_{kappa}_topk_{topk}_{gate_construction}{shuffle_suffix}"
            print(f"\nRunning {exp_name}...")

            exp_params = {
                'use_feature_gradients': True,
                'feature_gradient_layers': layers,
                'kappa': kappa,
                'topk_features': topk,
                'gate_construction': gate_construction,
                'shuffle_decoder': shuffle_decoder
            }

            exp_dir = output_base_dir / dataset_name / exp_name
            result = run_single_experiment(
                dataset_name=dataset_name,
                source_path=source_path,
                experiment_params=exp_params,
                output_dir=exp_dir,
                model=model,
                steering_resources=steering_resources,
                clip_classifier=clip_classifier,
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

        # Clean up model and SAE resources after dataset
        print(f"Cleaning up model and SAE resources for {dataset_name}...")

        # Clean up model
        if hasattr(model, 'to'):
            model.to("cpu")
        del model

        # Clean up CLIP classifier if it exists
        if clip_classifier is not None:
            if hasattr(clip_classifier, 'text_model') and clip_classifier.text_model is not None:
                if hasattr(clip_classifier.text_model, 'to'):
                    clip_classifier.text_model.to("cpu")
                del clip_classifier.text_model
            del clip_classifier

        # Clean up SAE resources
        for layer_idx, resources in steering_resources.items():
            if 'sae' in resources:
                if hasattr(resources['sae'], 'to'):
                    resources['sae'].to("cpu")
                del resources['sae']
        del steering_resources

        # Also clean up dataset config to free any cached data
        del dataset_config
        del temp_config

        # Force garbage collection multiple times
        import gc
        for _ in range(5):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force another round after sync
            for _ in range(2):
                gc.collect()
            torch.cuda.empty_cache()
            print(
                f"GPU Memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated, "
                f"{torch.cuda.memory_reserved()/1024**2:.1f} MB reserved"
            )

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
        ("hyperkvasir", Path("./data/hyperkvasir/labeled-images/")),
        # ("imagenet", Path("./data/imagenet/raw")),
        # ("waterbirds", Path("./data/waterbirds/waterbird_complete95_forest2water2")),
        # ("covidquex", Path("./data/covidquex/data/lung/")),
    ]

    # Define parameter grid
    layer_combinations = [
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
        #[8, 10]
        # [4, 8]
    ]

    # covidquex
    # [3, 4, 5]

    # hyprkvasir
    # [4, 6]

    # waterbirds
    # [6, 7, 8, 9]

    kappa_values = [0.5]  # Gating strength
    topk_values = [None]  # Top-k features per patch

    # Gate construction types for interaction ablation
    gate_constructions = ["combined", "gradient_only", "activation_only"]

    # Decoder shuffling options for semantic alignment ablation
    shuffle_decoder_options = [False]  # Test both normal and shuffled - WARNING: works only with combined

    # Run sweep
    results = run_parameter_sweep(
        datasets=datasets,
        layer_combinations=layer_combinations,
        kappa_values=kappa_values,
        topk_values=topk_values,
        gate_constructions=gate_constructions,
        shuffle_decoder_options=shuffle_decoder_options,
        subset_size=500,  # Use 100 images for quick testing, set to None for full dataset
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
    # run_best_performers(subset_size=500)
