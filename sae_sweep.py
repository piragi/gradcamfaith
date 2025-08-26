import copy
import gc
import itertools
import json
import logging
import os
import subprocess
from pathlib import Path

import torch
import torchvision
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.sae import VisionSAETrainer
from vit_prisma.sae.config import VisionModelSAERunnerConfig

import wandb
from dataset_config import get_dataset_config
from pipeline_unified import load_model_for_dataset

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

# ============ SWEEP CONFIG ============
SWEEP_CONFIG = {
    'dataset': 'hyperkvasir',  # single dataset for sweep
    'layers': [1, 2, 3, 5, 7, 8, 9, 10],  # which layers to train

    # Hyperparameters to sweep
    'expansion_factors': [64],  # Higher = more features to capture variance
    'k_values': [128],  # TopK active features
    'learning_rates': [5e-4],  # Learning rates

    # Fixed parameters
    'epochs': 3,  # Reduced to 3 epochs
    'batch_size': 4096,
    'num_workers': 10,  # Reduced from 16 to avoid warning
    'wandb_project': 'vit_sae_sweep',
    'log_to_wandb': True,
}


def train_single_config(dataset_name, layer_idx, expansion_factor, k, lr):
    """Train a single SAE configuration."""

    print(f"\n{'='*60}")
    print(f"Training: expansion={expansion_factor}, k={k}, lr={lr}")
    print(f"{'='*60}")

    # Load dataset config
    dataset_config = get_dataset_config(dataset_name)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hooked_model = load_model_for_dataset(dataset_config, device)

    # Set up data paths
    data_path = Path(f"data/{dataset_name}_unified")
    train_path = data_path / "train"
    val_path = data_path / "val"

    # Get dataset-specific transforms
    transform = dataset_config.get_transforms('train')
    val_transform = dataset_config.get_transforms('test')  # No augmentations for validation

    # Load datasets
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform)
    val_dataset = torchvision.datasets.ImageFolder(val_path, val_transform)

    # SAE config
    run_config = VisionModelSAERunnerConfig(
        model_name="vit_base_patch16_224",
        layer_subtype='hook_resid_post',
        cls_token_only=False,
        context_size=197,
        expansion_factor=expansion_factor,
        activation_fn_str="topk",
        activation_fn_kwargs={'k': k},
        lr=lr,
        train_batch_size=SWEEP_CONFIG['batch_size'],
        num_epochs=SWEEP_CONFIG['epochs'],
        num_workers=SWEEP_CONFIG['num_workers'],
        lr_scheduler_name="cosineannealingwarmup",
        lr_warm_up_steps=500,
        n_batches_in_buffer=64,
        initialization_method="encoder_transpose_decoder",
        dataset_name=dataset_name,
        log_to_wandb=SWEEP_CONFIG['log_to_wandb'],
        wandb_project=SWEEP_CONFIG['wandb_project'],
        n_checkpoints=1,
        use_ghost_grads=True,
        dead_feature_threshold=1e-8,
        feature_sampling_window=1000,
        dead_feature_window=5000,
    )

    run_config.hook_point_layer = layer_idx
    run_config.dataset_size = len(train_dataset)

    # Set up save directory
    save_dir = Path(f"data/sae_sweep/{dataset_name}/layer_{layer_idx}/exp{expansion_factor}_k{k}_lr{lr}")
    save_dir.mkdir(parents=True, exist_ok=True)
    run_config.checkpoint_path = str(save_dir)

    # Add sweep info to wandb config
    if SWEEP_CONFIG['log_to_wandb']:
        wandb.init(
            project=SWEEP_CONFIG['wandb_project'],
            config={
                'expansion_factor': expansion_factor,
                'k': k,
                'lr': lr,
                'layer': layer_idx,
                'dataset': dataset_name,
                'epochs': SWEEP_CONFIG['epochs'],
            },
            name=f"exp{expansion_factor}_k{k}_lr{lr}_l{layer_idx}",
            reinit=True
        )

    trainer = VisionSAETrainer(run_config, hooked_model, train_dataset, val_dataset)

    try:
        trained_sae = trainer.run()

        # Extract metrics
        metrics = {
            'expansion_factor': expansion_factor,
            'k': k,
            'lr': lr,
            'layer': layer_idx,
            'explained_variance': wandb.run.summary.get('metrics/explained_variance', 0),
            'mse_loss': wandb.run.summary.get('losses/mse_loss', 0),
            'dead_features': wandb.run.summary.get('sparsity/dead_features', 0),
            'l0': wandb.run.summary.get('metrics/l0', 0),
        }

        # Save metrics
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"SUCCESS: Explained variance = {metrics['explained_variance']:.4f}")

        return metrics

    except Exception as e:
        print(f"ERROR: Training failed - {e}")
        return None

    finally:
        wandb.finish()

        # Cleanup
        if 'trained_sae' in locals():
            del trained_sae
        del trainer
        del hooked_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


def main():
    """Run the hyperparameter sweep across all layers."""

    dataset_name = SWEEP_CONFIG['dataset']
    layers = SWEEP_CONFIG['layers']

    # Generate all combinations of hyperparameters and layers
    hyperparams = list(
        itertools.product(SWEEP_CONFIG['expansion_factors'], SWEEP_CONFIG['k_values'], SWEEP_CONFIG['learning_rates'])
    )

    all_configs = list(itertools.product(layers, hyperparams))

    print(f"Running sweep with {len(all_configs)} total configurations")
    print(f"Dataset: {dataset_name}")
    print(f"Layers: {layers}")
    print(f"Hyperparameter combinations per layer: {len(hyperparams)}")

    all_results = {}  # Store results by layer

    for i, (layer_idx, (exp_factor, k, lr)) in enumerate(all_configs, 1):
        print(f"\n[{i}/{len(all_configs)}] Running Layer {layer_idx} with exp_factor={exp_factor}, k={k}, lr={lr}")

        metrics = train_single_config(dataset_name, layer_idx, exp_factor, k, lr)

        if metrics:
            # Initialize layer results list if needed
            if layer_idx not in all_results:
                all_results[layer_idx] = []

            all_results[layer_idx].append(metrics)

            # Save intermediate results for this layer
            with open(f'sweep_results_{dataset_name}_layer{layer_idx}.json', 'w') as f:
                json.dump(all_results[layer_idx], f, indent=2)

            # Also save combined results
            with open(f'sweep_results_{dataset_name}_all_layers.json', 'w') as f:
                json.dump(all_results, f, indent=2)

    # Find best configuration per layer and overall
    if all_results:
        print(f"\n{'='*60}")
        print(f"BEST CONFIGURATIONS PER LAYER:")
        print(f"{'='*60}")

        overall_best = None
        overall_best_variance = -float('inf')

        for layer_idx in sorted(all_results.keys()):
            layer_results = all_results[layer_idx]
            if layer_results:
                best = max(layer_results, key=lambda x: x['explained_variance'])
                print(f"\nLayer {layer_idx}:")
                print(f"  Expansion Factor: {best['expansion_factor']}")
                print(f"  K (TopK): {best['k']}")
                print(f"  Learning Rate: {best['lr']}")
                print(f"  Explained Variance: {best['explained_variance']:.4f}")

                # Track overall best
                if best['explained_variance'] > overall_best_variance:
                    overall_best = best
                    overall_best_variance = best['explained_variance']

        if overall_best:
            print(f"\n{'='*60}")
            print(f"OVERALL BEST CONFIGURATION:")
            print(f"Layer: {overall_best['layer']}")
            print(f"Expansion Factor: {overall_best['expansion_factor']}")
            print(f"K (TopK): {overall_best['k']}")
            print(f"Learning Rate: {overall_best['lr']}")
            print(f"Explained Variance: {overall_best['explained_variance']:.4f}")
            print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    results = main()
