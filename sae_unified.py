import copy
import gc
import os
from pathlib import Path

import torch
import torchvision
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.sae import VisionSAETrainer
from vit_prisma.sae.config import VisionModelSAERunnerConfig
import wandb

from dataset_config import get_dataset_config
from pipeline_unified import load_model_for_dataset
import logging

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

# ============ CONFIG ============
config = {
    'dataset': 'covidquex',  # 'covidquex' or 'hyperkvasir'
    'layers': [6],     # which layers to train
    'k': 64,                    # topk activation
    'expansion_factor': 64,
    'lr': 2e-5,
    'epochs': 3,
    'batch_size': 4096,
    'wandb_project': 'vit_unified_sae',
    'log_to_wandb': True,
}
# ================================

# Load dataset config
dataset_config = get_dataset_config(config['dataset'])
print(f"Training SAEs for {dataset_config.name} dataset")
print(f"Number of classes: {dataset_config.num_classes}")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hooked_model = load_model_for_dataset(dataset_config, device)

# Set up data paths
data_path = Path(f"data/{config['dataset']}_unified")
train_path = data_path / "train"
val_path = data_path / "val"

# Get transform for preprocessed 224x224 images
from vit.preprocessing import get_processor_for_precached_224_images
transform = get_processor_for_precached_224_images()

# Load datasets
train_dataset = torchvision.datasets.ImageFolder(train_path, transform)
val_dataset = torchvision.datasets.ImageFolder(val_path, transform)
print(f"Training dataset size: {len(train_dataset)} images")
print(f"Validation dataset size: {len(val_dataset)} images")

# Base SAE config
base_config = VisionModelSAERunnerConfig(
    model_name="vit_base_patch16_224",
    layer_subtype='hook_resid_post',
    cls_token_only=False,
    context_size=197,
    expansion_factor=config['expansion_factor'],
    activation_fn_str="topk",
    activation_fn_kwargs={'k': config['k']},
    lr=config['lr'],
    train_batch_size=config['batch_size'],
    num_epochs=config['epochs'],
    lr_scheduler_name="cosineannealingwarmup",
    lr_warm_up_steps=100,
    n_batches_in_buffer=64,
    initialization_method="encoder_transpose_decoder",
    dataset_name=config['dataset'],
    log_to_wandb=config['log_to_wandb'],
    wandb_project=config['wandb_project'],
    n_checkpoints=1,
    use_ghost_grads=True,
    dead_feature_threshold=1e-9,
    feature_sampling_window=1000,
    dead_feature_window=20,
)

# Train SAE for each layer
for layer_idx in config['layers']:
    # Clear memory at start of each iteration
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    run_config = copy.deepcopy(base_config)
    run_config.hook_point_layer = layer_idx

    # Set up save directory in our organized structure
    save_dir = Path(f"data/sae_{config['dataset']}/layer_{layer_idx}")
    save_dir.mkdir(parents=True, exist_ok=True)
    run_config.checkpoint_path = str(save_dir)

    run_name = f"sae_{config['dataset']}_l{layer_idx}_k{config['k']}_exp{config['expansion_factor']}"
    print(f"\nSTARTING: {run_name}")
    print(f"Layer {layer_idx}, k={config['k']}, expansion={config['expansion_factor']}")
    print(f"Saving to: {save_dir}")

    trainer = VisionSAETrainer(run_config, hooked_model, train_dataset, val_dataset)

    try:
        trained_sae = trainer.run()
        print(f"SUCCESS: Finished training for layer {layer_idx}")
        print(f"SAE saved in: {save_dir}/n_images_{len(train_dataset)}.pt")
    except Exception as e:
        print(f"ERROR: Training failed for layer {layer_idx}")
        print(f"Error details: {e}")
        continue
    finally:
        print("Cleaning up memory...")
        wandb.finish()

        # Cleanup
        if 'trained_sae' in locals():
            del trained_sae
        del trainer

        # Clear model gradients
        for param in hooked_model.parameters():
            param.grad = None

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()

print(f"\nAll SAEs trained and saved in data/sae_{config['dataset']}/")
