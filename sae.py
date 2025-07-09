import copy
import os

import torch
import torchvision
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.sae import VisionSAETrainer
from vit_prisma.sae.config import VisionModelSAERunnerConfig

import wandb
from vit.preprocessing import get_processor_for_precached_224_images

model_name = "vit_base_patch16_224"
hooked_model = HookedSAEViT.from_pretrained(model_name, load_pretrained_model=False)

num_classes = 6
hooked_model.head = torch.nn.Linear(hooked_model.cfg.d_model, num_classes)

# Load your fine-tuned model checkpoint
checkpoint_path = "./model/vit_b-ImageNet_class_init-frozen_False-dataset_Hyperkvasir_anatomical.pth"
checkpoint = torch.load(checkpoint_path, weights_only=False)

state_dict = checkpoint.get('model_state_dict', checkpoint)
if 'lin_head.weight' in state_dict:
    state_dict['head.weight'] = state_dict.pop('lin_head.weight')
if 'lin_head.bias' in state_dict:
    state_dict['head.bias'] = state_dict.pop('lin_head.bias')
hooked_model.load_state_dict(state_dict, strict=False)
hooked_model.to('cuda')
hooked_model.eval()
print(f"Successfully loaded fine-tuned model from {checkpoint_path}")

train_path = "./hyper-kvasir_imagefolder/train"
val_path = "./hyper-kvasir_imagefolder/val"

transform = get_processor_for_precached_224_images()

label_map = {2: 3, 3: 2}


def custom_target_transform(target):
    return label_map.get(target, target)


train_dataset = torchvision.datasets.ImageFolder(train_path, transform, target_transform=custom_target_transform)
val_dataset = torchvision.datasets.ImageFolder(val_path, transform, target_transform=custom_target_transform)
print(f"Training dataset size: {len(train_dataset)} images")

base_config = VisionModelSAERunnerConfig(
    model_name=model_name,
    # hook_point_layer=9,
    layer_subtype='hook_resid_post',
    cls_token_only=False,
    context_size=197,
    expansion_factor=32,
    activation_fn_str="topk",
    activation_fn_kwargs={'k': 1024},
    lr=0.00002,
    # l1_coefficient=2e-4,
    train_batch_size=4096,
    num_epochs=3,
    lr_scheduler_name="cosineannealingwarmup",
    lr_warm_up_steps=100,
    n_batches_in_buffer=64,
    initialization_method="encoder_transpose_decoder",
    dataset_name="hyper-kvasir",
    log_to_wandb=False,
    wandb_project='vit_medical_sae_k_sweep',
    n_checkpoints=1,
    use_ghost_grads=True,
    dead_feature_threshold=1e-9,
    feature_sampling_window=1000,
    dead_feature_window=20,
)

sweep_parameters = {'hook_point_layer': [6, 7, 8, 9, 10], 'k_values': [2048]}

for layer_idx in sweep_parameters['hook_point_layer']:
    run_config = copy.deepcopy(base_config)

    run_config.hook_point_layer = layer_idx

    run_name = f"sae_k{run_config.activation_fn_kwargs['k']}_exp{run_config.expansion_factor}_lr{run_config.lr}"
    # run_config.wandb_run_name = run_name

    checkpoint_dir = f'./models/sweep/{run_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_config.checkpoint_path = checkpoint_dir

    print(f"STARTING SWEEP RUN: {run_name}")
    print(f"Configuration: k={run_config.activation_fn_kwargs['k']}, expansion={run_config.expansion_factor}")

    trainer = VisionSAETrainer(run_config, hooked_model, train_dataset, val_dataset)

    try:
        trained_sae = trainer.run()
        print(f"\nSUCCESS: Finished training for {run_name}")
        print(f"Final SAE saved in: {run_config.checkpoint_path}")
    except Exception as e:
        print(f"\nERROR: Training failed for {run_name}")
        print(f"Error details: {e}")
        continue
    finally:
        print("Finishing wandb run")
        # wandb.finish()
