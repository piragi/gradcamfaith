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
    hook_point_layer=11,
    layer_subtype='hook_resid_post',
    cls_token_only=False,
    context_size=197,
    expansion_factor=16,
    lr=1e-5,
    l1_coefficient=1e-5,
    train_batch_size=4096,
    num_epochs=3,
    lr_scheduler_name="cosineannealingwarmup",
    lr_warm_up_steps=200,
    n_batches_in_buffer=64,
    initialization_method="encoder_transpose_decoder",
    dataset_name="hyper-kvasir",
    log_to_wandb=True,
    wandb_project='vit_medical_sae_vanilla_sweep',
    n_checkpoints=1,
    use_ghost_grads=True,
    dead_feature_threshold=1e-9,
    feature_sampling_window=1000,
    dead_feature_window=20,
    activation_fn_str="relu",
)

sweep_parameters = {
    'l1_coefficient': [1e-5]  # Sweep around the paper's successful 1e-5 value
}

for l1_val in sweep_parameters['l1_coefficient']:
    run_config = copy.deepcopy(base_config)

    # Set the L1 coefficient for this specific run
    run_config.l1_coefficient = l1_val

    run_name = f"vanilla_l1_{l1_val}_exp{run_config.expansion_factor}_lr{run_config.lr}"
    run_config.wandb_run_name = run_name

    checkpoint_dir = f'./models/sweep/{run_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_config.checkpoint_path = checkpoint_dir

    print(f"STARTING SWEEP RUN: {run_name}")
    print(f"Configuration: l1_coefficient={l1_val}")

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
        wandb.finish()
