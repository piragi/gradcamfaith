import torch
import torchvision
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.models.weight_conversion import convert_timm_weights
from vit_prisma.sae import VisionModelSAERunnerConfig, VisionSAETrainer

from vit.preprocessing import get_processor_for_precached_224_images

model_name = "vit_base_patch16_224"
hooked_model = HookedSAEViT.from_pretrained(model_name)

num_classes = 6
hooked_model.head = torch.nn.Linear(hooked_model.cfg.d_model, num_classes)

checkpoint = torch.load(
    "./model/vit_b-ImageNet_class_init-frozen_False-dataset_Hyperkvasir_anatomical.pth", weights_only=False
)

state_dict = checkpoint['model_state_dict'].copy()
if 'lin_head.weight' in state_dict:
    state_dict['head.weight'] = state_dict.pop('lin_head.weight')
if 'lin_head.bias' in state_dict:
    state_dict['head.bias'] = state_dict.pop('lin_head.bias')

converted_weights = convert_timm_weights(state_dict, hooked_model.cfg)
hooked_model.load_state_dict(converted_weights)
hooked_model.to('cuda')

label_map = {2: 3, 3: 2}  # original label 0 -> new label 2, original label 1 -> new label 0


def custom_target_transform(target):
    return label_map.get(target, target)


train_path = "./hyper-kvasir_imagefolder/train"
val_path = "./hyper-kvasir_imagefolder/val"
train_dataset = torchvision.datasets.ImageFolder(
    train_path, get_processor_for_precached_224_images(), target_transform=custom_target_transform
)
val_dataset = torchvision.datasets.ImageFolder(
    val_path, get_processor_for_precached_224_images(), target_transform=custom_target_transform
)

print(f"Training dataset size: {len(train_dataset)} images")
print(f"Validation dataset size: {len(val_dataset)} images")
print(f"Total dataset size: {len(train_dataset) + len(val_dataset)} images")

sae_trainer_cfg = VisionModelSAERunnerConfig(
    hook_point_layer=10,
    model_name=model_name,
    layer_subtype='hook_resid_post',
    dataset_name="hyper-kvasir",
    activation_fn_str='relu',
    expansion_factor=8,  # Middle ground
    l1_coefficient=1e-3,  # Encourage more sparsity

    # Training settings that worked
    train_batch_size=128,
    lr=5e-6,
    lr_scheduler_name="cosineannealingwarmup",
    lr_warm_up_steps=10,
    num_epochs=50,

    # Feature resampling
    feature_sampling_window=100,
    dead_feature_window=100,
    use_ghost_grads=False,

    # CLS-specific
    cls_token_only=True,
    context_size=197,

    # Logging
    checkpoint_path='./models/cls_sae/',
    wandb_project='hyperkvasir_cls_sae',
    n_checkpoints=5
)
trainer = VisionSAETrainer(sae_trainer_cfg, hooked_model, train_dataset, val_dataset)
sae = trainer.run()
