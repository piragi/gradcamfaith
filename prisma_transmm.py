# attribution_prisma.py
import gc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from vit_prisma.models.base_vit import HookedViT  # Import the new model class
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.models.weight_conversion import convert_timm_weights
from vit_prisma.sae import SparseAutoencoder

import vit.model as model_handler  # You might need to adapt or replace this too
from config import PipelineConfig
from vit.model import IDX2CLS
from vit.preprocessing import get_processor_for_precached_224_images


def load_models():
    """Load SAE and fine-tuned model"""
    # Load SAE
    sae_path = "./models/cls_sae//bfa31d82-hyperkvasir_cls_sae/n_images_4168.pt"
    sae = SparseAutoencoder.load_from_pretrained(sae_path)
    sae.cuda().eval()

    # Load model
    model = HookedSAEViT.from_pretrained("vit_base_patch16_224")
    model.head = torch.nn.Linear(model.cfg.d_model, 6)

    checkpoint = torch.load(
        "./model/vit_b-ImageNet_class_init-frozen_False-dataset_Hyperkvasir_anatomical.pth", weights_only=False
    )
    state_dict = checkpoint['model_state_dict'].copy()

    if 'lin_head.weight' in state_dict:
        state_dict['head.weight'] = state_dict.pop('lin_head.weight')
    if 'lin_head.bias' in state_dict:
        state_dict['head.bias'] = state_dict.pop('lin_head.bias')

    converted_weights = convert_timm_weights(state_dict, model.cfg)
    model.load_state_dict(converted_weights)
    model.cuda().eval()

    return sae, model


def find_class_specific_features(model, sae, n_batches=1000):
    """Find features that activate strongly for specific classes"""
    class_feature_scores = torch.zeros(6, sae.cfg.d_sae).cuda()
    class_counts = torch.zeros(6).cuda()

    label_map = {2: 3, 3: 2}  # original label 0 -> new label 2, original label 1 -> new label 0

    def custom_target_transform(target):
        return label_map.get(target, target)

    train_path = "./hyper-kvasir_imagefolder/train"
    train_dataset = torchvision.datasets.ImageFolder(
        train_path, get_processor_for_precached_224_images(), target_transform=custom_target_transform
    )
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if i >= n_batches:
                break

            # Get activations
            _, cache = model.run_with_cache(imgs.cuda(), names_filter=sae.cfg.hook_point)
            acts = cache[sae.cfg.hook_point][:, 0:1, :]  # CLS only

            # Encode
            feature_acts = sae.encode(acts)[1]  # [batch, 1, d_sae]

            # Accumulate by class
            for j in range(imgs.shape[0]):
                label = labels[j].item()
                if label < 6:  # Safety check
                    class_feature_scores[label] += feature_acts[j, 0]
                    class_counts[label] += 1

    # Check which classes we found
    print("Samples per class:", class_counts.cpu().numpy())

    # Normalize
    for i in range(6):
        if class_counts[i] > 0:
            class_feature_scores[i] /= class_counts[i]

    # Find most discriminative features
    discriminative_features = {}
    for i in range(6):
        # Features that fire strongly for class i
        top_features = class_feature_scores[i].topk(20)

        # Features that fire strongly for i but not others
        other_classes_mean = (class_feature_scores.sum(0) - class_feature_scores[i]) / 5
        discrimination_score = class_feature_scores[i] - other_classes_mean
        most_discriminative = discrimination_score.topk(10)

        discriminative_features[i] = {
            'top_features': top_features.indices.tolist(),
            'most_discriminative': most_discriminative.indices.tolist(),
            'discrimination_scores': most_discriminative.values.tolist()
        }

    return discriminative_features


# --- These helper functions do not need to change ---
def avg_heads(cam: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    cam = cam.cpu()
    grad = grad.cpu()
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def apply_self_attention_rules(R_ss: torch.Tensor, cam_ss: torch.Tensor) -> torch.Tensor:
    R_ss = R_ss.cpu()
    cam_ss = cam_ss.cpu()
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


# --- The main refactored function ---


def transmm_prisma(
    model_prisma: HookedViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    device: Optional[torch.device] = None,
    img_size: int = 224,
    sae: Optional[SparseAutoencoder] = None,  # The SAE model, needed for steering
    discriminative_features: Optional[Dict[str, Any]] = None,  # Dict with steering params
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Memory-efficient implementation of TransMM, adapted for vit-prisma's HookedViT.
    Returns: (prediction_dict, positive_attr_np)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steering_options = {'layer_idx': 10, 'multiplier': 2.0, 'feature_idx': "predicted_class"}
    class_feature_map = {0: 165, 2: 3301, 3: 3714, 4: 3714, 5: 4167}
    # class_feature_map = {
    # class_idx: data['top_features'][0]
    # for class_idx, data in discriminative_features.items() if data['most_discriminative']
    # }

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # We only need one forward/backward pass. We'll use hooks to capture
    # everything we need in that single pass.
    model_prisma.reset_hooks()
    model_prisma.zero_grad()

    # Create a cache to store the activations and gradients from our hooks
    cache = {}

    def save_activation_hook(tensor: torch.Tensor, hook: Any):
        """A forward hook to save an activation to the cache."""
        cache[hook.name] = tensor

    def save_gradient_hook(tensor: torch.Tensor, hook: Any):
        """A backward hook to save a gradient to the cache."""
        # The key name is predictable: {hook_name}_grad
        cache[hook.name + "_grad"] = tensor

    # Define the names of the hooks we want to attach to.
    # 'hook_pattern' is the attention probability map post-softmax.
    attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(model_prisma.cfg.n_layers)]

    # Create the list of hooks to add
    fwd_hooks = [(name, save_activation_hook) for name in attn_hook_names]
    bwd_hooks = [(name, save_gradient_hook) for name in attn_hook_names]

    # --- DYNAMIC STEERING LOGIC ---
    if steering_options and sae and steering_options.get('feature_idx') == "predicted_class":
        if not class_feature_map:
            raise ValueError("`class_feature_map` is required for 'predicted_class' steering.")

        # 1. Perform an initial, cheap forward pass to get the prediction
        with torch.no_grad():
            initial_logits = model_prisma(input_tensor)
            predicted_class_idx = torch.argmax(initial_logits, dim=-1).item()

        # 2. Look up the corresponding feature from our pre-computed map
        feature_to_steer = class_feature_map.get(predicted_class_idx)

    # If we determined a feature to steer with (either statically or dynamically)
    if feature_to_steer is not None and isinstance(feature_to_steer, int) and False:
        layer_idx = steering_options['layer_idx']
        multiplier = steering_options['multiplier']

        steering_vector = sae.W_dec[feature_to_steer, :].to(device) * multiplier

        def steering_hook(resid_stream, hook):
            resid_stream[:, 0, :] += steering_vector
            return resid_stream

        steering_hook_name = f"blocks.{layer_idx}.hook_resid_post"
        fwd_hooks.append((steering_hook_name, steering_hook))

    # Use the model.hooks() context manager to temporarily add the hooks
    with model_prisma.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        # Perform the forward pass to get logits. Activations are cached automatically.
        logits = model_prisma(input_tensor)

        # This part remains the same: get target class, create one-hot, calculate loss
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

        # Build the prediction dictionary. You might want to customize this.
        prediction_result_dict = {
            "logits": logits,
            "probabilities": probabilities.squeeze().cpu().detach().numpy(),
            "predicted_class_idx": predicted_class_idx,
            "predicted_class_label": IDX2CLS[predicted_class_idx]
        }

        one_hot = torch.zeros((1, logits.size(-1)), dtype=torch.float32, device=device)
        one_hot[0, predicted_class_idx] = 1
        one_hot.requires_grad_(True)

        loss = torch.sum(one_hot * logits)

        # Perform the backward pass. Gradients are now cached automatically by the backward hooks.
        loss.backward(retain_graph=True)  # Retain graph if you might need it later

    model_prisma.reset_hooks()
    # ---- Attribution Calculation (This section is almost identical to your original) ----

    # The number of tokens includes the CLS token
    num_tokens = cache[attn_hook_names[0]].shape[-1]
    R_pos = torch.eye(num_tokens, num_tokens, device='cpu')

    for i in range(model_prisma.cfg.n_layers):
        hook_name = f"blocks.{i}.attn.hook_pattern"

        # Retrieve the saved attention map and gradient from our cache
        grad = cache[hook_name + "_grad"]
        cam = cache[hook_name]

        # Your original logic works perfectly from here
        cam_pos_avg = avg_heads(cam, grad)  # Returns CPU tensor
        R_pos = R_pos + apply_self_attention_rules(R_pos, cam_pos_avg)

        # Optional: clean up the cache as we go to save memory
        del cache[hook_name], cache[hook_name + "_grad"], grad, cam, cam_pos_avg

    # Extract attribution for patch tokens from the CLS token's relevance
    transformer_attribution_pos = R_pos[0, 1:].clone()
    del R_pos, cache
    gc.collect()

    # ---- Reshaping and Normalization (This section is identical) ----
    def process_attribution_map(attr_tensor: torch.Tensor) -> np.ndarray:
        side_len = int(np.sqrt(attr_tensor.size(0)))
        attr_tensor = attr_tensor.reshape(1, 1, side_len, side_len)
        attr_tensor_device = attr_tensor.to(device)
        attr_interpolated = F.interpolate(
            attr_tensor_device, size=(img_size, img_size), mode='bilinear', align_corners=False
        )
        return attr_interpolated.squeeze().cpu().detach().numpy()

    attribution_pos_np = process_attribution_map(transformer_attribution_pos)

    normalize_fn = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8) if (np.max(x) - np.min(x)) > 1e-8 else x
    attribution_pos_np = normalize_fn(attribution_pos_np)

    # ---- Cleanup (This section is identical) ----
    del transformer_attribution_pos, input_tensor, one_hot, loss
    torch.cuda.empty_cache()
    gc.collect()

    # If the probabilities in the dict are still a tensor, convert them
    if isinstance(prediction_result_dict["probabilities"], (torch.Tensor, np.ndarray)):
        prediction_result_dict["probabilities"] = prediction_result_dict["probabilities"].tolist()

    return (
        prediction_result_dict,
        attribution_pos_np,
    )


def generate_attribution_prisma(
    model: HookedSAEViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    device: Optional[torch.device] = None,
    sae: Optional[SparseAutoencoder] = None,  # The SAE model, needed for steering
    steering_options: Optional[Dict[str, Any]] = None,  # Dict with steering params
) -> Dict[str, Any]:
    """
    Unified interface for generating attribution maps.
    Returns a dictionary formatted for pipeline.py.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure input_tensor is on the correct device
    input_tensor = input_tensor.to(device)

    (pred_dict, pos_attr_np) = transmm_prisma(
        model_prisma=model, input_tensor=input_tensor, config=config, sae=sae, discriminative_features=steering_options
    )

    # Structure the output dictionary as expected by pipeline.py
    return {
        "predictions": pred_dict,  # The dict from model_handler.get_prediction via transmm
        "attribution_positive": pos_attr_np,
        "logits": None,  # This will be None if transmm returns None
        "ffn_activity": [],  # List of dicts
        "class_embedding_representation": [],  # List of dicts
        "head_contribution": []
    }
