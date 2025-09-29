"""
Enhanced TransLRP with Feature Gradient Gating
Integrates SAE feature gradients for improved faithfulness
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from vit_prisma.models.base_vit import HookedSAEViT, HookedViT

from config import PipelineConfig
from feature_gradient_gating import apply_feature_gradient_gating


def apply_gradient_gating_to_cam(
    cam_pos_avg: torch.Tensor, layer_idx: int, gradients: Dict[str, torch.Tensor], residuals: Dict[int, torch.Tensor],
    steering_resources: Dict[int, Dict[str, Any]], config: PipelineConfig, device: torch.device
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Apply feature gradient gating to a CAM tensor at a specific layer."""
    resid_grad_key = f"blocks.{layer_idx}.hook_resid_post_grad"

    if resid_grad_key not in gradients or layer_idx not in residuals:
        return cam_pos_avg, {}

    # Get gradient and residual - already on CPU from hooks
    residual_grad_cpu = gradients[resid_grad_key]
    residual_tensor_cpu = residuals[layer_idx]
    sae = steering_resources[layer_idx]["sae"]

    # Compute SAE codes on-demand
    with torch.no_grad():
        residual_tensor_gpu = residual_tensor_cpu.to(device)
        _, codes = sae.encode(residual_tensor_gpu)
        codes = codes.detach().cpu()
        del residual_tensor_gpu

    # Move gradient to GPU for computation
    residual_grad = residual_grad_cpu.to(device)
    codes_gpu = codes.to(device)

    if residual_grad.dim() == 3:
        residual_grad = residual_grad[0]
    residual_grad = residual_grad[1:]  # Remove CLS

    if codes_gpu.dim() == 3:
        codes_gpu = codes_gpu[0]
    codes_gpu = codes_gpu[1:]  # Remove CLS

    # Apply feature gradient gating - get parameters from config
    topk_value = getattr(config.classify.boosting, 'top_k_features', 5)
    if topk_value is None:
        topk_value = 15  # Default when None
    gating_config = {
        'top_k_features': topk_value,
        'kappa': getattr(config.classify.boosting, 'kappa', 50.0),
        'clamp_min': 0.1,
        'clamp_max': 10.0,
        'denoise_gradient': False,
        'denoise_alpha': 5.0,
        'denoise_min': 0.6,
        'gate_construction': getattr(config.classify.boosting, 'gate_construction', 'combined'),
        'shuffle_decoder': getattr(config.classify.boosting, 'shuffle_decoder', False),
    }

    # Get residuals for denoising if available
    layer_residuals = None  # Disable denoising for now

    gated_cam, layer_debug = apply_feature_gradient_gating(
        cam_pos_avg=cam_pos_avg,
        residual_grad=residual_grad,
        sae_codes=codes_gpu,
        sae=sae,
        config=gating_config,
        enable_denoising=False,  # Disabled due to shape issues
        residuals=layer_residuals,
        debug=False  # Disable debug to avoid storing large arrays
    )

    return gated_cam, layer_debug


def compute_layer_attribution(
    model_cfg: HookedViTConfig, activations: Dict[str, torch.Tensor], gradients: Dict[str, torch.Tensor],
    residuals: Dict[int, torch.Tensor], feature_gradient_layers: List[int],
    steering_resources: Optional[Dict[int, Dict[str, Any]]], config: PipelineConfig, device: torch.device
) -> torch.Tensor:
    """Compute attribution by iterating through layers and applying attention rules."""
    attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(model_cfg.n_layers)]
    num_tokens = activations[attn_hook_names[0]].shape[-1]
    R_pos = torch.eye(num_tokens, num_tokens, device='cpu')

    for i in range(model_cfg.n_layers):
        hname = f"blocks.{i}.attn.hook_pattern"
        grad = gradients[hname + "_grad"]
        cam = activations[hname]
        cam_pos_avg = avg_heads(cam, grad)

        # Apply feature gradient gating if configured
        if (i in feature_gradient_layers and steering_resources is not None and i in steering_resources):
            cam_pos_avg, _ = apply_gradient_gating_to_cam(
                cam_pos_avg, i, gradients, residuals, steering_resources, config, device
            )

        R_pos = R_pos + apply_self_attention_rules(R_pos, cam_pos_avg)

    transformer_attribution_pos = R_pos[0, 1:].clone()

    return transformer_attribution_pos


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


def run_model_forward_backward(
    model_prisma: HookedViT, input_tensor: torch.Tensor, clip_classifier: Optional[Any], device: torch.device
) -> Tuple[Dict[str, Any], int]:
    """Run forward and backward pass, return predictions and predicted class."""
    if clip_classifier is not None:
        clip_result = clip_classifier.forward(input_tensor, requires_grad=True)
        logits = clip_result["logits"]
        probabilities = clip_result["probabilities"]
        predicted_class_idx = clip_result["predicted_class_idx"]
    else:
        logits = model_prisma(input_tensor)
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

    # Backward pass
    num_classes = logits.size(-1)
    one_hot = torch.zeros((1, num_classes), dtype=torch.float32, device=device)
    one_hot[0, int(predicted_class_idx)] = 1
    one_hot.requires_grad_(True)
    loss = torch.sum(one_hot * logits)
    loss.backward(retain_graph=False)

    prediction_result = {
        "logits":
        logits.detach().cpu().numpy(),
        "probabilities":
        probabilities.squeeze().cpu().detach().numpy().tolist()
        if isinstance(probabilities, torch.Tensor) else probabilities,
        "predicted_class_idx":
        predicted_class_idx,
    }

    return prediction_result, int(predicted_class_idx)


def setup_hooks(model_prisma: HookedViT, feature_gradient_layers: List[int]) -> Tuple[List, List, Dict, Dict, Dict]:
    """Setup forward and backward hooks, return hook lists and storage dictionaries."""
    gradients = {}
    activations = {}
    residuals = {}

    attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(model_prisma.cfg.n_layers)]

    def save_activation_hook(tensor: torch.Tensor, hook: Any):
        activations[hook.name] = tensor.detach().cpu()

    def save_gradient_hook(grad: torch.Tensor, hook: Any):
        if grad is not None:
            gradients[hook.name + "_grad"] = grad.detach().cpu()

    fwd_hooks = [(name, save_activation_hook) for name in attn_hook_names]
    bwd_hooks = [(name, save_gradient_hook) for name in attn_hook_names]

    # Add residual hooks for feature gradient layers
    all_resid_layers = set(feature_gradient_layers)
    if all_resid_layers:

        def save_resid_hook(tensor, hook):
            layer_idx = int(hook.name.split('.')[1])
            if layer_idx in feature_gradient_layers:
                residuals[layer_idx] = tensor.detach().cpu()
            return tensor

        for layer_idx in all_resid_layers:
            resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
            fwd_hooks.append((resid_hook_name, save_resid_hook))
            bwd_hooks.append((resid_hook_name, save_gradient_hook))

    return fwd_hooks, bwd_hooks, gradients, activations, residuals


def transmm_prisma_enhanced(
    model_prisma: HookedViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],
    device: Optional[torch.device] = None,
    img_size: int = 224,
    steering_resources: Optional[Dict[int, Dict[str, Any]]] = None,
    enable_feature_gradients: bool = True,
    feature_gradient_layers: Optional[List[int]] = None,
    clip_classifier: Optional[Any] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    TransMM with feature gradient gating for improved faithfulness.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default to applying feature gradients at layers 9-10 if not specified
    if feature_gradient_layers is None:
        feature_gradient_layers = [9, 10] if enable_feature_gradients else []

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # Setup hooks for data collection
    fwd_hooks, bwd_hooks, gradients, activations, residuals = setup_hooks(model_prisma, feature_gradient_layers)

    # Run model with hooks to collect gradients and activations
    with model_prisma.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, reset_hooks_end=True):
        prediction_result, predicted_class_idx = run_model_forward_backward(
            model_prisma, input_tensor, clip_classifier, device
        )

    # Add class label to prediction result
    prediction_result["predicted_class_label"] = idx_to_class.get(predicted_class_idx, f"class_{predicted_class_idx}")

    if not gradients:
        raise RuntimeError("No gradients captured!")

    # Compute attribution
    transformer_attribution_pos = compute_layer_attribution(
        model_prisma.cfg, activations, gradients, residuals, feature_gradient_layers, steering_resources, config, device
    )

    # Convert to numpy and process attribution map
    raw_patch_map = transformer_attribution_pos.detach().cpu().numpy()

    # Reshape and interpolate to image size
    side_len = int(np.sqrt(transformer_attribution_pos.size(0)))
    attribution_reshaped = transformer_attribution_pos.reshape(1, 1, side_len, side_len)
    attribution_pos_np = F.interpolate(
        attribution_reshaped, size=(img_size, img_size), mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()

    # Normalize
    normalize_fn = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8) if (np.max(x) - np.min(x)) > 1e-8 else x
    attribution_pos_np = normalize_fn(attribution_pos_np)

    return (prediction_result, attribution_pos_np, raw_patch_map)


def generate_attribution_prisma_enhanced(
    model: HookedSAEViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],
    device: Optional[torch.device] = None,
    steering_resources: Optional[Dict[int, Dict[str, Any]]] = None,
    enable_feature_gradients: bool = True,
    feature_gradient_layers: Optional[List[int]] = None,
    clip_classifier: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate attribution with feature gradient gating for improved faithfulness.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = input_tensor.to(device)

    (pred_dict, pos_attr_np, raw_patch_map) = transmm_prisma_enhanced(
        model_prisma=model,
        input_tensor=input_tensor,
        steering_resources=steering_resources,
        config=config,
        enable_feature_gradients=enable_feature_gradients,
        feature_gradient_layers=feature_gradient_layers,
        idx_to_class=idx_to_class,
        clip_classifier=clip_classifier,
    )

    return {
        "predictions": pred_dict,
        "attribution_positive": pos_attr_np,
        "raw_attribution": raw_patch_map,
    }
