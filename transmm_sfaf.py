"""
Enhanced TransLRP with Feature Gradient Gating
Integrates SAE feature gradients for improved faithfulness
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from vit_prisma.models.base_vit import HookedSAEViT, HookedViT
from vit_prisma.models.weight_conversion import convert_timm_weights
from vit_prisma.sae import SparseAutoencoder

from build_boost_mask_improved import (
    build_additive_correction_mask, build_bias_multiplicative_mask, build_boost_mask_improved, build_class_aware_mask,
    precache_bias_multiplicative_features, precache_sorted_features
)
from config import PipelineConfig
from feature_gradient_gating import (
    apply_feature_gradient_gating, compute_feature_gradient_gate, compute_reconstruction_denoise_gate
)


def load_models():
    """Load SAE and fine-tuned model"""
    model = HookedSAEViT.from_pretrained("vit_base_patch16_224")
    model.head = torch.nn.Linear(model.cfg.d_model, 3)

    checkpoint = torch.load("./model/model_best.pth.tar")
    state_dict = checkpoint['state_dict'].copy()

    if 'lin_head.weight' in state_dict:
        state_dict['head.weight'] = state_dict.pop('lin_head.weight')
    if 'lin_head.bias' in state_dict:
        state_dict['head.bias'] = state_dict.pop('lin_head.bias')

    converted_weights = convert_timm_weights(state_dict, model.cfg)
    model.load_state_dict(converted_weights)
    model.cuda().eval()

    return model


def load_steering_resources(layers: List[int], dataset_name: str = None) -> Dict[int, Dict[str, Any]]:
    """
    Loads SAEs and S_f/A_f dictionaries for the specified layers.
    Dynamically finds SAE and feature dict paths based on dataset and layer.
    
    Args:
        layers: List of layer indices to load
        dataset_name: Name of the dataset ('covidquex' or 'hyperkvasir')
    """
    resources = {}

    for layer_idx in layers:
        try:
            # TEST: Use ImageNet CLIP B-32 SAE for waterbirds layer 5
            if dataset_name == "waterbirds":
                print("=" * 60)
                print("TEST MODE: Using ImageNet CLIP B-32 SAE for steering")
                print("=" * 60)
                sae_path = Path(f"data/sae_waterbirds_clip_b32/layer_{layer_idx}/weights.pt")
                if not sae_path.exists():
                    print(f"Warning: ImageNet SAE not found at {sae_path}")
                    continue
                print(f"Loading ImageNet SAE from {sae_path}")
                sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
                sae.cuda().eval()
            else:
                # Original logic for other datasets
                sae_dir = Path("data") / f"sae_{dataset_name}" / f"layer_{layer_idx}"
                sae_files = list(sae_dir.glob("**/n_images_*.pt"))
                # Filter out log_feature_sparsity files
                sae_files = [f for f in sae_files if 'log_feature_sparsity' not in str(f)]

                if not sae_files:
                    print(f"Warning: No SAE found for {dataset_name} layer {layer_idx} in {sae_dir}")
                    continue

                # Use the most recent SAE file
                sae_path = sorted(sae_files)[-1]
                print(f"Loading SAE from {sae_path}")

                sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
                sae.cuda().eval()

            resources[layer_idx] = {"sae": sae}

            # Load SaCo feature dictionary - try inference dict first, then fall back to old format
            inference_dict_path = Path(f"data/featuredict_{dataset_name}/layer_{layer_idx}_inference_dict.pt")
            old_saco_path = Path(f"data/featuredict_{dataset_name}/layer_{layer_idx}_saco_features.pt")

            if inference_dict_path.exists():
                # New robust inference dictionary with class-aware features
                inference_dict = torch.load(inference_dict_path, weights_only=False)
                resources[layer_idx]["inference_dict"] = inference_dict
                resources[layer_idx]["use_class_aware"] = True
                print(f"Loaded class-aware inference dict for {dataset_name} layer {layer_idx}")
                print(f"  Classes: {inference_dict.get('metadata', {}).get('classes', [])}")
                print(f"  Reliable features: {inference_dict.get('metadata', {}).get('reliable_features', 0)}")
            elif old_saco_path.exists():
                # Old format SaCo results
                saco_results = torch.load(old_saco_path, weights_only=False)
                resources[layer_idx]["saco_dict"] = saco_results
                resources[layer_idx]["use_class_aware"] = False
                print(f"Loaded old-format SaCo dictionary for {dataset_name} layer {layer_idx}")

                # Pre-cache sorted features for performance
                try:
                    # Pre-cache for the old method (if still used)
                    precache_sorted_features(saco_results)
                    # Pre-cache for the new bias multiplicative method
                    precache_bias_multiplicative_features(
                        saco_results,
                        min_occurrences=50,  # Match updated config defaults
                        max_occurrences=500,
                        min_abs_bias=0.001
                    )
                except Exception as e:
                    print(f"Warning: Could not pre-cache features: {e}")
            else:
                print(f"ERROR: SaCo dict not found at {saco_dict_path}")
                print(f"Please run saco_feature_analysis.py first to generate feature dictionaries")

        except Exception as e:
            print(f"Error loading resources for {dataset_name} layer {layer_idx}: {e}")

    return resources


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


def transmm_prisma_enhanced(
    model_prisma: HookedViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],
    device: Optional[torch.device] = None,
    img_size: int = 224,
    steering_resources: Optional[Dict[int, Dict[str, Any]]] = None,
    enable_steering: bool = True,
    enable_feature_gradients: bool = True,  # NEW: Enable feature gradient gating
    feature_gradient_layers: Optional[List[int]] = None,  # NEW: Which layers to apply
    steering_strength: float = 1.5,
    clip_classifier: Optional[Any] = None,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """
    Enhanced TransMM with both feature boosting and feature gradient gating.
    
    Additional parameters:
        enable_feature_gradients: Whether to use feature gradient gating
        feature_gradient_layers: Which layers to apply feature gradients 
                                (default: last 2 layers)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default to applying feature gradients at layers 9-10 if not specified
    if feature_gradient_layers is None:
        feature_gradient_layers = [9, 10] if enable_feature_gradients else []

    active_steering_layers = list(steering_resources.keys()) if enable_steering and steering_resources else []

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # Storage for gradients, activations, SAE codes, and residuals
    gradients = {}
    activations = {}
    sae_codes = {}
    residuals = {}  # Store residuals for feature gradient computation

    attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(model_prisma.cfg.n_layers)]

    def save_activation_hook(tensor: torch.Tensor, hook: Any):
        activations[hook.name] = tensor.detach().cpu()

    def save_gradient_hook(grad: torch.Tensor, hook: Any):
        if grad is not None:
            gradients[hook.name + "_grad"] = grad.detach().cpu()

    fwd_hooks = [(name, save_activation_hook) for name in attn_hook_names]
    bwd_hooks = [(name, save_gradient_hook) for name in attn_hook_names]

    # Add residual hooks for steering and feature gradient layers
    all_resid_layers = set(active_steering_layers) | set(feature_gradient_layers)

    if all_resid_layers:
        # Create a single hook function that checks layer index from hook.name
        def save_resid_hook(tensor, hook):
            # Extract layer index from hook name (e.g., "blocks.4.hook_resid_post" -> 4)
            layer_idx = int(hook.name.split('.')[1])
            if layer_idx in feature_gradient_layers:
                residuals[layer_idx] = tensor.detach().cpu()
            return tensor

        for layer_idx in all_resid_layers:
            resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
            fwd_hooks.append((resid_hook_name, save_resid_hook))  # Use the same function for all

            # NEW: Add backward hook for residual gradients
            if layer_idx in feature_gradient_layers:
                bwd_hooks.append((resid_hook_name, save_gradient_hook))

    # Forward and backward pass - ensure hooks are reset after
    with model_prisma.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, reset_hooks_end=True):
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
        one_hot[0, predicted_class_idx] = 1
        one_hot.requires_grad_(True)
        loss = torch.sum(one_hot * logits)
        loss.backward(retain_graph=False)  # Don't retain graph - we only do one backward pass!

    prediction_result_dict = {
        "logits":
        logits.detach().cpu().numpy(),
        "probabilities":
        probabilities.squeeze().cpu().detach().numpy().tolist()
        if isinstance(probabilities, torch.Tensor) else probabilities,
        "predicted_class_idx":
        predicted_class_idx,
        "predicted_class_label":
        idx_to_class.get(predicted_class_idx, f"class_{predicted_class_idx}")
    }

    if not gradients:
        raise RuntimeError("No gradients captured!")

    # Attribution loop with feature gradient gating
    num_tokens = activations[attn_hook_names[0]].shape[-1]
    R_pos = torch.eye(num_tokens, num_tokens, device='cpu')
    all_boosted_features = {}
    accumulated_correction_mask = None
    feature_gradient_debug = {}

    for i in range(model_prisma.cfg.n_layers):
        hname = f"blocks.{i}.attn.hook_pattern"
        grad = gradients[hname + "_grad"]
        cam = activations[hname]
        cam_pos_avg = avg_heads(cam, grad)

        # Apply feature gradient gating if configured
        if i in feature_gradient_layers and i in steering_resources:
            # print(f"Applying feature gradient gating at layer {i}")

            resid_grad_key = f"blocks.{i}.hook_resid_post_grad"
            if resid_grad_key in gradients and i in residuals:
                # Get gradient and residual - already on CPU from hooks
                residual_grad_cpu = gradients[resid_grad_key]
                residual_tensor_cpu = residuals[i]
                sae = steering_resources[i]["sae"]

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
                gating_config = {
                    'top_k_features': getattr(config.classify.boosting, 'topk_active', 5),
                    'kappa': getattr(config.classify.boosting, 'kappa', 50.0),
                    'clamp_min': 0.2,
                    'clamp_max': 5.0,
                    'denoise_gradient': True,
                    'denoise_alpha': 5.0,
                    'denoise_min': 0.6,
                }

                # Get residuals for denoising if available
                # NOTE: Disabling denoising for now due to shape mismatch
                # The SAE expects full tokens including CLS, but we've removed CLS
                # from codes. To enable denoising, we'd need to keep the original codes
                # with CLS for reconstruction.
                layer_residuals = None  # Disable denoising for now

                cam_pos_avg, layer_debug = apply_feature_gradient_gating(
                    cam_pos_avg=cam_pos_avg,
                    residual_grad=residual_grad,
                    sae_codes=codes_gpu,
                    sae=sae,
                    config=gating_config,
                    enable_denoising=False,  # Disabled due to shape issues
                    residuals=layer_residuals,
                    debug=False  # Disable debug to avoid storing large arrays
                )

                feature_gradient_debug[i] = layer_debug

                # if layer_debug.get('feature_gating', {}).get('mean_gate'):
                # print(f"  Mean gate multiplier: {layer_debug['feature_gating']['mean_gate']:.3f}")

        # Apply existing feature boosting if configured
        if i in active_steering_layers:
            resources = steering_resources[i]
            codes_for_layer = sae_codes.get(i)

            if codes_for_layer is not None and ("saco_dict" in resources or "inference_dict" in resources):
                boosting_config = config.classify.boosting

                if resources.get('use_class_aware', False) and 'inference_dict' in resources:
                    # Class-aware correction
                    inference_dict = resources['inference_dict']
                    pred_class_idx = prediction_result_dict['predicted_class_idx']
                    pred_class_name = idx_to_class.get(pred_class_idx, f'class_{pred_class_idx}')

                    correction_mask = build_class_aware_mask(
                        codes_for_layer[0, 1:],
                        inference_dict,
                        pred_class_name,
                        top_L_per_patch=10,
                        strength_k=boosting_config.bias_scale_factor,
                        clamp_min=0.2,
                        clamp_max=5.0,
                        debug=False
                    )

                    if accumulated_correction_mask is None:
                        accumulated_correction_mask = correction_mask.cpu()
                    else:
                        accumulated_correction_mask = torch.sqrt(accumulated_correction_mask * correction_mask.cpu())

                elif "saco_dict" in resources:
                    # Original SaCo-based correction
                    saco_results = resources["saco_dict"]
                    correction_mask, selected_feat_ids, debug_info = build_bias_multiplicative_mask(
                        sae_codes=codes_for_layer,
                        saco_results=saco_results,
                        device=device,
                        min_activation=boosting_config.min_activation,
                        correction_method=boosting_config.correction_method,
                        scale_factor=boosting_config.bias_scale_factor,
                        min_abs_bias=boosting_config.min_abs_bias,
                        min_occurrences=boosting_config.min_occurrences,
                        max_occurrences=boosting_config.max_occurrences,
                        topk_active=boosting_config.topk_active,
                        aggregation=boosting_config.aggregation,
                        debug=False
                    )

                    if accumulated_correction_mask is None:
                        accumulated_correction_mask = correction_mask.cpu()
                    else:
                        accumulated_correction_mask = torch.sqrt(accumulated_correction_mask * correction_mask.cpu())

                    all_boosted_features[i] = selected_feat_ids

        R_pos = R_pos + apply_self_attention_rules(R_pos, cam_pos_avg)

    transformer_attribution_pos = R_pos[0, 1:].clone()

    # Apply accumulated correction
    if accumulated_correction_mask is not None and enable_steering:
        # print(f"Applying accumulated correction mask")
        transformer_attribution_pos = transformer_attribution_pos * accumulated_correction_mask

    # Convert to numpy first to avoid keeping tensor references
    raw_patch_map = transformer_attribution_pos.detach().cpu().numpy()

    # Reshape and normalize - work with tensor copy to avoid keeping references
    def process_attribution_map(attr_tensor: torch.Tensor) -> np.ndarray:
        side_len = int(np.sqrt(attr_tensor.size(0)))
        # Clone the tensor to ensure we don't keep references
        attr_tensor = attr_tensor.clone().reshape(1, 1, side_len, side_len)
        attr_tensor_device = attr_tensor.to(device)
        attr_interpolated = F.interpolate(
            attr_tensor_device, size=(img_size, img_size), mode='bilinear', align_corners=False
        )
        result = attr_interpolated.squeeze().cpu().detach().numpy()
        # Clean up intermediate tensors
        del attr_tensor_device, attr_interpolated, attr_tensor
        torch.cuda.empty_cache()
        return result

    # Clone before passing to avoid keeping the original tensor alive
    attribution_pos_np = process_attribution_map(transformer_attribution_pos.clone())

    # Normalize
    normalize_fn = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8) if (np.max(x) - np.min(x)) > 1e-8 else x
    attribution_pos_np = normalize_fn(attribution_pos_np)

    # Add feature gradient debug info to output
    # Make sure we're not keeping any tensor references
    extra_info = {
        'boosted_features': all_boosted_features,
        'feature_gradient_debug': feature_gradient_debug if enable_feature_gradients else {}
    }

    # Clear the debug dictionary if it has tensor references
    for layer_key in feature_gradient_debug.keys():
        layer_debug = feature_gradient_debug[layer_key]
        if 'feature_gating' in layer_debug:
            # These should already be numpy, but ensure they're not keeping references
            for key in [
                'raw_contributions', 'normalized_contributions', 'gate_values', 'top_features_per_patch',
                'feature_contributions'
            ]:
                if key in layer_debug['feature_gating']:
                    # Ensure it's a proper numpy array with no tensor backing
                    layer_debug['feature_gating'][key] = np.array(layer_debug['feature_gating'][key])

    return (prediction_result_dict, attribution_pos_np, raw_patch_map, extra_info)


def generate_attribution_prisma_enhanced(
    model: HookedSAEViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],
    device: Optional[torch.device] = None,
    steering_resources: Optional[Dict[int, Dict[str, Any]]] = None,
    enable_steering: bool = True,
    enable_feature_gradients: bool = True,  # NEW
    feature_gradient_layers: Optional[List[int]] = None,  # NEW
    clip_classifier: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate attribution with both feature boosting and gradient gating.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = input_tensor.to(device)

    (pred_dict, pos_attr_np, raw_patch_map, extra_info) = transmm_prisma_enhanced(
        model_prisma=model,
        input_tensor=input_tensor,
        steering_resources=steering_resources,
        config=config,
        enable_steering=enable_steering,
        enable_feature_gradients=enable_feature_gradients,
        feature_gradient_layers=feature_gradient_layers,
        idx_to_class=idx_to_class,
        clip_classifier=clip_classifier,
    )

    return {
        "predictions": pred_dict,
        "attribution_positive": pos_attr_np,
        "raw_attribution": raw_patch_map,
        "logits": None,
        "ffn_activity": [],
        "class_embedding_representation": [],
        "head_contribution": [],
        "extra_info": extra_info
    }
