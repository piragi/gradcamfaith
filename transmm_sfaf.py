# transmm_sfaf.py
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from vit_prisma.models.base_vit import HookedViT  # Import the new model class
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.models.weight_conversion import convert_timm_weights
from vit_prisma.sae import SparseAutoencoder

from build_boost_mask_improved import (
    build_additive_correction_mask, build_bias_multiplicative_mask, build_boost_mask_improved,
    precache_bias_multiplicative_features, precache_sorted_features
)
from config import PipelineConfig


def load_models():
    """Load SAE and fine-tuned model"""
    # Load model
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
    print(f"\n[DEBUG] load_steering_resources called from transmm_sfaf.py for layers {layers}")
    from memory_debug import print_memory_status
    print_memory_status("BEFORE_LOADING_SAES")
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
                sae_files = list(sae_dir.glob("*/n_images_*.pt"))
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
                print(f"Please run saco_feature_analysis_simple.py first to generate feature dictionaries")

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


def transmm_prisma(
    model_prisma: HookedViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],  # Required parameter - no default
    device: Optional[torch.device] = None,
    img_size: int = 224,
    steering_resources: Optional[Dict[int, Dict[str, Any]]] = None,  # CHANGED: Pass resources in
    enable_steering: bool = True,
    steering_strength: float = 1.5,
    clip_classifier: Optional[Any] = None,  # Optional CLIP classifier
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """
    TransMM with S_f/A_f based patch boosting on multiple layers.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We now receive the resources directly
    active_steering_layers = list(steering_resources.keys()) if enable_steering and steering_resources else []

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    model_prisma.reset_hooks()
    model_prisma.zero_grad()

    # Storage for gradients, activations, and now layer-specific SAE codes
    gradients = {}
    activations = {}
    sae_codes = {}  # Will store codes as {layer_idx: codes_tensor}

    attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(model_prisma.cfg.n_layers)]

    def save_activation_hook(tensor: torch.Tensor, hook: Any):
        activations[hook.name] = tensor.detach()

    def save_gradient_hook(grad: torch.Tensor, hook: Any):
        # Backward hook receives gradient as first argument
        if grad is not None:
            gradients[hook.name + "_grad"] = grad.detach().clone()

    fwd_hooks = [(name, save_activation_hook) for name in attn_hook_names]
    bwd_hooks = [(name, save_gradient_hook) for name in attn_hook_names]

    # NEW: Add residual hooks for all active steering layers
    if enable_steering and active_steering_layers:

        def make_resid_hook(layer_idx):

            def save_resid_and_codes(tensor, hook):
                resource = steering_resources[layer_idx]
                with torch.no_grad():
                    _, codes = resource["sae"].encode(tensor)
                    sae_codes[layer_idx] = codes.detach().clone()
                return tensor

            return save_resid_and_codes

        for layer_idx in active_steering_layers:
            resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
            fwd_hooks.append((resid_hook_name, make_resid_hook(layer_idx)))

    # Forward and backward pass (modified for CLIP)
    with model_prisma.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        if clip_classifier is not None:
            # CLIP classification path
            # Use CLIP classifier to get logits (with gradients enabled)
            clip_result = clip_classifier.forward(input_tensor, requires_grad=True)
            logits = clip_result["logits"]
            probabilities = clip_result["probabilities"]
            predicted_class_idx = clip_result["predicted_class_idx"]
        else:
            # Regular ViT classification path
            logits = model_prisma(input_tensor)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

        # Create one-hot for gradient computation
        num_classes = logits.size(-1)
        one_hot = torch.zeros((1, num_classes), dtype=torch.float32, device=device)
        one_hot[0, predicted_class_idx] = 1
        one_hot.requires_grad_(True)
        loss = torch.sum(one_hot * logits)
        loss.backward(retain_graph=True)  # Use retain_graph for CLIP

    prediction_result_dict = {
        "logits":
        logits.detach(),
        "probabilities":
        probabilities.squeeze().cpu().detach().numpy().tolist()
        if isinstance(probabilities, torch.Tensor) else probabilities,
        "predicted_class_idx":
        predicted_class_idx,
        "predicted_class_label":
        idx_to_class.get(predicted_class_idx, f"class_{predicted_class_idx}")
    }

    # Check if gradients were captured (for debugging)
    if not gradients:
        raise RuntimeError("No gradients captured! Check backward hook implementation.")

    # -------------- Attribution loop (MODIFIED) --------------
    num_tokens = activations[attn_hook_names[0]].shape[-1]
    R_pos = torch.eye(num_tokens, num_tokens, device='cpu')
    all_boosted_features = {}

    # Storage for correction masks from each layer (NEW)
    accumulated_correction_mask = None

    for i in range(model_prisma.cfg.n_layers):
        hname = f"blocks.{i}.attn.hook_pattern"
        grad = gradients[hname + "_grad"]  # No fallback - we need real gradients
        cam = activations[hname]
        cam_pos_avg = avg_heads(cam, grad)

        if i in active_steering_layers:
            # print(f"--- Applying steering for layer {i} ---")
            resources = steering_resources[i]
            codes_for_layer = sae_codes.get(i)

            if codes_for_layer is not None:

                if "saco_dict" in resources or "inference_dict" in resources:
                    # Get boosting parameters from config
                    boosting_config = config.classify.boosting

                    # Get parameters for bias multiplicative correction
                    correction_method = getattr(boosting_config, 'correction_method', 'clamped')
                    scale_factor = getattr(boosting_config, 'bias_scale_factor', 1.0)
                    min_abs_bias = getattr(boosting_config, 'min_abs_bias', 0.0)
                    aggregation = getattr(boosting_config, 'aggregation', 'geometric_mean')

                    # Build correction mask but DON'T APPLY IT YET
                    # Enable debug for first few images to see what's happening
                    import os
                    debug_mode = os.environ.get('DEBUG_BOOST', '0') == '1'

                    # Check if we have class-aware inference dict
                    if resources.get('use_class_aware', False) and 'inference_dict' in resources:
                        # Use new class-aware method
                        from build_boost_mask_improved import \
                            build_class_aware_mask

                        inference_dict = resources['inference_dict']
                        # Get predicted class name from the prediction
                        pred_class_idx = prediction_result_dict['predicted_class_idx']
                        pred_class_name = idx_to_class.get(pred_class_idx, f'class_{pred_class_idx}')

                        if debug_mode:
                            print(f"Using class-aware correction for {pred_class_name}")

                        correction_mask = build_class_aware_mask(
                            codes_for_layer[0, 1:],  # Remove CLS token
                            inference_dict,
                            pred_class_name,
                            top_L_per_patch=5,
                            strength_k=boosting_config.bias_scale_factor,
                            clamp_min=0.3,
                            clamp_max=3.0,
                            debug=debug_mode
                        )

                        # Get some stats for debug info
                        selected_feat_ids = []  # Not tracked in new method
                        debug_info = {
                            'multiplier_range': (correction_mask.min().item(), correction_mask.max().item()),
                            'n_patches_boosted': (correction_mask > 1.1).sum().item(),
                            'n_patches_suppressed': (correction_mask < 0.9).sum().item()
                        }
                    elif "saco_dict" in resources:
                        # Use old method with saco_dict
                        saco_results = resources["saco_dict"]
                        correction_mask, selected_feat_ids, debug_info = build_bias_multiplicative_mask(
                            sae_codes=codes_for_layer,
                            saco_results=saco_results,
                            device=device,
                            min_activation=boosting_config.min_activation,
                            correction_method=correction_method,
                            scale_factor=scale_factor,
                            min_abs_bias=min_abs_bias,
                            min_occurrences=boosting_config.min_occurrences,
                            max_occurrences=boosting_config.max_occurrences,
                            topk_active=boosting_config.topk_active,
                            aggregation=aggregation,
                            debug=debug_mode
                        )
                    else:
                        # No dictionary available
                        print(f"Layer {i}: No feature dictionary available, skipping correction")
                        continue

                    if resources.get('use_class_aware', False):
                        print(f"Layer {i}: Built class-aware correction mask")
                        print(
                            f"  Multiplier range: [{debug_info['multiplier_range'][0]:.3f}, "
                            f"{debug_info['multiplier_range'][1]:.3f}]"
                        )
                        print(
                            f"  Patches boosted: {debug_info.get('n_patches_boosted', 0)}, "
                            f"suppressed: {debug_info.get('n_patches_suppressed', 0)}"
                        )
                    elif selected_feat_ids:
                        print(f"Layer {i}: Built correction mask using {len(selected_feat_ids)} features")
                        print(
                            f"  Multiplier range: [{debug_info['multiplier_range'][0]:.3f}, "
                            f"{debug_info['multiplier_range'][1]:.3f}]"
                        )

                        # Accumulate corrections from all layers
                        if accumulated_correction_mask is None:
                            accumulated_correction_mask = correction_mask.cpu()
                        else:
                            # Combine with geometric mean for multiple layers
                            accumulated_correction_mask = torch.sqrt(
                                accumulated_correction_mask * correction_mask.cpu()
                            )

                        all_boosted_features[i] = selected_feat_ids

                else:
                    print(f"ERROR: No SaCo dict or inference dict found for layer {i} - skipping steering")
                    continue  # Skip this layer if no dict available

        R_pos = R_pos + apply_self_attention_rules(R_pos, cam_pos_avg)

    transformer_attribution_pos = R_pos[0, 1:].clone()

    # Apply accumulated correction to final attribution (NEW)
    if accumulated_correction_mask is not None and enable_steering:
        print(f"\n=== Applying accumulated correction to final attribution ===")
        print(
            f"Correction range: [{accumulated_correction_mask.min():.3f}, "
            f"{accumulated_correction_mask.max():.3f}]"
        )
        print(f"Mean multiplier: {accumulated_correction_mask.mean():.3f}")

        # Store original for comparison
        original_max = transformer_attribution_pos.max().item()
        original_min = transformer_attribution_pos.min().item()
        original_std = transformer_attribution_pos.std().item()

        # Apply multiplicative correction - NO intermediate renormalization!
        # Let the final normalize_fn handle scaling to [0,1]
        transformer_attribution_pos = transformer_attribution_pos * accumulated_correction_mask

        # Check how much the distribution changed
        new_max = transformer_attribution_pos.max().item()
        new_min = transformer_attribution_pos.min().item()
        new_std = transformer_attribution_pos.std().item()

        print(f"Max: {original_max:.4f} -> {new_max:.4f} (×{new_max/original_max:.2f})")
        print(f"Min: {original_min:.4f} -> {new_min:.4f} (×{new_min/original_min:.2f})")
        print(f"Std: {original_std:.4f} -> {new_std:.4f} (×{new_std/original_std:.2f})")

        # The key metric: how much did we change the relative importance?
        if original_max > 0:
            print(f"Dynamic range: {(new_max - new_min)/(original_max - original_min):.2f}x")

    raw_patch_map = transformer_attribution_pos.cpu().numpy()  # shape (49,) for B-32 or (196,) for B-16

    # ============ RESHAPE AND NORMALIZE ============

    def process_attribution_map(attr_tensor: torch.Tensor) -> np.ndarray:
        side_len = int(np.sqrt(attr_tensor.size(0)))
        attr_tensor = attr_tensor.reshape(1, 1, side_len, side_len)
        attr_tensor_device = attr_tensor.to(device)
        attr_interpolated = F.interpolate(
            attr_tensor_device, size=(img_size, img_size), mode='bilinear', align_corners=False
        )
        return attr_interpolated.squeeze().cpu().detach().numpy()

    attribution_pos_np = process_attribution_map(transformer_attribution_pos)

    # Normalize
    normalize_fn = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8) if (np.max(x) - np.min(x)) > 1e-8 else x
    attribution_pos_np = normalize_fn(attribution_pos_np)

    # Clean up
    del transformer_attribution_pos, input_tensor, one_hot, loss
    del gradients, activations
    if 'codes' in sae_codes:
        del sae_codes
    torch.cuda.empty_cache()
    gc.collect()

    return (prediction_result_dict, attribution_pos_np, raw_patch_map)


def generate_attribution_prisma(
    model: HookedSAEViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],  # Required parameter - no default
    device: Optional[torch.device] = None,
    steering_resources: Optional[Dict[int, Dict[str, Any]]] = None,
    enable_steering: bool = True,
    clip_classifier: Optional[Any] = None,  # Optional CLIP classifier wrapper
) -> Dict[str, Any]:
    """
    Generate attribution with S_f/A_f based steering.
    
    Args:
        model: The hooked SAE ViT model
        input_tensor: Input image tensor
        config: Pipeline configuration
        idx_to_class: Required mapping from class indices to class names
        device: Device to run on
        steering_resources: Resources for steering
        enable_steering: Whether to enable steering
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = input_tensor.to(device)

    (pred_dict, pos_attr_np, raw_patch_map) = transmm_prisma(
        model_prisma=model,
        input_tensor=input_tensor,
        steering_resources=steering_resources,
        config=config,
        enable_steering=enable_steering,
        idx_to_class=idx_to_class,
        clip_classifier=clip_classifier,
    )

    # Structure output
    return {
        "predictions": pred_dict,
        "attribution_positive": pos_attr_np,
        "raw_attribution": raw_patch_map,
        "logits": None,
        "ffn_activity": [],
        "class_embedding_representation": [],
        "head_contribution": []
    }
