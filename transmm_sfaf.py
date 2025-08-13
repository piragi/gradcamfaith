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

from config import PipelineConfig
# Removed hardcoded IDX2CLS import - will use dataset-specific mapping
from build_boost_mask_improved import build_boost_mask_improved, precache_sorted_features

SAE_CONFIG = {
    1: {
        "sae_path": "models/sweep/sae_l1_k128_exp64_lr0.0002/48a0f474-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/sfaf_stealth_l2_alignment_min1_128k64.pt"
    },
    2: {
        "sae_path": "./models/sweep/sae_l2_k64_exp64_lr2e-05/92bcc2fc-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_l2_alignment_min1_128k64.pt"
    },
    3: {
        "sae_path": "./models/sweep/sae_l3_k64_exp64_lr2e-05/99defb16-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_l3_alignment_min1_128k64.pt"
    },
    4: {
        "sae_path": "./models/sweep/sae_l4_k64_exp64_lr2e-05/24d1b962-vit_covidquex_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_l4_alignment_min1_128k64.pt"
    },
    5: {
        "sae_path": "./models/sweep/sae_l5_k64_exp64_lr2e-05/d9216a1a-vit_covidquex_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_local_l5_alignment_min1_128k64.pt"
    },
    6: {
        # "sae_path": "data/sae_covidquex/layer_6/b562ac30-vit_unified_sae/n_images_65161.pt",
        # "dict_path": "./sae_dictionaries/steer_corr_local_l6_alignment_min1_128k64.pt",
        # "saco_dict_path": "data/featuredict_covidquex/layer_6_saco_features.pt"  # New location with correct format
        "sae_path": "data/sae_hyperkvasir/layer_6/7767f4dc-vit_unified_sae/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_local_l6_alignment_min1_128k64.pt",
        "saco_dict_path": "data/featuredict_hyperkvasir/layer_6_saco_features.pt"  # New location with correct format
    },
    7: {
        "sae_path": "./models/sweep/sae_l7_k64_exp64_lr2e-05/1a690d9c-vit_covidquex_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_local_l7_alignment_min1_64k64.pt",
        "saco_dict_path": "./results/saco_features_direct_l7.pt"
        # "saco_dict_path": "./results/refiltered_saco_features_min_occ_30.pt"
        # "saco_dict_path": "./results/saco_problematic_features_bins_l7.pt"

    },
    8: {
        "sae_path": "./models/sweep/sae_l8_k64_exp64_lr2e-05/e23b1351-vit_covidquex_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_local_l8_alignment_min1_128k64.pt",
        "saco_dict_path": "./results/saco_features_direct_l8.pt"
    },
    9: {
        "sae_path": "./models/sweep/sae_l9_k64_exp64_lr2e-05/9058e0a1-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_local_l9_alignment_min1_128k64.pt"
    },
    10: {
        "sae_path": "./models/sweep/sae_l10_k64_exp64_lr2e-05/d25db388-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_l10_alignment_min1_128k64.pt"
    },
    11: {
        "sae_path": "models/sweep/sae_l11_k128_exp64_lr0.0002/a29c74a6-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_l11_alignment_min1_128k64.pt"
    },
}


def load_models():
    """Load SAE and fine-tuned model"""
    # Load model
    model = HookedSAEViT.from_pretrained("vit_base_patch16_224")
    model.head = torch.nn.Linear(model.cfg.d_model, 3)

    checkpoint = torch.load(
        "./model/model_best.pth.tar"
    )
    state_dict = checkpoint['state_dict'].copy()

    if 'lin_head.weight' in state_dict:
        state_dict['head.weight'] = state_dict.pop('lin_head.weight')
    if 'lin_head.bias' in state_dict:
        state_dict['head.bias'] = state_dict.pop('lin_head.bias')

    converted_weights = convert_timm_weights(state_dict, model.cfg)
    model.load_state_dict(converted_weights)
    model.cuda().eval()

    return model


def load_steering_resources(layers: List[int]) -> Dict[int, Dict[str, Any]]:
    """Loads SAEs and S_f/A_f dictionaries for the specified layers from SAE_CONFIG."""
    resources = {}
    for layer_idx in layers:
        if layer_idx not in SAE_CONFIG:
            print(f"Warning: No configuration found for layer {layer_idx}. Skipping.")
            continue

        config = SAE_CONFIG[layer_idx]
        sae_path = Path(config["sae_path"])
        dict_path = Path(config["dict_path"])

        if not sae_path.exists():
            print(f"Warning: SAE file not found for layer {layer_idx} at {sae_path}. Skipping.")
            continue

        try:
            sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
            sae.cuda().eval()

            resources[layer_idx] = {"sae": sae}
            
            # Load SaCo dictionary if available
            if "saco_dict_path" in config:
                saco_dict_path = Path(config["saco_dict_path"])
                if saco_dict_path.exists():
                    saco_results = torch.load(saco_dict_path, weights_only=False)
                    resources[layer_idx]["saco_dict"] = saco_results  # Load once, store in memory
                    print(f"Loaded SaCo dictionary for layer {layer_idx}: {saco_dict_path}")
                    
                    # Pre-cache sorted features for performance
                    try:
                        precache_sorted_features(saco_results)
                    except Exception as e:
                        print(f"Warning: Could not pre-cache features: {e}")
                else:
                    print(f"Warning: SaCo dict path specified but file not found: {saco_dict_path}")
            
        except Exception as e:
            print(f"Error loading resources for layer {layer_idx}: {e}")

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

    def save_gradient_hook(tensor: torch.Tensor, hook: Any):
        gradients[hook.name + "_grad"] = tensor.detach().clone()

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

    # Forward and backward pass (unchanged)
    with model_prisma.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        logits = model_prisma(input_tensor)
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        one_hot = torch.zeros((1, logits.size(-1)), dtype=torch.float32, device=device)
        one_hot[0, predicted_class_idx] = 1
        one_hot.requires_grad_(True)
        loss = torch.sum(one_hot * logits)
        loss.backward()

    prediction_result_dict = {
        "logits": logits.detach(),
        "probabilities": probabilities.squeeze().cpu().detach().numpy().tolist(),
        "predicted_class_idx": predicted_class_idx,
        "predicted_class_label": idx_to_class.get(predicted_class_idx, f"class_{predicted_class_idx}")
    }

    # -------------- Attribution loop (MODIFIED) --------------
    num_tokens = activations[attn_hook_names[0]].shape[-1]
    R_pos = torch.eye(num_tokens, num_tokens, device='cpu')
    all_boosted_features = {}

    for i in range(model_prisma.cfg.n_layers):
        hname = f"blocks.{i}.attn.hook_pattern"
        grad = gradients[hname + "_grad"]
        cam = activations[hname]
        cam_pos_avg = avg_heads(cam, grad)

        if i in active_steering_layers:
            print(f"--- Applying steering for layer {i} ---")
            resources = steering_resources[i]
            codes_for_layer = sae_codes.get(i)

            if codes_for_layer is not None:
                
                # ===== RANDOM BASELINE BOOST =====
                if "saco_dict" in resources:
                    # Toggle between random baseline and SACO-based method
                    use_random_baseline = False  # Set to False to use SACO method
                    
                    if use_random_baseline:
                        print(f"Using RANDOM BASELINE boosting for layer {i}")
                        boost_mask, selected_feat_ids = build_boost_mask_random(
                            sae_codes=codes_for_layer,
                            device=device,
                            suppress_strength=0.2,  # Match improved method
                            boost_strength=5.0,      # Match improved method
                            top_k_suppress=5,       # Match improved method (max_suppress)
                            top_k_boost=15,          # Match improved method (max_boost)
                            min_activation=0.05,     # Match improved method
                            seed=42,  # For reproducibility
                            debug=False
                        )
                    else:
                        # ===== IMPROVED FEATURE-BASED BOOST =====
                        print(f"Using SACO-BASED boosting for layer {i}")
                        saco_results = resources["saco_dict"]
                        boost_mask, selected_feat_ids = build_boost_mask_improved(
                            sae_codes=codes_for_layer,
                            saco_results=saco_results,
                            predicted_class=predicted_class_idx,
                            idx_to_class=idx_to_class,
                            device=device,
                            debug=False
                        )
                    
                else:
                    print(f"No SaCo dict found for layer {i}, using random baseline")
                    boost_mask, selected_feat_ids = build_boost_mask_random(
                        sae_codes=codes_for_layer,
                        device=device,
                        suppress_strength=0.5,
                        boost_strength=2.0,
                        top_k_suppress=10,
                        top_k_boost=8,
                        min_activation=0.05,
                        debug=False
                    )

                if selected_feat_ids:
                    print(f"Predicted class. {idx_to_class.get(predicted_class_idx, f'class_{predicted_class_idx}')}")
                    print(f"Layer {i}: Boosting based on {len(selected_feat_ids)} features: {selected_feat_ids}")
                    cam_pos_avg[0, 1:] *= boost_mask.cpu()
                    all_boosted_features[i] = selected_feat_ids

        R_pos = R_pos + apply_self_attention_rules(R_pos, cam_pos_avg)

    transformer_attribution_pos = R_pos[0, 1:].clone()
    raw_patch_map = transformer_attribution_pos.cpu().numpy()  # shape (196,)

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




@torch.no_grad()
def build_boost_mask_random(
    sae_codes: torch.Tensor,
    device: torch.device,
    *,
    suppress_strength: float = 0.5,
    boost_strength: float = 2.0,
    min_activation: float = 0.05,
    top_k_suppress: int = 10,
    top_k_boost: int = 8,
    seed: Optional[int] = None,
    debug: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Random feature boosting/suppression for baseline comparison.
    Randomly selects features that are active in the current sample.
    
    Args:
        sae_codes: SAE feature codes [1+T, k]
        device: Device to run on
        suppress_strength: Suppression strength (< 1.0)
        boost_strength: Boost strength (> 1.0)
        min_activation: Minimum activation threshold
        top_k_suppress: Number of features to suppress
        top_k_boost: Number of features to boost
        seed: Random seed for reproducibility
        debug: Print debug information
    
    Returns:
        boost_mask: Multiplicative mask for patches
        selected_features: List of selected feature IDs
    """
    codes = sae_codes[0, 1:].to(device)  # Remove CLS token
    n_patches, n_feats = codes.shape
    boost_mask = torch.ones(n_patches, device=device)
    selected_features = []
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        import random
        random.seed(seed)
    
    # Vectorized: Find all active features in this sample
    active_mask = (codes > min_activation).any(dim=0)  # [n_feats]
    active_features = active_mask.nonzero(as_tuple=True)[0]  # Tensor of active feature indices
    
    if len(active_features) == 0:
        if debug:
            print("No active features found for random boosting")
        return boost_mask, selected_features
    
    # Randomly sample features for suppression and boosting (vectorized)
    n_suppress = min(top_k_suppress, len(active_features))
    n_boost = min(top_k_boost, len(active_features))
    n_total = min(n_suppress + n_boost, len(active_features))
    
    if n_total > 0:
        # Single random permutation for both suppress and boost
        perm = torch.randperm(len(active_features))[:n_total]
        selected_feat_indices = active_features[perm]
        
        # Split into suppress and boost
        suppress_indices = selected_feat_indices[:n_suppress] if n_suppress > 0 else torch.tensor([], dtype=torch.long, device=device)
        boost_indices = selected_feat_indices[n_suppress:n_suppress + n_boost] if n_boost > 0 else torch.tensor([], dtype=torch.long, device=device)
        
        # Vectorized suppression
        if len(suppress_indices) > 0:
            suppress_activations = codes[:, suppress_indices]  # [n_patches, n_suppress]
            suppress_masks = suppress_strength + (1.0 - suppress_strength) * (1.0 - suppress_activations.clamp(0, 1))
            boost_mask *= suppress_masks.prod(dim=1)  # Multiply all suppression effects
            
            if debug:
                for i, feat_id in enumerate(suppress_indices):
                    n_active = (suppress_activations[:, i] > min_activation).sum().item()
                    print(f"  SUPPRESS feature {feat_id.item()} (random): active_patches={n_active}")
        
        # Vectorized boosting  
        if len(boost_indices) > 0:
            boost_activations = codes[:, boost_indices]  # [n_patches, n_boost]
            boost_masks = 1.0 + boost_activations.clamp(0, 1) * (boost_strength - 1.0)
            boost_mask *= boost_masks.prod(dim=1)  # Multiply all boost effects
            
            if debug:
                for i, feat_id in enumerate(boost_indices):
                    n_active = (boost_activations[:, i] > min_activation).sum().item()
                    print(f"  BOOST feature {feat_id.item()} (random): active_patches={n_active}")
        
        # Convert to list for return value
        selected_features = selected_feat_indices.cpu().tolist()
    
    if debug:
        total_active = len(active_features)
        print(f"Random mask: {len(selected_features)} total features selected "
              f"(from {total_active} active features)")
    
    return boost_mask, selected_features


def generate_attribution_prisma(
    model: HookedSAEViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],  # Required parameter - no default
    device: Optional[torch.device] = None,
    steering_resources: Optional[Dict[int, Dict[str, Any]]] = None,
    enable_steering: bool = True,
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
