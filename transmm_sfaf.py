# transmm_sfaf.py
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from vit_prisma.models.base_vit import HookedViT  # Import the new model class
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.models.weight_conversion import convert_timm_weights
from vit_prisma.sae import SparseAutoencoder

import vit.model as model_handler  # You might need to adapt or replace this too
from config import PipelineConfig
from vit.model import IDX2CLS
from vit.preprocessing import get_processor_for_precached_224_images

SAE_CONFIG = {
    1: {
        "sae_path": "models/sweep/sae_l1_k128_exp64_lr0.0002/48a0f474-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/sfaf_stealth_l2_alignment_min1_128k64.pt"
        #"dict_path": "./sae_dictionaries/sfaf_stealth_l1_alignment_min3_128k64.pt"
    },
    2: {
        # "sae_path": "models/sweep/sae_l2_k128_exp64_lr0.0002/41db76e2-vit_medical_sae_k_sweep/n_images_49276.pt",
        "sae_path": "./models/sweep/sae_l2_k64_exp64_lr2e-05/92bcc2fc-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_l2_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/sfaf_stealth_l2_alignment_min3_128k64.pt"
    },
    3: {
        # "sae_path": "models/sweep/sae_l3_k128_exp64_lr0.0002/6fd8fb1a-vit_medical_sae_k_sweep/n_images_49276.pt",
        "sae_path": "./models/sweep/sae_l3_k64_exp64_lr2e-05/99defb16-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_l3_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/sfaf_stealth_l3_alignment_min3_128k64.pt"
    },
    4: {
        # "sae_path": "models/sweep/sae_l4_k128_exp64_lr0.0002/20673e0c-vit_medical_sae_k_sweep/n_images_49276.pt",
        "sae_path": "./models/sweep/sae_l4_k64_exp64_lr2e-05/72421a1a-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_l4_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/sfaf_stealth_l4_alignment_min3_128k64.pt"
    },
    5: {
        # "sae_path": "models/sweep/sae_l5_k128_exp64_lr0.0002/e7fdbb62-vit_medical_sae_k_sweep/n_images_49276.pt",
        "sae_path": "./models/sweep/sae_l5_k64_exp64_lr2e-05/09853b08-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_local_l5_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/steer_corr_l5_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/sfaf_stealth_l5_alignment_min3_128k64.pt"
    },
    6: {
        # "sae_path": "models/sweep/sae_l6_k128_exp64_lr0.0002/becaec1e-vit_medical_sae_k_sweep/n_images_49276.pt",
        "sae_path": "./models/sweep/sae_l6_k64_exp64_lr2e-05/81ea5ed2-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_local_l6_alignment_min1_128k64.pt",
        # "dict_path": "./sae_dictionaries/steer_corr_l6_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/sfaf_stealth_l6_alignment_min3_128k64.pt"
        "saco_dict_path": "./results/saco_problematic_features_bins_l6.pt"
    },
    7: {
        # "sae_path": "models/sweep/sae_l7_k64_exp64_lr0.0002/21922d4b-vit_medical_sae_k_sweep/n_images_49276.pt",
        "sae_path": "./models/sweep/sae_l7_k64_exp64_lr2e-05/31a0aa2d-vit_medical_sae_k_sweep/n_images_49276.pt",
        # "sae_path": "./models/sweep/sae_l7_k128_exp32_lr2e-05/d77c1ce8-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_local_l7_alignment_min1_64k64.pt"
        # "dict_path": "./sae_dictionaries/steer_corr_local_l7_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/steer_corr_l7_alignment_min1_32k128.pt"
        # "dict_path": "./sae_dictionaries/sfaf_stealth_l7_alignment_min1_32k128.pt"
    },
    8: {
        # "sae_path": "models/sweep/sae_l8_k128_exp64_lr0.0002/dc5d1afd-vit_medical_sae_k_sweep/n_images_49276.pt",
        "sae_path": "./models/sweep/sae_l8_k64_exp64_lr2e-05/a43b7675-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_local_l8_alignment_min1_128k64.pt",
        # "dict_path": "./sae_dictionaries/steer_corr_l8_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/sfaf_stealth_l8_alignment_min3_128k64.pt"
        # "saco_dict_path": "./results/saco_problematic_features_l8_moderate.pt"  # NEW: SaCo results
        "saco_dict_path": "./results/saco_problematic_features_bins_l8.pt"
    },
    9: {
        # "sae_path": "models/sweep/sae_l9_k128_exp64_lr0.0002/e06c6b1d-vit_medical_sae_k_sweep/n_images_49276.pt",
        "sae_path": "./models/sweep/sae_l9_k64_exp64_lr2e-05/9058e0a1-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_local_l9_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/steer_corr_l9_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/sfaf_stealth_l9_alignment_min3_128k64.pt"
    },
    10: {
        # "sae_path": "models/sweep/sae_l10_k128_exp64_lr0.0002/4ade9e1f-vit_medical_sae_k_sweep/n_images_49276.pt",
        "sae_path": "./models/sweep/sae_l10_k64_exp64_lr2e-05/d25db388-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_l10_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/sfaf_stealth_l10_alignment_min3_128k64.pt"
    },
    11: {
        "sae_path": "models/sweep/sae_l11_k128_exp64_lr0.0002/a29c74a6-vit_medical_sae_k_sweep/n_images_49276.pt",
        "dict_path": "./sae_dictionaries/steer_corr_l11_alignment_min1_128k64.pt"
        # "dict_path": "./sae_dictionaries/sfaf_stealth_l11_alignment_min3_128k64.pt"
    },
}


def load_models():
    """Load SAE and fine-tuned model"""
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
        if not dict_path.exists():
            print(f"Warning: S_f/A_f dictionary not found for layer {layer_idx} at {dict_path}. Skipping.")
            continue

        try:
            sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
            sae.cuda().eval()
            sf_af_dict = torch.load(dict_path, weights_only=False)

            resources[layer_idx] = {"sae": sae, "dict": sf_af_dict}
            
            # Load SaCo dictionary if available
            if "saco_dict_path" in config:
                saco_dict_path = Path(config["saco_dict_path"])
                if saco_dict_path.exists():
                    saco_results = torch.load(saco_dict_path, weights_only=False)
                    resources[layer_idx]["saco_dict"] = saco_results  # Load once, store in memory
                    print(f"Loaded SaCo dictionary for layer {layer_idx}: {saco_dict_path}")
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
        print(input_tensor.shape)
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
        "predicted_class_label": IDX2CLS[predicted_class_idx]
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
                # ===== ORIGINAL BOOST METHODS (COMMENTED OUT) =====
                # boost_mask, selected_feat_ids = build_boost_mask_combined(
                #     sae_codes=codes_for_layer,
                #     alignment_dict=resources["dict"],
                #     predicted_class=predicted_class_idx,
                #     device=device,
                #     top_k=25,
                #     base_strength=10.,
                #     min_score=0.6,  # The combined score is 0-1, so this is a reasonable floor
                #     min_occurrences_for_class=1,
                #     debug=True
                # )
                
                # ===== RANDOM BASELINE BOOST =====
                if "saco_dict" in resources:
                    # print(f"Using random baseline boosting for layer {i}")
                    # boost_mask, selected_feat_ids = build_boost_mask_random(
                        # sae_codes=codes_for_layer,
                        # device=device,
                        # suppress_strength=0.3,
                        # boost_strength=10.0,
                        # top_k_suppress=0,
                        # top_k_boost=15,
                        # min_activation=0.05,
                        # seed=42,  # For reproducibility
                        # debug=True
                    # )
                    
                    # ===== SIMPLIFIED SACO-BASED BOOST =====
                    saco_results = resources["saco_dict"]  # Keep loading for potential future use
                    boost_mask, selected_feat_ids = build_boost_mask_saco_simple(
                    sae_codes=codes_for_layer,
                    saco_results=saco_results,
                    device=device,
                    suppress_strength=0.3,
                    boost_strength=5,
                    top_k_suppress=8,
                    top_k_boost=15,
                    min_activation=0.05,
                    debug=True
                    )
                    
                    # Option 2: Conservative suppression only
                    # boost_mask, selected_feat_ids = build_boost_mask_saco_suppress_only(
                        # sae_codes=codes_for_layer,
                        # saco_results=saco_results,
                        # predicted_class=predicted_class_idx,
                        # device=device,
                        # suppress_strength=0.7,
                        # top_k=15,
                        # overlap_threshold=0.7,
                        # debug=True
                    # )
                else:
                    print(f"No SaCo dict found for layer {i}, falling back to combined boost")
                    boost_mask, selected_feat_ids = build_boost_mask_combined(
                        sae_codes=codes_for_layer,
                        alignment_dict=resources["dict"],
                        predicted_class=predicted_class_idx,
                        device=device,
                        top_k=25,
                        base_strength=10.,
                        min_score=0.6,
                        min_occurrences_for_class=1,
                        debug=True
                    )
                # Option 2: Use correlation-based boosting (original)
                # boost_mask, selected_feat_ids = build_boost_mask_hybrid_tier(
                #     sae_codes=codes_for_layer,
                #     alignment_dict=resources["dict"],
                #     predicted_class=predicted_class_idx,
                #     device=device,
                #     layer_idx=i,  # ADD THIS
                #     debug=True  # To see what's being selected
                # )
                # boost_mask, selected_feat_ids = build_boost_mask_light(
                # sae_codes=codes_for_layer,
                # alignment_dict=resources["dict"],
                # predicted_class=predicted_class_idx,  # already here
                # device=device,
                # min_corr=0.55,
                # min_occ=1,  # keep minimal
                # top_k=6,
                # base_str=2.1
                # )
                # boost_mask, selected_feat_ids = build_boost_mask_hybrid(
                # sae_codes=codes_for_layer,
                # alignment_dict=resources["dict"],
                # predicted_class=predicted_class_idx,
                # device=device,
                # # You can now easily tune the strategy here
                # reliable_corr_thresh=0.35,
                # reliable_occ_thresh=20,
                # rare_corr_thresh=0.65,  # High bar for rare features!
                # top_k=5,
                # base_strength=2.0
                # )
                # boost_mask, selected_feat_ids = build_boost_mask_simple(
                # sae_codes=codes_for_layer,
                # alignment_dict=resources["dict"],
                # predicted_class=predicted_class_idx,
                # device=device,
                # top_k=15,
                # base_strength=1.2,
                # min_pfac_for_consideration=0.1,
                # min_occurrences_for_class=10
                # )
                #
                if selected_feat_ids:
                    print(f"Predicted class. {IDX2CLS[predicted_class_idx]}")
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
def build_boost_mask_hybrid_tier(
    sae_codes: torch.Tensor,
    alignment_dict: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    layer_idx: int,  # NEW: Need to know which layer we're in
    *,
    # --- Three-tier strategy ---
    # Super gems: Ultra-rare, ultra-high correlation
    super_gem_corr_thresh: float = 0.99,
    super_gem_occ_thresh: int = 5,
    super_gem_boost: float = 3.0,

    # Rare gems: Rare but highly correlated
    rare_corr_thresh: float = 0.99,
    rare_occ_range: Tuple[int, int] = (1, 10),
    rare_boost: float = 2.5,

    # Reliable workhorses: Common, moderately correlated
    reliable_corr_thresh: float = 0.2,
    reliable_occ_thresh: int = 1,
    reliable_boost: float = 5.,

    # General parameters
    top_k: int = 3,
    min_activation: float = 0.05,
    correlation_weight: bool = True,  # Weight boost by correlation strength
    debug: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Three-tier boost mask strategy:
    1. Super gems: Extremely rare (<5 occurrences) with very high correlation (>0.8)
    2. Rare gems: Rare (1-10 occurrences) with high correlation (>0.65)
    3. Reliable workhorses: Common (>20 occurrences) with moderate correlation (>0.35)
    """
    codes = sae_codes[0, 1:].to(device)
    n_patches, n_feats = codes.shape
    act_mask = (codes > min_activation)
    feat_any = act_mask.any(dim=0)

    if not feat_any.any():
        return torch.ones(n_patches, device=device), []

    # Pull pre-computed stats
    stats = alignment_dict["feature_stats"]
    pfac_means = torch.tensor([
        stats[f]['class_mean_pfac'].get(predicted_class, 0.0) if f in stats else 0.0 for f in range(n_feats)
    ],
                              device=device)

    class_counts = torch.tensor([
        stats[f]['class_count_map'].get(predicted_class, 0) if f in stats else 0 for f in range(n_feats)
    ],
                                device=device)

    # Layer-specific threshold adjustments
    if layer_idx <= 3:  # Early layers: be more lenient
        rare_corr_thresh *= 0.9
        reliable_corr_thresh *= 0.9
    elif layer_idx >= 8:  # Late layers: be more strict
        rare_corr_thresh *= 1.05
        reliable_corr_thresh *= 1.1

    # --- THREE-TIER FILTERING ---
    # Tier 1: Super gems
    is_super_gem = (
        feat_any & (class_counts <= super_gem_occ_thresh) & (class_counts > 0) & (pfac_means >= super_gem_corr_thresh)
    )

    # Tier 2: Rare gems
    is_rare_gem = (
        feat_any & (class_counts >= rare_occ_range[0]) & (class_counts <= rare_occ_range[1]) &
        (pfac_means >= rare_corr_thresh)
    )

    # Tier 3: Reliable workhorses
    is_reliable = (feat_any & (class_counts >= reliable_occ_thresh) & (pfac_means >= reliable_corr_thresh))

    # Create tier assignments for each feature
    feature_tiers = torch.zeros(n_feats, device=device)
    feature_tiers[is_super_gem] = 3
    feature_tiers[is_rare_gem & (feature_tiers == 0)] = 2  # Don't override super gems
    feature_tiers[is_reliable & (feature_tiers == 0)] = 1

    valid = feature_tiers > 0

    if not valid.any():
        return torch.ones(n_patches, device=device), []

    # Score features by tier and correlation
    valid_idx = valid.nonzero(as_tuple=True)[0]

    # Create composite scores: tier priority + correlation within tier
    tier_scores = feature_tiers[valid_idx]
    pfac_scores = pfac_means[valid_idx]

    # Composite score: heavily weight tier, then correlation
    composite_scores = tier_scores * 10 + pfac_scores

    # Select top-k by composite score
    k_top = min(top_k, valid_idx.size(0))
    top_scores, top_pos = torch.topk(composite_scores, k_top, sorted=True)
    selected_features = valid_idx[top_pos]
    selected_tiers = feature_tiers[selected_features]

    if debug:
        print(f"Layer {layer_idx} selection:")
        for i, (feat_id, tier, corr) in enumerate(
            zip(selected_features.tolist(), selected_tiers.tolist(), pfac_means[selected_features].tolist())
        ):
            tier_name = {3: "super gem", 2: "rare gem", 1: "reliable"}[int(tier)]
            count = class_counts[feat_id].item()
            print(f"  Feature {feat_id}: {tier_name}, corr={corr:.3f}, count={count}")

    # Build boost mask with tier-specific strengths
    boost_mask = torch.ones(n_patches, device=device)

    for feat_id, tier in zip(selected_features, selected_tiers):
        feat_act = act_mask[:, feat_id].float()

        # Determine boost strength based on tier
        if tier == 3:
            base_boost = super_gem_boost
        elif tier == 2:
            base_boost = rare_boost
        else:
            base_boost = reliable_boost

        # Optionally weight by correlation strength
        if correlation_weight:
            corr_factor = pfac_means[feat_id].item()
            # Scale boost: at threshold = 1.0x, at perfect correlation = 1.5x
            corr_multiplier = 1.0 + 0.5 * min((corr_factor - 0.3) / 0.7, 1.0)
            effective_boost = 1 + (base_boost - 1) * corr_multiplier
        else:
            effective_boost = base_boost

        # Apply multiplicative boost where feature is active
        patch_boost = 1 + feat_act * (effective_boost - 1)
        boost_mask *= patch_boost

    return boost_mask, selected_features.tolist()


@torch.no_grad()
def build_boost_mask_hybrid(
    sae_codes: torch.Tensor,
    alignment_dict: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    *,
    # --- Parameters for the new HYBRID strategy ---
    # For common, "reliable" features
    reliable_corr_thresh: float = 0.35,
    reliable_occ_thresh: int = 20,
    # For rare, "gem" features
    rare_corr_thresh: float = 0.65,  # Much higher bar for correlation
    # General parameters
    top_k: int = 10,
    base_strength: float = 2.0,
    min_activation: float = 0.05
) -> Tuple[torch.Tensor, List[int]]:
    """
    Builds a boost mask using a hybrid strategy that considers both
    reliable "workhorse" features and potent but rare "gem" features.
    """
    codes = sae_codes[0, 1:].to(device)
    n_patches, n_feats = codes.shape

    act_mask = (codes > min_activation)
    feat_any = act_mask.any(dim=0)
    if not feat_any.any():
        return torch.ones(n_patches, device=device), []

    # Pull pre-computed stats from the dictionary
    stats = alignment_dict["feature_stats"]
    pfac_means = torch.tensor([
        stats[f]['class_mean_pfac'].get(predicted_class, 0.0) if f in stats else 0.0 for f in range(n_feats)
    ],
                              device=device)
    class_counts = torch.tensor([
        stats[f]['class_count_map'].get(predicted_class, 0) if f in stats else 0 for f in range(n_feats)
    ],
                                device=device)

    # --- HYBRID FILTERING LOGIC ---
    # Condition 1: Is it a reliable workhorse? (High occurrence, medium correlation)
    is_reliable = (class_counts >= reliable_occ_thresh) & (pfac_means >= reliable_corr_thresh)

    # Condition 2: Is it a rare gem? (Low occurrence, very high correlation)
    is_gem = (class_counts < reliable_occ_thresh) & (pfac_means >= rare_corr_thresh)

    # A feature is valid if it's active AND it's either a workhorse OR a gem
    valid = feat_any & (is_reliable | is_gem)

    if not valid.any():
        return torch.ones(n_patches, device=device), []

    # Proceed with top-k selection from the valid features
    valid_idx = valid.nonzero(as_tuple=True)[0]
    pfac_valid = pfac_means[valid_idx]

    k_top = min(top_k, pfac_valid.size(0))
    top_vals, top_pos = torch.topk(pfac_valid, k_top, sorted=False)
    selected_features = valid_idx[top_pos]

    # Build the final boost mask
    sel_act = act_mask[:, selected_features].float()
    boost_mask = (1 + sel_act * (base_strength - 1)).prod(dim=1)

    return boost_mask, selected_features.tolist()


@torch.no_grad()
def build_boost_mask_simple(
    sae_codes: torch.Tensor,  # [1+T, k]
    alignment_dict: Dict[str, Any],  # your improved dict
    predicted_class: int,
    device: torch.device,
    *,
    top_k: int = 5,
    base_strength: float = 2.5,
    min_activation: float = 0.05,
    min_pfac_for_consideration: float = 0.10,
    min_occurrences_for_class: int = 3,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Vectorized boost–mask builder:
      1) Find all features with any activation > min_activation
      2) Vector‐gather per‐feature PFAC mean _for this class_ and occurrence counts
      3) Mask out those below thresholds, take top_k by PFAC
      4) Build one shot patch‐wise mask by a single tensor‐prod
    """
    # strip CLS token → [T, k]
    codes = sae_codes[0, 1:].to(device)  # [n_patches, n_feats]
    n_patches, n_feats = codes.shape

    # 1) which features fire at least once?
    act_mask = (codes > min_activation)  # [T, k]
    feat_any = act_mask.any(dim=0)  # [k]
    if not feat_any.any():
        return torch.ones(n_patches, device=device), []

    # 2) pull out the two per‐feature scalars we need: avg PFAC for this class, and count of occ.
    #    (we pre‐computed these into your dict at build time)
    stats = alignment_dict["feature_stats"]
    # build tensors of shape [k] by list‐comprehension then moving to device
    # this is still Python for the gather but only k ops, once per image
    pfac_means = torch.tensor([
        stats[f]['class_mean_pfac'].get(predicted_class, 0.0) if f in stats else 0.0 for f in range(n_feats)
    ],
                              device=device)
    class_counts = torch.tensor([
        stats[f]['class_count_map'].get(predicted_class, 0) if f in stats else 0 for f in range(n_feats)
    ],
                                device=device)

    # 3) mask out
    valid = feat_any \
            & (pfac_means >= min_pfac_for_consideration) \
            & (class_counts >= min_occurrences_for_class)
    if not valid.any():
        return torch.ones(n_patches, device=device), []

    # restrict to valid feature indices
    valid_idx = valid.nonzero(as_tuple=True)[0]  # [m]
    pfac_valid = pfac_means[valid_idx]  # [m]

    # top‐k by PFAC
    k_top = min(top_k, pfac_valid.size(0))
    top_vals, top_pos = torch.topk(pfac_valid, k_top, sorted=False)  # [k_top]
    selected = valid_idx[top_pos]  # [k_top]

    # 4) build boost mask in one go:
    #    for each selected feature, wherever it fires, multiply by base_strength
    #    →   mask = ∏₍f∈selected₎ [1 + (codes[:,f]>min_activation)*(base_strength-1)]
    sel_act = act_mask[:, selected].float()  # [T, k_top]
    boost_mask = (1 + sel_act * (base_strength - 1)).prod(dim=1)  # [T]

    return boost_mask, selected.tolist()


@torch.no_grad()
def build_boost_mask_combined(
    sae_codes: torch.Tensor,
    alignment_dict: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    *,
    top_k: int = 10,
    base_strength: float = 3.0,
    min_activation: float = 0.05,
    min_score: float = 0.3,  # Adjust based on new score distribution
    min_occurrences_for_class: int = 3,
    debug: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Builds a boost/suppress mask using the combined score. It selects top features
    based on the score (which considers absolute steerability) and then applies a
    boost for positive steerability or a suppression for negative steerability.
    """
    # ... (initial part of the function is the same) ...
    codes = sae_codes[0, 1:].to(device)
    n_patches, n_feats = codes.shape
    act_mask = (codes > min_activation)
    feat_any = act_mask.any(dim=0)
    if not feat_any.any(): return torch.ones(n_patches, device=device), []

    stats = alignment_dict.get("feature_stats", {})
    if not stats:
        if debug: print("Warning: 'feature_stats' not found.")
        return torch.ones(n_patches, device=device), []

    combined_scores_per_class = torch.tensor([
        stats[f].get('class_combined_scores', {}).get(predicted_class, 0.0) if f in stats else 0.0
        for f in range(n_feats)
    ],
                                             device=device)

    class_counts = torch.tensor([
        stats[f].get('class_count_map', {}).get(predicted_class, 0) if f in stats else 0 for f in range(n_feats)
    ],
                                device=device)

    valid = feat_any \
            & (combined_scores_per_class >= min_score) \
            & (class_counts >= min_occurrences_for_class)

    if not valid.any():
        if debug: print(f"No valid features found for class {predicted_class} with score >= {min_score}")
        return torch.ones(n_patches, device=device), []

    valid_idx = valid.nonzero(as_tuple=True)[0]
    scores_valid = combined_scores_per_class[valid_idx]

    k_top = min(top_k, scores_valid.size(0))
    top_vals, top_pos = torch.topk(scores_valid, k_top, sorted=False)
    selected_features = valid_idx[top_pos]

    # --- MODIFICATION START ---
    # We now need the *actual steerability values* (with sign) to decide the action.
    steerability_values = torch.tensor([
        stats[f].get('class_mean_steerability', {}).get(predicted_class, 0.0) if f in stats else 0.0
        for f in selected_features.tolist()
    ],
                                       device=device)

    if debug:
        print(f"Selected top {k_top} features for class {predicted_class} by combined score:")
        for i, feat_id in enumerate(selected_features.tolist()):
            score_val = combined_scores_per_class[feat_id].item()
            steer_val = steerability_values[i].item()
            action = "BOOST" if steer_val > 0 else "SUPPRESS" if steer_val < 0 else "NEUTRAL"
            print(f"  Feature {feat_id}: score={score_val:.3f}, steer={steer_val:.3f} -> {action}")

    # Build the final boost/suppress mask
    boost_mask = torch.ones(n_patches, device=device)
    suppress_strength = 1.0 / base_strength

    for i, feat_id in enumerate(selected_features):
        feat_act = act_mask[:, feat_id].float()  # [T]
        steer_val = steerability_values[i].item()

        if steer_val > 0:
            # Positive steerability -> BOOST (multiply by > 1)
            # This feature helps the class, so we amplify its contribution.
            patch_effect = 1.0 + feat_act * (base_strength - 1.0)
        elif steer_val < 0:
            # Negative steerability -> SUPPRESS (multiply by < 1)
            # This feature confuses the model, so we reduce its contribution.
            patch_effect = 1.0 + feat_act * (base_strength - 1.0)
            # patch_effect = 1.0 + feat_act * (suppress_strength - 1.0)
        else:
            # Steerability is zero, no effect.
            patch_effect = 1.0

        boost_mask *= patch_effect
    # --- MODIFICATION END ---

    return boost_mask, selected_features.tolist()


@torch.no_grad()
def build_boost_mask_steerability(
    sae_codes: torch.Tensor,
    alignment_dict: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    *,
    top_k: int = 5,
    base_strength: float = 2.0,
    min_activation: float = 0.05,
    min_steerability: float = 5.,
    min_occurrences_for_class: int = 1,
    debug: bool = True
) -> Tuple[torch.Tensor, List[int]]:
    """
    Builds a boost/suppress mask using the most steerable features per class.
    Features with positive steerability are boosted, negative steerability are suppressed.
    
    Steerability interpretation:
    - Positive: Feature hurts target class -> BOOST patches to emphasize negative evidence
    - Negative: Feature helps target class -> SUPPRESS patches to de-emphasize positive evidence
    
    Args:
        sae_codes: SAE feature codes [1+T, k]
        alignment_dict: Dictionary with feature stats including steerability
        predicted_class: Target class for boosting/suppressing
        device: Device to run on
        top_k: Number of top absolute steerable features to use
        base_strength: Multiplicative boost/suppress strength (>1.0)
        min_activation: Minimum activation threshold
        min_steerability: Minimum absolute steerability threshold for consideration
        min_occurrences_for_class: Minimum class occurrences for consideration
        debug: Print debug information
    
    Returns:
        boost_mask: Tensor to multiply with attribution map (>1 = boost, <1 = suppress)
        selected_features: List of selected feature IDs
    """
    # Strip CLS token → [T, k]
    codes = sae_codes[0, 1:].to(device)  # [n_patches, n_feats]
    n_patches, n_feats = codes.shape

    # 1) Find features with any activation > min_activation
    act_mask = (codes > min_activation)  # [T, k]
    feat_any = act_mask.any(dim=0)  # [k]
    if not feat_any.any():
        return torch.ones(n_patches, device=device), []

    # 2) Extract steerability and occurrence data for this class
    stats = alignment_dict["feature_stats"]

    # Build tensors of shape [k] for steerability and class counts
    steerability_means = torch.tensor([
        stats[f]['class_mean_steerability'].get(predicted_class, 0.0) if f in stats else 0.0 for f in range(n_feats)
    ],
                                      device=device)

    class_counts = torch.tensor([
        stats[f]['class_count_map'].get(predicted_class, 0) if f in stats else 0 for f in range(n_feats)
    ],
                                device=device)

    # 3) Filter features based on absolute steerability (can be positive or negative)
    abs_steerability = torch.abs(steerability_means)
    valid = feat_any \
            & (abs_steerability >= min_steerability) \
            & (class_counts >= min_occurrences_for_class)

    if not valid.any():
        if debug:
            print(f"No valid steerable features found for class {predicted_class}")
        return torch.ones(n_patches, device=device), []

    # 4) Select top-k by absolute steerability
    valid_idx = valid.nonzero(as_tuple=True)[0]  # [m]
    abs_steerability_valid = abs_steerability[valid_idx]  # [m]

    k_top = min(top_k, abs_steerability_valid.size(0))
    top_vals, top_pos = torch.topk(abs_steerability_valid, k_top, sorted=False)  # [k_top]
    selected = valid_idx[top_pos]  # [k_top]

    # Get the actual steerability values (with sign) for selected features
    selected_steerabilities = steerability_means[selected]  # [k_top]

    if debug:
        print(f"Selected top {k_top} steerable features for class {predicted_class}:")
        for i, feat_id in enumerate(selected.tolist()):
            steer_val = selected_steerabilities[i].item()
            count = class_counts[feat_id].item()
            action = "BOOST" if steer_val > 0 else "SUPPRESS"
            print(f"  Feature {feat_id}: steerability={steer_val:.3f}, count={count} -> {action}")

    # 5) Build boost/suppress mask based on sign of steerability
    boost_mask = torch.ones(n_patches, device=device)

    for i, feat_id in enumerate(selected):
        feat_act = act_mask[:, feat_id].float()  # [T]
        steer_val = selected_steerabilities[i].item()

        if steer_val < 0:
            # Positive steerability: boost (multiply by base_strength)
            patch_effect = 1 + feat_act * (base_strength - 1)
        else:
            # Negative steerability: suppress (divide by base_strength)
            suppress_strength = 1.0 / base_strength
            patch_effect = 1 + feat_act * (suppress_strength - 1)

        boost_mask *= patch_effect

    return boost_mask, selected.tolist()


# ==================== SACO-BASED BOOST FUNCTIONS ====================

@torch.no_grad()
def build_boost_mask_saco_bidirectional(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    *,
    suppress_strength: float = 0.7,  # Suppress over-attributed (multiply by this)
    boost_strength: float = 1.4,    # Boost under-attributed (multiply by this)
    min_activation: float = 0.01,
    top_k_suppress: int = 10,       # Top features to suppress per class
    top_k_boost: int = 5,           # Top features to boost per class
    class_specific_boost: bool = True,  # NEW: Use class-specific feature selection
    strict_class_filter: bool = False,  # NEW: Only use features that are dominant for target class
    debug: bool = False
) -> Tuple[torch.Tensor, Dict[str, List[int]]]:
    """
    Build boost/suppress mask using SaCo analysis results.
    
    Strategy:
    - SUPPRESS features that are over-attributed (negative weighted SaCo)
    - BOOST features that are under-attributed (positive weighted SaCo)
    """
    codes = sae_codes[0, 1:].to(device)  # Remove CLS token, shape: [n_patches, n_features]
    n_patches, n_feats = codes.shape
    
    # Initialize mask as all ones (no effect)
    boost_mask = torch.ones(n_patches, device=device)
    selected_features = {'suppress': [], 'boost': []}
    
    # Get results by patch type
    results_by_type = saco_results.get('results_by_type', {})
    
    # Process over-attributed features (SUPPRESS)
    over_attributed = results_by_type.get('over_attributed', {})
    if over_attributed:
        suppress_features = []
        
        # NEW: Class-specific feature selection
        if class_specific_boost:
            predicted_class_name = IDX2CLS.get(predicted_class, 'unknown')
            
            # Filter features that are relevant for this specific class
            class_relevant_features = []
            for feat_id, stats in over_attributed.items():
                class_dist = stats.get('class_distribution', {})
                if predicted_class_name in class_dist:
                    # Check strict class filter: only use features where target class is dominant
                    if strict_class_filter:
                        dominant_class = stats.get('dominant_class', '')
                        if dominant_class != predicted_class_name:
                            continue  # Skip features not dominated by target class
                    
                    # Score by how much this feature appears in the target class
                    class_frequency = class_dist.get(predicted_class_name, 0)
                    total_frequency = sum(class_dist.values())
                    class_ratio = class_frequency / max(total_frequency, 1)
                    
                    # Use raw SaCo score for ranking (don't dilute with class ratio)
                    raw_saco_score = stats['mean_weighted_saco']
                    class_relevant_features.append((feat_id, stats, raw_saco_score))
            
            # Sort by raw SaCo score and take top_k
            class_relevant_features.sort(key=lambda x: x[2])  # Most negative first
            selected_features_data = class_relevant_features[:top_k_suppress]
            
            if debug:
                print(f"Found {len(class_relevant_features)} class-relevant suppress features for {predicted_class_name}")
        else:
            # Original method: just take top global features
            selected_features_data = [(feat_id, stats, stats['mean_weighted_saco']) 
                                      for feat_id, stats in list(over_attributed.items())[:top_k_suppress]]
        
        for feat_id, stats, _ in selected_features_data:
            # Check if feature exists and is active in this sample
            if feat_id >= n_feats:
                continue
                
            feat_activations = codes[:, feat_id]
            active_mask = feat_activations > min_activation
            
            # Skip if feature is not active in this sample
            if not active_mask.any():
                continue
            
            # Apply suppression (class relevance already checked above)
            # Apply suppression where feature is active
            # More negative weighted_saco = stronger suppression
            saco_score = abs(stats['mean_weighted_saco'])
            adaptive_suppress = suppress_strength * min(1.0, saco_score)
            
            # Apply multiplicative suppression
            patch_suppression = 1.0 - (feat_activations * (1.0 - adaptive_suppress))
            boost_mask *= patch_suppression
            
            suppress_features.append(feat_id)
            
            if debug:
                n_active_patches = active_mask.sum().item()
                predicted_class_name = IDX2CLS.get(predicted_class, 'unknown')
                print(f"  Suppressing feature {feat_id}: saco={stats['mean_weighted_saco']:.3f}, "
                      f"class={stats['dominant_class']}, strength={adaptive_suppress:.3f}, "
                      f"active_patches={n_active_patches}")
        
        selected_features['suppress'] = suppress_features
    
    # Process under-attributed features (BOOST)
    under_attributed = results_by_type.get('under_attributed', {})
    if under_attributed:
        boost_features = []
        
        # NEW: Class-specific feature selection for boosting
        if class_specific_boost:
            predicted_class_name = IDX2CLS.get(predicted_class, 'unknown')
            
            # Filter features that are relevant for this specific class
            class_relevant_features = []
            for feat_id, stats in under_attributed.items():
                class_dist = stats.get('class_distribution', {})
                if predicted_class_name in class_dist:
                    # Check strict class filter: only use features where target class is dominant
                    if strict_class_filter:
                        dominant_class = stats.get('dominant_class', '')
                        if dominant_class != predicted_class_name:
                            continue  # Skip features not dominated by target class
                    
                    # Score by how much this feature appears in the target class
                    class_frequency = class_dist.get(predicted_class_name, 0)
                    total_frequency = sum(class_dist.values())
                    class_ratio = class_frequency / max(total_frequency, 1)
                    
                    # Use raw SaCo score for ranking (don't dilute with class ratio)
                    raw_saco_score = stats['mean_weighted_saco']
                    class_relevant_features.append((feat_id, stats, raw_saco_score))
            
            # Sort by raw SaCo score and take top_k (most positive first for boosting)
            class_relevant_features.sort(key=lambda x: x[2], reverse=True)
            selected_features_data = class_relevant_features[:top_k_boost]
            
            if debug:
                print(f"Found {len(class_relevant_features)} class-relevant boost features for {predicted_class_name}")
        else:
            # Original method: just take top global features
            selected_features_data = [(feat_id, stats, stats['mean_weighted_saco']) 
                                      for feat_id, stats in list(under_attributed.items())[:top_k_boost]]
        
        for feat_id, stats, _ in selected_features_data:
            # Check if feature exists and is active in this sample
            if feat_id >= n_feats:
                continue
                
            feat_activations = codes[:, feat_id]
            active_mask = feat_activations > min_activation
            
            # Skip if feature is not active in this sample
            if not active_mask.any():
                continue
            
            # Apply boosting (class relevance already checked above)
            # Apply boosting where feature is active
            # More positive weighted_saco = stronger boost
            saco_score = stats['mean_weighted_saco']
            adaptive_boost = 1.0 + (boost_strength - 1.0) * min(1.0, saco_score)
            
            # Apply multiplicative boost
            patch_boost = 1.0 + feat_activations * (adaptive_boost - 1.0)
            boost_mask *= patch_boost
            
            boost_features.append(feat_id)
            
            if debug:
                n_active_patches = active_mask.sum().item()
                print(f"  Boosting feature {feat_id}: saco={stats['mean_weighted_saco']:.3f}, "
                      f"class={stats['dominant_class']}, strength={adaptive_boost:.3f}, "
                      f"active_patches={n_active_patches}")
        
        selected_features['boost'] = boost_features
    
    if debug:
        # Count total active features for context
        total_active_features = (codes.abs() > min_activation).any(dim=0).sum().item()
        print(f"SaCo mask applied: {len(selected_features['suppress'])} suppressed, "
              f"{len(selected_features['boost'])} boosted features "
              f"(from {total_active_features} total active features)")
    
    return boost_mask, selected_features


@torch.no_grad()
def build_boost_mask_saco_suppress_only(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    *,
    suppress_strength: float = 0.6,  # Suppress over-attributed features
    min_activation: float = 0.05,
    top_k: int = 15,                # Top over-attributed features to suppress
    overlap_threshold: float = 0.7,  # Only suppress features with high overlap in problematic patches
    debug: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Conservative version that only suppresses over-attributed features.
    """
    codes = sae_codes[0, 1:].to(device)
    n_patches, n_feats = codes.shape
    boost_mask = torch.ones(n_patches, device=device)
    
    # Get over-attributed features
    over_attributed = saco_results.get('results_by_type', {}).get('over_attributed', {})
    if not over_attributed:
        if debug:
            print("No over-attributed features found in SaCo results")
        return boost_mask, []
    
    suppress_features = []
    predicted_class_name = IDX2CLS.get(predicted_class, 'unknown')
    
    # Sort by most problematic (most negative weighted_saco)
    sorted_features = sorted(over_attributed.items(), key=lambda x: x[1]['mean_weighted_saco'])
    
    for feat_id, stats in sorted_features[:top_k]:
        # Check if feature exists and is active in this sample first
        if feat_id >= n_feats:
            continue
            
        feat_activations = codes[:, feat_id]
        active_mask = feat_activations > min_activation
        
        # Skip if feature is not active in this sample
        if not active_mask.any():
            continue
            
        # Filter by overlap ratio - only suppress features highly concentrated in problematic patches
        if stats['mean_overlap_ratio'] < overlap_threshold:
            continue
            
        # Check class relevance
        class_dist = stats.get('class_distribution', {})
        if predicted_class_name not in class_dist and stats.get('dominant_class') != predicted_class_name:
            continue
            
        # Apply suppression for this active, relevant feature
        # Adaptive suppression based on how problematic the feature is
        saco_score = abs(stats['mean_weighted_saco'])
        overlap_ratio = stats['mean_overlap_ratio']
        
        # Stronger suppression for more problematic features with higher overlap
        adaptive_strength = suppress_strength * min(1.0, saco_score * overlap_ratio)
        
        # Apply suppression
        patch_suppression = 1.0 - feat_activations * (1.0 - adaptive_strength)
        boost_mask *= patch_suppression
        
        suppress_features.append(feat_id)
        
        if debug:
            n_active_patches = active_mask.sum().item()
            print(f"  Suppressing feature {feat_id}: saco={stats['mean_weighted_saco']:.3f}, "
                  f"overlap={stats['mean_overlap_ratio']:.3f}, class={stats['dominant_class']}, "
                  f"active_patches={n_active_patches}")
    
    if debug:
        # Count total active features for context
        total_active_features = (codes.abs() > min_activation).any(dim=0).sum().item()
        print(f"SaCo suppression applied to {len(suppress_features)} features "
              f"(from {total_active_features} total active features)")
    
    return boost_mask, suppress_features


@torch.no_grad()
def build_boost_mask_saco_under_attributed_focus(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    *,
    boost_strength: float = 8.0,        # Higher boost for focused strategy
    min_activation: float = 0.01,       # Much lower threshold to capture more active features
    top_k_boost: int = 20,              # More features to find the good ones
    class_specific_boost: bool = True,
    strict_class_filter: bool = False,   # Use looser signal-focused filter
    debug: bool = False
) -> Tuple[torch.Tensor, Dict[str, List[int]]]:
    """
    Focus ONLY on boosting under-attributed features with looser restrictions.
    Designed for classes with heavy under-attribution bias (pylorus, z-line, etc.).
    """
    codes = sae_codes[0, 1:].to(device)  # Remove CLS token
    n_patches, n_feats = codes.shape
    
    # Initialize mask as all ones (no effect)
    boost_mask = torch.ones(n_patches, device=device)
    selected_features = {'boost': []}
    
    # Get results by patch type - ONLY under-attributed
    results_by_type = saco_results.get('results_by_type', {})
    under_attributed = results_by_type.get('under_attributed', {})
    
    if not under_attributed:
        if debug:
            print("No under-attributed features found in SaCo results")
        return boost_mask, selected_features
    
    boost_features = []
    
    if class_specific_boost:
        predicted_class_name = IDX2CLS.get(predicted_class, 'unknown')
        
        # Filter features with LOOSER restrictions for more coverage
        class_relevant_features = []
        for feat_id, stats in under_attributed.items():
            class_dist = stats.get('class_distribution', {})
            if predicted_class_name in class_dist:
                
                # Calculate quality metrics
                class_count = class_dist.get(predicted_class_name, 0)
                total_occurrences = sum(class_dist.values())
                class_frequency = class_count / max(total_occurrences, 1)
                overlap_ratio = stats.get('mean_overlap_ratio', 0.0)
                saco_strength = abs(stats.get('mean_weighted_saco', 0.0))
                mean_activation = stats.get('mean_activation_in_problematic', 0.0)
                
                # ACTIVATION-FOCUSED filtering: prioritize features that will actually fire
                if strict_class_filter:
                    # Much simpler filter focused on activation potential and basic quality
                    if (mean_activation >= 2.0 and        # Must have decent activation in problematic patches
                        class_count >= 1 and              # Just needs to appear for this class
                        saco_strength >= 0.6):            # Moderate signal requirement
                        pass  # Feature passes activation-focused filter
                    else:
                        continue  # Skip features unlikely to be active
                
                # ACTIVATION-FOCUSED scoring: prioritize features that actually activate
                activation_score = mean_activation * saco_strength * (1 + class_frequency)
                class_relevant_features.append((feat_id, stats, activation_score))
        
        # Sort by activation score and take top_k (highest activation score first)
        class_relevant_features.sort(key=lambda x: x[2], reverse=True)
        selected_features_data = class_relevant_features[:top_k_boost]
        
        if debug:
            print(f"Found {len(class_relevant_features)} class-relevant boost features for {predicted_class_name}")
            print(f"Selected top {len(selected_features_data)} features with ACTIVATION-FOCUSED strategy")
    else:
        # Original method: just take top global features
        selected_features_data = [(feat_id, stats, stats['mean_weighted_saco']) 
                                  for feat_id, stats in list(under_attributed.items())[:top_k_boost]]
    
    for feat_id, stats, _ in selected_features_data:
        # Check if feature exists and is active in this sample
        if feat_id >= n_feats:
            continue
            
        feat_activations = codes[:, feat_id]
        active_mask = feat_activations > min_activation
        
        # Skip if feature is not active in this sample
        if not active_mask.any():
            continue
        
        # Apply boosting where feature is active
        saco_score = stats['mean_weighted_saco']
        adaptive_boost = 1.0 + (boost_strength - 1.0) * min(1.0, saco_score)
        
        # Apply multiplicative boost
        patch_boost = 1.0 + feat_activations * (adaptive_boost - 1.0)
        boost_mask *= patch_boost
        
        boost_features.append(feat_id)
        
        if debug:
            n_active_patches = active_mask.sum().item()
            print(f"  Boosting feature {feat_id}: saco={stats['mean_weighted_saco']:.3f}, "
                  f"class={stats['dominant_class']}, strength={adaptive_boost:.3f}, "
                  f"active_patches={n_active_patches}")
    
    selected_features['boost'] = boost_features
    
    if debug:
        # Count total active features for context
        total_active_features = (codes.abs() > min_activation).any(dim=0).sum().item()
        print(f"Under-attributed focus applied: {len(boost_features)} boosted features "
              f"(from {total_active_features} total active features)")
    
    return boost_mask, selected_features

# =====================================================================

@torch.no_grad()
def build_boost_mask_saco_simple(
    sae_codes: torch.Tensor,
    saco_results: Dict[str, Any],
    device: torch.device,
    *,
    suppress_strength: float = 0.5,  # Suppress over-attributed features
    boost_strength: float = 2.0,     # Boost under-attributed features  
    min_activation: float = 0.05,
    top_k_suppress: int = 10,        # Top over-attributed features to suppress
    top_k_boost: int = 8,            # Top under-attributed features to boost
    debug: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Simplified SaCo-based boosting without class filtering.
    Just uses the top features by confidence-adjusted score regardless of class.
    """
    codes = sae_codes[0, 1:].to(device)  # Remove CLS token
    n_patches, n_feats = codes.shape
    boost_mask = torch.ones(n_patches, device=device)
    selected_features = []
    
    results_by_type = saco_results.get('results_by_type', {})
    
    # Process over-attributed features (SUPPRESS) - no class filtering
    over_attributed = results_by_type.get('over_attributed', {})
    if over_attributed:
        # Sort by confidence score (already computed in analysis)
        sorted_features = sorted(
            over_attributed.items(), 
            key=lambda x: x[1].get('saco_adjusted_score', 0.0), 
            reverse=True
        )
        
        suppress_count = 0
        for feat_id, stats in sorted_features:
            if suppress_count >= top_k_suppress:
                break
                
            # Check if feature exists and is active
            if feat_id >= n_feats:
                continue
                
            feat_activations = codes[:, feat_id]
            active_mask = feat_activations > min_activation
            
            if not active_mask.any():
                continue
            
            # Apply suppression - simpler formula
            patch_suppression = suppress_strength + (1.0 - suppress_strength) * (1.0 - feat_activations.clamp(0, 1))
            boost_mask *= patch_suppression
            
            selected_features.append(feat_id)
            suppress_count += 1
            
            if debug:
                n_active = active_mask.sum().item()
                print(f"  SUPPRESS feature {feat_id}: conf_score={stats.get('saco_adjusted_score', 0):.3f}, "
                      f"active_patches={n_active}")
    
    # Process under-attributed features (BOOST) - no class filtering  
    under_attributed = results_by_type.get('under_attributed', {})
    if under_attributed:
        # Sort by confidence score 
        sorted_features = sorted(
            under_attributed.items(),
            key=lambda x: x[1].get('saco_adjusted_score', 0.0),
            reverse=True
        )
        
        boost_count = 0
        for feat_id, stats in sorted_features:
            if boost_count >= top_k_boost:
                break
                
            # Check if feature exists and is active
            if feat_id >= n_feats:
                continue
                
            feat_activations = codes[:, feat_id] 
            active_mask = feat_activations > min_activation
            
            if not active_mask.any():
                continue
            
            # Apply boost - simpler formula
            patch_boost = 1.0 + feat_activations.clamp(0, 1) * (boost_strength - 1.0)
            boost_mask *= patch_boost
            
            selected_features.append(feat_id)
            boost_count += 1
            
            if debug:
                n_active = active_mask.sum().item()
                print(f"  BOOST feature {feat_id}: conf_score={stats.get('saco_adjusted_score', 0):.3f}, "
                      f"active_patches={n_active}")
    
    if debug:
        total_active = (codes.abs() > min_activation).any(dim=0).sum().item()
        print(f"Simple SaCo mask: {len(selected_features)} total features selected "
              f"(from {total_active} active features)")
    
    return boost_mask, selected_features


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
    device: Optional[torch.device] = None,
    steering_resources: Optional[Dict[int, Dict[str, Any]]] = None,
    enable_steering: bool = True,
) -> Dict[str, Any]:
    """
    Generate attribution with S_f/A_f based steering.
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
