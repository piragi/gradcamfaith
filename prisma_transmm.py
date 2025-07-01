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
    sae_path = "./models/sweep/sae_k128_exp8_lr0.0002/1756558b-vit_medical_sae_k_sweep/n_images_49276.pt"
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


def find_class_specific_features(model, sae, n_batches=1000, aggregation='mean'):
    """Find features that activate strongly for specific classes
    
    Args:
        model: The ViT model
        sae: The Sparse Autoencoder
        n_batches: Number of batches to process
        aggregation: How to aggregate features across tokens ('mean', 'max', or 'sum')
    """
    class_feature_scores = torch.zeros(6, sae.cfg.d_sae).cuda()
    class_counts = torch.zeros(6).cuda()

    label_map = {2: 3, 3: 2}

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
            acts = cache[sae.cfg.hook_point]  # [batch, num_tokens, d_model] - ALL tokens now

            # Encode all tokens
            feature_acts = sae.encode(acts)[1]  # [batch, num_tokens, d_sae]

            # Average the feature activations across all tokens
            aggregated_features = feature_acts.sum(dim=1)  # [batch, d_sae]

            # Accumulate by class
            for j in range(imgs.shape[0]):
                label = labels[j].item()
                if label < 6:  # Safety check
                    class_feature_scores[label] += aggregated_features[j]
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


def _select_active_features(
    wanted: List[int],
    sae_codes: torch.Tensor,
    thresh: float = 0.0,
) -> List[int]:
    """
    Select all wanted features that have activations above the threshold.

    Returns list of active features or [] if none exceed thresh.
    """
    if not wanted:
        return []

    codes = sae_codes.detach()  # (B,T,F) or (B,F)

    # Extract activations for the wanted features only → (B, |wanted|)
    wanted_idx = torch.tensor(wanted, device=codes.device)
    wanted_codes = codes[:, wanted_idx]  # (B, W)

    # Take max over batch → (W,)
    max_per_feat = wanted_codes.max(dim=0).values

    # Find all features above threshold
    active_mask = max_per_feat > thresh

    if not active_mask.any():
        return []  # nothing active

    # Get indices of active features
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    active_features = [wanted[idx.item()] for idx in active_indices]

    # Print activation values for active features
    print(f"Active features: {active_features}")
    print(f"Activation values: {max_per_feat[active_mask].tolist()}")

    return active_features


def transmm_prisma(
    model_prisma: HookedViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    device: Optional[torch.device] = None,
    img_size: int = 224,
    sae: Optional[SparseAutoencoder] = None,
    discriminative_features: Optional[Dict[str, Any]] = None,
    enable_steering_enhancement: bool = True,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """
    Two-pass implementation of TransMM with multi-layer steering:
    1. First pass: Compute gradients from ORIGINAL model (no steering)
    2. Second pass: Apply steering at multiple layers (7-11) for enhanced attention
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Multi-layer steering configuration
    # Stronger steering at earlier layers for more propagation effect
    steering_layers = {
        6: {
            'multiplier': 5.0
        },
    }

    steering_options = {
        'layers': steering_layers,
        'feature_idx': "predicted_class",
        'reference_layer': 6  # Layer to use for feature selection
    }

    class_feature_map = {
        0: [
            3380, 1909, 887, 1648, 6047, 4369, 4355, 1201, 4049, 772, 2004, 4351, 3102, 5927, 5446, 2939, 1499, 985,
            5249, 5228, 3741, 108, 888, 242, 1631, 698, 3133, 430, 3874, 6033, 3580, 4833, 231, 3359, 1013, 1920, 168,
            3403, 5391, 129, 443, 3651, 2157, 3792, 5096, 4735, 5526, 2905, 4852, 620
        ],
        1: [
            3184, 1564, 5357, 3575, 3809, 48, 3882, 1730, 39, 529, 1199, 1451, 2188, 2189, 2198, 3047, 4409, 5280, 974,
            1725, 2054, 2252, 2586, 2618, 4765, 4857, 5088, 5101, 5780, 374, 1981, 2949, 1769, 2678, 5391, 2721, 4808,
            2766, 334, 3646, 5310, 5527, 5916, 207, 4635, 5586, 5618, 3794, 3442, 1695
        ],
        2: [
            2155, 3298, 1976, 175, 3572, 2811, 4123, 3874, 3557, 5234, 174, 1872, 1366, 378, 1320, 2426, 36, 4260, 2499,
            1924, 4137, 4945, 3325, 728, 1155, 2282, 585, 5526, 3802, 5691, 4417, 2905, 3102, 3594, 1114, 3637, 5665,
            1868, 5859, 1911, 2581, 1395, 484, 2937, 374, 430, 168, 2766, 3294, 3786
        ],
        3: [
            15, 4468, 4355, 231, 3129, 787, 5840, 3217, 4015, 4762, 1361, 3559, 5455, 4370, 429, 3209, 5228, 5460, 1977,
            2321, 422, 1817, 5123, 5990, 3514, 5905, 4699, 3904, 2363, 2108, 3516, 1939, 6051, 2326, 2371, 5986, 1666,
            1244, 175, 4094, 5564, 1573, 1978, 522, 2038, 760, 2976, 5849, 1422, 5106
        ],
        4: [
            4800, 4074, 2231, 1419, 4412, 1900, 2941, 1783, 3886, 96, 5662, 5689, 1281, 417, 2796, 2554, 2514, 1868,
            4063, 85, 6052, 5097, 5228, 2984, 985, 5975, 807, 5186, 1539, 4122, 635, 756, 2832, 5567, 4852, 4365, 3327,
            2352, 2023, 6042, 1884, 4917, 4493, 6129, 4307, 1366, 159, 174, 1173, 2326
        ],
        5: [
            1676, 3527, 2502, 1872, 4382, 137, 3218, 3794, 2698, 1824, 5180, 2322, 1725, 4925, 5437, 2207, 3325, 5871,
            997, 4678, 1451, 4880, 1132, 2630, 2856, 5883, 5803, 5315, 5795, 1856, 4934, 349, 417, 6042, 1336, 3055,
            4304, 487, 1911, 1449, 852, 5984, 5128, 537, 3427, 5119, 2991, 3393, 5707, 1173
        ],
    }

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    model_prisma.reset_hooks()
    model_prisma.zero_grad()

    # Storage for gradients and activations
    original_gradients = {}
    original_activations = {}
    enhanced_activations = {}
    gradient_info = {}

    # Define hook names
    attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(model_prisma.cfg.n_layers)]

    # ============ PASS 1: ORIGINAL MODEL (NO STEERING) ============
    print("Pass 1: Computing gradients from original model (no steering)...")

    # Hooks for first pass
    def save_activation_hook(tensor: torch.Tensor, hook: Any):
        original_activations[hook.name] = tensor.detach().clone()

    def save_gradient_hook(tensor: torch.Tensor, hook: Any):
        original_gradients[hook.name + "_grad"] = tensor.detach().clone()

    fwd_hooks_pass1 = [(name, save_activation_hook) for name in attn_hook_names]
    bwd_hooks_pass1 = [(name, save_gradient_hook) for name in attn_hook_names]

    # Add residual stream hooks for all steering layers
    resid_data = {}
    if sae and steering_options:
        for layer_idx in steering_layers.keys():
            resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
            resid_data[layer_idx] = {}

            def make_save_resid_activation(layer):

                def save_resid_activation(tensor: torch.Tensor, hook: Any):
                    resid_data[layer]['activation'] = tensor.detach().clone()

                return save_resid_activation

            def make_save_resid_gradient(layer):

                def save_resid_gradient(tensor: torch.Tensor, hook: Any):
                    resid_data[layer]['gradient'] = tensor.detach().clone()

                return save_resid_gradient

            fwd_hooks_pass1.append((resid_hook_name, make_save_resid_activation(layer_idx)))
            bwd_hooks_pass1.append((resid_hook_name, make_save_resid_gradient(layer_idx)))

    # Run first pass WITHOUT any steering
    with model_prisma.hooks(fwd_hooks=fwd_hooks_pass1, bwd_hooks=bwd_hooks_pass1):
        # Forward pass
        logits = model_prisma(input_tensor)
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

        # Create one-hot for predicted class
        one_hot = torch.zeros((1, logits.size(-1)), dtype=torch.float32, device=device)
        one_hot[0, predicted_class_idx] = 1
        one_hot.requires_grad_(True)

        loss = torch.sum(one_hot * logits)

        # Backward pass - this captures the TRUE gradients
        loss.backward()

    print(f"Predicted class: {predicted_class_idx}")

    # Build prediction dictionary
    prediction_result_dict = {
        "logits": logits,
        "probabilities": probabilities.squeeze().cpu().detach().numpy(),
        "predicted_class_idx": predicted_class_idx,
        "predicted_class_label": IDX2CLS[predicted_class_idx]
    }

    # ============ IDENTIFY FEATURES TO ENHANCE ============
    features_to_enhance = []
    features_by_layer = {}  # Track which features are active at each layer

    if sae and steering_options and steering_options.get('feature_idx') == "predicted_class":
        if not class_feature_map:
            raise ValueError("`class_feature_map` is required for 'predicted_class' steering.")

        # Get features for predicted class
        wanted_features = class_feature_map.get(predicted_class_idx, [])

        if wanted_features:
            # Use reference layer for initial feature selection
            ref_layer = steering_options.get('reference_layer', 9)
            if ref_layer in resid_data and 'activation' in resid_data[ref_layer]:
                _, activations = sae.encode(resid_data[ref_layer]['activation'])
                features_to_enhance = _select_active_features(wanted_features, activations[:, 0, :], thresh=0.0)
                print(f"Features to enhance: {features_to_enhance} (from {len(wanted_features)} candidates)")

                # Check feature activity at each steering layer
                for layer_idx in steering_layers.keys():
                    if layer_idx in resid_data and 'activation' in resid_data[layer_idx]:
                        _, layer_acts = sae.encode(resid_data[layer_idx]['activation'])
                        active_features = []
                        for feat in features_to_enhance:
                            if layer_acts[0, 0, feat] > 0.1:  # Check CLS token
                                active_features.append(feat)
                        features_by_layer[layer_idx] = active_features
                        print(f"Layer {layer_idx}: {len(active_features)} features active")

    # ============ PASS 2: ENHANCED ATTENTION (OPTIONAL) ============
    if enable_steering_enhancement and features_to_enhance:
        print(f"Pass 2: Computing enhanced attention with multi-layer steering...")

        # Clear previous hooks
        model_prisma.reset_hooks()

        # Hooks for second pass - only save attention
        def save_enhanced_activation(tensor: torch.Tensor, hook: Any):
            enhanced_activations[hook.name] = tensor.detach().clone()

        fwd_hooks_pass2 = [(name, save_enhanced_activation) for name in attn_hook_names]

        def choose_active_features(codes, wanted, top_k=20, act_thresh=0.1):
            """
            codes   : (B,T,F)   SAE codes of current layer
            wanted  : list[int] candidate feature ids for that class
            returns : 1-D tensor of at most `top_k` feature ids that are
                      (a) in `wanted` and (b) have |code| > act_thresh
                      somewhere in the image.
            """
            with torch.no_grad():
                if not wanted:
                    return torch.empty(0, dtype=torch.long, device=codes.device)
                wanted_mask = torch.zeros(codes.shape[-1], dtype=torch.bool, device=codes.device)
                wanted_mask[wanted] = True
                active = (codes.abs() > act_thresh).any(dim=(0, 1))
                feats = torch.nonzero(wanted_mask & active, as_tuple=False).squeeze(-1)
                if feats.numel() > top_k:
                    # take those with largest |CLS code|
                    cls_mag = codes[:, 0, feats].abs().mean(0)
                    feats = feats[torch.topk(cls_mag, top_k).indices]
                return feats

        def make_scaling_hook(feats, strength):
            """
            feats     : 1-D tensor of feature ids to boost
            strength  : float, e.g. 0.5  (= +50 %)
            returns   : a forward hook that scales those features
            """
            if feats.numel() == 0:
                # nothing to do – return identity hook
                return lambda x, *, hook=None: x

            W_sub = sae.W_dec[feats].to(device)  # (k, D_model)

            def hook_fn(resid, *, hook=None):
                codes = sae.encode(resid)[1]  # (B,T,F)
                code_sel = codes[..., feats]  # (B,T,k)
                # Δresid = strength · code_sel · W_sub
                resid += strength * torch.einsum('btk,kd->btd', code_sel, W_sub)
                return resid

            return hook_fn

        # ------------------------------------------------------------------
        # --- build hooks for each steering layer -------------------------
        # ------------------------------------------------------------------
        top_k = 20  # keep at most 20 features per image
        layer_strength = 0.5  # +50 % of current activation

        for layer_idx, cfg_layer in steering_layers.items():
            # we reuse the strength from the dict if you set one
            strength = cfg_layer.get('multiplier', layer_strength)

            resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"

            # Get the codes *once* to decide which features are active
            if layer_idx not in resid_data or 'activation' not in resid_data[layer_idx]:
                continue
            with torch.no_grad():
                _, layer_codes = sae.encode(resid_data[layer_idx]['activation'])

            feats_here = choose_active_features(layer_codes, features_to_enhance, top_k=top_k, act_thresh=0.1)

            print(f"Layer {layer_idx}: steering {len(feats_here)} features "
                  f"with strength {strength}")

            fwd_hooks_pass2.append((resid_hook_name, make_scaling_hook(feats_here, strength)))
        # ------------------------------------------------------------------

        # Run second pass with multi-layer steering (no gradients needed)
        with torch.no_grad():
            with model_prisma.hooks(fwd_hooks=fwd_hooks_pass2):
                steered_logits = model_prisma(input_tensor)
                steered_probs = F.softmax(steered_logits, dim=-1)

        # Log probability change from steering
        prob_change = steered_probs[0, predicted_class_idx] - probabilities[0, predicted_class_idx]
        print(f"Probability change from steering: {prob_change.item():.4f}")
    else:
        print("Pass 2: Skipped (using original attention)")
        enhanced_activations = original_activations.copy()

    # ============ GRADIENT ANALYSIS ============
    if sae and features_to_enhance:
        gradient_info = {
            'enhanced_features': features_to_enhance,
            'num_enhanced_features': len(features_to_enhance),
            'steering_layers': list(steering_layers.keys()),
            'features_by_layer': features_by_layer,
            'enhancement_applied': enable_steering_enhancement and bool(features_to_enhance)
        }

        # Analyze gradients at reference layer
        ref_layer = steering_options.get('reference_layer', 9)
        if ref_layer in resid_data and 'activation' in resid_data[ref_layer] and 'gradient' in resid_data[ref_layer]:
            with torch.no_grad():
                resid_act = resid_data[ref_layer]['activation'].cpu()
                resid_grad = resid_data[ref_layer]['gradient'].cpu()

                sae_device = next(sae.parameters()).device
                sae = sae.cpu()

                _, feature_acts = sae.encode(resid_act)
                cls_feature_acts = feature_acts[:, 0, :]
                cls_resid_grad = resid_grad[:, 0, :]

                # Gradient projection
                try:
                    if sae.W_enc.shape[0] == sae.cfg.d_sae:
                        grad_feature_projection = torch.matmul(cls_resid_grad, sae.W_enc.T)
                    else:
                        grad_feature_projection = torch.matmul(cls_resid_grad, sae.W_enc)
                except RuntimeError:
                    if sae.W_enc.shape[1] == cls_resid_grad.shape[1]:
                        grad_feature_projection = torch.einsum('bd,fd->bf', cls_resid_grad, sae.W_enc)
                    else:
                        grad_feature_projection = torch.einsum('bd,df->bf', cls_resid_grad, sae.W_enc)

                sae = sae.to(sae_device)

                # Per-feature analysis
                feature_gradient_info = {}
                for feature_idx in features_to_enhance:
                    grad_mag = grad_feature_projection[0, feature_idx].abs().item()
                    act_strength = cls_feature_acts[0, feature_idx].item()

                    feature_gradient_info[feature_idx] = {
                        'gradient_magnitude': grad_mag,
                        'activation_strength': act_strength,
                        'grad_to_act_ratio': grad_mag / (act_strength + 1e-8),
                    }

                gradient_info['per_feature_analysis'] = feature_gradient_info
                gradient_info['reference_layer'] = ref_layer

    # ============ ATTRIBUTION CALCULATION ============
    # CRITICAL: Use ORIGINAL gradients with (optionally) ENHANCED attention

    num_tokens = original_activations[attn_hook_names[0]].shape[-1]
    R_pos = torch.eye(num_tokens, num_tokens, device='cpu')

    for i in range(model_prisma.cfg.n_layers):
        hook_name = f"blocks.{i}.attn.hook_pattern"

        # ALWAYS use ORIGINAL gradients
        grad = original_gradients[hook_name + "_grad"]

        # Choose attention: enhanced or original
        if enable_steering_enhancement and hook_name in enhanced_activations:
            cam = enhanced_activations[hook_name]
        else:
            cam = original_activations[hook_name]

        # Compute attribution
        cam_pos_avg = avg_heads(cam, grad)
        R_pos = R_pos + apply_self_attention_rules(R_pos, cam_pos_avg)

    # Extract attribution for patch tokens from the CLS token's relevance
    transformer_attribution_pos = R_pos[0, 1:].clone()
    del R_pos

    # ============ RESHAPING AND NORMALIZATION ============
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

    # ============ CLEANUP ============
    del transformer_attribution_pos, input_tensor, one_hot, loss
    del original_gradients, original_activations, enhanced_activations
    torch.cuda.empty_cache()
    gc.collect()

    # Convert probabilities if needed
    if isinstance(prediction_result_dict["probabilities"], (torch.Tensor, np.ndarray)):
        prediction_result_dict["probabilities"] = prediction_result_dict["probabilities"].tolist()

    return (prediction_result_dict, attribution_pos_np, gradient_info)


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

    (pred_dict, pos_attr_np, gradient_info) = transmm_prisma(
        model_prisma=model, input_tensor=input_tensor, config=config, sae=sae, discriminative_features=steering_options
    )

    # Structure the output dictionary as expected by pipeline.py
    return {
        "predictions": pred_dict,  # The dict from model_handler.get_prediction via transmm
        "attribution_positive": pos_attr_np,
        "gradient_analysis": gradient_info,  # Add gradient analysis here
        "logits": None,  # This will be None if transmm returns None
        "ffn_activity": [],  # List of dicts
        "class_embedding_representation": [],  # List of dicts
        "head_contribution": []
    }
