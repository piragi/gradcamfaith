# transmm_sfaf.py
import gc
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
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
    # layer 6
    sae_path = "./models/sweep/sae_k128_exp8_lr0.0002/1756558b-vit_medical_sae_k_sweep/n_images_49276.pt"
    # layer 9
    # sae_path = "./models/sweep/sae_k128_exp8_lr0.0002/e1074fed-vit_medical_sae_k_sweep/n_images_49276.pt"
    # layer 10
    # sae_path = "./models/sweep/vanilla_l1_5e-06_exp8_lr1e-05/28ebc3ab-vit_medical_sae_vanilla_sweep/n_images_49276.pt"
    # layer 11
    # sae_path = "./models/sweep/vanilla_l1_1e-05_exp8_lr1e-05/518dec78-vit_medical_sae_vanilla_sweep/n_images_49276.pt"
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


def analyze_single_image_correlation(
    sae_codes: torch.Tensor,  # From your existing code
    attribution_map: np.ndarray,  # The final attribution result
    predicted_class: int,
    sf_af_dict: Dict[str, torch.Tensor],
    feature_id: Optional[int] = None  # If you want to analyze a specific feature
) -> Dict[str, Any]:
    """
    Simplified version for analyzing correlation in a single image.
    Can be called right after attribution generation.
    """
    if feature_id is not None:
        # Analyze specific feature
        codes_spatial = sae_codes[0, 1:, feature_id]  # Spatial tokens for this feature
        active_patches = (codes_spatial > 0.1).cpu().numpy()

        if active_patches.sum() < 2:
            return {'correlation': 0, 'active': False}

        # Resize attribution to patch space
        n_patches = len(codes_spatial)
        n_patches_per_side = int(np.sqrt(n_patches))

        attr_tensor = torch.tensor(attribution_map).unsqueeze(0).unsqueeze(0)
        attr_patches = F.adaptive_avg_pool2d(attr_tensor, (n_patches_per_side, n_patches_per_side))
        attr_patches = attr_patches.squeeze().flatten().numpy()

        # Compute correlation
        corr, p_val = stats.pearsonr(active_patches.astype(float), attr_patches)

        return {
            'feature_id': feature_id,
            'correlation': corr,
            'p_value': p_val,
            'active': True,
            'n_active_patches': active_patches.sum(),
            's_f': sf_af_dict['S_f'][feature_id, predicted_class].item(),
            'a_f': sf_af_dict['A_f'][feature_id, predicted_class].item()
        }
    else:
        # Analyze all features - use the full function
        return compute_feature_attribution_correlation(sae_codes, attribution_map)


def build_distributional_sa_dictionary(
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    n_samples: int = 10000,
    layer_idx: int = 6,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Builds a "more reflected" dictionary by storing the full distribution of
    S_f (steerability) and A_f (attention) values for each feature,
    instead of just their mean. This captures context-dependency and avoids
    the "averaging out" problem.

    Returns a dictionary containing lists of lists for S_f and A_f values.
    """
    device = next(model.parameters()).device
    n_features = sae.cfg.d_sae
    n_classes = model.head.out_features

    # --- NEW: Initialize data structures for storing distributions ---
    # We use a nested list structure: [feature][class] -> [list of values]
    # This is more memory-efficient than a giant tensor for sparse data.
    S_f_distributions = [[[] for _ in range(n_classes)] for _ in range(n_features)]
    A_f_distributions = [[[] for _ in range(n_classes)] for _ in range(n_features)]
    feature_class_counts = torch.zeros(n_features, n_classes, device=device)

    # --- Hook setup (remains the same) ---
    attn_hook_name = f"blocks.{layer_idx}.attn.hook_pattern"
    resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"

    samples_processed = 0
    print(f"Building Distributional S_f/A_f dictionary for layer {layer_idx}...")

    for imgs, labels in tqdm(dataloader, desc="Processing batches"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = imgs.shape[0]

        # --- Forward pass & Gradient calculation (remains the same) ---
        resid_storage, attn_storage = {}, {}

        def save_resid_hook(tensor, hook):
            tensor.requires_grad_(True)
            resid_storage['resid'] = tensor

        def save_attn_hook(tensor, hook):
            tensor.requires_grad_(True)
            attn_storage['attn'] = tensor

        fwd_hooks = [(resid_hook_name, save_resid_hook), (attn_hook_name, save_attn_hook)]
        model.zero_grad()
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(imgs)
        resid, attn = resid_storage['resid'], attn_storage['attn']

        with torch.no_grad():
            _, codes = sae.encode(resid)

        one_hot_labels = F.one_hot(labels, num_classes=n_classes).float()
        target_values = (logits * one_hot_labels).sum()
        resid_grad, attn_grad = torch.autograd.grad(outputs=target_values, inputs=[resid, attn])

        # --- S_f and A_f calculation per image (remains the same) ---
        with torch.no_grad():
            grad_weighted_attn = (attn * attn_grad.abs()).sum(dim=1)
            cls_to_patch_attn = grad_weighted_attn[:, 0, 1:]
            active_codes_mask = codes[:, 1:, :] > 0.0
            A_f_per_img = torch.einsum('bt,btf->bf', cls_to_patch_attn, active_codes_mask.float())

            dir_deriv = torch.einsum('bd,fd->bf', resid_grad[:, 0, :], sae.W_dec)
            codes_no_cls = codes[:, 1:, :]
            S_f_per_img = (codes_no_cls * dir_deriv.unsqueeze(1)).sum(1)

        # --- NEW: Distributional Aggregation ---
        # This part is completely different. We loop through the batch and append.
        active_features_per_img = (codes[:, 1:, :] > 0.0).any(1)  # Shape: (B, F)

        # Move to CPU for efficient list appending
        S_batch_cpu = S_f_per_img.cpu()
        A_batch_cpu = A_f_per_img.cpu()
        active_features_cpu = active_features_per_img.cpu()
        labels_cpu = labels.cpu()

        for i in range(batch_size):  # Loop over each image in the batch
            img_label = labels_cpu[i].item()
            # Find which features were active for this image
            active_indices = active_features_cpu[i].nonzero(as_tuple=True)[0]

            if len(active_indices) == 0:
                continue

            # Get the S_f and A_f values for this image's active features
            s_values_for_active_feats = S_batch_cpu[i, active_indices]
            a_values_for_active_feats = A_batch_cpu[i, active_indices]

            # Append these values to our global distribution lists
            for j, feat_idx in enumerate(active_indices):
                S_f_distributions[feat_idx][img_label].append(s_values_for_active_feats[j].item())
                A_f_distributions[feat_idx][img_label].append(a_values_for_active_feats[j].item())

            # Update counts (can still be done on GPU for speed)
            feat_idx_gpu = active_indices.to(device)
            class_idx_gpu = labels[i].expand(len(feat_idx_gpu))
            feature_class_counts.index_put_((feat_idx_gpu, class_idx_gpu),
                                            torch.ones_like(feat_idx_gpu, dtype=torch.float),
                                            accumulate=True)

        samples_processed += batch_size
        if n_samples is not None and samples_processed >= n_samples:
            break

    # --- Final Dictionary Assembly ---
    # No normalization needed, we have the raw distributions.
    dictionary = {
        'S_f_distributions': S_f_distributions,
        'A_f_distributions': A_f_distributions,
        'feature_counts': feature_class_counts.cpu(),
        'layer_idx': layer_idx,
        'n_samples': samples_processed,
        'n_features': n_features,
        'n_classes': n_classes
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # Use pickle for saving complex Python objects like lists of lists
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(dictionary, f)
        print(f"Saved distributional dictionary to {save_path}")

    return dictionary


def build_steerability_attention_dictionary(
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    n_samples: int = 10000,
    layer_idx: int = 6,
    save_path: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Build dictionary of S_f (steerability) and A_f (attention relevance) using
    a vectorized and correct gradient calculation approach.
    """
    #TODO: The big question here is how we are going to put Sf and Af into relation. There are several ideas that we can employ.
    # But we have to consider that Af might be roughly between [0,1] due to normalization, while Sf is unbounded.
    # Needs tests and analyzing
    # Normalize each separately:
    # A_f_norm = (A_f - A_f.mean()) / A_f.std()
    # S_f_norm = (S_f - S_f.mean()) / S_f.std()
    # ratio = S_f_norm / (A_f_norm + eps)
    # Feature selction without ratio:
    # high_impact_low_attention = (S_f > S_f.quantile(0.9)) & (A_f < A_f.quantile(0.5))

    device = next(model.parameters()).device
    n_features = sae.cfg.d_sae
    n_classes = model.head.out_features

    # Initialize accumulators
    S_f_sum = torch.zeros(n_features, n_classes, device=device)
    A_f_sum = torch.zeros(n_features, n_classes, device=device)
    feature_class_counts = torch.zeros(n_features, n_classes, device=device)

    hook_point = sae.cfg.hook_point
    attn_hook_name = f"blocks.{layer_idx}.attn.hook_pattern"
    resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"

    samples_processed = 0
    print(f"Building S_f/A_f dictionary for layer {layer_idx}...")

    for imgs, labels in tqdm(dataloader, desc="Processing batches"):

        imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = imgs.shape[0]

        # --- 1. Forward Pass to get Activations and Logits ---
        # We need to capture the intermediate tensors we'll need gradients for.
        resid_storage = {}
        attn_storage = {}

        def save_resid_hook(tensor, hook):
            tensor.requires_grad_(True)  # Ensure this tensor can have a grad
            resid_storage['resid'] = tensor

        def save_attn_hook(tensor, hook):
            tensor.requires_grad_(True)
            attn_storage['attn'] = tensor

        fwd_hooks = [(resid_hook_name, save_resid_hook), (attn_hook_name, save_attn_hook)]

        model.zero_grad()
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(imgs)

        resid = resid_storage['resid']  # Shape: (B, T, D_model)
        attn = attn_storage['attn']  # Shape: (B, H, T, T)

        # --- 2. Get SAE Codes for the batch ---
        with torch.no_grad():
            _, codes = sae.encode(resid)  # Shape: (B, T, D_sae)

            batch_active = (codes[:, 1:, :] > 0).any((0, 1)).float().mean() * 100
            print(f"active features this batch: {batch_active:.2f} %")

        # --- 3. Compute Gradients Efficiently using torch.autograd.grad ---
        # We want the gradient of the PROBABILITY for the *correct class* for each image
        # w.r.t the residual stream and the attention pattern.

        # Create a one-hot vector for the batch labels
        one_hot_labels = F.one_hot(labels, num_classes=n_classes).float()

        # We want the gradient of sum(logits * one_hot_labels)
        target_values = (logits * one_hot_labels).sum()

        # Compute gradients for both tensors in a single call
        # The 'outputs' is now target_values (based on logits) instead of target_logits
        resid_grad, attn_grad = torch.autograd.grad(outputs=target_values, inputs=[resid, attn])
        # resid_grad shape: (B, T, D_model)
        # attn_grad shape: (B, H, T, T)

        # --- 4. Calculate S_f and A_f for the batch --------------------
        with torch.no_grad():
            # 4-a  attention relevance A_f
            grad_weighted_attn = (attn * attn_grad.abs()).sum(dim=1)  # (B,T,T)
            cls_to_patch_attn = grad_weighted_attn[:, 0, 1:]  # (B,T-1)

            active_codes_mask = codes[:, 1:, :] > 0.0  # (B,T-1,F)
            A_f_per_img = torch.einsum('bt,btf->bf', cls_to_patch_attn, active_codes_mask.float())

            # 4-b  steerability S_f
            dir_deriv = torch.einsum(
                'bd,fd->bf',  # (B,F)
                resid_grad[:, 0, :],
                sae.W_dec
            )
            codes_no_cls = codes[:, 1:, :]  # (B,T-1,F)
            S_f_per_img = (codes_no_cls * dir_deriv.unsqueeze(1)).sum(1)  # (B,F)

            # ----------------------------------------------------------
            # 5.  accumulate   (features × classes table)
            # ----------------------------------------------------------
            active_features_per_img = active_codes_mask.any(1)  # (B,F)  bool

            # mask-out inactive ones so we don’t count them
            S_batch = S_f_per_img * active_features_per_img  # (B,F)
            A_batch = A_f_per_img * active_features_per_img.float()  # (B,F)

            # indices for index_put_
            feat_idx = torch.arange(n_features, device=device).expand(batch_size, -1)  # (B,F)
            class_idx = labels.view(-1, 1).expand(-1, n_features)  # (B,F)

            # flatten everything -> 1-D
            flat_feat = feat_idx.reshape(-1)
            flat_class = class_idx.reshape(-1)

            S_f_sum.index_put_((flat_feat, flat_class), S_batch.reshape(-1), accumulate=True)
            A_f_sum.index_put_((flat_feat, flat_class), A_batch.reshape(-1), accumulate=True)
            feature_class_counts.index_put_((flat_feat, flat_class),
                                            active_features_per_img.float().reshape(-1),
                                            accumulate=True)

        # ----------------------------------------------------------------
        samples_processed += batch_size
        if samples_processed >= n_samples:
            break

    # --- 6. Final Normalization ---
    with torch.no_grad():
        # Avoid division by zero
        safe_counts = feature_class_counts.clamp_min(1.0)
        A_f = A_f_sum / safe_counts
        S_f = S_f_sum / safe_counts

    dictionary = {
        'S_f': S_f.cpu(),
        'A_f': A_f.cpu(),
        'feature_counts': feature_class_counts.cpu(),
        'layer_idx': layer_idx,
        'n_samples': samples_processed
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dictionary, save_path)
        print(f"Saved S_f/A_f dictionary to {save_path}")

    return dictionary


def load_or_build_sf_af_dictionary(
    model: HookedSAEViT,
    sae: SparseAutoencoder,
    layer_idx: int = 6,
    dict_path: str = "./sae_dictionaries/sf_af_dict.pt",
    rebuild: bool = False,
    n_samples: int = 1000,
) -> Dict[str, torch.Tensor]:
    """Load existing dictionary or build a new one"""
    if not rebuild and Path(dict_path).exists():
        print(f"Loading existing S_f/A_f dictionary from {dict_path}")
        return torch.load(dict_path, weights_only=False)

    # Build new dictionary
    label_map = {2: 3, 3: 2}

    def custom_target_transform(target):
        return label_map.get(target, target)

    train_path = "./hyper-kvasir_imagefolder/train"
    train_dataset = torchvision.datasets.ImageFolder(
        train_path, get_processor_for_precached_224_images(), target_transform=custom_target_transform
    )
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    return build_steerability_attention_dictionary(
        model, sae, dataloader, n_samples=n_samples, layer_idx=layer_idx, save_path=dict_path
    )


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
    sae: Optional[SparseAutoencoder] = None,
    sf_af_dict: Optional[Dict[str, torch.Tensor]] = None,
    enable_steering: bool = True,
    steering_layer: int = 6,
    steering_strength: float = 1.5,
    class_analysis=None
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """
    TransMM with S_f/A_f based patch boosting.
    
    Single pass approach:
    1. Compute standard TransMM attribution
    2. Identify high S_f/A_f features and boost patches where they activate
    
    Note: Parameters keep original names for compatibility but are repurposed:
    - enable_steering: enables SAE-based patch boosting
    - steering_layer: layer to analyze SAE features
    - steering_strength: boost strength (now as fraction, e.g., 0.3)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    model_prisma.reset_hooks()
    model_prisma.zero_grad()

    # Storage for gradients and activations
    gradients = {}
    activations = {}
    sae_codes = {}

    # Define hook names
    attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(model_prisma.cfg.n_layers)]

    # Hooks
    def save_activation_hook(tensor: torch.Tensor, hook: Any):
        activations[hook.name] = tensor.detach().clone()

    def save_gradient_hook(tensor: torch.Tensor, hook: Any):
        gradients[hook.name + "_grad"] = tensor.detach().clone()

    fwd_hooks = [(name, save_activation_hook) for name in attn_hook_names]
    bwd_hooks = [(name, save_gradient_hook) for name in attn_hook_names]

    # Add residual hook for SAE analysis
    if enable_steering and sae is not None:
        resid_hook_name = f"blocks.{steering_layer}.hook_resid_post"

        def save_resid_and_codes(tensor, hook):
            with torch.no_grad():
                _, codes = sae.encode(tensor)
                sae_codes['codes'] = codes.detach().clone()
                sae_codes['layer'] = steering_layer
            return tensor

        fwd_hooks.append((resid_hook_name, save_resid_and_codes))

    # Forward and backward pass
    with model_prisma.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        logits = model_prisma(input_tensor)
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

        # Backward pass for gradients
        one_hot = torch.zeros((1, logits.size(-1)), dtype=torch.float32, device=device)
        one_hot[0, predicted_class_idx] = 1
        one_hot.requires_grad_(True)
        loss = torch.sum(one_hot * logits)
        loss.backward()

    # print(f"Predicted class: {predicted_class_idx}")

    # Build prediction dictionary
    prediction_result_dict = {
        "logits": logits.detach(),
        "probabilities": probabilities.squeeze().cpu().detach().numpy().tolist(),
        "predicted_class_idx": predicted_class_idx,
        "predicted_class_label": IDX2CLS[predicted_class_idx]
    }

    # ---------- build boost mask ONCE ----------
    # Extract CLS    boost_mask, selected_feat_ids = (torch.ones(196), [])  # defaults to no boost (1.0)

    if enable_steering and sae and sf_af_dict and "codes" in sae_codes:

        # boost_mask, selected_feat_ids = build_adaptive_boost_mask(
        # sae_codes=sae_codes["codes"],
        # stealth_dict=sf_af_dict,  # Load this once at the start
        # predicted_class=predicted_class_idx,
        # base_strength=1.5,  # Tune this
        # top_k=2000,  # Tune this
        # device=device,
        # )
        # boost_mask, selected_feat_ids = build_patch_boost_mask_from_stealth(
        # sae_codes["codes"],
        # sf_af_dict,
        # boost_strength=1.1,  # or expose this through your config
        # device=device,
        # )
        #
        # boost_mask, boosted_ids, avoided_ids = build_class_specific_boost_mask(
        # sae_codes["codes"],
        # sf_af_dict,
        # class_analysis,
        # target_class=predicted_class_idx,  # e.g., for "polyp" class
        # top_k=30,
        # boost_strength=2.0
        # )

        print(f'Predicted class: {IDX2CLS[predicted_class_idx]}')

        boost_mask, selected_feat_ids = build_aligned_boost_mask(
            sae_codes=sae_codes["codes"],
            alignment_dict=sf_af_dict,
            predicted_class=predicted_class_idx,
            device=device,
            # These hyperparameters should ideally come from your config object
            top_k=5,  # e.g., 5
            base_strength=2.,  # e.g., 1.8
            min_pfac_for_consideration=0.15  # e.g., 0.1
        )

        if selected_feat_ids:
            print(f"Boosting based on {len(selected_feat_ids)} features: {selected_feat_ids}")

        # boost_mask, selected_feat_ids = build_adaptive_boost_mask_class(
        # sae_codes=sae_codes["codes"],
        # stealth_dict=sf_af_dict,
        # predicted_class=predicted_class_idx,
        # base_strength=5,  # Tune this
        # top_k=1,  # Tune this
        # ranking_metric='s_f',  # Or 'logit_impact'
        # device=device,
        # )

        # boost_mask, selected_feat_ids = build_patch_boost_mask_adaptive_sign(
        # sae_codes['codes'],
        # sf_af_dict,
        # predicted_class_idx,
        # config=config,
        # device=device,
        # )
        # boost_mask, selected_feat_ids = build_patch_boost_mask_adaptive(
        # sae_codes['codes'],
        # sf_af_dict,
        # predicted_class_idx,
        # base_strength=config.classify.base_strength,  # e.g., 2.0
        # percentile_threshold=config.classify.percentile_threshold,  # top 10% |S_f|
        # attention_threshold=config.classify.attention_threshold,  # bottom 50% A_f
        # device=device,
        # top_k_features=config.classify.top_k_features,
        # )
        # boost_mask, selected_feat_ids = build_patch_boost_mask_advanced(
        # sae_codes['codes'],
        # sf_af_dict,
        # predicted_class_idx,
        # base_strength=steering_strength,  # e.g., 2.0
        # percentile_threshold=60.0,  # top 10% |S_f|
        # attention_threshold=30.0,  # bottom 50% A_f
        # device=device,
        # use_percentile_weighting=False,
        # top_k_features=10,
        # )
        # boost_mask, selected_feat_ids = build_patch_boost_mask_simple(
        # sae_codes['codes'],
        # sf_af_dict,
        # predicted_class_idx,
        # strength=steering_strength,  # e.g., 2.0
        # percentile_threshold=70.0,  # top 10% |S_f|
        # attention_threshold=40.0,  # bottom 50% A_f
        # device=device,
        # )
    else:
        boost_mask = torch.ones(196, device=device)
        selected_feat_ids = []

    # -------------- Attribution loop --------------
    num_tokens = activations[attn_hook_names[0]].shape[-1]
    R_pos = torch.eye(num_tokens, num_tokens, device='cpu')

    # Add this diagnostic code in your attribution loop
    for i in range(model_prisma.cfg.n_layers):
        hname = f"blocks.{i}.attn.hook_pattern"
        grad = gradients[hname + "_grad"]
        cam = activations[hname]

        cam_pos_avg = avg_heads(cam, grad)

        # Diagnostic: track R magnitude
        # if i == steering_layer - 1:
        # print(f"Layer {i} R[0,1:] mean: {R_pos[0, 1:].mean():.4f}, max: {R_pos[0, 1:].max():.4f}")

        if enable_steering and i == steering_layer and len(selected_feat_ids) > 0:
            cam_pos_avg_before = cam_pos_avg.clone()
            cam_pos_avg[0, 1:] *= boost_mask.cpu()

            # # More detailed diagnostics
            # print(f"\nLayer {i} diagnostics:")
            # print(f"R[0,1:] before boost - mean: {R_pos[0, 1:].mean():.4f}, max: {R_pos[0, 1:].max():.4f}")
            # print(
            # f"cam_pos_avg[0,1:] - mean: {cam_pos_avg_before[0, 1:].mean():.4f}, max: {cam_pos_avg_before[0, 1:].max():.4f}"
            # )
            # print(f"Boost mask - min: {boost_mask.min():.2f}, max: {boost_mask.max():.2f}")

            # Calculate actual contribution
            R_contribution = apply_self_attention_rules(R_pos, cam_pos_avg
                                                        ) - apply_self_attention_rules(R_pos, cam_pos_avg_before)
            # print(
            # f"Boost contribution to R[0,1:] - mean: {R_contribution[0, 1:].mean():.6f}, max: {R_contribution[0, 1:].max():.6f}"
            # )

        R_pos = R_pos + apply_self_attention_rules(R_pos, cam_pos_avg)
    transformer_attribution_pos = R_pos[0, 1:].clone()

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

    # Gradient info
    gradient_info = {
        'boosted_features': selected_feat_ids,
        'num_boosted_features': len(selected_feat_ids),
        'steering_layer': steering_layer,
        'lambda_used': 0.5
    }
    # Clean up
    del transformer_attribution_pos, input_tensor, one_hot, loss
    del gradients, activations
    if 'codes' in sae_codes:
        del sae_codes
    torch.cuda.empty_cache()
    gc.collect()

    return (prediction_result_dict, attribution_pos_np, gradient_info)


def build_aligned_boost_mask_vectorized(
    sae_codes: torch.Tensor,
    alignment_dict: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    top_k: int = 5,
    base_strength: float = 1.8,
    min_activation: float = 0.05,
    min_pfac_for_consideration: float = 0.1,
    min_occurrences_for_class: int = 3,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Vectorized version of boost mask building for improved performance.
    """
    codes_patches = sae_codes[0, 1:]
    n_patches, n_features = codes_patches.shape

    feature_stats = alignment_dict.get('feature_stats')
    if not feature_stats:
        return torch.ones(n_patches, device=device), []

    # Pre-compute active features
    active_in_image_mask = (codes_patches > min_activation).any(dim=0)
    active_indices = active_in_image_mask.nonzero(as_tuple=True)[0].cpu().numpy()

    if len(active_indices) == 0:
        return torch.ones(n_patches, device=device), []

    # Pre-extract all relevant data into arrays for vectorized processing
    n_active = len(active_indices)
    pfac_scores = np.zeros(n_active)
    valid_mask = np.zeros(n_active, dtype=bool)
    feature_ids = []

    # Batch process all active features
    for i, feat_id in enumerate(active_indices):
        if feat_id not in feature_stats:
            continue

        stats = feature_stats[feat_id]
        raw_metrics = stats.get('raw_metrics', {})
        if not raw_metrics:
            continue

        # Vectorized extraction of class-specific metrics
        pfac_all = np.array(raw_metrics.get('pfac_corrs', []))
        classes_all = np.array(raw_metrics.get('classes', []))

        # Boolean mask for target class
        class_mask = classes_all == predicted_class
        n_class_occurrences = class_mask.sum()

        if n_class_occurrences >= min_occurrences_for_class:
            avg_pfac = pfac_all[class_mask].mean()
            if not np.isnan(avg_pfac) and avg_pfac >= min_pfac_for_consideration:
                pfac_scores[i] = avg_pfac
                valid_mask[i] = True
                feature_ids.append(feat_id)

    if not any(valid_mask):
        return torch.ones(n_patches, device=device), []

    # Get top-k features using vectorized operations
    valid_scores = pfac_scores[valid_mask]
    valid_indices_filtered = np.array(feature_ids)

    # Use argpartition for efficient top-k selection (faster than full sort)
    if len(valid_scores) > top_k:
        top_k_indices = np.argpartition(valid_scores, -top_k)[-top_k:]
        # Sort just the top-k
        top_k_indices = top_k_indices[np.argsort(valid_scores[top_k_indices])[::-1]]
    else:
        top_k_indices = np.argsort(valid_scores)[::-1]

    selected_feature_ids = valid_indices_filtered[top_k_indices].tolist()

    # Vectorized boost mask creation
    boost_mask = torch.ones(n_patches, device=device)

    # Create a tensor of selected feature indices for vectorized access
    selected_features_tensor = torch.tensor(selected_feature_ids, device=device)

    # Get all activation masks at once
    activation_masks = codes_patches[:, selected_features_tensor] > min_activation  # (n_patches, top_k)

    # Apply boost to any patch that has at least one active selected feature
    patches_to_boost = activation_masks.any(dim=1)
    boost_mask[patches_to_boost] *= base_strength

    return boost_mask, selected_feature_ids


def build_aligned_boost_mask(
    sae_codes: torch.Tensor,
    alignment_dict: Dict[str, Any],
    predicted_class: int,
    device: torch.device,
    top_k: int = 5,
    base_strength: float = 1.8,
    min_activation: float = 0.05,
    min_pfac_for_consideration: float = 0.1,
    # Let's keep this gate, it's important!
    min_occurrences_for_class: int = 1,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Builds a boost mask by selecting features that are highly aligned with
    baseline attribution maps for the predicted class. (Corrected version)
    """
    codes_patches = sae_codes[0, 1:]
    n_patches, n_features = codes_patches.shape

    feature_stats = alignment_dict.get('feature_stats')
    if not feature_stats:
        return torch.ones(n_patches, device=device), []

    active_in_image_mask = (codes_patches > min_activation).any(dim=0)
    class_specific_candidates = []

    for feat_id_tensor in active_in_image_mask.nonzero(as_tuple=True)[0]:
        feat_id = feat_id_tensor.item()
        if feat_id not in feature_stats:
            continue

        stats = feature_stats[feat_id]
        raw_metrics = stats.get('raw_metrics', {})
        if not raw_metrics:
            continue

        pfac_for_class = [
            pfac for pfac, cls in zip(raw_metrics.get('pfac_corrs', []), raw_metrics.get('classes', []))
            if cls == predicted_class
        ]

        if len(pfac_for_class) < min_occurrences_for_class:
            continue

        avg_pfac = np.mean(pfac_for_class)

        if np.isnan(avg_pfac) or avg_pfac < min_pfac_for_consideration:
            continue

        class_specific_candidates.append({'id': feat_id, 'score': avg_pfac, 'gini': stats.get('mean_gini_score', 0)})

    if not class_specific_candidates:
        return torch.ones(n_patches, device=device), []

    # Sort candidates by their alignment score, highest first
    # Add a guard against NaN values in sorting, which can cause errors
    class_specific_candidates.sort(key=lambda x: x.get('score', -1), reverse=True)

    top_features = class_specific_candidates[:top_k]
    # This print statement is for debugging, you can remove it in the final version
    selected_feature_ids = [f['id'] for f in top_features]

    boost_mask = torch.ones(n_patches, device=device)
    for feat_info in top_features:
        feat_id = feat_info['id']
        active_patches_mask = (codes_patches[:, feat_id] > min_activation)
        boost_mask[active_patches_mask] *= base_strength

    return boost_mask, selected_feature_ids


def build_adaptive_boost_mask_class(
    sae_codes: torch.Tensor,
    stealth_dict: Dict[str, Any],
    predicted_class: int,
    base_strength: float = 1.8,
    top_k: int = 10,
    min_activation: float = 0.05,
    ranking_metric: str = 's_f',  # 's_f' or 'logit_impact'
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, List[int]]:
    """
    Builds a highly adaptive, class-sensitive boost/suppression mask.

    This function identifies the most relevant stealth features for a specific
    image and class prediction, then creates a mask to intelligently
    modify the attention mechanism.

    Args:
        sae_codes: SAE activations for the current image. Shape: (1, T, F).
        stealth_dict: The dictionary from build_stealth_feature_dictionary.
        predicted_class: The model's predicted class index for this image.
        base_strength: The maximum boost/suppression factor (e.g., 1.8 means boost
                       up to 1.8x, suppress down to 1/1.8x).
        top_k: The number of top stealth features to consider for this image.
        min_activation: The threshold to consider an SAE feature "active" in a patch.
        ranking_metric: Which metric from the dictionary to use for ranking features.
                        's_f' is generally preferred.
        device: The device to perform computations on.

    Returns:
        A tuple of (boost_mask, selected_feature_ids).
        - boost_mask: A (num_patches,) tensor of multiplicative factors.
        - selected_feature_ids: A list of the integer IDs of the features used.
    """
    # --- 1. Unpack Data and Identify Contextually Active Stealth Features ---
    codes_patches = sae_codes[0, 1:]
    n_patches, _ = codes_patches.shape

    all_stealth_ids = stealth_dict.get('feature_ids')
    if all_stealth_ids is None or all_stealth_ids.numel() == 0:
        return torch.ones(n_patches, device=device), []

    all_stealth_ids = all_stealth_ids.to(device)
    active_in_image_mask = (codes_patches[:, all_stealth_ids] > min_activation).any(dim=0)
    if not active_in_image_mask.any():
        return torch.ones(n_patches, device=device), []

    contextual_stealth_ids = all_stealth_ids[active_in_image_mask]

    # --- 2. Rank Active Features by Class-Specific Impact ---
    raw_feature_data = stealth_dict.get('raw_data', {})
    class_specific_impacts = []

    # <<< CHANGED START: Map ranking_metric string to the actual key in the dictionary.
    metric_key_map = {'s_f': 'all_s_f', 'logit_impact': 'all_impacts'}
    # Ensure the provided ranking_metric is valid
    if ranking_metric not in metric_key_map:
        raise ValueError(f"Invalid ranking_metric: '{ranking_metric}'. Must be 's_f' or 'logit_impact'.")
    metric_list_name = metric_key_map[ranking_metric]

    for fid in contextual_stealth_ids.tolist():
        feature_data = raw_feature_data.get(fid, {})

        metric_values = feature_data.get(metric_list_name, [])
        feature_classes = feature_data.get('all_classes', [])

        # The list comprehension now zips the correct metric values with the classes.
        metric_values_for_class = [
            metric_val  # `metric_val` is now a float from the correct list
            for metric_val, cls in zip(metric_values, feature_classes) if cls == predicted_class
        ]

        if metric_values_for_class:
            avg_impact = np.mean(metric_values_for_class)
            class_specific_impacts.append({'id': fid, 'impact': avg_impact})

    if not class_specific_impacts:
        return torch.ones(n_patches, device=device), []

    class_specific_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)  # print(class_specific_impacts)

    # --- 3. Top-K Filtering ---
    top_k_features = [f for f in class_specific_impacts][:top_k]
    if predicted_class == 5:
        top_k_features = [f for f in class_specific_impacts if f['id'] == 839]

    if not top_k_features:
        return torch.ones(n_patches, device=device), []

    selected_feature_ids = [f['id'] for f in top_k_features]
    print(selected_feature_ids)
    selected_impacts = torch.tensor([f['impact'] for f in top_k_features], device=device)

    # --- 4. Build the Adaptive Boost/Suppression Mask ---
    boost_mask = torch.ones(n_patches, device=device)

    max_abs_impact = selected_impacts.abs().max()
    if max_abs_impact > 0:
        strength_weights = selected_impacts.abs() / max_abs_impact
    else:
        strength_weights = torch.ones_like(selected_impacts)

    for i, feat_stats in enumerate(top_k_features):
        feat_id = feat_stats['id']
        impact_value = feat_stats['impact']
        strength_weight = strength_weights[i].item()

        active_patches_mask = codes_patches[:, feat_id] > min_activation
        if not active_patches_mask.any():
            continue

        if impact_value > 0:
            adaptive_boost = 1.0 + (base_strength - 1.0) * strength_weight
            print(f"feat_id: {feat_id}, boost: {adaptive_boost}")
            boost_mask[active_patches_mask] *= adaptive_boost
        else:
            adaptive_suppression = 1.0 - (1.0 - 1.0 / base_strength) * strength_weight
            print(f"feat_id: {feat_id}, suppression: {adaptive_suppression}")
            boost_mask[active_patches_mask] *= adaptive_suppression

    boost_mask = boost_mask.clamp(min=(1.0 / base_strength), max=base_strength)

    return boost_mask, selected_feature_ids


def build_class_specific_boost_mask(
    sae_codes: torch.Tensor,  # (1, T, F)
    stealth_dict: Dict[str, Any],
    class_analysis: Dict[str, Any],  # output of analyze_class_specific_stealth_behavior
    target_class: int,
    boost_strength: float = 1.05,
    activation_threshold: float = 0.10,
    top_k: int = 500,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """
    Builds a class-specific boost mask that only boosts features beneficial 
    for the target class and avoids features harmful to that class.
    
    Args:
        sae_codes: SAE feature activations (1, T, F)
        stealth_dict: Dictionary from build_stealth_feature_dictionary
        class_analysis: Dictionary from analyze_class_specific_stealth_behavior
        target_class: Class ID to optimize for
        boost_strength: Multiplicative boost factor
        activation_threshold: Minimum activation threshold
        top_k: Maximum number of features to consider
        device: Device for tensor operations
        
    Returns:
        boost_mask: (T-1,) tensor with boost factors
        boosted_feat_ids: List of feature IDs that were boosted
        avoided_feat_ids: List of feature IDs that were avoided
    """
    print(sae_codes)
    codes_patches = sae_codes[0, 1:]  # (196, F)
    n_patches, _ = codes_patches.shape

    if target_class not in class_analysis['class_stats']:
        # Fallback to universal boosting
        return build_patch_boost_mask_from_stealth(
            sae_codes, stealth_dict, boost_strength, activation_threshold, top_k, device
        ) + ([], )  # Add empty avoided list

    class_stats = class_analysis['class_stats'][target_class]
    feature_details = class_stats['feature_details']

    # Get beneficial and harmful features for this class
    beneficial_features = []
    harmful_features = []

    for feat_id, details in feature_details.items():
        if details['beneficial'] and details['count'] >= 2:  # Must appear multiple times
            beneficial_features.append((feat_id, details['mean_impact']))
        elif details['harmful']:
            harmful_features.append(feat_id)

    # Sort beneficial features by impact and take top-k
    beneficial_features.sort(key=lambda x: x[1], reverse=True)
    selected_beneficial = [fid for fid, _ in beneficial_features[:top_k]]

    # Build boost mask
    boost_mask = torch.ones(n_patches, device=device)
    boosted_feat_ids = []

    for fid in selected_beneficial:
        # Check if feature is active in any patches
        active_patches = codes_patches[:, fid] > activation_threshold
        if not active_patches.any():
            continue

        boost_mask[active_patches.cpu()] *= boost_strength
        boosted_feat_ids.append(fid)

    return boost_mask, boosted_feat_ids, harmful_features


def calculate_adaptive_strength_sign(s_value: float, s_percentile_rank: float, base_strength: float) -> float:
    """
    Calculates an adaptive strength factor.
    Features with higher impact (higher percentile rank) get a stronger effect.
    
    Args:
        s_value: The raw S_f value of the feature (sign matters).
        s_percentile_rank: The percentile rank of the feature's |S_f| (0.0 to 1.0).
        base_strength: The maximum possible boost/suppression effect.

    Returns:
        A strength factor. > 1.0 for boosting, < 1.0 for suppression.
    """
    # The magnitude of the effect is based on the percentile rank.
    # A feature at the 95th percentile will have a stronger effect than one at the 91st.
    # Map [0.0, 1.0] rank to a [0.0, 1.0] weight.
    magnitude_weight = (
        s_percentile_rank - 0.5
    ) * 2 if s_percentile_rank > 0.5 else 0.0  # Scale from 0 to 1 for top 50%

    if s_value > 0:  # Constructive feature -> Boost
        # Linearly scale from 1.0 (no boost) to base_strength (max boost)
        return 1.0 + (base_strength - 1.0) * magnitude_weight
    else:  # Destructive feature -> Suppress
        # Linearly scale from 1.0 (no suppression) to 1/base_strength (max suppression)
        return 1.0 - (1.0 - 1.0 / base_strength) * magnitude_weight


def build_patch_boost_mask_adaptive_sign(
    sae_codes: torch.Tensor,  # (1, T, F)
    sf_af_dict: Dict[str, torch.Tensor],
    predicted_class: int,
    config: PipelineConfig,
    device: torch.device
) -> Tuple[torch.Tensor, List[int]]:
    """
    Adaptive, sign-aware patch boosting and suppression.

    1. Selects candidate "stealth" features (high |S_f|, low A_f).
    2. Filters for the top_k most impactful candidates to reduce noise.
    3. For each selected feature, applies a boost (>1) or suppression (<1) to
       patches where it activates, depending on the sign of its S_f value.
    4. The strength of the effect is adaptive based on the feature's |S_f| rank.
    """
    cfg = config.classify
    codes_patches = sae_codes[0, 1:]  # (T-1, F)
    n_patches, n_features = codes_patches.shape

    # --- 1. Get Feature Properties for Predicted Class ---
    S_f = sf_af_dict['S_f'][:, predicted_class].to(device)
    A_f = sf_af_dict['A_f'][:, predicted_class].to(device)
    S_f_abs = S_f

    # --- 2. Select Candidate Stealth Features ---
    # Find all features active in the current image to avoid calculating for all 65k
    codes = sae_codes[0]
    codes_patches = codes[1:]  # (T-1, F)
    active_in_image_mask = (codes_patches > 0.1).any(dim=0)

    # Calculate thresholds based on the full distribution of features
    s_abs_thresh = torch.quantile(S_f_abs, cfg.percentile_threshold / 100.0)
    a_thresh = torch.quantile(A_f, cfg.attention_threshold / 100.0)

    # Identify stealth candidates among the active features
    stealth_candidate_mask = ((S_f_abs > s_abs_thresh) & (A_f < a_thresh) & active_in_image_mask)
    stealth_candidate_indices = stealth_candidate_mask.nonzero(as_tuple=True)[0]

    if len(stealth_candidate_indices) == 0:
        return torch.ones(n_patches, device=device), []

    # --- 3. Filter for Top-K Most Impactful Candidates ---
    candidate_magnitudes = S_f_abs[stealth_candidate_indices]

    num_to_select = min(cfg.top_k_features, len(candidate_magnitudes))
    if num_to_select == 0:
        return torch.ones(n_patches, device=device), []

    _, top_k_in_candidate_indices = torch.topk(candidate_magnitudes, k=num_to_select)
    selected_feature_indices = stealth_candidate_indices[top_k_in_candidate_indices]

    # --- 4. Apply Adaptive Boosting & Suppression ---
    boost_mask = torch.ones(n_patches, device=device)

    # Pre-calculate percentile ranks for all features for efficiency
    s_abs_ranks = S_f_abs.argsort().argsort().float() / (len(S_f_abs) - 1)

    for feat_idx in selected_feature_indices:
        # Get patches where this specific feature is active
        active_patches_mask = codes_patches[:, feat_idx] > 0.1  # min_activation
        if not active_patches_mask.any():
            continue

        s_value = S_f[feat_idx].item()
        s_rank = s_abs_ranks[feat_idx].item()

        # Calculate the adaptive strength for this feature
        strength_factor = calculate_adaptive_strength_sign(s_value, s_rank, cfg.base_strength)

        # Apply the factor multiplicatively to the relevant patches
        boost_mask[active_patches_mask] *= strength_factor

    # Clamp to prevent extreme values
    boost_mask = boost_mask.clamp(min=1.0 / cfg.base_strength, max=cfg.base_strength)

    return boost_mask, selected_feature_indices.tolist()


def build_patch_boost_mask_simplified(
    sae_codes: torch.Tensor,  # (1, T, F)
    sf_af_dict: Dict[str, torch.Tensor],
    predicted_class: int,
    base_strength: float = 2.0,
    percentile_threshold: float = 90.0,  # For |S_f| selection
    attention_threshold: float = 40.0,  # For A_f selection
    activation_threshold: float = 0.1,
    top_k_features: int = 20,
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, List[int]]:
    """
    Simplified boost mask with sign-aware boosting and adaptive strength.
    
    Core logic:
    - Select "stealth" features: high |S_f|, low A_f
    - Boost patches where positive S_f features activate
    - Suppress patches where negative S_f features activate
    - Use adaptive strength based on feature importance
    """
    codes = sae_codes[0]
    codes_patches = codes[1:]  # (T-1, F)
    n_patches = codes_patches.size(0)

    S_f = sf_af_dict['S_f'][:, predicted_class].to(device)
    A_f = sf_af_dict['A_f'][:, predicted_class].to(device)

    # Find active features
    active_mask = (codes_patches > activation_threshold).any(dim=0)
    active_indices = active_mask.nonzero(as_tuple=True)[0]

    if len(active_indices) == 0:
        return torch.ones(n_patches, device=device), []

    # Get thresholds
    S_f_abs = S_f.abs()
    s_threshold = torch.quantile(S_f_abs, percentile_threshold / 100.0)
    a_threshold = torch.quantile(A_f, attention_threshold / 100.0)

    # Select stealth features: high |S_f|, low A_f
    S_f_active = S_f[active_indices]
    A_f_active = A_f[active_indices]
    stealth_mask = (S_f_active.abs() > s_threshold) & (A_f_active < a_threshold)
    stealth_indices = active_indices[stealth_mask]


def build_patch_boost_mask_adaptive(
    sae_codes: torch.Tensor,  # (1, T, F)
    sf_af_dict: Dict[str, torch.Tensor],
    predicted_class: int,
    base_strength: float = 2.0,
    percentile_threshold: float = 90.0,  # top X% |S_f| features
    attention_threshold: float = 40.0,  # bottom Y% A_f features
    top_k_features: int = 20,
    min_activation: float = 0.1,
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, List[int]]:
    """
    Simplified adaptive boosting with sign awareness.
    
    Key principles:
    - Select "stealth" features: high |S_f|, low A_f
    - Boost patches for positive S_f (constructive)
    - Suppress patches for negative S_f (destructive)
    - Use adaptive strength based on feature properties
    """
    codes = sae_codes[0]
    codes_patches = codes[1:]  # (T-1, F)
    n_patches = codes_patches.size(0)

    # Get S_f and A_f for predicted class
    S_f = sf_af_dict['S_f'][:, predicted_class].to(device)
    A_f = sf_af_dict['A_f'][:, predicted_class].to(device)

    # Find active features
    active_mask = (codes_patches > min_activation).any(dim=0)
    active_indices = active_mask.nonzero(as_tuple=True)[0]

    if len(active_indices) == 0:
        return torch.ones(n_patches, device=device), []

    # Get values for active features
    S_f_active = S_f[active_indices]
    A_f_active = A_f[active_indices]

    # Calculate thresholds
    s_threshold = torch.quantile(S_f.abs(), percentile_threshold / 100.0)
    a_threshold = torch.quantile(A_f, attention_threshold / 100.0)

    # Select stealth features: high |S_f| AND low A_f
    stealth_mask = (S_f_active.abs() > s_threshold) & (A_f_active < a_threshold)
    stealth_indices = active_indices[stealth_mask]

    if len(stealth_indices) == 0:
        return torch.ones(n_patches, device=device), []

    # Sort by |S_f| and take top-k
    s_magnitudes = S_f[stealth_indices].abs()
    _, top_k_idx = torch.topk(s_magnitudes, k=min(top_k_features, len(stealth_indices)))
    selected_features = stealth_indices[top_k_idx]

    # Initialize boost mask
    boost_mask = torch.ones(n_patches, device=device)

    # Apply adaptive boosting/suppressing for each selected feature
    for feat_idx in selected_features:
        # Get patches where this feature is active
        feature_activations = codes_patches[:, feat_idx]
        active_patches = feature_activations > min_activation

        if not active_patches.any():
            continue

        # Calculate adaptive strength for this feature
        s_value = S_f[feat_idx].item()
        a_value = A_f[feat_idx].item()
        s_percentile = (S_f.abs() < abs(s_value)).float().mean().item()

        adaptive_strength = calculate_adaptive_strength(s_value, a_value, s_percentile, base_strength)

        # Apply sign-aware boosting/suppressing
        if s_value > 0:  # Constructive feature - boost
            boost_mask[active_patches] *= adaptive_strength
        else:  # Destructive feature - suppress
            # Convert strength to suppression factor (inverse relationship)
            suppress_factor = 1.0 / adaptive_strength
            boost_mask[active_patches] *= suppress_factor

    # Clamp to reasonable range
    boost_mask = boost_mask.clamp(min=0.2, max=base_strength * 1.0)

    # Log statistics
    # print(f"\nBoost mask statistics:")
    # print(f"- Selected {len(selected_features)} features")
    # print(f"- Boosted patches (>1.0): {(boost_mask > 1.0).sum().item()}")
    # print(f"- Suppressed patches (<1.0): {(boost_mask < 1.0).sum().item()}")
    # print(f"- Boost range: [{boost_mask.min():.2f}, {boost_mask.max():.2f}]")

    return boost_mask, selected_features.tolist()


def calculate_adaptive_strength(
    s_value: float,
    a_value: float,
    s_percentile: float,
    base_strength: float = 2.0,
    stealth_weight: float = 0.3,
    magnitude_weight: float = 0.7
) -> float:
    """
    Calculate adaptive boost/suppress strength based on feature properties.
    
    Args:
        s_value: S_f value (can be positive or negative)
        a_value: A_f value (attention relevance)
        s_percentile: Percentile rank of |s_value| (0-1)
        base_strength: Maximum boost strength
        stealth_weight: Weight for stealth score (high |S_f|, low A_f)
        magnitude_weight: Weight for absolute magnitude importance
    
    Returns:
        Adaptive strength value (>1 for boosting)
    """
    # Stealth score: high impact, low attention (normalized to ~[0,1])
    # Using tanh to prevent extreme values
    stealth_score = torch.tanh(torch.tensor(abs(s_value) / (a_value + 1e-6)) / 10.0).item()

    # Magnitude importance: higher percentile = more important
    magnitude_score = s_percentile

    # Combined score (weighted average)
    combined_score = (stealth_weight * stealth_score + magnitude_weight * magnitude_score)

    # Map to strength range [1.0, base_strength]
    adaptive_strength = 1.0 + (base_strength - 1.0) * combined_score

    return adaptive_strength


def build_patch_boost_mask_advanced(
    sae_codes: torch.Tensor,  # (1, T, F)
    sf_af_dict: Dict[str, torch.Tensor],
    predicted_class: int,
    base_strength: float = 3.0,  # Base multiplicative factor
    percentile_threshold: float = 70.0,  # For S_f selection
    attention_threshold: float = 40.0,  # For A_f selection
    activation_threshold: float = 0.1,
    use_percentile_weighting: bool = True,
    top_k_features: int = 20,
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, List[int]]:
    """
    Advanced boosting with:
    - Sign-aware boosting (boost constructive, suppress destructive)
    - Activation-weighted boost strength
    - Percentile-based importance weighting
    """
    codes = sae_codes[0]
    codes_patches = codes[1:]  # (T-1, F)
    n_patches = codes_patches.size(0)

    S_f = sf_af_dict['S_f'][:, predicted_class].to(device)
    A_f = sf_af_dict['A_f'][:, predicted_class].to(device)

    # Find active features
    active_mask = (codes_patches > activation_threshold).any(dim=0)
    active_indices = active_mask.nonzero(as_tuple=True)[0]

    if len(active_indices) == 0:
        return torch.ones(n_patches, device=device), []

    S_f_active = S_f[active_indices]
    A_f_active = A_f[active_indices]

    # Compute percentiles for S_f magnitude
    S_f_abs = S_f.abs()
    s_threshold = torch.quantile(S_f_abs, percentile_threshold / 100.0)
    s_threshold_upper = torch.quantile(S_f_abs, 100 / 100.0)
    a_threshold = torch.quantile(A_f, attention_threshold / 100.0)

    # Select stealth features (high |S_f|, low A_f)
    stealth_mask = (S_f_active.abs() > s_threshold) & (S_f_active.abs()
                                                       < s_threshold_upper) & (A_f_active < a_threshold)
    stealth_features = active_indices[stealth_mask]

    if len(stealth_features) == 0:
        return torch.ones(n_patches, device=device), []

    # NEW: Top-K selection based on |S_f| magnitude
    if len(stealth_features) > top_k_features:
        # Get S_f magnitudes for stealth candidates
        candidate_magnitudes = S_f[stealth_features].abs()

        # Select top-k by magnitude
        _, top_k_indices = torch.topk(candidate_magnitudes, k=min(top_k_features, len(stealth_features)))
        stealth_features = stealth_features[top_k_indices]

        print(f"Selected top {len(stealth_features)} features from {len(stealth_features)} candidates")
    else:
        stealth_features = stealth_features
        print(f"Using all {len(stealth_features)} stealth features (< top_k={top_k_features})")

    # Initialize boost mask
    boost_mask = torch.ones(n_patches, device=device).cpu()

    # Process each stealth feature
    for feat_idx in stealth_features:
        patch_activity = codes_patches[:, feat_idx] > activation_threshold
        if not patch_activity.any():
            continue

        # Get feature properties
        s_value = S_f[feat_idx].item()
        s_magnitude = abs(s_value)

        # 1. Activation-based weighting
        # Normalize activations for this feature across active patches
        feature_activations = codes_patches[:, feat_idx][patch_activity.cpu()]
        max_activation = codes_patches[:, feat_idx].max()
        if max_activation > 0:
            activation_weights = (feature_activations / max_activation).clamp(min=0.3)
        else:
            activation_weights = torch.ones_like(feature_activations) * 0.5

        # 2. Percentile-based importance weighting
        if use_percentile_weighting:
            # What percentile is this feature in?
            percentile_rank = (S_f_abs < s_magnitude).float().mean()
            # Convert to weight (higher percentile = stronger weight)
            percentile_weight = 0.5 + 0.5 * percentile_rank  # 0.5 to 1.0
        else:
            percentile_weight = 1.0

        # 3. Sign-aware boost calculation
        if s_value > 0:  # Constructive feature
            # Boost = base * activation_weight * percentile_weight
            patch_boost = 1.0 + (base_strength - 1.0) * activation_weights * percentile_weight
            # Apply multiplicatively
            boost_mask[patch_activity.cpu()] *= patch_boost.cpu()
        else:  # Destructive feature
            # For destructive features, we suppress (reduce attention)
            # Stronger suppression for more negative S_f
            suppression_strength = min(s_magnitude / s_threshold, 2.0)  # Cap at 2x threshold
            suppression_factor = 0.2 + 0.4 * (1 - suppression_strength)  # 0.1 to 0.5
            # Weight by activation but inverse for suppression
            patch_suppress = suppression_factor + (1 - suppression_factor) * (1 - activation_weights * 0.5)
            boost_mask[patch_activity.cpu()] *= patch_suppress.cpu()

    # Prevent extreme values
    boost_mask = boost_mask.clamp(min=0.3, max=base_strength * 1)

    # Log statistics
    n_boosted = (boost_mask > 1.0).sum().item()
    n_suppressed = (boost_mask < 1.0).sum().item()
    print(f"Boosted {n_boosted} patches, suppressed {n_suppressed} patches")
    if n_boosted > 0:
        print(f"Boost range: {boost_mask[boost_mask > 1.0].min():.2f} - {boost_mask[boost_mask > 1.0].max():.2f}")
    if n_suppressed > 0:
        print(f"Suppression range: {boost_mask[boost_mask < 1.0].min():.2f} - {boost_mask[boost_mask < 1.0].max():.2f}")

    return boost_mask, stealth_features.tolist()


def build_patch_boost_mask_simple(
    sae_codes: torch.Tensor,  # (1, T, F)
    sf_af_dict: Dict[str, torch.Tensor],
    predicted_class: int,
    strength: float = 2.0,  # multiplicative boost factor
    percentile_threshold: float = 90.0,  # top X% S_f features
    attention_threshold: float = 50.0,  # bottom Y% A_f features
    activation_threshold: float = 0.1,
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, List[int]]:
    """
    Simple boosting based on your S_f/A_f analysis.
    
    Returns:
        boost_mask: (T-1,) multiplicative factors for CLS→patch attention
        selected_features: list of feature indices that were boosted
    """
    codes = sae_codes[0]  # (T, F)
    codes_patches = codes[1:]  # (T-1, F) excluding CLS
    n_patches = codes_patches.size(0)

    # Get S_f and A_f for predicted class
    S_f = sf_af_dict['S_f'][:, predicted_class].to(device)  # (F,)
    A_f = sf_af_dict['A_f'][:, predicted_class].to(device)  # (F,)

    S_f = torch.clamp(S_f, max=0).abs()  # (F,)

    # Find active features in this image
    active_mask = (codes_patches > activation_threshold).any(dim=0)  # (F,)
    active_indices = active_mask.nonzero(as_tuple=True)[0]

    if len(active_indices) == 0:
        return torch.ones(n_patches, device=device), []

    # Get S_f and A_f for active features
    S_f_active = S_f[active_indices]
    A_f_active = A_f[active_indices]

    # Compute percentiles for thresholding
    s_threshold = torch.quantile(S_f.abs(), percentile_threshold / 100.0)
    a_threshold = torch.quantile(A_f, attention_threshold / 100.0)

    # Select "stealth" features: high |S_f| AND low A_f
    stealth_mask = (S_f_active.abs() > s_threshold) & (A_f_active < a_threshold)
    stealth_features = active_indices[stealth_mask]

    if len(stealth_features) == 0:
        return torch.ones(n_patches, device=device), []

    # Initialize boost mask
    boost_mask = torch.ones(n_patches, device=device)

    # For each stealth feature, boost patches where it's active
    for feat_idx in stealth_features:
        # Find patches where this feature fires
        patch_activity = codes_patches[:, feat_idx] > activation_threshold

        # Apply boost based on S_f sign
        s_value = S_f[feat_idx]
        if s_value > 0:  # Constructive feature
            boost_mask[patch_activity] *= strength
        else:  # Destructive feature - also boost to highlight counter-evidence
            boost_mask[patch_activity] *= strength

    # Prevent extreme values
    boost_mask = boost_mask.clamp(max=strength * 1)

    return boost_mask, stealth_features.tolist()


from typing import Any, Dict, List, Tuple

import torch


def build_adaptive_boost_mask(
    sae_codes: torch.Tensor,
    stealth_dict: Dict[str, Any],
    predicted_class: int,
    base_strength: float = 1.5,
    top_k: int = 10,
    min_activation: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, List[int]]:
    """
    Builds a highly adaptive boost/suppression mask using a pre-computed
    stealth feature dictionary.

    This combines four strategies:
    1.  **Context-Aware Filtering:** Only considers features active in the current image.
    2.  **Top-K Selection:** Focuses on the most impactful (high |S_f|) features.
    3.  **Sign-Awareness:** Boosts for constructive (S_f > 0) and suppresses for destructive (S_f < 0) features.
    4.  **Adaptive Strength:** The effect size scales with the feature's |S_f| magnitude.

    Args:
        sae_codes: SAE activations for the current image. Shape: (1, T, F).
        stealth_dict: The dictionary from your build_stealth_feature_dictionary function.
        predicted_class: The model's predicted class index for this image.
        base_strength: The maximum boost/suppression factor (e.g., 1.5 means boost up to 1.5x, suppress down to 1/1.5x).
        top_k: The number of top stealth features to consider for this image.
        min_activation: The threshold to consider an SAE feature "active" in a patch.
        device: The device to perform computations on.

    Returns:
        A tuple of (boost_mask, selected_feature_ids).
    """
    # --- 1. Unpack data and identify active features in this image ---
    codes_patches = sae_codes[0, 1:]  # (196, F) - Patches only
    n_patches, _ = codes_patches.shape

    # Get all potential stealth features from the dictionary
    all_stealth_ids = stealth_dict.get('feature_ids')
    if all_stealth_ids is None or all_stealth_ids.numel() == 0:
        return torch.ones(n_patches, device=device), []

    all_stealth_ids = all_stealth_ids.to(device)

    # Find which of these are actually active in this specific image
    active_mask = (codes_patches[:, all_stealth_ids] > min_activation).any(dim=0)
    if not active_mask.any():
        return torch.ones(n_patches, device=device), []

    contextual_stealth_ids = all_stealth_ids[active_mask]

    # --- 2. Get S_f values and select Top-K most impactful features ---
    # We need to find the S_f values for these specific features.
    # Your dictionary stores detailed stats we can use.
    detailed_stats = stealth_dict['detailed_stats']

    s_f_values = []
    valid_ids = []
    for fid in contextual_stealth_ids.tolist():
        # Using mean_s_f as the representative S_f for this feature
        s_f_values.append(detailed_stats[fid]['mean_s_f'])
        valid_ids.append(fid)

    if not valid_ids:
        return torch.ones(n_patches, device=device), []

    s_f_tensor = torch.tensor(s_f_values, device=device)
    valid_ids_tensor = torch.tensor(valid_ids, device=device, dtype=torch.long)

    # Select top-k based on magnitude |S_f|
    s_f_magnitudes = s_f_tensor.abs()
    num_to_select = min(top_k, len(s_f_magnitudes))

    _, top_indices = torch.topk(s_f_magnitudes, k=num_to_select)

    selected_feature_ids = valid_ids_tensor[top_indices]
    selected_s_f_values = s_f_tensor[top_indices]

    # --- 3. Build the adaptive boost/suppression mask ---
    boost_mask = torch.ones(n_patches, device=device)

    # For adaptive strength, we can normalize magnitudes to [0, 1]
    max_magnitude = s_f_magnitudes[top_indices].max()
    if max_magnitude > 0:
        strength_weights = s_f_magnitudes[top_indices] / max_magnitude
    else:
        strength_weights = torch.ones_like(s_f_magnitudes[top_indices])

    for i, feat_idx in enumerate(selected_feature_ids):
        s_value = selected_s_f_values[i].item()
        strength_weight = strength_weights[i].item()

        # Find patches where this feature is active
        active_patches_mask = codes_patches[:, feat_idx] > min_activation
        if not active_patches_mask.any():
            continue

        if s_value > 0:  # Constructive -> BOOST
            # Scale boost from 1.0 to base_strength
            adaptive_boost = 1.0 + (base_strength - 1.0) * strength_weight
            boost_mask[active_patches_mask] *= adaptive_boost
        else:  # Destructive -> SUPPRESS
            # Scale suppression from 1.0 to 1/base_strength
            adaptive_suppression = 1.0 + (base_strength - 1.0) * strength_weight
            boost_mask[active_patches_mask] *= adaptive_suppression

    # Clamp to prevent extreme values from multiple overlapping features
    boost_mask = boost_mask.clamp(min=(1.0 / base_strength), max=base_strength)

    return boost_mask, selected_feature_ids.tolist()


def generate_attribution_prisma(
    model: HookedSAEViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    device: Optional[torch.device] = None,
    sae: Optional[SparseAutoencoder] = None,
    sf_af_dict: Optional[Dict[str, torch.Tensor]] = None,
    enable_steering: bool = True,
    class_analysis=None
) -> Dict[str, Any]:
    """
    Generate attribution with S_f/A_f based steering.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load S_f/A_f dictionary if not provided
    if sf_af_dict is None and sae is not None:
        sf_af_dict = load_or_build_sf_af_dictionary(model, sae)

    input_tensor = input_tensor.to(device)

    (pred_dict, pos_attr_np, gradient_info) = transmm_prisma(
        model_prisma=model,
        input_tensor=input_tensor,
        config=config,
        sae=sae,
        sf_af_dict=sf_af_dict,
        enable_steering=enable_steering,
        class_analysis=class_analysis
    )

    # Structure output
    return {
        "predictions": pred_dict,
        "attribution_positive": pos_attr_np,
        "gradient_analysis": gradient_info,
        "logits": None,
        "ffn_activity": [],
        "class_embedding_representation": [],
        "head_contribution": []
    }
