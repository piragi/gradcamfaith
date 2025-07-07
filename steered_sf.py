import gc
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from vit_prisma.models.base_vit import HookedViT  # Import the new model class
from vit_prisma.models.base_vit import HookedSAEViT
from vit_prisma.models.weight_conversion import convert_timm_weights
from vit_prisma.sae import SparseAutoencoder

from config import PipelineConfig
from vit.model import IDX2CLS


def load_models():
    """Load SAE and fine-tuned model"""
    # Load SAE
    # layer 6
    sae_path = "./models/sweep/sae_k128_exp8_lr0.0002/1756558b-vit_medical_sae_k_sweep/n_images_49276.pt"
    # sae_path = "./models/sweep/sae_k128_exp8_lr0.0002/e1074fed-vit_medical_sae_k_sweep/n_images_49276.pt"
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


def transmm(
    model_prisma,
    gradients,
    activations,
    attn_hook_names,
    device,
    img_size,
):
    num_tokens = activations[attn_hook_names[0]].shape[-1]
    R_pos = torch.eye(num_tokens, num_tokens, device='cpu')

    for i in range(model_prisma.cfg.n_layers):
        hname = f"blocks.{i}.attn.hook_pattern"
        grad = gradients[hname + "_grad"]
        cam = activations[hname]

        cam_pos_avg = avg_heads(cam, grad)
        R_pos = R_pos + apply_self_attention_rules(R_pos, cam_pos_avg)
    transformer_attribution_pos = R_pos[0, 1:].clone()

    def process_attribution_map(attr_tensor: torch.Tensor) -> np.ndarray:
        side_len = int(np.sqrt(attr_tensor.size(0)))
        attr_tensor = attr_tensor.reshape(1, 1, side_len, side_len)
        attr_tensor_device = attr_tensor.to(device)
        attr_interpolated = F.interpolate(
            attr_tensor_device, size=(img_size, img_size), mode='bilinear', align_corners=False
        )
        return attr_interpolated.squeeze().cpu().detach().numpy()

    attribution_pos_np = process_attribution_map(transformer_attribution_pos)
    del transformer_attribution_pos

    # Normalize
    normalize_fn = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8) if (np.max(x) - np.min(x)) > 1e-8 else x
    attribution_pos_np = normalize_fn(attribution_pos_np)

    return attribution_pos_np


# ------------------------------------------------------------
#  helpers
# ------------------------------------------------------------
def compute_s_f(
    sae: SparseAutoencoder,
    resid_grad_cls: torch.Tensor  # (D_model,)
) -> torch.Tensor:  # (F,)
    """
    Directional derivative of the logit wrt every SAE feature.
    S_f = < ∂logit/∂resid_CLS ,  W_dec_f >
    """
    # sae.W_dec : (F , D_model)
    return torch.matmul(sae.W_dec, resid_grad_cls)  # (F,)


def select_positive_features(
    s_f: torch.Tensor,  # (F,)
    codes: torch.Tensor,  # (T-1 , F)  (patch tokens only)
    top_k: int = 20,
    min_activation: float = 0.10
) -> list[int]:
    """
    Keep only features that
      – have S_f > 0
      – are active in at least one patch
    Then return the top-k by |S_f|.
    """
    active = (codes > min_activation).any(0)  # (F,)
    constructive = s_f > 0
    candidate_mask = active & constructive

    if candidate_mask.sum() == 0:
        return []

    cand_idx = candidate_mask.nonzero(as_tuple=True)[0]
    magnitudes = s_f[cand_idx]

    k = min(top_k, len(cand_idx))
    _, top_idx = torch.topk(magnitudes, k=k)
    return cand_idx[top_idx].tolist()


# ──────────────────────────────────────────────────────────────
# adaptive, sign–aware strength
# ──────────────────────────────────────────────────────────────
def adaptive_strength_sign(s_val: float, s_rank: float, base_strength: float = 2.0) -> float:
    """
    > 1  : boost      (constructive)
    < 1  : suppress   (destructive)

    `s_rank` – percentile rank of |S_f|  in  [0 … 1]
    """
    # only the top half (rank > 0.5) receive a scaling > 1.0
    mag_weight = max(0.0, (s_rank - 0.5) * 2)  # 0 … 1

    if s_val >= 0:  # constructive  -> boost
        return 1.0 + (base_strength - 1.0) * mag_weight
    else:  # destructive  -> suppress
        return 1.0 - (1.0 - 1.0 / base_strength) * mag_weight


# ──────────────────────────────────────────────────────────────
# build boost mask from ALL (positive and negative) features
# ──────────────────────────────────────────────────────────────
def build_boost_mask_adaptive_sign(
    codes_patches: torch.Tensor,  # (T-1 , F)
    s_f: torch.Tensor,  # (F,)
    *,
    min_activation: float = 0.10,
    percentile_thresh: float = 0.80,
    top_k: int = 40,
    base_strength: float = 2.0,
    device: torch.device
) -> tuple[torch.Tensor, list[int]]:
    """
    Returns
        mask : (T-1,)  multiplicative factors  (clipped to [1/base_strength , base_strength])
        selected_feature_ids : list[int]
    """
    n_patches, n_features = codes_patches.shape
    boost_mask = torch.ones(n_patches, device=device)

    # ── 1. keep only features that fire in at least one patch ─────────
    active = (codes_patches > min_activation).any(0)  # (F,)
    active_idx = active.nonzero(as_tuple=True)[0]
    if active_idx.numel() == 0:
        return boost_mask, []

    s_active = s_f[active_idx]
    s_abs = s_f.abs()

    # ── 2. threshold by |S_f| percentile ──────────────────────────────
    abs_threshold = torch.quantile(s_abs, percentile_thresh)
    strong_mask = s_abs[active_idx] > abs_threshold
    strong_idx = active_idx[strong_mask]

    if strong_idx.numel() == 0:
        return boost_mask, []

    # ── 3. take top-k by |S_f| magnitude ──────────────────────────────
    k = min(top_k, strong_idx.numel())
    _, top_local = torch.topk(s_abs[strong_idx], k=k)  # indices inside strong_idx
    chosen_idx = strong_idx[top_local]  # real feature ids

    # ── 4. percentile ranks (for adaptive strength) ───────────────────
    s_abs_ranks = s_abs.argsort().argsort().float() / (len(s_abs) - 1)

    # ── 5. build mask ─────────────────────────────────────────────────
    for f_id in chosen_idx.tolist():
        patch_activity = codes_patches[:, f_id] > min_activation
        if not patch_activity.any():
            continue
        s_val = s_f[f_id].item()
        s_rank = s_abs_ranks[f_id].item()
        factor = adaptive_strength_sign(s_val, s_rank, base_strength)
        boost_mask[patch_activity] *= factor

    boost_mask = boost_mask.clamp(min=1.0 / base_strength, max=base_strength)
    return boost_mask, chosen_idx.tolist()


def build_residual_steering_table(
    s_f: torch.Tensor,  # (F,)
    codes_pch: torch.Tensor,  # (T-1 , F)
    *,
    min_activation=0.10,
    percentile_thresh=0.80,  # keep top 20 % |S_f|
    top_k=40,
    base_strength=2.0,
    device=torch.device("cpu")
) -> tuple[list[int], torch.Tensor]:
    """
    Returns
        chosen_feats : list[int]                   feature ids
        lambda_vec   : (k,) tensor of signed λ_f   ( >0 boost,  <0 suppress)
    """
    n_features = s_f.size(0)
    active = (codes_pch > min_activation).any(0)  # (F,)

    # ------------ select candidates by |S_f| -----------------
    s_abs = s_f.abs()
    abs_threshold = torch.quantile(s_abs, percentile_thresh)
    candidate_mask = active & (s_abs > abs_threshold)
    cand_idx = candidate_mask.nonzero(as_tuple=True)[0]
    if cand_idx.numel() == 0:
        return [], torch.empty(0, device=device)

    # ------------ top-k by |S_f| magnitude -------------------
    k = min(top_k, cand_idx.numel())
    _, top_local = torch.topk(s_abs[cand_idx], k=k)
    chosen_idx = cand_idx[top_local]  # (k,)

    # ------------ build signed λ_f ---------------------------
    s_abs_ranks = s_abs.argsort().argsort().float() / (n_features - 1)
    lambdas = []
    for f in chosen_idx:
        rank = s_abs_ranks[f].item()
        strength = adaptive_strength_sign(s_f[f].item(), rank, base_strength=base_strength)
        # convert multiplicative factor → additive λ
        lambdas.append(strength - 1.0)  # boost:  +,  suppress: -
    lambda_vec = torch.tensor(lambdas, device=device)  # (k,)

    return chosen_idx.tolist(), lambda_vec


# ------------------------------------------------------------
#  two–pass TransMM + dynamic steering
# ------------------------------------------------------------
def transmm_prisma(
    model_prisma: HookedSAEViT,
    input_tensor: torch.Tensor,
    sae: SparseAutoencoder,
    steering_layer: int = 6,
    boost_strength: float = 1.6,
    img_size: int = 224,
    device: torch.device | None = None,
):
    """
    1) Find constructive SAE features for *this* image.
    2) Boost their patches in layer `steering_layer`.
    3) Produce TransMM attribution on the boosted run.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_prisma.cuda().eval()
    sae.cuda().eval()

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # ------------------------------------------------------------------
    #   --------  PASS 1 :  identify constructive features  ------------
    # ------------------------------------------------------------------
    model_prisma.reset_hooks()
    act_store, grad_store, resid_store = {}, {}, {}

    attn_names = [f"blocks.{i}.attn.hook_pattern" for i in range(model_prisma.cfg.n_layers)]
    resid_name = f"blocks.{steering_layer}.hook_resid_post"

    def save_resid(t, hook):  # resid_post of steering layer
        t.requires_grad_(True)
        resid_store["resid"] = t

    fwd_hooks = [(resid_name, save_resid)]
    with model_prisma.hooks(fwd_hooks=fwd_hooks):
        logits = model_prisma(input_tensor)

    pred_idx = logits.argmax(-1).item()

    # ❶ ----- autograd.grad instead of backward() -----------------------
    n_classes = logits.size(-1)
    one_hot = F.one_hot(torch.tensor(pred_idx, device=device), num_classes=n_classes).float()
    target_val = (logits * one_hot).sum()

    # Compute both gradients in one call   ❷
    resid_tensor = resid_store["resid"]

    (resid_grad, ) = torch.autograd.grad(outputs=target_val, inputs=[resid_tensor])

    # resid_grad : (1 , T , D)   attn_grad : (1 , H , T , T)
    # -------------------------------------------------------------------
    # ❸ we need only resid_grad[0,0] for S_f
    s_f = compute_s_f(sae, resid_grad[0, 0])

    with torch.no_grad():
        _, codes_full = sae.encode(resid_tensor)  # (1 , T , F)
    codes_patches = codes_full[0, 1:]  # (T-1 , F)
    chosen_features = select_positive_features(s_f, codes_patches, top_k=20, min_activation=0.10)
    boost_mask, chosen_features = build_boost_mask_adaptive_sign(
        codes_patches=codes_patches,
        s_f=s_f,
        min_activation=0.10,
        percentile_thresh=0.80,  # keep top 20 % |S_f|
        top_k=5,  # at most 40 features
        base_strength=boost_strength,
        device=device,
    )
    # ------------------------------------------------------------------
    #   --------  PASS 2 :  boosted forward + attribution  -------------
    # ------------------------------------------------------------------
    model_prisma.reset_hooks()
    activations, gradients = {}, {}

    def save_act(t, hook):
        activations[hook.name] = t.detach()

    def save_grad(t, hook):
        gradients[hook.name + "_grad"] = t.detach()

    chosen_feats, lambda_vec = build_residual_steering_table(
        s_f=s_f,
        codes_pch=codes_patches,
        min_activation=0.10,
        percentile_thresh=0.80,
        top_k=40,
        base_strength=boost_strength,
        device=device,
    )

    print(f"Will steer {len(chosen_feats)} features :", chosen_feats)

    W_sub = sae.W_dec[chosen_feats].to(device)  # (k , D_model)

    def resid_steer_hook(resid, *, hook=None):
        # resid : (B , T , D)
        B, T, _ = resid.shape
        if len(chosen_feats) == 0:
            return resid
        codes = sae.encode(resid)[1]  # (B , T , F)
        codes_sel = codes[..., chosen_feats]  # (B , T , k)
        # scaled feature reconstructions
        weighted_codes = codes_sel * lambda_vec  # broadcast (k,)
        delta = torch.einsum('btk,kd->btd', weighted_codes, W_sub)  # (B,T,D)
        resid += delta
        return resid

    steer_hook_name = f"blocks.{steering_layer}.hook_resid_post"
    fwd_hooks.append((steer_hook_name, resid_steer_hook))

    fwd_hooks = [(name, save_act) for name in attn_names]
    bwd_hooks = [(name, save_grad) for name in attn_names]

    with model_prisma.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        logits_boost = model_prisma(input_tensor)
        probs = logits_boost.softmax(-1)

        one_hot2 = torch.zeros_like(logits_boost, device=device)
        one_hot2[0, pred_idx] = 1.0
        (logits_boost * one_hot2).sum().backward()

    # ------------------------------------------------------------------
    # attribution (same routine you already had)
    # ------------------------------------------------------------------
    attr_map = transmm(model_prisma, gradients, activations, attn_names, device, img_size)

    prediction = {
        "logits": logits_boost.detach(),
        "probabilities": probs.squeeze().cpu().tolist(),
        "predicted_class_label": IDX2CLS[pred_idx],
        "predicted_class_idx": pred_idx,
        "chosen_features": chosen_features,
    }

    # tidy-up
    del activations, gradients, resid_store
    torch.cuda.empty_cache()
    gc.collect()

    return prediction, attr_map


def generate_attribution_prisma(
    model: HookedSAEViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    device: Optional[torch.device] = None,
    sae: Optional[SparseAutoencoder] = None,
    sf_af_dict: Optional[Dict[str, torch.Tensor]] = None,
    enable_steering: bool = True,
) -> Dict[str, Any]:
    """
    Generate attribution with S_f/A_f based steering.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = input_tensor.to(device)

    pred_dict, pos_attr_np = transmm_prisma(
        model_prisma=model,
        input_tensor=input_tensor,
        sae=sae,
    )

    # Structure output
    return {
        "predictions": pred_dict,
        "attribution_positive": pos_attr_np,
        "gradient_analysis": None,
        "logits": None,
        "ffn_activity": [],
        "class_embedding_representation": [],
        "head_contribution": []
    }
