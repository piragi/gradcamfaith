"""
Feature Gradient Gating for TransLRP
Implements Option 4: Using SAE feature gradients to create cleaner per-patch attribution scalars
"""

from typing import Any, Dict, Optional, Tuple

import torch


def compute_feature_gradient_gate(
    residual_grad: torch.Tensor,
    sae_codes: torch.Tensor,
    sae_decoder: torch.Tensor,
    top_k: int = 5,
    normalize_decoder: bool = True,
    denoise_gradient: bool = False,
    kappa: float = 3.0,
    clamp_min: float = 0.2,
    clamp_max: float = 5.0,
    gate_construction: str = "combined",
    shuffle_decoder: bool = False,
    debug: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute per-patch gating multipliers using SAE feature gradients.
    
    This implements the feature gradient decomposition:
    1. Project gradient to feature space: h = D^T g
    2. Weight by activation: s_k = h_k * f_k  
    3. Sum top-K features: s = Σ_k s_k
    4. Map to multiplier: w = exp(κ * normalize(s))
    
    Args:
        residual_grad: Gradient w.r.t. residual [n_patches, d_model]
        sae_codes: SAE feature activations [n_patches, n_features]
        sae_decoder: SAE decoder matrix [d_model, n_features]
        top_k: Number of top features to use per patch
        normalize_decoder: Whether to normalize decoder columns
        denoise_gradient: Whether to project gradient onto decoder subspace first
        kappa: Scaling factor for exponential mapping
        clamp_min: Minimum multiplier value
        clamp_max: Maximum multiplier value
        gate_construction: Gate construction type: "activation_only", "gradient_only", or "combined"
        shuffle_decoder: Whether to shuffle decoder columns to break semantic alignment
        debug: Whether to return debug information
        
    Returns:
        gate: Per-patch multipliers [n_patches]
        debug_info: Dictionary with debug information
    """
    device = residual_grad.device
    n_patches = residual_grad.shape[0]

    # Handle top_k=None case (use all features)
    if top_k is None:
        top_k = sae_codes.shape[1]

    # --- Stable shuffle (if requested) ---
    if shuffle_decoder:
        # If a perm was not provided, create one deterministically ONCE
        # Be deterministic across calls in this process:
        g = torch.Generator(device=sae_decoder.device)
        g.manual_seed(12345)  # or pass from config
        shuffle_perm = torch.randperm(sae_decoder.shape[1], generator=g, device=sae_decoder.device)
        sae_decoder_shuffled = sae_decoder[:, shuffle_perm]
        # IMPORTANT: shuffling + denoising cancels the effect; turn it off when shuffled
        denoise_gradient = False
    else:
        sae_decoder_shuffled = sae_decoder

    # Normalize decoder columns for stability
    if normalize_decoder:
        decoder_norm = sae_decoder_shuffled / (sae_decoder_shuffled.norm(dim=0, keepdim=True) + 1e-8)
    else:
        decoder_norm = sae_decoder_shuffled

    # Optional: Denoise gradient by projecting onto decoder subspace
    if denoise_gradient:
        # Project: g' = D(D^T g) - removes components not in SAE basis
        feature_proj = residual_grad @ decoder_norm  # [n_patches, n_features]
        residual_grad = feature_proj @ decoder_norm.t()  # [n_patches, d_model]

    # Compute feature gradients: h = D^T g
    feature_grads = residual_grad @ decoder_norm  # [n_patches, n_features]

    # Compute per-patch scalar based on gate construction type
    if gate_construction == "activation_only":
        # Activation-only: sum of top-K activations
        top_vals, top_idx = torch.topk(sae_codes, k=min(top_k, sae_codes.shape[1]), dim=1)
        s_t = top_vals.sum(dim=1)  # [n_patches]
        contributions = top_vals  # For debug info

    elif gate_construction == "gradient_only":
        # Gradient-only: sum of top-K absolute gradients in feature space
        abs_feature_grads = torch.abs(feature_grads)
        top_vals, top_idx = torch.topk(abs_feature_grads, k=min(top_k, abs_feature_grads.shape[1]), dim=1)
        s_t = top_vals.sum(dim=1)  # [n_patches]
        contributions = top_vals  # For debug info

    elif gate_construction == "combined":
        # Combined (default): gradient * activation for top-K features by activation
        top_vals, top_idx = torch.topk(sae_codes, k=min(top_k, sae_codes.shape[1]), dim=1)
        h_top = torch.gather(feature_grads, 1, top_idx)  # [n_patches, top_k]
        contributions = h_top * top_vals  # [n_patches, top_k]
        s_t = contributions.sum(dim=1)  # [n_patches]

    else:
        raise ValueError(f"Unknown gate_construction type: {gate_construction}")

    # Normalize across patches (z-score normalization)
    s_median = s_t.median()
    s_mad = (s_t - s_median).abs().median() + 1e-8

    s_norm = (s_t - s_median) / (1.4826 * s_mad)
    # Map tanh output from [-1, 1] to [0.1, 10]
    gate_min = 0.1
    gate_max = 10.0
    gate_center = (gate_min + gate_max) / 2  # 5.05
    gate_range = (gate_max - gate_min) / 2  # 4.95

    temperature = kappa
    gate = gate_center + gate_range * torch.tanh(s_norm * temperature)
    gate = gate.detach()

    # Collect debug information
    debug_info = {}
    if debug:
        debug_info = {
            'raw_contributions': s_t.detach().cpu().numpy(),
            'normalized_contributions': s_norm.detach().cpu().numpy(),
            'gate_values': gate.detach().cpu().numpy(),
            'top_features_per_patch': top_idx.detach().cpu().numpy(),
            'feature_contributions': contributions.detach().cpu().numpy(),
            'n_positive': (s_t > 0).sum().item(),
            'n_negative': (s_t < 0).sum().item(),
            'mean_gate': gate.mean().item(),
            'std_gate': gate.std().item(),
        }
    else:
        debug_info = {
            'mean_gate': gate.mean().item(),
            'std_gate': gate.std().item(),
        }

    return gate, debug_info


def compute_reconstruction_denoise_gate(
    residuals: torch.Tensor,
    reconstructions: torch.Tensor,
    alpha: float = 5.0,
    min_gate: float = 0.6,
    debug: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute per-patch denoising gate based on reconstruction quality.
    
    Patches that are poorly reconstructed by the SAE are likely noise,
    so we downweight them.
    
    Args:
        residuals: Original residual vectors [n_patches, d_model]
        reconstructions: SAE reconstructions [n_patches, d_model]
        alpha: Sensitivity parameter for sigmoid
        min_gate: Minimum gate value
        debug: Whether to return debug information
        
    Returns:
        gate: Per-patch denoising multipliers [n_patches]
        debug_info: Dictionary with debug information
    """
    # Compute reconstruction error per patch
    recon_error = (residuals - reconstructions).norm(dim=1)
    residual_norm = residuals.norm(dim=1) + 1e-8

    # Normalized reconstruction error
    rel_error = recon_error / residual_norm

    # Map to gate using sigmoid (high error -> low gate)
    gate = torch.sigmoid(-alpha * rel_error)
    gate = torch.clamp(gate, min=min_gate, max=1.0)

    debug_info = {}
    if debug:
        debug_info = {
            'reconstruction_errors': rel_error.detach().cpu().numpy(),
            'gate_values': gate.detach().cpu().numpy(),
            'mean_error': rel_error.mean().item(),
            'mean_gate': gate.mean().item(),
            'n_suppressed': (gate < 0.9).sum().item(),
        }

    return gate, debug_info


def apply_feature_gradient_gating(
    cam_pos_avg: torch.Tensor,
    residual_grad: torch.Tensor,
    sae_codes: torch.Tensor,
    sae: Any,  # SAE model with decoder
    config: Optional[Dict[str, Any]] = None,
    enable_denoising: bool = False,
    residuals: Optional[torch.Tensor] = None,
    debug: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply feature gradient gating to attention CAM.
    
    This is the main entry point that combines feature gradient gating
    with optional reconstruction-based denoising.
    
    Args:
        cam_pos_avg: Averaged attention CAM [n_patches, n_patches]
        residual_grad: Gradient w.r.t. residual (spatial tokens only) [n_patches-1, d_model]
        sae_codes: SAE codes (spatial tokens only) [n_patches-1, n_features]
        sae: SAE model with decoder weight
        config: Configuration dictionary for gating parameters
        enable_denoising: Whether to also apply reconstruction denoising
        residuals: Original residuals if denoising is enabled [n_patches-1, d_model]
        debug: Whether to collect debug information
        
    Returns:
        gated_cam: Modified attention CAM
        debug_info: Combined debug information
    """
    if config is None:
        config = {}

    # Get configuration parameters
    top_k = None  #config.get('top_k_features', 15)
    kappa = config.get('kappa', 10.0)
    clamp_min = config.get('clamp_min', 0.5)
    clamp_max = config.get('clamp_max', 2.0)
    denoise_gradient = config.get('denoise_gradient', False)
    gate_construction = config.get('gate_construction', 'combined')
    shuffle_decoder = config.get('shuffle_decoder', False)

    # Get decoder matrix - handle different SAE implementations
    if hasattr(sae, 'W_dec'):
        decoder = sae.W_dec.T  # StandardSparseAutoencoder uses W_dec [n_features, d_model]
    elif hasattr(sae, 'decoder'):
        decoder = sae.decoder.weight.t()  # Other implementations might use decoder module
    else:
        raise AttributeError(f"SAE of type {type(sae).__name__} has no decoder attribute (W_dec or decoder)")

    # Compute feature gradient gate
    feature_gate, feature_debug = compute_feature_gradient_gate(
        residual_grad=residual_grad,
        sae_codes=sae_codes,
        sae_decoder=decoder,
        top_k=top_k,
        denoise_gradient=denoise_gradient,
        kappa=kappa,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        gate_construction=gate_construction,
        shuffle_decoder=shuffle_decoder,
        debug=debug
    )

    # Optionally compute denoising gate (disabled when decoder is shuffled)
    denoise_gate = None
    denoise_debug = {}
    if enable_denoising and residuals is not None and not shuffle_decoder:
        with torch.no_grad():
            try:
                reconstructions = sae.decode(sae_codes)
            except Exception as e:
                # SAE decode failed - disable denoising
                enable_denoising = False
                denoise_gate = None

        if enable_denoising:
            denoise_gate, denoise_debug = compute_reconstruction_denoise_gate(
                residuals=residuals,
                reconstructions=reconstructions,
                alpha=config.get('denoise_alpha', 5.0),
                min_gate=config.get('denoise_min', 0.6),
                debug=debug
            )

    # Combine gates
    if denoise_gate is not None:
        combined_gate = feature_gate * denoise_gate
    else:
        combined_gate = feature_gate

    # Apply gate to CAM
    # Note: cam_pos_avg shape is [n_patches, n_patches] where first is CLS
    # We gate the spatial tokens (1:) in the second dimension

    # Ensure both tensors are on the same device
    device = cam_pos_avg.device
    combined_gate = combined_gate.to(device)

    # Create a new tensor to avoid in-place modification issues
    # The issue: cam_pos_avg is [197, 197] and combined_gate is [196]
    # We need to properly handle the CLS token dimension
    if cam_pos_avg.shape[0] == combined_gate.shape[0] + 1:
        gated_cam = cam_pos_avg.clone()
        gate_col = combined_gate.unsqueeze(0)  # [1, S]
        gated_cam[:, 1:] = gated_cam[:, 1:] * gate_col

        # row-normalize to keep rows stochastic
        # gated_cam = gated_cam / (gated_cam.sum(-1, keepdim=True) + 1e-8)
    else:
        # CAM is spatial-only
        gated_cam = cam_pos_avg * combined_gate.unsqueeze(0)

    # Compile debug info
    debug_info = {
        'feature_gating': feature_debug,
        'denoising': denoise_debug,
        'combined_gate': combined_gate.detach().cpu().numpy() if debug else None,
    }

    return gated_cam, debug_info
