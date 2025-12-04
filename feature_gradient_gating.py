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
    kappa: float = 3.0,
    clamp_max: float = 5.0,
    gate_construction: str = "combined",
    shuffle_decoder: bool = False,
    shuffle_decoder_seed: int = 12345,
    active_feature_threshold: float = 0.1,
    debug: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute per-patch gating multipliers using SAE feature gradients.

    This implements the feature gradient decomposition:
    1. Project gradient to feature space: h = D^T g
    2. Weight by activation: s_k = h_k * f_k
    3. Sum all features: s = Σ_k s_k
    4. Map to multiplier: w = exp(κ * normalize(s))

    Args:
        residual_grad: Gradient w.r.t. residual [n_patches, d_model]
        sae_codes: SAE feature activations [n_patches, n_features]
        sae_decoder: SAE decoder matrix [d_model, n_features]
        kappa: Scaling factor for exponential mapping
        clamp_max: Maximum multiplier value (gate range: [1/clamp_max, clamp_max])
        gate_construction: Gate construction type: "activation_only", "gradient_only", or "combined"
        shuffle_decoder: Whether to shuffle decoder columns to break semantic alignment
        shuffle_decoder_seed: Random seed for decoder shuffling (for reproducibility)
        active_feature_threshold: Threshold for considering a feature "active" in debug mode
        debug: Whether to return debug information

    Returns:
        gate: Per-patch multipliers [n_patches]
        debug_info: Dictionary with debug information
    """
    if shuffle_decoder:
        # Create deterministic shuffle using provided seed
        g = torch.Generator(device=sae_decoder.device)
        g.manual_seed(shuffle_decoder_seed)
        shuffle_perm = torch.randperm(sae_decoder.shape[1], generator=g, device=sae_decoder.device)
        sae_decoder_shuffled = sae_decoder[:, shuffle_perm]
    else:
        sae_decoder_shuffled = sae_decoder

    decoder_norm = sae_decoder_shuffled

    # Compute feature gradients: h = D^T g
    feature_grads = residual_grad @ decoder_norm  # [n_patches, n_features]

    # Compute per-patch scalar based on gate construction type
    if gate_construction == "activation_only":
        s_t = sae_codes.sum(dim=1)  # [n_patches]
        contributions = sae_codes  # For debug info

    elif gate_construction == "gradient_only":
        s_t = feature_grads.sum(dim=1)  # [n_patches]
        contributions = feature_grads  # For debug info

    elif gate_construction == "combined":
        contributions = sae_codes * feature_grads  # [n_patches, n_features]
        s_t = contributions.sum(dim=1)  # [n_patches]

    else:
        raise ValueError(f"Unknown gate_construction type: {gate_construction}")

    # Normalize across patches (z-score normalization)
    s_median = s_t.median()
    s_mad = (s_t - s_median).abs().median() + 1e-8

    s_norm = (s_t - s_median) / (1.4826 * s_mad)

    # Symmetric exponential mapping: gate = clamp_max^(tanh(kappa * s_norm))
    # Range: [1/clamp_max, clamp_max], centered at 1
    gate = torch.exp(
        torch.log(torch.tensor(clamp_max, device=s_norm.device, dtype=s_norm.dtype)) * torch.tanh(kappa * s_norm)
    )
    gate = gate.detach()

    # Collect debug information
    debug_info = {}
    if debug:
        # Collect sparse features (activation > threshold)
        active_mask = sae_codes > active_feature_threshold
        sparse_indices = []
        sparse_activations = []
        sparse_gradients = []
        sparse_contributions = []

        for patch_idx in range(sae_codes.shape[0]):
            mask = active_mask[patch_idx]
            indices = torch.where(mask)[0]
            sparse_indices.append(indices.detach().cpu().numpy())
            sparse_activations.append(sae_codes[patch_idx, mask].detach().cpu().numpy())
            sparse_gradients.append(feature_grads[patch_idx, mask].detach().cpu().numpy())
            sparse_contributions.append(contributions[patch_idx, mask].detach().cpu().numpy())

        # Compute total contribution magnitude per patch (for cancellation analysis)
        total_contribution_magnitude = torch.abs(contributions).sum(dim=1)  # [n_patches]

        debug_info = {
            'gate_values': gate.detach().cpu().numpy(),
            'sparse_features_indices': sparse_indices,
            'sparse_features_activations': sparse_activations,
            'sparse_features_gradients': sparse_gradients,
            'sparse_features_contributions': sparse_contributions,
            'contribution_sum': s_t.detach().cpu().numpy(),  # Net sum (can be canceled)
            'total_contribution_magnitude': total_contribution_magnitude.detach().cpu().numpy(),  # Total magnitude
            'mean_gate': gate.mean().item(),
            'std_gate': gate.std().item(),
        }
    else:
        debug_info = {
            'mean_gate': gate.mean().item(),
            'std_gate': gate.std().item(),
        }

    return gate, debug_info


def apply_feature_gradient_gating(
    cam_pos_avg: torch.Tensor,
    residual_grad: torch.Tensor,
    sae_codes: torch.Tensor,
    sae: Any,
    config: Optional[Dict[str, Any]] = None,
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
    kappa = config.get('kappa', 10.0)
    clamp_max = config.get('clamp_max', 5.0)
    gate_construction = config.get('gate_construction', 'combined')
    shuffle_decoder = config.get('shuffle_decoder', False)
    shuffle_decoder_seed = config.get('shuffle_decoder_seed', 12345)
    active_feature_threshold = config.get('active_feature_threshold', 0.1)

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
        kappa=kappa,
        clamp_max=clamp_max,
        gate_construction=gate_construction,
        shuffle_decoder=shuffle_decoder,
        shuffle_decoder_seed=shuffle_decoder_seed,
        active_feature_threshold=active_feature_threshold,
        debug=debug
    )

    # Ensure both tensors are on the same device
    device = cam_pos_avg.device
    feature_gate = feature_gate.to(device)

    # Create a new tensor to avoid in-place modification issues
    # The issue: cam_pos_avg is [197, 197] and combined_gate is [196]
    # We need to properly handle the CLS token dimension
    if cam_pos_avg.shape[0] == feature_gate.shape[0] + 1:
        gated_cam = cam_pos_avg.clone()
        gate_col = feature_gate.unsqueeze(0)  # [1, S]
        gated_cam[:, 1:] = gated_cam[:, 1:] * gate_col
    else:
        # CAM is spatial-only
        gated_cam = cam_pos_avg * feature_gate.unsqueeze(0)

    # Compute attribution delta (how much did gating change the CAM?)
    cam_delta = gated_cam - cam_pos_avg

    # Extract per-patch attribution deltas (signed sum of changes per column)
    # Positive = boosted attribution, Negative = deboosted attribution
    # This represents the net change in each patch's incoming attention
    if cam_pos_avg.shape[0] == feature_gate.shape[0] + 1:
        # Has CLS token, compute delta for spatial patches only
        patch_attribution_deltas = cam_delta[:, 1:].sum(dim=0)  # [196]
    else:
        # Spatial only
        patch_attribution_deltas = cam_delta.sum(dim=0)  # [196]

    # Compile debug info
    debug_info = {
        'feature_gating': feature_debug,
        'combined_gate': feature_gate.detach().cpu().numpy() if debug else None,
        'cam_delta': cam_delta.detach().cpu().numpy() if debug else None,
        'patch_attribution_deltas': patch_attribution_deltas.detach().cpu().numpy() if debug else None,
    }

    return gated_cam, debug_info
