# attribution.py
import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from translrp.ViT_new import VisionTransformer


def avg_heads(cam: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Rule 5 from paper: Average attention heads weighted by gradients.
    
    Args:
        cam: Attention map tensor
        grad: Gradient tensor
        
    Returns:
        Weighted attention map
    """
    # Move to CPU to save GPU memory
    cam = cam.cpu()
    grad = grad.cpu()

    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def avg_heads_min(cam: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Rule 5 variant: Average attention heads weighted by gradients with negative values.
    
    Args:
        cam: Attention map tensor
        grad: Gradient tensor
        
    Returns:
        Weighted attention map for negative contributions
    """
    # Move to CPU to save GPU memory
    cam = cam.cpu()
    grad = grad.cpu()

    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(max=0).abs().mean(dim=0)
    return cam


def apply_self_attention_rules(R_ss: torch.Tensor,
                               cam_ss: torch.Tensor) -> torch.Tensor:
    """
    Rule 6 from paper: Apply self-attention propagation rule.
    
    Args:
        R_ss: Current relevance tensor
        cam_ss: Attention map tensor
        
    Returns:
        Updated relevance tensor
    """
    # Ensure both are on the same device (CPU to save memory)
    R_ss = R_ss.cpu()
    cam_ss = cam_ss.cpu()

    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def gini_based_normalization(attention_head: torch.Tensor,
                             gini_threshold: float = 0.65,
                             steepness: float = 8.0,
                             max_power: float = 0.5) -> torch.Tensor:
    """
    Apply power normalization based on Gini coefficient.
    Only transforms dispersed attention (low Gini), leaving concentrated attention (high Gini) unchanged.
    
    Args:
        attention_head: Attention tensor
        gini_threshold: Threshold for applying transformation
        steepness: Controls transform sharpness
        max_power: Maximum transform strength
        
    Returns:
        Normalized attention tensor
    """
    # Calculate Gini coefficient
    sorted_attn = torch.sort(attention_head.flatten())[0]
    n = sorted_attn.numel()
    index = torch.arange(1, n + 1, device=sorted_attn.device)
    gini = torch.sum((2 * index - n - 1) *
                     sorted_attn) / (n * torch.sum(sorted_attn) + 1e-10)

    # Calculate transformation factor using sigmoid function
    transformation_factor = 1.0 - (1.0 /
                                   (1.0 + torch.exp(-steepness *
                                                    (gini - gini_threshold))))

    # Calculate adaptive power
    power = 1.0 - (max_power * transformation_factor)

    # Apply power transformation
    transformed = torch.pow(attention_head, power)

    # Preserve the sum
    return transformed * (torch.sum(attention_head) /
                          (torch.sum(transformed) + 1e-10))


def calculate_transformation_weights(
        input_tokens: torch.Tensor,
        output_tokens: torch.Tensor) -> torch.Tensor:
    """
    Calculate token transformation weights based on length ratio and directional correlation.
    
    Args:
        input_tokens: Input token embeddings
        output_tokens: Output token embeddings
        
    Returns:
        Transformation weights
    """
    # Calculate L2 norm (length) for each token
    input_lengths = torch.norm(input_tokens, p=2,
                               dim=-1)  # [batch_size, num_tokens]
    output_lengths = torch.norm(output_tokens, p=2,
                                dim=-1)  # [batch_size, num_tokens]

    # Calculate ratio (handle division by zero)
    eps = 1e-8  # small epsilon to avoid division by zero
    length_ratio = output_lengths / (input_lengths + eps)

    # Normalize tokens for cosine similarity
    input_norm = F.normalize(input_tokens, p=2, dim=-1)
    output_norm = F.normalize(output_tokens, p=2, dim=-1)

    # Calculate cosine similarity between original and transformed tokens
    cosine_sim = torch.sum(input_norm * output_norm,
                           dim=-1)  # [batch_size, num_tokens]

    # Apply softmax to get NECC (across tokens for each batch)
    necc = F.softmax(cosine_sim, dim=-1)

    # Combine both measurements into final transformation weights
    return length_ratio * necc


def calculate_ffn_activity(ffn_input: torch.Tensor,
                           ffn_output: torch.Tensor) -> torch.Tensor:
    """
    Calculate FFN activity based on length change and directional shift.
    
    Args:
        ffn_input: Input tokens to FFN [batch_size, num_tokens, dim]
        ffn_output: Output tokens from FFN [batch_size, num_tokens, dim]
        
    Returns:
        FFN activity metric [batch_size, num_tokens]
    """
    # Normalize tokens for cosine similarity
    input_norm = F.normalize(ffn_input, p=2, dim=-1)
    output_norm = F.normalize(ffn_output, p=2, dim=-1)

    # Calculate cosine similarity
    cosine_sim = torch.sum(input_norm * output_norm, dim=-1)

    # Calculate directional change component: (1 - cos⟨Eᵢ, Ẽᵢ⟩)
    directional_change = 1 - cosine_sim

    # Calculate length ratios
    input_lengths = torch.norm(ffn_input, p=2, dim=-1)
    output_lengths = torch.norm(ffn_output, p=2, dim=-1)
    eps = 1e-8  # Avoid division by zero
    length_ratio = output_lengths / (input_lengths + eps)

    # Calculate magnitude change component: |log(L(Ẽᵢ)/L(Eᵢ))|
    magnitude_change = torch.abs(torch.log(length_ratio + eps))

    # Combine for final metric
    ffn_activity = directional_change * magnitude_change

    return ffn_activity


def transmm(
    model: VisionTransformer,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    gini_params: Optional[Tuple[float, float, float]] = None,
    device: Optional[torch.device] = None,
    img_size: int = 224,
    weigh_by_class_embedding: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[Dict]]:
    """
    Memory-efficient implementation of TransMM.
    
    Args:
        model: Vision Transformer model
        input_tensor: Preprocessed input tensor
        original_image: Original image as numpy array
        target_class: Target class index (if None, uses predicted class)
        pretransform: Whether to apply Gini-based normalization
        gini_params: Parameters for Gini normalization (threshold, steepness, max_power)
        device: Device for computation
        img_size: Image size for attribution map
        
    Returns:
        Tuple of (original image array, (positive attribution map, negative attribution map))
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure input tensor is on the correct device
    input_tensor = input_tensor.to(device)

    # Add batch dimension if needed
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # Forward pass without hooks first to determine class
    if target_class is None:
        with torch.no_grad():
            outputs = model(input_tensor.detach())
            target_class = outputs.argmax(dim=1).item()
            del outputs
            torch.cuda.empty_cache()

    model.zero_grad()

    output = model(input_tensor, register_hook=True)

    # Create one-hot vector on CPU to save memory
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, target_class] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)

    # Scale output by one_hot
    loss = torch.sum(one_hot * output)

    model.zero_grad()
    loss.backward(retain_graph=True)

    # Get attention maps and gradients
    num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).to('cpu')  # Keep on CPU
    R_neg = torch.eye(num_tokens, num_tokens).to('cpu')  # Keep on CPU
    ffn_activities = []

    # Process each block
    for i, blk in enumerate(model.blocks):
        grad = blk.attn.get_attn_gradients().detach()
        cam = blk.attn.get_attention_map().detach()

        # Preprocess attn maps for dispersion
        if gini_params:
            gini_threshold, steepness, max_power = gini_params
            cam = gini_based_normalization(cam,
                                           gini_threshold=gini_threshold,
                                           steepness=steepness,
                                           max_power=max_power)

        # Process on CPU to save GPU memory
        cam_pos = avg_heads(cam, grad)
        cam_neg = avg_heads_min(cam, grad)

        if weigh_by_class_embedding and i >= 7 and target_class >= 1:  # Only apply to layers 8-11
            if target_class == 1:
                output = blk.attn.output_tokens.detach()
            if target_class == 2:
                output = blk.mlp.output_tokens.detach()

            # Get class logits for each token
            class_logits = model.get_class_embedding_space_representation(
                output)
            target_class_logits = class_logits[:, :, target_class]

            # Find tokens with high class relevance (top 50%)
            threshold = torch.quantile(target_class_logits.flatten(), 0.0)
            boost_mask = (target_class_logits > threshold).float()

            # Create weights array (1.0 for normal, 1.2 for boosted)
            boost_factor = 1.5
            weights = torch.ones_like(boost_mask) + boost_factor * boost_mask
            weights = weights.cpu()
            # Apply weights to attention map
            weights_rows = weights.view(-1, 1)
            cam_pos = cam_pos * weights_rows

        R = R + apply_self_attention_rules(R, cam_pos)
        R_neg = R_neg + apply_self_attention_rules(R_neg, cam_neg)

        # Calculate FFN activity
        ffn_input = blk.mlp.input_tokens.detach().cpu()
        ffn_output = blk.mlp.output_tokens.detach().cpu()
        ffn_activity = calculate_ffn_activity(ffn_input, ffn_output)

        # Store activity data (convert to numpy to save memory)
        ffn_activities.append({
            'layer':
            i,
            'activity':
            ffn_activity.squeeze(0).numpy()
            if ffn_activity.dim() > 1 else ffn_activity.numpy(),
            'mean_activity':
            ffn_activity.mean().item(),
            'cls_activity':
            ffn_activity[0, 0].item()
            if ffn_activity.dim() > 1 else ffn_activity[0].item()
        })

        del grad, cam, cam_pos, cam_neg, ffn_input, ffn_output, ffn_activity

    # Extract patch tokens relevance (excluding CLS token)
    transformer_attribution = R[0, 1:].clone()
    transformer_attribution_neg = R_neg[0, 1:].clone()
    del R, R_neg

    # Process attributions
    def process_attribution(attribution: torch.Tensor) -> np.ndarray:
        # Reshape to patch grid
        side_size = int(np.sqrt(attribution.size(0)))
        attribution = attribution.reshape(1, 1, side_size, side_size)

        # Move back to GPU only for interpolation
        attribution = attribution.to(device)

        # Upscale to image size
        attribution = F.interpolate(attribution,
                                    size=(img_size, img_size),
                                    mode='bilinear')

        # Convert to numpy and reshape
        return attribution.reshape(img_size, img_size).cpu().detach().numpy()

    attribution = process_attribution(transformer_attribution)
    attribution_neg = process_attribution(transformer_attribution_neg)

    # Clean up
    del transformer_attribution, transformer_attribution_neg, input_tensor, output, one_hot, loss
    torch.cuda.empty_cache()
    gc.collect()

    # Normalize
    normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)

    attribution = normalize(attribution)
    attribution_neg = normalize(attribution_neg)

    class_embedding_representations = token_class_embedding_representation(
        model)

    return attribution, attribution_neg, ffn_activities, class_embedding_representations


def token_class_embedding_representation(model: VisionTransformer):
    class_embedding_representations = []
    for i, blk in enumerate(model.blocks):
        attention_class_representation = model.get_class_embedding_space_representation(
            blk.attn.output_tokens).detach().cpu()
        mlp_class_representation = model.get_class_embedding_space_representation(
            blk.mlp.output_tokens).detach().cpu()
        class_embedding_representations.append({
            'layer':
            i,
            'attention_class_representation':
            attention_class_representation.squeeze(0).numpy(),
            'mlp_class_representation':
            mlp_class_representation.squeeze(0).numpy()
        })

    return class_embedding_representations


def generate_attribution(model: VisionTransformer,
                         input_tensor: torch.Tensor,
                         method: str = "translrp",
                         target_class: Optional[int] = None,
                         device: Optional[torch.device] = None,
                         img_size: int = 224,
                         **kwargs) -> Dict[str, Any]:
    """
    Unified interface for generating attribution maps using different methods.
    
    Args:
        model: Vision Transformer model
        input_tensor: Preprocessed input tensor
        method: Attribution method ('translrp', 'transmm', 'gae_tokentm')
        target_class: Target class for attribution (None for predicted class)
        device: Device for computation
        img_size: Image size for attribution maps
        **kwargs: Additional method-specific parameters
        
    Returns:
        Dictionary with attribution results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if method.lower() == "transmm":
        pos_attr, neg_attr, ffn_activity, class_embedding_representation = transmm(
            model=model,
            input_tensor=input_tensor,
            target_class=target_class,
            device=device,
            img_size=img_size,
            **kwargs)
        return {
            "method": "transmm",
            "attribution_positive": pos_attr,
            "attribution_negative": neg_attr,
            "ffn_activity": ffn_activity,
            "class_embedding_representation": class_embedding_representation
        }

    else:
        raise ValueError(f"Unsupported attribution method: {method}")
