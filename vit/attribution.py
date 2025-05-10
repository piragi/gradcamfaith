# attribution.py
import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import vit.model as model_handler
from translrp.ViT_new import Block, VisionTransformer


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
                             gini_threshold: float = 0.7,
                             steepness: float = 4.0,
                             max_power: float = 0.7) -> torch.Tensor:
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
    weigh_by_class_embedding: bool = False,
    data_collection: bool = False
) -> Tuple[Dict, np.ndarray, None, Optional[List[Dict]], Optional[List[Dict]]]:
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

    model.zero_grad()

    #output = model(input_tensor, register_hook=True)
    outputs = model_handler.get_prediction(model, input_tensor, eval=False)
    output = outputs['logits']
    target_class = outputs['predicted_class_idx']

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
        # if weigh_by_class_embedding:
        # cam = apply_head_specific_weighting(cam, blk, i, target_class,
        # model)

        # Process on CPU to save GPU memory
        cam_pos = avg_heads(cam, grad)
        # cam_neg = avg_heads_min(cam, grad)

        if weigh_by_class_embedding:
            cam_pos = cam_pos * adaptive_weighting(i, target_class, blk, model)

        R = R + apply_self_attention_rules(R, cam_pos)
        # R_neg = R_neg + apply_self_attention_rules(R_neg, cam_neg)

        if data_collection:
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

        del grad, cam, cam_pos  #, cam_neg
        if data_collection: del ffn_input, ffn_output, ffn_activity

    # Extract patch tokens relevance (excluding CLS token)
    transformer_attribution = R[0, 1:].clone()
    # transformer_attribution_neg = R_neg[0, 1:].clone()
    del R  # , R_neg

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
    # attribution_neg = process_attribution(transformer_attribution_neg)

    # Clean up
    del transformer_attribution, input_tensor, output, one_hot, loss
    torch.cuda.empty_cache()
    gc.collect()

    # Normalize
    normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)

    attribution = normalize(attribution)
    # attribution_neg = normalize(attribution_neg)

    class_embedding_representations = token_class_embedding_representation(
        model) if data_collection else None

    return outputs, attribution, None, ffn_activities, class_embedding_representations


def apply_head_specific_weighting(cam, blk, layer_idx, target_class, model):
    """Boost specific attention connections based on cross-class information flow - conservative version"""

    if target_class != 0:  # Only for COVID
        return cam

    batch_size, num_heads, seq_len_q, seq_len_k = cam.shape

    # Get token representations
    input_tokens = blk.attn.input_tokens
    output_tokens = blk.attn.output_tokens

    # Get class identifiability
    input_logits = model.get_class_embedding_space_representation(input_tokens)
    output_logits = model.get_class_embedding_space_representation(
        output_tokens)

    # Identify source tokens (high class 1/2 identifiability)
    class1_sources = (input_logits[:, :, 1] > input_logits[:, :,
                                                           1].mean()).float()
    class2_sources = (input_logits[:, :, 2] > input_logits[:, :,
                                                           2].mean()).float()

    # Identify COVID receivers (tokens gaining COVID representation)
    covid_gain = output_logits[:, :, 0] - input_logits[:, :, 0]
    covid_receivers = (covid_gain > 0).float()

    # Critical heads based on correlation data
    critical_heads = {
        0: {
            3: 0.245,
            4: 0.238,
            5: 0.229,
            7: 0.198,
            9: 0.197
        },
        1: {
            11: 0.204
        },
        5: {
            6: 0.185,
            8: 0.196,
            9: 0.183,
            10: 0.202
        },
        6: {
            0: 0.190,
            7: 0.240
        },
        7: {
            4: 0.263,
            5: 0.254
        },
        9: {
            2: 0.184,
            5: 0.210,
            7: 0.207
        },
        10: {
            9: 0.186,
            10: 0.181
        },
        11: {
            1: 0.180,
            8: 0.251,
            9: 0.182
        }
    }

    # Create attention weight mask
    weight_mask = torch.ones_like(cam)

    if layer_idx in critical_heads:
        for head_idx, correlation in critical_heads[layer_idx].items():
            # For this head, identify important connections
            if layer_idx <= 1:
                # Early layers: boost attention from class 2 sources
                important_sources = class2_sources
            elif 5 <= layer_idx <= 7:
                # Middle layers: boost from both class 1 and 2
                important_sources = class1_sources + class2_sources
            else:
                # Late layers: boost class 1 sources more (contrast)
                important_sources = class1_sources + 0.5 * class2_sources

            # Create connection importance matrix
            connection_importance = torch.einsum('bi,bj->bij', covid_receivers,
                                                 important_sources)

            # CONSERVATIVE BOOST: Use much smaller factors
            # Instead of 1.0 + correlation * 2.0, use 1.0 + correlation * 0.2
            boost_factor = 1.0 + correlation * 0.2  # Max boost ~1.05 even for highest correlation

            weight_mask[:, head_idx, :, :] = torch.where(
                connection_importance > 0,
                torch.full_like(connection_importance, boost_factor),
                torch.ones_like(connection_importance))

    # Apply targeted weighting
    weighted_cam = cam * weight_mask

    return weighted_cam


def token_class_embedding_representation(model: VisionTransformer):
    class_embedding_representations = []
    for i, blk in enumerate(model.blocks):
        attention_class_representation_input = model.get_class_embedding_space_representation(
            blk.attn.input_tokens).detach().cpu()
        attention_map = blk.attn.attention_map.detach().cpu()
        attention_class_representation = model.get_class_embedding_space_representation(
            blk.attn.output_tokens).detach().cpu()
        mlp_class_representation_input = model.get_class_embedding_space_representation(
            blk.mlp.input_tokens).detach().cpu()
        mlp_class_representation = model.get_class_embedding_space_representation(
            blk.mlp.output_tokens).detach().cpu()
        class_embedding_representations.append({
            'layer':
            i,
            'attention_class_representation_input':
            attention_class_representation_input.squeeze(0).numpy(),
            'attention_map':
            attention_map.squeeze(0).numpy(),
            'attention_class_representation':
            attention_class_representation.squeeze(0).numpy(),
            'mlp_class_representation_input':
            mlp_class_representation_input.squeeze(0).numpy(),
            'mlp_class_representation':
            mlp_class_representation.squeeze(0).numpy()
        })

    return class_embedding_representations


def adaptive_weighting(layer_idx: int, target_class: int, blk: Block,
                       model: VisionTransformer) -> torch.Tensor:
    # Get batch size and sequence length for default weights
    batch_size = blk.attn.output_tokens.shape[0]
    seq_len = blk.attn.output_tokens.shape[1]

    weights_rows = torch.ones((batch_size * seq_len, 1), device='cpu')

    if target_class == 2 and layer_idx >= 7:
        # Enhanced approach for class 2 (normal class) based on correlation analysis

        # For class 2, leverage both attention and MLP outputs
        attn_output = blk.attn.output_tokens.detach()
        mlp_output = blk.mlp.output_tokens.detach()

        # Get class logits for both outputs
        attn_class_logits = model.get_class_embedding_space_representation(
            attn_output)
        mlp_class_logits = model.get_class_embedding_space_representation(
            mlp_output)

        # Get target class logits
        attn_target_logits = attn_class_logits[:, :, target_class]
        mlp_target_logits = mlp_class_logits[:, :, target_class]

        # Define weights for combining attention and MLP features based on correlations
        # if layer_idx == 7:
        # # Layer 7: strong correlations in attention
        # attn_weight, mlp_weight = 0.8, 0.2
        # threshold_percentile = 0.0
        # elif layer_idx == 8:
        # # Layer 8: strong in both attention and MLP
        # attn_weight, mlp_weight = 0.5, 0.5
        # threshold_percentile = 0.0
        # elif layer_idx == 9:
        # # Layer 9: strongest in attention, negative in MLP
        # attn_weight, mlp_weight = 1.0, 0.0
        # threshold_percentile = 0.0
        if layer_idx == 10:
            # Layer 10: strongest overall in attention and strong in MLP
            attn_weight, mlp_weight = 0.6, 0.4
            threshold_percentile = 0.0
        elif layer_idx == 11:
            # Layer 10: strongest overall in attention and strong in MLP
            attn_weight, mlp_weight = 0.9, 0.1
            threshold_percentile = 0.0
        else:
            # Default weights for other layers
            attn_weight, mlp_weight = 0.5, 0.5
            threshold_percentile = 0.0

        # Create masks for tokens with high relevance
        attn_threshold = torch.quantile(attn_target_logits.flatten(),
                                        threshold_percentile)
        mlp_threshold = torch.quantile(mlp_target_logits.flatten(),
                                       threshold_percentile)

        attn_boost_mask = (attn_target_logits > attn_threshold).float()
        mlp_boost_mask = (mlp_target_logits > mlp_threshold).float()

        # Combine the masks with layer-specific weights
        combined_mask = attn_weight * attn_boost_mask
        if mlp_weight > 0:  # Only add MLP contribution if weight is positive
            combined_mask += mlp_weight * mlp_boost_mask

        # Layer-specific boost factors based on correlation strength
        layer_boost_factors = {
            7: 2.0,  # Layer 7: high correlation
            8: 2.2,  # Layer 8: very high correlation
            9: 2.5,  # Layer 9: very high correlation (attention)
            10: 5.0,  # Layer 10: highest correlation
            11: 1.2  # Layer 11: high correlation
        }
        boost_factor = layer_boost_factors.get(layer_idx, 1.0)

        # Create weights
        weights = torch.ones_like(combined_mask) + boost_factor * combined_mask

        # Apply extra boost to CLS token
        weights[:, 0] *= 1.5  # 50% more boost for CLS token

        weights = weights.cpu()
        weights_rows = weights.view(-1, 1)

    elif target_class == 1 and layer_idx >= 4:
        # Enhanced approach for class 1 (non-covid class) based on correlation analysis

        # For class 1, leverage both attention and MLP outputs
        attn_output = blk.attn.output_tokens.detach()
        mlp_output = blk.mlp.output_tokens.detach()

        # Get class logits for all classes from both outputs
        attn_class_logits = model.get_class_embedding_space_representation(
            attn_output)
        mlp_class_logits = model.get_class_embedding_space_representation(
            mlp_output)

        if layer_idx == 10:
            # Layer 10: Strong positive in class 1 attention, strong negative in class 0 attention/MLP
            attn_target_logits = attn_class_logits[:, :,
                                                   1]  # Positive in class 1 attention
            mlp_target_logits = -1.0 * attn_class_logits[:, :,
                                                         0]  # Negative in class 0, negate
            attn_weight, mlp_weight = 1.0, 0.0
            threshold_percentile = 0.0
            boost_factor = 2.5

        elif layer_idx == 11:
            # Layer 11: Strong correlation with class 2 (normal) attention representations
            # The strongest signals are class 2 attention logits and negative class 0
            attn_target_logits = attn_class_logits[:, :,
                                                   2]  # Class 2 attention (r=0.369)
            mlp_target_logits = -1.0 * mlp_class_logits[:, :,
                                                        0]  # Negative class 0 in MLP

            # Weight more on attention which showed stronger correlations
            attn_weight, mlp_weight = 1.0, 0.0
            threshold_percentile = 0.6
            boost_factor = 1.5  # Slightly lower than layer 10 to avoid overpowering it

        else:
            # Default parameters for other layers with weaker correlations
            attn_target_logits = attn_class_logits[:, :, 1]
            mlp_target_logits = mlp_class_logits[:, :, 1]
            attn_weight, mlp_weight = 0.5, 0.5
            threshold_percentile = 0.0
            boost_factor = 1.0

        # Create masks for tokens with high relevance
        attn_threshold = torch.quantile(attn_target_logits.flatten(),
                                        threshold_percentile)
        mlp_threshold = torch.quantile(mlp_target_logits.flatten(),
                                       threshold_percentile)

        attn_boost_mask = (attn_target_logits > attn_threshold).float()
        mlp_boost_mask = (mlp_target_logits > mlp_threshold).float()

        # Combine the masks with layer-specific weights
        combined_mask = attn_weight * attn_boost_mask
        if mlp_weight > 0:  # Only add MLP contribution if weight is positive
            combined_mask += mlp_weight * mlp_boost_mask

        # Create weights
        weights = torch.ones_like(combined_mask) + boost_factor * combined_mask

        # Apply extra boost to CLS token
        weights[:, 0] *= 1.5  # 50% more boost for CLS token

        weights = weights.cpu()
        weights_rows = weights.view(-1, 1)

    return weights_rows


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
        predictions, pos_attr, neg_attr, ffn_activity, class_embedding_representation = transmm(
            model=model,
            input_tensor=input_tensor,
            target_class=target_class,
            device=device,
            img_size=img_size,
            **kwargs)
        return {
            "method": "transmm",
            "predictions": predictions,
            "attribution_positive": pos_attr,
            "attribution_negative": neg_attr,
            "ffn_activity": ffn_activity,
            "class_embedding_representation": class_embedding_representation
        }

    else:
        raise ValueError(f"Unsupported attribution method: {method}")
