# attribution.py
import gc
from typing import Any, Dict, List, Optional, Tuple, Union  # Added Union

import numpy as np
import torch
import torch.nn.functional as F

import vit.model as model_handler
from config import PipelineConfig
from translrp.ViT_new import Block, VisionTransformer


def avg_heads(cam: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    cam = cam.cpu()
    grad = grad.cpu()
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def avg_heads_min(cam: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    cam = cam.cpu()
    grad = grad.cpu()
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(max=0).abs().mean(dim=0)
    return cam


def apply_self_attention_rules(R_ss: torch.Tensor, cam_ss: torch.Tensor) -> torch.Tensor:
    R_ss = R_ss.cpu()
    cam_ss = cam_ss.cpu()
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def gini_based_normalization(
    attention_head: torch.Tensor,
    gini_threshold: float = 0.7,
    steepness: float = 4.0,
    max_power: float = 0.7
) -> torch.Tensor:
    sorted_attn = torch.sort(attention_head.flatten())[0]
    n = sorted_attn.numel()
    index = torch.arange(1, n + 1, device=sorted_attn.device)
    gini = torch.sum((2 * index - n - 1) * sorted_attn) / (n * torch.sum(sorted_attn) + 1e-10)
    transformation_factor = 1.0 - (1.0 / (1.0 + torch.exp(-steepness * (gini - gini_threshold))))
    power = 1.0 - (max_power * transformation_factor)
    transformed = torch.pow(attention_head, power)
    return transformed * (torch.sum(attention_head) / (torch.sum(transformed) + 1e-10))


def calculate_ffn_activity(ffn_input: torch.Tensor, ffn_output: torch.Tensor) -> torch.Tensor:
    input_norm = F.normalize(ffn_input, p=2, dim=-1)
    output_norm = F.normalize(ffn_output, p=2, dim=-1)
    cosine_sim = torch.sum(input_norm * output_norm, dim=-1)
    directional_change = 1 - cosine_sim
    input_lengths = torch.norm(ffn_input, p=2, dim=-1)
    output_lengths = torch.norm(ffn_output, p=2, dim=-1)
    eps = 1e-8
    length_ratio = output_lengths / (input_lengths + eps)
    magnitude_change = torch.abs(torch.log(length_ratio + eps))
    ffn_activity = directional_change * magnitude_change
    return ffn_activity


def adaptive_hybrid_masking(
    layer_idx: int,
    target_class: int,
    config: PipelineConfig,
    num_total_tokens: int = 197,
    num_heads: int = 12
) -> torch.Tensor:
    """
    Creates a hybrid attention mask that boosts heads generally and tokens specifically.
    """
    attention_mask_boost = torch.ones((1, 1, num_heads, num_total_tokens, num_total_tokens))

    base_boost = config.classify.token_boost_value
    class_multiplier = config.classify.class_boost_multipliers.get(target_class, 1.0)
    actual_boost_value = base_boost * class_multiplier

    token_boost_map = config.classify.token_boost_factors.get(target_class, {})
    token_boost_for_layer = token_boost_map.get(layer_idx, {})

    if token_boost_for_layer:
        for head_id, token_ids in token_boost_for_layer.items():
            for token_id in token_ids:
                attention_mask_boost[:, :, head_id, token_id, :] = actual_boost_value
                attention_mask_boost[:, :, head_id, :, token_id] = actual_boost_value

    return attention_mask_boost


def adaptive_hybrid_masking_debug(
    layer_idx: int,
    target_class: int,
    config: PipelineConfig,
    num_total_tokens: int = 197,
    num_heads: int = 12
) -> torch.Tensor:
    """
    Debug version that shows what's being boosted
    """
    attention_mask_boost = torch.ones((1, 1, num_heads, num_total_tokens, num_total_tokens))

    token_boost_map = config.classify.token_boost_factors.get(target_class, {})
    token_boost_for_layer = token_boost_map.get(layer_idx, {})

    if token_boost_for_layer:
        for head_id, token_ids in token_boost_for_layer.items():
            for token_id in token_ids:
                # Try both directions to see which works
                attention_mask_boost[:, :, head_id, token_id, :] = config.classify.token_boost_value  # Outgoing
                attention_mask_boost[:, :, head_id, :, token_id] = config.classify.token_boost_value  # Incoming

    return attention_mask_boost


def adaptive_weighting_per_head(layer_idx: int, target_class: int, config: PipelineConfig) -> torch.Tensor:
    cam_ones = torch.ones((1, 1, 12, 197, 197))
    head_boost_factor = config.classify.head_boost_factor_per_head_per_class[target_class]
    head_ids = head_boost_factor.get(layer_idx, None)

    if head_ids:
        for head_id in head_ids:
            cam_ones[:, :, head_id, :, :] *= config.classify.head_boost_value

    return cam_ones


def adaptive_weighting(layer_idx: int, target_class: int, blk: Block, model_nn: VisionTransformer) -> torch.Tensor:
    batch_size = blk.attn.output_tokens.shape[0]
    seq_len = blk.attn.output_tokens.shape[1]
    weights_rows = torch.ones((batch_size * seq_len, 1), device='cpu')

    correlation_boost_factor_class2 = 1.2
    correlation_boost_factor_class1 = 1.7

    if target_class == 2:
        boost_factor_formula = lambda x: (
            1 + x.get(layer_idx, 0.0)
        ) * correlation_boost_factor_class2 if layer_idx in x else 1.0

        # Get attention outputs and compute class logits
        attn_output = blk.attn.output_tokens.detach()
        attn_class_logits = model_nn.get_class_embedding_space_representation(attn_output)
        attn_target_logits = attn_class_logits[:, :, target_class]
        argmax_classes = torch.argmax(attn_class_logits, dim=2)
        argmax_mask = (argmax_classes == target_class).float()

        # Set threshold percentile (always 0.0 for target_class 2)
        threshold_percentile = 0.0

        # Calculate threshold and create boost mask
        attn_threshold = torch.quantile(attn_target_logits.flatten(), threshold_percentile)
        attn_boost_mask = (attn_target_logits > attn_threshold).float()

        # Apply layer-specific boost factors
        layer_boost_factors = {7: 0.45, 8: 0.43, 9: 0.51, 10: 0.48, 11: 0.52}

        class_boost_factors = {
            5: 0.428,
            8: 0.29,
        }

        boost_factor = boost_factor_formula(layer_boost_factors)
        cls_boost_factor = boost_factor_formula(class_boost_factors)

        # Create weights with boosting
        boost_mask = attn_boost_mask * argmax_mask
        weights = torch.ones_like(attn_boost_mask) + boost_factor * boost_mask

        # Special boost for CLS token
        weights[:, 0] *= cls_boost_factor

        # Convert to CPU and reshape
        weights = weights.cpu()
        weights_rows = weights.view(-1, 1)

    elif target_class == 1 and layer_idx >= 8:
        boost_factor_formula = lambda x: (
            1 + x.get(layer_idx, 0.0)
        ) * correlation_boost_factor_class1 if layer_idx in x else 1.0

        # Get attention outputs and compute class logits
        attn_output = blk.attn.output_tokens.detach()
        attn_class_logits = model_nn.get_class_embedding_space_representation(attn_output)

        class_alignment_change = {7: 2, 8: 2, 9: 2, 10: 1, 11: 1}

        layer_boost_factors = {
            7: 0.36,
            9: 0.20,
            10: 0.18,
            # 11: 0.36,
        }
        class_boost_factors = {7: 0.22, 8: 0.19, 10: 0.43, 11: 0.1}

        # Different handling for specific layers
        class_alignment = class_alignment_change.get(layer_idx, 1)
        attn_target_logits = attn_class_logits[:, :, class_alignment]
        argmax_classes = torch.argmax(attn_class_logits, dim=2)
        argmax_mask = (argmax_classes == target_class).float()
        threshold_percentile = 0.0

        # Calculate threshold and create boost mask
        attn_threshold = torch.quantile(attn_target_logits.flatten(), threshold_percentile)
        attn_boost_mask = (attn_target_logits > attn_threshold).float()

        boost_factor = boost_factor_formula(layer_boost_factors)
        cls_boost_factor = boost_factor_formula(class_boost_factors)
        # Create weights with boosting
        boost_mask = attn_boost_mask * argmax_mask
        weights = torch.ones_like(attn_boost_mask) + boost_factor * boost_mask

        # Special boost for CLS token
        weights[:, 0] *= cls_boost_factor

        # Convert to CPU and reshape
        weights = weights.cpu()
        weights_rows = weights.view(-1, 1)

    return weights_rows


def class_embedding_representation_data_capture(blk: Block, i: int, model_nn: VisionTransformer) -> Dict[str, Any]:
    """
    Collects class embedding representations for attention and MLP outputs per layer.
    Returns data structured for FFNActivityItem and ClassEmbeddingRepresentationItem.
    """
    attn_input_tokens = getattr(blk.attn, 'input_tokens', None)
    attn_output_tokens = getattr(blk.attn, 'output_tokens', None)
    attn_map = getattr(blk.attn, 'attention_map', None).detach().cpu().squeeze(0).numpy()

    # MLP part
    mlp_input_tokens = getattr(blk.mlp, 'input_tokens', None)
    mlp_output_tokens = getattr(blk.mlp, 'output_tokens', None)

    def get_representation(tokens):
        return model_nn.get_class_embedding_space_representation(tokens).detach().cpu().squeeze(0).numpy()

    attn_class_repr_data = get_representation(attn_output_tokens)
    mlp_class_repr_data = get_representation(mlp_output_tokens)

    return {
        'layer': i,
        'attention_class_representation_output': attn_class_repr_data,
        'mlp_class_representation_output': mlp_class_repr_data,
        'attention_map': attn_map,
        'attention_class_representation_input': get_representation(attn_input_tokens),
        'mlp_class_representation_input': get_representation(mlp_input_tokens),
    }


def ffn_activities_data_capture(blk: Block, i: int) -> Dict[str, Any]:
    ffn_input = getattr(blk.mlp, 'input_tokens', None)
    ffn_output = getattr(blk.mlp, 'output_tokens', None)
    ffn_activity = {}

    if ffn_input is not None and ffn_output is not None:
        ffn_input_cpu = ffn_input.detach().cpu()
        ffn_output_cpu = ffn_output.detach().cpu()
        ffn_activity_metric = calculate_ffn_activity(ffn_input_cpu, ffn_output_cpu)  # Returns CPU tensor

        activity_data_np = ffn_activity_metric.squeeze(0).numpy() if ffn_activity_metric.dim(
        ) > 1 else ffn_activity_metric.numpy()

        ffn_activity = {
            'layer':
            i,
            'activity':
            activity_data_np,  # This is the np.ndarray for 'activity_data'
            'mean_activity':
            ffn_activity_metric.mean().item(),
            'cls_activity':
            ffn_activity_metric[0, 0].item() if ffn_activity_metric.dim() > 1 and ffn_activity_metric.shape[0] > 0 and
            ffn_activity_metric.shape[1] > 0 else (
                ffn_activity_metric[0].item()
                if ffn_activity_metric.dim() > 0 and ffn_activity_metric.shape[0] > 0 else 0.0
            )
        }
    return ffn_activity


def head_contribution_data_capture(blk: Block, i: int) -> Dict[str, Any]:
    """Capture head contributions after projection"""
    head_outputs = blk.attn.head_contributions  # Shape: [b, h, n, d_head]
    b, h, n, d_head = head_outputs.shape
    h_dim = h * d_head

    # Process all heads at once for efficiency
    contributions = []
    for head_idx in range(h):
        # Create tensor with only this head's output
        single_head = torch.zeros(b, n, h_dim, device=head_outputs.device)
        single_head[:, :, head_idx * d_head:(head_idx + 1) * d_head] = head_outputs[:, head_idx]

        # Apply projection
        head_contrib = blk.attn.proj(single_head)
        contributions.append(head_contrib)

    # Stack and convert to numpy
    stacked_contributions = torch.stack(contributions)  # Shape: [h, b, n, d_model]
    return {'layer': i, 'stacked_contribution': stacked_contributions.detach().cpu().numpy()}


def transmm(
    model_nn: VisionTransformer,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    target_class: Optional[int] = None,
    gini_params: Optional[Tuple[float, float, float]] = None,
    device: Optional[torch.device] = None,
    img_size: int = 224,
) -> Tuple[Dict[str, Any], np.ndarray, Optional[np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[
    str, Any]]]:
    """
    Memory-efficient implementation of TransMM.
    Returns: (prediction_dict, positive_attr_np, negative_attr_np_or_None, ffn_activity_list, cer_list)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    model_nn.zero_grad()

    prediction_result_dict = model_handler.get_prediction(model_nn, input_tensor, device=device, eval=False)

    logits = prediction_result_dict["logits"]
    effective_target_class = prediction_result_dict["predicted_class_idx"]

    # Create one-hot vector for the target class
    one_hot = torch.zeros((1, logits.size(-1)), dtype=torch.float32, device=device)
    one_hot[0, effective_target_class] = 1
    one_hot.requires_grad_(True)

    loss = torch.sum(one_hot * logits)

    model_nn.zero_grad()
    loss.backward(retain_graph=True)

    num_tokens = model_nn.blocks[0].attn.get_attention_map().shape[-1]
    R_pos = torch.eye(num_tokens, num_tokens, device='cpu')

    class_embedding_data_list: List[Dict[str, Any]] = []
    head_contribution_list: List[Dict[str, Any]] = []
    collected_ffn_activities: List[Dict[str, Any]] = []
    for i, blk in enumerate(model_nn.blocks):
        grad = blk.attn.get_attn_gradients().detach()
        cam = blk.attn.get_attention_map().detach().cpu()

        if config.file.weighted:
            # cam = cam * adaptive_weighting_per_head(i, effective_target_class, config)
            cam = cam * adaptive_hybrid_masking(i, effective_target_class, config)

        if gini_params:
            gini_threshold, steepness, max_power = gini_params
            cam = gini_based_normalization(cam, gini_threshold, steepness, max_power)

        cam_pos_avg = avg_heads(cam, grad)  # Returns CPU tensor

        if config.file.weighted and False:
            cam_pos_avg = cam_pos_avg * adaptive_weighting(i, effective_target_class, blk, model_nn)

        R_pos = R_pos + apply_self_attention_rules(R_pos, cam_pos_avg)

        if config.classify.data_collection:
            # collected_ffn_activities.append(ffn_activities_data_capture(blk, i))
            # class_embedding_data_list.append(class_embedding_representation_data_capture(blk, i, model_nn))
            if i >= 8:
                head_contribution_list.append(head_contribution_data_capture(blk, i))

        del grad, cam, cam_pos_avg  #, cam_neg_avg

    transformer_attribution_pos = R_pos[0, 1:].clone()
    del R_pos  #, R_neg

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
    torch.cuda.empty_cache()
    gc.collect()

    if isinstance(prediction_result_dict["probabilities"], torch.Tensor):
        prediction_result_dict["probabilities"] = prediction_result_dict["probabilities"].tolist()

    return (
        prediction_result_dict,  # Dict from model_handler.get_prediction
        attribution_pos_np,
        logits.cpu().detach().numpy() if config.classify.data_collection else None,  # Optional[np.ndarray]
        collected_ffn_activities,  # List[Dict for FFNActivityItem]
        class_embedding_data_list,  # List[Dict for ClassEmbeddingRepresentationItem]
        head_contribution_list
    )


def generate_attribution(
    model: VisionTransformer,  # Type hint for clarity
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
    img_size: int = 224,
    **kwargs: Any  # For other params like gini_params
) -> Dict[str, Any]:
    """
    Unified interface for generating attribution maps.
    Returns a dictionary formatted for pipeline.py.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure input_tensor is on the correct device
    input_tensor = input_tensor.to(device)

    if config.classify.attribution_method.lower() == "transmm":
        gini_params = kwargs.get('gini_params')
        weigh_by_class_embedding = kwargs.get('weigh_by_class_embedding', False)

        (pred_dict, pos_attr_np, logits, ffn_list, cer_list, head_contribution) = transmm(
            model_nn=model,  # Pass the VisionTransformer instance
            input_tensor=input_tensor,
            config=config,
            target_class=target_class,
            gini_params=gini_params,
            device=device,
            img_size=img_size
        )

        # Structure the output dictionary as expected by pipeline.py
        return {
            "predictions": pred_dict,  # The dict from model_handler.get_prediction via transmm
            "attribution_positive": pos_attr_np,
            "logits": logits,  # This will be None if transmm returns None
            "ffn_activity": ffn_list,  # List of dicts
            "class_embedding_representation": cer_list,  # List of dicts
            "head_contribution": head_contribution
        }
    else:
        raise ValueError(f"Unsupported attribution method: {method}")
