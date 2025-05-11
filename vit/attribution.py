# attribution.py
import gc
from typing import Any, Dict, List, Optional, Tuple, Union  # Added Union

import numpy as np
import torch
import torch.nn.functional as F

import vit.model as model_handler
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


def apply_self_attention_rules(R_ss: torch.Tensor,
                               cam_ss: torch.Tensor) -> torch.Tensor:
    R_ss = R_ss.cpu()
    cam_ss = cam_ss.cpu()
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def gini_based_normalization(attention_head: torch.Tensor,
                             gini_threshold: float = 0.7,
                             steepness: float = 4.0,
                             max_power: float = 0.7) -> torch.Tensor:
    sorted_attn = torch.sort(attention_head.flatten())[0]
    n = sorted_attn.numel()
    index = torch.arange(1, n + 1, device=sorted_attn.device)
    gini = torch.sum((2 * index - n - 1) *
                     sorted_attn) / (n * torch.sum(sorted_attn) + 1e-10)
    transformation_factor = 1.0 - (1.0 /
                                   (1.0 + torch.exp(-steepness *
                                                    (gini - gini_threshold))))
    power = 1.0 - (max_power * transformation_factor)
    transformed = torch.pow(attention_head, power)
    return transformed * (torch.sum(attention_head) /
                          (torch.sum(transformed) + 1e-10))


def calculate_ffn_activity(ffn_input: torch.Tensor,
                           ffn_output: torch.Tensor) -> torch.Tensor:
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


def adaptive_weighting(layer_idx: int, target_class: int, blk: Block,
                       model_nn: VisionTransformer) -> torch.Tensor:
    batch_size = blk.attn.output_tokens.shape[0]
    seq_len = blk.attn.output_tokens.shape[1]
    weights_rows = torch.ones((batch_size * seq_len, 1), device='cpu')
    if target_class == 2 and layer_idx >= 7:
        attn_output = blk.attn.output_tokens.detach()
        mlp_output = blk.mlp.output_tokens.detach()
        attn_class_logits = model_nn.get_class_embedding_space_representation(
            attn_output)  # model -> model_nn
        mlp_class_logits = model_nn.get_class_embedding_space_representation(
            mlp_output)  # model -> model_nn
        attn_target_logits = attn_class_logits[:, :, target_class]
        mlp_target_logits = mlp_class_logits[:, :, target_class]
        if layer_idx == 10:
            attn_weight, mlp_weight = 0.6, 0.4
            threshold_percentile = 0.0
        elif layer_idx == 11:
            attn_weight, mlp_weight = 0.9, 0.1
            threshold_percentile = 0.0
        else:
            attn_weight, mlp_weight = 0.5, 0.5
            threshold_percentile = 0.0
        attn_threshold = torch.quantile(attn_target_logits.flatten(),
                                        threshold_percentile)
        mlp_threshold = torch.quantile(mlp_target_logits.flatten(),
                                       threshold_percentile)
        attn_boost_mask = (attn_target_logits > attn_threshold).float()
        mlp_boost_mask = (mlp_target_logits > mlp_threshold).float()
        combined_mask = attn_weight * attn_boost_mask
        if mlp_weight > 0: combined_mask += mlp_weight * mlp_boost_mask
        layer_boost_factors = {7: 2.0, 8: 2.2, 9: 2.5, 10: 5.0, 11: 1.2}
        boost_factor = layer_boost_factors.get(layer_idx, 1.0)
        weights = torch.ones_like(combined_mask) + boost_factor * combined_mask
        weights[:, 0] *= 1.5
        weights = weights.cpu()
        weights_rows = weights.view(-1, 1)
    elif target_class == 1 and layer_idx >= 4:
        attn_output = blk.attn.output_tokens.detach()
        mlp_output = blk.mlp.output_tokens.detach()
        attn_class_logits = model_nn.get_class_embedding_space_representation(
            attn_output)  # model -> model_nn
        mlp_class_logits = model_nn.get_class_embedding_space_representation(
            mlp_output)  # model -> model_nn
        if layer_idx == 10:
            attn_target_logits = attn_class_logits[:, :, 1]
            mlp_target_logits = -1.0 * attn_class_logits[:, :, 0]
            attn_weight, mlp_weight = 1.0, 0.0
            threshold_percentile = 0.0
            boost_factor = 2.5
        elif layer_idx == 11:
            attn_target_logits = attn_class_logits[:, :, 2]
            mlp_target_logits = -1.0 * mlp_class_logits[:, :, 0]
            attn_weight, mlp_weight = 1.0, 0.0
            threshold_percentile = 0.6
            boost_factor = 1.5
        else:
            attn_target_logits = attn_class_logits[:, :, 1]
            mlp_target_logits = mlp_class_logits[:, :, 1]
            attn_weight, mlp_weight = 0.5, 0.5
            threshold_percentile = 0.0
            boost_factor = 1.0
        attn_threshold = torch.quantile(attn_target_logits.flatten(),
                                        threshold_percentile)
        mlp_threshold = torch.quantile(mlp_target_logits.flatten(),
                                       threshold_percentile)
        attn_boost_mask = (attn_target_logits > attn_threshold).float()
        mlp_boost_mask = (mlp_target_logits > mlp_threshold).float()
        combined_mask = attn_weight * attn_boost_mask
        if mlp_weight > 0: combined_mask += mlp_weight * mlp_boost_mask
        weights = torch.ones_like(combined_mask) + boost_factor * combined_mask
        weights[:, 0] *= 1.5
        weights = weights.cpu()
        weights_rows = weights.view(-1, 1)
    return weights_rows


def token_class_embedding_representation(
        model_nn: VisionTransformer) -> List[Dict[str, Any]]:
    """
    Collects class embedding representations for attention and MLP outputs per layer.
    Returns data structured for FFNActivityItem and ClassEmbeddingRepresentationItem.
    """
    class_embedding_data_list: List[Dict[str, Any]] = []
    if not hasattr(model_nn, 'blocks') or not isinstance(
            model_nn.blocks, torch.nn.ModuleList):
        print(
            "Warning: Model does not have 'blocks' or it's not a ModuleList. Skipping CER collection."
        )
        return class_embedding_data_list

    for i, blk in enumerate(model_nn.blocks):
        attn_input_tokens = getattr(blk.attn, 'input_tokens', None)
        attn_output_tokens = getattr(blk.attn, 'output_tokens', None)

        # MLP part
        mlp_input_tokens = getattr(blk.mlp, 'input_tokens', None)
        mlp_output_tokens = getattr(blk.mlp, 'output_tokens', None)

        def get_representation(tokens):
            return model_nn.get_class_embedding_space_representation(
                tokens).detach().cpu().squeeze(0).numpy()

        attn_class_repr_data = get_representation(attn_output_tokens)
        mlp_class_repr_data = get_representation(mlp_output_tokens)

        class_embedding_data_list.append({
            'layer':
            i,
            'attention_class_representation_output':
            attn_class_repr_data,
            'mlp_class_representation_output':
            mlp_class_repr_data,
            'attention_map':
            blk.attn.attention_map.detach().cpu().squeeze(0).numpy()
            if hasattr(blk.attn, 'attention_map')
            and blk.attn.attention_map is not None else np.array([]),
            'attention_class_representation_input':
            get_representation(attn_input_tokens),
            'mlp_class_representation_input':
            get_representation(mlp_input_tokens),
        })
    return class_embedding_data_list


def transmm(
    model_nn: VisionTransformer,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    gini_params: Optional[Tuple[float, float, float]] = None,
    device: Optional[torch.device] = None,
    img_size: int = 224,
    weigh_by_class_embedding: bool = True,
    data_collection: bool = False
) -> Tuple[Dict[str, Any], np.ndarray, Optional[np.ndarray], List[Dict[
        str, Any]], List[Dict[str, Any]]]:
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

    prediction_result_dict = model_handler.get_prediction(model_nn,
                                                          input_tensor,
                                                          device=device,
                                                          eval=False)

    logits = prediction_result_dict["logits"]
    effective_target_class = prediction_result_dict["predicted_class_idx"]

    # Create one-hot vector for the target class
    one_hot = torch.zeros((1, logits.size(-1)),
                          dtype=torch.float32,
                          device=device)
    one_hot[0, effective_target_class] = 1
    one_hot.requires_grad_(True)

    loss = torch.sum(one_hot * logits)

    model_nn.zero_grad()
    loss.backward(retain_graph=True)

    num_tokens = model_nn.blocks[0].attn.get_attention_map().shape[-1]
    R_pos = torch.eye(num_tokens, num_tokens, device='cpu')
    # R_neg = torch.eye(num_tokens, num_tokens, device='cpu') # If negative attribution is calculated

    collected_ffn_activities: List[Dict[str, Any]] = []

    for i, blk in enumerate(model_nn.blocks):
        grad = blk.attn.get_attn_gradients().detach()
        cam = blk.attn.get_attention_map().detach()

        if gini_params:
            gini_threshold, steepness, max_power = gini_params
            cam = gini_based_normalization(cam, gini_threshold, steepness,
                                           max_power)

        cam_pos_avg = avg_heads(cam, grad)  # Returns CPU tensor
        # cam_neg_avg = avg_heads_min(cam, grad) # If negative attribution calculated

        if weigh_by_class_embedding:
            cam_pos_avg = cam_pos_avg * adaptive_weighting(
                i, effective_target_class, blk, model_nn)

        R_pos = R_pos + apply_self_attention_rules(R_pos, cam_pos_avg)
        # R_neg = R_neg + apply_self_attention_rules(R_neg, cam_neg_avg)

        if data_collection:
            ffn_input = getattr(blk.mlp, 'input_tokens', None)
            ffn_output = getattr(blk.mlp, 'output_tokens', None)

            if ffn_input is not None and ffn_output is not None:
                ffn_input_cpu = ffn_input.detach().cpu()
                ffn_output_cpu = ffn_output.detach().cpu()
                ffn_activity_metric = calculate_ffn_activity(
                    ffn_input_cpu, ffn_output_cpu)  # Returns CPU tensor

                activity_data_np = ffn_activity_metric.squeeze(
                    0).numpy() if ffn_activity_metric.dim(
                    ) > 1 else ffn_activity_metric.numpy()

                collected_ffn_activities.append({
                    'layer':
                    i,
                    'activity':
                    activity_data_np,  # This is the np.ndarray for 'activity_data'
                    'mean_activity':
                    ffn_activity_metric.mean().item(),
                    'cls_activity':
                    ffn_activity_metric[0, 0].item()
                    if ffn_activity_metric.dim() > 1
                    and ffn_activity_metric.shape[0] > 0
                    and ffn_activity_metric.shape[1] > 0 else
                    (ffn_activity_metric[0].item()
                     if ffn_activity_metric.dim() > 0
                     and ffn_activity_metric.shape[0] > 0 else 0.0)
                })
                del ffn_input_cpu, ffn_output_cpu, ffn_activity_metric, activity_data_np
            del ffn_input, ffn_output

        del grad, cam, cam_pos_avg  #, cam_neg_avg

    transformer_attribution_pos = R_pos[0, 1:].clone()
    # transformer_attribution_neg = R_neg[0, 1:].clone()
    del R_pos  #, R_neg

    def process_attribution_map(attr_tensor: torch.Tensor) -> np.ndarray:
        side_len = int(np.sqrt(attr_tensor.size(0)))
        attr_tensor = attr_tensor.reshape(1, 1, side_len, side_len)
        attr_tensor_device = attr_tensor.to(device)
        attr_interpolated = F.interpolate(attr_tensor_device,
                                          size=(img_size, img_size),
                                          mode='bilinear',
                                          align_corners=False)
        return attr_interpolated.squeeze().cpu().detach().numpy()

    attribution_pos_np = process_attribution_map(transformer_attribution_pos)
    # attribution_neg_np = process_attribution_map(transformer_attribution_neg) # If calculated
    attribution_neg_np = None  # Per current transmm, it returns None for negative

    # Normalize
    normalize_fn = lambda x: (x - np.min(x)) / (np.max(x) - np.min(
        x) + 1e-8) if (np.max(x) - np.min(x)) > 1e-8 else x

    attribution_pos_np = normalize_fn(attribution_pos_np)
    # attribution_neg_np = normalize_fn(attribution_neg_np)

    collected_class_embedding_representations: List[Dict[str, Any]] = []
    if data_collection:
        collected_class_embedding_representations = token_class_embedding_representation(
            model_nn)

    # Clean up
    del transformer_attribution_pos, input_tensor, logits, one_hot, loss
    if 'transformer_attribution_neg' in locals():
        del transformer_attribution_neg
    torch.cuda.empty_cache()
    gc.collect()

    if isinstance(prediction_result_dict["probabilities"], torch.Tensor):
        prediction_result_dict["probabilities"] = prediction_result_dict[
            "probabilities"].tolist()

    return (
        prediction_result_dict,  # Dict from model_handler.get_prediction
        attribution_pos_np,
        attribution_neg_np,  # Optional[np.ndarray]
        collected_ffn_activities,  # List[Dict for FFNActivityItem]
        collected_class_embedding_representations  # List[Dict for ClassEmbeddingRepresentationItem]
    )


def generate_attribution(
    model: VisionTransformer,  # Type hint for clarity
    input_tensor: torch.Tensor,
    method: str = "transmm",  # Defaulting to transmm as it's the one detailed
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
    img_size: int = 224,
    data_collection: bool = False,  # Pass this through
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

    if method.lower() == "transmm":
        gini_params = kwargs.get('gini_params')
        weigh_by_class_embedding = kwargs.get('weigh_by_class_embedding',
                                              False)

        (pred_dict, pos_attr_np, neg_attr_np, ffn_list, cer_list) = transmm(
            model_nn=model,  # Pass the VisionTransformer instance
            input_tensor=input_tensor,
            target_class=target_class,
            gini_params=gini_params,
            device=device,
            img_size=img_size,
            weigh_by_class_embedding=weigh_by_class_embedding,
            data_collection=data_collection  # Pass this flag
        )

        # Structure the output dictionary as expected by pipeline.py
        return {
            "predictions":
            pred_dict,  # The dict from model_handler.get_prediction via transmm
            "attribution_positive": pos_attr_np,
            "attribution_negative":
            neg_attr_np,  # This will be None if transmm returns None
            "ffn_activity": ffn_list,  # List of dicts
            "class_embedding_representation": cer_list  # List of dicts
        }
    else:
        raise ValueError(f"Unsupported attribution method: {method}")
