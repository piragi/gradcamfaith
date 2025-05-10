# model.py
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the model architectures
from translrp.layers_ours import Linear
from translrp.ViT_new import VisionTransformer
from translrp.ViT_new import vit_base_patch16_224 as vit_mm

# Constants
CLS2IDX = {0: 'COVID-19', 1: 'Non-COVID', 2: 'Normal'}


def load_vit_model(num_classes: int = 3,
                   model_path: str = './model/model_best.pth.tar',
                   device: Optional[torch.device] = None) -> VisionTransformer:
    """
    Load a Vision Transformer model with specified configuration.
    
    Args:
        model_type: Type of model architecture ('translrp' or 'base')
        num_classes: Number of output classes
        model_path: Path to model weights
        device: Device to load the model on (defaults to CUDA if available)
        
    Returns:
        Loaded and configured model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vit_mm().to(device)

    # Replace the classification head for the specified number of classes
    model.head = Linear(model.head.in_features, num_classes).to(device)

    # Load pretrained weights if available
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model weights not found at {model_path}")

    model.eval()
    return model


def get_prediction(model: nn.Module,
                   input_tensor: torch.Tensor,
                   class_map: Optional[Dict[int, str]] = None,
                   device: Optional[torch.device] = None,
                   eval: bool = True) -> Dict[str, Any]:
    """
    Get prediction from model for an input tensor.
    
    Args:
        model: Model to use for prediction
        input_tensor: Preprocessed input tensor
        class_map: Mapping from class indices to class names
        device: Device to run inference on
        
    Returns:
        Dictionary with prediction details
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if class_map is None:
        class_map = CLS2IDX

    # Ensure model is in eval mode
    if eval: model.eval()

    # Add batch dimension if needed
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    if eval:
        with torch.no_grad():
            outputs = model(input_tensor)
    else:
        outputs = model(input_tensor, register_hook=True)

    probs = F.softmax(outputs, dim=1)[0]
    pred_class_idx = probs.argmax().item()

    # Get human-readable label
    pred_class_label = class_map.get(pred_class_idx, f"Class {pred_class_idx}")

    return {
        "logits": outputs,
        "probabilities": probs,
        "predicted_class_idx": pred_class_idx,
        "predicted_class_label": pred_class_label,
        "all_probabilities": {
            class_map.get(i, f"Class {i}"): probs[i].item()
            for i in range(len(probs))
        }
    }


def register_model_hooks(model: VisionTransformer,
                         register_hooks: bool = True) -> VisionTransformer:
    """
    Prepare a model for attribution by registering necessary hooks.
    
    Args:
        model: Model to prepare
        register_hooks: Whether to register forward/backward hooks
        
    Returns:
        Prepared model
    """
    # Make sure model is in eval mode
    model.eval()

    # Register hooks for attribution if requested
    if register_hooks and hasattr(model, "blocks"):
        for block in model.blocks:
            # Register hooks on attention and MLP modules
            if hasattr(block, "attn"):
                # This assumes the attn module has a register_hooks method
                if hasattr(block.attn, "register_hooks"):
                    block.attn.register_hooks()
            if hasattr(block, "mlp"):
                # This assumes the mlp module has a register_hooks method
                if hasattr(block.mlp, "register_hooks"):
                    block.mlp.register_hooks()

    return model


def get_available_models() -> List[str]:
    """
    Get list of available model architectures.
    
    Returns:
        List of model type identifiers
    """
    return ["translrp", "base"]


def get_model_info(model: VisionTransformer) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with model information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)

    # Get layer information
    layers_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            layers_info.append({
                "name": name,
                "type": module.__class__.__name__,
                "parameters": params
            })

    return {
        "total_parameters":
        total_params,
        "trainable_parameters":
        trainable_params,
        "has_attention":
        hasattr(model, "blocks") and hasattr(model.blocks[0], "attn"),
        "input_size":
        model.patch_embed.img_size if hasattr(model, "patch_embed") else None,
        "num_classes":
        model.head.out_features if hasattr(model, "head") else None
    }
