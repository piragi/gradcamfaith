# model.py
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the model architectures
from translrp.layers_ours import Linear
from translrp.ViT_new import VisionTransformer
from translrp.ViT_new import vit_base_patch16_224 as vit_mm

# Constants
CLASSES = ['COVID-19','Non-COVID','Normal']
IDX2CLS = {i: cls for i, cls in enumerate(CLASSES)}
CLS2IDX = {cls: i for i, cls in enumerate(CLASSES)}


def load_vit_model(
    num_classes: int = 6, model_path: str = "", device: Optional[torch.device] = None
) -> VisionTransformer:
    """
    Load a Vision Transformer model with specified configuration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the base model (same as training)
    model = vit_mm().to(device)
    model.head = Linear(model.head.in_features, num_classes).to(device)

    # Load weights if available
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")

        try:
            # Try to load the checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Falling back to ImageNet pretrained model...")
            model = vit_mm(pretrained=True, num_classes=num_classes)
            model.eval()
            return model

        # Extract model state dict based on checkpoint structure
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New checkpoint format
            state_dict = checkpoint['model_state_dict']
            print(f"✅ Using model_state_dict from checkpoint")
            print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"Validation F1: {checkpoint.get('val_f1', 'unknown')}")
            print(f"Classes: {checkpoint.get('class_names', 'unknown')}")
        else:
            print("⚠️ No 'model_state_dict' found, assuming entire checkpoint is state dict")
            state_dict = checkpoint

        try:
            model.load_state_dict(state_dict)
            print("✅ Model weights loaded successfully")
        except RuntimeError as e:
            print(f"❌ Error loading state dict: {e}")
            print("Falling back to ImageNet pretrained model...")
            model = vit_mm(pretrained=True, num_classes=num_classes)
            model.eval()
            return model
    else:
        print("No trained model found, using ImageNet pretrained")
        model = vit_mm(pretrained=True, num_classes=num_classes)

    model.eval()
    return model


def get_prediction(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_map: Optional[Dict[int, str]] = None,
    device: Optional[torch.device] = None,
    eval: bool = True
) -> Dict[str, Any]:
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
        class_map = IDX2CLS

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


def load_clip_model(
    model_name: str = "openai/clip-vit-base-patch16",
    device: Optional[torch.device] = None,
    cache_dir: Optional[str] = None
) -> Tuple[Any, Any]:  # Returns (model, processor)
    """
    Load CLIP model and processor for zero-shot classification.
    
    Args:
        model_name: Name of the CLIP model to load from Hugging Face
        device: Device to load model on
        cache_dir: Optional cache directory for model weights
        
    Returns:
        Tuple of (model, processor)
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError(
            "transformers library is required for CLIP. "
            "Install with: pip install transformers"
        )
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading CLIP model: {model_name}")
    
    # Load model and processor
    model = CLIPModel.from_pretrained(
        model_name,
        cache_dir=cache_dir
    ).to(device)
    
    processor = CLIPProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    model.eval()
    print(f"✅ CLIP model loaded successfully on {device}")
    
    return model, processor


def get_clip_prediction(
    model: Any,  # CLIPModel
    processor: Any,  # CLIPProcessor
    image: Any,  # PIL Image or tensor
    text_prompts: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Get prediction from CLIP model for an image.
    
    Args:
        model: CLIP model
        processor: CLIP processor
        image: Input image (PIL Image or preprocessed tensor)
        text_prompts: List of text descriptions for each class
        device: Device to run inference on
        
    Returns:
        Dictionary with prediction details similar to get_prediction
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if text_prompts is None:
        # Default prompts for waterbirds
        text_prompts = ["a landbird", "a waterbird"]
    
    # Process inputs
    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]  # Shape: [num_classes]
        probs = F.softmax(logits, dim=0)
    
    pred_class_idx = probs.argmax().item()
    
    return {
        "logits": logits,
        "probabilities": probs,
        "predicted_class_idx": pred_class_idx,
        "predicted_class_label": text_prompts[pred_class_idx],
        "all_probabilities": {
            text_prompts[i]: probs[i].item()
            for i in range(len(text_prompts))
        },
        "text_prompts": text_prompts
    }


def register_model_hooks(model: VisionTransformer, register_hooks: bool = True) -> VisionTransformer:
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get layer information
    layers_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            layers_info.append({"name": name, "type": module.__class__.__name__, "parameters": params})

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "has_attention": hasattr(model, "blocks") and hasattr(model.blocks[0], "attn"),
        "input_size": model.patch_embed.img_size if hasattr(model, "patch_embed") else None,
        "num_classes": model.head.out_features if hasattr(model, "head") else None
    }
