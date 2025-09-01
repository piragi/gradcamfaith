"""
CLIP classifier wrapper for zero-shot classification with HookedViT.

This module provides a wrapper to use CLIP's vision encoder (loaded as HookedViT)
for zero-shot classification, maintaining compatibility with the attribution pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class CLIPClassifier:
    """
    Wrapper for CLIP zero-shot classification using HookedViT vision encoder.
    
    This class handles:
    1. Text encoding to get class embeddings
    2. Image encoding through HookedViT
    3. Computing similarity scores as logits
    """

    def __init__(
        self,
        vision_model: Any,  # HookedViT
        text_model: Any,  # CLIP text encoder
        processor: Any,  # CLIP processor
        class_names: List[str],
        device: torch.device,
        temperature: float = 100.
    ):
        """
        Initialize CLIP classifier.
        
        Args:
            vision_model: HookedViT vision encoder from vit_prisma
            text_model: Original CLIP model for text encoding
            processor: CLIP processor for preprocessing
            class_names: List of class names for classification
            device: Device to run on
            temperature: Temperature scaling for logits
        """
        self.vision_model = vision_model
        self.text_model = text_model
        self.processor = processor
        self.class_names = class_names
        self.device = device
        self.temperature = temperature

        # Pre-compute text embeddings
        self.text_embeddings = self._encode_text_classes()

    def _encode_text_classes(self) -> torch.Tensor:
        """
        Encode text class descriptions to embeddings.
        
        Returns:
            Normalized text embeddings for each class
        """
        # Use class names directly (they already include articles if needed)
        text_prompts = self.class_names

        # Check if processor is a text processor or just image transforms
        if hasattr(self.processor, '__call__') and not hasattr(self.processor, 'tokenizer'):
            # processor is just transforms (OpenCLIP case), need OpenCLIP for text encoding
            import open_clip

            # Use the text model passed in (should be OpenCLIP model)
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            text = tokenizer(text_prompts).to(self.device)
            with torch.no_grad():
                text_features = self.text_model.encode_text(text)
        else:
            # processor is a full CLIPProcessor
            text_inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                # Get text features from CLIP text encoder
                text_features = self.text_model.get_text_features(**text_inputs)

            # Normalize for cosine similarity
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, images: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
        """
        Get image features from HookedViT.
        
        Args:
            images: Preprocessed image tensors
            requires_grad: Whether to compute gradients (for attribution)
            
        Returns:
            Image embeddings from vision encoder
        """
        # HookedViT outputs the final representation
        # We need the CLS token output
        if requires_grad:
            # Keep gradients for attribution
            output = self.vision_model(images)
        else:
            with torch.no_grad():
                output = self.vision_model(images)

        # For HookedViT, the output is typically the CLS token representation
        # after the final layer norm
        if hasattr(output, 'shape') and len(output.shape) == 3:
            # If output is (batch, seq_len, dim), take CLS token
            image_features = output[:, 0, :]
        else:
            # If output is already (batch, dim)
            image_features = output

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def forward(self,
                images: torch.Tensor,
                return_features: bool = False,
                requires_grad: bool = True) -> Dict[str, Any]:
        """
        Perform zero-shot classification.
        
        Args:
            images: Input images (already preprocessed)
            return_features: Whether to return intermediate features
            requires_grad: Whether to compute gradients (for attribution)
            
        Returns:
            Dictionary with logits, probabilities, and predictions
        """
        # Get image features
        image_features = self.get_image_features(images, requires_grad=requires_grad)

        # Compute similarity scores (these are our "logits")
        logits = (image_features @ self.text_embeddings.T) * self.temperature

        # Get probabilities and predictions
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

        result = {
            "logits": logits,
            "probabilities": probs,
            "predicted_class_idx": predictions[0].item() if images.shape[0] == 1 else predictions,
            "predicted_class_label": self.class_names[predictions[0].item()] if images.shape[0] == 1 else None
        }

        if return_features:
            result["image_features"] = image_features
            result["text_features"] = self.text_embeddings

        return result


def create_clip_classifier_for_waterbirds(
    vision_model: Any,
    processor: Any,
    device: torch.device,
    custom_prompts: Optional[List[str]] = None
) -> CLIPClassifier:
    """
    Create a CLIP classifier specifically for Waterbirds dataset.
    
    Args:
        vision_model: HookedViT vision model
        processor: CLIP processor
        device: Device to run on
        custom_prompts: Optional custom text prompts
        
    Returns:
        Configured CLIPClassifier
    """
    # For OpenCLIP models, load the full model for text encoding
    import open_clip

    # Load the same OpenCLIP model for text encoding (matching vision model)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="datacomp_xl_s13b_b90k", device=device
    )

    # Default prompts for waterbirds
    if custom_prompts is None:
        class_names = ["landbird", "waterbird"]
    else:
        class_names = custom_prompts

    classifier = CLIPClassifier(
        vision_model=vision_model, text_model=clip_model, processor=processor, class_names=class_names, device=device
    )

    return classifier


class CLIPModelWrapper(torch.nn.Module):
    """
    Wrapper that makes a CLIP classifier behave like a standard model for inference.
    
    This is needed because the attribution binning code expects to call model(input)
    directly, but CLIP needs the full classifier with text embeddings.
    """

    def __init__(self, clip_classifier: CLIPClassifier):
        """
        Initialize the wrapper.
        
        Args:
            clip_classifier: CLIPClassifier instance
        """
        super().__init__()
        self.clip_classifier = clip_classifier
        self.training = False  # Always in eval mode for attribution

    def eval(self):
        """Set to evaluation mode (no-op, always in eval)."""
        self.training = False
        return self

    def train(self, mode: bool = True):
        """Prevent training mode."""
        if mode:
            raise ValueError("CLIPModelWrapper should not be used in training mode")
        self.training = False
        return self

    def to(self, device):
        # move underlying models if they exist
        if hasattr(self.clip_classifier, "vision_model"):
            self.clip_classifier.vision_model.to(device)
        if hasattr(self.clip_classifier, "text_model") and self.clip_classifier.text_model is not None:
            self.clip_classifier.text_model.to(device)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward method required by nn.Module.
        
        Args:
            images: Input images tensor
            
        Returns:
            Logits tensor
        """
        # Use the CLIP classifier without gradients for inference
        result = self.clip_classifier.forward(images, requires_grad=False)
        return result["logits"]

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns logits.
        
        Args:
            images: Input images tensor
            
        Returns:
            Logits tensor
        """
        return self.forward(images)

    @property
    def cfg(self):
        """Access to model configuration through vision model."""
        if hasattr(self.clip_classifier.vision_model, 'cfg'):
            return self.clip_classifier.vision_model.cfg
        return None


def create_clip_classifier_for_oxford_pets(
    vision_model: Any,
    processor: Any,
    device: torch.device,
    custom_prompts: Optional[List[str]] = None
) -> CLIPClassifier:
    """
    Create a CLIP classifier specifically for Oxford-IIIT Pet dataset.
    
    Args:
        vision_model: HookedViT vision model
        processor: CLIP processor
        device: Device to run on
        custom_prompts: Optional custom text prompts
        
    Returns:
        Configured CLIPClassifier
    """
    # For OpenCLIP models, load the full model for text encoding
    import open_clip

    # Load the same OpenCLIP model for text encoding (matching vision model)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="datacomp_xl_s13b_b90k", device=device
    )

    # Default prompts for oxford pets
    if custom_prompts is None:
        class_names = ["cat", "dog"]
    else:
        class_names = custom_prompts

    classifier = CLIPClassifier(
        vision_model=vision_model, text_model=clip_model, processor=processor, class_names=class_names, device=device
    )

    return classifier
