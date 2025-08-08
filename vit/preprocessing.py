# preprocessing.py
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def get_default_processor(
        img_size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None) -> transforms.Compose:
    """
    Get default image processor/transform pipeline.
    
    Args:
        img_size: Target image size
        mean: Normalization mean (default: [0.56, 0.56, 0.56])
        std: Normalization std (default: [0.21, 0.21, 0.21])
        
    Returns:
        Composed transform pipeline
    """
    if mean is None:
        mean=[0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    pil_transform = transforms.Compose([
        transforms.Resize(256),  # Resize to img_size * 8/7
        transforms.CenterCrop(img_size)
    ])

    normalize = transforms.Normalize(mean=mean, std=std)
    preprocess_transform = transforms.Compose(
        [transforms.ToTensor(), normalize])

    return transforms.Compose([pil_transform, preprocess_transform])

def get_processor_for_precached_224_images(
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None) -> transforms.Compose:
    """
    Processor for images that are ALREADY 224x224 (due to prior Resize(256) + CenterCrop(224)).
    Only applies ToTensor and Normalize.
    """
    if mean is None:
        mean=[0.485, 0.456, 0.406] # ImageNet Mean
    if std is None:
        std = [0.229, 0.224, 0.225] # ImageNet Std

    normalize = transforms.Normalize(mean=mean, std=std)
    return transforms.Compose([transforms.ToTensor(), normalize])


def load_image(image_path: str, convert_to_rgb: bool = True) -> Image.Image:
    """
    Load an image from path and handle grayscale conversion if needed.
    
    Args:
        image_path: Path to the image
        convert_to_rgb: Whether to convert grayscale to RGB
        
    Returns:
        PIL Image object
    """
    image = np.asarray(Image.open(image_path))

    # Convert grayscale to RGB if needed
    if convert_to_rgb and len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(image, mode="RGB")
    elif convert_to_rgb and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(image, mode="RGB")

    return Image.fromarray(image)


def preprocess_image(
        image_path: str,
        processor: Optional[transforms.Compose] = None,
        img_size: int = 224,
        convert_to_rgb: bool = True) -> Tuple[Image.Image, torch.Tensor]:
    """
    Preprocess an image for model input.
    
    Args:
        image_path: Path to the image
        processor: Transform pipeline (if None, uses default)
        img_size: Target image size
        convert_to_rgb: Whether to convert grayscale to RGB
        
    Returns:
        Tuple of (original PIL image, preprocessed tensor)
    """
    # Load the image
    image = load_image(image_path, convert_to_rgb)

    # Get default processor if none provided
    if processor is None:
        # Check if image is already 224x224 (preprocessed)
        if image.size == (224, 224):
            processor = get_processor_for_precached_224_images()
        else:
            processor = get_default_processor(img_size)

    # Apply preprocessing
    input_tensor = processor(image)

    return image, input_tensor


def denormalize_tensor(tensor: torch.Tensor,
                       mean: List[float] = [0.56, 0.56, 0.56],
                       std: List[float] = [0.21, 0.21, 0.21]) -> torch.Tensor:
    """
    Denormalize a tensor to original image range.
    
    Args:
        tensor: Input tensor [C, H, W]
        mean: Normalization mean used
        std: Normalization std used
        
    Returns:
        Denormalized tensor
    """
    # Make a deep copy of the tensor to avoid modifying the original
    result = tensor.clone().detach()

    # Apply denormalization
    for t, m, s in zip(result, mean, std):
        t.mul_(s).add_(m)

    # Clamp values to [0, 1] for display
    result = torch.clamp(result, 0, 1)

    return result


def tensor_to_numpy(tensor: torch.Tensor,
                    denormalize: bool = True,
                    mean: Optional[List[float]] = None,
                    std: Optional[List[float]] = None) -> np.ndarray:
    """
    Convert a tensor to a numpy array suitable for visualization.
    
    Args:
        tensor: Input tensor [C, H, W] or [B, C, H, W]
        denormalize: Whether to denormalize the tensor
        mean: Normalization mean (default: [0.56, 0.56, 0.56])
        std: Normalization std (default: [0.21, 0.21, 0.21])
        
    Returns:
        Numpy array [H, W, C] with values in [0, 1]
    """
    if mean is None:
        mean = [0.56, 0.56, 0.56]
    if std is None:
        std = [0.21, 0.21, 0.21]

    # Make sure input is on CPU and detached from computation graph
    if tensor.dim() == 4:  # Batch dimension present
        tensor = tensor[0]  # Take first image

    tensor = tensor.cpu().detach()

    # Denormalize if requested
    if denormalize:
        tensor = denormalize_tensor(tensor, mean, std)

    # Convert to numpy and transpose from [C, H, W] to [H, W, C]
    np_image = tensor.numpy().transpose(1, 2, 0)

    return np_image
