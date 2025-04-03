import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from PIL import Image, ImageStat
from typing import Tuple, Union, Literal, List
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

device = "cuda"

def load_sd_model(model_path: str = "IrohXu/stable-diffusion-mimic-cxr-v0.1") -> StableDiffusionInpaintPipeline:
    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
    return sd_pipe

def patch_similarity(original_patch: Image.Image, perturbed_patch: Image.Image):
    transform = transforms.ToTensor()
    original_tensor = transform(original_patch).numpy().transpose(1, 2, 0)
    perturbed_tensor = transform(perturbed_patch).numpy().transpose(1, 2, 0)
    
    # Calculate SSIM
    similarity = ssim(
        original_tensor, 
        perturbed_tensor, 
        data_range=1.0,
        channel_axis=2
    )
    return similarity

def generate_patch_coordinates(
    image_width: int, 
    image_height: int, 
    patch_size: int
) -> List[Tuple[int, int, int]]:
    patches = []
    patch_id = 0
    
    for y in range(0, image_height - patch_size + 1, patch_size):
        for x in range(0, image_width - patch_size + 1, patch_size):
            patches.append((patch_id, x, y))
            patch_id += 1
            
    return patches

def create_patch_id(
    patient_id: str, 
    original_filename: str, 
    patch_id: int, 
    x: int, 
    y: int, 
    method: str = "sd", 
    strength: float = 0.2
) -> str:
    if method == "sd":
        return f"{patient_id}_{original_filename}_patch{patch_id}_x{x}_y{y}_s{strength}"
    else:
        return f"{patient_id}_{original_filename}_patch{patch_id}_x{x}_y{y}_{method}"

def perturb_patch_sd(
    model_pipe: StableDiffusionInpaintPipeline,
    image_identifier: Tuple[str, str],
    patch_info: Tuple[int, int, int],
    strength: float = 0.2
) -> Tuple[Image.Image, np.ndarray]:
    """
    Perturb a single patch using Stable Diffusion.
    
    Args:
        model_pipe: StableDiffusionInpaintPipeline instance
        image_identifier: Tuple of (patient_id, original_filename)
        patch_info: Tuple of (x, y, patch_size) defining the patch location and size
        strength: Perturbation strength (0.0 to 1.0)
        
    Returns:
        Tuple of (perturbed_image, mask) where perturbed_image is a PIL Image
        and mask is a numpy boolean array marking the perturbed area
    """
    patient_id, original_filename = image_identifier
    x, y, patch_size = patch_info
    
    original_512 = Image.open(f'./chexpert/{patient_id}/study1/{original_filename}.jpg').resize((512, 512), Image.LANCZOS).convert('RGB')
    mask_512 = Image.new("L", (512, 512), 0)
    
    # Scale coordinates from 224x224 to 512x512
    scale_factor = 512 / 224  # Assuming original is 224×224
    x_512 = int(x * scale_factor)
    y_512 = int(y * scale_factor)
    patch_512 = int(patch_size * scale_factor)
    
    patch_mask = Image.new("L", (patch_512, patch_512), 255)
    mask_512.paste(patch_mask, (x_512, y_512))
    np_mask = np.zeros((224, 224), dtype=bool)
    np_mask[y:y+patch_size, x:x+patch_size] = True
    
    device = model_pipe.device
    generator = torch.Generator(device=device).manual_seed(42)
    
    perturbed_512 = model_pipe(
        prompt="",
        image=original_512,
        mask_image=mask_512,
        guidance_scale=0.0,  # Unconditional generation
        num_inference_steps=10,
        strength=strength,
        generator=generator
    ).images[0]
    
    perturbed_224 = perturbed_512.resize((224, 224), Image.LANCZOS)
    original_224 = Image.open(f'./images/{patient_id}_{original_filename}.jpg')
    
    np_original = np.array(original_224)
    np_perturbed = np.array(perturbed_224)

    result = np.copy(np_original)
    result[np_mask] = np_perturbed[np_mask]
    
    result_image = Image.fromarray(result)
    
    return result_image, np_mask

def perturb_patch_mean(
    image_identifier: Tuple[str, str],
    patch_info: Tuple[int, int, int]
) -> Tuple[Image.Image, np.ndarray]:
    """
    Replace a patch with the mean color of the image.
    
    Args:
        image_identifier: Tuple of (patient_id, original_filename) 
        patch_info: Tuple of (x, y, patch_size) defining the patch
        
    Returns:
        Tuple of (perturbed_image, mask)
    """
    patient_id, original_filename = image_identifier
    x, y, patch_size = patch_info

    image = Image.open(f'./images/{patient_id}_{original_filename}.jpg')
    
    mean_channels = ImageStat.Stat(image).mean
    mean_color = tuple(int(channel) for channel in mean_channels)
    patch = Image.new("RGB", (patch_size, patch_size), mean_color)
    
    np_mask = np.zeros((224, 224), dtype=bool)
    np_mask[y:y+patch_size, x:x+patch_size] = True
    
    result = image.copy()
    result.paste(patch, (x, y))
    
    return result, np_mask