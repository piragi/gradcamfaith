import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from PIL import Image
from typing import Tuple
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim


device = "cuda"

def load_model(model_path: str = "IrohXu/stable-diffusion-mimic-cxr-v0.1") -> StableDiffusionInpaintPipeline:
    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
    return sd_pipe

def mask_image(image: Image.Image, patch_size: int, patch_position: Tuple[int, int]) -> Image.Image:
    width, height = image.size
    x, y = patch_position
    original_patch = image.crop((x, y, x + patch_size, y + patch_size))
    mask = Image.new("L", (width, height), 0)
    patch_mask = Image.new("L", (patch_size, patch_size), 255)
    mask.paste(patch_mask, (x, y))
    return mask, original_patch

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

def perturb_patch(model_pipe: StableDiffusionInpaintPipeline, image_path: str, patch_position: Tuple[int, int], patch_size = 16, strength = 0.2):
    original_image = Image.open(image_path).convert('RGB')
    image = original_image.copy()
    mask, original_patch = mask_image(image, patch_size, patch_position)
    generator = torch.Generator(device=device).manual_seed(42)
    result = model_pipe(
            prompt="",
            image=image,
            mask_image=mask,
            guidance_scale=0.0,  # Unconditional generation
            num_inference_steps=5,
            strength=strength,
            generator=generator
        ).images[0]

    x, y = patch_position
    perturbed_patch = result.crop((x, y, x + patch_size, y + patch_size)) 
    similarity = patch_similarity(original_patch, perturbed_patch)
    image.paste(perturbed_patch, patch_position)
    image.save("./images/xray_perturbed.jpg")
    return image, similarity

def create_attribution_mask(attribution: np.ndarray, percentile_threshold: int = 80) -> Image.Image:
    norm_attr = (attribution - np.min(attribution)) / (np.max(attribution) - np.min(attribution))
    threshold = np.percentile(norm_attr, percentile_threshold)
    binary_mask = (norm_attr >= threshold).astype(np.uint8)
    mask = binary_mask * 255
    return mask

def perturb_non_attribution(model_pipe: StableDiffusionInpaintPipeline, image_path: str, attribution: np.ndarray, percentile_threshold: int = 80, strength: float = 0.2):
    original_image = Image.open(image_path).convert('RGB')
    
    # Create the attribution mask as numpy array
    mask_np = create_attribution_mask(attribution, percentile_threshold)
    
    # Create the inverse mask (areas to perturb)
    inv_mask_np = 255 - mask_np
    
    # Convert to PIL image
    pil_mask = Image.fromarray(inv_mask_np).convert('L')
    
    # IMPORTANT: Resize the mask to match the original image dimensions
    pil_mask = pil_mask.resize(original_image.size, Image.NEAREST)
    
    device = model_pipe.device
    generator = torch.Generator(device=device).manual_seed(420)
    result = model_pipe(
        prompt="Severe edema in the left and right lower lobes, severity. Severe right and left pleural effusion is larger.",
        image=original_image,
        mask_image=pil_mask,
        guidance_scale=0.0,
        num_inference_steps=10,
        strength=strength,
        generator=generator
    ).images[0]
    
    # Create the final result
    result_image = original_image.copy()
    
    # Create mask that matches the image dimensions
    np_mask = np.array(pil_mask) > 0
    np_perturbed = np.array(result)
    np_result = np.array(result_image)
    
    # Now the dimensions should match
    np_result[np_mask] = np_perturbed[np_mask]
    result_image = Image.fromarray(np_result)
    result_image.save("./images/xray_perturbed.jpg")
    
    return result_image