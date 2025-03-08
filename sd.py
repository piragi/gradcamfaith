import os
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

def perturb_patch(model_pipe, image_path: str, patch_position: Tuple[int, int], patch_size = 16, strength = 0.2):
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