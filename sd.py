from pathlib import Path
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
            prompt="Normal chest X-ray with clear lung fields. No cardiomegaly. No pleural effusion. Normal cardiomediastinal silhouette. No consolidation or edema. Costophrenic angles are clear.",
            image=image,
            mask_image=mask,
            guidance_scale=50.0,  # Unconditional generation
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

def create_patch_attribution_mask(attribution: np.ndarray, percentile_threshold: int = 80, patch_size = 16) -> Image.Image:
    height, width = attribution.shape
    patches_y, patches_x = height // patch_size, width // patch_size
    patch_mask = np.zeros((patches_y, patches_x), dtype=np.float32)

    for y in range(patches_y):
        for x in range(patches_x):
            y_start = y * patch_size
            x_start = x * patch_size
            patch = attribution[y_start:y_start+patch_size, x_start:x_start+patch_size]
            patch_mask[y, x] = np.mean(patch)
    
    norm_patch_mask = (patch_mask - np.min(patch_mask)) / (np.max(patch_mask) - np.min(patch_mask))
    threshold = np.percentile(norm_patch_mask, percentile_threshold)
    binary_mask = (norm_patch_mask >= threshold).astype(np.uint8)

    # Upscale back to pixel level
    pixel_mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(patches_y):
        for x in range(patches_x):
            y_start = y * patch_size
            x_start = x * patch_size
            pixel_mask[y_start:y_start+patch_size, x_start:x_start+patch_size] = binary_mask[y,x]

    mask = pixel_mask * 255
    return mask

def perturb_non_attribution(model_pipe: StableDiffusionInpaintPipeline, image_path: str, attribution: np.ndarray, percentile_threshold: int = 80, strength: float = 0.2, patch_size: int = 16):
    original_512 = Image.open(image_path).convert('RGB')    

    # Create the attribution mask as numpy array
    mask_224 = create_patch_attribution_mask(attribution, percentile_threshold, patch_size)    

    # Create the inverse mask (areas to perturb)
    inv_mask_224 = 255 - mask_224
    
    # Convert to PIL image
    pil_mask_224 = Image.fromarray(inv_mask_224).convert('L')

    vit_mask = (mask_224 > 0)
    
    # IMPORTANT: Resize the mask to match the original image dimensions
    pil_mask_512 = pil_mask_224.resize((512,512), Image.NEAREST)
    
    device = model_pipe.device
    generator = torch.Generator(device=device).manual_seed(420)
    perturbed_512 = model_pipe(
        prompt="Bilateral pulmonary edema with patchy infiltrates in lower lobes. Perihilar haziness. Interstitial opacities.",
        image=original_512,
        mask_image=pil_mask_512,
        guidance_scale=0.0,
        num_inference_steps=5,
        strength=strength,
        generator=generator
    ).images[0]
    
    # Downscale both original and perturbed images to 224x224 for comparison
    original_224 = Image.open(f'./results/vit_inputs/{Path(image_path).stem}.jpg')
    perturbed_224 = perturbed_512.resize((224, 224), Image.LANCZOS)
    
    # Final composition: preserve high attribution areas from original
    np_original = np.array(original_224)
    np_perturbed = np.array(perturbed_224)
    
    result = np.copy(np_original)
    result[~vit_mask] = np_perturbed[~vit_mask]  # Only replace low attribution areas
    
    result_image = Image.fromarray(result)
    
    # Return the combined image and areas that were perturbed
    return result_image, ~vit_mask