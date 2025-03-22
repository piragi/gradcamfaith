import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from PIL import Image
from typing import Tuple, Union, Literal
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

def perturb_non_attribution(model_pipe: StableDiffusionInpaintPipeline, image_identifier: Union[str,str], attribution: np.ndarray, percentile_threshold: int = 80, strength: float = 0.2, patch_size: int = 16):
    patient_id, original_filename = image_identifier
    original_512 = Image.open(f'./chexpert/{patient_id}/study1/{original_filename}.jpg').resize((512,512), Image.LANCZOS).convert('RGB')    

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
        num_inference_steps=10,
        strength=strength,
        generator=generator
    ).images[0]
    
    # Downscale both original and perturbed images to 224x224 for comparison
    original_224 = Image.open(f'./images/{patient_id}_{original_filename}.jpg')
    perturbed_224 = perturbed_512.resize((224, 224), Image.LANCZOS)
    
    # Final composition: preserve high attribution areas from original
    np_original = np.array(original_224)
    np_perturbed = np.array(perturbed_224)
    
    result = np.copy(np_original)
    result[~vit_mask] = np_perturbed[~vit_mask]  # Only replace low attribution areas
    
    result_image = Image.fromarray(result)
    
    # Return the combined image and areas that were perturbed
    return result_image, ~vit_mask

def select_tiles(attribution: np.ndarray, 
                num_tiles: int = 30, 
                strategy: Literal["quantile", "lowest", "highest", "random"] = "quantile",
                patch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select tiles based on attribution values using different strategies.
    
    Args:
        attribution: Attribution map
        num_tiles: Number of tiles to select
        strategy: Tile selection strategy:
            - "quantile": Select tiles at regular quantile intervals (distribution-based)
            - "lowest": Select tiles with lowest attribution values
            - "highest": Select tiles with highest attribution values
            - "random": Select tiles randomly
        patch_size: Size of each tile/patch
    
    Returns:
        Tuple of (binary_mask, selected_patch_indices)
    """
    height, width = attribution.shape
    patches_y, patches_x = height // patch_size, width // patch_size
    total_patches = patches_y * patches_x
    
    # Ensure num_tiles doesn't exceed total patches
    num_tiles = min(num_tiles, total_patches)
    
    # Calculate mean attribution for each patch
    patch_values = np.zeros((patches_y, patches_x), dtype=np.float32)
    for y in range(patches_y):
        for x in range(patches_x):
            y_start = y * patch_size
            x_start = x * patch_size
            patch = attribution[y_start:y_start+patch_size, x_start:x_start+patch_size]
            patch_values[y, x] = np.mean(patch)
    
    # Flatten patch values and indices
    flat_values = patch_values.flatten()
    flat_indices = np.arange(total_patches)
    
    # Select indices based on strategy
    if strategy == "lowest":
        # Lowest attribution values
        selected_indices = flat_indices[np.argsort(flat_values)[:num_tiles]]
    
    elif strategy == "highest":
        # Highest attribution values
        selected_indices = flat_indices[np.argsort(flat_values)[-num_tiles:]]
    
    elif strategy == "quantile":
        # Select tiles at regular quantile intervals (distribution-based)
        sorted_indices = flat_indices[np.argsort(flat_values)]
        
        # Calculate indices to sample at regular intervals
        if num_tiles > 1:
            interval = (len(sorted_indices) - 1) / (num_tiles - 1)
            indices_to_sample = [int(round(interval * i)) for i in range(num_tiles)]
            # Ensure we don't have duplicates and exceed array bounds
            indices_to_sample = np.unique(np.clip(indices_to_sample, 0, len(sorted_indices) - 1))
            
            # If we lost some tiles due to duplicates, add more from regularly spaced samples
            if len(indices_to_sample) < num_tiles:
                remaining = num_tiles - len(indices_to_sample)
                additional_indices = np.linspace(0, len(sorted_indices) - 1, remaining + 2)[1:-1]
                additional_indices = [int(idx) for idx in additional_indices
                                     if int(idx) not in indices_to_sample]
                indices_to_sample = np.append(indices_to_sample, additional_indices[:remaining])
            
            selected_indices = sorted_indices[indices_to_sample]
        else:
            # If only one tile, pick the median
            median_idx = len(sorted_indices) // 2
            selected_indices = np.array([sorted_indices[median_idx]])
    
    elif strategy == "random":
        # Simple random selection
        selected_indices = np.random.choice(flat_indices, num_tiles, replace=False)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'quantile', 'lowest', 'highest', or 'random'.")
    
    # Create a binary mask for selected patches
    patch_mask = np.zeros((patches_y, patches_x), dtype=bool)
    for idx in selected_indices:
        y, x = idx // patches_x, idx % patches_x
        patch_mask[y, x] = True
    
    # Convert to pixel-level mask
    pixel_mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(patches_y):
        for x in range(patches_x):
            if patch_mask[y, x]:
                y_start = y * patch_size
                x_start = x * patch_size
                pixel_mask[y_start:y_start+patch_size, x_start:x_start+patch_size] = 1
    
    # Create mask for inpainting (255 for selected areas)
    mask_255 = pixel_mask * 255
    
    # Also return the selected patches
    selected_patches = np.argwhere(patch_mask)
    
    return mask_255, selected_patches

def visualize_selected_tiles(attribution: np.ndarray, 
                           mask: np.ndarray,
                           strategy: str,
                           save_path: str = None):
    """
    Visualize selected tiles overlaid on the attribution map.
    
    Args:
        attribution: Attribution map
        mask: Binary mask indicating selected tiles
        strategy: Name of the selection strategy used
        save_path: Path to save the visualization (if None, display instead)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # Normalize attribution for visualization
    norm_attr = (attribution - attribution.min()) / (attribution.max() - attribution.min())
    
    plt.figure(figsize=(10, 8))
    
    # Plot attribution map
    plt.imshow(norm_attr, cmap='viridis')
    plt.title(f"Selected Tiles: {strategy.capitalize()} Strategy")
    
    # Add rectangles for selected tiles
    patch_size = 16
    height, width = attribution.shape
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Check if this patch is selected
            if mask[y:y+patch_size, x:x+patch_size].any():
                rect = Rectangle((x, y), patch_size, patch_size, 
                              linewidth=1, edgecolor='red', facecolor='none')
                plt.gca().add_patch(rect)
    
    plt.colorbar(label='Attribution Value')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def perturb_image(model_pipe, 
                image_identifier: Tuple[str, str], 
                attribution: np.ndarray, 
                num_tiles: int = 20,
                strategy: str = "quantile",
                strength: float = 0.2, 
                patch_size: int = 16,
                output_dir: str = "./results/comparative"):
    """
    Perturb an image by selecting tiles using the specified strategy.
    
    Args:
        model_pipe: Stable Diffusion inpainting model
        image_identifier: Tuple of (patient_id, original_filename)
        attribution: Attribution map
        num_tiles: Number of tiles to perturb
        strategy: Tile selection strategy
        strength: Perturbation strength
        patch_size: Size of each tile/patch
        output_dir: Directory to save outputs
        
    Returns:
        Tuple of (perturbed_image, binary_mask, output_paths)
    """
    import os
    from pathlib import Path
    
    patient_id, original_filename = image_identifier
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create unique experiment ID
    exp_id = f"{patient_id}_{original_filename}_tiles{num_tiles}_{strategy}_s{strength}"
    perturbed_image_path = output_dir / f"{exp_id}.jpg"
    mask_path = output_dir / f"{exp_id}_mask.npy"
    viz_path = output_dir / f"{exp_id}_viz.png"
    
    # Skip if already processed
    if perturbed_image_path.exists() and mask_path.exists():
        print(f"Skipping {exp_id} - already exists")
        return None, None, (perturbed_image_path, mask_path, viz_path)
    
    # Load and resize original image
    original_512 = Image.open(f'./chexpert/{patient_id}/study1/{original_filename}.jpg').resize((512,512), Image.LANCZOS).convert('RGB')
    
    # Select tiles based on strategy
    mask_224, selected_patches = select_tiles(
        attribution, 
        num_tiles=num_tiles, 
        strategy=strategy,
        patch_size=patch_size
    )
    
    # Create binary mask for visualization
    binary_mask = mask_224 > 0
    
    # Visualize selected tiles
    visualize_selected_tiles(
        attribution, 
        binary_mask,
        strategy,
        save_path=str(viz_path)
    )
    
    # Convert to PIL image
    pil_mask_224 = Image.fromarray(mask_224).convert('L')
    
    # Resize mask to match the original image dimensions
    pil_mask_512 = pil_mask_224.resize((512,512), Image.NEAREST)
    
    # Run the inpainting model
    device = model_pipe.device
    generator = torch.Generator(device=device).manual_seed(420)
    perturbed_512 = model_pipe(
        prompt="Bilateral pulmonary edema with patchy infiltrates in lower lobes. Perihilar haziness. Interstitial opacities.",
        image=original_512,
        mask_image=pil_mask_512,
        guidance_scale=0.0,
        num_inference_steps=10,
        strength=strength,
        generator=generator
    ).images[0]
    
    # Downscale both original and perturbed images to 224x224 for comparison
    original_224 = Image.open(f'./images/{patient_id}_{original_filename}.jpg')
    perturbed_224 = perturbed_512.resize((224, 224), Image.LANCZOS)
    
    # Final composition: use perturbed image only in the masked areas
    np_original = np.array(original_224)
    np_perturbed = np.array(perturbed_224)
    
    result = np.copy(np_original)
    result[binary_mask] = np_perturbed[binary_mask]
    
    result_image = Image.fromarray(result)
    
    # Save results
    result_image.save(perturbed_image_path)
    np.save(mask_path, binary_mask)
    
    print(f"Created perturbed image with {strategy} strategy: {perturbed_image_path}")
    
    # Return the perturbed image, binary mask, and output paths
    return result_image, binary_mask, (perturbed_image_path, mask_path, viz_path)

def run_comparative_experiment(model_pipe, results_df, num_tiles=20, strength=0.2, 
                              strategies=["quantile", "lowest", "highest", "random"]):
    """
    Run a comparative perturbation experiment using the specified strategies.
    
    Args:
        model_pipe: Stable Diffusion inpainting model
        results_df: DataFrame with original classification results
        num_tiles: Number of tiles to perturb
        strength: Perturbation strength
        strategies: List of strategies to test
        
    Returns:
        Dictionary mapping strategy names to lists of perturbed image paths
    """
    from pathlib import Path
    
    output_dir = Path("./results/comparative_experiment")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    perturbed_paths = {strategy: [] for strategy in strategies}
    
    for _, row in results_df.iterrows():
        try:
            image_path = Path(row["image_path"])
            patient_id = image_path.stem.split('_')[0]  # Assuming format "patientID_view1_frontal.jpg"
            
            # Extract the original filename without the extension
            original_filename = "_".join(image_path.stem.split('_')[1:])
            
            # Load the attribution map
            attribution = np.load(row["attribution_path"])
            
            for strategy in strategies:
                # Create strategy-specific directory
                strategy_dir = output_dir / strategy
                strategy_dir.mkdir(exist_ok=True, parents=True)
                
                # Apply perturbation with the current strategy
                try:
                    _, _, (img_path, _, _) = perturb_image(
                        model_pipe=model_pipe,
                        image_identifier=(patient_id, original_filename),
                        attribution=attribution,
                        num_tiles=num_tiles,
                        strategy=strategy,
                        strength=strength,
                        output_dir=strategy_dir
                    )
                    
                    if img_path:
                        perturbed_paths[strategy].append(img_path)
                    
                except Exception as e:
                    print(f"Error processing {image_path.name} with {strategy} strategy: {e}")
        
        except Exception as e:
            print(f"Error processing row: {e}")
    
    return perturbed_paths

def perturb_single_patch(model_pipe, image_identifier, patch_info, strength=0.2):
    """
    Perturbs a single patch in an image using the Stable Diffusion model.
    
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
        prompt="Bilateral pulmonary edema with patchy infiltrates in lower lobes. Perihilar haziness. Interstitial opacities.",
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