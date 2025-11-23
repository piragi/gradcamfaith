"""
Lightweight SAE Activation Extraction
Extracts SAE feature activations from a dataset without computing attributions.
Much faster than full pipeline - only forward pass + SAE encoding.
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

import config
from dataset_config import get_dataset_config
from pipeline import load_model_for_dataset, load_steering_resources
from unified_dataloader import create_dataloader


def extract_sae_activations(
    dataset_name: str,
    layers: List[int],
    split: str = 'val',
    output_dir: Path = None,
    subset_size: int = None,
    batch_size: int = 32,
    use_clip: bool = True
):
    """
    Extract SAE activations for a dataset split without computing attributions.

    Args:
        dataset_name: Name of the dataset ('imagenet', 'covidquex', etc.)
        layers: List of layer indices to extract SAE activations from
        split: Dataset split to process ('train', 'val', 'test')
        output_dir: Where to save the activations (defaults to ./sae_activations/{dataset}_{split})
        subset_size: If specified, only process this many images
        batch_size: Batch size for processing
        use_clip: Whether the model uses CLIP
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(f"./sae_activations/{dataset_name}_{split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset config
    dataset_config = get_dataset_config(dataset_name)
    print(f"Dataset: {dataset_name} ({dataset_config.num_classes} classes)")

    # Load model
    print(f"Loading model for {dataset_name}...")
    temp_config = config.PipelineConfig()
    temp_config.file.set_dataset(dataset_name)
    if use_clip:
        temp_config.classify.use_clip = True
        temp_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"

    model, clip_classifier = load_model_for_dataset(dataset_config, device, temp_config)
    model.eval()

    # Load SAEs
    print(f"Loading SAE resources for layers: {layers}")
    steering_resources = load_steering_resources(layers, dataset_name=dataset_name)

    # Create dataloader
    print(f"Creating dataloader for {split} split...")
    prepared_path = Path(f"./data/{dataset_name}_unified")
    dataset_loader = create_dataloader(dataset_name=dataset_name, data_path=prepared_path)

    # Get image list
    image_data = list(dataset_loader.get_numeric_samples(split))
    total_samples = len(image_data)

    if subset_size is not None and subset_size < total_samples:
        import random
        random.seed(42)
        image_data = random.sample(image_data, subset_size)
        print(f"Processing {len(image_data)} randomly selected images (subset of {total_samples})")
    else:
        print(f"Processing all {total_samples} images")

    # Initialize storage for each layer
    # We'll save incrementally to avoid running out of RAM
    checkpoint_interval = 10000  # Save every 5000 images
    layer_data = {layer_idx: {'sparse_indices': [], 'sparse_activations': []} for layer_idx in layers}

    # Setup output directory
    debug_dir = output_dir / "debug_data"
    debug_dir.mkdir(exist_ok=True, parents=True)

    # Process images
    print(f"\nExtracting SAE activations (saving every {checkpoint_interval} images)...")

    from unified_dataloader import get_single_image_loader

    for img_idx, (image_path, label) in enumerate(tqdm(image_data, desc="Processing images")):
        try:
            # Load and preprocess image
            input_tensor = get_single_image_loader(image_path, dataset_config, use_clip=use_clip)
            input_tensor = input_tensor.to(device)

            # Setup hooks to capture residuals
            residuals = {}

            def save_resid_hook(tensor, hook):
                layer_idx = int(hook.name.split('.')[1])
                if layer_idx in layers:
                    residuals[layer_idx] = tensor.detach().cpu()
                return tensor

            # Register forward hooks
            fwd_hooks = []
            for layer_idx in layers:
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                fwd_hooks.append((hook_name, save_resid_hook))

            # Forward pass with hooks
            with torch.no_grad():
                with model.hooks(fwd_hooks=fwd_hooks, reset_hooks_end=True):
                    _ = model(input_tensor)

            # Encode residuals with SAE
            for layer_idx in layers:
                if layer_idx not in residuals:
                    continue

                resid = residuals[layer_idx].to(device)  # [1, n_patches+1, d_model]
                sae = steering_resources[layer_idx]['sae']

                # Encode with SAE
                with torch.no_grad():
                    _, codes = sae.encode(resid)  # [1, n_patches+1, n_features]

                # Remove batch dimension and CLS token (matching full pipeline behavior)
                codes = codes.cpu()
                if codes.dim() == 3:
                    codes = codes[0]  # [n_patches+1, n_features]
                codes = codes[1:]  # Remove CLS: [n_patches, n_features]

                # Debug: verify patch count on first image
                if img_idx == 0:
                    print(f"\n  Debug info for layer {layer_idx}:")
                    print(f"    Codes shape after CLS removal: {codes.shape}")
                    print(f"    Expected patches: 49 (for B/32) or 196 (for B/16)")

                # Convert to sparse format (only store active features > threshold)
                threshold = 0.1
                sparse_indices_per_patch = []
                sparse_activations_per_patch = []

                for patch_idx in range(codes.shape[0]):
                    patch_codes = codes[patch_idx]  # [n_features]
                    active_mask = patch_codes > threshold
                    active_indices = torch.where(active_mask)[0].numpy()
                    active_values = patch_codes[active_mask].numpy()

                    sparse_indices_per_patch.append(active_indices)
                    sparse_activations_per_patch.append(active_values)

                # Verify we have correct number of patches
                assert len(sparse_indices_per_patch) == codes.shape[0], \
                    f"Patch count mismatch: {len(sparse_indices_per_patch)} != {codes.shape[0]}"

                # Store sparse data
                layer_data[layer_idx]['sparse_indices'].append(sparse_indices_per_patch)
                layer_data[layer_idx]['sparse_activations'].append(sparse_activations_per_patch)

            # Save checkpoint every N images
            if (img_idx + 1) % checkpoint_interval == 0 or (img_idx + 1) == len(image_data):
                print(f"\n  Saving checkpoint at image {img_idx + 1}...")
                for layer_idx in layers:
                    checkpoint_file = debug_dir / f"layer_{layer_idx}_checkpoint_{img_idx + 1}.npz"

                    sparse_indices = np.array(layer_data[layer_idx]['sparse_indices'], dtype=object)
                    sparse_activations = np.array(layer_data[layer_idx]['sparse_activations'], dtype=object)

                    np.savez_compressed(
                        checkpoint_file, sparse_indices=sparse_indices, sparse_activations=sparse_activations
                    )

                # Clear memory after checkpoint
                layer_data = {layer_idx: {'sparse_indices': [], 'sparse_activations': []} for layer_idx in layers}
                torch.cuda.empty_cache()

            # Periodic GPU cleanup
            elif img_idx % 100 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue

    # Merge all checkpoints into final files
    print("\nMerging checkpoint files...")
    for layer_idx in layers:
        checkpoint_files = sorted(debug_dir.glob(f"layer_{layer_idx}_checkpoint_*.npz"))

        all_indices = []
        all_activations = []

        for checkpoint_file in checkpoint_files:
            data = np.load(checkpoint_file, allow_pickle=True)
            all_indices.extend(data['sparse_indices'])
            all_activations.extend(data['sparse_activations'])

        # Save final merged file
        output_file = debug_dir / f"layer_{layer_idx}_activations.npz"
        np.savez_compressed(
            output_file,
            sparse_indices=np.array(all_indices, dtype=object),
            sparse_activations=np.array(all_activations, dtype=object)
        )

        print(f"  Layer {layer_idx}: Merged {len(all_indices)} images to {output_file}")

        # Clean up checkpoint files
        for checkpoint_file in checkpoint_files:
            checkpoint_file.unlink()
        print(f"  Layer {layer_idx}: Cleaned up {len(checkpoint_files)} checkpoint files")

    # Save metadata including image path mapping
    # This is crucial for correctly mapping debug_idx to actual images later
    image_path_mapping = {idx: str(image_path) for idx, (image_path, label) in enumerate(image_data)}

    metadata = {
        'dataset_name': dataset_name,
        'split': split,
        'layers': layers,
        'n_images': len(image_data),
        'use_clip': use_clip,
        'image_paths': image_path_mapping  # Map debug_idx -> image_path
    }

    import json
    with open(output_dir / "extraction_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtraction complete! Saved to {output_dir}")

    # Cleanup
    del model
    if clip_classifier is not None:
        del clip_classifier
    for layer_idx, resources in steering_resources.items():
        if 'sae' in resources:
            del resources['sae']
    torch.cuda.empty_cache()

    return output_dir


if __name__ == "__main__":
    # Configuration
    dataset = "imagenet"
    layers = [6, 9, 10]
    split = "val"  # or 'train'

    # Extract activations
    output_dir = extract_sae_activations(
        dataset_name=dataset,
        layers=layers,
        split=split,
        subset_size=None,  # Process all images
        batch_size=4096,
        use_clip=True  # ImageNet uses CLIP
    )

    print(f"\nActivations saved to: {output_dir}")
    print("\nYou can now use these for prototype visualization in case studies!")
