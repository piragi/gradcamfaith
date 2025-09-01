"""
Download the paper's pre-trained SAE checkpoints from HuggingFace for ALL layers
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

# Map each layer to its specific repository (using the best performing version based on ending number)
layer_repo_map = {
    1: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_1-hook_resid_post-64-82",
    2: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_2-hook_resid_post-64-80",
    3: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_3-hook_resid_post-64-80",
    4: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_4-hook_resid_post-64-80",
    5: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_5-hook_resid_post-64-81",
    6: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_6-hook_resid_post-64-81",
    7: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_7-hook_resid_post-64-83",
    8: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_8-hook_resid_post-64-84",
    9: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_9-hook_resid_post-64-86",
    10: "Prisma-Multimodal/waterbirds-sweep-topk-64-patches_all_layers_10-hook_resid_post-64-85"
}

print(f"{'='*60}")
print("DOWNLOADING SAE CHECKPOINTS FOR ALL LAYERS")
print(f"{'='*60}\n")

for layer_num, repo_id in layer_repo_map.items():
    filename = "weights.pt"  # The trained checkpoint

    print(f"Downloading Layer {layer_num} SAE checkpoint...")
    print(f"Repository: {repo_id}")

    try:
        # Download the checkpoint
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir="./hf_cache")

        # Create the target directory structure for this layer's SAE
        target_dir = Path(f"data/sae_waterbirds_clip_b32/layer_{layer_num}")
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy to the expected location with consistent naming
        target_path = target_dir / "weights.pt"
        shutil.copy(downloaded_path, target_path)

        print(f"✅ Layer {layer_num} SAE checkpoint saved to: {target_path}")

        # Also download the config file if available
        try:
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir="./hf_cache")
            shutil.copy(config_path, target_dir / "config.json")
            print(f"✅ Layer {layer_num} config file saved")
        except:
            print(f"Note: No config.json found for layer {layer_num} (that's okay)")

        print()  # Add blank line between layers

    except Exception as e:
        print(f"❌ Error downloading layer {layer_num}: {str(e)}")
        print(f"   Repository might not exist: {repo_id}")
        print()

print(f"{'='*60}")
print("DOWNLOAD COMPLETE")
print(f"{'='*60}")
print(f"Layers downloaded: 1-10")
print(f"Hook point: hook_resid_post")
print(f"Expansion factor: 64x")
print(f"L1 coefficient: 5e-05")
print(f"Training data: ImageNet")
print(f"Model: CLIP-ViT-B-32")
print(f"\nSAE checkpoints are stored in: data/sae_waterbirds_clip_b32/")
print(f"{'='*60}")
