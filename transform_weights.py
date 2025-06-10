import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from translrp.ViT_new import vit_base_patch16_224

# --- SSL4GIE's VisionTransformer_from_Any (Simplified for this use case) ---
# We only need the parts relevant to head=True, dense=False, det=False
class SSL4GIE_ViT(TimmVisionTransformer):
    def __init__(
        self,
        num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        # Unused for this specific loading but kept for signature compatibility if strict loading was used
        head=True, 
        frozen=False,
        dense=False,
        det=False,
        fixed_size=224,
        out_token='cls',
        ImageNet_weights=False, # Will be False when loading their fine-tuned checkpoint
    ):
        super().__init__(
            patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads, num_classes=0 # Base timm ViT's head is Identity if num_classes=0
        )
        # Original SSL4GIE model sets self.head = nn.Identity() and then adds self.lin_head
        # Replicating that structure for faithful intermediate loading:
        self.head = nn.Identity() # This is the timm.ViT's head
        self.lin_head = nn.Linear(embed_dim, num_classes) # This is SSL4GIE's specific head
        
        self.head_bool = head # For their forward logic
        self.out_token = out_token # For their forward logic

        # Unused for this specific task but part of their class structure
        self.frozen = frozen
        self.dense = dense
        self.det = det
        if ImageNet_weights: # This path won't be taken when loading their fine-tuned ckpt
            # This is a placeholder, as the actual download URL is for their ImageNet_class init
            # which we are bypassing by loading their finetuned weights directly.
            print("WARNING: ImageNet_weights=True in SSL4GIE_ViT init, but we load custom weights.")


    def forward(self, x): # Simplified forward for classification
        x = self.forward_features(x) # From TimmVisionTransformer
        if self.out_token == "cls":
            x = x[:, 0]
        elif self.out_token == "spatial": # Not typical for their classification ViT-B
            x = x[:, 1:].mean(1)
        # If self.head_bool is True (which it is for classification)
        x = self.lin_head(x) # Use their specific linear head
        return x


def transform_and_save_weights(
    input_checkpoint_path,
    output_weights_path,
    num_classes,
    verbose=True):
    """
    Loads weights from an SSL4GIE checkpoint, transforms them for a
    translrp.ViT_new architecture, and saves the transformed state_dict.

    Args:
        input_checkpoint_path (str): Path to the SSL4GIE .pth checkpoint file.
        output_weights_path (str): Path to save the transformed model's state_dict.
        num_classes (int): Number of output classes for the final model.
        verbose (bool): Whether to print detailed loading messages.
    """
    if verbose: print(f"--- Starting weight transformation ---")
    if verbose: print(f"Input checkpoint: {input_checkpoint_path}")
    if verbose: print(f"Output for transformed weights: {output_weights_path}")
    if verbose: print(f"Number of classes: {num_classes}")

    # 1. Instantiate SSL4GIE's model structure
    ssl_temp_model = SSL4GIE_ViT(
        num_classes=num_classes,
        embed_dim=768, depth=12, num_heads=12, # Standard ViT-B params
        # Other params are defaults or not affecting weight loading structure for this case
    )
    ssl_temp_model.eval()

    # 2. Load the downloaded fine-tuned weights into ssl_temp_model
    actual_model_state_dict_from_checkpoint = None
    try:
        try:
            checkpoint = torch.load(input_checkpoint_path, map_location='cpu', weights_only=True)
            if verbose: print("Loaded checkpoint with weights_only=True")
        except Exception:
            checkpoint = torch.load(input_checkpoint_path, map_location='cpu', weights_only=False)
            if verbose: print("Loaded checkpoint with weights_only=False (fallback)")

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            actual_model_state_dict_from_checkpoint = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict):
            actual_model_state_dict_from_checkpoint = checkpoint
            if verbose: print("Checkpoint is a dict, but no 'model_state_dict' key. Assuming it's the state_dict.")
        else:
            raise ValueError("Loaded checkpoint is not a dictionary or 'model_state_dict' key not found.")

        if actual_model_state_dict_from_checkpoint:
            new_state_dict_for_ssl_model = {}
            has_module_prefix = any(key.startswith('module.') for key in actual_model_state_dict_from_checkpoint.keys())
            if has_module_prefix and verbose:
                print("Detected 'module.' prefix in checkpoint. Removing it.")
            for k, v in actual_model_state_dict_from_checkpoint.items():
                name = k[7:] if has_module_prefix and k.startswith('module.') else k
                new_state_dict_for_ssl_model[name] = v
            
            ssl_temp_model.load_state_dict(new_state_dict_for_ssl_model, strict=True) # Be strict here
            if verbose: print("Successfully loaded weights into intermediate SSL4GIE model structure.")
        else:
            raise ValueError("Could not extract a valid model_state_dict from the checkpoint.")

    except Exception as e:
        print(f"ERROR loading weights into intermediate SSL4GIE model: {e}")
        return

    # 3. Get the state_dict from the populated ssl_temp_model
    ssl_model_clean_state_dict = ssl_temp_model.state_dict()

    # 4. Instantiate your target model structure
    if verbose: print(f"Instantiating your target ViT-B model with {num_classes} classes...")
    target_model = vit_base_patch16_224(pretrained=False, num_classes=num_classes)
    target_model.eval()

    # 5. Prepare state_dict for your target model (mapping SSL4GIE's 'lin_head' to your 'head')
    final_state_dict_for_target_model = {}
    for key, value in ssl_model_clean_state_dict.items():
        if key == "lin_head.weight": # SSL4GIE_ViT uses lin_head
            final_state_dict_for_target_model["head.weight"] = value # Your ViT_new uses head
        elif key == "lin_head.bias":
            final_state_dict_for_target_model["head.bias"] = value
        else:
            final_state_dict_for_target_model[key] = value
    
    # 6. Load into your target model
    try:
        target_model.load_state_dict(final_state_dict_for_target_model, strict=True) # Be strict
        if verbose: print("Successfully loaded transformed weights into your target model architecture!")
    except Exception as e:
        print(f"ERROR loading transformed weights into your target model: {e}")
        # You might want to try strict=False here for debugging, but True is preferred for final check
        # missing_keys, unexpected_keys = target_model.load_state_dict(final_state_dict_for_target_model, strict=False)
        # print(f"  Missing keys: {missing_keys}")
        # print(f"  Unexpected keys: {unexpected_keys}")
        return

    # 7. Save the state_dict of your transformed model
    try:
        torch.save(target_model.state_dict(), output_weights_path)
        if verbose: print(f"Successfully saved transformed model state_dict to: {output_weights_path}")
    except Exception as e:
        print(f"ERROR saving transformed model state_dict: {e}")
        return

    if verbose: print(f"--- Weight transformation complete ---")


if __name__ == '__main__':
    # --- Configuration ---
    NUM_CLASSES = 6 # For Hyperkvasir anatomical landmarks
    
    # Path to the downloaded SSL4GIE fine-tuned weights
    INPUT_SSL4GIE_CHECKPOINT_PATH = './model/vit_b-ImageNet_class_init-frozen_False-dataset_Hyperkvasir_anatomical.pth'
    
    # Path where you want to save the transformed weights compatible with your translrp.ViT_new
    OUTPUT_TRANSFORMED_WEIGHTS_PATH = './model/vit_b_hyperkvasir_anatomical_for_translrp.pth'

    transform_and_save_weights(
        input_checkpoint_path=INPUT_SSL4GIE_CHECKPOINT_PATH,
        output_weights_path=OUTPUT_TRANSFORMED_WEIGHTS_PATH,
        num_classes=NUM_CLASSES,
        verbose=True
    )

    # --- Optional: Test loading the saved transformed weights ---
    print("\n--- Testing loading the newly saved transformed weights ---")
    try:
        test_model = vit_base_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
        test_model.load_state_dict(torch.load(OUTPUT_TRANSFORMED_WEIGHTS_PATH, map_location='cpu'))
        test_model.eval()
        print(f"Successfully loaded saved transformed weights into a new instance of your model.")
        
        # Dummy input test
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = test_model(dummy_input)
        print(f"Dummy input passed through test_model. Output shape: {output.shape}")
        assert output.shape == (1, NUM_CLASSES)
        print("Output shape is correct.")

    except FileNotFoundError:
        print(f"Test load failed: Transformed weights file not found at {OUTPUT_TRANSFORMED_WEIGHTS_PATH}")
    except Exception as e:
        print(f"Test load failed: {e}")
