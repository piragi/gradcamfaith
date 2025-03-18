import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
from collections import OrderedDict
from translrp.ViT_LRP import vit_base_patch16_224

class TransLRPExplainer:
    def __init__(self, huggingface_model_name="codewithdark/vit-chest-xray", img_size=224):
        """Initialize the TransLRP explainer"""
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load Hugging Face model and processor
        print(f"Loading Hugging Face model: {huggingface_model_name}")
        self.processor = AutoImageProcessor.from_pretrained(huggingface_model_name)
        self.hf_model = AutoModelForImageClassification.from_pretrained(huggingface_model_name)
        self.num_classes = self.hf_model.config.num_labels
        self.label_columns = self.hf_model.config.id2label
        
        # Initialize Chefer's ViT model
        print("Initializing Chefer's ViT model")
        self.model = vit_base_patch16_224(
            pretrained=False,
            img_size=img_size,
            num_classes=self.num_classes
        ).to(self.device)
        
        # Transfer weights and prepare model
        self._transfer_weights()
        print("Model initialized successfully")
    
    def _transfer_weights(self):
        """Transfer weights from Hugging Face model to Chefer's implementation"""
        hf_state_dict = self.hf_model.state_dict()
        new_state_dict = OrderedDict()
        
        # Create mapping between HF and Chefer models
        key_mapping = {
            'vit.embeddings.patch_embeddings.projection.weight': 'patch_embed.proj.weight',
            'vit.embeddings.patch_embeddings.projection.bias': 'patch_embed.proj.bias',
            'vit.embeddings.position_embeddings': 'pos_embed',
            'vit.embeddings.cls_token': 'cls_token',
            'vit.layernorm.weight': 'norm.weight',
            'vit.layernorm.bias': 'norm.bias',
            'classifier.weight': 'head.weight',
            'classifier.bias': 'head.bias',
        }
        
        # Encoder block mapping
        for i in range(12):
            key_mapping.update({
                f'vit.encoder.layer.{i}.layernorm_before.weight': f'blocks.{i}.norm1.weight',
                f'vit.encoder.layer.{i}.layernorm_before.bias': f'blocks.{i}.norm1.bias',
                f'vit.encoder.layer.{i}.layernorm_after.weight': f'blocks.{i}.norm2.weight',
                f'vit.encoder.layer.{i}.layernorm_after.bias': f'blocks.{i}.norm2.bias',
                f'vit.encoder.layer.{i}.attention.output.dense.weight': f'blocks.{i}.attn.proj.weight',
                f'vit.encoder.layer.{i}.attention.output.dense.bias': f'blocks.{i}.attn.proj.bias',
                f'vit.encoder.layer.{i}.intermediate.dense.weight': f'blocks.{i}.mlp.fc1.weight',
                f'vit.encoder.layer.{i}.intermediate.dense.bias': f'blocks.{i}.mlp.fc1.bias',
                f'vit.encoder.layer.{i}.output.dense.weight': f'blocks.{i}.mlp.fc2.weight',
                f'vit.encoder.layer.{i}.output.dense.bias': f'blocks.{i}.mlp.fc2.bias',
            })
        
        # Transfer mapped weights
        for hf_key, chefer_key in key_mapping.items():
            if hf_key in hf_state_dict:
                new_state_dict[chefer_key] = hf_state_dict[hf_key]
        
        # Handle QKV weights
        for i in range(12):
            q_weight = hf_state_dict.get(f'vit.encoder.layer.{i}.attention.attention.query.weight')
            k_weight = hf_state_dict.get(f'vit.encoder.layer.{i}.attention.attention.key.weight')
            v_weight = hf_state_dict.get(f'vit.encoder.layer.{i}.attention.attention.value.weight')
            
            q_bias = hf_state_dict.get(f'vit.encoder.layer.{i}.attention.attention.query.bias')
            k_bias = hf_state_dict.get(f'vit.encoder.layer.{i}.attention.attention.key.bias')
            v_bias = hf_state_dict.get(f'vit.encoder.layer.{i}.attention.attention.value.bias')
            
            if q_weight is not None and k_weight is not None and v_weight is not None:
                new_state_dict[f'blocks.{i}.attn.qkv.weight'] = torch.cat([q_weight, k_weight, v_weight], dim=0)
            
            if q_bias is not None and k_bias is not None and v_bias is not None:
                new_state_dict[f'blocks.{i}.attn.qkv.bias'] = torch.cat([q_bias, k_bias, v_bias], dim=0)
        
        # Load weights
        self.model.load_state_dict(new_state_dict, strict=False)
    
    def preprocess_image(self, image_path):
        """Preprocess image for the model"""
        img = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=img, return_tensors="pt")
        input_tensor = inputs["pixel_values"].to(self.device)
        return img, input_tensor
    
    def classify_image(self, image_path):
        """Classify an image using the model"""
        _, input_tensor = self.preprocess_image(image_path)
        
        # Forward pass
        outputs = self.model(input_tensor.detach())
        
        # Get probabilities and predicted class
        probs = F.softmax(outputs, dim=1)[0]
        pred_class_idx = probs.argmax().item()
        
        # Get label
        pred_class_label = self.label_columns.get(str(pred_class_idx), 
                           self.label_columns.get(pred_class_idx, f"Class {pred_class_idx}"))
        
        return {
            "logits": outputs,
            "probabilities": probs,
            "predicted_class_idx": pred_class_idx,
            "predicted_class_label": pred_class_label,
        }
    
    def explain(self, image_path, target_class=None, method="transformer_attribution"):
        """Generate an explanation for the prediction"""
        # Preprocess image
        img, input_tensor = self.preprocess_image(image_path)
        
        # Get prediction if no target class provided
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(input_tensor.detach())
                target_class = outputs.argmax(dim=1).item()
            label = self.label_columns.get(str(target_class), f"Class {target_class}")
            print(f"Explaining prediction: {label} (Class {target_class})")
        
        # Forward pass with gradients
        outputs = self.model(input_tensor)
        
        # Create one-hot encoding for the target class
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1
        
        # Zero gradients and backward pass
        self.model.zero_grad()
        outputs.backward(gradient=one_hot, retain_graph=True)
        
        # Generate explanation
        explanation = self.model.relprop(
            torch.eye(outputs.shape[1], device=outputs.device)[target_class].unsqueeze(0),
            method=method,
            alpha=1.0
        )
        
        # Process the explanation
        num_patches_side = int(np.sqrt(explanation.shape[1]))
        attribution = explanation.reshape(1, 1, num_patches_side, num_patches_side)
        attribution = F.interpolate(attribution, size=(self.img_size, self.img_size), mode='bilinear')
        attribution = attribution.squeeze().cpu().detach().numpy()
        
        # Normalize for visualization
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        return np.array(img.resize((self.img_size, self.img_size))), attribution
    
    def visualize(self, image, attribution, save_path=None, alpha=0.5):
        """Visualize the attribution map overlaid on the original image"""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Attribution map
        plt.subplot(1, 3, 2)
        plt.imshow(attribution, cmap='jet')
        plt.title('TransLRP Attribution')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.imshow(attribution, cmap='jet', alpha=alpha)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def explain_image(image_path, model_name="codewithdark/vit-chest-xray", method="transformer_attribution", save_dir="./explanations"):
    """Generate and visualize an explanation for an image"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize explainer
    explainer = TransLRPExplainer(model_name)
    
    # Classify image
    results = explainer.classify_image(image_path)
    pred_class = results["predicted_class_idx"]
    pred_label = results["predicted_class_label"]
    
    print(f"Prediction: {pred_label} (Class {pred_class})")
    print(f"Probabilities: {results['probabilities']}")
    
    # Generate explanation
    image, attribution = explainer.explain(image_path, pred_class, method)
    inspect_attribution(attribution)
    
    # Visualize and save
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f"{base_filename}_{method}.png")
    explainer.visualize(image, attribution, save_path=save_path)
    
    print(f"Explanation saved to {save_path}")
    
    return results, attribution

def inspect_attribution(attribution):
    # Print basic statistics
    min_val = np.min(attribution)
    max_val = np.max(attribution)
    mean_val = np.mean(attribution)
    median_val = np.median(attribution)
    print(f"Attribution range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"Mean: {mean_val:.6f}, Median: {median_val:.6f}")

    # Check percentiles to understand distribution
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("Percentiles:")
    for p in percentiles:
        print(f"{p}th: {np.percentile(attribution, p):.6f}")

    # Plot histogram of attribution values
    plt.figure(figsize=(10, 6))
    plt.hist(attribution.flatten(), bins=50)
    plt.title('Distribution of Attribution Values')
    plt.xlabel('Attribution Value')
    plt.ylabel('Frequency')
    plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join("./explanations/", f"attribution_histogram.png"))
    plt.show()

    # Visualize attribution as heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(attribution, cmap='hot')
    plt.colorbar(label='Attribution Value')
    plt.title('Attribution Heatmap')
    plt.savefig(os.path.join("./explanations/", f"attribution_histogram.png"))
    plt.show()

def explain_attribution_diff(attribution, perturbed_attribution, np_mask, base_name=None, save_dir="./explanations"):
    """
    Compare original and perturbed attribution maps, focusing on masked areas.
    
    Parameters:
    -----------
    attribution : numpy.ndarray
        Original attribution map
    perturbed_attribution : numpy.ndarray
        Attribution map after perturbation
    np_mask : numpy.ndarray
        Binary mask indicating areas that were perturbed (True/1 for perturbed areas)
    base_name : str, optional
        Base name for saving files (e.g., the perturbed image name without extension)
    save_dir : str
        Directory to save visualizations
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Use a default base name if none provided
    if base_name is None:
        base_name = "comparison"
    
    # Ensure mask has proper dimensions for masking the attribution maps
    if np_mask.shape != attribution.shape:
        print(f"Warning: Mask shape {np_mask.shape} differs from attribution shape {attribution.shape}")
        np_mask = resize(np_mask, attribution.shape, order=0, preserve_range=True).astype(bool)
    
    # Create masked versions of the attributions
    masked_original = np.where(np_mask, attribution, np.nan)
    masked_perturbed = np.where(np_mask, perturbed_attribution, np.nan)
    
    # Calculate the difference
    attribution_diff = perturbed_attribution - attribution
    masked_diff = np.where(np_mask, attribution_diff, np.nan)
    
    # Calculate statistics for the masked areas
    # For original attribution in masked area
    orig_masked_values = attribution[np_mask]
    orig_min = np.min(orig_masked_values)
    orig_max = np.max(orig_masked_values)
    orig_mean = np.mean(orig_masked_values)
    orig_median = np.median(orig_masked_values)
    
    # For perturbed attribution in masked area
    pert_masked_values = perturbed_attribution[np_mask]
    pert_min = np.min(pert_masked_values)
    pert_max = np.max(pert_masked_values)
    pert_mean = np.mean(pert_masked_values)
    pert_median = np.median(pert_masked_values)
    
    # For difference in masked area
    diff_masked_values = attribution_diff[np_mask]
    diff_min = np.min(diff_masked_values)
    diff_max = np.max(diff_masked_values)
    diff_mean = np.mean(diff_masked_values)
    diff_median = np.median(diff_masked_values)
    
    # Print statistics
    print("\nAttribution Statistics in Perturbed Areas:")
    print(f"{'':20} {'Original':15} {'Perturbed':15} {'Difference':15}")
    print(f"{'-'*65}")
    print(f"{'Min':20} {orig_min:.6f}{' '*9} {pert_min:.6f}{' '*9} {diff_min:.6f}")
    print(f"{'Max':20} {orig_max:.6f}{' '*9} {pert_max:.6f}{' '*9} {diff_max:.6f}")
    print(f"{'Mean':20} {orig_mean:.6f}{' '*9} {pert_mean:.6f}{' '*9} {diff_mean:.6f}")
    print(f"{'Median':20} {orig_median:.6f}{' '*9} {pert_median:.6f}{' '*9} {diff_median:.6f}")
    
    # Create visualizations
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original attribution (masked)
    im0 = axs[0].imshow(masked_original, cmap='viridis')
    axs[0].set_title('Original Attribution\n(Perturbed Areas Only)')
    axs[0].axis('off')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    # Perturbed attribution (masked)
    im1 = axs[1].imshow(masked_perturbed, cmap='viridis')
    axs[1].set_title('Perturbed Attribution\n(Perturbed Areas Only)')
    axs[1].axis('off')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    # Difference
    im2 = axs[2].imshow(masked_diff, cmap='coolwarm', vmin=-max(abs(diff_min), abs(diff_max)), vmax=max(abs(diff_min), abs(diff_max)))
    axs[2].set_title('Difference\n(Perturbed - Original)')
    axs[2].axis('off')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, f"{base_name}_comparison.png")
    plt.savefig(comparison_path)
    plt.close() 
    
    # Distribution of difference values
    plt.figure(figsize=(10, 6))
    plt.hist(diff_masked_values.flatten(), bins=50)
    plt.title('Distribution of Attribution Differences in Perturbed Areas')
    plt.xlabel('Difference Value (Perturbed - Original)')
    plt.ylabel('Frequency')
    plt.axvline(0, color='r', linestyle='dashed', linewidth=1, label='No Change')
    plt.axvline(diff_mean, color='g', linestyle='dashed', linewidth=1, label=f'Mean: {diff_mean:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    histogram_path = os.path.join(save_dir, f"{base_name}_histogram.png")
    plt.savefig(histogram_path)
    plt.close() 
    
    # Calculate percentage of masked area where attribution increased/decreased
    increased = np.sum(diff_masked_values > 0) / diff_masked_values.size * 100
    decreased = np.sum(diff_masked_values < 0) / diff_masked_values.size * 100
    unchanged = np.sum(diff_masked_values == 0) / diff_masked_values.size * 100
    
    print(f"\nPercentage of perturbed areas where attribution:")
    print(f"Increased: {increased:.2f}%")
    print(f"Decreased: {decreased:.2f}%")
    print(f"Unchanged: {unchanged:.2f}%")
    
    # Print the absolute change on average
    abs_mean_change = np.mean(np.abs(diff_masked_values))
    print(f"\nAbsolute mean change in attribution: {abs_mean_change:.6f}")
    
    return {
        "original_stats": {
            "min": float(orig_min),
            "max": float(orig_max),
            "mean": float(orig_mean),
            "median": float(orig_median)
        },
        "perturbed_stats": {
            "min": float(pert_min),
            "max": float(pert_max),
            "mean": float(pert_mean),
            "median": float(pert_median)
        },
        "difference_stats": {
            "min": float(diff_min),
            "max": float(diff_max),
            "mean": float(diff_mean),
            "median": float(diff_median),
            "abs_mean": float(abs_mean_change),
            "increased_pct": float(increased),
            "decreased_pct": float(decreased),
            "unchanged_pct": float(unchanged)
        },
        "visualization_paths": {
            "comparison": comparison_path,
            "histogram": histogram_path
        }
    }
if __name__ == "__main__":
    # Example usage
    image_path = "./images/xray.jpg"
    explain_image(image_path)
    
    image_path = "./images/xray_perturbed.jpg"
    explain_image(image_path)