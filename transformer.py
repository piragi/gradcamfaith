import gc
import os
from collections import OrderedDict
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from translrp.layers_ours import Linear
from translrp.ViT_LRP import vit_base_patch16_224 as vit_lrp
from translrp.ViT_new import vit_base_patch16_224 as vit_mm

CLS2IDX = {0: 'COVID-19', 1: 'Non-COVID', 2: 'Normal'}


class ViT:

    def __init__(self, img_size=224, method: str = "translrp"):
        """Initialize the TransLRP explainer"""
        self.img_size = img_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = self.get_processor()
        print(f"Using device: {self.device}")

        # Initialize Chefer's ViT model
        print("Initializing Chefer's ViT model")
        if method == "translrp":
            self.model = vit_lrp().to(self.device)
            self.model.head = Linear(self.model.head.in_features,
                                     3).to(self.device)
            self.model.load_state_dict(
                torch.load('./model/model_best.pth.tar')['state_dict'])
            self.model.eval()
        else:
            self.model = vit_mm().to(self.device)
            self.model.head = Linear(self.model.head.in_features,
                                     3).to(self.device)
            self.model.load_state_dict(
                torch.load('./model/model_best.pth.tar')['state_dict'])
            self.model.eval()

    def get_processor(self):
        pil_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224)])
        normalize = transforms.Normalize(mean=[0.56, 0.56, 0.56],
                                         std=[0.21, 0.21, 0.21])
        preprocess_transform = transforms.Compose(
            [transforms.ToTensor(), normalize])

        return transforms.Compose([pil_transform, preprocess_transform])

    def preprocess_image(self, image_path):
        """Preprocess image for the model"""
        image = np.asarray(Image.open(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image, mode="RGB")
        return image

    def classify_image(self, image_path):
        """Classify an image using the model"""
        img = self.preprocess_image(image_path)
        input_tensor = self.processor(img)

        # Forward pass
        outputs = self.model(input_tensor.unsqueeze(0).to(self.device))

        # Get probabilities and predicted class
        probs = F.softmax(outputs, dim=1)[0]
        pred_class_idx = probs.argmax().item()

        # Get label
        pred_class_label = CLS2IDX[int(pred_class_idx)]

        return img, {
            "logits": outputs,
            "probabilities": probs,
            "predicted_class_idx": pred_class_idx,
            "predicted_class_label": pred_class_label,
        }

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


def translrp(vit: ViT,
             image_path,
             target_class=None,
             method="transformer_attribution"):
    """Generate an explanation for the prediction"""
    # Preprocess image
    img, input_tensor = vit.preprocess_image(image_path)

    # Get prediction if no target class provided
    if target_class is None:
        with torch.no_grad():
            outputs = vit.model(input_tensor.detach())
            target_class = outputs.argmax(dim=1).item()
        label = vit.label_columns.get(str(target_class),
                                      f"Class {target_class}")
        print(f"Explaining prediction: {label} (Class {target_class})")

    # Forward pass with gradients
    outputs = vit.model(input_tensor)

    # Create one-hot encoding for the target class
    one_hot = torch.zeros_like(outputs)
    one_hot[0, target_class] = 1

    # Zero gradients and backward pass
    vit.model.zero_grad()
    outputs.backward(gradient=one_hot, retain_graph=True)

    # Generate explanation
    explanation = vit.model.relprop(torch.eye(
        outputs.shape[1], device=outputs.device)[target_class].unsqueeze(0),
                                    method=method,
                                    alpha=1.0)

    # Process the explanation
    num_patches_side = int(np.sqrt(explanation.shape[1]))
    attribution = explanation.reshape(1, 1, num_patches_side, num_patches_side)
    attribution = F.interpolate(attribution,
                                size=(vit.img_size, vit.img_size),
                                mode='bilinear')
    attribution = attribution.squeeze().cpu().detach().numpy()

    # Normalize for visualization
    attribution = (attribution - attribution.min()) / (
        attribution.max() - attribution.min() + 1e-8)

    return np.array(img), attribution


def avg_heads(cam, grad):
    """Rule 5 from paper: Average attention heads weighted by gradients."""
    # Move to CPU to save GPU memory
    cam = cam.cpu()
    grad = grad.cpu()

    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def avg_heads_min(cam, grad):
    """Rule 5 from paper: Average attention heads weighted by gradients."""
    # Move to CPU to save GPU memory
    cam = cam.cpu()
    grad = grad.cpu()

    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(max=0).abs().mean(dim=0)
    return cam


def apply_self_attention_rules(R_ss, cam_ss):
    """Rule 6 from paper: Apply self-attention propagation rule."""
    # Ensure both are on the same device (CPU to save memory)
    R_ss = R_ss.cpu()
    cam_ss = cam_ss.cpu()

    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def adaptive_power_normalization(attention_head):
    """
    Apply power normalization that adapts based on the attention distribution.
    Works with PyTorch tensors.
    """
    # Measure dispersion using entropy or Gini coefficient
    norm_attn = attention_head / (torch.sum(attention_head) + 1e-10)
    entropy = -torch.sum(norm_attn * torch.log2(norm_attn + 1e-10))

    # Calculate adaptive power (high entropy → lower power)
    max_entropy = torch.log2(torch.tensor(float(
        attention_head.numel())))  # Theoretical maximum entropy
    dispersion_ratio = entropy / max_entropy

    # Power ranges from 0.3 (very dispersed) to 1. (very concentrated)
    power = 1. - (0.2 * dispersion_ratio)

    # Apply power transformation
    transformed = torch.pow(attention_head, power)

    # Preserve the sum
    return transformed * (torch.sum(attention_head) /
                          (torch.sum(transformed) + 1e-10))


def improved_adaptive_power_normalization(attention_head,
                                          threshold=0.7,
                                          steepness=10.0,
                                          min_power=0.4):
    """
    Apply power normalization with a smooth transition based on dispersion.
    Only significantly transforms highly dispersed attention distributions,
    while leaving concentrated ones nearly unchanged.
    
    Args:
        attention_head: The attention tensor
        threshold: Dispersion ratio threshold where transition occurs
        steepness: Controls transition sharpness (higher = sharper)
        min_power: Minimum power value for highly dispersed attention
    """
    # Measure dispersion using entropy
    norm_attn = attention_head / (torch.sum(attention_head) + 1e-10)
    entropy = -torch.sum(norm_attn * torch.log2(norm_attn + 1e-10))

    # Calculate dispersion ratio
    max_entropy = torch.log2(torch.tensor(float(attention_head.numel())))
    dispersion_ratio = entropy / max_entropy

    # Sigmoid function for smooth transition
    # When dispersion_ratio = threshold, sigmoid = 0.5
    # For low dispersion, sigmoid ≈ 0 → power ≈ 1.0 (unchanged)
    # For high dispersion, sigmoid ≈ 1 → power ≈ min_power (full transformation)
    sigmoid = 1.0 / (1.0 + torch.exp(-steepness *
                                     (dispersion_ratio - threshold)))

    # Power ranges from 1.0 (no transformation) to min_power (full transformation)
    power = 1.0 - ((1.0 - min_power) * sigmoid)

    # Apply power transformation
    transformed = torch.pow(attention_head, power)

    # Preserve the sum
    return transformed * (torch.sum(attention_head) /
                          (torch.sum(transformed) + 1e-10))


def gini_based_normalization(
    attention_head,
    gini_threshold=0.65,  # Based on distribution analysis
    steepness=8.0,  # Good balance between sharp and smooth
    max_power=0.5):  # Moderate transformation strength
    """
    Apply power normalization based on Gini coefficient.
    Only transforms dispersed attention (low Gini), leaving concentrated attention (high Gini) unchanged.
    """
    # Calculate Gini coefficient
    sorted_attn = torch.sort(attention_head.flatten())[0]
    n = sorted_attn.numel()
    index = torch.arange(1, n + 1, device=sorted_attn.device)
    gini = torch.sum((2 * index - n - 1) *
                     sorted_attn) / (n * torch.sum(sorted_attn) + 1e-10)

    # Calculate transformation factor using sigmoid function
    transformation_factor = 1.0 - (1.0 /
                                   (1.0 + torch.exp(-steepness *
                                                    (gini - gini_threshold))))

    # Calculate adaptive power
    power = 1.0 - (max_power * transformation_factor)

    # Apply power transformation
    transformed = torch.pow(attention_head, power)

    # Preserve the sum
    return transformed * (torch.sum(attention_head) /
                          (torch.sum(transformed) + 1e-10))


def per_head_gini_normalization(attention_heads,
                                gini_threshold=0.65,
                                steepness=5.0,
                                max_power=0.6):
    """
    Apply different transformations to each attention head based on its dispersion characteristics.
    """
    # Process each head individually
    transformed_heads = []

    for head_idx in range(attention_heads.shape[0]):
        head = attention_heads[head_idx]

        # Calculate head-specific Gini
        sorted_attn = torch.sort(head.flatten())[0]
        n = sorted_attn.numel()
        index = torch.arange(1, n + 1, device=sorted_attn.device)
        gini = torch.sum((2 * index - n - 1) *
                         sorted_attn) / (n * torch.sum(sorted_attn) + 1e-10)

        # Adjust threshold based on head's Gini value
        # Only transform very dispersed heads (relative to this specific head's typical pattern)
        if gini < gini_threshold:
            # Apply transformation only to dispersed heads
            gini_factor = 1.0 - (1.0 /
                                 (1.0 + torch.exp(-steepness *
                                                  (gini - gini_threshold))))
            power = 1.0 - (max_power * gini_factor)

            # Apply power transformation
            transformed = torch.pow(head, power)
            # Preserve the sum
            transformed = transformed * (torch.sum(head) /
                                         (torch.sum(transformed) + 1e-10))
        else:
            # Leave concentrated heads untouched
            transformed = head

        transformed_heads.append(transformed)

    return torch.stack(transformed_heads)


def transmm(vit: ViT,
            image_path,
            target_class=None,
            pretransform=False,
            gini_params: Tuple[float, float, float] = (0.65, 8.0, 0.5)):
    """
    Memory-efficient implementation of TransMM.
    
    Args:
        vit: ViT model instance
        image_path: Path to the image
        target_class: Target class index (if None, use predicted class)
        
    Returns:
        Tuple of (original_image_array, attribution_map)
    """
    # Preprocess image
    img = vit.preprocess_image(image_path)
    input_tensor = vit.processor(img)
    device = vit.device

    # Forward pass without hooks first to determine class
    if target_class is None:
        with torch.no_grad():
            outputs = vit.model(input_tensor.unsqueeze(0).to(device).detach())
            target_class = outputs.argmax(dim=1).item()
            del outputs
            torch.cuda.empty_cache()

    # Clear any existing gradients
    vit.model.zero_grad()

    # Forward pass with hook registration
    output = vit.model(input_tensor.unsqueeze(0).to(device),
                       register_hook=True)

    # Create one-hot vector on CPU to save memory
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, target_class] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)

    # Scale output by one_hot
    loss = torch.sum(one_hot * output)

    vit.model.zero_grad()
    loss.backward(retain_graph=True)

    # Get attention maps and gradients
    num_tokens = vit.model.blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).to('cpu')  # Keep on CPU
    R_neg = torch.eye(num_tokens, num_tokens).to('cpu')  # Keep on CPU

    # Process each block
    for i, blk in enumerate(vit.model.blocks):
        # Get data and immediately move to CPU
        grad = blk.attn.get_attn_gradients().detach()
        cam = blk.attn.get_attention_map().detach()

        # Preprocess attn maps for dispersion
        if pretransform:
            gini_threshold, steepness, max_power = gini_params
            cam = gini_based_normalization(cam,
                                           gini_threshold=gini_threshold,
                                           steepness=steepness,
                                           max_power=max_power)

        # Process on CPU to save GPU memory
        cam = avg_heads(cam, grad)
        cam_neg = avg_heads_min(cam, grad)
        R = R + apply_self_attention_rules(R, cam)
        R_neg = R_neg + apply_self_attention_rules(R_neg, cam_neg)

        # Explicitly delete intermediates
        del grad, cam

    # Extract patch tokens relevance (excluding CLS token)
    transformer_attribution = R[0, 1:].clone()
    transformer_attribution_neg = R_neg[0, 1:].clone()
    del R, R_neg

    def process_transformer_attribution(transformer_attribution):
        # Reshape to patch grid
        side_size = int(np.sqrt(transformer_attribution.size(0)))
        transformer_attribution = transformer_attribution.reshape(
            1, 1, side_size, side_size)

        # Move back to GPU only for interpolation
        transformer_attribution = transformer_attribution.to(device)

        # Upscale to image size
        transformer_attribution = F.interpolate(transformer_attribution,
                                                size=(vit.img_size,
                                                      vit.img_size),
                                                mode='bilinear')

        # Convert to numpy and normalize
        return transformer_attribution.reshape(
            vit.img_size, vit.img_size).cpu().detach().numpy()

    attribution = process_transformer_attribution(transformer_attribution)
    attribution_neg = process_transformer_attribution(
        transformer_attribution_neg)

    # Clean up
    del transformer_attribution, transformer_attribution_neg, input_tensor, output, one_hot, loss
    torch.cuda.empty_cache()
    gc.collect()

    # Normalize
    normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)

    attribution = normalize(attribution)
    attribution_neg = normalize(attribution_neg)

    return np.array(img), (attribution, attribution_neg)


def explain_image(image_path,
                  model_name="",
                  method="transformer_attribution",
                  save_dir="./explanations",
                  visualize=False):
    """Generate and visualize an explanation for an image"""
    os.makedirs(save_dir, exist_ok=True)

    # Initialize explainer
    explainer = ViT(model_name)

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
    if visualize:
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
    plt.axvline(mean_val,
                color='r',
                linestyle='dashed',
                linewidth=1,
                label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val,
                color='g',
                linestyle='dashed',
                linewidth=1,
                label=f'Median: {median_val:.4f}')
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


def explain_attribution_diff(attribution,
                             perturbed_attribution,
                             np_mask,
                             base_name=None,
                             save_dir="./explanations",
                             visualize=False):
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

    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.transform import resize

    os.makedirs(save_dir, exist_ok=True)

    # Use a default base name if none provided
    if base_name is None:
        base_name = "comparison"

    # Ensure mask has proper dimensions for masking the attribution maps
    if np_mask.shape != attribution.shape:
        print(
            f"Warning: Mask shape {np_mask.shape} differs from attribution shape {attribution.shape}"
        )
        np_mask = resize(np_mask,
                         attribution.shape,
                         order=0,
                         preserve_range=True).astype(bool)

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

    if visualize:
        # Print statistics
        print("\nAttribution Statistics in Perturbed Areas:")
        print(f"{'':20} {'Original':15} {'Perturbed':15} {'Difference':15}")
        print(f"{'-'*65}")
        print(
            f"{'Min':20} {orig_min:.6f}{' '*9} {pert_min:.6f}{' '*9} {diff_min:.6f}"
        )
        print(
            f"{'Max':20} {orig_max:.6f}{' '*9} {pert_max:.6f}{' '*9} {diff_max:.6f}"
        )
        print(
            f"{'Mean':20} {orig_mean:.6f}{' '*9} {pert_mean:.6f}{' '*9} {diff_mean:.6f}"
        )
        print(
            f"{'Median':20} {orig_median:.6f}{' '*9} {pert_median:.6f}{' '*9} {diff_median:.6f}"
        )
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
        im2 = axs[2].imshow(masked_diff,
                            cmap='coolwarm',
                            vmin=-max(abs(diff_min), abs(diff_max)),
                            vmax=max(abs(diff_min), abs(diff_max)))
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
        plt.axvline(0,
                    color='r',
                    linestyle='dashed',
                    linewidth=1,
                    label='No Change')
        plt.axvline(diff_mean,
                    color='g',
                    linestyle='dashed',
                    linewidth=1,
                    label=f'Mean: {diff_mean:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        histogram_path = os.path.join(save_dir, f"{base_name}_histogram.png")
        plt.savefig(histogram_path)
        plt.close()

    # Calculate percentage of masked area where attribution increased/decreased
    increased = np.sum(diff_masked_values > 0) / diff_masked_values.size * 100
    decreased = np.sum(diff_masked_values < 0) / diff_masked_values.size * 100
    unchanged = np.sum(diff_masked_values == 0) / diff_masked_values.size * 100

    # print(f"\nPercentage of perturbed areas where attribution:")
    # print(f"Increased: {increased:.2f}%")
    # print(f"Decreased: {decreased:.2f}%")
    # print(f"Unchanged: {unchanged:.2f}%")

    # Print the absolute change on average
    abs_mean_change = np.mean(np.abs(diff_masked_values))
    # print(f"\nAbsolute mean change in attribution: {abs_mean_change:.6f}")

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
    }
