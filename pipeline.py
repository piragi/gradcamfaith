import explanation as expl
import transformer as tf
from typing import Tuple
import sd

def classify_image(image, model=None, processor=None, label_columns=None):
    # Use provided models if available, otherwise load them
    if model is None or processor is None or label_columns is None:
        processor, model, label_columns = tf.load_classifier()
    
    results = tf.predict(image, model, processor, label_columns)
    print(results)
    return results

def perturb_image(image: str, patch_position: Tuple[int, int], strength: float, sd_pipe=None):
    # Use provided model if available, otherwise load it
    if sd_pipe is None:
        sd_pipe = sd.load_model()
    
    perturbed_image, similarity = sd.perturb_patch(sd_pipe, image, patch_position, strength=strength)
    print(similarity)
    return perturbed_image

def perturb_classify(image: str):
    results, attribution = expl.explain_image(image)
    print(results)

    sd_pipe = sd.load_model()
    result_image, np_mask = sd.perturb_non_attribution(sd_pipe, image, attribution, strength=0.8, percentile_threshold=15)
    results, perturbed_attribution = expl.explain_image("./images/xray_perturbed.jpg")
    expl.explain_attribution_diff(attribution, perturbed_attribution, np_mask)
    print(results)

def perturb_classify_all_patches(image: str, strength: float = 0.3):
    """
    Perturbs and classifies each 16x16 patch in a 512x512 image.
    
    Args:
        image (str): Path to the input image
        strength (float): Strength of the perturbation
    """
    patch_size = 32
    image_size = 512
    
    # Calculate the number of patches in each dimension
    num_patches_x = image_size // patch_size  # 32
    num_patches_y = image_size // patch_size  # 32
    
    total_patches = num_patches_x * num_patches_y  # 1024
    print(f"Processing {total_patches} patches of size {patch_size}x{patch_size} in a {image_size}x{image_size} image")
    
    # Load models once
    print("Loading models...")
    sd_pipe = sd.load_model()
    processor, model, label_columns = tf.load_classifier()
    print("Models loaded successfully")
    
    # Classify the original image once
    print("\nClassifying original image:")
    original_results = classify_image(image, model, processor, label_columns)
    
    # Track important information
    changed_patches = []
    confidence_impacts = []
    original_class = original_results['predicted_class_label']
    original_class_idx = original_results['predicted_class_idx']
    original_confidence = original_results['probabilities'][original_class_idx].item()
    print(f"Original confidence for {original_class}: {original_confidence:.4f}")
    
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            patch_position = (x * patch_size, y * patch_size)
            patch_num = y*num_patches_x + x + 1
            
            print(f"\nPatch {patch_num}/{total_patches} at position {patch_position}:")
            
            # Perturb the image at the current patch using the pre-loaded model
            perturbed_image = perturb_image(image, patch_position, strength, sd_pipe)
            
            # Classify the perturbed image using the pre-loaded model
            print(f"Classification results:")
            results = classify_image(perturbed_image, model, processor, label_columns)
            
            # Get new confidence for the original class (even if classification changed)
            new_confidence_for_original = results['probabilities'][original_class_idx].item()
            confidence_drop = original_confidence - new_confidence_for_original
            
            # Track all relevant information
            if results['predicted_class_label'] != original_class:
                changed_patches.append((patch_position, results['predicted_class_label'], new_confidence_for_original))
                
            # Record confidence impact regardless of classification change
            confidence_impacts.append((patch_position, confidence_drop, new_confidence_for_original))
    
    print(f"\nCompleted processing all {total_patches} patches.")
    
    # Sort patches by confidence impact (highest drop first)
    confidence_impacts.sort(key=lambda x: x[1], reverse=True)
    
    # Summary of patches that changed the classification
    if changed_patches:
        print(f"\nFound {len(changed_patches)} patches that changed classification from {original_class}:")
        for pos, new_class, conf in changed_patches:
            print(f"  Position {pos}: Changed to {new_class} (confidence for {original_class}: {conf:.4f})")
    else:
        print("\nNo patches changed the classification.")
    
    # Summary of patches with highest confidence impact
    print(f"\nTop 10 patches with greatest reduction in confidence for {original_class}:")
    for i, (pos, drop, new_conf) in enumerate(confidence_impacts[:10]):
        print(f"  {i+1}. Position {pos}: Confidence reduced by {drop:.4f} (from {original_confidence:.4f} to {new_conf:.4f})")