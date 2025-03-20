import explanation as expl
import transformer as tf
from typing import Optional, Tuple, List
import sd
import pandas as pd
from pathlib import Path
import numpy as np
import json
from PIL import Image

DATA_DIR = "./images"
OUTPUT_DIR = "./results"
CACHE_DIR = "./cache"

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
    result_image, np_mask = sd.perturb_non_attribution(sd_pipe, image, attribution, strength=0.2, percentile_threshold=20)
    results, perturbed_attribution = expl.explain_image("./images/xray_perturbed.jpg")
    expl.explain_attribution_diff(attribution, perturbed_attribution, np_mask)
    print(results)

def preprocess_dataset(source_dir: str = "./chexpert", 
                       dest_dir: str = "./images", 
                       target_size: tuple = (224,224)) -> List[Path]:
    """
    Preprocess JPG images by resizing them to target_size and saving them to dest_dir.
    Can recursively search through subfolders and filter for frontal X-rays.
    
    Args:
        source_dir: Directory containing original images
        dest_dir: Directory to save resized images
        target_size: Target size (width, height) for the resized images
        recursive: If True, search recursively through all subfolders
        only_frontal: If True, only process images with '_frontal' in the filename
        
    Returns:
        List of paths to the processed images
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(exist_ok=True, parents=True)
    
    image_files = list(source_path.glob("**/*.jpg"))
    
    image_files = [img for img in image_files if "_frontal" in img.name]
    print(f"Found {len(image_files)} frontal X-rays")
    
    processed_paths = []
    for image_file in image_files:
        # Create a unique output filename to avoid conflicts from different subfolders
        patient_id = image_file.parent.parent.name
        output_filename = f"{patient_id}_{image_file.name}"
        output_path = dest_path / output_filename
        
        if output_path.exists():
            print(f"Skipping {output_path.name} - already exists in {dest_dir}")
            processed_paths.append(output_path)
            continue
        
        try:
            img = Image.open(image_file).convert('RGB')
            resized_img = img.resize(target_size, Image.LANCZOS)
            resized_img.save(output_path)
            processed_paths.append(output_path)
            
            print(f"Processed: {image_file.name} -> {output_path}")
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
    
    print(f"Preprocessing complete. {len(processed_paths)} images saved to {dest_dir}")
    return processed_paths

def classify(data_directory: Optional[str] = None, output_suffix: str = "") -> pd.DataFrame:
    """
    Classify images and generate attribution maps.
    
    Args:
        data_directory: Optional custom directory containing images to classify.
                       If None, uses default DATA_DIR.
        output_suffix: Optional suffix for output files to distinguish different runs
        
    Returns:
        DataFrame with classification results and paths to attribution maps.
    """
    output_dir = Path(OUTPUT_DIR)
    cache_dir = Path(CACHE_DIR)
    attribution_dir = output_dir / f"attributions{output_suffix}"
    vit_inputs_dir = output_dir / "vit_inputs"
    
    if data_directory is None:
        data_dir = Path(DATA_DIR)
    else:
        data_dir = Path(data_directory)
    
    image_paths = list(data_dir.glob("*.jpg"))
    
    for directory in [output_dir, cache_dir, attribution_dir, vit_inputs_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    explainer = expl.TransLRPExplainer()
    results = []
    
    for image_path in image_paths:
        # check cache
        cache_file = cache_dir / f"{image_path.stem}_classification{output_suffix}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                result = json.load(f)
                results.append(result)
                continue
        
        image, classification = explainer.classify_image(image_path=str(image_path))

        # save classified image for similarity comparisons
        vit_input_path = vit_inputs_dir / f"{image_path.stem}{output_suffix}.jpg"
        image.save(vit_input_path)

        image, attribution = explainer.explain(image_path, classification['predicted_class_idx'])
        attribution_path = attribution_dir / f"{image_path.stem}_attribution.npy"
        np.save(attribution_path, attribution)
        vis_path = attribution_dir / f"{image_path.stem}_vis.png"
        explainer.visualize(image, attribution, save_path=str(vis_path))
        
        result = {
            "image_path": str(image_path),
            "vit_input_path": str(vit_input_path),
            "predicted_class": classification["predicted_class_label"],
            "predicted_class_idx": classification["predicted_class_idx"],
            "confidence": float(classification["probabilities"][classification["predicted_class_idx"]]),
            "attribution_path": str(attribution_path),
            "attribution_vis_path": str(vis_path)
        }
        
        with open(cache_file, 'w') as f:
            json.dump(result, f)
        
        results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / f"classification_results{output_suffix}.csv", index=False)
    
    return df

def perturb_low_attribution_areas(results_df: pd.DataFrame, percentile_threshold: int = 15, strength: float = 0.2) -> List[Path]:
    """
    Perturb low attribution areas of all images and save the results.
    
    Args:
        results_df: DataFrame with classification results (from run_initial_classify)
        percentile_threshold: Percentile threshold for determining attribution regions
        strength: Perturbation strength
        
    Returns:
        List of paths to perturbed images
    """
    output_dir = Path(OUTPUT_DIR)
    perturbed_dir = output_dir / "perturbed"
    mask_dir = output_dir / "masks"
    for directory in [perturbed_dir, mask_dir]:
        directory.mkdir(exist_ok=True, parents=True)

    sd_pipe = sd.load_model()
    perturbed_image_paths = []

    for _, row in results_df.iterrows():
        processed_filename = Path(row["image_path"]).name
        print(processed_filename)
        patient_id, original_filename = processed_filename.split('_', 1)
        original_filename = original_filename.split('.')[0]
        image_path = f'./chexpert/{patient_id}/study1/{original_filename}.jpg'
        attribution_path = row["attribution_path"]
        exp_id = f"{patient_id}_{original_filename}_non_attr_p{percentile_threshold}_s{strength}"
        perturbed_image_path = perturbed_dir / f'{exp_id}.jpg'
        mask_path = mask_dir / f'{exp_id}_mask.npy'

        if perturbed_image_path.exists():
            perturbed_image_paths.append(perturbed_image_path)
            continue

        attribution = np.load(attribution_path)
        result_image, np_mask = sd.perturb_non_attribution(
            sd_pipe, 
            (patient_id, original_filename), 
            attribution, 
            percentile_threshold=percentile_threshold,
            strength=strength
        )
        
        # Save the perturbed image
        result_image.save(perturbed_image_path)
        np.save(mask_path, np_mask)
        perturbed_image_paths.append(perturbed_image_path)
        
        print(f"Perturbed and saved: {perturbed_image_path}")
    
    return perturbed_image_paths

def compare_attributions(original_results_df: pd.DataFrame, perturbed_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare attributions between original and perturbed images.

    Args:
        original_results_df: DataFrame with original classification results
        perturbed_results_df: DataFrame with perturbed classification results

    Returns:
        DataFrame with attribution comparison results
    """
    output_dir = Path(OUTPUT_DIR)
    comparison_dir = output_dir / "comparisons"
    mask_dir = output_dir / "masks"
    comparison_dir.mkdir(exist_ok=True, parents=True)
    comparison_results = []

    for _, perturbed_row in perturbed_results_df.iterrows():
        perturbed_path = Path(perturbed_row["image_path"])
        perturbed_filename = perturbed_path.stem
        original_name = perturbed_filename.split("_non_attr")[0]
        original_row = original_results_df[original_results_df["image_path"].str.contains(f"/{original_name}.jpg")].iloc[0]

        original_attribution = np.load(original_row["attribution_path"])
        perturbed_attribution = np.load(perturbed_row["attribution_path"])
        np_mask = np.load(mask_dir / f'{perturbed_filename}_mask.npy')

        comparison_path = comparison_dir / f"{perturbed_filename}_comparison.png"
        diff_stats = expl.explain_attribution_diff(
            original_attribution, 
            perturbed_attribution, 
            np_mask,
            base_name=perturbed_filename,
            save_dir=str(comparison_dir)
        )

        # Calculate SSIM between the actual 224x224 ViT inputs
        ssim_score = None
        if "vit_input_path" in original_row and "vit_input_path" in perturbed_row:
            original_vit_img = Image.open(original_row["vit_input_path"]).convert('RGB')
            perturbed_vit_img = Image.open(perturbed_row["vit_input_path"]).convert('RGB')
            ssim_score = sd.patch_similarity(original_vit_img, perturbed_vit_img)


        result = {
            "original_image": original_row["image_path"],
            "perturbed_image": perturbed_path,
            "original_class": original_row["predicted_class"],
            "perturbed_class": perturbed_row["predicted_class"],
            "class_changed": original_row["predicted_class_idx"] != perturbed_row["predicted_class_idx"],
            "original_confidence": original_row["confidence"],
            "perturbed_confidence": perturbed_row["confidence"],
            "confidence_delta": perturbed_row["confidence"] - original_row["confidence"],
            "vit_input_ssim": ssim_score,
            "comparison_path": str(comparison_path)
        }

        # Add key metrics from diff_stats
        for category in ["original_stats", "perturbed_stats", "difference_stats"]:
            if category in diff_stats:
                for key, value in diff_stats[category].items():
                    result[f"{category}_{key}"] = value

        comparison_results.append(result)

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(output_dir / "attribution_comparisons.csv", index=False)
    
    return comparison_df

def calculate_perturbation_statistics(mask_dir: str = "./results/masks", image_size=(224, 224), patch_size=16):
    """
    Calculate statistics about the amount of patches perturbed across all masks.
    
    Args:
        mask_dir: Directory containing mask .npy files
        image_size: Size of the images (default: 224x224)
        patch_size: Size of each patch (default: 16)
        
    Returns:
        Dictionary with perturbation statistics
    """
    from pathlib import Path
    import numpy as np
    
    mask_dir_path = Path(mask_dir)
    mask_files = list(mask_dir_path.glob("*_mask.npy"))
    
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        return {}
    
    # Calculate grid dimensions
    patch_grid = (image_size[0] // patch_size, image_size[1] // patch_size)
    total_patches = patch_grid[0] * patch_grid[1]
    print(f"Image size: {image_size}, Patch size: {patch_size}, Total patches: {total_patches}")
    
    # Calculate perturbation stats for each mask
    perturbation_fractions = []
    patch_counts = []
    
    for mask_file in mask_files:
        # Load pixel-level mask
        pixel_mask = np.load(mask_file)
        
        if pixel_mask.shape != image_size:
            print(f"Warning: Mask shape {pixel_mask.shape} doesn't match expected image size {image_size}")
            # Skip this mask or try to resize it
            continue
        
        # Convert pixel-level mask to patch-level mask
        patch_mask = np.zeros((patch_grid[0], patch_grid[1]), dtype=bool)
        
        # For each patch, check if any pixel in that patch is perturbed
        for i in range(patch_grid[0]):
            for j in range(patch_grid[1]):
                patch_pixels = pixel_mask[
                    i * patch_size:(i + 1) * patch_size,
                    j * patch_size:(j + 1) * patch_size
                ]
                # If any pixel in the patch is True (perturbed), mark the patch as perturbed
                patch_mask[i, j] = np.any(patch_pixels)
        
        # Count perturbed patches
        perturbed_count = np.count_nonzero(patch_mask)
        perturbation_fraction = perturbed_count / total_patches
        
        patch_counts.append(perturbed_count)
        perturbation_fractions.append(perturbation_fraction)
        
    # Calculate statistics
    stats = {
        "total_masks": len(mask_files),
        "total_patches_per_image": total_patches,
        "mean_perturbed_patches": np.mean(patch_counts),
        "median_perturbed_patches": np.median(patch_counts),
        "mean_perturbation_fraction": np.mean(perturbation_fractions),
        "median_perturbation_fraction": np.median(perturbation_fractions),
        "min_perturbation_fraction": np.min(perturbation_fractions),
        "max_perturbation_fraction": np.max(perturbation_fractions)
    }
    
    print(f"Perturbation Statistics:")
    print(f"Total masks analyzed: {stats['total_masks']}")
    print(f"Total patches per image: {stats['total_patches_per_image']}")
    print(f"Mean perturbed patches: {stats['mean_perturbed_patches']:.2f} / {total_patches}")
    print(f"Median perturbed patches: {stats['median_perturbed_patches']:.2f} / {total_patches}")
    print(f"Mean perturbation fraction: {stats['mean_perturbation_fraction']:.2%}")
    print(f"Median perturbation fraction: {stats['median_perturbation_fraction']:.2%}")
    print(f"Min perturbation fraction: {stats['min_perturbation_fraction']:.2%}")
    print(f"Max perturbation fraction: {stats['max_perturbation_fraction']:.2%}")
    
    # Create a histogram of perturbation fractions
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(perturbation_fractions, bins=20, edgecolor='black')
        plt.title('Distribution of Perturbation Fractions')
        plt.xlabel('Fraction of Patches Perturbed')
        plt.ylabel('Number of Images')
        plt.axvline(np.mean(perturbation_fractions), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(perturbation_fractions):.2%}')
        plt.axvline(np.median(perturbation_fractions), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(perturbation_fractions):.2%}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the histogram
        histogram_path = Path(mask_dir).parent / "perturbation_histogram.png"
        plt.savefig(histogram_path)
        print(f"Histogram saved to: {histogram_path}")
        
        # Create a second histogram showing the count of perturbed patches
        plt.figure(figsize=(10, 6))
        plt.hist(patch_counts, bins=20, edgecolor='black')
        plt.title('Distribution of Perturbed Patch Counts')
        plt.xlabel('Number of Patches Perturbed')
        plt.ylabel('Number of Images')
        plt.axvline(np.mean(patch_counts), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(patch_counts):.2f}')
        plt.axvline(np.median(patch_counts), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(patch_counts):.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the second histogram
        histogram_path2 = Path(mask_dir).parent / "perturbation_count_histogram.png"
        plt.savefig(histogram_path2)
        print(f"Patch count histogram saved to: {histogram_path2}")
    except Exception as e:
        print(f"Could not create histogram: {e}")
    
    return stats