import explanation as expl
import transformer as tf
from typing import Optional, Tuple, List
import sd
import pandas as pd
from pathlib import Path
import numpy as np
import json
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

DATA_DIR = "./images"
OUTPUT_DIR = "./results"
CACHE_DIR = "./cache"
PATCH_DIR = "./results/patches"

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

        # Convert probabilities to a list for JSON serialization
        probabilities_list = classification["probabilities"].tolist()

        
        result = {
            "image_path": str(image_path),
            "vit_input_path": str(vit_input_path),
            "predicted_class": classification["predicted_class_label"],
            "predicted_class_idx": classification["predicted_class_idx"],
            "confidence": float(classification["probabilities"][classification["predicted_class_idx"]]),
            "probabilities": probabilities_list,
            "attribution_path": str(attribution_path),
            "attribution_vis_path": str(vis_path)
        }
        
        with open(cache_file, 'w') as f:
            json.dump(result, f)
        
        results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / f"classification_results{output_suffix}.csv", index=False)
    
    return df

def perturb_low_attribution_areas(results_df: pd.DataFrame, sd_pipe: StableDiffusionInpaintPipeline, percentile_threshold: int = 15, strength: float = 0.2) -> List[Path]:
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

    perturbed_image_paths = []

    for _, row in results_df.iterrows():
        processed_filename = Path(row["image_path"]).name
        print(processed_filename)
        patient_id, original_filename = processed_filename.split('_', 1)
        original_filename = original_filename.split('.')[0]
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

def perturb_all_patches(results_df, sd_pipe, patch_size=16, strength=0.2, max_images=None):
    """
    Perturbs each individual patch in the images one by one and saves the results.
    
    Args:
        results_df: DataFrame with classification results
        sd_pipe: StableDiffusionInpaintPipeline instance (optional)
        patch_size: Size of each patch to perturb
        strength: Perturbation strength
        max_images: Maximum number of images to process (for testing)
        
    Returns:
        List of paths to perturbed images
    """
    output_dir = Path(OUTPUT_DIR)
    perturbed_dir = output_dir / "patches"
    mask_dir = output_dir / "patch_masks"
    for directory in [perturbed_dir, mask_dir]:
        directory.mkdir(exist_ok=True, parents=True)

    perturbed_image_paths = []
    
    if max_images:
        processing_df = results_df.head(max_images)
    else:
        processing_df = results_df

    for _, row in processing_df.iterrows():
        image_path = row["image_path"]
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        processed_filename = Path(image_path).name
        print(f"Processing: {processed_filename}")
        patient_id, original_filename = processed_filename.split('_', 1)
        original_filename = original_filename.split('.')[0]
        attribution = np.load(row["attribution_path"])
        
        num_patches_x = width // patch_size
        
        # Process each patch
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                patch_id = (y // patch_size) * num_patches_x + (x // patch_size)
                
                # Calculate mean attribution in this patch
                patch_attribution = attribution[y:y+patch_size, x:x+patch_size]
                mean_attribution = np.mean(patch_attribution)
                
                # Create unique patch identifier
                exp_id = f"{patient_id}_{original_filename}_patch{patch_id}_x{x}_y{y}_s{strength}"
                perturbed_image_path = perturbed_dir / f'{exp_id}.jpg'
                mask_path = mask_dir / f'{exp_id}_mask.npy'

                if perturbed_image_path.exists():
                    perturbed_image_paths.append(perturbed_image_path)
                else: 
                    try:
                        result_image, np_mask = sd.perturb_single_patch(
                            sd_pipe,
                            (patient_id, original_filename),
                            (x, y, patch_size),
                            strength=strength
                        )
                        
                        # Save the perturbed image
                        result_image.save(perturbed_image_path)
                        np.save(mask_path, np_mask)
                        perturbed_image_paths.append(perturbed_image_path)
                        
                        print(f"Perturbed patch {patch_id} at ({x}, {y}) - Attribution: {mean_attribution:.4f}")

                    except Exception as e:
                        print(f"Error perturbing patch {patch_id} in {processed_filename}: {e}")
                        continue
                
                mean_id = f"{patient_id}_{original_filename}_patch{patch_id}_x{x}_y{y}_mean"
                mean_image_path = perturbed_dir / f'{mean_id}.jpg'
                mask_path = mask_dir / f'{mean_id}_mask.npy'

                if mean_image_path.exists():
                    perturbed_image_paths.append(mean_image_path)
                else:
                    mean_image, mean_mask = sd.mean_image_patch((patient_id, original_filename), (x,y,patch_size))
                    mean_image.save(mean_image_path)
                    np.save(mask_path, mean_mask)
                    perturbed_image_paths.append(mean_image_path)

    return perturbed_image_paths