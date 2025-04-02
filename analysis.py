import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import explanation as expl
import sd

OUTPUT_DIR = "./results"

def compare_attributions(original_results_df: pd.DataFrame, perturbed_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare attributions between original and perturbed images.
    
    Args:
        original_results_df: DataFrame with original classification results
        perturbed_results_df: DataFrame with perturbed classification results (patch-perturbed)
        
    Returns:
        DataFrame with attribution comparison results
    """
    output_dir = Path(OUTPUT_DIR)
    comparison_dir = output_dir / "comparisons"
    patch_mask_dir = output_dir / "patch_masks"
    comparison_dir.mkdir(exist_ok=True, parents=True)
    comparison_results = []
    
    for _, perturbed_row in perturbed_results_df.iterrows():
        perturbed_path = Path(perturbed_row["image_path"])
        perturbed_filename = perturbed_path.stem
        
        # Extract the patient ID and original filename from the perturbed filename
        # Format: {patient_id}_{original_filename}_patch{patch_id}_x{x}_y{y}_s{strength}
        filename_parts = perturbed_filename.split('_', 1)
        if len(filename_parts) < 2:
            print(f"Skipping {perturbed_filename}: unexpected filename format")
            continue
            
        patient_id = filename_parts[0]
        
        # Extract the original filename part (before "_patch")
        if "_patch" not in perturbed_filename:
            print(f"Skipping {perturbed_filename}: not a patch-perturbed file")
            continue
            
        original_part = perturbed_filename.split("_patch")[0]
        
        # Find corresponding original image
        matching_rows = original_results_df[original_results_df["image_path"].str.contains(original_part)]
        
        if matching_rows.empty:
            print(f"No matching original image found for {perturbed_filename}")
            continue
            
        original_row = matching_rows.iloc[0]
        
        # Get mask path
        mask_path = patch_mask_dir / f"{perturbed_filename}_mask.npy"
        
        if not mask_path.exists():
            print(f"Mask file not found: {mask_path}")
            continue
            
        try:
            # Load attributions
            original_attribution = np.load(original_row["attribution_path"])
            perturbed_attribution = np.load(perturbed_row["attribution_path"])
            np_mask = np.load(mask_path)
            
            # Generate comparison visualization
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
            
            # Extract patch coordinates from filename
            x, y = -1, -1
            patch_id = -1
            
            # Try to extract patch info from filename
            filename_parts = perturbed_filename.split('_')
            for part in filename_parts:
                if part.startswith('patch'):
                    try:
                        patch_id = int(part[5:])
                    except ValueError:
                        pass
                elif part.startswith('x'):
                    try:
                        x = int(part[1:])
                    except ValueError:
                        pass
                elif part.startswith('y'):
                    try:
                        y = int(part[1:])
                    except ValueError:
                        pass
            
            # Calculate mean attribution in the patch if coordinates are found
            mean_attribution = None
            if x >= 0 and y >= 0:
                patch_size = 16  # Default patch size
                patch_end_x = min(x + patch_size, original_attribution.shape[1])
                patch_end_y = min(y + patch_size, original_attribution.shape[0])
                
                if x < original_attribution.shape[1] and y < original_attribution.shape[0]:
                    patch_attribution = original_attribution[y:patch_end_y, x:patch_end_x]
                    mean_attribution = np.mean(patch_attribution)
            
            result = {
                "original_image": original_row["image_path"],
                "perturbed_image": str(perturbed_path),
                "patch_id": patch_id,
                "x": x,
                "y": y,
                "mean_attribution": mean_attribution,
                "original_class": original_row["predicted_class"],
                "perturbed_class": perturbed_row["predicted_class"],
                "class_changed": original_row["predicted_class_idx"] != perturbed_row["predicted_class_idx"],
                "original_confidence": original_row["confidence"],
                "perturbed_confidence": perturbed_row["confidence"],
                "confidence_delta": perturbed_row["confidence"] - original_row["confidence"],
                "confidence_delta_abs": abs(perturbed_row["confidence"] - original_row["confidence"]),
                "vit_input_ssim": ssim_score,
                "comparison_path": str(comparison_path)
            }
            
            # Add key metrics from diff_stats
            for category in ["original_stats", "perturbed_stats", "difference_stats"]:
                if category in diff_stats:
                    for key, value in diff_stats[category].items():
                        result[f"{category}_{key}"] = value
            
            comparison_results.append(result)
            
        except Exception as e:
            print(f"Error processing {perturbed_filename}: {e}")
            continue
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    if not comparison_df.empty:
        # Add rankings for impact and attribution
        if "confidence_delta_abs" in comparison_df.columns:
            comparison_df["impact_rank"] = comparison_df["confidence_delta_abs"].rank(ascending=False)
        
        if "mean_attribution" in comparison_df.columns:
            comparison_df["attribution_rank"] = comparison_df["mean_attribution"].rank(ascending=False)
            
        if "impact_rank" in comparison_df.columns and "attribution_rank" in comparison_df.columns:
            comparison_df["rank_difference"] = comparison_df["attribution_rank"] - comparison_df["impact_rank"]
    
        comparison_df.to_csv(output_dir / "patch_attribution_comparisons.csv", index=False)
    else:
        print("Warning: No comparison results were generated.")
    
    return comparison_df

def calculate_saco_with_details(data_path: str = "./results/patch_attribution_comparisons.csv", method: str = "mean"):
    data_df = pd.read_csv(data_path)
    if method:
        data_df = data_df[data_df['perturbed_image'].str.contains(f"_{method}.jpg")]
    
    results = {}
    pair_data = {}
    
    for image_name, image_data in data_df.groupby('original_image'):
        image_data = image_data.sort_values('mean_attribution', ascending=False).reset_index(drop=True)
        attributions = image_data['mean_attribution'].values
        confidence_impacts = image_data['confidence_delta_abs'].values
        patch_ids = image_data['patch_id'].values
        
        saco_score, image_pair_data = calculate_image_saco_with_details(
            attributions, confidence_impacts, patch_ids)
        
        results[image_name] = saco_score
        pair_data[image_name] = image_pair_data
        print(f"Calculated SaCo for {image_name}: {saco_score:.4f}")
    
    print(f"Average SaCo score: {sum(list(results.values())) / len(results)}")
    return results, pair_data

def calculate_image_saco_with_details(attributions, confidence_impacts, patch_ids):
    """Calculate SaCo and return detailed pair-wise comparison data"""
    F = 0
    total_weight = 0
    pairs_data = []
    
    for i in range(len(attributions)-1):
        for j in range(i+1, len(attributions)):
            attr_diff = attributions[i] - attributions[j]
            impact_i, impact_j = confidence_impacts[i], confidence_impacts[j]
            patch_i, patch_j = patch_ids[i], patch_ids[j]
            
            # Calculate weight for SaCo
            weight = attr_diff if impact_i >= impact_j else -attr_diff
            F += weight
            total_weight += abs(weight)
            
            # Store pair data
            pair_info = {
                'patch_i': patch_i,
                'patch_j': patch_j,
                'is_faithful': impact_i >= impact_j,
                'weight': weight
            }
            pairs_data.append(pair_info)
    
    F /= total_weight
    return F, pd.DataFrame(pairs_data)

def analyze_patch_metrics(pair_data):
    """Analyze the pair-wise comparison data and return a flat DataFrame with patch-specific metrics"""
    rows = []
    
    for image_name, image_pairs in pair_data.items():
        unique_patches = set(image_pairs['patch_i'].tolist() + image_pairs['patch_j'].tolist())
        
        for patch_id in unique_patches:
            pairs_with_i = image_pairs[image_pairs['patch_i'] == patch_id]
            pairs_with_j = image_pairs[image_pairs['patch_j'] == patch_id]
            
            # For pairs where patch is the first one (i)
            weights_i = pairs_with_i['weight'].tolist()
            faithful_i = pairs_with_i['is_faithful'].sum()
            
            # For pairs where patch is the second one (j)
            weights_j = pairs_with_j['weight'].tolist()
            faithful_j = pairs_with_j['is_faithful'].sum()
            
            # Combine data from both directions
            all_weights = weights_i + weights_j
            total_faithful = faithful_i + faithful_j
            total_pairs = len(pairs_with_i) + len(pairs_with_j)
            
            # Calculate patch-specific SaCo
            patch_saco = sum(all_weights) / sum(abs(w) for w in all_weights) if all_weights else 0
            
            # Create a row for this patch
            row = {
                'image_name': image_name,
                'patch_id': patch_id,
                'faithful_pairs_count': total_faithful,
                'unfaithful_pairs_count': total_pairs - total_faithful,
                'faithful_pairs_pct': (total_faithful / total_pairs * 100) if total_pairs else 0,
                'patch_saco': patch_saco
            }
            
            rows.append(row)

    return pd.DataFrame(rows)