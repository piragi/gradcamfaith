from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

import perturbation
import transformer as trans

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
            diff_stats = trans.explain_attribution_diff(
                original_attribution,
                perturbed_attribution,
                np_mask,
                base_name=perturbed_filename,
                save_dir=str(comparison_dir)
            )
            
            # Calculate SSIM between the actual 224x224 ViT inputs
            ssim_score = None
            if "attribution_vis_path" in original_row and "attribution_vis_path" in perturbed_row:
                original_vit_img = Image.open(original_row["attribution_vis_path"]).convert('RGB')
                perturbed_vit_img = Image.open(perturbed_row["attribution_vis_path"]).convert('RGB')
                ssim_score = perturbation.patch_similarity(original_vit_img, perturbed_vit_img)
            
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

def analyze_faithfulness_vs_correctness(saco_scores):
    """
    Analyze the relationship between attribution faithfulness and prediction correctness
    using both patient_id and study_id for accurate matching.
    Disregards images where the filename doesn't contain study ID.
    """
    # Convert SaCo scores to DataFrame
    image_metrics = pd.DataFrame.from_dict(saco_scores, orient='index', columns=['image_saco'])
    image_metrics.index.name = 'image_name'
    image_metrics.reset_index(inplace=True)
    
    # Load validation labels
    valid_df = pd.read_csv("valid.csv")
    
    # Print debugging info
    print(f"Number of images with SaCo scores: {len(image_metrics)}")
    print(f"Number of images in validation set: {len(valid_df)}")
    
    # Extract patient_id AND study_id from image names
    def extract_ids_from_image(image_path):
        """Extract both patient_id and study_id from an image path"""
        filename = image_path.split('/')[-1]
        parts = filename.split('_')
        
        # Check if "study" is in the filename
        if len(parts) >= 2 and "study" in parts[1]:
            patient_id = parts[0].replace('patient', '')
            study_id = parts[1].replace('study', '')
            return patient_id, study_id
        else:
            # For debugging purposes, still print a warning
            print(f"Warning: No study ID in filename {filename}, disregarding")
            return None, None
    
    # Extract IDs from SaCo image names
    image_metrics[['patient_id', 'study_id']] = image_metrics['image_name'].apply(
        lambda x: pd.Series(extract_ids_from_image(x))
    )
    
    # Filter out images without study ID
    initial_count = len(image_metrics)
    image_metrics = image_metrics.dropna(subset=['study_id'])
    filtered_count = initial_count - len(image_metrics)
    print(f"Filtered out {filtered_count} images without study ID from SaCo scores")
    
    # Extract IDs from validation data paths
    valid_df[['patient_id', 'study_id']] = valid_df['Path'].apply(
        lambda x: pd.Series((x.split('/')[-3].replace('patient', ''), 
                            x.split('/')[-2].replace('study', '')))
    )
    
    # Print sample entries to verify extraction
    print("\nSample SaCo image names with extracted IDs:")
    print(image_metrics[['image_name', 'patient_id', 'study_id']].head())
    
    print("\nSample validation paths with extracted IDs:")
    print(valid_df[['Path', 'patient_id', 'study_id']].head())
    
    # Target classes in order
    target_classes = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']
    
    # Load classification results
    class_results = pd.read_csv("./results/classification_results.csv")
    
    # Extract original image filename and IDs
    class_results['original_image'] = class_results['image_path'].apply(
        lambda x: 'images/' + x.split('/')[-1].split('_patch')[0] + '.jpg')
    
    # Extract both patient_id and study_id from classification results
    class_results[['patient_id', 'study_id']] = class_results['original_image'].apply(
        lambda x: pd.Series(extract_ids_from_image(x))
    )
    
    # Filter out classification results without study ID
    initial_count = len(class_results)
    class_results = class_results.dropna(subset=['study_id'])
    filtered_count = initial_count - len(class_results)
    print(f"Filtered out {filtered_count} images without study ID from classification results")
    
    # Parse the probabilities column
    class_results['prob_list'] = class_results['probabilities'].apply(
        lambda x: eval(x))
    
    # Print sample class results to verify extraction
    print("\nSample classification results with extracted IDs:")
    print(class_results[['original_image', 'patient_id', 'study_id']].head())
    
    # Merge data on BOTH patient_id AND study_id
    # First merge SaCo scores with classification results
    merged_df = image_metrics.merge(
        class_results[['original_image', 'patient_id', 'study_id', 'predicted_class', 
                      'predicted_class_idx', 'confidence', 'prob_list']].drop_duplicates('original_image'),
        on=['patient_id', 'study_id'], how='inner')
    
    print(f"\nAfter first merge - matching records: {len(merged_df)}")
    
    # Then merge with validation data
    results_df = merged_df.merge(
        valid_df[['patient_id', 'study_id'] + target_classes], 
        on=['patient_id', 'study_id'], how='inner')
    
    sample_case = results_df.sample(5)
    print(sample_case[['image_name', 'patient_id', 'study_id', 
                      'predicted_class', 'Cardiomegaly', 'Edema', 
                      'Consolidation', 'Pneumonia', 'No Finding']])
    print(f"After second merge - matching records: {len(results_df)}")
    
    # If we have matches, proceed with analysis
    if len(results_df) > 0:
        # Determine if prediction is correct
        results_df['predicted_class'] = results_df['predicted_class'].apply(
            lambda x: target_classes[int(x.split('_')[1])])
        
        # Add is_correct column
        results_df['is_correct'] = results_df.apply(
            lambda row: row[row['predicted_class']] == 1.0, axis=1)
        
        # Calculate metrics
        correct_saco = results_df[results_df['is_correct']]['image_saco'].mean()
        incorrect_saco = results_df[~results_df['is_correct']]['image_saco'].mean()
        
        print(f"Average SaCo for correct predictions: {correct_saco:.4f}")
        print(f"Average SaCo for incorrect predictions: {incorrect_saco:.4f}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='is_correct', y='image_saco', data=results_df)
        plt.title('SaCo Scores by Prediction Correctness')
        plt.xlabel('Prediction Correct')
        plt.ylabel('SaCo Score')
        plt.savefig('faithfulness_vs_correctness.png')
        plt.close()
        
        # Additional analysis: Calculate metrics per class
        print("\nClass-wise breakdown:")
        for cls in target_classes:
            class_data = results_df[results_df['predicted_class'] == cls]
            if len(class_data) > 0:
                correct_class = class_data[class_data['is_correct']]
                incorrect_class = class_data[~class_data['is_correct']]
                
                print(f"\n{cls}:")
                print(f"  Total predictions: {len(class_data)}")
                print(f"  Correct: {len(correct_class)} ({len(correct_class)/len(class_data)*100:.1f}%)")
                
                if len(correct_class) > 0:
                    print(f"  Avg SaCo (correct): {correct_class['image_saco'].mean():.4f}")
                
                if len(incorrect_class) > 0:
                    print(f"  Avg SaCo (incorrect): {incorrect_class['image_saco'].mean():.4f}")
        
        return results_df
    else:
        print("No matching records found. Check the file naming patterns and IDs.")
        
        # Detailed diagnostic information
        print("\nDiagnostic Information:")
        
        print("\n1. Unique patient IDs in SaCo data:")
        print(image_metrics['patient_id'].value_counts().head())
        
        print("\n2. Unique patient IDs in validation data:")
        print(valid_df['patient_id'].value_counts().head())
        
        print("\n3. Unique patient IDs in classification results:")
        print(class_results['patient_id'].value_counts().head())
        
        print("\n4. Unique study IDs in SaCo data:")
        print(image_metrics['study_id'].value_counts().head())
        
        print("\n5. Unique study IDs in validation data:")
        print(valid_df['study_id'].value_counts().head())
        
        print("\n6. Sample filenames from each source:")
        print("SaCo image names:", image_metrics['image_name'].head(3).tolist())
        print("Validation paths:", valid_df['Path'].head(3).tolist())
        print("Classification images:", class_results['original_image'].head(3).tolist())
        
        # Check for any potential matches (patient_id only)
        patient_matches = set(image_metrics['patient_id']) & set(valid_df['patient_id']) & set(class_results['patient_id'])
        print(f"\nNumber of patient IDs common across all datasets: {len(patient_matches)}")
        
        # Check potential study matches
        if len(patient_matches) > 0:
            print("\nAnalyzing a sample patient to debug study ID matching:")
            sample_patient = list(patient_matches)[0]
            print(f"Sample patient ID: {sample_patient}")
            
            saco_studies = set(image_metrics[image_metrics['patient_id'] == sample_patient]['study_id'])
            valid_studies = set(valid_df[valid_df['patient_id'] == sample_patient]['study_id'])
            class_studies = set(class_results[class_results['patient_id'] == sample_patient]['study_id'])
            
            print(f"SaCo studies: {saco_studies}")
            print(f"Validation studies: {valid_studies}")
            print(f"Classification studies: {class_studies}")
            
            common_studies = saco_studies & valid_studies & class_studies
            print(f"Common studies: {common_studies}")
        
        return None
