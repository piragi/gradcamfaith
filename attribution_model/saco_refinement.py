# refinement.py
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def refine_attribution_map(attribution_map: np.ndarray,
                           patch_metrics: pd.DataFrame,
                           patch_size: int = 16,
                           learning_rate: float = 0.1) -> np.ndarray:
    """
    Refine attribution map based on patch-specific SaCo metrics.
    
    Args:
        attribution_map: Original attribution map
        patch_metrics: DataFrame with patch-specific SaCo metrics
        patch_size: Size of each patch in pixels
        learning_rate: Rate of adjustment for each patch
        
    Returns:
        Refined attribution map
    """
    refined_map = attribution_map.copy()

    # Calculate grid size based on the attribution map dimensions
    map_size = attribution_map.shape[0]
    grid_size = map_size // patch_size

    for _, row in patch_metrics.iterrows():
        try:
            patch_id = row['patch_id']
            patch_saco = row['patch_saco']

            # Calculate grid coordinates from patch_id
            y = (patch_id // grid_size) * patch_size
            x = (patch_id % grid_size) * patch_size

            # Skip if coordinates are out of bounds
            if y >= map_size or x >= map_size:
                continue

            # Get patch region
            y_end = min(y + patch_size, map_size)
            x_end = min(x + patch_size, map_size)
            patch_region = refined_map[y:y_end, x:x_end]

            # Determine adjustment direction and amount based on patch SaCo
            # If patch_saco is negative, the attribution is overestimated
            adjustment_factor = learning_rate * (1 - patch_saco)

            if patch_saco < 0:  # Overattributed
                # Reduce attribution (multiply by a factor < 1)
                reduction_factor = max(0.5, 1 - abs(adjustment_factor))
                patch_region *= reduction_factor
            else:  # Underattributed
                # Increase attribution (add small amount based on current values)
                # Avoid exceeding 1.0 by calculating the max possible increase
                max_value = patch_region.max()
                if max_value < 1.0:
                    max_increase = min(0.2,
                                       1.0 - max_value)  # Limit max increase
                    increase_amount = adjustment_factor * max_increase
                    patch_region += increase_amount * patch_region  # Proportional increase

            # Update the refined map with the adjusted patch
            refined_map[y:y_end, x:x_end] = patch_region

        except Exception as e:
            print(f"Error processing patch {patch_id}: {e}")
            continue

    # Renormalize the entire map to maintain the [0,1] range
    refined_map = (refined_map - refined_map.min()) / (
        refined_map.max() - refined_map.min() + 1e-8)

    return refined_map


def regularized_refinement(original_map: np.ndarray,
                           current_map: np.ndarray,
                           patch_metrics: pd.DataFrame,
                           learning_rate: float = 0.1,
                           reg_weight: float = 0.5,
                           patch_size: int = 16) -> np.ndarray:
    """
    Refine attribution map with regularization to stay close to original.
    
    Args:
        original_map: Original attribution map
        current_map: Current attribution map being refined
        patch_metrics: DataFrame with patch-specific SaCo metrics
        learning_rate: Rate of adjustment for each patch
        reg_weight: Weight for regularization (higher = closer to original)
        patch_size: Size of each patch in pixels
        
    Returns:
        Regularized and refined attribution map
    """
    # Apply refinement to the current map
    unregularized_refinement = refine_attribution_map(current_map,
                                                      patch_metrics,
                                                      patch_size,
                                                      learning_rate)

    # Apply regularization toward original map
    regularized_map = (
        1 - reg_weight) * unregularized_refinement + reg_weight * original_map

    # Normalize
    regularized_map = (regularized_map - regularized_map.min()) / \
                      (regularized_map.max() - regularized_map.min() + 1e-8)

    return regularized_map


def update_attribution_values(working_data: pd.DataFrame,
                              attribution_map: np.ndarray,
                              patch_size: int = 16) -> pd.DataFrame:
    """
    Update mean_attribution values in working data based on the refined attribution map.
    
    Args:
        working_data: DataFrame with patch IDs and mean_attribution values
        attribution_map: Refined attribution map
        patch_size: Size of each patch in pixels
        
    Returns:
        Updated working data with new mean_attribution values
    """
    map_size = attribution_map.shape[0]
    grid_size = map_size // patch_size

    updated_data = working_data.copy()

    for idx, row in updated_data.iterrows():
        patch_id = row['patch_id']

        # Calculate coordinates from patch_id
        y = (patch_id // grid_size) * patch_size
        x = (patch_id % grid_size) * patch_size

        # Skip if coordinates are out of bounds
        if y >= map_size or x >= map_size:
            continue

        # Get patch region
        y_end = min(y + patch_size, map_size)
        x_end = min(x + patch_size, map_size)
        patch_region = attribution_map[y:y_end, x:x_end]

        # Calculate new mean attribution
        updated_data.at[idx, 'mean_attribution'] = np.mean(patch_region)

    return updated_data


def iterate_saco_refinement(
    original_attribution: np.ndarray,
    comparison_df: pd.DataFrame,
    initial_saco: float,
    image_name: str,
    max_iterations: int = 10,
    learning_rate: float = 0.05,
    initial_reg_weight: float = 0.7,
    min_reg_weight: float = 0.2,
    decay_rate: float = 0.9,
    patch_size: int = 16,
    output_dir: Optional[str] = "./results/refined"
) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
    """
    Iteratively refine attribution map to improve SaCo score - simplified version.
    
    Args:
        original_attribution: Original attribution map
        comparison_df: DataFrame with patch attribution comparisons
        initial_saco: Initial SaCo score for this image
        image_name: Name of the image being processed
        max_iterations: Maximum number of refinement iterations
        learning_rate: Learning rate for refinement
        initial_reg_weight: Initial regularization weight (higher = closer to original)
        min_reg_weight: Minimum regularization weight
        decay_rate: Rate at which regularization weight decays
        patch_size: Size of each patch in pixels
        output_dir: Directory to save refinement results
        
    Returns:
        Tuple of (refined_attribution, final_saco, iteration_stats)
    """
    from analysis import (analyze_patch_metrics,
                          calculate_image_saco_with_details)

    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(exist_ok=True, parents=True)

    # Initialize with original attribution
    current_map = original_attribution.copy()

    # Get current SaCo score
    current_saco = initial_saco

    # Extract only the columns needed for SaCo calculation
    image_data = comparison_df[comparison_df['original_image'] == image_name]
    image_data = image_data.sort_values('mean_attribution',
                                        ascending=False).reset_index(drop=True)

    # Create a working copy of the data
    working_data = image_data[[
        'patch_id', 'mean_attribution', 'confidence_delta_abs'
    ]].copy()

    # Initialize variables for tracking
    reg_weight = initial_reg_weight
    iteration_stats = []
    best_map = current_map.copy()
    best_saco = current_saco

    print(f"Initial SaCo score for {image_name}: {current_saco:.4f}")

    for iteration in range(max_iterations):
        try:
            # Get current attribution values and confidence impacts
            attributions = working_data['mean_attribution'].values
            confidence_impacts = working_data['confidence_delta_abs'].values
            patch_ids = working_data['patch_id'].values

            # Calculate patch-specific metrics
            _, pair_data = calculate_image_saco_with_details(
                attributions, confidence_impacts, patch_ids)

            patch_metrics_df = analyze_patch_metrics({image_name: pair_data})

            # Apply regularized refinement
            refined_map = regularized_refinement(
                original_map=original_attribution,
                current_map=current_map,
                patch_metrics=patch_metrics_df,
                learning_rate=learning_rate,
                reg_weight=reg_weight,
                patch_size=patch_size)

            # Update mean_attribution values based on refined map
            working_data = update_attribution_values(working_data, refined_map,
                                                     patch_size)

            # Recalculate SaCo with updated attribution values
            new_attributions = working_data['mean_attribution'].values
            new_saco, _ = calculate_image_saco_with_details(
                new_attributions, confidence_impacts, patch_ids)

            # Record iteration stats
            iteration_stats.append({
                'iteration': iteration + 1,
                'saco_score': new_saco,
                'reg_weight': reg_weight,
                'improvement': new_saco - current_saco
            })

            print(
                f"Iteration {iteration+1}: SaCo {current_saco:.4f} -> {new_saco:.4f} "
                f"(reg_weight: {reg_weight:.2f})")

            # Save intermediate results
            if output_path:
                np.save(
                    output_path /
                    f"{Path(image_name).stem}_iter{iteration+1}.npy",
                    refined_map)

            # Keep changes if SaCo improved - simplified logic
            if new_saco > current_saco:
                current_map = refined_map.copy()
                current_saco = new_saco

                # Update best if this is the highest SaCo so far
                if new_saco > best_saco:
                    best_map = refined_map.copy()
                    best_saco = new_saco
            # If SaCo didn't improve, just keep the current map (no conservative step)

            # Decay regularization weight
            reg_weight = max(min_reg_weight, reg_weight * decay_rate)

        except Exception as e:
            print(f"Error in iteration {iteration+1}: {e}")
            continue

    # Save final refined map
    if output_path:
        np.save(output_path / f"{Path(image_name).stem}_final.npy", best_map)

    print(
        f"Refinement complete for {image_name}. Initial SaCo: {initial_saco:.4f}, Final SaCo: {best_saco:.4f}"
    )
    return best_map, best_saco, iteration_stats


def run_refinement_pipeline(comparison_df: pd.DataFrame,
                            saco_scores: Dict[str, float],
                            original_results_df: pd.DataFrame,
                            output_dir: str = "./results/refined",
                            max_iterations: int = 10,
                            learning_rate: float = 0.05) -> pd.DataFrame:
    """
    Run the refinement pipeline on a dataset using existing comparison data and SaCo scores.
    
    Args:
        comparison_df: DataFrame with patch attribution comparisons
        saco_scores: Dictionary mapping image names to their SaCo scores
        original_results_df: DataFrame with original classification results
        output_dir: Directory to save refinement results
        max_iterations: Maximum refinement iterations
        learning_rate: Learning rate for refinement
        
    Returns:
        DataFrame with refinement results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create a DataFrame for SaCo scores
    saco_df = pd.DataFrame({
        'image_name': list(saco_scores.keys()),
        'saco_score': list(saco_scores.values())
    })
    print(f"Initial average SaCo score: {saco_df['saco_score'].mean():.4f}")

    refinement_results = []

    # Process each unique original image
    unique_images = comparison_df['original_image'].unique()

    for image_name in unique_images:
        try:
            # Get original attribution
            image_rows = original_results_df[original_results_df['image_path']
                                             == image_name]
            if image_rows.empty:
                print(
                    f"Warning: No data found for {image_name} in original_results_df"
                )
                continue

            original_attribution_path = image_rows['attribution_path'].iloc[0]
            original_attribution = np.load(original_attribution_path)

            print(f"\nRefining attribution for {image_name}")

            # Get the initial SaCo score for this image
            initial_saco = saco_scores.get(image_name, 0.0)

            # Iteratively refine attribution - no per-image directories
            refined_map, final_saco, iteration_stats = iterate_saco_refinement(
                original_attribution=original_attribution,
                comparison_df=comparison_df,
                initial_saco=initial_saco,  # Pass only the relevant score
                image_name=image_name,
                max_iterations=max_iterations,
                learning_rate=learning_rate,
                output_dir=output_dir  # Save directly to the output directory
            )

            # Save final refined attribution
            refined_path = output_path / f"{Path(image_name).stem}_refined.npy"
            np.save(refined_path, refined_map)

            # Record results
            refinement_results.append({
                'original_image':
                image_name,
                'original_attribution_path':
                original_attribution_path,
                'refined_attribution_path':
                str(refined_path),
                'initial_saco':
                initial_saco,
                'final_saco':
                final_saco,
                'improvement':
                final_saco - initial_saco,
                'iterations':
                len(iteration_stats)
            })

        except Exception as e:
            print(f"Error refining attribution for {image_name}: {e}")
            continue

    # Create summary DataFrame
    results_df = pd.DataFrame(refinement_results)

    # Save summary results
    if not results_df.empty:
        results_path = output_path / "refinement_results.csv"
        results_df.to_csv(results_path, index=False)

        # Print summary statistics
        avg_improvement = results_df['improvement'].mean()
        avg_final_saco = results_df['final_saco'].mean()
        print(f"\nRefinement complete. Summary:")
        print(f"Average SaCo improvement: {avg_improvement:.4f}")
        print(f"Average final SaCo: {avg_final_saco:.4f}")
        print(f"Results saved to {results_path}")

    return results_df
