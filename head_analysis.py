"""
Head analysis module for calculating and analyzing head contributions to class directions.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from config import PipelineConfig
from data_types import ClassificationResult
from vit.model import VisionTransformer


def calculate_head_direction_similarities(
        original_results: List[ClassificationResult],
        model: VisionTransformer,
        config: PipelineConfig,
        num_layers_to_analyze: int = 5,
        save_dir: Optional[Path] = None) -> Dict:
    """
    Calculate cosine similarity between head outputs and class directional vectors.
    """
    if save_dir is None:
        save_dir = config.file.output_dir / "head_analysis"
    os.makedirs(save_dir, exist_ok=True)

    # Get class weight vectors
    class_weights = model.head.weight.detach().cpu().numpy()
    num_classes = len(class_weights)

    results_by_image = {}

    print(f"Processing {len(original_results)} images")
    for result in tqdm(original_results,
                       desc="Calculating direction similarities"):
        # Skip if no head contribution data
        if not result.attribution_paths or not result.attribution_paths.head_contribution_path.exists(
        ):
            continue

        # Get predicted class and other classes
        predicted_class = result.prediction.predicted_class_idx
        other_classes = [i for i in range(num_classes) if i != predicted_class]

        # Get predicted class vector
        class_vector_predicted = class_weights[predicted_class]

        # Calculate direction vectors
        # 1. For each other class individually
        direction_vectors = {}
        for other_class in other_classes:
            direction = class_vector_predicted - class_weights[other_class]
            # Normalize direction
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
            direction_vectors[other_class] = direction

        # 2. Direction against SUM of ALL other classes
        other_class_vectors = class_weights[other_classes]
        other_class_sum = np.sum(other_class_vectors, axis=0)
        direction_all = class_vector_predicted - other_class_sum
        direction_all_norm = np.linalg.norm(direction_all)
        if direction_all_norm > 0:
            direction_all = direction_all / direction_all_norm
        direction_vectors['all'] = direction_all

        # Load head contributions
        head_contributions = np.load(
            result.attribution_paths.head_contribution_path, allow_pickle=True)
        n_layers = len(head_contributions)

        # Focus on last num_layers_to_analyze layers
        layers_to_analyze = min(num_layers_to_analyze, n_layers)
        start_layer = n_layers - layers_to_analyze

        # Get sample dimensions - handle 4-dimensional data
        sample_layer_data = head_contributions[start_layer]["activity_data"]

        # Handle 4D shape: (n_heads, batch, n_tokens, d_model)
        if len(sample_layer_data.shape) == 4:
            n_heads, batch_size, n_tokens, d_model = sample_layer_data.shape
            # We'll use batch_idx = 0 since we're processing one image at a time
            batch_idx = 0
        else:
            # Handle 3D shape if it's structured differently: (n_heads, n_tokens, d_model)
            n_heads, n_tokens, d_model = sample_layer_data.shape
            batch_idx = None

        # Initialize arrays to store similarities for this image
        # One for each direction: [layers, heads, tokens]
        similarities = {}
        for direction_key in direction_vectors.keys():
            similarities[direction_key] = np.zeros(
                (layers_to_analyze, n_heads, n_tokens))

        # Calculate similarities
        for layer_offset in range(layers_to_analyze):
            layer_idx = start_layer + layer_offset
            layer_data = head_contributions[layer_idx]["activity_data"]

            # Calculate for all heads and tokens
            for head_idx in range(n_heads):
                for token_idx in range(n_tokens):
                    # Get token representation, accounting for batch dimension if present
                    if batch_idx is not None:
                        token_repr = layer_data[head_idx, batch_idx, token_idx]
                    else:
                        token_repr = layer_data[head_idx, token_idx]

                    token_norm = np.linalg.norm(token_repr)

                    if token_norm == 0:
                        continue

                    # Normalize token
                    norm_token = token_repr / token_norm

                    # Calculate cosine similarity with each direction
                    for direction_key, direction_vector in direction_vectors.items(
                    ):
                        # Calculate cosine similarity (directions are already normalized)
                        cosine_sim = np.dot(norm_token, direction_vector)
                        similarities[direction_key][layer_offset, head_idx,
                                                    token_idx] = cosine_sim

        # Store results for this image
        results_by_image[result.image_path.stem] = {
            'predicted_class': predicted_class,
            'similarities': similarities
        }

    # Save combined results
    output_path = save_dir / f"head_direction_similarities{config.file.output_suffix}.npy"
    np.save(output_path, results_by_image)
    print(f"Saved head direction similarities to {output_path}")

    return results_by_image


def analyze_class_specific_head_importance(
        direction_similarities: Dict,
        num_classes: int,
        num_heads: int = 12,
        num_layers: int = 5,
        importance_threshold: float = 0.5,
        cls_token_only: bool = True,
        save_dir: Optional[Path] = None,
        config: Optional[PipelineConfig] = None) -> Dict:
    """
    Analyze head importance based on direction similarity scores, broken down by class.
    
    Args:
        direction_similarities: Output from calculate_head_direction_similarities
        num_classes: Number of classes in the model
        num_heads: Number of heads per layer
        num_layers: Number of layers analyzed
        importance_threshold: Threshold for considering heads as important
        cls_token_only: Whether to focus only on CLS token
        save_dir: Directory to save results
        config: Pipeline configuration (for output suffix)
        
    Returns:
        Dictionary with class-specific head importance analysis
    """
    # Initialize class-specific storage
    class_head_importance = {
        cls_idx: np.zeros((num_layers, num_heads))
        for cls_idx in range(num_classes)
    }
    class_image_counts = {cls_idx: 0 for cls_idx in range(num_classes)}

    # Track all images by class
    images_by_class = {cls_idx: [] for cls_idx in range(num_classes)}

    # Track all similarity values for stats
    all_similarity_values = []

    # First pass: analyze similarities and track important heads
    for image_name, result in direction_similarities.items():
        pred_class = result['predicted_class']
        similarities = result['similarities']

        # Increment count for this class
        class_image_counts[pred_class] += 1
        images_by_class[pred_class].append(image_name)

        # Extract class direction similarity
        all_direction_sim = similarities[
            'all']  # Shape: [layers, heads, tokens]

        # Focus only on CLS token if specified
        if cls_token_only:
            focus_similarities = all_direction_sim[:, :, 0]  # CLS token only
        else:
            # Take mean across all tokens
            focus_similarities = np.mean(all_direction_sim, axis=2)

        # Collect all values for statistical analysis
        all_similarity_values.extend(focus_similarities.flatten())

        # Update importance for heads exceeding threshold
        for layer in range(num_layers):
            for head in range(num_heads):
                similarity = focus_similarities[layer, head]
                if abs(similarity) <= importance_threshold:
                    class_head_importance[pred_class][layer, head] += 1

    # Calculate similarity stats for informative threshold setting
    similarity_stats = {
        'mean': np.mean(all_similarity_values),
        'median': np.median(all_similarity_values),
        'std': np.std(all_similarity_values),
        'min': np.min(all_similarity_values),
        'max': np.max(all_similarity_values),
        'percentiles': {
            '25': np.percentile(all_similarity_values, 25),
            '50': np.percentile(all_similarity_values, 50),
            '75': np.percentile(all_similarity_values, 75),
            '90': np.percentile(all_similarity_values, 90),
            '95': np.percentile(all_similarity_values, 95),
        }
    }

    # Second pass: normalize by count to get percentage importance per class
    for cls_idx in range(num_classes):
        if class_image_counts[cls_idx] > 0:  # Avoid division by zero
            class_head_importance[cls_idx] /= class_image_counts[cls_idx]

    # Generate ranked heads for each class
    class_ranked_heads = {}
    for cls_idx in range(num_classes):
        flat_importances = []
        for layer in range(num_layers):
            for head in range(num_heads):
                importance = class_head_importance[cls_idx][layer, head]
                flat_importances.append((layer, head, importance))

        # Sort by importance score (descending)
        sorted_heads = sorted(flat_importances,
                              key=lambda x: x[2],
                              reverse=True)
        class_ranked_heads[cls_idx] = sorted_heads

    # Create a more readable output of top heads per class
    readable_top_heads = {}
    start_layer = 12 - num_layers
    for cls_idx in range(num_classes):
        if cls_idx in class_ranked_heads and class_image_counts[cls_idx] > 0:
            top_5_heads = class_ranked_heads[cls_idx][:5]  # Top 5 heads
            readable_top_heads[cls_idx] = [
                f"Layer {h[0] + start_layer}, Head {h[1]}: {h[2]:.4f}"
                for h in top_5_heads
            ]

    results = {
        'class_head_importance': class_head_importance,
        'class_ranked_heads': class_ranked_heads,
        'images_by_class': images_by_class,
        'class_image_counts': class_image_counts,
        'similarity_stats': similarity_stats,
        'top_heads_per_class': readable_top_heads
    }

    # Save results if path provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        output_suffix = config.file.output_suffix if config else ""
        output_path = save_dir / f"head_importance_analysis{output_suffix}.npy"
        np.save(output_path, results)
        print(f"Saved head importance analysis to {output_path}")

        # Also save as text for readability
        with open(save_dir / f"head_importance_summary{output_suffix}.txt",
                  'w') as f:
            f.write("CLASS-SPECIFIC HEAD IMPORTANCE ANALYSIS\n")
            f.write("=====================================\n\n")

            f.write("Similarity Statistics:\n")
            for k, v in similarity_stats.items():
                if k != 'percentiles':
                    f.write(f"  {k}: {v:.4f}\n")
                else:
                    f.write("  Percentiles:\n")
                    for pk, pv in v.items():
                        f.write(f"    {pk}%: {pv:.4f}\n")
            f.write("\n")

            f.write("Top Heads by Class:\n")
            for cls_idx, heads in readable_top_heads.items():
                f.write(
                    f"  Class {cls_idx} ({class_image_counts[cls_idx]} images):\n"
                )
                for i, head_desc in enumerate(heads):
                    f.write(f"    {i+1}. {head_desc}\n")
                f.write("\n")

    return results


def analyze_token_activation_patterns(direction_similarities,
                                      cls_idx,
                                      pca_components=20,
                                      n_clusters=5):
    """
    Find recurring patterns of token activations across images of the same class.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # Extract activation data for all images in this class
    class_images = []
    activation_matrices = []

    for img_name, data in direction_similarities.items():
        if data['predicted_class'] == cls_idx:
            class_images.append(img_name)
            # Get similarity values for 'all' direction
            similarities = data['similarities'][
                'all']  # [layers, heads, tokens]
            # Flatten to 1D vector
            flat_activations = similarities.flatten()
            activation_matrices.append(flat_activations)

    if not activation_matrices:
        return None

    # Create feature matrix where rows=images, columns=token-head activations
    feature_matrix = np.array(activation_matrices)

    # Step 1: Apply PCA to reduce dimensions
    pca = PCA(n_components=min(pca_components, len(activation_matrices)))
    pca_result = pca.fit_transform(feature_matrix)

    # Step 2: Apply k-means to identify clusters
    kmeans = KMeans(n_clusters=min(n_clusters, len(activation_matrices)))
    cluster_labels = kmeans.fit_predict(pca_result)

    # Analyze clusters
    cluster_stats = []
    for cluster_id in range(kmeans.n_clusters):
        # Get images in this cluster
        cluster_images = [
            class_images[i] for i in range(len(class_images))
            if cluster_labels[i] == cluster_id
        ]

        # Get centroid in original space
        centroid_pca = kmeans.cluster_centers_[cluster_id]
        centroid_original = pca.inverse_transform(centroid_pca.reshape(
            1, -1)).flatten()

        # Reshape back to [layers, heads, tokens]
        layers, heads, tokens = similarities.shape
        centroid_reshaped = centroid_original.reshape(layers, heads, tokens)

        # Find top activated token-head pairs
        flat_indices = np.argsort(centroid_original)[
            -20:]  # Top 20 activations
        top_activations = []
        for idx in flat_indices:
            l = idx // (heads * tokens)
            h = (idx % (heads * tokens)) // tokens
            t = idx % tokens
            activation = centroid_original[idx]
            top_activations.append((l, h, t, activation))

        cluster_stats.append({
            'cluster_id':
            cluster_id,
            'size':
            len(cluster_images),
            'percentage':
            len(cluster_images) / len(class_images),
            'top_activations':
            top_activations,
            'image_examples':
            cluster_images[:5]  # First 5 examples
        })

    return {
        'n_images': len(class_images),
        'clusters': cluster_stats,
        'explained_variance': pca.explained_variance_ratio_,
        'cluster_labels': cluster_labels,
        'pca_result': pca_result
    }


def run_head_analysis(original_results: List[ClassificationResult],
                      model: VisionTransformer,
                      config: PipelineConfig,
                      num_layers_to_analyze: int = 5,
                      importance_threshold: float = 0.2,
                      cls_token_only: bool = True,
                      analyze_token_patterns: bool = True,
                      pca_components: int = 10,
                      n_clusters: int = 3):
    """
    Run the complete head analysis process:
    1. Calculate head direction similarities
    2. Analyze class-specific head importance
    3. (Optional) Analyze token activation patterns
    
    Args:
        original_results: List of classification results
        model: Loaded ViT model
        config: Pipeline configuration
        num_layers_to_analyze: Number of last layers to analyze
        importance_threshold: Threshold for considering heads as important
        cls_token_only: Whether to focus only on CLS token
        analyze_token_patterns: Whether to perform token pattern analysis
        pca_components: Number of PCA components for pattern analysis
        n_clusters: Number of clusters for pattern analysis
        
    Returns:
        Tuple of (direction_similarities, head_importance_analysis, token_patterns)
    """
    save_dir = config.file.output_dir / "head_analysis"
    os.makedirs(save_dir, exist_ok=True)

    print("Step 1: Calculating head direction similarities...")
    direction_similarities = calculate_head_direction_similarities(
        original_results, model, config, num_layers_to_analyze, save_dir)

    # Get number of classes from the model
    num_classes = model.head.weight.shape[0]
    num_heads = 12

    print("Step 2: Analyzing class-specific head importance...")
    head_importance = analyze_class_specific_head_importance(
        direction_similarities, num_classes, num_heads, num_layers_to_analyze,
        importance_threshold, cls_token_only, save_dir, config)

    token_patterns = {}
    if analyze_token_patterns:
        print("Step 3: Analyzing token activation patterns...")
        # Check which classes have images
        classes_with_images = {
            cls_idx: count
            for cls_idx, count in
            head_importance['class_image_counts'].items() if count > 0
        }

        for cls_idx, count in classes_with_images.items():
            if count >= n_clusters:  # Only run if enough images for clustering
                print(
                    f"  Analyzing patterns for class {cls_idx} ({count} images)..."
                )
                patterns = analyze_token_activation_patterns(
                    direction_similarities,
                    cls_idx,
                    pca_components=pca_components,
                    n_clusters=min(n_clusters, count))

                if patterns:
                    token_patterns[cls_idx] = patterns

                    # Save patterns to file
                    output_path = save_dir / f"token_patterns_class_{cls_idx}{config.file.output_suffix}.npy"
                    np.save(output_path, patterns)

                    # Create human-readable summary
                    with open(
                            save_dir /
                            f"token_patterns_summary_class_{cls_idx}{config.file.output_suffix}.txt",
                            'w') as f:
                        f.write(
                            f"TOKEN ACTIVATION PATTERNS FOR CLASS {cls_idx}\n")
                        f.write("=======================================\n\n")
                        f.write(
                            f"Total images analyzed: {patterns['n_images']}\n")
                        f.write(
                            f"PCA explained variance: {sum(patterns['explained_variance'][:pca_components]):.4f}\n\n"
                        )

                        f.write("PATTERN CLUSTERS:\n")
                        for i, cluster in enumerate(patterns['clusters']):
                            f.write(f"\nCluster {i+1}:\n")
                            f.write(
                                f"  Size: {cluster['size']} images ({cluster['percentage']*100:.1f}%)\n"
                            )
                            f.write(
                                f"  Example images: {', '.join(cluster['image_examples'])}\n"
                            )
                            f.write(
                                "  Top token activations (Layer, Head, Token, Value):\n"
                            )

                            # Correct layer numbering - add offset to show actual layer numbers
                            # (if analyzing last 5 layers of a 12-layer model)
                            layer_offset = 12 - num_layers_to_analyze

                            for l, h, t, v in sorted(
                                    cluster['top_activations'],
                                    key=lambda x: x[3],
                                    reverse=True)[:10]:
                                actual_layer = l + layer_offset
                                f.write(
                                    f"    Layer {actual_layer}, Head {h}, Token {t}: {v:.5f}\n"
                                )

                        f.write("\n\n")
            else:
                print(
                    f"  Skipping class {cls_idx}: not enough images for clustering ({count})"
                )

    print("Head analysis complete!")
    print(f"Results saved to {save_dir}")

    return direction_similarities, head_importance, token_patterns
