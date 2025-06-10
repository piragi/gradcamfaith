"""
Head analysis module for calculating and analyzing head contributions to class directions.
"""

import os
import pprint
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from config import PipelineConfig
from data_types import ClassificationResult
from vit.model import VisionTransformer


def calculate_head_direction_similarities(
    original_results: List[ClassificationResult],
    model: VisionTransformer,
    config: PipelineConfig,
    num_layers_to_analyze: int = 5,
    save_dir: Optional[Path] = None
) -> Dict:
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
    for result in tqdm(original_results, desc="Calculating direction similarities"):
        # Skip if no head contribution data
        if not result.attribution_paths or not result.attribution_paths.head_contribution_path.exists():
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

        other_class_mean = np.mean(other_class_vectors, axis=0)
        direction_all_mean = class_vector_predicted - other_class_mean
        direction_all_mean_norm = np.linalg.norm(direction_all_mean)
        # if direction_all_mean_norm > 0:
        # direction_all_mean = direction_all_mean / direction_all_mean_norm
        direction_vectors['all_mean'] = direction_all_mean

        # Load head contributions
        head_contributions = result.attribution_paths.load_head_contributions()
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
            similarities[direction_key] = np.zeros((layers_to_analyze, n_heads, n_tokens))

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
                    for direction_key, direction_vector in direction_vectors.items():
                        # Calculate cosine similarity (directions are already normalized)
                        # TODO: direct logit attribution rather than similarity
                        # cosine_sim = np.dot(norm_token, direction_vector)
                        direct_attribution = np.dot(token_repr, direction_vector)
                        similarities[direction_key][layer_offset, head_idx, token_idx] = direct_attribution

        # Store results for this image
        results_by_image[result.image_path.stem] = {'predicted_class': predicted_class, 'similarities': similarities}

    # Save combined results
    output_path = save_dir / f"head_direction_similarities{config.file.output_suffix}.npy"
    np.save(output_path, results_by_image)
    print(f"Saved head direction similarities to {output_path}")

    return results_by_image


def generate_and_print_token_config(
    token_patterns: Dict,
    layer_offset: int,
    activation_threshold: float,
    top_k_tokens: int,
    min_cluster_size: int = 100
) -> Dict[int, Dict[int, Dict[int, List[int]]]]:
    """
    Analyzes token patterns, extracts a robust configuration for boosting,
    and prints it as a clean, copy-paste-ready Python dictionary string.
    """
    token_boost_config = {}

    print("\n--- Generating Clean Token Boost Config ---\n")

    for class_idx_str, class_pattern_data in token_patterns.items():
        class_idx = int(class_idx_str)

        if not class_pattern_data or 'clusters' not in class_pattern_data:
            continue

        current_class_config: Dict[int, Dict[int, List[int]]] = {}
        print(f"Processing Class {class_idx}...")

        for cluster_info in class_pattern_data['clusters']:
            if cluster_info['size'] < min_cluster_size:
                continue

            sorted_activations = sorted(cluster_info['top_activations'], key=lambda x: x[3], reverse=True)

            tokens_to_add = []
            for rel_layer, head, token, activation in sorted_activations[:top_k_tokens]:
                if activation >= activation_threshold:
                    tokens_to_add.append((rel_layer, head, token))
                else:
                    break

            for rel_layer, head, token in tokens_to_add:
                actual_layer_idx = int(rel_layer) + layer_offset
                head_idx = int(head)
                token_idx = int(token)

                layer_dict = current_class_config.setdefault(actual_layer_idx, {})
                head_list = layer_dict.setdefault(head_idx, [])

                if token_idx not in head_list:
                    head_list.append(token_idx)

        if current_class_config:
            for l_idx in current_class_config:
                for h_idx in current_class_config[l_idx]:
                    current_class_config[l_idx][h_idx].sort()
            token_boost_config[class_idx] = current_class_config

    # Use a custom function to print to avoid any lingering type issues
    def dict_to_pretty_string(d, indent=0):
        lines = []
        is_multiline = len(d) > 1 or any(isinstance(v, dict) for v in d.values())

        if is_multiline:
            lines.append('{')
            for key, value in sorted(d.items()):
                lines.append(' ' * (indent + 4) + f"{key}: " + dict_to_pretty_string(value, indent + 4) + ",")
            lines.append(' ' * indent + '}')
            return '\n'.join(lines)
        else:  # Handle the innermost list of tokens on one line for compactness
            key, value = next(iter(d.items()))
            return '{' + f"{key}: {sorted(value)}" + '}'

    config_string = pprint.pformat(token_boost_config, indent=4, width=120)

    print("\n" + "=" * 80)
    print("COPY AND PASTE THE FOLLOWING DICTIONARY INTO YOUR PIPELINE CONFIG:")
    print("=" * 80)
    # The standard pprint should now work correctly after casting to int
    print(f"token_boost_details_per_class: Dict[int, Dict[int, Dict[int, List[int]]]] = \\\n{config_string}")
    print("=" * 80 + "\n")

    return token_boost_config


def analyze_class_specific_head_importance(
    direction_similarities: Dict,
    num_classes: int,
    num_heads: int = 12,
    num_layers: int = 4,
    importance_threshold: float = 0.5,
    cls_token_only: bool = True,
    save_dir: Optional[Path] = None,
    config: Optional[PipelineConfig] = None
) -> Dict:
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
    class_head_importance = {cls_idx: np.zeros((num_layers, num_heads)) for cls_idx in range(num_classes)}
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
        all_direction_sim = similarities['all_mean']  # Shape: [layers, heads, tokens]

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
                if similarity > importance_threshold:
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
        sorted_heads = sorted(flat_importances, key=lambda x: x[2], reverse=True)
        class_ranked_heads[cls_idx] = sorted_heads

    # Create a more readable output of top heads per class
    readable_top_heads = {}
    start_layer = 12 - num_layers
    for cls_idx in range(num_classes):
        if cls_idx in class_ranked_heads and class_image_counts[cls_idx] > 0:
            top_5_heads = class_ranked_heads[cls_idx][:5]  # Top 5 heads
            readable_top_heads[cls_idx] = [f"Layer {h[0] + start_layer}, Head {h[1]}: {h[2]:.4f}" for h in top_5_heads]

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
        with open(save_dir / f"head_importance_summary{output_suffix}.txt", 'w') as f:
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
                f.write(f"  Class {cls_idx} ({class_image_counts[cls_idx]} images):\n")
                for i, head_desc in enumerate(heads):
                    f.write(f"    {i+1}. {head_desc}\n")
                f.write("\n")

    return results


def analyze_token_activation_patterns(direction_similarities, cls_idx, pca_components=20, n_clusters=5):
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
            similarities = data['similarities']['all_mean']  # [layers, heads, tokens]
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
        cluster_images = [class_images[i] for i in range(len(class_images)) if cluster_labels[i] == cluster_id]

        # Get centroid in original space
        centroid_pca = kmeans.cluster_centers_[cluster_id]
        centroid_original = pca.inverse_transform(centroid_pca.reshape(1, -1)).flatten()

        # Reshape back to [layers, heads, tokens]
        layers, heads, tokens = similarities.shape
        centroid_reshaped = centroid_original.reshape(layers, heads, tokens)

        # Find top activated token-head pairs
        flat_indices = np.argsort(centroid_original)[-50:]  # Top 20 activations
        top_activations = []
        for idx in flat_indices:
            l = idx // (heads * tokens)
            h = (idx % (heads * tokens)) // tokens
            t = idx % tokens
            activation = centroid_original[idx]
            top_activations.append((l, h, t, activation))

        cluster_stats.append({
            'cluster_id': cluster_id,
            'size': len(cluster_images),
            'percentage': len(cluster_images) / len(class_images),
            'top_activations': top_activations,
            'image_examples': cluster_images[:5]  # First 5 examples
        })

    return {
        'n_images': len(class_images),
        'clusters': cluster_stats,
        'explained_variance': pca.explained_variance_ratio_,
        'cluster_labels': cluster_labels,
        'pca_result': pca_result
    }


def analyze_and_summarize_important_components(direction_similarities, num_classes, top_k=100):
    """
    Find top contributing components and analyze their consistency patterns to identify different classification strategies.
    """
    print("=" * 80)
    print("IMPORTANT COMPONENTS ANALYSIS - STRATEGY DISCOVERY")
    print("=" * 80)

    for class_idx in range(num_classes):
        # Get images for this class
        class_images = [
            name for name, result in direction_similarities.items() if result['predicted_class'] == class_idx
        ]

        if len(class_images) < 5:  # Skip classes with too few examples
            continue

        print(f"\nCLASS {class_idx} ({len(class_images)} images)")
        print("-" * 50)

        # Collect ALL component contributions (no filtering yet)
        component_stats = {}
        contributions_by_image = {}  # Track which images use which components

        for image_name in class_images:
            contributions = direction_similarities[image_name]['similarities']['all_mean']
            layers, heads, tokens = contributions.shape
            contributions_by_image[image_name] = contributions

            for layer in range(layers):
                for head in range(heads):
                    for token in range(tokens):
                        key = (layer, head, token)
                        if key not in component_stats:
                            component_stats[key] = []
                        component_stats[key].append(contributions[layer, head, token])

        # Rank ALL components by mean contribution (no consistency filter)
        all_components = []
        for (layer, head, token), contribs in component_stats.items():
            mean_contrib = np.mean(contribs)
            std_contrib = np.std(contribs)
            positive_ratio = sum(1 for c in contribs if c > 0) / len(contribs)
            max_contrib = np.max(contribs)

            all_components.append({
                'layer': layer,
                'head': head,
                'token': token,
                'mean': mean_contrib,
                'std': std_contrib,
                'consistency': positive_ratio,
                'max': max_contrib,
                'values': contribs
            })

        # Sort by mean contribution (absolute value to catch both positive and negative)
        all_components.sort(key=lambda x: abs(x['mean']), reverse=True)

        if all_components:
            print("Top Contributing Components (ranked by contribution magnitude):")
            for i, comp in enumerate(all_components[:top_k]):
                layer_actual = comp['layer'] + (12 - len(range(layers)))
                token_type = "CLS" if comp['token'] == 0 else f"Token{comp['token']}"

                # Consistency pattern analysis
                consistency_desc = "Always" if comp['consistency'] > 0.9 else \
                                  "Often" if comp['consistency'] > 0.7 else \
                                  "Sometimes" if comp['consistency'] > 0.3 else "Rarely"

                print(
                    f"  {i+1:2d}. Layer {layer_actual:2d} | Head {comp['head']:2d} | {token_type:>6} | "
                    f"Contrib: {comp['mean']:+.4f} | Consistency: {comp['consistency']*100:4.1f}% ({consistency_desc}) | "
                    f"Range: [{np.min(comp['values']):.3f}, {np.max(comp['values']):.3f}]"
                )

            # Strategy analysis: Look for different patterns
            print(f"\nStrategy Analysis:")

            # Find high contributors with different consistency patterns
            always_on = [c for c in all_components[:top_k * 2] if c['consistency'] > 0.9 and c['mean'] > 0]
            sometimes_on = [c for c in all_components[:top_k * 2] if 0.3 < c['consistency'] < 0.7 and c['mean'] > 0]

            if always_on:
                print(f"  Core Strategy Components (always active): {len(always_on)} components")
                for c in always_on[:3]:
                    layer_actual = c['layer'] + (12 - len(range(layers)))
                    token_type = "CLS" if c['token'] == 0 else f"Token{c['token']}"
                    print(f"    Layer {layer_actual}, Head {c['head']}, {token_type} (contrib: {c['mean']:+.3f})")

            if sometimes_on:
                print(f"  Conditional Strategy Components (selective): {len(sometimes_on)} components")

                # For conditional components, show which images activate them most
                for c in sometimes_on[:3]:
                    layer_actual = c['layer'] + (12 - len(range(layers)))
                    token_type = "CLS" if c['token'] == 0 else f"Token{c['token']}"

                    # Find images where this component is most active
                    component_activations = []
                    for img_name in class_images:
                        activation = contributions_by_image[img_name][c['layer'], c['head'], c['token']]
                        component_activations.append((activation, img_name))

                    component_activations.sort(reverse=True)

                    print(f"    Layer {layer_actual}, Head {c['head']}, {token_type} (contrib: {c['mean']:+.3f})")
                    print(
                        f"      Most active in: {[name.split('_')[-1][:15] for _, name in component_activations[:3]]}"
                    )
                    print(
                        f"      Least active in: {[name.split('_')[-1][:15] for _, name in component_activations[-3:]]}"
                    )

        print(
            f"\nSummary: Found {len([c for c in all_components if c['consistency'] > 0.9])} 'always-on' and "
            f"{len([c for c in all_components if 0.3 < c['consistency'] < 0.7])} 'conditional' components"
        )

    print("\n" + "=" * 80)


def analyze_discriminative_components(
    direction_similarities: Dict,
    num_classes: int,
    sparsity_threshold: float = 0.3,  # Activation frequency threshold
    magnitude_threshold: float = 0.5,  # Minimum activation strength
    top_k: int = 20
) -> Dict:
    """
    Find components with high discriminative power using multiple criteria:
    1. Sparsity: Tokens that activate rarely but strongly
    2. Specificity: High activation for target class, low for others
    3. Consistency: Reliable activation patterns within class
    """

    # Collect all activations by component and class
    component_stats = {}  # (layer, head, token) -> {class_idx: [activations]}

    for image_name, result in direction_similarities.items():
        pred_class = result['predicted_class']
        similarities = result['similarities']['all_mean']  # [layers, heads, tokens]

        for layer in range(similarities.shape[0]):
            for head in range(similarities.shape[1]):
                for token in range(similarities.shape[2]):
                    key = (layer, head, token)
                    if key not in component_stats:
                        component_stats[key] = {cls: [] for cls in range(num_classes)}

                    activation = similarities[layer, head, token]
                    component_stats[key][pred_class].append(activation)

    # Analyze discriminative power for each component
    discriminative_components = []

    for (layer, head, token), class_activations in component_stats.items():
        for target_class in range(num_classes):
            target_activations = class_activations[target_class]

            if len(target_activations) < 3:  # Need minimum samples
                continue

            # Calculate discriminative metrics
            target_mean = np.mean(target_activations)
            target_std = np.std(target_activations)
            target_max = np.max(target_activations)

            # Sparsity: How often does this component activate strongly?
            strong_activations = [a for a in target_activations if a > magnitude_threshold]
            activation_frequency = len(strong_activations) / len(target_activations)

            # Specificity: How much higher for target vs other classes?
            other_activations = []
            for other_class in range(num_classes):
                if other_class != target_class:
                    other_activations.extend(class_activations[other_class])

            if other_activations:
                other_mean = np.mean(other_activations)
                specificity_ratio = target_mean / (other_mean + 1e-8)  # Avoid division by zero
                specificity_diff = target_mean - other_mean
            else:
                specificity_ratio = float('inf')
                specificity_diff = target_mean

            # Discriminative power score (combine multiple factors)
            # High score = rare but strong + specific to this class
            rarity_bonus = 1.0 if activation_frequency < sparsity_threshold else activation_frequency
            discriminative_score = (target_max * specificity_ratio * rarity_bonus) / (target_std + 1e-8)

            discriminative_components.append({
                'layer': layer,
                'head': head,
                'token': token,
                'target_class': target_class,
                'discriminative_score': discriminative_score,
                'target_mean': target_mean,
                'target_max': target_max,
                'target_std': target_std,
                'activation_frequency': activation_frequency,
                'specificity_ratio': specificity_ratio,
                'specificity_diff': specificity_diff,
                'n_samples': len(target_activations),
                'strong_activations': strong_activations
            })

    # Sort by discriminative score
    discriminative_components.sort(key=lambda x: x['discriminative_score'], reverse=True)

    # Group by class for easier analysis
    by_class = {cls: [] for cls in range(num_classes)}
    for comp in discriminative_components:
        by_class[comp['target_class']].append(comp)

    return {
        'all_components': discriminative_components[:top_k * num_classes],
        'by_class': {
            cls: comps[:top_k]
            for cls, comps in by_class.items()
        },
        'stats': {
            'total_components_analyzed':
            len(component_stats),
            'components_with_discriminative_power':
            len([c for c in discriminative_components if c['discriminative_score'] > 1.0])
        }
    }


def find_sparse_high_impact_tokens(
    direction_similarities: Dict,
    activation_percentile: float = 90,  # Only look at top 10% of activations
    frequency_threshold: float = 0.4,  # Activate in <40% of images
    min_peak_activation: float = 0.3  # Minimum peak activation value
) -> List:
    """
    Specifically find tokens that 'light up only half the time but go way higher'
    """

    # Collect all activations per component across all images
    component_activations = {}  # (layer, head, token) -> [all_activations]

    for image_name, result in direction_similarities.items():
        similarities = result['similarities']['all_mean']

        for layer in range(similarities.shape[0]):
            for head in range(similarities.shape[1]):
                for token in range(similarities.shape[2]):
                    key = (layer, head, token)
                    if key not in component_activations:
                        component_activations[key] = []

                    component_activations[key].append(similarities[layer, head, token])

    sparse_high_impact = []

    for (layer, head, token), activations in component_activations.items():
        activations = np.array(activations)

        # Calculate activation statistics
        threshold = np.percentile(activations, activation_percentile)
        high_activations = activations[activations > threshold]

        if len(high_activations) == 0:
            continue

        # Frequency: How often does it activate highly?
        frequency = len(high_activations) / len(activations)

        # Peak impact: What's the maximum activation?
        peak_activation = np.max(activations)
        mean_high_activation = np.mean(high_activations)

        # Variance in activations (high variance suggests selective activation)
        activation_variance = np.var(activations)

        # Select sparse but high-impact components
        if (
            frequency < frequency_threshold and peak_activation > min_peak_activation and activation_variance > 0.01
        ):  # Some minimum variance

            sparse_high_impact.append({
                'layer': layer,
                'head': head,
                'token': token,
                'frequency': frequency,
                'peak_activation': peak_activation,
                'mean_high_activation': mean_high_activation,
                'activation_variance': activation_variance,
                'n_total': len(activations),
                'n_high': len(high_activations),
                'impact_score': peak_activation * (1 - frequency) * activation_variance
            })

    # Sort by impact score (high peaks, low frequency, high variance)
    sparse_high_impact.sort(key=lambda x: x['impact_score'], reverse=True)

    return sparse_high_impact


def generate_class_specific_token_report(
    direction_similarities: Dict, discriminative_analysis: Dict, sparse_analysis: List, num_layers_analyzed: int,
    save_dir: Path
):
    """
    Generate a comprehensive report of class-specific token findings
    """
    layer_offset = 12 - num_layers_analyzed

    with open(save_dir / "discriminative_token_report.txt", 'w') as f:
        f.write("DISCRIMINATIVE TOKEN ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        # Section 1: Top discriminative components per class
        f.write("1. TOP DISCRIMINATIVE COMPONENTS PER CLASS\n")
        f.write("-" * 40 + "\n")

        for cls_idx, components in discriminative_analysis['by_class'].items():
            if not components:
                continue

            f.write(f"\nClass {cls_idx}:\n")
            for i, comp in enumerate(components[:50]):
                actual_layer = comp['layer'] + layer_offset
                token_name = "CLS" if comp['token'] == 0 else f"T{comp['token']}"

                f.write(f"  {i+1:2d}. L{actual_layer} H{comp['head']} {token_name} | ")
                f.write(f"Score: {comp['discriminative_score']:.3f} | ")
                f.write(f"Freq: {comp['activation_frequency']*100:.1f}% | ")
                f.write(f"Peak: {comp['target_max']:.3f} | ")
                f.write(f"Specificity: {comp['specificity_ratio']:.2f}x\n")

        # Section 2: Sparse high-impact tokens
        f.write(f"\n\n2. SPARSE HIGH-IMPACT TOKENS (Top 20)\n")
        f.write("-" * 40 + "\n")
        f.write("These tokens activate rarely but with high magnitude:\n\n")

        for i, comp in enumerate(sparse_analysis[:50]):
            actual_layer = comp['layer'] + layer_offset
            token_name = "CLS" if comp['token'] == 0 else f"T{comp['token']}"

            f.write(f"{i+1:2d}. L{actual_layer} H{comp['head']} {token_name} | ")
            f.write(f"Peak: {comp['peak_activation']:.3f} | ")
            f.write(f"Freq: {comp['frequency']*100:.1f}% | ")
            f.write(f"Variance: {comp['activation_variance']:.4f}\n")

        # Section 3: Implementation suggestions
        f.write(f"\n\n3. IMPLEMENTATION SUGGESTIONS\n")
        f.write("-" * 40 + "\n")
        f.write("Based on this analysis, consider:\n\n")
        f.write("A. Boosting high-discriminative components for each class\n")
        f.write("B. Using sparse tokens as 'signature' features\n")
        f.write("C. Implementing conditional boosting based on activation patterns\n")
        f.write("D. Creating ensemble methods that weight different token types\n")

    print(f"Comprehensive discriminative analysis saved to {save_dir / 'discriminative_token_report.txt'}")


def analyze_frequent_class_specific_components(
    direction_similarities: Dict,
    num_classes: int,
    min_activation_threshold: float = 0.1,  # Minimum logit contribution to count as "active"
    top_k: int = 20
) -> Dict:
    """
    Extension to existing analysis: Find components that are frequent within their class
    but rare across other classes, using the same logit attribution data.
    """

    # Collect activations per component per class (same as your existing approach)
    component_stats = {}  # (layer, head, token) -> {class_idx: [activations]}

    for image_name, result in direction_similarities.items():
        pred_class = result['predicted_class']
        similarities = result['similarities']['all_mean']  # Your logit attributions

        for layer in range(similarities.shape[0]):
            for head in range(similarities.shape[1]):
                for token in range(similarities.shape[2]):
                    key = (layer, head, token)
                    if key not in component_stats:
                        component_stats[key] = {cls: [] for cls in range(num_classes)}

                    activation = similarities[layer, head, token]
                    component_stats[key][pred_class].append(activation)

    # Analyze class-specific frequency patterns
    frequent_components = []

    for (layer, head, token), class_activations in component_stats.items():
        for target_class in range(num_classes):
            target_activations = class_activations[target_class]

            if len(target_activations) < 3:
                continue

            target_activations = np.array(target_activations)

            # Calculate target class metrics
            target_active_count = np.sum(target_activations > min_activation_threshold)
            target_frequency = target_active_count / len(target_activations)
            target_mean = np.mean(target_activations)
            target_max = np.max(target_activations)
            target_std = np.std(target_activations)

            # Calculate other classes metrics
            other_frequencies = []
            other_means = []
            all_other_activations = []

            for other_class in range(num_classes):
                if other_class != target_class and class_activations[other_class]:
                    other_activations = np.array(class_activations[other_class])
                    other_active_count = np.sum(other_activations > min_activation_threshold)
                    other_frequency = other_active_count / len(other_activations)
                    other_mean = np.mean(other_activations)

                    other_frequencies.append(other_frequency)
                    other_means.append(other_mean)
                    all_other_activations.extend(other_activations)

            if not other_frequencies:
                continue

            # Summary statistics for other classes
            max_other_frequency = np.max(other_frequencies)
            mean_other_frequency = np.mean(other_frequencies)
            overall_other_mean = np.mean(all_other_activations)

            # Class specificity metrics
            frequency_advantage = target_frequency - mean_other_frequency
            frequency_ratio = target_frequency / (mean_other_frequency + 1e-8)
            magnitude_advantage = target_mean - overall_other_mean

            # Combined score emphasizing both frequency and magnitude
            class_specificity_score = frequency_advantage * magnitude_advantage * target_max

            frequent_components.append({
                'layer': layer,
                'head': head,
                'token': token,
                'target_class': target_class,
                'class_specificity_score': class_specificity_score,
                'target_frequency': target_frequency,
                'target_mean': target_mean,
                'target_max': target_max,
                'target_std': target_std,
                'max_other_frequency': max_other_frequency,
                'mean_other_frequency': mean_other_frequency,
                'overall_other_mean': overall_other_mean,
                'frequency_advantage': frequency_advantage,
                'frequency_ratio': frequency_ratio,
                'magnitude_advantage': magnitude_advantage,
                'n_target_samples': len(target_activations),
                'n_other_samples': len(all_other_activations)
            })

    # Sort by class specificity score
    frequent_components.sort(key=lambda x: x['class_specificity_score'], reverse=True)

    # Group by class
    by_class = {cls: [] for cls in range(num_classes)}
    for comp in frequent_components:
        by_class[comp['target_class']].append(comp)

    return {
        'all_components': frequent_components[:top_k * num_classes],
        'by_class': {
            cls: comps[:top_k]
            for cls, comps in by_class.items()
        },
        'stats': {
            'total_analyzed': len(component_stats),
            'components_with_freq_advantage': len([c for c in frequent_components if c['frequency_advantage'] > 0.2]),
            'components_by_target_frequency': {
                'high': len([c for c in frequent_components if c['target_frequency'] > 0.7]),
                'medium': len([c for c in frequent_components if 0.3 <= c['target_frequency'] <= 0.7]),
                'low': len([c for c in frequent_components if c['target_frequency'] < 0.3])
            }
        }
    }


def compare_frequency_patterns(
    direction_similarities: Dict, discriminative_analysis: Dict, frequent_analysis: Dict, num_layers_analyzed: int,
    save_dir: Path
):
    """
    Compare your original discriminative analysis with the frequency-focused analysis
    """
    layer_offset = 12 - num_layers_analyzed

    with open(save_dir / "frequency_pattern_comparison.txt", 'w') as f:
        f.write("FREQUENCY PATTERN ANALYSIS COMPARISON\n")
        f.write("=" * 60 + "\n\n")

        f.write("SUMMARY STATISTICS:\n")
        f.write(f"Original discriminative components: {len(discriminative_analysis['all_components'])}\n")
        f.write(f"Frequent class-specific components: {len(frequent_analysis['all_components'])}\n")
        f.write(
            f"High frequency components (>70%): {frequent_analysis['stats']['components_by_target_frequency']['high']}\n"
        )
        f.write(
            f"Medium frequency components (30-70%): {frequent_analysis['stats']['components_by_target_frequency']['medium']}\n"
        )
        f.write(
            f"Components with freq advantage >20%: {frequent_analysis['stats']['components_with_freq_advantage']}\n\n"
        )

        for cls_idx in range(len(discriminative_analysis['by_class'])):
            if cls_idx not in discriminative_analysis['by_class'] or not discriminative_analysis['by_class'][cls_idx]:
                continue

            f.write(f"CLASS {cls_idx} COMPARISON:\n")
            f.write("-" * 40 + "\n")

            f.write("Original Discriminative (Top 5):\n")
            for i, comp in enumerate(discriminative_analysis['by_class'][cls_idx][:25]):
                actual_layer = comp['layer'] + layer_offset
                token_name = "CLS" if comp['token'] == 0 else f"T{comp['token']}"
                f.write(f"  {i+1}. L{actual_layer} H{comp['head']} {token_name} | ")
                f.write(f"Freq: {comp['activation_frequency']*100:.1f}% | ")
                f.write(f"Peak: {comp['target_max']:.3f} | ")
                f.write(f"Spec: {comp['specificity_ratio']:.1f}x\n")

            f.write("\nFrequent Class-Specific (Top 5):\n")
            if cls_idx in frequent_analysis['by_class']:
                for i, comp in enumerate(frequent_analysis['by_class'][cls_idx][:25]):
                    actual_layer = comp['layer'] + layer_offset
                    token_name = "CLS" if comp['token'] == 0 else f"T{comp['token']}"
                    f.write(f"  {i+1}. L{actual_layer} H{comp['head']} {token_name} | ")
                    f.write(f"Freq: {comp['target_frequency']*100:.1f}% | ")
                    f.write(f"Others: {comp['mean_other_frequency']*100:.1f}% | ")
                    f.write(f"Advantage: +{comp['frequency_advantage']*100:.1f}% | ")
                    f.write(f"Mean: {comp['target_mean']:.3f}\n")
            else:
                f.write("  No frequent components found for this class.\n")

            f.write("\n")

    print(f"Frequency pattern comparison saved to {save_dir / 'frequency_pattern_comparison.txt'}")


def analyze_activation_distribution_per_class(
    direction_similarities: Dict, num_classes: int, percentiles: List[float] = [10, 25, 50, 75, 90, 95, 99]
) -> Dict:
    """
    Analyze the distribution of logit attribution values per class to understand activation patterns
    """

    class_distributions = {}

    for target_class in range(num_classes):
        # Collect all activations for this class
        class_activations = []

        for image_name, result in direction_similarities.items():
            if result['predicted_class'] == target_class:
                similarities = result['similarities']['all_mean']
                class_activations.extend(similarities.flatten())

        if class_activations:
            class_activations = np.array(class_activations)

            class_distributions[target_class] = {
                'n_samples': len(class_activations),
                'mean': np.mean(class_activations),
                'std': np.std(class_activations),
                'min': np.min(class_activations),
                'max': np.max(class_activations),
                'percentiles': {
                    p: np.percentile(class_activations, p)
                    for p in percentiles
                },
                'positive_ratio': np.sum(class_activations > 0) / len(class_activations),
                'high_activation_ratio': np.sum(class_activations > 0.5) / len(class_activations),
                'very_high_activation_ratio': np.sum(class_activations > 1.0) / len(class_activations)
            }

    return class_distributions


def extended_discriminative_analysis_report(
    direction_similarities: Dict, discriminative_analysis: Dict, frequent_analysis: Dict, distribution_analysis: Dict,
    num_layers_analyzed: int, save_dir: Path, num_classes: int
):
    """
    Generate comprehensive report combining all analyses
    """
    layer_offset = 12 - num_layers_analyzed

    with open(save_dir / "extended_discriminative_analysis.txt", 'w') as f:
        f.write("EXTENDED DISCRIMINATIVE COMPONENT ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        f.write("ACTIVATION DISTRIBUTION BY CLASS:\n")
        f.write("-" * 40 + "\n")
        for cls_idx, dist in distribution_analysis.items():
            f.write(f"Class {cls_idx}:\n")
            f.write(f"  Samples: {dist['n_samples']}\n")
            f.write(f"  Mean activation: {dist['mean']:.4f}\n")
            f.write(f"  Std: {dist['std']:.4f}\n")
            f.write(f"  Range: [{dist['min']:.4f}, {dist['max']:.4f}]\n")
            f.write(f"  Positive activations: {dist['positive_ratio']*100:.1f}%\n")
            f.write(f"  High activations (>0.5): {dist['high_activation_ratio']*100:.1f}%\n")
            f.write(f"  Very high activations (>1.0): {dist['very_high_activation_ratio']*100:.1f}%\n")
            f.write(
                f"  Key percentiles: 50th={dist['percentiles'][50]:.3f}, 90th={dist['percentiles'][90]:.3f}, 99th={dist['percentiles'][99]:.3f}\n\n"
            )

        f.write("FREQUENT VS RARE COMPONENT COMPARISON:\n")
        f.write("-" * 40 + "\n")

        for cls_idx in range(num_classes):
            if cls_idx not in discriminative_analysis['by_class'] or not discriminative_analysis['by_class'][cls_idx]:
                continue

            f.write(f"\nClass {cls_idx}:\n")

            # Original analysis - typically finds rare but highly specific components
            rare_comps = [c for c in discriminative_analysis['by_class'][cls_idx] if c['activation_frequency'] < 0.3]
            frequent_rare_comps = [
                c for c in discriminative_analysis['by_class'][cls_idx] if c['activation_frequency'] >= 0.3
            ]

            f.write(f"  Original analysis: {len(discriminative_analysis['by_class'][cls_idx])} components\n")
            f.write(f"    Rare (<30%): {len(rare_comps)}\n")
            f.write(f"    Frequent (≥30%): {len(frequent_rare_comps)}\n")

            # New analysis - focuses on frequently activating components
            if cls_idx in frequent_analysis['by_class']:
                freq_comps = frequent_analysis['by_class'][cls_idx]
                high_freq = [c for c in freq_comps if c['target_frequency'] >= 0.7]
                med_freq = [c for c in freq_comps if 0.3 <= c['target_frequency'] < 0.7]

                f.write(f"  Frequent analysis: {len(freq_comps)} components\n")
                f.write(f"    High frequency (≥70%): {len(high_freq)}\n")
                f.write(f"    Medium frequency (30-70%): {len(med_freq)}\n")

                if freq_comps:
                    f.write(
                        f"  Top frequent component: L{freq_comps[0]['layer'] + layer_offset} H{freq_comps[0]['head']} "
                    )
                    f.write(f"T{freq_comps[0]['token']} ({freq_comps[0]['target_frequency']*100:.1f}% freq, ")
                    f.write(f"{freq_comps[0]['frequency_advantage']*100:.1f}% advantage)\n")


from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def analyze_intra_class_variation(direction_similarities: Dict, num_classes: int, min_samples: int = 10) -> Dict:
    """
    Analyze variation patterns within each class to discover if classes have multiple strategies
    """

    class_variation_analysis = {}

    for class_idx in range(num_classes):
        # Collect all data for this class
        class_data = []
        image_names = []

        for image_name, result in direction_similarities.items():
            if result['predicted_class'] == class_idx:
                similarities = result['similarities']['all_mean']
                class_data.append(similarities.flatten())
                image_names.append(image_name)

        if len(class_data) < min_samples:
            continue

        class_data = np.array(class_data)

        # Calculate variation metrics
        mean_activation = np.mean(class_data, axis=0)
        std_activation = np.std(class_data, axis=0)
        cv_activation = std_activation / (np.abs(mean_activation) + 1e-8)  # Coefficient of variation

        # Measure overall class coherence
        class_coherence = 1.0 - np.mean(cv_activation)

        # Find components with high variation (potential multiple strategies)
        high_variation_indices = np.where(cv_activation > 1.5)[0]  # CV > 1.5 indicates high variation

        # Apply PCA to understand dimensionality
        scaler = StandardScaler()
        class_data_scaled = scaler.fit_transform(class_data)

        pca = PCA()
        pca.fit(class_data_scaled)

        # Calculate effective dimensionality (components needed for 80% variance)
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        effective_dims = np.argmax(cumsum_variance >= 0.8) + 1

        # Detect potential clustering using elbow method on inertia
        inertias = []
        max_clusters = min(8, len(class_data) // 3)

        if max_clusters >= 2:
            for n_clusters in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(class_data_scaled)
                inertias.append(kmeans.inertia_)

            # Simple elbow detection
            if len(inertias) >= 3:
                # Calculate second derivative to find elbow
                second_derivatives = np.diff(np.diff(inertias))
                optimal_clusters = np.argmax(second_derivatives) + 2  # +2 because of double diff
            else:
                optimal_clusters = 1
        else:
            optimal_clusters = 1
            inertias = [0]

        class_variation_analysis[class_idx] = {
            'n_samples': len(class_data),
            'class_coherence': class_coherence,
            'high_variation_components': len(high_variation_indices),
            'total_components': len(mean_activation),
            'effective_dimensionality': effective_dims,
            'total_variance_in_first_pc': pca.explained_variance_ratio_[0],
            'optimal_clusters': optimal_clusters,
            'clustering_inertias': inertias,
            'mean_cv': np.mean(cv_activation),
            'median_cv': np.median(cv_activation)
        }

    return class_variation_analysis


def analyze_component_consistency(direction_similarities: Dict, num_classes: int, min_samples: int = 5) -> Dict:
    """
    Analyze consistency/reliability of each component across images within each class
    """

    consistency_analysis = {}

    for class_idx in range(num_classes):
        # Collect activations per component for this class
        component_activations = defaultdict(list)

        for image_name, result in direction_similarities.items():
            if result['predicted_class'] == class_idx:
                similarities = result['similarities']['all_mean']

                for layer in range(similarities.shape[0]):
                    for head in range(similarities.shape[1]):
                        for token in range(similarities.shape[2]):
                            key = (layer, head, token)
                            component_activations[key].append(similarities[layer, head, token])

        if not component_activations:
            continue

        # Analyze consistency for each component
        component_stats = []

        for (layer, head, token), activations in component_activations.items():
            if len(activations) < min_samples:
                continue

            activations = np.array(activations)

            # Basic statistics
            mean_val = np.mean(activations)
            std_val = np.std(activations)
            cv = std_val / (abs(mean_val) + 1e-8)

            # Consistency metrics
            signal_to_noise = abs(mean_val) / (std_val + 1e-8)

            # Activation frequency (how often > threshold)
            activation_freq_low = np.mean(activations > 0.1)
            activation_freq_med = np.mean(activations > 0.3)
            activation_freq_high = np.mean(activations > 0.5)

            # Stability metric (inverse of coefficient of variation)
            stability = 1.0 / (cv + 1e-8)

            # Predictability (how "gaussian" is the distribution)
            _, p_value = stats.normaltest(activations)
            predictability = p_value  # Higher p-value = more gaussian = more predictable

            component_stats.append({
                'layer': layer,
                'head': head,
                'token': token,
                'mean': mean_val,
                'std': std_val,
                'cv': cv,
                'snr': signal_to_noise,
                'stability': stability,
                'predictability': predictability,
                'freq_low': activation_freq_low,
                'freq_med': activation_freq_med,
                'freq_high': activation_freq_high,
                'n_samples': len(activations),
                'range': np.max(activations) - np.min(activations)
            })

        # Sort by signal-to-noise ratio
        component_stats.sort(key=lambda x: x['snr'], reverse=True)

        consistency_analysis[class_idx] = {
            'components': component_stats,
            'summary': {
                'total_components': len(component_stats),
                'high_snr_components': len([c for c in component_stats if c['snr'] > 3.0]),
                'stable_components': len([c for c in component_stats if c['cv'] < 0.5]),
                'frequent_components': len([c for c in component_stats if c['freq_med'] > 0.7]),
                'mean_snr': np.mean([c['snr'] for c in component_stats]),
                'mean_stability': np.mean([c['stability'] for c in component_stats])
            }
        }

    return consistency_analysis


def discover_component_co_activation_patterns(
    direction_similarities: Dict,
    num_classes: int,
    activation_threshold: float = 0.3,
    min_co_occurrence: float = 0.6
) -> Dict:
    """
    Discover which components tend to activate together (co-activation patterns)
    """

    co_activation_analysis = {}

    for class_idx in range(num_classes):
        # Collect binary activation patterns for this class
        activation_patterns = []

        for image_name, result in direction_similarities.items():
            if result['predicted_class'] == class_idx:
                similarities = result['similarities']['all_mean']
                # Convert to binary activation pattern
                binary_pattern = (similarities > activation_threshold).astype(int)
                activation_patterns.append(binary_pattern.flatten())

        if len(activation_patterns) < 5:
            continue

        activation_patterns = np.array(activation_patterns)  # [images, flattened_components]

        # Find frequently co-activating component pairs
        n_components = activation_patterns.shape[1]
        co_activation_pairs = []

        # Sample pairs to avoid O(n²) complexity
        max_pairs_to_check = 10000
        total_possible_pairs = n_components * (n_components - 1) // 2

        if total_possible_pairs > max_pairs_to_check:
            # Sample random pairs
            component_indices = np.arange(n_components)
            sampled_pairs = np.random.choice(component_indices, size=(max_pairs_to_check, 2), replace=True)
            pairs_to_check = [(i, j) for i, j in sampled_pairs if i != j]
        else:
            # Check all pairs
            pairs_to_check = [(i, j) for i in range(n_components) for j in range(i + 1, n_components)]

        for i, j in pairs_to_check:
            comp_i_active = activation_patterns[:, i]
            comp_j_active = activation_patterns[:, j]

            # Calculate co-activation metrics
            both_active = comp_i_active & comp_j_active
            either_active = comp_i_active | comp_j_active

            if np.sum(either_active) == 0:
                continue

            co_activation_rate = np.sum(both_active) / np.sum(either_active)  # Jaccard index
            joint_frequency = np.sum(both_active) / len(activation_patterns)

            if co_activation_rate > min_co_occurrence and joint_frequency > 0.3:
                # Convert flat indices back to (layer, head, token)
                layers, heads, tokens = direction_similarities[list(direction_similarities.keys()
                                                                    )[0]]['similarities']['all_mean'].shape

                layer_i = i // (heads * tokens)
                head_i = (i % (heads * tokens)) // tokens
                token_i = i % tokens

                layer_j = j // (heads * tokens)
                head_j = (j % (heads * tokens)) // tokens
                token_j = j % tokens

                co_activation_pairs.append({
                    'comp1': (layer_i, head_i, token_i),
                    'comp2': (layer_j, head_j, token_j),
                    'co_activation_rate': co_activation_rate,
                    'joint_frequency': joint_frequency,
                    'individual_freq_1': np.mean(comp_i_active),
                    'individual_freq_2': np.mean(comp_j_active)
                })

        # Sort by co-activation rate
        co_activation_pairs.sort(key=lambda x: x['co_activation_rate'], reverse=True)

        co_activation_analysis[class_idx] = {
            'co_activation_pairs': co_activation_pairs[:20],  # Top 20 pairs
            'summary': {
                'pairs_found':
                len(co_activation_pairs),
                'strong_pairs':
                len([p for p in co_activation_pairs if p['co_activation_rate'] > 0.8]),
                'mean_co_activation':
                np.mean([p['co_activation_rate'] for p in co_activation_pairs]) if co_activation_pairs else 0
            }
        }

    return co_activation_analysis


def analyze_decision_boundary_components(
    direction_similarities: Dict, discriminative_analysis: Dict, num_classes: int
) -> Dict:
    """
    Analyze which components are most important for class boundaries vs within-class variation
    """

    boundary_analysis = {}

    for class_idx in range(num_classes):
        if class_idx not in discriminative_analysis['by_class']:
            continue

        discriminative_components = discriminative_analysis['by_class'][class_idx]

        if not discriminative_components:
            continue

        # Collect target class activations
        target_activations = []
        other_activations = []

        for image_name, result in direction_similarities.items():
            similarities = result['similarities']['all_mean']

            if result['predicted_class'] == class_idx:
                target_activations.append(similarities)
            else:
                other_activations.append(similarities)

        if not target_activations or not other_activations:
            continue

        target_activations = np.array(target_activations)
        other_activations = np.array(other_activations)

        # Analyze top discriminative components
        boundary_components = []

        for comp in discriminative_components[:20]:  # Top 20 discriminative
            layer, head, token = comp['layer'], comp['head'], comp['token']

            # Get activations for this component
            target_comp_acts = target_activations[:, layer, head, token]
            other_comp_acts = other_activations[:, layer, head, token]

            # Calculate separation metrics
            target_mean = np.mean(target_comp_acts)
            other_mean = np.mean(other_comp_acts)
            target_std = np.std(target_comp_acts)
            other_std = np.std(other_comp_acts)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((target_std**2 + other_std**2) / 2)
            cohens_d = abs(target_mean - other_mean) / (pooled_std + 1e-8)

            # Statistical significance
            _, p_value = stats.ttest_ind(target_comp_acts, other_comp_acts)

            # Overlap measure
            if target_std > 0 and other_std > 0:
                # Calculate overlap coefficient
                overlap = min(target_mean + target_std,
                              other_mean + other_std) - max(target_mean - target_std, other_mean - other_std)
                total_range = max(target_mean + target_std,
                                  other_mean + other_std) - min(target_mean - target_std, other_mean - other_std)
                overlap_coefficient = overlap / (total_range + 1e-8)
            else:
                overlap_coefficient = 0

            boundary_components.append({
                'layer': layer,
                'head': head,
                'token': token,
                'cohens_d': cohens_d,
                'p_value': p_value,
                'overlap_coefficient': overlap_coefficient,
                'target_mean': target_mean,
                'other_mean': other_mean,
                'separation': abs(target_mean - other_mean),
                'discriminative_score': comp.get('discriminative_score', 0)
            })

        # Sort by Cohen's d (effect size)
        boundary_components.sort(key=lambda x: x['cohens_d'], reverse=True)

        boundary_analysis[class_idx] = {
            'boundary_components': boundary_components,
            'summary': {
                'strong_separators': len([c for c in boundary_components if c['cohens_d'] > 0.8]),
                'significant_separators': len([c for c in boundary_components if c['p_value'] < 0.01]),
                'low_overlap_components': len([c for c in boundary_components if c['overlap_coefficient'] < 0.2]),
                'mean_effect_size': np.mean([c['cohens_d'] for c in boundary_components]),
                'mean_separation': np.mean([c['separation'] for c in boundary_components])
            }
        }

    return boundary_analysis


def generate_comprehensive_pattern_report(
    direction_similarities: Dict, variation_analysis: Dict, consistency_analysis: Dict, co_activation_analysis: Dict,
    boundary_analysis: Dict, num_layers_analyzed: int, save_dir: Path
):
    """
    Generate comprehensive analytical report of all discovered patterns
    """
    layer_offset = 12 - num_layers_analyzed

    with open(save_dir / "comprehensive_pattern_analysis.txt", 'w') as f:
        f.write("COMPREHENSIVE PATTERN ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Summary statistics
        f.write("EXECUTIVE SUMMARY:\n")
        f.write("-" * 20 + "\n")

        classes_analyzed = set(variation_analysis.keys()) | set(consistency_analysis.keys())
        f.write(f"Classes analyzed: {sorted(classes_analyzed)}\n")

        # Variation analysis summary
        if variation_analysis:
            f.write(f"\nCLASS VARIATION PATTERNS:\n")
            for cls_idx, analysis in variation_analysis.items():
                f.write(f"  Class {cls_idx}:\n")
                f.write(f"    Coherence: {analysis['class_coherence']:.3f} (1.0 = perfectly coherent)\n")
                f.write(f"    Effective dimensionality: {analysis['effective_dimensionality']} components\n")
                f.write(f"    Optimal clusters: {analysis['optimal_clusters']}\n")
                f.write(
                    f"    High-variation components: {analysis['high_variation_components']}/{analysis['total_components']}\n"
                )

                # Interpretation
                if analysis['class_coherence'] > 0.8:
                    f.write(f"    → INTERPRETATION: Highly coherent class (single strategy)\n")
                elif analysis['class_coherence'] > 0.6:
                    f.write(f"    → INTERPRETATION: Moderately coherent (minor sub-strategies)\n")
                else:
                    f.write(f"    → INTERPRETATION: Low coherence (multiple distinct strategies)\n")
                f.write("\n")

        # Consistency analysis
        f.write(f"\nCOMPONENT CONSISTENCY ANALYSIS:\n")
        f.write("-" * 35 + "\n")

        for cls_idx in sorted(consistency_analysis.keys()):
            analysis = consistency_analysis[cls_idx]
            summary = analysis['summary']

            f.write(f"\nClass {cls_idx} Component Reliability:\n")
            f.write(f"  Total components analyzed: {summary['total_components']}\n")
            total_components = summary['total_components']
            if total_components > 0:
                f.write(
                    f"  High SNR components (>3.0): {summary['high_snr_components']} ({summary['high_snr_components']/total_components*100:.1f}%)\n"
                )
                f.write(
                    f"  Stable components (CV<0.5): {summary['stable_components']} ({summary['stable_components']/total_components*100:.1f}%)\n"
                )
                f.write(
                    f"  Frequently active (>70%): {summary['frequent_components']} ({summary['frequent_components']/total_components*100:.1f}%)\n"
                )
            else:
                f.write(f"  High SNR components (>3.0): {summary['high_snr_components']} (0.0%)\n")
                f.write(f"  Stable components (CV<0.5): {summary['stable_components']} (0.0%)\n")
                f.write(f"  Frequently active (>70%): {summary['frequent_components']} (0.0%)\n")
            f.write(f"  Mean Signal-to-Noise: {summary['mean_snr']:.2f}\n")

            # Top reliable components
            f.write(f"  Top 5 most reliable components:\n")
            for i, comp in enumerate(analysis['components'][:5]):
                actual_layer = comp['layer'] + layer_offset
                token_name = "CLS" if comp['token'] == 0 else f"T{comp['token']}"
                f.write(f"    {i+1}. L{actual_layer} H{comp['head']} {token_name} | ")
                f.write(f"SNR: {comp['snr']:.2f} | Freq: {comp['freq_med']*100:.1f}% | CV: {comp['cv']:.2f}\n")

        # Co-activation patterns
        f.write(f"\nCO-ACTIVATION PATTERNS:\n")
        f.write("-" * 25 + "\n")

        for cls_idx in sorted(co_activation_analysis.keys()):
            analysis = co_activation_analysis[cls_idx]
            summary = analysis['summary']

            f.write(f"\nClass {cls_idx}:\n")
            f.write(f"  Co-activation pairs found: {summary['pairs_found']}\n")
            f.write(f"  Strong pairs (>80% co-activation): {summary['strong_pairs']}\n")

            if analysis['co_activation_pairs']:
                f.write(f"  Top 3 co-activation patterns:\n")
                for i, pair in enumerate(analysis['co_activation_pairs'][:3]):
                    l1, h1, t1 = pair['comp1']
                    l2, h2, t2 = pair['comp2']
                    actual_l1 = l1 + layer_offset
                    actual_l2 = l2 + layer_offset
                    t1_name = "CLS" if t1 == 0 else f"T{t1}"
                    t2_name = "CLS" if t2 == 0 else f"T{t2}"

                    f.write(f"    {i+1}. L{actual_l1}H{h1}{t1_name} + L{actual_l2}H{h2}{t2_name} | ")
                    f.write(f"Co-activation: {pair['co_activation_rate']*100:.1f}% | ")
                    f.write(f"Joint freq: {pair['joint_frequency']*100:.1f}%\n")

        # Decision boundary analysis
        f.write(f"\nDECISION BOUNDARY ANALYSIS:\n")
        f.write("-" * 30 + "\n")

        for cls_idx in sorted(boundary_analysis.keys()):
            analysis = boundary_analysis[cls_idx]
            summary = analysis['summary']

            f.write(f"\nClass {cls_idx}:\n")
            f.write(f"  Strong separators (Cohen's d > 0.8): {summary['strong_separators']}\n")
            f.write(f"  Statistically significant (p < 0.01): {summary['significant_separators']}\n")
            f.write(f"  Low overlap components (<20%): {summary['low_overlap_components']}\n")
            f.write(f"  Mean effect size: {summary['mean_effect_size']:.2f}\n")

            f.write(f"  Top 3 boundary-defining components:\n")
            for i, comp in enumerate(analysis['boundary_components'][:3]):
                actual_layer = comp['layer'] + layer_offset
                token_name = "CLS" if comp['token'] == 0 else f"T{comp['token']}"
                f.write(f"    {i+1}. L{actual_layer} H{comp['head']} {token_name} | ")
                f.write(f"Effect size: {comp['cohens_d']:.2f} | ")
                f.write(f"Overlap: {comp['overlap_coefficient']*100:.1f}% | ")
                f.write(f"p-value: {comp['p_value']:.2e}\n")

        # Strategic insights
        f.write(f"\n\nSTRATEGIC INSIGHTS:\n")
        f.write("-" * 20 + "\n")

        f.write("Classes with multiple strategies (low coherence):\n")
        multi_strategy_classes = [
            cls for cls, analysis in variation_analysis.items() if analysis['class_coherence'] < 0.6
        ]
        if multi_strategy_classes:
            for cls in multi_strategy_classes:
                f.write(f"  Class {cls}: {variation_analysis[cls]['optimal_clusters']} strategies detected\n")
        else:
            f.write("  None detected - all classes show coherent strategies\n")

        f.write("\nClasses with highly reliable components:\n")
        reliable_classes = [
            cls for cls, analysis in consistency_analysis.items() if analysis['summary']['mean_snr'] > 2.0
        ]
        if reliable_classes:
            for cls in reliable_classes:
                snr = consistency_analysis[cls]['summary']['mean_snr']
                f.write(f"  Class {cls}: Mean SNR = {snr:.2f}\n")
        else:
            f.write("  No classes with consistently high SNR components\n")

        f.write("\nClasses with strong co-activation patterns:\n")
        cooperative_classes = [
            cls for cls, analysis in co_activation_analysis.items() if analysis['summary']['strong_pairs'] > 0
        ]
        if cooperative_classes:
            for cls in cooperative_classes:
                pairs = co_activation_analysis[cls]['summary']['strong_pairs']
                f.write(f"  Class {cls}: {pairs} strong co-activation pairs\n")
        else:
            f.write("  No strong co-activation patterns detected\n")

    print(f"Comprehensive pattern analysis saved to {save_dir / 'comprehensive_pattern_analysis.txt'}")


# Integration function for your existing head_analysis.py
def run_comprehensive_pattern_analysis(
    direction_similarities: Dict, discriminative_analysis: Dict, num_classes: int, num_layers_analyzed: int,
    save_dir: Path
):
    """
    Run all pattern analyses and generate comprehensive report
    """

    print("Step 8: Analyzing intra-class variation patterns...")
    variation_analysis = analyze_intra_class_variation(direction_similarities, num_classes)

    print("Step 9: Analyzing component consistency...")
    consistency_analysis = analyze_component_consistency(direction_similarities, num_classes)

    print("Step 10: Discovering co-activation patterns...")
    co_activation_analysis = discover_component_co_activation_patterns(direction_similarities, num_classes)

    print("Step 11: Analyzing decision boundary components...")
    boundary_analysis = analyze_decision_boundary_components(
        direction_similarities, discriminative_analysis, num_classes
    )

    print("Step 12: Generating comprehensive pattern report...")
    generate_comprehensive_pattern_report(
        direction_similarities, variation_analysis, consistency_analysis, co_activation_analysis, boundary_analysis,
        num_layers_analyzed, save_dir
    )

    return variation_analysis, consistency_analysis, co_activation_analysis, boundary_analysis


def analyze_class_strategies(
    direction_similarities: Dict,
    target_class: int,
    max_clusters: int = 5,
    min_cluster_size: int = 5,
    activation_threshold: float = 0.3
) -> Dict:
    """
    Discover and analyze different strategies within a single class
    """

    # Collect all data for this class
    class_data = []
    image_names = []

    for image_name, result in direction_similarities.items():
        if result['predicted_class'] == target_class:
            similarities = result['similarities']['all_mean']
            class_data.append(similarities.flatten())
            image_names.append(image_name)

    if len(class_data) < min_cluster_size * 2:
        return None

    class_data = np.array(class_data)

    # Standardize the data
    scaler = StandardScaler()
    class_data_scaled = scaler.fit_transform(class_data)

    # Apply PCA for dimensionality reduction and visualization
    pca = PCA(n_components=min(50, len(class_data) - 1))
    class_data_pca = pca.fit_transform(class_data_scaled)

    # Find optimal number of clusters using silhouette score
    max_possible_clusters = min(max_clusters, len(class_data) // min_cluster_size)

    if max_possible_clusters < 2:
        return None

    silhouette_scores = []
    cluster_results = {}

    for n_clusters in range(2, max_possible_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_data_pca)

        # Check if all clusters have minimum size
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
        if min(cluster_sizes) < min_cluster_size:
            continue

        silhouette_avg = silhouette_score(class_data_pca, cluster_labels)
        silhouette_scores.append((n_clusters, silhouette_avg))
        cluster_results[n_clusters] = (kmeans, cluster_labels, silhouette_avg)

    if not silhouette_scores:
        return None

    # Choose optimal number of clusters (highest silhouette score)
    optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    kmeans, cluster_labels, best_silhouette = cluster_results[optimal_n_clusters]

    # Analyze each strategy
    strategies = {}
    layers, heads, tokens = direction_similarities[image_names[0]]['similarities']['all_mean'].shape

    for strategy_id in range(optimal_n_clusters):
        strategy_mask = cluster_labels == strategy_id
        strategy_size = np.sum(strategy_mask)

        if strategy_size < min_cluster_size:
            continue

        # Get images and data for this strategy
        strategy_images = [image_names[i] for i in range(len(image_names)) if strategy_mask[i]]
        strategy_data = class_data[strategy_mask]

        # Calculate strategy signature (mean activation pattern)
        strategy_mean = np.mean(strategy_data, axis=0)
        strategy_std = np.std(strategy_data, axis=0)

        # Reshape back to [layers, heads, tokens]
        strategy_mean_reshaped = strategy_mean.reshape(layers, heads, tokens)
        strategy_std_reshaped = strategy_std.reshape(layers, heads, tokens)

        # Find defining components for this strategy
        defining_components = []

        for layer in range(layers):
            for head in range(heads):
                for token in range(tokens):
                    activation = strategy_mean_reshaped[layer, head, token]
                    consistency = 1.0 - (strategy_std_reshaped[layer, head, token] / (abs(activation) + 1e-8))

                    # Component is defining if it's strong and consistent
                    if abs(activation) > activation_threshold and consistency > 0.7:
                        defining_components.append({
                            'layer': layer,
                            'head': head,
                            'token': token,
                            'activation': activation,
                            'consistency': consistency,
                            'importance': abs(activation) * consistency
                        })

        # Sort by importance
        defining_components.sort(key=lambda x: x['importance'], reverse=True)

        # Calculate distinctiveness from other strategies
        distinctiveness_scores = []
        other_strategy_means = []

        for other_id in range(optimal_n_clusters):
            if other_id != strategy_id:
                other_mask = cluster_labels == other_id
                if np.sum(other_mask) >= min_cluster_size:
                    other_data = class_data[other_mask]
                    other_mean = np.mean(other_data, axis=0)
                    other_strategy_means.append(other_mean)

        if other_strategy_means:
            # Calculate how different this strategy is from others
            for other_mean in other_strategy_means:
                difference = np.linalg.norm(strategy_mean - other_mean)
                distinctiveness_scores.append(difference)

            avg_distinctiveness = np.mean(distinctiveness_scores)
        else:
            avg_distinctiveness = 0

        # Find strategy's preferred components (compared to class average)
        class_mean = np.mean(class_data, axis=0)
        strategy_preferences = strategy_mean - class_mean
        strategy_preferences_reshaped = strategy_preferences.reshape(layers, heads, tokens)

        preferred_components = []
        for layer in range(layers):
            for head in range(heads):
                for token in range(tokens):
                    preference = strategy_preferences_reshaped[layer, head, token]
                    if abs(preference) > 0.2:  # Significant preference
                        preferred_components.append({
                            'layer': layer,
                            'head': head,
                            'token': token,
                            'preference': preference,
                            'direction': 'higher' if preference > 0 else 'lower'
                        })

        preferred_components.sort(key=lambda x: abs(x['preference']), reverse=True)

        strategies[strategy_id] = {
            'id': strategy_id,
            'size': strategy_size,
            'percentage': strategy_size / len(class_data),
            'images': strategy_images,
            'defining_components': defining_components[:15],  # Top 15
            'preferred_components': preferred_components[:15],  # Top 15 preferences
            'distinctiveness': avg_distinctiveness,
            'silhouette_score': best_silhouette,
            'strategy_mean': strategy_mean_reshaped,
            'strategy_std': strategy_std_reshaped
        }

    return {
        'class': target_class,
        'n_strategies': optimal_n_clusters,
        'strategies': strategies,
        'silhouette_score': best_silhouette,
        'pca_explained_variance': pca.explained_variance_ratio_[:10].sum(),
        'all_silhouette_scores': silhouette_scores,
        'total_images': len(class_data)
    }


def compare_strategies_within_class(class_strategy_analysis: Dict, num_layers_analyzed: int) -> Dict:
    """
    Compare strategies within a class to understand their differences
    """

    if not class_strategy_analysis or 'strategies' not in class_strategy_analysis:
        return None

    strategies = class_strategy_analysis['strategies']
    layer_offset = 12 - num_layers_analyzed

    comparison = {
        'class': class_strategy_analysis['class'],
        'strategy_comparison': [],
        'shared_components': [],
        'unique_components': {},
        'strategy_specializations': {}
    }

    # Compare each pair of strategies
    strategy_ids = list(strategies.keys())

    for i, strategy_1_id in enumerate(strategy_ids):
        for j, strategy_2_id in enumerate(strategy_ids[i + 1:], i + 1):
            strategy_1 = strategies[strategy_1_id]
            strategy_2 = strategies[strategy_2_id]

            # Find overlapping defining components
            s1_components = {(c['layer'], c['head'], c['token']) for c in strategy_1['defining_components']}
            s2_components = {(c['layer'], c['head'], c['token']) for c in strategy_2['defining_components']}

            overlap = s1_components & s2_components
            s1_unique = s1_components - s2_components
            s2_unique = s2_components - s1_components

            # Calculate strategy similarity
            s1_mean = strategy_1['strategy_mean'].flatten()
            s2_mean = strategy_2['strategy_mean'].flatten()
            correlation = np.corrcoef(s1_mean, s2_mean)[0, 1]

            comparison['strategy_comparison'].append({
                'strategy_1':
                strategy_1_id,
                'strategy_2':
                strategy_2_id,
                'shared_components':
                len(overlap),
                'strategy_1_unique':
                len(s1_unique),
                'strategy_2_unique':
                len(s2_unique),
                'correlation':
                correlation,
                'distinctiveness':
                abs(strategy_1['distinctiveness'] - strategy_2['distinctiveness'])
            })

    # Find components that are important across multiple strategies
    all_component_importance = defaultdict(list)

    for strategy_id, strategy in strategies.items():
        for comp in strategy['defining_components']:
            key = (comp['layer'], comp['head'], comp['token'])
            all_component_importance[key].append({
                'strategy': strategy_id,
                'importance': comp['importance'],
                'activation': comp['activation']
            })

    # Identify shared vs unique components
    for component, importances in all_component_importance.items():
        if len(importances) > 1:  # Shared component
            comparison['shared_components'].append({
                'component': component,
                'strategies': [imp['strategy'] for imp in importances],
                'avg_importance': np.mean([imp['importance'] for imp in importances])
            })
        else:  # Unique component
            strategy_id = importances[0]['strategy']
            if strategy_id not in comparison['unique_components']:
                comparison['unique_components'][strategy_id] = []
            comparison['unique_components'][strategy_id].append({
                'component': component,
                'importance': importances[0]['importance']
            })

    # Identify strategy specializations (head/layer preferences)
    for strategy_id, strategy in strategies.items():
        head_usage = defaultdict(int)
        layer_usage = defaultdict(int)

        for comp in strategy['defining_components'][:10]:  # Top 10 components
            head_key = (comp['layer'], comp['head'])
            layer_key = comp['layer']

            head_usage[head_key] += comp['importance']
            layer_usage[layer_key] += comp['importance']

        # Find most used heads and layers
        top_heads = sorted(head_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        top_layers = sorted(layer_usage.items(), key=lambda x: x[1], reverse=True)[:3]

        comparison['strategy_specializations'][strategy_id] = {
            'preferred_heads':
            [(f"L{layer + layer_offset}H{head}", importance) for (layer, head), importance in top_heads],
            'preferred_layers': [(f"L{layer + layer_offset}", importance) for layer, importance in top_layers]
        }

    return comparison


def generate_strategy_report(
    multi_strategy_classes: List[int], direction_similarities: Dict, num_layers_analyzed: int, save_dir: Path
):
    """
    Generate comprehensive report on strategies for classes with multiple approaches
    """

    layer_offset = 12 - num_layers_analyzed
    all_strategy_analyses = {}

    with open(save_dir / "multi_strategy_analysis.txt", 'w') as f:
        f.write("MULTI-STRATEGY CLASS ANALYSIS\n")
        f.write("=" * 50 + "\n\n")

        for class_idx in multi_strategy_classes:
            print(f"Analyzing strategies for Class {class_idx}...")

            # Analyze strategies for this class
            strategy_analysis = analyze_class_strategies(
                direction_similarities, class_idx, max_clusters=6, min_cluster_size=8
            )

            if not strategy_analysis:
                f.write(f"CLASS {class_idx}: Insufficient data for strategy analysis\n\n")
                continue

            all_strategy_analyses[class_idx] = strategy_analysis

            f.write(f"CLASS {class_idx} STRATEGY ANALYSIS:\n")
            f.write("-" * 35 + "\n")
            f.write(f"Total images: {strategy_analysis['total_images']}\n")
            f.write(f"Number of strategies detected: {strategy_analysis['n_strategies']}\n")
            f.write(f"Strategy separation quality (silhouette): {strategy_analysis['silhouette_score']:.3f}\n")
            f.write(f"PCA variance explained: {strategy_analysis['pca_explained_variance']:.3f}\n\n")

            # Analyze each strategy
            for strategy_id, strategy in strategy_analysis['strategies'].items():
                f.write(f"  STRATEGY {strategy_id + 1}:\n")
                f.write(f"    Size: {strategy['size']} images ({strategy['percentage']*100:.1f}% of class)\n")
                f.write(f"    Distinctiveness: {strategy['distinctiveness']:.3f}\n")

                f.write(f"    Top defining components:\n")
                for i, comp in enumerate(strategy['defining_components'][:8]):
                    actual_layer = comp['layer'] + layer_offset
                    token_name = "CLS" if comp['token'] == 0 else f"T{comp['token']}"
                    f.write(f"      {i+1}. L{actual_layer} H{comp['head']} {token_name} | ")
                    f.write(f"Activation: {comp['activation']:+.3f} | ")
                    f.write(f"Consistency: {comp['consistency']:.3f}\n")

                f.write(f"    Strategy preferences (vs class average):\n")
                for i, pref in enumerate(strategy['preferred_components'][:5]):
                    actual_layer = pref['layer'] + layer_offset
                    token_name = "CLS" if pref['token'] == 0 else f"T{pref['token']}"
                    f.write(f"      {i+1}. L{actual_layer} H{pref['head']} {token_name} | ")
                    f.write(f"Prefers {pref['direction']} by {abs(pref['preference']):.3f}\n")

                f.write(f"    Example images: {strategy['images'][:5]}\n\n")

            # Compare strategies within class
            print(f"Comparing strategies within Class {class_idx}...")
            comparison = compare_strategies_within_class(strategy_analysis, num_layers_analyzed)

            if comparison:
                f.write(f"  STRATEGY COMPARISON:\n")

                # Strategy similarities
                f.write(f"    Strategy pair correlations:\n")
                for comp in comparison['strategy_comparison']:
                    f.write(f"      Strategy {comp['strategy_1']+1} vs {comp['strategy_2']+1}: ")
                    f.write(f"correlation = {comp['correlation']:.3f}, ")
                    f.write(f"shared components = {comp['shared_components']}\n")

                # Shared components
                if comparison['shared_components']:
                    f.write(f"    Components used by multiple strategies:\n")
                    for i, shared in enumerate(comparison['shared_components'][:5]):
                        layer, head, token = shared['component']
                        actual_layer = layer + layer_offset
                        token_name = "CLS" if token == 0 else f"T{token}"
                        f.write(f"      {i+1}. L{actual_layer} H{head} {token_name} | ")
                        f.write(f"Used by strategies: {[s+1 for s in shared['strategies']]}\n")

                # Strategy specializations
                f.write(f"    Strategy specializations:\n")
                for strategy_id, spec in comparison['strategy_specializations'].items():
                    f.write(f"      Strategy {strategy_id+1}:\n")
                    f.write(
                        f"        Preferred heads: {[f'{head} ({imp:.2f})' for head, imp in spec['preferred_heads']]}\n"
                    )
                    f.write(
                        f"        Preferred layers: {[f'{layer} ({imp:.2f})' for layer, imp in spec['preferred_layers']]}\n"
                    )

            f.write("\n" + "=" * 50 + "\n\n")

        # Cross-class strategy summary
        f.write("CROSS-CLASS STRATEGY SUMMARY:\n")
        f.write("-" * 30 + "\n")

        for class_idx, analysis in all_strategy_analyses.items():
            f.write(f"Class {class_idx}: {analysis['n_strategies']} strategies, ")
            f.write(f"separation quality = {analysis['silhouette_score']:.3f}\n")

            # Show strategy size distribution
            strategy_sizes = [s['percentage'] for s in analysis['strategies'].values()]
            strategy_sizes.sort(reverse=True)
            f.write(f"  Strategy distribution: {[f'{size*100:.1f}%' for size in strategy_sizes]}\n")

        f.write(f"\nInterpretation:\n")
        f.write(f"- Silhouette score > 0.5: Well-separated strategies\n")
        f.write(f"- Silhouette score 0.2-0.5: Moderately distinct strategies\n")
        f.write(f"- Silhouette score < 0.2: Weakly separated strategies\n")

    print(f"Multi-strategy analysis saved to {save_dir / 'multi_strategy_analysis.txt'}")
    return all_strategy_analyses


# Integration function for head_analysis.py
def run_strategy_analysis_for_multi_strategy_classes(
    direction_similarities: Dict,
    variation_analysis: Dict,
    num_layers_analyzed: int,
    save_dir: Path,
    coherence_threshold: float = 0.6
):
    """
    Run detailed strategy analysis for classes identified as having multiple strategies
    """

    # Identify classes with multiple strategies (low coherence)
    multi_strategy_classes = []

    for class_idx, analysis in variation_analysis.items():
        if analysis['class_coherence'] < coherence_threshold:
            multi_strategy_classes.append(class_idx)

    if not multi_strategy_classes:
        print("No classes with multiple strategies detected.")
        return None

    print(f"Detected classes with multiple strategies: {multi_strategy_classes}")
    print("Running detailed strategy analysis...")

    # Generate comprehensive strategy analysis
    strategy_analyses = generate_strategy_report(
        multi_strategy_classes, direction_similarities, num_layers_analyzed, save_dir
    )

    return strategy_analyses


def run_head_analysis(
    original_results: List[ClassificationResult],
    model: VisionTransformer,
    config: PipelineConfig,
    num_layers_to_analyze: int = 4,
    importance_threshold: float = 0.10,
    cls_token_only: bool = True,
    analyze_token_patterns: bool = True,
    pca_components: int = 35,
    n_clusters: int = 10
):
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
        original_results, model, config, num_layers_to_analyze, save_dir
    )

    # Get number of classes from the model
    num_classes = model.head.weight.shape[0]
    num_heads = 12

    print("Step 2: Analyzing class-specific head importance...")
    head_importance = analyze_class_specific_head_importance(
        direction_similarities, num_classes, num_heads, num_layers_to_analyze, importance_threshold, cls_token_only,
        save_dir, config
    )

    token_patterns = {}
    if analyze_token_patterns:
        print("Step 3: Analyzing token activation patterns...")
        # Check which classes have images
        classes_with_images = {
            cls_idx: count
            for cls_idx, count in head_importance['class_image_counts'].items() if count > 0
        }

        for cls_idx, count in classes_with_images.items():
            if count >= n_clusters:  # Only run if enough images for clustering
                print(f"  Analyzing patterns for class {cls_idx} ({count} images)...")
                patterns = analyze_token_activation_patterns(
                    direction_similarities, cls_idx, pca_components=pca_components, n_clusters=min(n_clusters, count)
                )

                if patterns:
                    token_patterns[cls_idx] = patterns

                    # Save patterns to file
                    output_path = save_dir / f"token_patterns_class_{cls_idx}{config.file.output_suffix}.npy"
                    np.save(output_path, patterns)

                    token_boost_config = generate_and_print_token_config(
                        token_patterns=token_patterns,
                        layer_offset=8,
                        activation_threshold=0.13,  # Our recommended gatekeeper threshold
                        top_k_tokens=5,  # Our recommended safety cap
                        min_cluster_size=25  # Ignore clusters with fewer than 25 images
                    )

                    # Create human-readable summary
                    with open(
                        save_dir / f"token_patterns_summary_class_{cls_idx}{config.file.output_suffix}.txt", 'w'
                    ) as f:
                        f.write(f"TOKEN ACTIVATION PATTERNS FOR CLASS {cls_idx}\n")
                        f.write("=======================================\n\n")
                        f.write(f"Total images analyzed: {patterns['n_images']}\n")
                        f.write(
                            f"PCA explained variance: {sum(patterns['explained_variance'][:pca_components]):.4f}\n\n"
                        )

                        f.write("PATTERN CLUSTERS:\n")
                        for i, cluster in enumerate(patterns['clusters']):
                            f.write(f"\nCluster {i+1}:\n")
                            f.write(f"  Size: {cluster['size']} images ({cluster['percentage']*100:.1f}%)\n")
                            f.write(f"  Example images: {', '.join(cluster['image_examples'])}\n")
                            f.write("  Top token activations (Layer, Head, Token, Value):\n")

                            # Correct layer numbering - add offset to show actual layer numbers
                            # (if analyzing last 5 layers of a 12-layer model)
                            layer_offset = 12 - num_layers_to_analyze

                            for l, h, t, v in sorted(cluster['top_activations'], key=lambda x: x[3], reverse=True)[:25]:
                                actual_layer = l + layer_offset
                                f.write(f"    Layer {actual_layer}, Head {h}, Token {t}: {v:.5f}\n")

                        f.write("\n\n")
            else:
                print(f"  Skipping class {cls_idx}: not enough images for clustering ({count})")

    print("Step 4: Analyze and summarize important components")
    analyze_and_summarize_important_components(direction_similarities, num_classes=10, top_k=8)

    # Add discriminative analysis
    print("Step 5: Analyzing discriminative components...")
    discriminative_analysis = analyze_discriminative_components(
        direction_similarities, num_classes=model.head.weight.shape[0]
    )

    print("Step 6: Finding sparse high-impact tokens...")
    sparse_analysis = find_sparse_high_impact_tokens(direction_similarities)

    print("Step 7: Generating comprehensive report...")
    save_dir = config.file.output_dir / "head_analysis"
    generate_class_specific_token_report(direction_similarities, discriminative_analysis, sparse_analysis, 4, save_dir)
    print("Head analysis complete!")
    print(f"Results saved to {save_dir}")

    variation_analysis, consistency_analysis, co_activation_analysis, boundary_analysis = run_comprehensive_pattern_analysis(
        direction_similarities, discriminative_analysis, num_classes, num_layers_to_analyze, save_dir
    )

    print("Step 8: Analyzing frequent class-specific components...")
    frequent_analysis = analyze_frequent_class_specific_components(direction_similarities, model.head.weight.shape[0])

    print("Step 9: Analyzing activation distributions...")
    distribution_analysis = analyze_activation_distribution_per_class(
        direction_similarities, model.head.weight.shape[0]
    )

    print("Step 10: Comparing frequency patterns...")
    compare_frequency_patterns(
        direction_similarities, discriminative_analysis, frequent_analysis, num_layers_to_analyze, save_dir
    )

    print("Step 11: Generating extended analysis report...")
    extended_discriminative_analysis_report(
        direction_similarities, discriminative_analysis, frequent_analysis, distribution_analysis,
        num_layers_to_analyze, save_dir, model.head.weight.shape[0]
    )

    # After your run_comprehensive_pattern_analysis call, add:
    print("Step 13: Analyzing strategies for multi-strategy classes...")
    strategy_analyses = run_strategy_analysis_for_multi_strategy_classes(
        direction_similarities, variation_analysis, num_layers_to_analyze, save_dir
    )

    return direction_similarities, head_importance, token_patterns
