# vision_sae_feature_selection.py
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from prisma_transmm import load_models
from vit.preprocessing import get_processor_for_precached_224_images


class VisionSAEFeatureSelector:
    """
    Feature selection for SAE-based steering in vision models
    Adapted specifically for vision transformers
    """

    def __init__(self, sae, model, n_classes=6, device='cuda'):
        self.sae = sae
        self.model = model
        self.n_classes = n_classes
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Cache for feature statistics
        self.feature_stats = {
            'activation_frequency': None,
            'spatial_variance': None,
            'steerability_scores': None,
            'class_effects': None,  # Replace output_scores with class_effects
            'mean_magnitude': None
        }

    def compute_activation_statistics(self, dataloader, n_batches=100):
        """Compute comprehensive activation statistics for all features"""

        n_features = self.sae.cfg.d_sae
        activation_counts = torch.zeros(n_features).to(self.device)
        activation_magnitudes = torch.zeros(n_features).to(self.device)
        spatial_activations = defaultdict(lambda: defaultdict(list))
        total_tokens = 0

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(dataloader):
                if i >= n_batches:
                    break

                imgs = imgs.to(self.device)

                # Get model activations at the SAE hook point
                _, cache = self.model.run_with_cache(imgs, names_filter=self.sae.cfg.hook_point)
                acts = cache[self.sae.cfg.hook_point]

                # For vision models, acts shape is typically [batch, num_patches+1, hidden_dim]
                # Where patch 0 is the CLS token

                # Encode through SAE
                feature_acts = self.sae.encode(acts)[1]  # [batch, seq_len, d_sae]

                # Track activation frequency
                activation_counts += (feature_acts > 0).float().sum(dim=[0, 1])
                activation_magnitudes += feature_acts.sum(dim=[0, 1])

                # Track spatial patterns for patches (excluding CLS token)
                if acts.shape[1] > 1:  # Has spatial patches
                    # Separate CLS token from patches
                    cls_features = feature_acts[:, 0, :]  # CLS token
                    patch_features = feature_acts[:, 1:, :]  # Spatial patches

                    # Store CLS separately
                    spatial_activations['cls'][0].append((cls_features > 0).float().mean(dim=0))

                    # For patches, we can reshape to spatial grid if needed
                    n_patches = patch_features.shape[1]
                    grid_size = int(np.sqrt(n_patches))

                    if grid_size * grid_size == n_patches:
                        # Reshape to spatial grid
                        patch_features_grid = patch_features.reshape(imgs.shape[0], grid_size, grid_size, -1)

                        # Track activations by spatial position
                        for row in range(grid_size):
                            for col in range(grid_size):
                                patch_idx = row * grid_size + col
                                spatial_activations['patches'][patch_idx].append(
                                    (patch_features_grid[:, row, col, :] > 0).float().mean(dim=0)
                                )

                total_tokens += imgs.shape[0] * acts.shape[1]

        # Compute statistics
        self.feature_stats['activation_frequency'] = activation_counts / total_tokens
        self.feature_stats['mean_magnitude'] = activation_magnitudes / activation_counts.clamp(min=1)

        # Compute spatial variance
        if spatial_activations['patches']:
            spatial_patterns = []
            for patch_idx in sorted(spatial_activations['patches'].keys()):
                spatial_patterns.append(torch.stack(spatial_activations['patches'][patch_idx]).mean(dim=0))
            spatial_patterns = torch.stack(spatial_patterns)  # [n_patches, n_features]
            self.feature_stats['spatial_variance'] = spatial_patterns.var(dim=0)

            # Also compute CLS vs patch differences
            cls_pattern = torch.stack(spatial_activations['cls'][0]).mean(dim=0)
            patch_mean = spatial_patterns.mean(dim=0)
            self.feature_stats['cls_patch_diff'] = (cls_pattern - patch_mean).abs()

    def compute_steerability_scores(self, dataloader, n_samples=100, strengths=[10, 50, 150]):
        """
        Compute steerability scores for vision model features
        """
        n_features = self.sae.cfg.d_sae
        steerability_scores = torch.zeros(n_features).to(self.device)

        # Collect test samples
        test_images = []
        test_labels = []
        for imgs, labels in dataloader:
            test_images.append(imgs)
            test_labels.append(labels)
            if sum(len(b) for b in test_images) >= n_samples:
                break

        test_images = torch.cat(test_images)[:n_samples].to(self.device)
        test_labels = torch.cat(test_labels)[:n_samples].to(self.device)

        # Test each feature
        for feature_idx in range(n_features):
            if self.feature_stats['activation_frequency'][feature_idx] < 0.001:
                continue  # Skip dead features

            feature_effects = []

            for strength in strengths:
                with torch.no_grad():
                    # Get baseline predictions
                    baseline_logits = self.model(test_images)
                    baseline_probs = F.softmax(baseline_logits, dim=-1)

                    # Define steering hook
                    def steering_hook(resid, hook):
                        # Only steer the CLS token for classification
                        sae_encoded = self.sae.encode(resid)[1]
                        sae_encoded[:, 0, feature_idx] = strength
                        return self.sae.decode(sae_encoded)

                    # Run with steering
                    with self.model.hooks(fwd_hooks=[(self.sae.cfg.hook_point, steering_hook)]):
                        steered_logits = self.model(test_images)
                        steered_probs = F.softmax(steered_logits, dim=-1)

                    # Measure effect concentration
                    prob_changes = (steered_probs - baseline_probs)**2
                    feature_effects.append(prob_changes.mean().item())

            # Steerability is the maximum effect across strengths
            steerability_scores[feature_idx] = max(feature_effects)

        self.feature_stats['steerability_scores'] = steerability_scores

    def compute_class_effects(self, dataloader, n_samples=1000):
        """
        Compute how features affect class predictions (vision-specific)
        """
        n_features = self.sae.cfg.d_sae
        class_effects = torch.zeros(n_features, self.n_classes).to(self.device)
        feature_class_counts = torch.zeros(n_features, self.n_classes).to(self.device)

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(dataloader):
                if i * imgs.shape[0] >= n_samples:
                    break

                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # Get activations and features
                _, cache = self.model.run_with_cache(imgs, names_filter=self.sae.cfg.hook_point)
                acts = cache[self.sae.cfg.hook_point]

                # Get SAE features (focus on CLS token for classification)
                feature_acts = self.sae.encode(acts)[1]
                cls_features = feature_acts[:, 0, :]  # [batch, d_sae]

                # For each sample, track which features are active for which class
                for b in range(imgs.shape[0]):
                    active_features = torch.where(cls_features[b] > 0)[0]
                    label = labels[b].item()

                    # Increment counts for active features
                    feature_class_counts[active_features, label] += 1

                    # Accumulate feature magnitudes by class
                    class_effects[active_features, label] += cls_features[b, active_features]

        # Normalize by counts
        class_effects = class_effects / (feature_class_counts + 1e-6)

        # Compute class selectivity (how much a feature prefers one class over others)
        class_probs = F.softmax(class_effects, dim=-1)
        class_selectivity = -(class_probs * (class_probs + 1e-10).log()).sum(dim=-1)
        class_selectivity = -class_selectivity  # Invert so higher is more selective

        self.feature_stats['class_effects'] = class_effects
        self.feature_stats['class_selectivity'] = class_selectivity

    def find_discriminative_features(self, dataloader, n_batches=100):
        """
        Find class-discriminative features (improved version of your original method)
        """
        n_features = self.sae.cfg.d_sae
        class_feature_scores = torch.zeros(self.n_classes, n_features).to(self.device)
        class_counts = torch.zeros(self.n_classes).to(self.device)

        # Map labels if needed (from your original code)
        label_map = {2: 3, 3: 2}

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(dataloader):
                if i >= n_batches:
                    break

                imgs = imgs.to(self.device)

                # Apply label mapping
                mapped_labels = torch.tensor([label_map.get(l.item(), l.item()) for l in labels]).to(self.device)

                # Get features
                _, cache = self.model.run_with_cache(imgs, names_filter=self.sae.cfg.hook_point)
                acts = cache[self.sae.cfg.hook_point]
                feature_acts = self.sae.encode(acts)[1]

                # Focus on CLS token for classification
                cls_features = feature_acts[:, 0, :]

                # Accumulate by class
                for j in range(imgs.shape[0]):
                    label = mapped_labels[j].item()
                    if label < self.n_classes:
                        class_feature_scores[label] += cls_features[j]
                        class_counts[label] += 1

        # Normalize
        for i in range(self.n_classes):
            if class_counts[i] > 0:
                class_feature_scores[i] /= class_counts[i]

        # Find discriminative features
        discriminative_features = {}
        for i in range(self.n_classes):
            # Features that fire strongly for class i
            class_scores = class_feature_scores[i]

            # Features that fire strongly for i but not others
            other_classes_mean = (class_feature_scores.sum(0) - class_scores) / (self.n_classes - 1)
            discrimination_scores = class_scores - other_classes_mean

            # Get top features
            top_k = 20
            top_indices = discrimination_scores.topk(top_k).indices

            # Filter by steerability if available
            if self.feature_stats['steerability_scores'] is not None:
                steerable_mask = self.feature_stats['steerability_scores'][top_indices] > 0.1
                steerable_features = top_indices[steerable_mask]

                if len(steerable_features) < 5:
                    # If too few steerable features, keep top 5 regardless
                    steerable_features = top_indices[:5]
            else:
                steerable_features = top_indices

            discriminative_features[i] = {
                'top_features': steerable_features.tolist(),
                'discrimination_scores': discrimination_scores[steerable_features].tolist(),
                'mean_activation': class_scores[steerable_features].tolist()
            }

        return discriminative_features

    def filter_always_on_features(self, threshold=0.8):
        """Filter out features that activate too frequently"""
        if self.feature_stats['activation_frequency'] is None:
            raise ValueError("Must compute activation statistics first")

        # Always-on: activate too frequently
        always_on_mask = self.feature_stats['activation_frequency'] > threshold

        # Spatially uniform: same activation pattern everywhere
        if self.feature_stats.get('spatial_variance') is not None:
            spatially_uniform_mask = self.feature_stats['spatial_variance'] < 0.01
        else:
            spatially_uniform_mask = torch.zeros_like(always_on_mask)

        # Dead features: never activate
        dead_mask = self.feature_stats['activation_frequency'] < 0.001

        # Valid features are not always-on, not uniform, and not dead
        valid_mask = ~(always_on_mask | spatially_uniform_mask | dead_mask)

        self.logger.info(f"Feature filtering summary:")
        self.logger.info(f"  Always-on: {always_on_mask.sum().item()}")
        self.logger.info(f"  Spatially uniform: {spatially_uniform_mask.sum().item()}")
        self.logger.info(f"  Dead: {dead_mask.sum().item()}")
        self.logger.info(f"  Valid: {valid_mask.sum().item()}")

        return valid_mask

    def select_features_for_steering(self,
                                     n_features_per_class: int = 10,
                                     weights: Optional[Dict[str, float]] = None) -> Dict[int, Dict[str, List[int]]]:
        """
        Select best features for steering each class
        """
        if weights is None:
            weights = {'steerability': 0.4, 'class_selectivity': 0.3, 'discrimination': 0.3}

        # Get valid features
        valid_mask = self.filter_always_on_features()

        # Get discriminative features
        discriminative_features = self.find_discriminative_features(train_loader)

        steering_features = {}

        for class_idx in range(self.n_classes):
            # Get candidate features for this class
            candidates = discriminative_features[class_idx]['top_features']

            # Filter by validity
            candidates = [f for f in candidates if valid_mask[f]]

            if len(candidates) < n_features_per_class:
                # If not enough valid candidates, expand search
                class_effects = self.feature_stats['class_effects'][:, class_idx]
                additional = class_effects.topk(n_features_per_class * 3).indices
                candidates.extend([f.item() for f in additional if valid_mask[f] and f not in candidates])

            # Score each candidate
            scores = []
            for feat_idx in candidates[:n_features_per_class * 2]:  # Consider 2x candidates
                score = 0

                if 'steerability' in weights and self.feature_stats['steerability_scores'] is not None:
                    steer_score = self.feature_stats['steerability_scores'][feat_idx].item()
                    score += weights['steerability'] * steer_score

                if 'class_selectivity' in weights and self.feature_stats.get('class_selectivity') is not None:
                    select_score = self.feature_stats['class_selectivity'][feat_idx].item()
                    score += weights['class_selectivity'] * select_score

                if 'discrimination' in weights:
                    disc_idx = discriminative_features[class_idx]['top_features'].index(feat_idx) \
                              if feat_idx in discriminative_features[class_idx]['top_features'] else -1
                    if disc_idx >= 0:
                        disc_score = discriminative_features[class_idx]['discrimination_scores'][disc_idx]
                        # Normalize to [0, 1]
                        disc_score = (disc_score - min(discriminative_features[class_idx]['discrimination_scores'])) / \
                                   (max(discriminative_features[class_idx]['discrimination_scores']) -
                                    min(discriminative_features[class_idx]['discrimination_scores']) + 1e-6)
                        score += weights['discrimination'] * disc_score

                scores.append((feat_idx, score))

            # Sort by score and select top features
            scores.sort(key=lambda x: x[1], reverse=True)
            selected = [s[0] for s in scores[:n_features_per_class]]

            steering_features[class_idx] = {
                'features': selected,
                'scores': [s[1] for s in scores[:n_features_per_class]]
            }

            print(scores)
            if scores == []:
                continue

            self.logger.info(
                f"Class {class_idx}: selected {len(selected)} features, "
                f"top score: {scores[0][1]:.3f}"
            )

        return steering_features


# Example usage in your find_features.py
if __name__ == "__main__":
    import torchvision

    from find_features import load_models  # Your existing imports

    # Load models
    sae, model = load_models()

    # Create dataset
    train_path = "./hyper-kvasir_imagefolder/train"
    train_dataset = torchvision.datasets.ImageFolder(
        train_path,
        get_processor_for_precached_224_images(),
        target_transform=lambda t: {
            2: 3,
            3: 2
        }.get(t, t)  # Your label mapping
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Create feature selector
    selector = VisionSAEFeatureSelector(sae, model, n_classes=6)

    # Compute statistics
    print("Computing activation statistics...")
    selector.compute_activation_statistics(train_loader, n_batches=1000)

    print("Computing steerability scores...")
    selector.compute_steerability_scores(train_loader, n_samples=100)

    print("Computing class effects...")
    selector.compute_class_effects(train_loader, n_samples=100)

    # Select features for steering
    print("\nSelecting features for steering...")
    steering_features = selector.select_features_for_steering(
        n_features_per_class=10, weights={
            'steerability': 0.5,
            'class_selectivity': 0.3,
            'discrimination': 0.2
        }
    )

    # Print results
    print("\nSelected steering features by class:")
    for class_idx, data in steering_features.items():
        if data['features'] == []:
            continue

        print(f"\nClass {class_idx}:")
        print(f"  Features: {data['features'][:5]}...")  # Show first 5
        print(f"  Top score: {data['scores'][0]:.3f}")

        # Check statistics for top feature
        top_feat = data['features'][0]
        print(f"  Top feature {top_feat} stats:")
        print(f"    Activation frequency: {selector.feature_stats['activation_frequency'][top_feat]:.3f}")
        print(f"    Steerability: {selector.feature_stats['steerability_scores'][top_feat]:.3f}")

