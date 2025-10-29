"""
Unit tests for Faithfulness Correlation implementation.

Based on "Evaluating and Aggregating Feature-based Model Explanations" by Bhatt et al. (2020).

Faithfulness Correlation measures the correlation between:
- Sum of attribution scores for random patch subsets
- Prediction change when those patches are perturbed

Key property: Good explanations → high correlation (≈1.0)
              Bad explanations → low/zero correlation (≈0.0)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from faithfulness import FaithfulnessCorrelation

# ============= TEST UTILITIES =============


class MockModelForCorrelation:
    """
    Mock model where prediction drop is proportional to sum of perturbed patch attributions.
    This allows us to create controlled correlation scenarios.
    """

    def __init__(self, attributions, correlation_type="perfect"):
        """
        Args:
            attributions: Ground truth attributions (n_patches,)
            correlation_type: "perfect", "none", or "anti"
        """
        self.true_attributions = attributions
        self.correlation_type = correlation_type
        self.original_called = False
        self.original_pred = None
        self.call_count = 0

    def eval(self):
        return self

    def __call__(self, x_batch):
        """
        Returns predictions based on which patches are perturbed.

        Perfect correlation: prediction drop ∝ sum of perturbed patch attributions
        No correlation: random prediction drops
        Anti-correlation: prediction drop ∝ negative sum of attributions
        """
        batch_size = x_batch.shape[0]

        # First call: return high predictions (original unperturbed)
        if not self.original_called:
            self.original_called = True
            self.original_pred = 0.9
            return torch.ones(batch_size, 10) * self.original_pred

        # Subsequent calls: return predictions based on perturbation
        predictions = []
        grid_size = int(np.sqrt(len(self.true_attributions)))
        patch_size = 224 // grid_size

        for i in range(batch_size):
            # Detect which patches are perturbed (set to image minimum)
            img = x_batch[i]
            img_min = img.min()
            perturbed_patches = []

            for patch_idx in range(len(self.true_attributions)):
                row = patch_idx // grid_size
                col = patch_idx % grid_size
                start_row = row * patch_size
                end_row = start_row + patch_size
                start_col = col * patch_size
                end_col = start_col + patch_size

                patch_region = img[:, start_row:end_row, start_col:end_col]

                # If patch is at minimum value (perturbed with "black" baseline), record it
                if np.allclose(patch_region, img_min, atol=1e-6):
                    perturbed_patches.append(patch_idx)

            # Calculate prediction based on correlation type
            attr_sum = sum(self.true_attributions[idx] for idx in perturbed_patches)

            if self.correlation_type == "perfect":
                # High attribution → large drop
                drop = attr_sum * 0.5  # Scale factor
            elif self.correlation_type == "none":
                # Deterministic but uncorrelated drop
                # Use a pattern independent of attribution: based on patch count
                # This ensures consistent results but no correlation with attr_sum
                n_perturbed = len(perturbed_patches)
                drop = 0.3 + 0.05 * np.sin(n_perturbed + self.call_count)
            elif self.correlation_type == "anti":
                # High attribution → small drop (anti-correlated)
                drop = -attr_sum * 0.5 + 0.3
            else:
                drop = 0.2

            self.call_count += 1
            pred = max(0.1, self.original_pred - drop)

            # Create logits
            logits = torch.ones(10) * -5.0
            logits[0] = pred * 5
            predictions.append(logits)

        return torch.stack(predictions)


# ============= TESTS WITH KNOWN EXPECTED RESULTS =============


@pytest.mark.unit
def test_perfect_correlation():
    """
    Test 1: Perfect correlation (correlation ≈ 1.0)

    When attributions perfectly match importance:
    - High attribution patches → large prediction drops
    - Low attribution patches → small prediction drops
    - Correlation should be close to 1.0
    """
    n_patches = 16  # 4x4 grid for simplicity

    # Create attributions with clear variance
    attributions = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01])

    model = MockModelForCorrelation(attributions, correlation_type="perfect")

    # Create test data
    x_batch = np.random.randn(1, 3, 224, 224).astype(np.float32)
    y_batch = np.array([0])
    a_batch = attributions.reshape(1, -1)

    # Run faithfulness correlation
    fc = FaithfulnessCorrelation(
        n_patches=n_patches,
        patch_size=56,  # 224/4 = 56
        subset_size=4,  # Perturb 4 patches at a time
        nr_runs=20,  # Multiple runs for stable correlation
        perturb_baseline="black"
    )

    correlation = fc.evaluate_batch(model, x_batch, y_batch, a_batch, device="cpu")[0]

    # With perfect correlation between attribution and impact, expect high positive correlation
    assert correlation > 0.8, \
        f"Perfect correlation case should give correlation > 0.7, got {correlation:.3f}"


@pytest.mark.unit
def test_zero_correlation():
    """
    Test 2: No correlation (correlation ≈ 0.0)

    When attributions don't match importance:
    - Attribution says some patches are important
    - But model's predictions drop regardless of attribution (uncorrelated pattern)
    - Correlation should be close to 0.0
    """
    # Set seed for reproducibility
    np.random.seed(42)

    n_patches = 16

    # Attributions with variance (but don't match actual importance)
    attributions = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01])

    # Model returns uncorrelated prediction drops (based on patch count, not attribution)
    model = MockModelForCorrelation(attributions, correlation_type="none")

    x_batch = np.random.randn(1, 3, 224, 224).astype(np.float32)
    y_batch = np.array([0])
    a_batch = attributions.reshape(1, -1)

    fc = FaithfulnessCorrelation(
        n_patches=n_patches, patch_size=56, subset_size=4, nr_runs=20, perturb_baseline="black"
    )

    correlation = fc.evaluate_batch(model, x_batch, y_batch, a_batch, device="cpu")[0]

    # With no relationship, expect correlation near 0
    # Using wider bounds since some spurious correlation can occur
    assert -0.4 < correlation < 0.4, \
        f"No correlation case should give correlation near 0, got {correlation:.3f}"


@pytest.mark.unit
def test_perfect_better_than_zero_correlation():
    """
    Test 3: Comparative test - perfect correlation should be much higher than random.

    This verifies that the metric distinguishes between good and bad explanations.
    """
    n_patches = 16
    attributions = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01])

    x_batch = np.random.randn(1, 3, 224, 224).astype(np.float32)
    y_batch = np.array([0])
    a_batch = attributions.reshape(1, -1)

    fc = FaithfulnessCorrelation(
        n_patches=n_patches, patch_size=56, subset_size=4, nr_runs=20, perturb_baseline="black"
    )

    # Perfect correlation
    model_perfect = MockModelForCorrelation(attributions, correlation_type="perfect")
    corr_perfect = fc.evaluate_batch(model_perfect, x_batch, y_batch, a_batch, device="cpu")[0]

    # No correlation
    model_none = MockModelForCorrelation(attributions, correlation_type="none")
    corr_none = fc.evaluate_batch(model_none, x_batch, y_batch, a_batch, device="cpu")[0]

    # Perfect should be significantly higher than random
    assert corr_perfect > corr_none + 0.3, \
        f"Perfect correlation ({corr_perfect:.3f}) should be much higher than no correlation ({corr_none:.3f})"


@pytest.mark.unit
def test_uniform_attributions_give_low_correlation():
    """
    Test 4: Uniform attributions (all patches equally important)

    When all attributions are equal:
    - No variance in attribution sums across different subsets
    - This should result in low/undefined correlation

    Note: This test will produce a ConstantInputWarning from scipy, which is expected.
    """
    n_patches = 16

    # All attributions are equal
    attributions = np.ones(n_patches) * 0.5

    model = MockModelForCorrelation(attributions, correlation_type="perfect")

    x_batch = np.random.randn(1, 3, 224, 224).astype(np.float32)
    y_batch = np.array([0])
    a_batch = attributions.reshape(1, -1)

    fc = FaithfulnessCorrelation(
        n_patches=n_patches, patch_size=56, subset_size=4, nr_runs=20, perturb_baseline="black"
    )

    correlation = fc.evaluate_batch(model, x_batch, y_batch, a_batch, device="cpu")[0]

    # With uniform attributions, all subset sums are equal → no variance → low/undefined correlation
    # The correlation might be NaN or close to 0
    assert np.isnan(correlation) or abs(correlation) < 0.3, \
        f"Uniform attributions should give low/undefined correlation, got {correlation:.3f}"


# ============= PYTEST CONFIGURATION =============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
