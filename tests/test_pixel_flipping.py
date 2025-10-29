"""
Unit tests for Pixel Flipping semantic correctness.

Based on "On Pixel-Wise Explanations for Non-Linear Classifier Decisions
by Layer-Wise Relevance Propagation" by Bach et al. (2015).

The pixel flipping method evaluates explanation faithfulness by:
1. Progressively perturbing the most important patches (highest attribution)
2. Measuring prediction degradation as patches are removed
3. Computing AUC to quantify faithfulness

These tests verify the calculation is correct by:
- Mocking model responses to return known prediction sequences
- Calculating expected AUC mathematically
- Verifying pixel flipping returns the expected AUC
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from faithfulness import PatchPixelFlipping


# ============= TEST UTILITIES =============


class MockModelWithPredefinedOutputs:
    """
    Mock model that returns a predefined sequence of predictions.
    This allows us to test with known outputs and verify the AUC calculation.
    """

    def __init__(self, prediction_sequence, target_class=0):
        """
        Args:
            prediction_sequence: List of probabilities for target class at each step
                                e.g., [0.9, 0.7, 0.5, 0.3, 0.1]
            target_class: Index of the target class
        """
        self.prediction_sequence = prediction_sequence
        self.target_class = target_class
        self.call_count = 0

    def eval(self):
        return self

    def __call__(self, x_batch):
        """
        Returns the next prediction in the sequence.
        """
        batch_size = x_batch.shape[0]

        # Get the current prediction value
        if self.call_count < len(self.prediction_sequence):
            prob = self.prediction_sequence[self.call_count]
        else:
            prob = 0.0  # Fallback if we run out of predictions

        self.call_count += 1

        # Create logits that will produce the desired probability after softmax
        # If we want P(target_class) = p with 10 total classes:
        # Set all non-target logits to 0, and target logit to x where:
        # p = exp(x) / (exp(x) + 9)  =>  x = log(9p / (1-p))
        logits = torch.zeros(batch_size, 10)  # All other classes at 0

        if prob > 0 and prob < 1.0:
            target_logit = np.log(9.0 * prob / (1.0 - prob))
            logits[:, self.target_class] = target_logit
        elif prob >= 1.0:
            logits[:, self.target_class] = 10.0  # Very high logit for prob ≈ 1
        else:  # prob == 0
            logits[:, self.target_class] = -10.0  # Very low logit for prob ≈ 0

        return logits


# ============= TESTS WITH KNOWN EXPECTED RESULTS =============


@pytest.mark.unit
def test_auc_calculation_linear_decline():
    """
    Test 1: Linear decline from 1.0 to 0.0

    Model returns: [1.0, 0.75, 0.5, 0.25, 0.0] as patches are perturbed
    Expected AUC = trapezoid([1.0, 0.75, 0.5, 0.25, 0.0], dx=1) = 1.5

    Calculation: (1.0+0.75)/2 + (0.75+0.5)/2 + (0.5+0.25)/2 + (0.25+0.0)/2
               = 0.875 + 0.625 + 0.375 + 0.125 = 2.0
    """
    prediction_probs = [1.0, 0.75, 0.5, 0.25, 0.0]
    expected_auc = np.trapezoid(prediction_probs, dx=1)

    assert np.isclose(expected_auc, 2.0), f"Manual calculation check failed: {expected_auc}"

    # Setup pixel flipping with 4 patches (5 predictions: initial + 4 perturbations)
    n_patches = 4
    model = MockModelWithPredefinedOutputs(prediction_probs, target_class=0)

    x_batch = np.random.randn(1, 3, 224, 224).astype(np.float32)
    y_batch = np.array([0])
    a_batch = np.array([[1.0, 0.8, 0.6, 0.4]])  # Order doesn't matter for this test

    pf = PatchPixelFlipping(n_patches=n_patches, patch_size=56, features_in_step=1)
    actual_auc = pf.evaluate_batch(model, x_batch, y_batch, a_batch, device="cpu")[0]

    # Verify we get the expected AUC
    assert np.isclose(actual_auc, expected_auc, atol=0.01), \
        f"Expected AUC={expected_auc:.3f}, got {actual_auc:.3f}"


@pytest.mark.unit
def test_auc_calculation_steep_drop():
    """
    Test 2: Steep drop (good explanation)

    Model returns: [0.9, 0.3, 0.1, 0.05] as patches are perturbed
    Expected AUC = trapezoid([0.9, 0.3, 0.1, 0.05], dx=1)

    Calculation: (0.9+0.3)/2 + (0.3+0.1)/2 + (0.1+0.05)/2
               = 0.6 + 0.2 + 0.075 = 0.875
    """
    prediction_probs = [0.9, 0.3, 0.1, 0.05]
    expected_auc = np.trapezoid(prediction_probs, dx=1)

    assert np.isclose(expected_auc, 0.875), f"Manual calculation check failed: {expected_auc}"

    n_patches = 3
    model = MockModelWithPredefinedOutputs(prediction_probs, target_class=0)

    x_batch = np.random.randn(1, 3, 224, 224).astype(np.float32)
    y_batch = np.array([0])
    a_batch = np.array([[1.0, 0.5, 0.1]])

    pf = PatchPixelFlipping(n_patches=n_patches, patch_size=74, features_in_step=1)
    actual_auc = pf.evaluate_batch(model, x_batch, y_batch, a_batch, device="cpu")[0]

    assert np.isclose(actual_auc, expected_auc, atol=0.01), \
        f"Expected AUC={expected_auc:.3f}, got {actual_auc:.3f}"


@pytest.mark.unit
def test_auc_calculation_gradual_drop():
    """
    Test 3: Gradual drop (bad explanation)

    Model returns: [0.9, 0.85, 0.8, 0.75, 0.7] as patches are perturbed
    Expected AUC = trapezoid([0.9, 0.85, 0.8, 0.75, 0.7], dx=1)

    Calculation: (0.9+0.85)/2 + (0.85+0.8)/2 + (0.8+0.75)/2 + (0.75+0.7)/2
               = 0.875 + 0.825 + 0.775 + 0.725 = 3.2
    """
    prediction_probs = [0.9, 0.85, 0.8, 0.75, 0.7]
    expected_auc = np.trapezoid(prediction_probs, dx=1)

    assert np.isclose(expected_auc, 3.2), f"Manual calculation check failed: {expected_auc}"

    n_patches = 4
    model = MockModelWithPredefinedOutputs(prediction_probs, target_class=0)

    x_batch = np.random.randn(1, 3, 224, 224).astype(np.float32)
    y_batch = np.array([0])
    a_batch = np.array([[1.0, 0.8, 0.6, 0.4]])

    pf = PatchPixelFlipping(n_patches=n_patches, patch_size=56, features_in_step=1)
    actual_auc = pf.evaluate_batch(model, x_batch, y_batch, a_batch, device="cpu")[0]

    assert np.isclose(actual_auc, expected_auc, atol=0.01), \
        f"Expected AUC={expected_auc:.3f}, got {actual_auc:.3f}"


@pytest.mark.unit
def test_steep_drop_lower_auc_than_gradual():
    """
    Test 4: Verify that steep drop (good explanation) has lower AUC than gradual drop (bad explanation).

    This is the key property: AUC measures area under prediction curve.
    - Steep drop → small area → low AUC → good explanation
    - Gradual drop → large area → high AUC → bad explanation
    """
    # Steep drop
    steep_probs = [0.9, 0.3, 0.1, 0.05]
    steep_auc_expected = np.trapezoid(steep_probs, dx=1)  # Should be ~0.875

    # Gradual drop
    gradual_probs = [0.9, 0.85, 0.8, 0.75]
    gradual_auc_expected = np.trapezoid(gradual_probs, dx=1)  # Should be ~2.475

    assert steep_auc_expected < gradual_auc_expected, \
        "Steep drop should have lower AUC than gradual drop"

    # Test with pixel flipping
    n_patches = 3
    x_batch = np.random.randn(1, 3, 224, 224).astype(np.float32)
    y_batch = np.array([0])
    a_batch = np.array([[1.0, 0.5, 0.1]])

    # Steep drop
    steep_model = MockModelWithPredefinedOutputs(steep_probs, target_class=0)
    pf = PatchPixelFlipping(n_patches=n_patches, patch_size=74, features_in_step=1)
    steep_auc = pf.evaluate_batch(steep_model, x_batch, y_batch, a_batch, device="cpu")[0]

    # Gradual drop
    gradual_model = MockModelWithPredefinedOutputs(gradual_probs, target_class=0)
    gradual_auc = pf.evaluate_batch(gradual_model, x_batch, y_batch, a_batch, device="cpu")[0]

    # Verify both match expected
    assert np.isclose(steep_auc, steep_auc_expected, atol=0.01), \
        f"Steep AUC: expected {steep_auc_expected:.3f}, got {steep_auc:.3f}"
    assert np.isclose(gradual_auc, gradual_auc_expected, atol=0.01), \
        f"Gradual AUC: expected {gradual_auc_expected:.3f}, got {gradual_auc:.3f}"

    # Verify steep < gradual
    assert steep_auc < gradual_auc, \
        f"Steep drop AUC ({steep_auc:.3f}) should be < gradual drop AUC ({gradual_auc:.3f})"


# ============= PYTEST CONFIGURATION =============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])