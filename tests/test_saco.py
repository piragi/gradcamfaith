"""
Unit tests for SaCo (Salience-guided Faithfulness Coefficient) implementation.

Based on the paper "On the Faithfulness of Vision Transformer Explanations" by Wu et al.
Tests verify that the implementation correctly calculates SaCo scores according to Algorithm 1.

Test Categories:
1. Core SaCo calculation tests (mathematical properties)
2. Separated function tests (unit tests for refactored functions)
3. Integration tests (full pipeline)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from attribution_binning import (
    BinInfo, BinnedPerturbationData, ImageData, calculate_binned_saco_for_image, calculate_saco_vectorized_with_bias,
    compute_saco_from_impacts, create_binned_perturbations, measure_bin_impacts
)
from data_types import (AttributionOutputPaths, ClassificationPrediction, ClassificationResult)

# ============= TEST UTILITIES & FIXTURES =============


def calculate_saco_simple(attributions: np.ndarray, confidence_impacts: np.ndarray) -> float:
    """Simple wrapper to get just the SaCo score from the existing function."""
    saco, _ = calculate_saco_vectorized_with_bias(attributions, confidence_impacts)
    return saco


@pytest.fixture
def mock_image_data():
    """Fixture for creating mock ImageData."""
    def _create(n_patches: int = 49, original_confidence: float = 0.9, 
                original_class_idx: int = 0, attributions: np.ndarray = None):
        if attributions is None:
            attributions = np.random.rand(n_patches)
        return ImageData(
            pil_image=None,  # Mock object
            tensor=torch.randn(3, 224, 224),
            raw_attributions=attributions,
            original_confidence=original_confidence,
            original_class_idx=original_class_idx
        )
    return _create


@pytest.fixture
def mock_bin_results():
    """Fixture for creating mock bin results."""
    def _create(attributions: list, impacts: list):
        results = []
        for i, (attr, impact) in enumerate(zip(attributions, impacts)):
            results.append({
                "bin_id": i,
                "mean_attribution": attr,
                "total_attribution": attr,
                "n_patches": 1,
                "confidence_delta": impact,
                "confidence_delta_abs": abs(impact),
                "class_changed": False
            })
        return results
    return _create


# ============= CATEGORY 1: CORE SACO CALCULATION TESTS =============


@pytest.mark.unit
def test_saco_perfect_alignment():
    """Test SaCo = 1.0 for perfect alignment between attribution and impact."""
    attributions = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    confidence_impacts = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

    saco = calculate_saco_simple(attributions, confidence_impacts)
    assert np.isclose(saco, 1.0, atol=1e-10), f"Expected SaCo=1.0, got {saco}"


@pytest.mark.unit
def test_saco_perfect_misalignment():
    """Test SaCo = -1.0 for perfect anti-correlation."""
    attributions = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    confidence_impacts = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Reversed

    saco = calculate_saco_simple(attributions, confidence_impacts)
    assert np.isclose(saco, -1.0, atol=1e-10), f"Expected SaCo=-1.0, got {saco}"


@pytest.mark.unit
def test_saco_no_correlation():
    """Test SaCo = 0.0 when all attributions are equal (no discrimination)."""
    attributions = np.array([5.0, 5.0, 5.0, 5.0])
    confidence_impacts = np.array([0.1, 0.3, 0.2, 0.4])

    saco = calculate_saco_simple(attributions, confidence_impacts)
    assert np.isclose(saco, 0.0, atol=1e-10), f"Expected SaCo=0.0, got {saco}"


@pytest.mark.unit
def test_saco_edge_cases():
    """Test edge cases: single bin, empty input, two bins."""
    # Single bin - no comparison possible
    saco_single = calculate_saco_simple(np.array([5.0]), np.array([0.5]))
    assert saco_single == 0.0

    # Empty input
    saco_empty = calculate_saco_simple(np.array([]), np.array([]))
    assert saco_empty == 0.0

    # Two bins aligned
    saco_two_aligned = calculate_saco_simple(np.array([10.0, 5.0]), np.array([0.8, 0.3]))
    assert np.isclose(saco_two_aligned, 1.0, atol=1e-10)

    # Two bins misaligned
    saco_two_misaligned = calculate_saco_simple(np.array([10.0, 5.0]), np.array([0.3, 0.8]))
    assert np.isclose(saco_two_misaligned, -1.0, atol=1e-10)


@pytest.mark.unit
def test_saco_paper_algorithm_verification():
    """Verify implementation matches Algorithm 1 from the paper exactly."""
    # Simple case we can verify by hand
    attributions = np.array([6.0, 3.0])
    confidence_impacts = np.array([0.8, 0.2])

    # One pair: (G1, G2), s(G1)=6, s(G2)=3, impact(G1)=0.8, impact(G2)=0.2
    # s(G1) > s(G2) and impact(G1) > impact(G2) → aligned
    # weight = s(G1) - s(G2) = 3
    # totalWeight = |weight| = 3
    # SaCo = 3/3 = 1.0

    saco = calculate_saco_simple(attributions, confidence_impacts)
    assert np.isclose(saco, 1.0, atol=1e-10)


# ============= CATEGORY 2: SEPARATED FUNCTION UNIT TESTS =============


@pytest.mark.unit
def test_compute_saco_from_impacts(mock_bin_results):
    """Test the compute_saco_from_impacts function directly."""
    # Create mock bin results
    bin_results = mock_bin_results(
        attributions=[10.0, 8.0, 6.0, 4.0, 2.0],
        impacts=[0.5, 0.4, 0.3, 0.2, 0.1]  # Perfectly aligned
    )

    impact_result = compute_saco_from_impacts(bin_results, compute_bias=True)

    assert np.isclose(impact_result.saco_score, 1.0, atol=1e-10)
    assert len(impact_result.bin_biases) == len(bin_results)
    assert all(np.isclose(bias, 0.0, atol=1e-10) for bias in impact_result.bin_biases)


@pytest.mark.unit
def test_compute_saco_detects_abs_bug():
    """
    This test MUST FAIL if compute_saco_from_impacts uses 
    confidence_delta_abs instead of confidence_delta.
    
    This test uses a negative impact that exposes the bug.
    """
    # Create bin results with the last impact being negative
    bin_results = [
        {
            "bin_id": 0,
            "mean_attribution": 10.0,
            "total_attribution": 10.0,
            "n_patches": 1,
            "confidence_delta": 0.5,
            "confidence_delta_abs": 0.5,
            "class_changed": False
        },
        {
            "bin_id": 1,
            "mean_attribution": 8.0,
            "total_attribution": 8.0,
            "n_patches": 1,
            "confidence_delta": 0.4,
            "confidence_delta_abs": 0.4,
            "class_changed": False
        },
        {
            "bin_id": 2,
            "mean_attribution": 6.0,
            "total_attribution": 6.0,
            "n_patches": 1,
            "confidence_delta": 0.3,
            "confidence_delta_abs": 0.3,
            "class_changed": False
        },
        {
            "bin_id": 3,
            "mean_attribution": 4.0,
            "total_attribution": 4.0,
            "n_patches": 1,
            "confidence_delta": 0.2,
            "confidence_delta_abs": 0.2,
            "class_changed": False
        },
        {
            "bin_id": 4,
            "mean_attribution": 2.0,
            "total_attribution": 2.0,
            "n_patches": 1,
            "confidence_delta": -0.25,  # NEGATIVE impact!
            "confidence_delta_abs": 0.25,  # Absolute is 0.25
            "class_changed": False
        }
    ]

    # Compute SaCo
    result = compute_saco_from_impacts(bin_results, compute_bias=True)

    # If using signed impacts (CORRECT):
    # Impacts: [0.5, 0.4, 0.3, 0.2, -0.25]
    # This is perfectly aligned (monotonically decreasing with attribution)
    # SaCo should be 1.0

    # If using absolute impacts (WRONG):
    # Impacts: [0.5, 0.4, 0.3, 0.2, 0.25]
    # This is NOT aligned! 0.25 > 0.2, so there's a violation
    # The pair (4.0, 2.0) has attribution 4 > 2 but impact 0.2 < 0.25
    # This should give SaCo < 1.0

    assert np.isclose(result.saco_score, 1.0, atol=1e-10), \
        f"Expected SaCo=1.0 with signed impacts, got {result.saco_score}. " \
        "This indicates compute_saco_from_impacts is using confidence_delta_abs instead of confidence_delta!"


@pytest.mark.unit
def test_measure_bin_impacts(mock_image_data):
    """Test the measure_bin_impacts function with mocked model."""
    # Setup test data
    image_data = mock_image_data(n_patches=4, original_confidence=0.9)

    mock_bins = [
        BinInfo(
            bin_id=i,
            min_value=val,
            max_value=val,
            patch_indices=[i],
            mean_attribution=val,
            total_attribution=val,
            n_patches=1
        ) for i, val in enumerate([0.8, 0.6, 0.4, 0.2])
    ]

    perturbation_data = BinnedPerturbationData(
        bins=mock_bins, perturbed_tensors=[torch.randn(3, 224, 224) for _ in mock_bins]
    )

    # Mock model predictions
    with patch('attribution_binning.batched_model_inference') as mock_inference:
        mock_inference.return_value = [
            {
                "predicted_class_idx": 0,
                "confidence": 0.3
            },  # impact = 0.6
            {
                "predicted_class_idx": 0,
                "confidence": 0.5
            },  # impact = 0.4
            {
                "predicted_class_idx": 0,
                "confidence": 0.7
            },  # impact = 0.2
            {
                "predicted_class_idx": 0,
                "confidence": 0.85
            },  # impact = 0.05
        ]

        bin_results = measure_bin_impacts(perturbation_data, image_data, Mock(), torch.device("cpu"))

    # Verify impacts
    expected_impacts = [0.6, 0.4, 0.2, 0.05]
    for result, expected in zip(bin_results, expected_impacts):
        assert np.isclose(result["confidence_delta"], expected, atol=1e-10)


# ============= CATEGORY 3: INTEGRATION TESTS =============


@pytest.mark.integration
def test_full_pipeline_with_mocked_io():
    """
    Integration test of the full pipeline with mocked file I/O.
    Tests that the refactored implementation correctly handles signed impacts.
    """
    # Create mock data
    raw_attributions = np.array([0.8, 0.6, 0.4, 0.2])

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f, raw_attributions)
        attr_file = Path(f.name)

    try:
        # Create mock classification result
        original_result = Mock(spec=ClassificationResult)
        original_result.image_path = Path("/fake/image.jpg")
        original_result.prediction = Mock(spec=ClassificationPrediction)
        original_result.prediction.confidence = 0.9
        original_result.prediction.predicted_class_idx = 0
        original_result.attribution_paths = Mock(spec=AttributionOutputPaths)
        original_result.attribution_paths.raw_attribution_path = attr_file

        # Mock config and model
        mock_config = Mock()
        mock_config.classify.target_size = [224, 224]
        mock_config.perturb.method = "mean"

        # Create predictions with mixed impacts (including negative)
        mock_predictions = [
            {
                "predicted_class_idx": 0,
                "confidence": 0.3
            },  # impact = 0.6
            {
                "predicted_class_idx": 0,
                "confidence": 0.95
            },  # impact = -0.05 (increases!)
            {
                "predicted_class_idx": 0,
                "confidence": 0.7
            },  # impact = 0.2
            {
                "predicted_class_idx": 0,
                "confidence": 0.85
            },  # impact = 0.05
        ]

        # Patch dependencies
        with patch('attribution_binning.preprocessing.preprocess_image') as mock_preprocess, \
             patch('attribution_binning.batched_model_inference') as mock_inference, \
             patch('attribution_binning.create_spatial_mask_for_bin') as mock_mask, \
             patch('attribution_binning.apply_binned_perturbation') as mock_perturb:

            mock_preprocess.return_value = (Mock(), torch.randn(3, 224, 224))
            mock_inference.return_value = mock_predictions
            mock_mask.return_value = torch.zeros(224, 224, dtype=torch.bool)
            mock_perturb.return_value = torch.randn(3, 224, 224)

            # Run the function
            saco_score, bin_results, _ = calculate_binned_saco_for_image(
                original_result, Mock(), mock_config, torch.device("cpu"), n_bins=4
            )

            # Verify that negative impacts exist (confidence increase)
            has_negative = any(r["confidence_delta"] < 0 for r in bin_results)
            assert has_negative, "Should handle negative impacts (confidence increases)"

            # SaCo should be valid
            assert -1.0 <= saco_score <= 1.0

    finally:
        attr_file.unlink()


@pytest.mark.integration
def test_separated_functions_integration(mock_image_data, mock_bin_results):
    """
    Test that the separated functions work together correctly
    without needing file I/O.
    """
    # Create test data
    image_data = mock_image_data(n_patches=4, attributions=np.array([0.8, 0.6, 0.4, 0.2]))

    # Create bins (simplified - normally done by create_binned_perturbations)
    bin_results = mock_bin_results(
        attributions=[0.8, 0.6, 0.4, 0.2],
        impacts=[0.5, 0.3, 0.2, 0.1]  # Well-aligned
    )

    # Compute SaCo
    impact_result = compute_saco_from_impacts(bin_results)

    # Should have high positive SaCo for aligned case
    assert impact_result.saco_score > 0.5
    assert len(impact_result.bin_results) == 4


# ============= PYTEST CONFIGURATION =============

if __name__ == "__main__":
    # Run pytest with current file when executed directly
    pytest.main([__file__, "-v"])
