#!/usr/bin/env python3
"""Test script to verify save/load functionality for SaCo results"""

import torch
from pathlib import Path

# Create dummy results that match the expected structure
dummy_results = {
    'results_by_type': {
        'under_attributed': {
            3559: {
                'mean_log_ratio': 4.713,
                'sum_of_means': 100.0,
                'sum_of_sums': 200.0,
                'confidence_score': 0.733,
                'n_occurrences': 43,
                'dominant_class': 'Non-COVID'
            },
            34104: {
                'mean_log_ratio': 3.277,
                'sum_of_means': 90.0,
                'sum_of_sums': 180.0,
                'confidence_score': 0.709,
                'n_occurrences': 500,
                'dominant_class': 'Non-COVID'
            }
        },
        'over_attributed': {
            46159: {
                'mean_log_ratio': -3.858,
                'sum_of_means': -80.0,
                'sum_of_sums': -160.0,
                'confidence_score': 0.590,
                'n_occurrences': 107,
                'dominant_class': 'COVID-19'
            }
        }
    },
    'analysis_params': {
        'layer_idx': 7,
        'activation_threshold': 0.01,
        'min_patches_per_feature': 3,
        'min_occurrences': 1,
        'n_images_processed': 5417
    }
}

# Save the dummy results
save_path = "results/saco_features_direct_l7_test.pt"
Path(save_path).parent.mkdir(parents=True, exist_ok=True)
torch.save(dummy_results, save_path)
print(f"Saved dummy results to {save_path}")

# Try to load them back
loaded = torch.load(save_path, weights_only=False)
print(f"Successfully loaded results")
print(f"Keys: {loaded.keys()}")
print(f"Number of under-attributed features: {len(loaded['results_by_type']['under_attributed'])}")
print(f"Number of over-attributed features: {len(loaded['results_by_type']['over_attributed'])}")

# Save with the real filename for testing
real_path = "results/saco_features_direct_l7.pt"
torch.save(dummy_results, real_path)
print(f"\nAlso saved to {real_path} for testing")