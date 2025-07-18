import argparse
from pathlib import Path

import torch

from vit.model import IDX2CLS

# Optional: To make the output more readable, you can add your class index to name mapping here.
# If you don't have one, it will just use the class indices (0, 1, 2...).
IDX_TO_CLASS_NAME = IDX2CLS


def analyze_steerability_pfac_overlap(class_feature_records, class_name):
    """
    Analyzes the overlap between top 50 steerable features and top 50 PFAC features.
    """
    if len(class_feature_records) < 50:
        print(f"Warning: Only {len(class_feature_records)} features available for {class_name}")
        top_n = len(class_feature_records)
    else:
        top_n = 50

    # Get top N features by absolute steerability and absolute PFAC
    top_steer = sorted(class_feature_records, key=lambda x: abs(x['steer']), reverse=True)[:top_n]
    top_pfac = sorted(class_feature_records, key=lambda x: abs(x['pfac']), reverse=True)[:top_n]

    # Convert to sets of feature IDs for overlap calculation
    steer_fids = {f['fid'] for f in top_steer}
    pfac_fids = {f['fid'] for f in top_pfac}

    # Calculate overlap
    overlap_fids = steer_fids & pfac_fids
    overlap_count = len(overlap_fids)
    overlap_percentage = (overlap_count / top_n) * 100

    print(f"\n{'='*15} STEERABILITY vs PFAC OVERLAP ANALYSIS {'='*15}")
    print(f"Top {top_n} steerable features: {len(steer_fids)}")
    print(f"Top {top_n} PFAC features: {len(pfac_fids)}")
    print(f"Overlap: {overlap_count} features ({overlap_percentage:.1f}%)")

    # Analyze characteristics of overlapping vs non-overlapping features
    overlapping_features = [f for f in class_feature_records if f['fid'] in overlap_fids]
    steer_only = [f for f in class_feature_records if f['fid'] in steer_fids - overlap_fids]
    pfac_only = [f for f in class_feature_records if f['fid'] in pfac_fids - overlap_fids]

    def get_stats(features, label):
        if not features:
            return
        avg_steer = sum(f['steer'] for f in features) / len(features)
        avg_pfac = sum(f['pfac'] for f in features) / len(features)
        avg_combined = sum(f['combined'] for f in features) / len(features)
        avg_local = sum(f['local'] for f in features) / len(features)

        print(f"\n{label} ({len(features)} features):")
        print(f"  Avg Steerability: {avg_steer:.3f}")
        print(f"  Avg PFAC: {avg_pfac:.3f}")
        print(f"  Avg Combined: {avg_combined:.3f}")
        print(f"  Avg Locality: {avg_local:.3f}")

    get_stats(overlapping_features, "Overlapping Features")
    get_stats(steer_only, "High Steer Only")
    get_stats(pfac_only, "High PFAC Only")

    # Show some examples of overlapping features
    if overlapping_features:
        print(f"\nTop 10 Overlapping Features:")
        overlapping_sorted = sorted(overlapping_features, key=lambda x: x['combined'], reverse=True)[:10]
        for i, feat in enumerate(overlapping_sorted):
            print(
                f"  {i+1}. Feature {feat['fid']}: Steer={feat['steer']:.3f}, PFAC={feat['pfac']:.3f}, Combined={feat['combined']:.3f}"
            )


def analyze_dictionary(file_path: Path):
    """
    Loads a feature dictionary and prints a per-class analysis of the top features
    for different metrics (Combined Score, Steerability, Frequency, etc.).
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Analyzing Feature Dictionary: {file_path.name} ---")
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    feature_stats = data.get('feature_stats')
    metadata = data.get('metadata', {})

    if not feature_stats:
        print("Error: 'feature_stats' not found in the dictionary.")
        return

    print(f"Layer: {metadata.get('layer_idx', 'N/A')}, Total Features in Dict: {len(feature_stats)}")
    print("-" * 50)

    # Discover all classes present in the dictionary
    all_classes = set()
    for info in feature_stats.values():
        all_classes.update(info.get('class_count_map', {}).keys())

    if not all_classes:
        print("No class-specific information found in the dictionary.")
        return

    # --- Main analysis loop for each class ---
    for class_id in sorted(list(all_classes)):
        class_name = IDX_TO_CLASS_NAME.get(class_id, f"Class {class_id}")
        print(f"\n{'='*20} ANALYSIS FOR: {class_name.upper()} {'='*20}")

        # 1. Prepare a list of features relevant to this class
        class_feature_records = []
        for fid, info in feature_stats.items():
            # Only include features that have appeared with this class
            if class_id in info.get('class_count_map', {}):
                record = {
                    'fid': fid,
                    'combined': info.get('class_combined_scores', {}).get(class_id, 0.0),
                    'steer': info.get('class_mean_steerability', {}).get(class_id, 0.0),
                    'freq': info.get('class_count_map', {}).get(class_id, 0),
                    'pfac': info.get('class_mean_pfac', {}).get(class_id, 0.0),
                    # Lower score is more local, so we'll sort ascending for "top"
                    'local': info.get('class_mean_locality', {}).get(class_id, 1.0)
                }
                class_feature_records.append(record)

        if not class_feature_records:
            print("No features recorded for this class.")
            continue

        # 2. Define the metrics to analyze
        metrics_to_analyze = {
            "Combined Score": ("combined", True),
            "Steerability": ("steer", True),
            "Frequency (for this class)": ("freq", True),
            "PFAC Correlation": ("pfac", True),
            "Locality (Most Local)": ("local", False),  # False means sort ascending
        }

        # 3. Perform and print the top 5 analysis for each metric
        for title, (key, reverse_sort) in metrics_to_analyze.items():
            print(f"\n--- Top 5 by {title} ---")

            # Sort the records by the current metric
            sorted_features = sorted(class_feature_records, key=lambda x: x[key], reverse=reverse_sort)[:5]

            if not sorted_features:
                print("  (No features found for this metric)")
                continue

            for i, feat in enumerate(sorted_features):
                # Print a detailed line for each top feature for easy comparison
                print(
                    f"  {i+1}. Feature {feat['fid']:<5}: "
                    f"{key.capitalize():<8}= {feat[key]:<6.3f} | "
                    f"Combined={feat['combined']:.2f}, "
                    f"Steer={feat['steer']:.2f}, "
                    f"Freq={feat['freq']:<4}, "
                    f"PFAC={feat['pfac']:.2f}, "
                    f"Local={feat['local']:.2f}"
                )

        # 4. Analyze overlap between top steerable and top PFAC features
        analyze_steerability_pfac_overlap(class_feature_records, class_name)


if __name__ == "__main__":
    for layer_ídx in range(2, 11):
        path = f"./sae_dictionaries/steer_corr_local_l{layer_ídx}_alignment_min1_64k64.pt"

        analyze_dictionary(Path(path))
