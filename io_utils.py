import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from data_types import ClassificationResult


def ensure_directories(directories: List[Path]) -> None:
    """Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)


def try_load_from_cache(cache_path: Path) -> Optional[ClassificationResult]:
    """Try to load cached results from a file.
    
    Args:
        cache_path: Path to the cache file
        
    Returns:
        Cached data as a dictionary if successful, None otherwise
    """
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
                return ClassificationResult.from_dict_for_cache(cached_data)
        except Exception as e:
            print(f"Error loading from cache {cache_path}: {e}. Recomputing.")
    return None


def save_to_cache(cache_path: Path, result: ClassificationResult) -> None:
    """Save results to cache.
    
    Args:
        cache_path: Path to the cache file
        result: Data to cache
    """
    try:
        with open(cache_path, 'w') as f:
            json.dump(result.to_dict_for_cache(), f, indent=4)
    except Exception as e:
        print(f"Error saving to cache {cache_path}: {e}")


def build_cache_path(cache_dir: Path,
                     image_path: Path,
                     suffix: str,
                     prefix: str = "") -> Path:
    """Build a cache path for an image.
    
    Args:
        cache_dir: Directory for cache files
        image_path: Path to the image
        suffix: Suffix to add to the filename
        prefix: Optional prefix to add to the filename
        
    Returns:
        Path to the cache file
    """
    return cache_dir / f"{prefix}{image_path.stem}{suffix}.json"


def save_classification_results_to_csv(results: List[ClassificationResult],
                                       output_path: Path):
    if not results:
        print("No results to save to CSV.")
        return
    flat_results_list = []
    for res in results:
        row = {
            'image_path': str(res.image_path),
            'predicted_class_label': res.prediction.predicted_class_label,
            'predicted_class_idx': res.prediction.predicted_class_idx,
            'confidence': res.prediction.confidence,
            'probabilities': res.prediction.probabilities,
        }
        if res.attribution_paths:
            row.update({
                'attribution_path':
                str(res.attribution_paths.attribution_path),
                'attribution_neg_path':
                str(res.attribution_paths.attribution_neg_path),
                'ffn_activity_path':
                str(res.attribution_paths.ffn_activity_path),
                'class_embedding_path':
                str(res.attribution_paths.class_embedding_path),
            })
        else:
            row.update({
                'attribution_path': None,
                'attribution_neg_path': None,
                'ffn_activity_path': None,
                'class_embedding_path': None
            })
        flat_results_list.append(row)
    df = pd.DataFrame.from_records(flat_results_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
