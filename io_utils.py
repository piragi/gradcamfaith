import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def ensure_directories(directories: List[Path]) -> None:
    """Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)


def try_load_from_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
    """Try to load cached results from a file.
    
    Args:
        cache_path: Path to the cache file
        
    Returns:
        Cached data as a dictionary if successful, None otherwise
    """
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def save_to_cache(cache_path: Path, result: Dict[str, Any]) -> None:
    """Save results to cache.
    
    Args:
        cache_path: Path to the cache file
        result: Data to cache
    """
    with open(cache_path, 'w') as f:
        json.dump(result, f)


def build_cache_path(cache_dir: Path, image_path: Path, suffix: str, prefix: str = "") -> Path:
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
