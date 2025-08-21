from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

Modes = Literal["train", "test", "val", "dev"]


@dataclass
class FileConfig:
    """Configuration for file paths and I/O operations."""
    dataset_name: str = "hyperkvasir"
    base_pipeline_dir: Path = field(default_factory=lambda: Path("./data/hyperkvasir_unified/results"))
    current_mode: Modes = "test"
    weighted = False

    output_suffix: str = ""

    # Caching behavior
    use_cached_original: bool = True
    use_cached_perturbed: str = ""

    @property
    def mode_dir(self) -> Path:
        return self.base_pipeline_dir / self.current_mode

    @property
    def output_dir(self) -> Path:
        weighted_suffix = "_weighted" if self.weighted else ""
        return self.base_pipeline_dir / f'{self.current_mode}{weighted_suffix}'

    @property
    def data_dir(self) -> Path:
        return self.mode_dir / "preprocessed"

    @property
    def cache_dir(self) -> Path:
        """Cache directory specific to this mode."""
        return self.output_dir / "cache"

    @property
    def attribution_dir(self) -> Path:
        # Suffix application is now more explicit for filenames/sub-subdirs if needed
        return self.output_dir / f"attributions{self.output_suffix}"

    @property
    def vit_inputs_dir(self) -> Path:
        return self.mode_dir / "vit_inputs"

    @property
    def perturbed_dir(self) -> Path:
        return self.mode_dir / "patches"  # Patches of perturbed images

    @property
    def mask_dir(self) -> Path:
        return self.mode_dir / "patch_masks"

    @property
    def directories(self) -> List[Path]:
        """Return a list of all directories used by the pipeline."""
        return [
            self.output_dir, self.data_dir, self.cache_dir, self.attribution_dir, self.vit_inputs_dir,
            self.perturbed_dir, self.mask_dir, self.mode_dir
        ]

    @property
    def directory_map(self) -> Dict[str, Path]:
        """Return a map of all directories."""
        return {
            "output_dir": self.output_dir,
            "mode_dir": self.mode_dir,
            "data": self.data_dir,
            "cache": self.cache_dir,
            "attribution": self.attribution_dir,
            "vit_inputs": self.vit_inputs_dir,
            "perturbed": self.perturbed_dir,
            "masks": self.mask_dir
        }

    def set_dataset(self, dataset_name: str):
        """Update the dataset name and base pipeline directory."""
        self.dataset_name = dataset_name
        self.base_pipeline_dir = Path(f"./data/{dataset_name}_unified/results")


@dataclass
class BoostingConfig:
    """Configuration for SAE-based feature boosting/suppression."""
    # Strength parameters (constant multipliers, no adaptation)
    boost_strength: float = 5.0  # Multiply attribution by this for boost features
    suppress_strength: float = 5.0  # Divide attribution by this for suppress features

    # Selection limits
    max_boost: int = 10
    max_suppress: int = 10

    # Frequency filtering
    min_occurrences: int = 1
    max_occurrences: int = 10000000

    # Ratio thresholds
    min_log_ratio: float = 0.

    # Activation threshold
    min_activation: float = 0.1

    # Top-k filtering
    topk_active: Optional[int] = None

    # Weighting strategy
    use_balanced_score: bool = True

    # Selection method: 'saco', 'topk_activation', 'random'
    selection_method: str = 'saco'

    # Class-aware corrections
    class_aware: bool = False

    # Random seed (for reproducibility)
    random_seed: int = 42

    # SAE layers to apply steering on
    steering_layers: List[int] = field(default_factory=lambda: [6])


@dataclass
class ClassificationConfig:
    """Configuration for image classification and explanation."""
    # Preprocessing parameters
    target_size: Tuple[int, int] = (224, 224)

    # Model parameters
    model_type: str = "vit_base_patch16_224"
    num_classes: int = 3
    
    # CLIP-specific parameters
    use_clip: bool = False
    clip_model_name: str = "openai/clip-vit-base-patch16"
    clip_text_prompts: Optional[List[str]] = None  # If None, uses dataset defaults

    attribution_method: str = "transmm"
    analysis = False
    data_collection = False

    percentile_threshold = 80
    attention_threshold = 30
    top_k_features = 20
    base_strength = 1.5

    # Device configuration
    device: Optional[str] = None  # None will use CUDA if available

    # Boosting configuration
    boosting: BoostingConfig = field(default_factory=BoostingConfig)


@dataclass
class PerturbationConfig:
    """Configuration for image perturbation."""
    # Patch parameters
    patch_size: int = 16

    # Perturbation method
    method: str = "mean"  # "mean" or "sd"

    # Processing limits
    max_images: Optional[int] = None  # None means process all images


@dataclass
class PipelineConfig:
    """Master configuration for the medical image analysis pipeline."""
    # Component configurations
    file: FileConfig = field(default_factory=FileConfig)
    classify: ClassificationConfig = field(default_factory=ClassificationConfig)
    perturb: PerturbationConfig = field(default_factory=PerturbationConfig)

    @property
    def directories(self) -> List[Path]:
        """Return a list of all directories used by the pipeline."""
        return self.file.directories

    @property
    def directory_map(self) -> Dict[str, Path]:
        """Return a map of all directories."""
        return self.file.directory_map
