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

    output_suffix: str = ""

    # Caching behavior
    use_cached_original: bool = True
    use_cached_perturbed: str = ""

    @property
    def mode_dir(self) -> Path:
        return self.base_pipeline_dir / self.current_mode

    @property
    def output_dir(self) -> Path:
        return self.base_pipeline_dir / self.current_mode

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
    # Random seed (for reproducibility)
    random_seed: int = 42

    # SAE layers to apply steering on
    steering_layers: List[int] = field(default_factory=lambda: [6])

    # Feature gradient gating
    enable_feature_gradients: bool = False  # Enable feature gradient gating
    feature_gradient_layers: List[int] = field(default_factory=lambda: [9, 10])  # Which layers to apply
    kappa: float = 10.0  # Scaling factor for exponential gate mapping
    clamp_min: float = 0.1  # Minimum gate value (1/clamp_max)
    clamp_max: float = 10.0  # Maximum gate value (range: [clamp_min, clamp_max])
    gate_construction: str = "combined"  # Type of gate: "activation_only", "gradient_only", or "combined"
    shuffle_decoder: bool = False  # Shuffle decoder columns to break semantic alignment
    shuffle_decoder_seed: int = 12345  # Random seed for decoder shuffling (reproducibility)

    # Debug mode - collect detailed feature data
    debug_mode: bool = False  # If True, collect sparse features, gradients, and gate values per image
    active_feature_threshold: float = 0.1  # Threshold for considering a feature "active" in debug mode


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

    analysis = False

    # Device configuration
    device: Optional[str] = None  # None will use CUDA if available

    # Boosting configuration
    boosting: BoostingConfig = field(default_factory=BoostingConfig)


@dataclass
class PerturbationConfig:
    """Configuration for image perturbation."""
    # Patch parameters
    patch_size: int = 16

    # TODO: there is only mean as a perturbation method
    # Perturbation method
    method: str = "mean"

    # Processing limits
    max_images: Optional[int] = None  # None means process all images


@dataclass
class FaithfulnessConfig:
    """Configuration for faithfulness evaluation metrics (includes SaCo and correlation metrics)."""
    # === Statistical robustness ===
    n_trials: int = 3  # Number of trials for statistical robustness

    # === Faithfulness correlation parameters ===
    nr_runs: int = 20  # Number of random perturbations per image
    subset_size: int = 20  # Size of feature subset to perturb for correlation (adaptive to n_patches)
    subset_size_b32: int = 10  # Size of feature subset to perturb for B-32 models

    # === Pixel flipping parameters ===
    features_in_step: int = 1  # Number of patches to perturb at each step

    # === SaCo (binned attribution analysis) parameters ===
    n_bins: int = 49  # Number of attribution bins (for B-16 models with 196 patches)
    n_bins_b32: int = 13  # Number of attribution bins (for B-32 models with 49 patches)
    saco_inference_batch_size: int = 32  # Batch size for model inference during SaCo perturbation

    # === Perturbation settings (shared across all metrics) ===
    perturb_baseline: str = "mean"  # Baseline for perturbation ("black", "white", "mean", etc.)

    # === GPU batching ===
    gpu_batch_size: int = 1024  # Batch size for GPU forward passes


@dataclass
class PipelineConfig:
    """Master configuration for the medical image analysis pipeline."""
    # Component configurations
    file: FileConfig = field(default_factory=FileConfig)
    classify: ClassificationConfig = field(default_factory=ClassificationConfig)
    perturb: PerturbationConfig = field(default_factory=PerturbationConfig)
    faithfulness: FaithfulnessConfig = field(default_factory=FaithfulnessConfig)

    @property
    def directories(self) -> List[Path]:
        """Return a list of all directories used by the pipeline."""
        return self.file.directories

    @property
    def directory_map(self) -> Dict[str, Path]:
        """Return a map of all directories."""
        return self.file.directory_map
