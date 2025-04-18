from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict


@dataclass
class FileConfig:
    """Configuration for file paths and I/O operations."""
    # Base directories
    data_dir: Path = Path("./images")
    output_dir: Path = Path("./results")
    cache_dir: Path = Path("./cache")
    
    # Output suffix for mean/sd perturbation
    output_suffix: str = ""
    
    # Caching behavior
    use_cached: bool = True
    
    # Output subdirectories - computed after initialization
    attribution_dir: Path = field(init=False)
    vit_inputs_dir: Path = field(init=False)
    perturbed_dir: Path = field(init=False)
    mask_dir: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.attribution_dir = self.output_dir / f"attributions{self.output_suffix}"
        self.vit_inputs_dir = self.output_dir / "vit_inputs"
        self.perturbed_dir = self.output_dir / "patches"
        self.mask_dir = self.output_dir / "patch_masks"
    
    @property
    def directories(self) -> List[Path]:
        """Return a list of all directories used by the pipeline."""
        return [
            self.data_dir,
            self.output_dir,
            self.cache_dir,
            self.attribution_dir,
            self.vit_inputs_dir,
            self.perturbed_dir,
            self.mask_dir
        ]
    
    @property
    def directory_map(self) -> Dict[str, Path]:
        """Return a map of all directories."""
        return {
            "data": self.data_dir,
            "output": self.output_dir,
            "cache": self.cache_dir,
            "attribution": self.attribution_dir,
            "vit_inputs": self.vit_inputs_dir,
            "perturbed": self.perturbed_dir,
            "masks": self.mask_dir
        }


@dataclass
class ClassificationConfig:
    """Configuration for image classification and explanation."""
    # Preprocessing parameters
    target_size: Tuple[int, int] = (224, 224)
    
    # Model parameters
    model_type: str = "vit_base_patch16_224"
    num_classes: int = 2
    
    # Explanation parameters
    pretransform: bool = False
    gini_params: Tuple[float, float, float] = (0.65, 8.0, 0.5)
    attribution_method: str = "transmm"
    
    # Device configuration
    device: Optional[str] = None  # None will use CUDA if available


@dataclass
class PerturbationConfig:
    """Configuration for image perturbation."""
    # Patch parameters
    patch_size: int = 16
    
    # Perturbation method
    method: str = "mean"  # "mean" or "sd"
    
    # Stable Diffusion parameters
    strength: float = 0.2
    prompt: str = "normal healthy tissue"
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    
    # Processing limits
    max_images: Optional[int] = None  # None means process all images


@dataclass
class PipelineConfig:
    """Master configuration for the medical image analysis pipeline."""
    # Component configurations
    file_config: FileConfig = field(default_factory=FileConfig)
    classification_config: ClassificationConfig = field(default_factory=ClassificationConfig)
    perturbation_config: PerturbationConfig = field(default_factory=PerturbationConfig)
    
    @property
    def directories(self) -> List[Path]:
        """Return a list of all directories used by the pipeline."""
        return self.file_config.directories
    
    @property
    def directory_map(self) -> Dict[str, Path]:
        """Return a map of all directories."""
        return self.file_config.directory_map
