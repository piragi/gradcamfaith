from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

Modes = Literal["train", "test", "val", "dev"]


@dataclass
class FileConfig:
    """Configuration for file paths and I/O operations."""
    base_pipeline_dir: Path = Path("./results")
    current_mode: Modes = "test"
    weighted = False

    output_suffix: str = ""

    # Caching behavior
    use_cached_original: bool = True
    use_cached_perturbed: bool = True

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


@dataclass
class ClassificationConfig:
    """Configuration for image classification and explanation."""
    # Preprocessing parameters
    target_size: Tuple[int, int] = (224, 224)

    # Model parameters
    model_type: str = "vit_base_patch16_224"
    num_classes: int = 3

    # Explanation parameters
    gini_params: Tuple[float, float, float] = (0.65, 8.0, 0.5)
    adaptive_weighting: Tuple[float, float] = (1.2, 1.7)
    head_boost_value: float = 200.0
    # Dict for head-specific boosting per class per layer
    # {class: {layer: head}}
    head_boost_factor_per_head_per_class: Dict[int, Dict[int, List[int]]] = field(
        default_factory=lambda: {
            0: {
                8: [1],
            },
            2: {},
            3: {},
            4: {},
            5: {},
        }
    )
    token_boost_value: float = 100.0
    token_boost_factors: Dict[int, Dict[int, Dict[int, List[int]]]] = field(
        default_factory=lambda: {
            0: {
                8: {
                    1: [8, 64, 7, 99]
                }
            },
            2: {
                8: {
                    5: [26, 41, 24, 11, 25]
                }
            },
            3: {},
            4: {},
            5: {
                10: {
                    1: [14, 1, 0, 70, 71]
                }
            }
        }
    )
    class_boost_multipliers = {
        0: 0.6,  # Moderate (high correlation but still distinct)
        2: 1.0,  # Keep working approach  
        # 3: 0.4,  # Low (conflicting strategies)
        # 4: 0.0,  # Moderate
        5: 1.2  # High (false multi-strategy, actually coherent)
    }
    attribution_method: str = "transmm"
    analysis = False
    data_collection = False

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
