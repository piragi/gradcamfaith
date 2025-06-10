from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from config import PipelineConfig


@dataclass
class ClassificationPrediction:
    predicted_class_label: str
    predicted_class_idx: int
    confidence: float
    probabilities: List[float]


@dataclass
class FFNActivityItem:
    layer: Union[int, str]
    mean_activity: float
    cls_activity: float
    activity_data: np.ndarray

    def __eq__(self, other):
        if not isinstance(other, FFNActivityItem):
            return NotImplemented
        return (
            self.layer == other.layer and np.isclose(self.mean_activity, other.mean_activity) and
            np.isclose(self.cls_activity, other.cls_activity) and
            np.array_equal(self.activity_data, other.activity_data)
        )


@dataclass
class HeadContributionItem:
    layer: Union[int, str]
    stacked_contribution: np.ndarray

    def __eq__(self, other):
        if not isinstance(other, HeadContributionItem):
            return NotImplemented
        return (self.layer == other.layer and np.array_equal(self.stacked_contribution, other.stacked_contribution))


@dataclass
class ClassEmbeddingRepresentationItem:
    layer: Union[int, str]
    attention_class_representation_input: np.ndarray
    mlp_class_representation_input: np.ndarray
    attention_class_representation_output: np.ndarray
    mlp_class_representation_output: np.ndarray
    attention_map: np.ndarray

    def __eq__(self, other):
        if not isinstance(other, ClassEmbeddingRepresentationItem):
            return NotImplemented
        return (
            self.layer == other.layer and
            np.array_equal(self.attention_class_representation_input, other.attention_class_representation_input) and
            np.array_equal(self.mlp_class_representation_input, other.mlp_class_representation_input) and
            np.array_equal(self.attention_class_representation_output, other.attention_class_representation_output) and
            np.array_equal(self.mlp_class_representation_output, other.mlp_class_representation_output) and
            np.array_equal(self.attention_map, other.attention_map)
        )


@dataclass
class AttributionDataBundle:
    positive_attribution: np.ndarray
    logits: Optional[np.ndarray]  # Can be None
    ffn_activities: List[FFNActivityItem]
    class_embedding_representations: List[ClassEmbeddingRepresentationItem]
    head_contribution: List[HeadContributionItem]

    def __eq__(self, other):
        if not isinstance(other, AttributionDataBundle):
            return NotImplemented

        logits_equal = False
        if self.logits is None and other.logits is None:
            logits_equal = True
        elif self.logits is not None and other.logits is not None:
            logits_equal = np.array_equal(self.logits, other.logits)
        else:  # one is None, other is not
            logits_equal = False

        return (
            np.array_equal(self.positive_attribution, other.positive_attribution) and logits_equal and
            self.ffn_activities == other.ffn_activities and  # Relies on FFNActivityItem.__eq__
            self.class_embedding_representations == other.class_embedding_representations
            and self.head_contribution == other.head_contribution
        )  # Relies on ClassEmbeddingRepresentationItem.__eq__


@dataclass
class AttributionOutputPaths:
    attribution_path: Path
    logits: Path  # Path to file which might contain an empty array
    ffn_activity_path: Path
    class_embedding_path: Path
    head_contribution_path: Path

    def load_head_contributions(self) -> List[Dict[str, Any]]:
        """Load head contribution data from .npz file in expected format"""
        if not self.head_contribution_path.exists():
            raise FileNotFoundError(f"Head contribution file not found: {self.head_contribution_path}")

        with np.load(self.head_contribution_path) as data:
            stacked_contributions = data['contributions']  # [num_layers, num_heads, batch_size, num_tokens, head_dim]
            layer_indices = data['layer_indices']  # [num_layers]

        # Convert back to expected format
        head_contributions = []
        for i, layer_idx in enumerate(layer_indices):
            head_contributions.append({'layer': int(layer_idx), 'activity_data': stacked_contributions[i]})
        return head_contributions


@dataclass
class ClassificationResult:
    image_path: Path
    prediction: ClassificationPrediction
    attribution_paths: Optional[AttributionOutputPaths] = None

    def to_dict_for_cache(self) -> Dict[str, Any]:
        data = asdict(self)
        data['image_path'] = str(data['image_path'])
        if data.get('attribution_paths'
                    ) and isinstance(data['attribution_paths'], dict):  # It will be a dict after asdict
            attrs = data['attribution_paths']
            for k_attr, v_attr in attrs.items():
                if isinstance(v_attr, Path):  # Should always be Path as per AttributionOutputPaths types
                    attrs[k_attr] = str(v_attr)
        return data

    @classmethod
    def from_dict_for_cache(cls, data: Dict[str, Any]) -> 'ClassificationResult':
        data['image_path'] = Path(data['image_path'])

        # Explicitly reconstruct nested dataclasses
        if 'prediction' in data and isinstance(data['prediction'], dict):
            data['prediction'] = ClassificationPrediction(**data['prediction'])

        if 'attribution_paths' in data and data['attribution_paths'] is not None:
            if isinstance(data['attribution_paths'], dict):
                attrs_data = data['attribution_paths']
                # Convert string paths back to Path objects before constructing AttributionOutputPaths
                converted_attrs_data = {k: Path(v) if isinstance(v, str) else v for k, v in attrs_data.items()}
                data['attribution_paths'] = AttributionOutputPaths(**converted_attrs_data)
            else:
                raise ValueError("attribution_paths in cache is not a dict or None")
        elif 'attribution_paths' not in data:  # if key is missing, treat as None
            data['attribution_paths'] = None

        return cls(**data)


@dataclass
class PerturbationPatchInfo:
    patch_id: int
    x: int
    y: int


@dataclass
class PerturbedImageRecord:
    original_image_path: Path
    perturbed_image_path: Path
    mask_path: Path
    patch_info: PerturbationPatchInfo
    perturbation_method: str
    perturbation_strength: Optional[float] = None


@dataclass
class LoadedAttributionData:
    positive_attribution: Optional[np.ndarray] = None

    @classmethod
    def from_positive_attribution_path(cls, path: Optional[Path]) -> 'LoadedAttributionData':
        if path is None or not path.exists():
            return cls()
        return cls(positive_attribution=np.load(path))


@dataclass
class AnalysisContext:
    """Holds all data for an analysis session, focused on SaCo for now."""
    config: PipelineConfig
    original_results: List[ClassificationResult]

    all_perturbed_records: List[PerturbedImageRecord]
    perturbed_classification_results_map: Dict[Path, ClassificationResult]

    @classmethod
    def build(
        cls, config: PipelineConfig, original_results: List[ClassificationResult],
        all_perturbed_records: List[PerturbedImageRecord], perturbed_classification_results: List[ClassificationResult]
    ) -> 'AnalysisContext':

        perturbed_results_map = {p_res.image_path: p_res for p_res in perturbed_classification_results}
        return cls(
            config=config,
            original_results=original_results,
            all_perturbed_records=all_perturbed_records,
            perturbed_classification_results_map=perturbed_results_map
        )
