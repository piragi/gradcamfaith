"""
Unified Pipeline Module

This is the updated pipeline that uses the unified dataloader system.
It can work with any dataset that has been converted to the standard format.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

import io_utils
import perturbation
import visualization
import vit.preprocessing as preprocessing
from config import FileConfig, PipelineConfig
from data_types import (
    AttributionDataBundle, AttributionOutputPaths, ClassEmbeddingRepresentationItem, ClassificationPrediction,
    ClassificationResult, FFNActivityItem, HeadContributionItem, PerturbationPatchInfo, PerturbedImageRecord
)
from faithfulness import evaluate_and_report_faithfulness
from translrp.ViT_new import VisionTransformer
from transmm_sfaf import generate_attribution_prisma, load_steering_resources
from attribution_binning import run_binned_attribution_analysis

# New imports for unified system
from dataset_config import DatasetConfig, get_dataset_config
from dataset_converters import convert_dataset
from unified_dataloader import UnifiedMedicalDataset, create_dataloader, get_single_image_loader


def load_model_for_dataset(dataset_config: DatasetConfig, device: torch.device):
    """
    Load the appropriate model for a given dataset configuration.
    
    Args:
        dataset_config: Configuration for the dataset
        device: Device to load the model on
        
    Returns:
        Loaded model with appropriate head size
    """
    from vit_prisma.models.base_vit import HookedSAEViT
    from vit_prisma.models.weight_conversion import convert_timm_weights
    
    # Create model with correct number of classes (don't load ImageNet weights)
    model = HookedSAEViT.from_pretrained("vit_base_patch16_224", load_pretrained_model=False)
    model.head = torch.nn.Linear(model.cfg.d_model, dataset_config.num_classes)
    
    # Load checkpoint if available
    checkpoint_path = Path(dataset_config.model_checkpoint)
    if checkpoint_path.exists():
        print(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict'].copy()
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict'].copy()
        else:
            state_dict = checkpoint
        
        # Rename linear head if needed
        if 'lin_head.weight' in state_dict:
            state_dict['head.weight'] = state_dict.pop('lin_head.weight')
        if 'lin_head.bias' in state_dict:
            state_dict['head.bias'] = state_dict.pop('lin_head.bias')
        
        # Convert weights to correct format
        converted_weights = convert_timm_weights(state_dict, model.cfg)
        model.load_state_dict(converted_weights)
    else:
        print(f"Warning: Model checkpoint not found at {checkpoint_path}, using random initialization")
    
    model.to(device).eval()
    return model


def prepare_dataset_if_needed(
    dataset_name: str,
    source_path: Path,
    prepared_path: Path,
    force_prepare: bool = False,
    **converter_kwargs
) -> Path:
    """
    Prepare dataset if not already prepared.
    
    Args:
        dataset_name: Name of the dataset
        source_path: Path to raw dataset
        prepared_path: Path where prepared dataset should be
        force_prepare: If True, force re-preparation even if exists
        **converter_kwargs: Additional arguments for converter
        
    Returns:
        Path to prepared dataset
    """
    metadata_file = prepared_path / "dataset_metadata.json"
    
    if not force_prepare and metadata_file.exists():
        print(f"Dataset already prepared at {prepared_path}")
        return prepared_path
    
    print(f"Preparing {dataset_name} dataset...")
    print("Images will be preprocessed to 224x224")
    convert_dataset(
        dataset_name=dataset_name,
        source_path=source_path,
        output_path=prepared_path,
        **converter_kwargs
    )
    
    return prepared_path


def classify_single_image(
    config: PipelineConfig,
    dataset_config: DatasetConfig,
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    true_label: Optional[str] = None
) -> ClassificationResult:
    """
    Classify a single image using the unified system.
    """
    cache_path = io_utils.build_cache_path(
        config.file.cache_dir, image_path, f"_classification_{dataset_config.name}.json"
    )
    
    # Try to load from cache
    loaded_result = io_utils.try_load_from_cache(cache_path)
    if config.file.use_cached_perturbed and loaded_result:
        return loaded_result
    
    # Load and preprocess image
    input_tensor = get_single_image_loader(image_path, dataset_config)
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_idx = torch.argmax(probabilities, dim=-1).item()
    
    current_prediction = ClassificationPrediction(
        predicted_class_label=dataset_config.idx_to_class[predicted_idx],
        predicted_class_idx=predicted_idx,
        confidence=float(probabilities[0, predicted_idx].item()),
        probabilities=probabilities[0].tolist()
    )
    
    result = ClassificationResult(
        image_path=image_path,
        prediction=current_prediction,
        true_label=true_label,
        attribution_paths=None
    )
    
    # Cache the result
    io_utils.save_to_cache(cache_path, result)
    
    return result


def save_attribution_bundle_to_files(
    image_stem: str,
    attribution_bundle: AttributionDataBundle,
    file_config: FileConfig
) -> AttributionOutputPaths:
    """Save attribution bundle contents to .npy files."""
    
    # Ensure attribution directory exists
    io_utils.ensure_directories([file_config.attribution_dir])
    
    attribution_path = file_config.attribution_dir / f"{image_stem}_attribution.npy"
    raw_attribution_path = file_config.attribution_dir / f"{image_stem}_raw_attribution.npy"
    logits_path = Path("")
    ffn_activity_path = Path("")
    class_embedding_path = Path("")
    head_contribution_path = Path("")
    
    # Save positive attribution
    np.save(attribution_path, attribution_bundle.positive_attribution)
    # Save raw attribution
    np.save(raw_attribution_path, attribution_bundle.raw_attribution)
    
    # Save logits if available
    if attribution_bundle.logits is not None:
        logits_path = file_config.attribution_dir / f"{image_stem}_logits.npy"
        np.save(logits_path, attribution_bundle.logits)
    
    # Save FFN activities
    if attribution_bundle.ffn_activities:
        ffn_activity_path = file_config.attribution_dir / f"{image_stem}_ffn_activity.npy"
        ffn_data_to_save = []
        for item in attribution_bundle.ffn_activities:
            ffn_data_to_save.append({
                'layer': item.layer,
                'mean_activity': item.mean_activity,
                'cls_activity': item.cls_activity,
                'activity_data': item.activity_data
            })
        np.save(ffn_activity_path, np.array(ffn_data_to_save, dtype=object))
    
    # Save head contributions
    if attribution_bundle.head_contribution:
        head_contribution_path = file_config.attribution_dir / f"{image_stem}_head_contribution.npz"
        
        layers = []
        layer_indices = []
        
        for item in attribution_bundle.head_contribution:
            layers.append(item.stacked_contribution)
            layer_indices.append(item.layer)
        
        stacked_contributions = np.stack(layers, axis=0)
        layer_indices = np.array(layer_indices)
        
        np.savez(head_contribution_path, contributions=stacked_contributions, layer_indices=layer_indices)
    
    # Save class embedding representations
    if attribution_bundle.class_embedding_representations:
        class_embedding_path = file_config.attribution_dir / f"{image_stem}_class_embedding_representation.npy"
        class_embedding_data_to_save = []
        for item in attribution_bundle.class_embedding_representations:
            class_embedding_data_to_save.append({
                'layer': item.layer,
                'attention_class_representation_input': item.attention_class_representation_input,
                'mlp_class_representation_input': item.mlp_class_representation_input,
                'attention_class_representation_output': item.attention_class_representation_output,
                'attention_map': item.attention_map,
                'mlp_class_representation_output': item.mlp_class_representation_output
            })
        np.save(class_embedding_path, np.array(class_embedding_data_to_save, dtype=object))
    
    return AttributionOutputPaths(
        attribution_path=attribution_path,
        raw_attribution_path=raw_attribution_path,
        logits=logits_path,
        ffn_activity_path=ffn_activity_path,
        class_embedding_path=class_embedding_path,
        head_contribution_path=head_contribution_path
    )


def classify_explain_single_image(
    config: PipelineConfig,
    dataset_config: DatasetConfig,
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    steering_resources: Optional[Dict[int, Dict[str, Any]]],
    true_label: Optional[str] = None,
) -> ClassificationResult:
    """
    Classify a single image AND generate explanations using unified system.
    """
    cache_path = io_utils.build_cache_path(
        config.file.cache_dir, image_path, f"_classification_explained_{dataset_config.name}.json"
    )
    
    # Try to load from cache
    loaded_result = io_utils.try_load_from_cache(cache_path)
    if config.file.use_cached_original and loaded_result:
        if loaded_result.attribution_paths is not None:
            return loaded_result
    
    # Load and preprocess image
    input_tensor = get_single_image_loader(image_path, dataset_config)
    input_tensor = input_tensor.to(device)
    
    # Generate attribution
    raw_attribution_result_dict = generate_attribution_prisma(
        model=model,
        input_tensor=input_tensor,
        config=config,
        idx_to_class=dataset_config.idx_to_class,  # Pass dataset-specific class mapping
        device=device,
        steering_resources=steering_resources,
        enable_steering=config.file.weighted,
    )
    
    # Extract raw attribution
    raw_attr = raw_attribution_result_dict.get("raw_attribution", np.array([]))
    
    # Create prediction
    prediction_data = raw_attribution_result_dict["predictions"]
    current_prediction = ClassificationPrediction(
        predicted_class_label=dataset_config.idx_to_class[prediction_data["predicted_class_idx"]],
        predicted_class_idx=prediction_data["predicted_class_idx"],
        confidence=float(prediction_data["probabilities"][prediction_data["predicted_class_idx"]]),
        probabilities=prediction_data["probabilities"]
    )
    
    # Process FFN activities
    ffn_activity_items: List[FFNActivityItem] = []
    if "ffn_activity" in raw_attribution_result_dict and raw_attribution_result_dict["ffn_activity"]:
        for ffn_dict in raw_attribution_result_dict["ffn_activity"]:
            ffn_activity_items.append(
                FFNActivityItem(
                    layer=ffn_dict["layer"],
                    mean_activity=ffn_dict["mean_activity"],
                    cls_activity=ffn_dict["cls_activity"],
                    activity_data=ffn_dict["activity"]
                )
            )
    
    # Process class embeddings
    class_embedding_items: List[ClassEmbeddingRepresentationItem] = []
    if "class_embedding_representation" in raw_attribution_result_dict and raw_attribution_result_dict["class_embedding_representation"]:
        for cer_dict in raw_attribution_result_dict["class_embedding_representation"]:
            class_embedding_items.append(
                ClassEmbeddingRepresentationItem(
                    layer=cer_dict["layer"],
                    attention_class_representation_output=cer_dict["attention_class_representation_output"],
                    mlp_class_representation_output=cer_dict["mlp_class_representation_output"],
                    attention_class_representation_input=cer_dict["attention_class_representation_input"],
                    attention_map=cer_dict["attention_map"],
                    mlp_class_representation_input=cer_dict["mlp_class_representation_input"]
                )
            )
    
    # Process head contributions
    head_contribution_items: List[HeadContributionItem] = []
    if "head_contribution" in raw_attribution_result_dict and raw_attribution_result_dict["head_contribution"]:
        for head_contribution_dict in raw_attribution_result_dict["head_contribution"]:
            head_contribution_items.append(
                HeadContributionItem(
                    layer=head_contribution_dict["layer"],
                    stacked_contribution=head_contribution_dict["stacked_contribution"]
                )
            )
    
    # Create attribution bundle
    attribution_bundle = AttributionDataBundle(
        positive_attribution=raw_attribution_result_dict["attribution_positive"],
        raw_attribution=raw_attr,
        logits=raw_attribution_result_dict.get("logits"),
        ffn_activities=ffn_activity_items,
        class_embedding_representations=class_embedding_items,
        head_contribution=head_contribution_items,
    )
    
    # Save attribution bundle
    saved_attribution_paths = save_attribution_bundle_to_files(
        image_path.stem, attribution_bundle, config.file
    )
    
    # Create final result
    final_result = ClassificationResult(
        image_path=image_path,
        prediction=current_prediction,
        true_label=true_label,
        attribution_paths=saved_attribution_paths
    )
    
    # Cache the result
    io_utils.save_to_cache(cache_path, final_result)
    
    return final_result


def run_unified_pipeline(
    config: PipelineConfig,
    dataset_name: str,
    source_data_path: Path,
    prepared_data_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    force_prepare: bool = False
) -> List[ClassificationResult]:
    """
    Run the unified pipeline for any supported dataset.
    
    Args:
        config: Pipeline configuration
        dataset_name: Name of the dataset ('covidquex' or 'hyperkvasir')
        source_data_path: Path to the source dataset
        prepared_data_path: Path for prepared data (default: ./data/{dataset_name}_unified)
        device: Device to use
        force_prepare: Force re-preparation of dataset
        
    Returns:
        List of classification results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    print(f"Loading configuration for {dataset_name} dataset")
    print(f"  Classes: {dataset_config.num_classes} - {dataset_config.class_names}")
    
    # Prepare dataset if needed
    if prepared_data_path is None:
        prepared_data_path = Path(f"./data/{dataset_name}_unified")
    
    # Prepare converter kwargs based on dataset
    converter_kwargs = {}
    
    prepared_path = prepare_dataset_if_needed(
        dataset_name=dataset_name,
        source_path=source_data_path,
        prepared_path=prepared_data_path,
        force_prepare=force_prepare,
        **converter_kwargs
    )
    
    # Create dataloader
    print(f"Creating dataloader from {prepared_path}")
    dataset_loader = create_dataloader(
        dataset_name=dataset_name,
        data_path=prepared_path,
        batch_size=config.classify.batch_size if hasattr(config.classify, 'batch_size') else 32
    )
    
    # Print dataset statistics
    stats = dataset_loader.get_statistics()
    for split, split_stats in stats['splits'].items():
        print(f"\n{split.upper()}: {split_stats['total_samples']} samples")
        for class_name, count in split_stats['class_distribution'].items():
            print(f"  {class_name}: {count}")
    
    # Load model
    print(f"\nLoading model for {dataset_name}")
    model = load_model_for_dataset(dataset_config, device)
    
    # Load steering resources if needed
    steering_layers = getattr(config.classify, 'steering_layers', [6])
    steering_resources = load_steering_resources(steering_layers)
    print("Steering resources loaded")
    
    # Process images from the specified split (mode)
    split_to_use = config.file.current_mode if config.file.current_mode in ['train', 'val', 'test', 'dev'] else 'test'
    
    split_dataset = dataset_loader.get_dataset(split_to_use)
    # Get both image paths and their true labels
    image_data = [(Path(img_path), label_idx) for img_path, label_idx in split_dataset.samples]
    
    print(f"\nProcessing {len(image_data)} {split_to_use} images")
    
    # Classify and explain
    results = []
    for image_path, true_label_idx in tqdm(image_data, desc="Classifying & Explaining"):
        try:
            # Convert label index to label name
            true_label = dataset_config.idx_to_class[true_label_idx]
            
            result = classify_explain_single_image(
                config=config,
                dataset_config=dataset_config,
                image_path=image_path,
                model=model,
                device=device,
                steering_resources=steering_resources,
                true_label=true_label
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
    
    # Save results
    if results:
        csv_path = config.file.output_dir / f"results_{dataset_name}_unified.csv"
        io_utils.save_classification_results_to_csv(results, csv_path)
        print(f"Results saved to {csv_path}")
    
    # Run faithfulness evaluation if configured
    if config.classify.analysis:
        print("Running faithfulness evaluation...")
        try:
            evaluate_and_report_faithfulness(config, model, device, results)
        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}")
    
    # Run attribution analysis
    print("Running attribution analysis...")
    run_binned_attribution_analysis(config, model, results, device, n_bins=49)
    
    print("\nPipeline complete!")
    return results


# Example usage
if __name__ == "__main__":
    from config import PipelineConfig
    
    # Example: Run pipeline for CovidQUEX
    config = PipelineConfig()
    
    results = run_unified_pipeline(
        config=config,
        dataset_name="covidquex",
        source_data_path=Path("./COVID-QU-Ex/"),
        force_prepare=False  # Set to True to force re-preparation
    )
    
    print(f"\nProcessed {len(results)} images successfully")