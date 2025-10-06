"""
Unified Pipeline Module

This is the updated pipeline that uses the unified dataloader system.
It can work with any dataset that has been converted to the standard format.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

from vit_prisma.sae import SparseAutoencoder

import io_utils
from config import FileConfig, PipelineConfig
from data_types import (AttributionDataBundle, AttributionOutputPaths, ClassificationPrediction, ClassificationResult)
# New imports for unified system
from dataset_config import DatasetConfig, get_dataset_config
from faithfulness import evaluate_and_report_faithfulness
from saco import run_binned_attribution_analysis
from setup import convert_dataset
from transmm import generate_attribution_prisma_enhanced
from unified_dataloader import create_dataloader, get_single_image_loader


def load_steering_resources(layers: List[int], dataset_name: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """
    Loads SAEs for the specified layers for feature gradient gating.
    
    Args:
        layers: List of layer indices to load
        dataset_name: Name of the dataset ('covidquex', 'hyperkvasir', 'waterbirds', etc.)
    """
    resources = {}

    for layer_idx in layers:
        try:
            if dataset_name in ["waterbirds", "imagenet"]:
                # Use CLIP Vanilla B-32 SAE for waterbirds
                sae_path = Path(f"data/sae_clip_vanilla_b32/layer_{layer_idx}/weights.pt")
                if not sae_path.exists():
                    print(f"Warning: CLIP Vanilla SAE not found at {sae_path}")
                    continue
                print(f"Loading CLIP Vanilla SAE from {sae_path}")
                sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
                sae.cuda().eval()
            else:
                # Load SAE for other datasets
                sae_dir = Path("data") / f"sae_{dataset_name}" / f"layer_{layer_idx}"
                sae_files = list(sae_dir.glob("**/n_images_*.pt"))
                # Filter out log_feature_sparsity files
                sae_files = [f for f in sae_files if 'log_feature_sparsity' not in str(f)]

                if not sae_files:
                    print(f"Warning: No SAE found for {dataset_name} layer {layer_idx} in {sae_dir}")
                    continue

                # Use the most recent SAE file
                sae_path = sorted(sae_files)[-1]
                print(f"Loading SAE from {sae_path}")

                sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
                sae.cuda().eval()

            resources[layer_idx] = {"sae": sae}

        except Exception as e:
            print(f"Error loading SAE for {dataset_name} layer {layer_idx}: {e}")

    return resources


def load_model_for_dataset(
    dataset_config: DatasetConfig, device: torch.device, config: Optional[PipelineConfig] = None
):
    """
    Load the appropriate model and CLIP classifier for a given dataset configuration.

    Args:
        dataset_config: Configuration for the dataset
        device: Device to load the model on
        config: Optional pipeline config for CLIP settings

    Returns:
        Tuple of (model, clip_classifier) where clip_classifier is None for non-CLIP models
    """
    # Check if we should use CLIP for this dataset
    use_clip = (config and config.classify.use_clip) or dataset_config.name == "waterbirds"

    if use_clip:
        from vit_prisma.models.model_loader import load_hooked_model

        print(f"Loading CLIP as HookedViT for {dataset_config.name}")

        # Use vit_prisma's load_hooked_model to get CLIP as HookedViT
        # This automatically converts CLIP weights to HookedViT format
        clip_model_name = config.classify.clip_model_name if config else "openai/clip-vit-base-patch32"

        model = load_hooked_model(clip_model_name, dtype=torch.float32, device=str(device))
        # Ensure model is on the correct device
        model = model.to(device)
        model.eval()

        print(f"CLIP loaded as HookedViT")

        # Create CLIP classifier for this model
        from clip_classifier import create_clip_classifier_for_waterbirds
        print("Creating CLIP classifier...")
        clip_model_name = config.classify.clip_model_name if config else "openai/clip-vit-base-patch32"
        clip_classifier = create_clip_classifier_for_waterbirds(
            vision_model=model,
            device=device,
            clip_model_name=clip_model_name,
            custom_prompts=config.classify.clip_text_prompts if config.classify.clip_text_prompts else None
        )

        return model, clip_classifier

    # Original ViT loading code
    from vit_prisma.models.base_vit import HookedSAEViT
    from vit_prisma.models.weight_conversion import convert_timm_weights

    # Create model with correct number of classes (don't load ImageNet weights)
    model = HookedSAEViT.from_pretrained("vit_base_patch16_224", load_pretrained_model=False)

    # Update the config and recreate the head with the correct number of classes
    model.cfg.n_classes = dataset_config.num_classes
    from vit_prisma.models.layers.head import Head
    model.head = Head(model.cfg)

    # Load checkpoint if available
    checkpoint_path = Path(dataset_config.model_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at '{checkpoint_path}'. ")

    print(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict'].copy()
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict'].copy()
    else:
        state_dict = checkpoint

    # Rename linear head if needed for compatibility
    if 'lin_head.weight' in state_dict:
        state_dict['head.weight'] = state_dict.pop('lin_head.weight')
    if 'lin_head.bias' in state_dict:
        state_dict['head.bias'] = state_dict.pop('lin_head.bias')

    # Convert weights to the correct format for the model and load them
    converted_weights = convert_timm_weights(state_dict, model.cfg)
    model.load_state_dict(converted_weights)

    model.to(device).eval()
    return model, None  # No CLIP classifier for regular ViT models


def prepare_dataset_if_needed(
    dataset_name: str, source_path: Path, prepared_path: Path, force_prepare: bool = False, **converter_kwargs
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
    convert_dataset(dataset_name=dataset_name, source_path=source_path, output_path=prepared_path, **converter_kwargs)

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
    cache_path = io_utils.build_cache_path(config.file.cache_dir, image_path, f"_classification_{dataset_config.name}")

    # Try to load from cache
    loaded_result = io_utils.try_load_from_cache(cache_path)
    if config.file.use_cached_perturbed and loaded_result:
        return loaded_result

    # Load and preprocess image
    # Check if we're using CLIP (config might be None in some cases)
    use_clip = config and config.classify.use_clip
    input_tensor = get_single_image_loader(image_path, dataset_config, use_clip=use_clip)
    input_tensor = input_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_idx = int(torch.argmax(probabilities, dim=-1).item())

    current_prediction = ClassificationPrediction(
        predicted_class_label=dataset_config.idx_to_class[predicted_idx],
        predicted_class_idx=predicted_idx,
        confidence=float(probabilities[0, predicted_idx].item()),
        probabilities=probabilities[0].tolist()
    )

    result = ClassificationResult(
        image_path=image_path, prediction=current_prediction, true_label=true_label, attribution_paths=None
    )

    # Cache the result
    io_utils.save_to_cache(cache_path, result)

    return result


def save_attribution_bundle_to_files(
    image_stem: str, attribution_bundle: AttributionDataBundle, file_config: FileConfig
) -> AttributionOutputPaths:
    """Save attribution bundle contents to .npy files."""

    # Ensure attribution directory exists
    io_utils.ensure_directories([file_config.attribution_dir])

    attribution_path = file_config.attribution_dir / f"{image_stem}_attribution.npy"
    raw_attribution_path = file_config.attribution_dir / f"{image_stem}_raw_attribution.npy"

    # Save positive attribution
    np.save(attribution_path, attribution_bundle.positive_attribution)
    # Save raw attribution
    np.save(raw_attribution_path, attribution_bundle.raw_attribution)

    return AttributionOutputPaths(
        attribution_path=attribution_path,
        raw_attribution_path=raw_attribution_path,
    )


def classify_explain_single_image(
    config: PipelineConfig,
    dataset_config: DatasetConfig,
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    steering_resources: Optional[Dict[int, Dict[str, Any]]],
    true_label: Optional[str] = None,
    clip_classifier: Optional[Any] = None,  # Pre-created CLIP classifier
) -> ClassificationResult:
    """
    Classify a single image AND generate explanations using unified system.
    """
    cache_path = io_utils.build_cache_path(
        config.file.cache_dir, image_path, f"_classification_explained_{dataset_config.name}"
    )

    # Try to load from cache
    loaded_result = io_utils.try_load_from_cache(cache_path)
    if config.file.use_cached_original and loaded_result:
        if loaded_result.attribution_paths is not None:
            return loaded_result

    # Load and preprocess image
    # Check if we're using CLIP (config might be None in some cases)
    use_clip = config and config.classify.use_clip
    input_tensor = get_single_image_loader(image_path, dataset_config, use_clip=use_clip)
    input_tensor = input_tensor.to(device)

    raw_attribution_result_dict = generate_attribution_prisma_enhanced(
        model=model,
        input_tensor=input_tensor,
        config=config,
        idx_to_class=dataset_config.idx_to_class,  # Pass dataset-specific class mapping
        device=device,
        steering_resources=steering_resources,
        enable_feature_gradients=config.classify.boosting.enable_feature_gradients,
        feature_gradient_layers=config.classify.boosting.feature_gradient_layers
        if config.classify.boosting.enable_feature_gradients else [],
        clip_classifier=clip_classifier,
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

    # Create attribution bundle
    attribution_bundle = AttributionDataBundle(
        positive_attribution=raw_attribution_result_dict["attribution_positive"],
        raw_attribution=raw_attr,
    )

    # Save attribution bundle
    saved_attribution_paths = save_attribution_bundle_to_files(image_path.stem, attribution_bundle, config.file)

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
    model: torch.nn.Module,
    steering_resources: Dict[int, Dict[str, Any]],
    clip_classifier: Optional[Any] = None,
    prepared_data_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    force_prepare: bool = False,
    subset_size: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Tuple[List[ClassificationResult], Dict[str, Any]]:
    """
    Run the unified pipeline for any supported dataset.

    Args:
        config: Pipeline configuration
        dataset_name: Name of the dataset ('covidquex' or 'hyperkvasir')
        source_data_path: Path to the source dataset
        model: Pre-loaded model
        steering_resources: Pre-loaded SAE resources
        clip_classifier: Pre-loaded CLIP classifier (None for non-CLIP models)
        prepared_data_path: Path for prepared data (default: ./data/{dataset_name}_unified)
        device: Device to use
        force_prepare: Force re-preparation of dataset
        subset_size: If specified, only use this many random images
        random_seed: Random seed for reproducible subset selection

    Returns:
        List of classification results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    print(f"Loading configuration for {dataset_name} dataset")

    # Ensure all required directories exist
    io_utils.ensure_directories(config.directories)

    # Prepare dataset if needed
    if prepared_data_path is None:
        prepared_data_path = Path(f"./data/{dataset_name}_unified")

    prepared_path = prepare_dataset_if_needed(
        dataset_name=dataset_name,
        source_path=source_data_path,
        prepared_path=prepared_data_path,
        force_prepare=force_prepare
    )

    # Create dataloader
    print(f"Creating dataloader from {prepared_path}")
    # Check if we're using CLIP for this dataset
    use_clip = config.classify.use_clip if hasattr(config.classify, 'use_clip') else False
    dataset_loader = create_dataloader(
        dataset_name=dataset_name,
        data_path=prepared_path
    )

    # Use pre-loaded model and steering resources
    print(f"\nUsing pre-loaded model for {dataset_name}")
    print(f"Using pre-loaded SAE resources for {dataset_name}")

    # Process images from the specified split (mode)
    split_to_use = config.file.current_mode if config.file.current_mode in ['train', 'val', 'test', 'dev'] else 'test'

    image_data = list(dataset_loader.get_numeric_samples(split_to_use))
    total_samples = len(image_data)

    # Apply subset if requested
    if subset_size is not None and subset_size < total_samples:
        import random
        if random_seed is not None:
            random.seed(random_seed)
        image_data = random.sample(image_data, subset_size)
        print(
            f"\nProcessing {len(image_data)} randomly selected {split_to_use} images (subset of {total_samples})"
        )
    else:
        print(f"\nProcessing {total_samples} {split_to_use} images")

    # Use pre-loaded CLIP classifier (created once per dataset)
    if clip_classifier is not None:
        print("Using pre-loaded CLIP classifier")

    # Classify and explain
    results = []

    for _, (image_path, true_label_idx) in enumerate(tqdm(image_data, desc="Classifying & Explaining")):
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
                true_label=true_label,
                clip_classifier=clip_classifier  # Pass pre-created classifier
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

    # For CLIP models, wrap the classifier for both faithfulness and attribution analysis
    model_for_analysis = model
    if clip_classifier is not None:
        from clip_classifier import CLIPModelWrapper
        model_for_analysis = CLIPModelWrapper(clip_classifier)
        print("Using CLIP wrapper for analysis")

    # Run faithfulness evaluation if configured
    if config.classify.analysis:
        print("Running faithfulness evaluation...")
        try:
            evaluate_and_report_faithfulness(
                config, model_for_analysis, device, results, clip_classifier=clip_classifier
            )
        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}")

    # Run attribution analysis
    # For B-32: 49 patches total, use fewer bins (e.g., 20)
    # For B-16: 196 patches total, can use 49 bins
    # Check for B-32 models (both patch32 and B-32 patterns)
    is_patch32 = False
    if hasattr(config.classify, 'clip_model_name') and config.classify.clip_model_name:
        model_name = config.classify.clip_model_name.lower()
        is_patch32 = "patch32" in model_name or "b-32" in model_name or "b32" in model_name
    n_bins = 13 if is_patch32 else 49
    print(f"Running attribution analysis with {n_bins} bins (patch-{'32' if is_patch32 else '16'})...")

    saco_analysis = run_binned_attribution_analysis(config, model_for_analysis, results, device, n_bins=n_bins)

    # Extract SaCo scores (overall and per-class)
    saco_results = {}
    if saco_analysis and "faithfulness_correctness" in saco_analysis:
        fc_df = saco_analysis["faithfulness_correctness"]

        # Overall statistics
        saco_results['mean'] = fc_df["saco_score"].mean()
        saco_results['std'] = fc_df["saco_score"].std()
        saco_results['n_samples'] = len(fc_df)

        # Per-class breakdown
        per_class_stats = fc_df.groupby('true_class')['saco_score'].agg(['mean', 'std', 'count'])
        saco_results['per_class'] = per_class_stats.to_dict('index')

        # Also include correctness breakdown
        correctness_stats = fc_df.groupby('is_correct')['saco_score'].agg(['mean', 'std', 'count'])
        saco_results['by_correctness'] = correctness_stats.to_dict('index')

    print("\nPipeline complete!")

    return results, saco_results
