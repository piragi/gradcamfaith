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
import vit.preprocessing as preprocessing
from attribution_binning import run_binned_attribution_analysis
from config import FileConfig, PipelineConfig
from data_types import (
    AttributionDataBundle, AttributionOutputPaths, ClassEmbeddingRepresentationItem, ClassificationPrediction,
    ClassificationResult, FFNActivityItem, HeadContributionItem, PerturbationPatchInfo, PerturbedImageRecord
)
# New imports for unified system
from dataset_config import DatasetConfig, get_dataset_config
from faithfulness import evaluate_and_report_faithfulness
from setup import convert_dataset
from transmm_sfaf import (generate_attribution_prisma_enhanced, load_steering_resources)
from unified_dataloader import (UnifiedMedicalDataset, create_dataloader, get_single_image_loader)


def load_model_for_dataset(dataset_config: DatasetConfig, device: torch.device, config: PipelineConfig = None):
    """
    Load the appropriate model for a given dataset configuration.
    
    Args:
        dataset_config: Configuration for the dataset
        device: Device to load the model on
        config: Optional pipeline config for CLIP settings
        
    Returns:
        Loaded model with appropriate head size (and processor if CLIP)
    """
    # Check if we should use CLIP for this dataset
    use_clip = (config and config.classify.use_clip) or dataset_config.name == "waterbirds"

    if use_clip:
        from transformers import CLIPProcessor
        from vit_prisma.models.model_loader import load_hooked_model

        print(f"Loading CLIP as HookedViT for {dataset_config.name}")

        # Use vit_prisma's load_hooked_model to get CLIP as HookedViT
        # This automatically converts CLIP weights to HookedViT format
        clip_model_name = config.classify.clip_model_name if config else "openai/clip-vit-base-patch32"

        model = load_hooked_model(clip_model_name, dtype=torch.float32, device=device)
        # Ensure model is on the correct device
        model = model.to(device)
        model.eval()

        # Also load the processor for text encoding
        # For OpenCLIP models, use vit_prisma's transforms
        if "open-clip:" in clip_model_name:
            from vit_prisma.transforms import get_clip_val_transforms
            processor = get_clip_val_transforms()
            print("Using vit_prisma transforms for OpenCLIP model")
        else:
            processor_name = config.classify.clip_model_name if config else "openai/clip-vit-base-patch32"
            processor = CLIPProcessor.from_pretrained(processor_name)

        print(f"✅ CLIP loaded as HookedViT")
        # Return tuple for CLIP (model, processor)
        return (model, processor)

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
        predicted_idx = torch.argmax(probabilities, dim=-1).item()

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
        logits_arr = attribution_bundle.logits
        if isinstance(logits_arr, torch.Tensor):
            logits_arr = logits_arr.detach().cpu().numpy()
        np.save(logits_path, logits_arr)

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
    processor: Optional[Any] = None,  # CLIP processor if using CLIP
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

    # Only create CLIP classifier if not provided (shouldn't happen in normal pipeline)
    if processor is not None and clip_classifier is None:
        from clip_classifier import (create_clip_classifier_for_oxford_pets, create_clip_classifier_for_waterbirds)
        print("WARNING: Creating CLIP classifier per image (inefficient!)")

        # Choose the appropriate factory based on dataset
        dataset_name = config.file.dataset_name
        if dataset_name == "oxford_pets":
            clip_classifier = create_clip_classifier_for_oxford_pets(
                vision_model=model,
                processor=processor,
                device=device,
                custom_prompts=config.classify.clip_text_prompts if config.classify.clip_text_prompts else None
            )
        else:  # default to waterbirds
            clip_classifier = create_clip_classifier_for_waterbirds(
                vision_model=model,
                processor=processor,
                device=device,
                custom_prompts=config.classify.clip_text_prompts if config.classify.clip_text_prompts else None
            )

    # Always use enhanced version for better performance
    # It will run vanilla TransLRP when both steering and feature gradients are disabled
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
    if "class_embedding_representation" in raw_attribution_result_dict and raw_attribution_result_dict[
        "class_embedding_representation"]:
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
    prepared_data_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    force_prepare: bool = False,
    subset_size: Optional[int] = None,
    random_seed: Optional[int] = None
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
    print(f"  Classes: {dataset_config.num_classes} - {dataset_config.class_names}")

    # Ensure all required directories exist
    io_utils.ensure_directories(config.directories)

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
    # Check if we're using CLIP for this dataset
    use_clip = config.classify.use_clip if hasattr(config.classify, 'use_clip') else False
    dataset_loader = create_dataloader(
        dataset_name=dataset_name,
        data_path=prepared_path,
        batch_size=config.classify.batch_size if hasattr(config.classify, 'batch_size') else 32,
        use_clip=use_clip
    )

    # Print dataset statistics
    stats = dataset_loader.get_statistics()
    for split, split_stats in stats['splits'].items():
        print(f"\n{split.upper()}: {split_stats['total_samples']} samples")
        for class_name, count in split_stats['class_distribution'].items():
            print(f"  {class_name}: {count}")

    # Load model
    print(f"\nLoading model for {dataset_name}")
    model_result = load_model_for_dataset(dataset_config, device, config)

    # Handle CLIP model (returns tuple) vs regular model
    if isinstance(model_result, tuple):
        model, processor = model_result
        print("CLIP model loaded with processor")
    else:
        model = model_result
        processor = None

    # Load steering resources if needed - for BOTH steering and feature gradient layers
    steering_layers = config.classify.boosting.steering_layers
    feature_gradient_layers = config.classify.boosting.feature_gradient_layers if config.classify.boosting.enable_feature_gradients else []

    # Combine both layer sets to load all necessary SAEs
    all_layers_needing_sae = list(set(steering_layers) | set(feature_gradient_layers))
    steering_resources = load_steering_resources(all_layers_needing_sae, dataset_name=dataset_name)
    print(
        f"Steering resources loaded for {dataset_name} layers: {all_layers_needing_sae} (steering: {steering_layers}, gradients: {feature_gradient_layers})"
    )

    # Process images from the specified split (mode)
    split_to_use = config.file.current_mode if config.file.current_mode in ['train', 'val', 'test', 'dev'] else 'test'

    split_dataset = dataset_loader.get_dataset(split_to_use)
    # Get both image paths and their true labels
    image_data = [(Path(img_path), label_idx) for img_path, label_idx in split_dataset.samples]

    # Apply subset if requested
    if subset_size is not None and subset_size < len(image_data):
        import random
        if random_seed is not None:
            random.seed(random_seed)
        image_data = random.sample(image_data, subset_size)
        print(
            f"\nProcessing {len(image_data)} randomly selected {split_to_use} images (subset of {len(split_dataset.samples)})"
        )
    else:
        print(f"\nProcessing {len(image_data)} {split_to_use} images")

    # Create CLIP classifier once if using CLIP (not per image!)
    clip_classifier = None
    if processor is not None:
        from clip_classifier import (create_clip_classifier_for_oxford_pets, create_clip_classifier_for_waterbirds)
        print("Creating CLIP classifier (once for all images)...")

        # Choose the appropriate factory based on dataset
        if dataset_name == "oxford_pets":
            clip_classifier = create_clip_classifier_for_oxford_pets(
                vision_model=model,
                processor=processor,
                device=device,
                custom_prompts=config.classify.clip_text_prompts if config.classify.clip_text_prompts else None
            )
        else:  # default to waterbirds
            clip_classifier = create_clip_classifier_for_waterbirds(
                vision_model=model,
                processor=processor,
                device=device,
                custom_prompts=config.classify.clip_text_prompts if config.classify.clip_text_prompts else None
            )

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
                processor=processor,  # Pass processor for CLIP
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
    n_bins = 20 if is_patch32 else 49
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

        print(f"Mean SaCo score: {saco_results['mean']:.4f} (std={saco_results['std']:.4f})")
        print(f"Per-class SaCo scores:")
        for class_name, stats in saco_results['per_class'].items():
            print(f"  {class_name}: {stats['mean']:.4f} (std={stats['std']:.4f}, n={stats['count']})")

    print("\nPipeline complete!")

    # Clean up models and SAEs to prevent memory leaks
    # IMPORTANT: We need to break circular references from backward pass
    if 'model' in locals():
        # Clear gradients first
        model.zero_grad(set_to_none=True)
        # Clear optimizer state if any
        for p in model.parameters():
            p.grad = None
            if hasattr(p, '_grad'):
                p._grad = None
        # Move to CPU and clear data
        model.to("cpu")
        for param in model.parameters():
            param.data = torch.empty(0)
        # Clear any hooks
        if hasattr(model, 'reset_hooks'):
            model.reset_hooks(including_permanent=True, clear_contexts=True)
        if hasattr(model, '_forward_hooks'):
            model._forward_hooks.clear()
        if hasattr(model, '_backward_hooks'):
            model._backward_hooks.clear()
        del model

    if 'processor' in locals():
        del processor

    if 'clip_classifier' in locals():
        if hasattr(clip_classifier, 'vision_model'):
            # Clear gradients
            clip_classifier.vision_model.zero_grad(set_to_none=True)
            for p in clip_classifier.vision_model.parameters():
                p.grad = None
            clip_classifier.vision_model.to("cpu")
            # Clear hooks
            if hasattr(clip_classifier.vision_model, 'reset_hooks'):
                clip_classifier.vision_model.reset_hooks(including_permanent=True, clear_contexts=True)
        del clip_classifier

    if 'model_for_analysis' in locals():
        if hasattr(model_for_analysis, 'to'):
            model_for_analysis.to("cpu")
        # For CLIPModelWrapper, clean the underlying model
        if hasattr(model_for_analysis, 'model') and hasattr(model_for_analysis.model, 'vision_model'):
            model_for_analysis.model.vision_model.zero_grad(set_to_none=True)
            if hasattr(model_for_analysis.model.vision_model, 'reset_hooks'):
                model_for_analysis.model.vision_model.reset_hooks(including_permanent=True, clear_contexts=True)
        del model_for_analysis

    # Clean up steering resources (SAEs)
    if 'steering_resources' in locals() and steering_resources:
        for layer_idx in list(steering_resources.keys()):
            if "sae" in steering_resources[layer_idx]:
                sae = steering_resources[layer_idx]["sae"]
                sae.to("cpu")
                for param in sae.parameters():
                    param.data = torch.empty(0)
                del steering_resources[layer_idx]["sae"]
            # Only SAE resources to clean up now
        steering_resources.clear()
        del steering_resources

    # Force garbage collection
    import gc
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    return results, saco_results


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
