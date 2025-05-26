# pipeline.py
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import quantus
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from tqdm import tqdm

import io_utils
import perturbation
import visualization
import vit.attribution as attribution
import vit.model as model
import vit.preprocessing as preprocessing
from config import FileConfig, PipelineConfig
from data_types import (AttributionDataBundle, AttributionOutputPaths,
                        ClassEmbeddingRepresentationItem,
                        ClassificationPrediction, ClassificationResult,
                        FFNActivityItem, HeadContributionItem,
                        PerturbationPatchInfo, PerturbedImageRecord)
from faithfulness import evaluate_and_report_faithfulness


def preprocess_dataset(config: PipelineConfig, source_dir: Path) -> List[Path]:
    """
    Preprocess images by resizing them to target_size and saving them to the data directory.
    
    Args:
        config: Pipeline configuration
        source_dir: Directory containing original images
        
    Returns:
        List of paths to the processed images
    """
    io_utils.ensure_directories([config.file.data_dir])

    image_files = list(source_dir.glob("**/*.png"))
    print(f"Found {len(image_files)} X-ray images")

    processed_paths = []
    for image_file in image_files:
        output_path = config.file.data_dir / image_file.name

        # Skip if exists
        if output_path.exists():
            processed_paths.append(output_path)
            continue

        try:
            img = preprocessing.load_image(str(image_file))
            processor = preprocessing.get_default_processor(
                img_size=config.classify.target_size[0])
            # Apply the PIL transforms (resize, crop) without tensor conversion
            pil_transform = processor.transforms[0]
            processed_img = pil_transform(img)
            processed_img.save(output_path)

            processed_paths.append(output_path)
            print(f"Processed: {image_file.name} -> {output_path}")
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            print("here")
            continue

    print(
        f"Preprocessing complete. {len(processed_paths)} images saved to {config.file.data_dir}"
    )
    return processed_paths


def save_attribution_bundle_to_files(
        image_stem: str, attribution_bundle: AttributionDataBundle,
        file_config: FileConfig) -> AttributionOutputPaths:
    """Save attribution bundle contents to .npy files."""

    # Ensure attribution directory exists
    io_utils.ensure_directories([file_config.attribution_dir])

    attribution_path = file_config.attribution_dir / f"{image_stem}_attribution.npy"
    logits_path = Path("")
    ffn_activity_path = Path("")
    class_embedding_path = Path("")
    head_contribution_path = Path("")

    # Save positive attribution
    np.save(attribution_path, attribution_bundle.positive_attribution)

    # Save logits (save empty array if None)
    if attribution_bundle.logits is not None:
        logits_path = file_config.attribution_dir / f"{image_stem}_logits.npy"
        np.save(logits_path, attribution_bundle.logits)

    # Save FFN activities
    # Convert List[FFNActivityItem] to a NumPy array of objects (dicts) for saving
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

    if attribution_bundle.head_contribution:
        head_contribution_path = file_config.attribution_dir / f"{image_stem}_head_contribution.npy"
        head_contribution_data_to_save = []
        for item in attribution_bundle.head_contribution:
            head_contribution_data_to_save.append({
                'layer':
                item.layer,
                'activity_data':
                item.stacked_contribution
            })
        np.save(head_contribution_path,
                np.array(head_contribution_data_to_save, dtype=object))

    # Save class embedding representations
    if attribution_bundle.class_embedding_representations:
        class_embedding_path = file_config.attribution_dir / f"{image_stem}_class_embedding_representation.npy"
        class_embedding_data_to_save = []
        for item in attribution_bundle.class_embedding_representations:
            class_embedding_data_to_save.append({
                'layer':
                item.layer,
                'attention_class_representation_input':
                item.attention_class_representation_input,
                'mlp_class_representation_input':
                item.mlp_class_representation_input,
                'attention_class_representation_output':
                item.attention_class_representation_output,
                'attention_map':
                item.attention_map,
                'mlp_class_representation_output':
                item.mlp_class_representation_output
            })
        np.save(class_embedding_path,
                np.array(class_embedding_data_to_save, dtype=object))

    return AttributionOutputPaths(
        attribution_path=attribution_path,
        logits=logits_path,
        ffn_activity_path=ffn_activity_path,
        class_embedding_path=class_embedding_path,
        head_contribution_path=head_contribution_path)


def classify_single_image(config: PipelineConfig, image_path: Path,
                          vit_model: model.VisionTransformer,
                          device: torch.device) -> ClassificationResult:
    """
    Classifies a single image and returns a ClassificationResult with prediction only.
    Uses its own caching mechanism for classification-only results.
    """
    cache_path = io_utils.build_cache_path(
        config.file.cache_dir, image_path,
        f"_classification_explained{config.file.output_suffix}.json")

    # Try to load from cache
    loaded_result = io_utils.try_load_from_cache(cache_path)
    if config.file.use_cached_perturbed and loaded_result:
        if loaded_result.attribution_paths is not None:
            print(
                f"Warning: Cache for classify_single_image contained attribution paths for {image_path.name}. Ignoring them."
            )
            loaded_result.attribution_paths = None
        return loaded_result

    _, input_tensor = preprocessing.preprocess_image(
        str(image_path), img_size=config.classify.target_size[0])
    input_tensor = input_tensor.to(device)

    prediction_dict = model.get_prediction(vit_model,
                                           input_tensor,
                                           device=device,
                                           eval=True)

    current_prediction = ClassificationPrediction(
        predicted_class_label=prediction_dict["predicted_class_label"],
        predicted_class_idx=prediction_dict["predicted_class_idx"],
        confidence=float(prediction_dict["probabilities"][
            prediction_dict["predicted_class_idx"]].item()),
        probabilities=prediction_dict["probabilities"].tolist())

    result = ClassificationResult(image_path=image_path,
                                  prediction=current_prediction,
                                  attribution_paths=None)

    # Cache the result
    io_utils.save_to_cache(cache_path, result)

    return result


def classify_explain_single_image(
    config: PipelineConfig,
    image_path: Path,
    vit_model: model.
    VisionTransformer,  # Hooks should be registered on this model instance
    device: torch.device
) -> ClassificationResult:
    """
    Classifies a single image AND generates explanations.
    Relies on vit.attribution.generate_attribution to return both prediction and attribution data.
    Manages caching for the combined ClassificationResult.
    """

    cache_path = io_utils.build_cache_path(
        config.file.cache_dir, image_path,
        f"_classification_explained{config.file.output_suffix}.json")

    # Try to load from cache
    loaded_result = io_utils.try_load_from_cache(cache_path)
    if config.file.use_cached_original and loaded_result:
        if loaded_result.attribution_paths is not None:
            return loaded_result

    # 1. Preprocess image
    _, input_tensor = preprocessing.preprocess_image(
        str(image_path), img_size=config.classify.target_size[0])
    input_tensor = input_tensor.to(device)

    # 2. Call attribution.generate_attribution
    raw_attribution_result_dict = attribution.generate_attribution(
        model=vit_model,
        input_tensor=input_tensor,
        method=config.classify.attribution_method,
        target_class=None,
        device=device,
        img_size=config.classify.target_size[0],
        weigh_by_class_embedding=config.file.weighted,
        data_collection=config.classify.data_collection)

    # 3. Instantiate ClassificationPrediction from the "predictions" field
    prediction_data = raw_attribution_result_dict["predictions"]
    current_prediction = ClassificationPrediction(
        predicted_class_label=prediction_data["predicted_class_label"],
        predicted_class_idx=prediction_data["predicted_class_idx"],
        confidence=float(prediction_data["probabilities"][
            prediction_data["predicted_class_idx"]]
                         ),  # Assuming probabilities is a list/tensor
        probabilities=prediction_data["probabilities"])  # Needs to be a list

    # 4. Instantiate FFNActivityItem and ClassEmbeddingRepresentationItem lists
    ffn_activity_items: List[FFNActivityItem] = []
    if "ffn_activity" in raw_attribution_result_dict and raw_attribution_result_dict[
            "ffn_activity"]:
        for ffn_dict in raw_attribution_result_dict["ffn_activity"]:
            ffn_activity_items.append(
                FFNActivityItem(layer=ffn_dict["layer"],
                                mean_activity=ffn_dict["mean_activity"],
                                cls_activity=ffn_dict["cls_activity"],
                                activity_data=ffn_dict["activity"]))

    class_embedding_items: List[ClassEmbeddingRepresentationItem] = []
    if "class_embedding_representation" in raw_attribution_result_dict and raw_attribution_result_dict[
            "class_embedding_representation"]:
        for cer_dict in raw_attribution_result_dict[
                "class_embedding_representation"]:
            class_embedding_items.append(
                ClassEmbeddingRepresentationItem(
                    layer=cer_dict["layer"],
                    attention_class_representation_output=cer_dict[
                        "attention_class_representation_output"],
                    mlp_class_representation_output=cer_dict[
                        "mlp_class_representation_output"],
                    attention_class_representation_input=cer_dict[
                        "attention_class_representation_input"],
                    attention_map=cer_dict["attention_map"],
                    mlp_class_representation_input=cer_dict[
                        "mlp_class_representation_input"]))

    head_contribution_items: List[HeadContributionItem] = []
    if "head_contribution" in raw_attribution_result_dict and raw_attribution_result_dict[
            "head_contribution"]:
        for head_contribution_dict in raw_attribution_result_dict[
                "head_contribution"]:
            head_contribution_items.append(
                HeadContributionItem(
                    layer=head_contribution_dict["layer"],
                    stacked_contribution=head_contribution_dict[
                        "stacked_contribution"]))

    # 5. Instantiate AttributionDataBundle
    attribution_bundle = AttributionDataBundle(
        positive_attribution=raw_attribution_result_dict[
            "attribution_positive"],
        logits=raw_attribution_result_dict.get("logits"),
        ffn_activities=ffn_activity_items,
        class_embedding_representations=class_embedding_items,
        head_contribution=head_contribution_items,
    )

    # 6. Save attribution bundle to files
    saved_attribution_paths = save_attribution_bundle_to_files(
        image_path.stem, attribution_bundle, config.file)

    # 7. Create final ClassificationResult
    final_result = ClassificationResult(
        image_path=image_path,
        prediction=current_prediction,
        attribution_paths=saved_attribution_paths)

    # 8. Cache the combined result
    io_utils.save_to_cache(cache_path, final_result)

    return final_result


def classify_and_explain_dataset(
        config: PipelineConfig, vit_model: model.VisionTransformer,
        device: torch.device, image_paths_to_process: List[Path],
        output_results_csv_path: Path) -> List[ClassificationResult]:
    collected_results: List[ClassificationResult] = []
    for image_path in tqdm(
            image_paths_to_process,
            desc=
            f"Classifying & Explaining (suffix: '{config.file.output_suffix}')"
    ):
        try:
            # This now uses the revised single image function
            result = classify_explain_single_image(config, image_path,
                                                   vit_model, device)
            collected_results.append(result)
        except Exception as e:
            print(f"Error C&E {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    if collected_results:
        io_utils.save_classification_results_to_csv(collected_results,
                                                    output_results_csv_path)
    else:
        print("No images successfully C&E.")
    return collected_results


def classify_dataset_only(
        config: PipelineConfig, vit_model: model.VisionTransformer,
        device: torch.device, image_paths_to_process: List[Path],
        output_results_csv_path: Path) -> List[ClassificationResult]:
    collected_results: List[ClassificationResult] = []
    for image_path in tqdm(
            image_paths_to_process,
            desc=f"Classifying dataset (suffix: '{config.file.output_suffix}')"
    ):
        try:
            result = classify_single_image(config, image_path, vit_model,
                                           device)
            collected_results.append(result)
        except Exception as e:
            print(f"Error classifying {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    if collected_results:
        io_utils.save_classification_results_to_csv(collected_results,
                                                    output_results_csv_path)
    else:
        print("No images successfully classified.")
    return collected_results


def generate_perturbed_identifier(config: PipelineConfig,
                                  original_image_stem: str,
                                  patch_info: PerturbationPatchInfo) -> str:
    if config.perturb.method == "sd":
        return f"{original_image_stem}_patch{patch_info.patch_id}_x{patch_info.x}_y{patch_info.y}_s{config.perturb.strength}"
    else:
        return f"{original_image_stem}_patch{patch_info.patch_id}_x{patch_info.x}_y{patch_info.y}_{config.perturb.method}"


def perturb_single_patch(
    config: PipelineConfig,
    original_image_path: Path,
    patch_info_dc: PerturbationPatchInfo,
    sd_pipe: Optional[StableDiffusionInpaintPipeline] = None
) -> Optional[PerturbedImageRecord]:
    original_image_stem = original_image_path.stem
    patch_size = config.perturb.patch_size
    perturbed_id = generate_perturbed_identifier(config, original_image_stem,
                                                 patch_info_dc)
    perturbed_image_path = config.file.perturbed_dir / f'{perturbed_id}.png'
    mask_path = config.file.mask_dir / f'{perturbed_id}_mask.npy'

    if config.file.use_cached_perturbed and perturbed_image_path.exists(
    ) and mask_path.exists():
        return PerturbedImageRecord(
            original_image_path=original_image_path,
            perturbed_image_path=perturbed_image_path,
            mask_path=mask_path,
            patch_info=patch_info_dc,
            perturbation_method=config.perturb.method,
            perturbation_strength=config.perturb.strength
            if config.perturb.method == "sd" else None)
    try:
        io_utils.ensure_directories(
            [config.file.perturbed_dir, config.file.mask_dir])
        patch_coordinates = (patch_info_dc.x, patch_info_dc.y, patch_size)
        if config.perturb.method == "sd":
            raise Exception("do not include sd anymore")
        elif config.perturb.method == "mean":
            result_image, np_mask = perturbation.perturb_patch_mean(
                str(original_image_path), patch_coordinates, config.file)
        else:
            raise ValueError(
                f"Unknown perturbation method: {config.perturb.method}")
        result_image.save(perturbed_image_path)
        np.save(mask_path, np_mask)
        return PerturbedImageRecord(
            original_image_path=original_image_path,
            perturbed_image_path=perturbed_image_path,
            mask_path=mask_path,
            patch_info=patch_info_dc,
            perturbation_method=config.perturb.method,
            perturbation_strength=config.perturb.strength
            if config.perturb.method == "sd" else None)
    except Exception as e:
        print(
            f"Error perturbing patch for {original_image_stem} (ID {patch_info_dc.patch_id}): {e}"
        )
        return None


def generate_patch_coordinates(image: Image.Image,
                               patch_size: int) -> List[Tuple[int, int, int]]:
    """Generate patch coordinates for an image.
    
    Args:
        image: Image to generate patches for
        patch_size: Size of patches
        
    Returns:
        List of patch coordinates as (patch_id, x, y)
    """
    width, height = image.size
    num_patches_x = width // patch_size
    patches = []

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch_id = (y // patch_size) * num_patches_x + (x // patch_size)
            patches.append((patch_id, x, y))

    return patches


def perturb_image_patches(
    config: PipelineConfig,
    original_image_path: Path,
    sd_pipe: Optional[StableDiffusionInpaintPipeline] = None
) -> List[PerturbedImageRecord]:
    """
    Perturb patches in a single image.
    
    Args:
        config: Pipeline configuration
        image_path: Path to the image to perturb
        sd_pipe: Optional StableDiffusionInpaintPipeline for SD method
        
    Returns:
        List of paths to perturbed images
    """
    perturbed_records: List[PerturbedImageRecord] = []
    try:
        image = preprocessing.load_image(str(original_image_path))
    except Exception as e:
        print(
            f"Error loading image {original_image_path} for perturbation: {e}")
        return []
    patch_coordinates_list = generate_patch_coordinates(
        image, config.perturb.patch_size)
    for patch_id, x, y in patch_coordinates_list:
        patch_info_dc = PerturbationPatchInfo(patch_id=patch_id, x=x, y=y)
        record = perturb_single_patch(config, original_image_path,
                                      patch_info_dc, sd_pipe)
        if record: perturbed_records.append(record)
    return perturbed_records


def perturb_dataset(
    config: PipelineConfig,
    image_paths_to_perturb: List[Path],
    sd_pipe: Optional[StableDiffusionInpaintPipeline] = None
) -> List[PerturbedImageRecord]:
    """
    Perturb patches in all images.
    
    Args:
        config: Pipeline configuration
        results_df: DataFrame with classification results
        sd_pipe: Optional StableDiffusionInpaintPipeline for SD method
        
    Returns:
        List of paths to perturbed images
    """
    io_utils.ensure_directories(
        [config.file.perturbed_dir, config.file.mask_dir])
    all_perturbed_records: List[PerturbedImageRecord] = []
    for image_path in tqdm(image_paths_to_perturb, desc="Perturbing dataset"):
        records_for_image = perturb_image_patches(config, image_path, sd_pipe)
        all_perturbed_records.extend(records_for_image)
    print(f"Generated {len(all_perturbed_records)} perturbed image records.")
    return all_perturbed_records


def run_perturbation(config: PipelineConfig,
                     image_paths: List[Path]) -> List[PerturbedImageRecord]:
    """
    Run the perturbation pipeline.
    
    Args:
        config: Pipeline configuration
        results_df: Classification results
        
    Returns:
        List of paths to perturbed images
    """
    # Load the SD pipeline if needed
    sd_pipe = None
    if config.perturb.method == "sd":
        sd_pipe = perturbation.load_sd_model()

    return perturb_dataset(config, image_paths, sd_pipe)


def visualize_attributions(results_df: pd.DataFrame):
    for _, row in results_df.iterrows():
        # Extract file locations from the DataFrame row
        image_path = row["image_path"]
        attribution_path = row["attribution_path"]

        # Load the original image
        original_image = Image.open(image_path)

        # Load the attribution map from the stored .npy file
        attribution_map = np.load(attribution_path)

        # Call the visualization function
        visualization.visualize_attribution_map(
            attribution_map,
            original_image,
            save_path=
            f'./results/vit_inputs_unweighted/{Path(image_path).stem}_visualization.png'
        )


def run_classification_standalone(
        config: PipelineConfig,
        device: torch.device,
        image_source_dir: Optional[Path] = None,
        output_suffix_override: Optional[str] = None,
        explain: bool = True) -> List[ClassificationResult]:
    io_utils.ensure_directories(config.directories)
    vit_model = model.load_vit_model(device=device)
    print("ViT model loaded")

    if image_source_dir:
        images_to_proc = sorted(list(image_source_dir.glob("*.png")))
    else:
        images_to_proc = sorted(list(config.file.data_dir.glob("*.png")))
    print(
        f"Processing {len(images_to_proc)} images from {'source' if image_source_dir else 'data_dir'}."
    )

    orig_suffix = config.file.output_suffix
    if output_suffix_override is not None:
        config.file.output_suffix = output_suffix_override

    explain_tag = "_explained" if explain else "_classified_only"
    csv_path = config.file.output_dir / f"results{explain_tag}{config.file.output_suffix}.csv"

    if explain:
        print(f"Running C&E (suffix: '{config.file.output_suffix}')")
        results = classify_and_explain_dataset(config, vit_model, device,
                                               images_to_proc, csv_path)
    else:
        print(f"Running Classify Only (suffix: '{config.file.output_suffix}')")
        results = classify_dataset_only(config, vit_model, device,
                                        images_to_proc, csv_path)

    if output_suffix_override is not None:
        config.file.output_suffix = orig_suffix
    return results


def run_pipeline(
    config: PipelineConfig,
    source_dir_for_preprocessing: Path,
    device: Optional[torch.device] = None
) -> Tuple[List[ClassificationResult], Tuple[List[PerturbedImageRecord],
                                             List[ClassificationResult]]]:
    """
    Streamlined full pipeline:
    1. (Optional) Preprocess dataset.
    2. Classify AND Explain original images.
    3. Run perturbation on original images.
    4. Classify ONLY perturbed images.
    Returns Tuple[List[ClassificationResult] (originals, explained), List[ClassificationResult] (perturbed, classified_only)].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    io_utils.ensure_directories(config.directories)

    original_image_paths = preprocess_dataset(config,
                                              source_dir_for_preprocessing)
    print(f"Found {len(original_image_paths)} original images for processing.")

    vit_model = model.load_vit_model(device=device)
    print("ViT model loaded, hooks registered, and set to eval mode.")

    originals_csv_path = config.file.output_dir / f"classification_results_originals_explained{config.file.output_suffix}.csv"
    print(f"Running Classify & Explain for original images")

    original_results_explained = classify_and_explain_dataset(
        config, vit_model, device, original_image_paths, originals_csv_path)

    if config.classify.analysis:
        print("Evaluate Faithfulness Pipeline")
        evaluate_and_report_faithfulness(config, vit_model, device,
                                         original_results_explained)

    paths_for_perturbation = [
        res.image_path for res in original_results_explained
    ]

    perturbed_image_records: List[PerturbedImageRecord] = run_perturbation(
        config, paths_for_perturbation)

    perturbed_image_paths_to_classify = [
        rec.perturbed_image_path for rec in perturbed_image_records
        if rec.perturbed_image_path.exists()
    ]

    config.file.output_suffix = config.file.output_suffix + "_perturbed"

    perturbed_csv_path = config.file.output_dir / f"classification_results_perturbed_classified_only{config.file.output_suffix}.csv"
    print(
        f"Running Classify ONLY for perturbed images (suffix: '{config.file.output_suffix}')"
    )

    perturbed_results_classified_only = classify_dataset_only(
        config, vit_model, device, perturbed_image_paths_to_classify,
        perturbed_csv_path)

    print("Full pipeline finished.")
    return original_results_explained, (perturbed_image_records,
                                        perturbed_results_classified_only)
