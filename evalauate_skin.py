import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor


def investigate_data_split():
    """Investigate potential data leakage or split issues"""

    # Load all splits to compare
    dataset = load_dataset("marmal88/skin_cancer")
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    print("=== DATASET SPLIT INVESTIGATION ===")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(validation_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print(f"Total: {len(train_dataset) + len(validation_dataset) + len(test_dataset)} samples")

    # Check for overlapping image_ids between splits
    train_ids = set(item["image_id"] for item in train_dataset)
    val_ids = set(item["image_id"] for item in validation_dataset)
    test_ids = set(item["image_id"] for item in test_dataset)

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    print(f"\n=== CHECKING FOR DATA LEAKAGE ===")
    print(f"Train-Validation overlap: {len(train_val_overlap)} images")
    print(f"Train-Test overlap: {len(train_test_overlap)} images")
    print(f"Validation-Test overlap: {len(val_test_overlap)} images")

    if train_val_overlap:
        print(f"⚠️ WARNING: Found {len(train_val_overlap)} overlapping images between train and validation!")
        print("Sample overlapping IDs:", list(train_val_overlap)[:5])

    # Check lesion_id overlap (same lesion, different images)
    train_lesions = set(item["lesion_id"] for item in train_dataset)
    val_lesions = set(item["lesion_id"] for item in validation_dataset)

    lesion_overlap = train_lesions.intersection(val_lesions)
    print(f"\n=== CHECKING LESION-LEVEL LEAKAGE ===")
    print(f"Same lesions in train and validation: {len(lesion_overlap)}")
    if lesion_overlap:
        print(f"⚠️ WARNING: Found {len(lesion_overlap)} lesions appearing in both train and validation!")
        print("This could explain the high accuracy - same patients/lesions in both splits")

    # Check class distribution
    print(f"\n=== CLASS DISTRIBUTION ===")
    for split_name, split_data in [("Train", train_dataset), ("Validation", validation_dataset),
                                   ("Test", test_dataset)]:
        class_counts = {}
        for item in split_data:
            dx = item["dx"]
            class_counts[dx] = class_counts.get(dx, 0) + 1

        print(f"\n{split_name} split:")
        total = len(split_data)
        for dx, count in sorted(class_counts.items()):
            percentage = (count / total) * 100
            print(f"  {dx}: {count} ({percentage:.1f}%)")


def verify_validation_accuracy():
    # Load your fine-tuned model and processor
    model_name = "sharren/vit-skin-demo-v5"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # First investigate the data split
    investigate_data_split()

    print(f"\n{'='*50}")
    print("PROCEEDING WITH ACCURACY CALCULATION...")
    print(f"{'='*50}")

    # Load the validation dataset
    dataset = load_dataset("marmal88/skin_cancer")
    validation_dataset = dataset["validation"]

    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Classes: {list(model.config.id2label.values())}")

    # Check dataset structure
    print(f"Dataset columns: {validation_dataset.column_names}")
    print(f"Sample data: {validation_dataset[0]}")

    # First, let's check what labels actually exist in the dataset
    print("Checking first few validation samples:")
    for i in range(min(5, len(validation_dataset))):
        sample = validation_dataset[i]
        print(f"Sample {i}: dx = '{sample['dx']}'")

    # Get all unique labels in the validation dataset
    all_dx_labels = set()
    for item in validation_dataset:
        all_dx_labels.add(item["dx"])
    print(f"\nAll unique labels in validation dataset: {sorted(all_dx_labels)}")

    # Model's expected labels
    label2id = model.config.label2id
    print(f"Model expects these labels: {sorted(label2id.keys())}")

    # Create label mapping from dataset labels to model labels
    # Based on the actual labels found in the dataset
    dataset_to_model_labels = {
        "actinic_keratoses": "akiec",
        "basal_cell_carcinoma": "bcc",
        "benign_keratosis-like_lesions": "bkl",
        "dermatofibroma": "df",
        "melanoma": "mel",
        "melanocytic_Nevi": "nv",  # Note the capital N
        "vascular_lesions": "vasc"
    }

    print(f"Label mapping created:")
    for dataset_label, model_label in dataset_to_model_labels.items():
        print(f"  '{dataset_label}' → '{model_label}'")

    # Verify all dataset labels can be mapped
    unmapped_labels = all_dx_labels - set(dataset_to_model_labels.keys())
    if unmapped_labels:
        print(f"WARNING: Unmapped labels found: {unmapped_labels}")
    else:
        print("✓ All dataset labels can be mapped to model labels")

    # Create DataLoader for batch processing
    def collate_fn(batch):
        # Extract images and labels from batch
        images = [item["image"].convert("RGB") for item in batch]
        # Use 'dx' column and map to model labels
        labels = []
        for item in batch:
            dx_label = item["dx"]

            # Map dataset label to model label
            if dx_label in dataset_to_model_labels:
                model_label = dataset_to_model_labels[dx_label]
                labels.append(label2id[model_label])
            else:
                raise ValueError(f"Unknown label '{dx_label}' found in dataset")

        # Process images
        inputs = processor(images, return_tensors="pt", padding=True)
        inputs["labels"] = torch.tensor(labels)
        return inputs

    # Create DataLoader
    dataloader = DataLoader(
        validation_dataset,
        batch_size=16,  # Adjust based on your GPU memory
        shuffle=False,
        collate_fn=collate_fn
    )

    # Evaluate the model
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []

    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            # Move to device
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Calculate accuracy for this batch
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            # Store for detailed analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final accuracy
    accuracy = correct_predictions / total_predictions

    print(f"\nValidation Results:")
    print(f"Total samples: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Expected Accuracy (from model card): 0.8492 (84.92%)")

    # Optional: Per-class accuracy
    print("\nPer-class accuracy:")
    id2label = model.config.id2label
    for class_id, class_name in id2label.items():
        class_id = int(class_id)
        class_mask = np.array(all_labels) == class_id
        if class_mask.sum() > 0:
            class_correct = np.array(all_predictions)[class_mask] == class_id
            class_accuracy = class_correct.sum() / class_mask.sum()
            class_count = class_mask.sum()
            print(f"{class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {class_count} samples")
        else:
            print(f"{class_name}: No samples found in validation set")

    return accuracy


if __name__ == "__main__":
    accuracy = verify_validation_accuracy()
