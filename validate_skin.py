from transformers import ViTImageProcessor, ViTForImageClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def verify_validation_accuracy():
    # Load your fine-tuned model and processor
    model_name = "sharren/vit-skin-demo-v5"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load the validation dataset
    dataset = load_dataset("marmal88/skin_cancer")
    validation_dataset = dataset["validation"]

    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Classes: {list(model.config.id2label.values())}")

    # Preprocessing function
    def preprocess_images(examples):
        # Process images
        images = [image.convert("RGB") for image in examples["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = torch.tensor(examples["label"])
        return inputs

    # Create DataLoader for batch processing
    def collate_fn(batch):
        # Extract images and labels from batch
        images = [item["image"].convert("RGB") for item in batch]
        labels = [item["label"] for item in batch]

        # Process images
        inputs = processor(images, return_tensors="pt", padding=True)
        inputs["labels"] = torch.tensor(labels)
        return inputs

    # Create DataLoader
    dataloader = DataLoader(
        validation_dataset,
        batch_size=16,  # Adjust based on your GPU memory
        shuffle=False,
        collate_fn=collate_fn)

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
    for class_id, class_name in model.config.id2label.items():
        class_id = int(class_id)
        class_mask = np.array(all_labels) == class_id
        if class_mask.sum() > 0:
            class_correct = np.array(all_predictions)[class_mask] == class_id
            class_accuracy = class_correct.sum() / class_mask.sum()
            print(
                f"{class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)"
            )

    return accuracy


if __name__ == "__main__":
    accuracy = verify_validation_accuracy()
