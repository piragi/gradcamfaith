from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import your existing modules
import vit.model as model
from vit.model import CLASSES, CLS2IDX, IDX2CLS
import vit.preprocessing as preprocessing
from config import PipelineConfig


class HyperkvasirDataset(Dataset):
    def __init__(self, image_dir, current_cls2idx, current_idx2cls, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))
        self.transform = transform
        
        # Use the mappings provided for THIS RUN
        self.cls_to_idx = current_cls2idx
        self.idx_to_cls = current_idx2cls # For potentially deriving ordered class names
        
        # Derive class_names for reports based on the numerical order of indices in idx_to_cls
        # This ensures the report labels match the 0-5 order of the model output
        self.class_names_for_report = [self.idx_to_cls[i] for i in range(len(self.idx_to_cls))]

        if not self.image_paths:
            print(f"Warning: No images found in {self.image_dir}")
        self._verify_filename_suffixes() # Important sanity check

    def _verify_filename_suffixes(self):
        unmappable_files = []
        for path in self.image_paths:
            class_name_from_file = path.stem.split('_')[-1]
            if class_name_from_file not in self.cls_to_idx:
                unmappable_files.append(path)
        if unmappable_files:
            print(f"ERROR: The following files have suffixes that cannot be mapped using the current CLS2IDX:")
            for f_path in unmappable_files[:5]:
                print(f"  - {f_path} (suffix: {f_path.stem.split('_')[-1]})")
            if len(unmappable_files) > 5:
                print(f"  ... and {len(unmappable_files) - 5} more.")
            print(f"Keys in current CLS2IDX: {list(self.cls_to_idx.keys())}")
            raise ValueError("Mismatch between filename suffixes and CLS2IDX keys. Check vit/model.py or your manual definition.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        class_name_from_file = image_path.stem.split('_')[-1]
        
        # This check should pass if _verify_filename_suffixes passed
        if class_name_from_file not in self.cls_to_idx:
             raise ValueError(f"Class '{class_name_from_file}' from file {image_path} not in CLS2IDX map.")
        
        class_idx_numerical = self.cls_to_idx[class_name_from_file]

        if self.transform:
            image = self.transform(image)
        return image, class_idx_numerical, str(image_path) # Return path for debugging if needed

# --- Modified create_data_loaders ---
# It will now implicitly use the mappings imported from vit.model at the time of the call
def create_data_loaders(config, batch_size=32, source_dir_root: Path = Path("./hyper-kvasir/preprocessed/")):
    # These will be the mappings currently defined in vit.model when this function is called
    current_classes_list = CLASSES
    current_cls2idx = CLS2IDX
    current_idx2cls = IDX2CLS

    # Sanity check: Ensure CLS2IDX and IDX2CLS are consistent with CLASSES
    if len(current_classes_list) != len(current_cls2idx) or len(current_classes_list) != len(current_idx2cls):
        raise ValueError("Mismatch in lengths of CLASSES, CLS2IDX, IDX2CLS in vit.model.py")
    for i, name in enumerate(current_classes_list):
        if current_cls2idx.get(name) != i or current_idx2cls.get(i) != name:
            raise ValueError(f"Inconsistency in mappings for class '{name}' at index {i} in vit.model.py. "
                             f"Expected CLS2IDX['{name}'] == {i} and IDX2CLS[{i}] == '{name}'. "
                             f"Got CLS2IDX.get('{name}') = {current_cls2idx.get(name)}, IDX2CLS.get({i}) = {current_idx2cls.get(i)}")


    processor = preprocessing.get_processor_for_precached_224_images()

    train_dataset = HyperkvasirDataset(source_dir_root / "train", current_cls2idx, current_idx2cls, transform=processor)
    val_dataset = HyperkvasirDataset(source_dir_root / "val", current_cls2idx, current_idx2cls, transform=processor)
    test_dataset = HyperkvasirDataset(source_dir_root / "test", current_cls2idx, current_idx2cls, transform=processor)

    class_names_for_report = test_dataset.class_names_for_report

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names_for_report

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels, _) in enumerate(tqdm(train_loader, desc="Training")):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, f1


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate for one epoch
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, f1


def train_vit_on_hyperkvasir(config, num_epochs=50, batch_size=32, learning_rate=1e-4):
    """
    Main training function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(config, batch_size)
    num_classes = len(class_names)
    criterion = nn.CrossEntropyLoss()

    # Create model
    vit_model = model.load_vit_model(device=device, model_path="./model/vit_b_hyperkvasir_anatomical_for_translrp.pth",num_classes=num_classes).cuda()
    print("Head layer weights (first few values per class):")
    print(vit_model.head.weight.data[:, :5])
    print("Head layer biases:")
    print(vit_model.head.bias.data)

    # Final test evaluation
    print("\nFinal evaluation on test set:")
    test_loss, test_acc, test_f1 = validate_epoch(vit_model, test_loader, criterion, device)
    print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    # Detailed test evaluation
    vit_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vit_model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nDetailed Test Results:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    return vit_model, class_names

# Usage example
def main():
    config = PipelineConfig()

    # Train the model
    trained_model, class_names = train_vit_on_hyperkvasir(
        config=config, num_epochs=50, batch_size=1, learning_rate=1e-4
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
