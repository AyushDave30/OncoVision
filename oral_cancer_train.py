#!/usr/bin/env python3
"""
Oral Cancer Histopathological Detection Training Script
Optimized for Kaggle environment with MobileNetV3-Small
Dataset: Normal epithelium vs Oral Squamous Cell Carcinoma (OSCC)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Configuration
class Config:
    # Paths for oral cancer dataset
    BASE_DIR = "/kaggle/input/dataset"
    TRAIN_DIR = "/kaggle/input/dataset/train"
    VAL_DIR = "/kaggle/input/dataset/validation_ignore"  # Force train/val split
    TEST_DIR = "/kaggle/input/dataset/test"

    # Training parameters - MAXIMUM GPU UTILIZATION
    BATCH_SIZE = 256  # Much larger batch for small dataset
    LEARNING_RATE = 5e-4  # Increased for larger batch
    EPOCHS = 15
    IMAGE_SIZE = 224
    NUM_WORKERS = 2  # Reduced workers for small validation set

    # Model parameters
    MODEL_NAME = "mobilenetv3_small_100"  # Lightweight model ~10MB
    NUM_CLASSES = 2  # 0: Normal, 1: OSCC (Oral Squamous Cell Carcinoma)

    # Output
    MODEL_SAVE_PATH = "/kaggle/working/oral_cancer_model.pth"
    RESULTS_PATH = "/kaggle/working/oral_cancer_results.txt"

    # Class names for better readability
    CLASS_NAMES = ["Normal", "OSCC"]


class OralCancerDataset(Dataset):
    """Custom dataset for oral cancer histopathological images"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load images and labels
        self._load_data()

    def _load_data(self):
        """Load image paths and labels from directory structure"""

        # Expected structure: data_dir/Normal/ and data_dir/OSCC/
        class_dirs = {"Normal": 0, "OSCC": 1}

        for class_name, class_id in class_dirs.items():
            class_path = os.path.join(self.data_dir, class_name)

            if not os.path.exists(class_path):
                print(f"Warning: {class_name} directory not found at {class_path}")
                continue

            # Get all image files
            image_files = []
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                for file in os.listdir(class_path):
                    if file.lower().endswith(ext.lower()):
                        image_files.append(file)

            print(f"Found {len(image_files)} {class_name} images")

            for img_file in image_files:
                self.image_paths.append(os.path.join(class_path, img_file))
                self.labels.append(class_id)

        print(f"Total images loaded: {len(self.image_paths)}")
        if len(self.labels) > 0:
            print(f"Class distribution: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image with error handling
        image_path = self.image_paths[idx]
        try:
            # Use faster image loading
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a dummy image in case of error
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_transforms():
    """Define optimized data augmentation transforms for training and validation"""

    # Optimized transforms - less CPU intensive
    train_transforms = transforms.Compose(
        [
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # Reduced rotation for speed
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transforms, val_transforms


def create_model():
    """Create MobileNetV3-Small model with custom classifier"""

    print("Creating model...")

    # Load pretrained MobileNetV3-Small
    model = timm.create_model(Config.MODEL_NAME, pretrained=True)

    # Replace the classifier
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2), nn.Linear(num_features, Config.NUM_CLASSES)
    )

    # Move to device
    model = model.to(device)

    print(f"Model created: {Config.MODEL_NAME}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Estimated model size: ~{sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.1f}MB"
    )

    return model


def train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler=None):
    """Train for one epoch with mixed precision and gradient clipping"""

    from torch.cuda.amp import autocast

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {batch_idx}")
                continue

        # Mixed precision backward pass with gradient clipping
        scaler.scale(loss).backward()

        # Gradient clipping to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Step scheduler after optimizer step (for CosineAnnealingLR)
        if scheduler is not None and batch_idx == len(train_loader) - 1:
            scheduler.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        accuracy = 100.0 * correct / total
        progress_bar.set_postfix(
            {
                "Loss": f"{total_loss/(batch_idx+1):.4f}",
                "Acc": f"{accuracy:.2f}%",
                "LR": f'{optimizer.param_groups[0]["lr"]:.2e}',
            }
        )

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion):
    """Validate for one epoch with mixed precision and NaN protection"""

    from torch.cuda.amp import autocast

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    valid_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")

        for images, labels in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Skip NaN losses
            if torch.isnan(loss):
                print("Warning: NaN loss in validation, skipping batch")
                continue

            valid_batches += 1
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            accuracy = 100.0 * correct / total
            avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
            progress_bar.set_postfix(
                {"Loss": f"{avg_loss:.4f}", "Acc": f"{accuracy:.2f}%"}
            )

    epoch_loss = total_loss / max(valid_batches, 1)
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc, all_predictions, all_labels


def main():
    """Main training function"""

    print("Starting Oral Cancer Histopathological Detection Training")
    print("=" * 70)

    # GPU and system info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # Check available CPU cores
    import multiprocessing

    print(f"CPU cores available: {multiprocessing.cpu_count()}")

    # Enable optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes

    # Check if dataset directories exist and determine validation strategy
    use_validation_split = False

    if not os.path.exists(Config.TRAIN_DIR):
        print(f"ERROR: Training directory not found: {Config.TRAIN_DIR}")
        return

    if not os.path.exists(Config.VAL_DIR):
        print(f"Validation directory not found: {Config.VAL_DIR}")
        print("Will create validation split from training data (80/20 split)")
        use_validation_split = True
    else:
        print("Found separate validation directory")
        use_validation_split = False

    # Get transforms
    train_transforms, val_transforms = get_data_transforms()

    # Create datasets based on validation strategy
    if use_validation_split:
        print("\nLoading data and creating train/validation split...")
        # Load all data from training directory
        full_dataset = OralCancerDataset(Config.TRAIN_DIR, None)  # No transforms yet

        if len(full_dataset) == 0:
            print("ERROR: No images found in training dataset!")
            return

        # Create train/validation split (80/20)
        from sklearn.model_selection import train_test_split

        # Get indices for each class to ensure balanced split
        class_0_indices = [
            i for i, label in enumerate(full_dataset.labels) if label == 0
        ]
        class_1_indices = [
            i for i, label in enumerate(full_dataset.labels) if label == 1
        ]

        # Split each class separately
        train_idx_0, val_idx_0 = train_test_split(
            class_0_indices, test_size=0.2, random_state=42
        )
        train_idx_1, val_idx_1 = train_test_split(
            class_1_indices, test_size=0.2, random_state=42
        )

        train_indices = train_idx_0 + train_idx_1
        val_indices = val_idx_0 + val_idx_1

        # Create subset datasets
        from torch.utils.data import Subset

        # Create train dataset with transforms
        train_dataset_full = OralCancerDataset(Config.TRAIN_DIR, train_transforms)
        train_dataset = Subset(train_dataset_full, train_indices)

        # Create validation dataset with transforms
        val_dataset_full = OralCancerDataset(Config.TRAIN_DIR, val_transforms)
        val_dataset = Subset(val_dataset_full, val_indices)

        print(
            f"Created train/val split: {len(train_dataset)} train, {len(val_dataset)} validation images"
        )

    else:
        # Use separate directories
        print("\nLoading training data...")
        train_dataset = OralCancerDataset(Config.TRAIN_DIR, train_transforms)

        print("\nLoading validation data...")
        val_dataset = OralCancerDataset(Config.VAL_DIR, val_transforms)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: No images found in dataset!")
        return

    # Create data loaders - OPTIMIZED FOR SMALL VALIDATION SET
    print("\nCreating data loaders...")

    # Adjust workers based on dataset size
    optimal_workers = (
        2
        if len(val_dataset) < 200
        else min(multiprocessing.cpu_count(), Config.NUM_WORKERS)
    )
    print(f"Using {optimal_workers} workers for data loading")
    print(
        f"Warning: Very small validation set ({len(val_dataset)} samples) - results may be unreliable"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True if optimal_workers > 0 else False,
        prefetch_factor=2,  # Reduced for small dataset
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=min(
            Config.BATCH_SIZE, len(val_dataset)
        ),  # Don't exceed dataset size
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True if optimal_workers > 0 else False,
        prefetch_factor=2,
        drop_last=False,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    model = create_model()

    # Loss function and optimizer - STABILIZED TO PREVENT NaN
    # Use class weights if dataset is imbalanced
    train_class_counts = np.bincount(
        train_dataset.labels
        if not use_validation_split
        else [train_dataset_full.labels[i] for i in train_indices]
    )
    if len(train_class_counts) == 2:
        total = sum(train_class_counts)
        # Moderate class weights to prevent instability
        weight_0 = min(total / train_class_counts[0], 3.0)  # Cap at 3x
        weight_1 = min(total / train_class_counts[1], 3.0)
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(
            device
        )
        print(f"Using capped class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=0.1
        )  # Add label smoothing
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Enable mixed precision for faster training
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()

    # More stable optimizer settings
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4, eps=1e-8
    )

    # More conservative learning rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-7
    )

    # Training loop
    print(f"\nStarting training for {Config.EPOCHS} epochs...")
    best_val_acc = 0.0
    training_history = []

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler
        )

        # Validate
        val_loss, val_acc, val_predictions, val_labels = validate_epoch(
            model, val_loader, criterion
        )

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Create serializable config dictionary
            config_dict = {
                "BATCH_SIZE": Config.BATCH_SIZE,
                "LEARNING_RATE": Config.LEARNING_RATE,
                "EPOCHS": Config.EPOCHS,
                "IMAGE_SIZE": Config.IMAGE_SIZE,
                "MODEL_NAME": Config.MODEL_NAME,
                "NUM_CLASSES": Config.NUM_CLASSES,
                "CLASS_NAMES": Config.CLASS_NAMES,
            }

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": config_dict,
                },
                Config.MODEL_SAVE_PATH,
            )
            print(f"  🎯 New best model saved! Val Acc: {val_acc:.2f}%")

        # Store history
        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time": epoch_time,
            }
        )

    # Final results
    print("\n" + "=" * 70)
    print("🏆 TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Classification report for final epoch
    print("\nFinal Validation Classification Report:")
    print(
        classification_report(
            val_labels, val_predictions, target_names=Config.CLASS_NAMES
        )
    )

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_predictions)
    print(f"\nConfusion Matrix:")
    print(f"             Predicted")
    print(f"Actual    Normal  OSCC")
    print(f"Normal    {cm[0,0]:6d}  {cm[0,1]:4d}")
    print(f"OSCC      {cm[1,0]:6d}  {cm[1,1]:4d}")

    # Save training results
    with open(Config.RESULTS_PATH, "w") as f:
        f.write("Oral Cancer Histopathological Detection - Training Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: Normal epithelium vs OSCC\n")
        f.write(f"Model: {Config.MODEL_NAME}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n\n")
        f.write("Training History:\n")
        for h in training_history:
            f.write(
                f"Epoch {h['epoch']:2d}: Train={h['train_acc']:5.2f}%, "
                f"Val={h['val_acc']:5.2f}%, Time={h['time']:5.1f}s\n"
            )
        f.write("\nFinal Classification Report:\n")
        f.write(
            classification_report(
                val_labels, val_predictions, target_names=Config.CLASS_NAMES
            )
        )
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"             Predicted\n")
        f.write(f"Actual    Normal  OSCC\n")
        f.write(f"Normal    {cm[0,0]:6d}  {cm[0,1]:4d}\n")
        f.write(f"OSCC      {cm[1,0]:6d}  {cm[1,1]:4d}\n")

    print(f"\n📁 Files saved:")
    print(f"   Model: {Config.MODEL_SAVE_PATH}")
    print(f"   Results: {Config.RESULTS_PATH}")
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
