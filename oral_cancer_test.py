#!/usr/bin/env python3
"""
Oral Cancer Histopathological Detection Testing Script
Tests the trained model on test dataset with limited samples for speed
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
import time
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")


class OralCancerPredictor:
    """Oral cancer detection predictor class"""

    def __init__(self, model_path, device="auto"):
        """
        Initialize the predictor

        Args:
            model_path (str): Path to the saved model file
            device (str): Device to use ('auto', 'cuda', 'cpu')
        """
        self.device = self._setup_device(device)
        self.model = None
        self.transform = None
        self.model_path = model_path
        self.class_names = ["Normal", "OSCC"]

        self._load_model()
        self._setup_transforms()

        print(f"🔬 Oral Cancer Predictor initialized on device: {self.device}")

    def _setup_device(self, device):
        """Setup computation device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self):
        """Load the trained model"""
        print("Loading model...")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Get model configuration
        if "config" in checkpoint:
            model_name = checkpoint["config"].get("MODEL_NAME", "mobilenetv3_small_100")
            num_classes = checkpoint["config"].get("NUM_CLASSES", 2)
            class_names = checkpoint["config"].get("CLASS_NAMES", ["Normal", "OSCC"])
        else:
            # Default values if config not found
            model_name = "mobilenetv3_small_100"
            num_classes = 2
            class_names = ["Normal", "OSCC"]

        self.class_names = class_names

        # Create model architecture
        self.model = timm.create_model(model_name, pretrained=False)

        # Replace classifier to match training
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(num_features, num_classes)
        )

        # Load trained weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded successfully from {self.model_path}")
        if "val_acc" in checkpoint:
            print(f"   Training validation accuracy: {checkpoint['val_acc']:.2f}%")

    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict_image(self, image_path):
        """
        Predict cancer probability for a single image

        Args:
            image_path (str): Path to the image file

        Returns:
            dict: Prediction results with probabilities and class
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            return {
                "image_path": image_path,
                "predicted_class": predicted_class,
                "predicted_label": self.class_names[predicted_class],
                "confidence": confidence,
                "probabilities": {
                    "normal": probabilities[0][0].item(),
                    "oscc": probabilities[0][1].item(),
                },
            }
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return None

    def test_on_dataset(self, test_dir, max_images_per_class=50):
        """
        Test model on organized test dataset

        Args:
            test_dir (str): Path to test directory with Normal/OSCC subdirectories
            max_images_per_class (int): Maximum images per class to test (for speed)

        Returns:
            dict: Test results with accuracy, predictions, and metrics
        """
        print(f"🧪 Testing on dataset: {test_dir}")
        print(f"   Limited to {max_images_per_class} images per class for speed")

        # Collect test images
        test_images = []
        true_labels = []

        class_dirs = {"Normal": 0, "OSCC": 1}

        for class_name, class_id in class_dirs.items():
            class_path = os.path.join(test_dir, class_name)

            if not os.path.exists(class_path):
                print(f"⚠️  Warning: {class_name} directory not found at {class_path}")
                continue

            # Get image files
            image_files = []
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                for file in os.listdir(class_path):
                    if file.lower().endswith(ext.lower()):
                        image_files.append(os.path.join(class_path, file))

            # Limit number of images per class
            if len(image_files) > max_images_per_class:
                image_files = image_files[:max_images_per_class]

            print(f"   {class_name}: {len(image_files)} images")

            for img_path in image_files:
                test_images.append(img_path)
                true_labels.append(class_id)

        if not test_images:
            print("❌ No test images found!")
            return None

        print(f"\n📊 Testing on {len(test_images)} images...")

        # Predict on all test images
        predictions = []
        predicted_labels = []

        for img_path in tqdm(test_images, desc="Testing images"):
            result = self.predict_image(img_path)
            if result:
                predictions.append(result)
                predicted_labels.append(result["predicted_class"])
            else:
                predicted_labels.append(-1)  # Error case

        # Calculate metrics
        valid_predictions = [
            p for p, t in zip(predicted_labels, true_labels) if p != -1
        ]
        valid_true_labels = [
            t for p, t in zip(predicted_labels, true_labels) if p != -1
        ]

        if not valid_predictions:
            print("❌ No valid predictions made!")
            return None

        accuracy = accuracy_score(valid_true_labels, valid_predictions)

        print(f"\n🎯 Test Results:")
        print(f"   Total images tested: {len(valid_predictions)}")
        print(f"   Accuracy: {accuracy:.2%}")

        # Classification report
        print(f"\n📈 Classification Report:")
        report = classification_report(
            valid_true_labels,
            valid_predictions,
            target_names=self.class_names,
            zero_division=0,
        )
        print(report)

        # Confusion matrix
        cm = confusion_matrix(valid_true_labels, valid_predictions)
        print(f"\n🔢 Confusion Matrix:")
        print(f"             Predicted")
        print(f"Actual    Normal  OSCC")
        print(f"Normal    {cm[0,0]:6d}  {cm[0,1]:4d}")
        print(f"OSCC      {cm[1,0]:6d}  {cm[1,1]:4d}")

        # Sample predictions
        print(f"\n🔍 Sample Predictions:")
        for i in range(min(10, len(predictions))):
            pred = predictions[i]
            true_class = self.class_names[true_labels[i]]
            filename = Path(pred["image_path"]).name
            correct = "✅" if pred["predicted_class"] == true_labels[i] else "❌"
            print(
                f"   {correct} {filename}: {pred['predicted_label']} "
                f"({pred['confidence']:.1%}) [True: {true_class}]"
            )

        # Save results
        results_df = pd.DataFrame(predictions)
        results_df["true_label"] = [
            self.class_names[label] for label in true_labels[: len(predictions)]
        ]
        results_df["correct"] = (
            results_df["predicted_class"] == true_labels[: len(predictions)]
        )

        output_path = "/kaggle/working/oral_cancer_test_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n📁 Detailed results saved to: {output_path}")

        return {
            "accuracy": accuracy,
            "total_tested": len(valid_predictions),
            "predictions": predictions,
            "classification_report": report,
            "confusion_matrix": cm,
            "results_df": results_df,
        }


def main():
    """Main testing function"""

    # Configuration
    MODEL_PATH = "/kaggle/working/oral_cancer_model.pth"
    TEST_DIR = "/kaggle/input/dataset/test"

    print("🦷 Oral Cancer Histopathological Detection - Testing Script")
    print("=" * 70)

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("   Please run oral_cancer_train.py first to train the model.")
        return

    # Check if test directory exists
    if not os.path.exists(TEST_DIR):
        print(f"❌ Test directory not found: {TEST_DIR}")
        print("   Expected structure:")
        print("   /kaggle/input/dataset/test/Normal/")
        print("   /kaggle/input/dataset/test/OSCC/")
        return

    # Initialize predictor
    try:
        predictor = OralCancerPredictor(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Test single image prediction capability
    print(f"\n1️⃣  Testing single image prediction...")

    # Find a test image to demonstrate
    test_image_found = False
    for class_name in ["Normal", "OSCC"]:
        class_dir = os.path.join(TEST_DIR, class_name)
        if os.path.exists(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    test_img_path = os.path.join(class_dir, file)
                    result = predictor.predict_image(test_img_path)
                    if result:
                        print(f"   ✅ Sample prediction successful:")
                        print(f"      Image: {Path(test_img_path).name}")
                        print(
                            f"      Prediction: {result['predicted_label']} ({result['confidence']:.1%})"
                        )
                        test_image_found = True
                    break
        if test_image_found:
            break

    if not test_image_found:
        print(f"   ⚠️  Could not find test images for demonstration")

    # Test on full test dataset
    print(f"\n2️⃣  Testing on test dataset...")

    test_results = predictor.test_on_dataset(TEST_DIR, max_images_per_class=100)

    if test_results:
        print(f"\n✅ Testing completed successfully!")
        print(f"   Final accuracy: {test_results['accuracy']:.2%}")
        print(f"   Images tested: {test_results['total_tested']}")
    else:
        print(f"❌ Testing failed!")

    print(f"\n" + "=" * 70)
    print(f"🏁 TESTING COMPLETED")
    print(f"=" * 70)
    print(f"📁 Files generated:")
    print(f"   - oral_cancer_test_results.csv (detailed predictions)")
    print(f"\n🚀 Ready for local deployment!")
    print(f"   Use oral_cancer_streamlit.py for local testing")


if __name__ == "__main__":
    main()
