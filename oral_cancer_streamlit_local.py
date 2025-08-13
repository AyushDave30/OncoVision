#!/usr/bin/env python3
"""
Oral Cancer Histopathological Detection - Streamlit App
Auto-loading deployment for testing and demonstration
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from PIL import Image
import numpy as np
import io
import time
import warnings
import os

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="🦷 Oral Cancer Detection",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class OralCancerPredictor:
    """Oral cancer detection predictor for Streamlit"""

    def __init__(self):
        self.device = torch.device("cpu")  # Use CPU for local deployment
        self.model = None
        self.transform = None
        self.class_names = ["Normal", "OSCC"]
        self.loaded = False
        self.val_acc = "Unknown"

        # Auto-detect model path
        self.model_paths = [
            "oral_cancer_model.pth",
            "model/oral_cancer_model.pth",
            "../model/oral_cancer_model.pth",
            "./model/oral_cancer_model.pth",
            r"D:\Xpython\Web App\Histo_patho_Cld\model\oral_cancer_model.pth",
            "/kaggle/working/oral_cancer_model.pth",
            "models/oral_cancer_model.pth",
        ]

    def find_model_path(self):
        """Find the model file automatically"""
        for path in self.model_paths:
            if os.path.exists(path):
                return path
        return None

    def load_model(self, model_path=None):
        """Load the trained model with PyTorch 2.6 compatibility"""
        try:
            # Auto-find model if path not provided
            if model_path is None:
                model_path = self.find_model_path()
                if model_path is None:
                    return False, "Model file not found in common locations"

            # Load checkpoint with PyTorch 2.6 compatibility
            try:
                # Try weights_only=False for compatibility with older checkpoints
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )
            except Exception as e:
                # Fallback for very old PyTorch versions
                checkpoint = torch.load(model_path, map_location=self.device)

            # Get model configuration
            if "config" in checkpoint:
                model_name = checkpoint["config"].get(
                    "MODEL_NAME", "mobilenetv3_small_100"
                )
                num_classes = checkpoint["config"].get("NUM_CLASSES", 2)
                class_names = checkpoint["config"].get(
                    "CLASS_NAMES", ["Normal", "OSCC"]
                )
                self.val_acc = checkpoint.get("val_acc", "Unknown")
            else:
                model_name = "mobilenetv3_small_100"
                num_classes = 2
                class_names = ["Normal", "OSCC"]

            self.class_names = class_names

            # Create model architecture
            self.model = timm.create_model(model_name, pretrained=False)

            # Replace classifier
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(num_features, num_classes)
            )

            # Load weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            # Setup transforms
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self.loaded = True
            return True, f"Model loaded successfully from {model_path}"

        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def predict(self, image):
        """Make prediction on uploaded image"""
        if not self.loaded:
            return None

        try:
            # Preprocess image
            if image.mode != "RGB":
                image = image.convert("RGB")

            image_tensor = self.transform(image).unsqueeze(0)

            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            return {
                "predicted_class": predicted_class,
                "predicted_label": self.class_names[predicted_class],
                "confidence": confidence,
                "probabilities": {
                    "normal": probabilities[0][0].item(),
                    "oscc": probabilities[0][1].item(),
                },
            }

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None


@st.cache_resource
def load_predictor():
    """Load predictor with caching for performance"""
    predictor = OralCancerPredictor()
    success, message = predictor.load_model()
    return predictor, success, message


def main():
    """Main Streamlit application"""

    # Header
    st.markdown(
        '<h1 class="main-header">🦷 Oral Cancer Detection</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">Histopathological Image Analysis using Deep Learning</p>',
        unsafe_allow_html=True,
    )

    # Auto-load model
    with st.spinner("🚀 Loading AI model..."):
        predictor, model_loaded, load_message = load_predictor()

    # Display model status
    if model_loaded:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### ✅ AI Model Ready")
        st.markdown(f"**Status:** {load_message}")
        if predictor.val_acc != "Unknown":
            st.markdown(f"**Training Accuracy:** {predictor.val_acc:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="danger-box">', unsafe_allow_html=True)
        st.markdown("### ❌ Model Loading Failed")
        st.markdown(f"**Error:** {load_message}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Show manual model upload option
        st.markdown("### 📁 Manual Model Upload")
        uploaded_model = st.file_uploader(
            "Upload your trained model (.pth file)",
            type=["pth"],
            help="Upload the oral_cancer_model.pth file",
        )

        if uploaded_model is not None:
            # Save uploaded model temporarily
            temp_path = "temp_model.pth"
            with open(temp_path, "wb") as f:
                f.write(uploaded_model.read())

            # Try to load the uploaded model
            success, message = predictor.load_model(temp_path)
            if success:
                st.success("✅ Model uploaded and loaded successfully!")
                model_loaded = True
                st.experimental_rerun()
            else:
                st.error(f"❌ Failed to load uploaded model: {message}")
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # Sidebar
    st.sidebar.title("📋 Information")
    st.sidebar.markdown(
        """
    ### About This Tool
    This application detects **Oral Squamous Cell Carcinoma (OSCC)** in histopathological images.
    
    ### Classes
    - **Normal**: Healthy oral epithelium
    - **OSCC**: Oral Squamous Cell Carcinoma
    
    ### Model Info
    - **Architecture**: MobileNetV3-Small
    - **Size**: ~10MB
    - **Input**: 224x224 RGB images
    
    ### How to Use
    1. Upload a histopathological image
    2. Click "Analyze Image"
    3. View the AI prediction results
    """
    )

    # Main prediction interface
    if model_loaded:
        st.markdown("---")
        st.markdown("### 🔬 Image Analysis")

        # File uploader
        uploaded_file = st.file_uploader(
            "📤 Upload a histopathological image",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
            help="Supported formats: JPG, PNG, BMP, TIFF",
        )

        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)

                # Image info
                with st.expander("📊 Image Details"):
                    st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
                    st.write(f"**Mode:** {image.mode}")
                    st.write(
                        f"**Format:** {image.format if hasattr(image, 'format') else 'Unknown'}"
                    )

            with col2:
                st.subheader("🎯 Analysis Results")

                # Auto-analyze or manual button
                analyze_mode = st.radio(
                    "Analysis Mode:",
                    ["Auto-analyze", "Manual analysis"],
                    horizontal=True,
                )

                should_analyze = False
                if analyze_mode == "Auto-analyze":
                    should_analyze = True
                    st.info("🤖 Auto-analyzing image...")
                else:
                    should_analyze = st.button("🔍 Analyze Image", type="primary")

                if should_analyze:
                    with st.spinner("🧠 AI is analyzing the image..."):
                        start_time = time.time()
                        result = predictor.predict(image)
                        inference_time = time.time() - start_time

                        if result:
                            # Main prediction
                            prediction_label = result["predicted_label"]
                            confidence = result["confidence"]

                            # Results display
                            if prediction_label == "OSCC":
                                st.markdown(
                                    '<div class="danger-box">', unsafe_allow_html=True
                                )
                                st.markdown("### ⚠️ OSCC Detected")
                                st.markdown(f"**Confidence:** {confidence:.1%}")
                                st.markdown(
                                    "**⚠️ Recommendation:** Consult an oncologist immediately"
                                )
                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    '<div class="success-box">', unsafe_allow_html=True
                                )
                                st.markdown("### ✅ Normal Tissue")
                                st.markdown(f"**Confidence:** {confidence:.1%}")
                                st.markdown(
                                    "**✅ Recommendation:** No abnormalities detected"
                                )
                                st.markdown("</div>", unsafe_allow_html=True)

                            # Detailed probabilities
                            st.markdown("### 📊 Detailed Probabilities")

                            normal_prob = result["probabilities"]["normal"]
                            oscc_prob = result["probabilities"]["oscc"]

                            # Progress bars for visual representation
                            st.progress(normal_prob, text=f"Normal: {normal_prob:.1%}")
                            st.progress(oscc_prob, text=f"OSCC: {oscc_prob:.1%}")

                            # Metrics in columns
                            col3, col4, col5 = st.columns(3)
                            with col3:
                                st.metric("Normal", f"{normal_prob:.1%}")
                            with col4:
                                st.metric("OSCC", f"{oscc_prob:.1%}")
                            with col5:
                                st.metric("Time", f"{inference_time:.2f}s")

                        else:
                            st.error("❌ Prediction failed. Please try another image.")

    # Disclaimer
    st.markdown("---")
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(
        """
    ### ⚠️ Medical Disclaimer
    
    **IMPORTANT:** This is a demonstration tool for educational purposes only. 
    
    - This tool should **NOT** be used for actual medical diagnosis
    - Always consult qualified medical professionals for health-related decisions
    - The predictions are based on a machine learning model trained on limited data
    - Results may not be accurate and should be verified by medical experts
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🔬 Oral Cancer Detection System | Built with Streamlit & PyTorch</p>
        <p>Model: MobileNetV3-Small | Dataset: Histopathological Images</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
