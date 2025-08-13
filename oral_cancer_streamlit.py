#!/usr/bin/env python3
"""
Oral Cancer Histopathological Detection - Streamlit App
Fixed version with improved functionality and user experience
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
from pathlib import Path

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
    .stProgress > div > div > div {
        background-image: linear-gradient(to right, #1f77b4, #28a745);
    }
</style>
""",
    unsafe_allow_html=True,
)


class OralCancerPredictor:
    """Oral cancer detection predictor for Streamlit"""

    def __init__(self):
        self.device = torch.device("cpu")  # Use CPU for web deployment
        self.model = None
        self.transform = None
        self.class_names = ["Normal", "OSCC"]
        self.loaded = False
        self.val_acc = "Unknown"
        self.model_info = {}

        # Comprehensive model path search
        self.model_paths = [
            "oral_cancer_model.pth",
            "model/oral_cancer_model.pth",
            "models/oral_cancer_model.pth",
            "../model/oral_cancer_model.pth",
            "./model/oral_cancer_model.pth",
            "/app/model/oral_cancer_model.pth",  # Docker path
            "/opt/conda/lib/python3.*/site-packages/model/oral_cancer_model.pth",
            r"D:\Xpython\Web App\Histo_patho_Cld\model\oral_cancer_model.pth",
            "/kaggle/working/oral_cancer_model.pth",
            os.path.expanduser("~/oral_cancer_model.pth"),
        ]

    def find_model_path(self):
        """Find the model file automatically"""
        for path in self.model_paths:
            if os.path.exists(path):
                return path
        return None

    def load_model(self, model_path=None):
        """Load the trained model with comprehensive error handling"""
        try:
            # Auto-find model if path not provided
            if model_path is None:
                model_path = self.find_model_path()
                if model_path is None:
                    return (
                        False,
                        "❌ Model file not found in any common locations. Please upload the model manually.",
                    )

            if not os.path.exists(model_path):
                return False, f"❌ Model file not found at: {model_path}"

            # Check file size for validation
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            if file_size < 1:
                return (
                    False,
                    f"❌ Model file seems too small ({file_size:.1f}MB). May be corrupted.",
                )

            # Load checkpoint with improved compatibility
            try:
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=False
                )
            except Exception as load_error:
                # Try alternative loading method
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                except Exception:
                    return (
                        False,
                        f"❌ Failed to load model file. Error: {str(load_error)}",
                    )

            # Validate checkpoint structure
            if "model_state_dict" not in checkpoint:
                return False, "❌ Invalid model file format. Missing model_state_dict."

            # Extract configuration with defaults
            config = checkpoint.get("config", {})
            model_name = config.get("MODEL_NAME", "mobilenetv3_small_100")
            num_classes = config.get("NUM_CLASSES", 2)
            self.class_names = config.get("CLASS_NAMES", ["Normal", "OSCC"])
            self.val_acc = checkpoint.get("val_acc", "Unknown")

            # Store model info
            self.model_info = {
                "model_name": model_name,
                "num_classes": num_classes,
                "file_size": f"{file_size:.1f}MB",
                "path": model_path,
                "validation_accuracy": self.val_acc,
            }

            # Create model architecture
            try:
                self.model = timm.create_model(model_name, pretrained=False)
            except Exception as e:
                return (
                    False,
                    f"❌ Failed to create model architecture '{model_name}'. Error: {str(e)}",
                )

            # Replace classifier
            try:
                num_features = self.model.classifier.in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(0.2), nn.Linear(num_features, num_classes)
                )
            except Exception as e:
                return False, f"❌ Failed to modify model classifier. Error: {str(e)}"

            # Load weights
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
            except Exception as e:
                return (
                    False,
                    f"❌ Failed to load model weights. Model architecture mismatch. Error: {str(e)}",
                )

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
            success_msg = f"✅ Model loaded successfully!\n📁 Path: {Path(model_path).name}\n📊 Size: {file_size:.1f}MB"
            if self.val_acc != "Unknown":
                success_msg += f"\n🎯 Training Accuracy: {self.val_acc:.2f}%"

            return True, success_msg

        except Exception as e:
            return False, f"❌ Unexpected error loading model: {str(e)}"

    def predict(self, image):
        """Make prediction on uploaded image with comprehensive error handling"""
        if not self.loaded:
            return None

        try:
            # Validate image
            if image is None:
                return None

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Check image size
            if min(image.size) < 50:
                st.warning("⚠️ Image seems very small. Results may not be accurate.")

            # Preprocess image
            image_tensor = self.transform(image).unsqueeze(0)

            # Predict with error handling
            with torch.no_grad():
                outputs = self.model(image_tensor)

                # Check for NaN outputs
                if torch.isnan(outputs).any():
                    st.error(
                        "❌ Model produced invalid outputs. Please try a different image."
                    )
                    return None

                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            # Validate predictions
            if not (0 <= predicted_class < len(self.class_names)):
                st.error("❌ Invalid prediction class. Model may be corrupted.")
                return None

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
            st.error(f"❌ Prediction error: {str(e)}")
            return None


@st.cache_resource
def load_predictor():
    """Load predictor with caching for performance"""
    predictor = OralCancerPredictor()
    success, message = predictor.load_model()
    return predictor, success, message


def display_model_info(predictor):
    """Display detailed model information"""
    if predictor.loaded and predictor.model_info:
        with st.expander("🔧 Model Technical Details"):
            info = predictor.model_info
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Architecture:** {info['model_name']}")
                st.write(f"**Classes:** {info['num_classes']}")
                st.write(f"**File Size:** {info['file_size']}")

            with col2:
                st.write(f"**Input Size:** 224×224 RGB")
                st.write(f"**Framework:** PyTorch + TIMM")
                if info["validation_accuracy"] != "Unknown":
                    st.write(f"**Val Accuracy:** {info['validation_accuracy']:.2f}%")


def main():
    """Main Streamlit application"""

    # Header
    st.markdown(
        '<h1 class="main-header">🦷 Oral Cancer Detection</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Histopathological Image Analysis</p>',
        unsafe_allow_html=True,
    )

    # Load model with progress indicator
    model_load_placeholder = st.empty()
    with model_load_placeholder.container():
        with st.spinner("🚀 Loading AI model..."):
            predictor, model_loaded, load_message = load_predictor()

    model_load_placeholder.empty()

    # Display model status
    status_container = st.container()
    with status_container:
        if model_loaded:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("### ✅ AI Model Ready")
            st.markdown(load_message.replace("\n", "  \n"))
            st.markdown("</div>", unsafe_allow_html=True)

            # Display technical details
            display_model_info(predictor)

        else:
            st.markdown('<div class="danger-box">', unsafe_allow_html=True)
            st.markdown("### ❌ Model Loading Failed")
            st.markdown(load_message)
            st.markdown("</div>", unsafe_allow_html=True)

            # Manual model upload interface
            st.markdown("---")
            st.markdown("### 📁 Upload Model File")
            st.markdown(
                "Please upload your trained model file (`oral_cancer_model.pth`) to continue:"
            )

            uploaded_model = st.file_uploader(
                "Choose model file",
                type=["pth"],
                help="Upload the oral_cancer_model.pth file generated during training",
                accept_multiple_files=False,
            )

            if uploaded_model is not None:
                with st.spinner("📤 Uploading and loading model..."):
                    # Save uploaded model temporarily
                    temp_path = f"temp_model_{int(time.time())}.pth"
                    try:
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_model.read())

                        # Try to load the uploaded model
                        success, message = predictor.load_model(temp_path)

                        if success:
                            st.success("✅ Model uploaded and loaded successfully!")
                            model_loaded = True
                            st.rerun()  # Refresh the page
                        else:
                            st.error(f"Failed to load uploaded model:\n{message}")
                    except Exception as e:
                        st.error(f"Error saving uploaded model: {str(e)}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except:
                                pass

    # Sidebar
    with st.sidebar:
        st.title("📋 Information")

        st.markdown("### 🔬 About This Tool")
        st.markdown(
            """
        This AI system analyzes histopathological images to detect **Oral Squamous Cell Carcinoma (OSCC)**.
        """
        )

        st.markdown("### 🏷️ Classification Classes")
        st.markdown("- **Normal**: Healthy oral epithelium")
        st.markdown("- **OSCC**: Oral Squamous Cell Carcinoma")

        st.markdown("### 🤖 Model Information")
        st.markdown("- **Architecture**: MobileNetV3-Small")
        st.markdown("- **Model Size**: ~10MB")
        st.markdown("- **Input**: 224×224 RGB images")
        st.markdown("- **Inference**: ~0.1-0.5 seconds")

        st.markdown("### 📝 Usage Instructions")
        st.markdown(
            """
        1. **Upload Image**: Select a histopathological image
        2. **Analyze**: Click the analyze button
        3. **Review Results**: Check AI predictions
        4. **Consult Expert**: Always verify with medical professionals
        """
        )

        st.markdown("### ⚠️ Important Notes")
        st.markdown(
            """
        - For **demonstration purposes only**
        - Not for clinical diagnosis
        - Always consult medical experts
        - Best with H&E stained slides
        """
        )

    # Main prediction interface
    if model_loaded:
        st.markdown("---")
        st.markdown("### 🔬 Image Analysis")

        # File uploader with better validation
        uploaded_file = st.file_uploader(
            "📤 Upload histopathological image",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
            help="Supported formats: JPG, JPEG, PNG, BMP, TIFF. Recommended: H&E stained histopathological slides",
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            # Validate file size
            file_size = len(uploaded_file.read()) / (1024 * 1024)  # MB
            uploaded_file.seek(0)  # Reset file pointer

            if file_size > 10:
                st.warning(f"⚠️ Large file ({file_size:.1f}MB). Processing may be slow.")
            elif file_size < 0.01:
                st.warning("⚠️ Very small file. May not be a valid image.")

            try:
                # Load and display image
                image = Image.open(uploaded_file)

                # Create two columns for image and results
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📷 Original Image")
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                    # Image metadata
                    with st.expander("📊 Image Information"):
                        st.write(
                            f"**Dimensions:** {image.size[0]} × {image.size[1]} pixels"
                        )
                        st.write(f"**Color Mode:** {image.mode}")
                        st.write(f"**Format:** {getattr(image, 'format', 'Unknown')}")
                        st.write(f"**File Size:** {file_size:.2f} MB")

                with col2:
                    st.subheader("🧠 AI Analysis")

                    # Single analyze button - FIXED UX ISSUE
                    if st.button(
                        "🔍 Analyze Image", type="primary", use_container_width=True
                    ):

                        # Progress indicator
                        with st.spinner(
                            "🧠 AI is analyzing the histopathological image..."
                        ):
                            start_time = time.time()
                            result = predictor.predict(image)
                            inference_time = time.time() - start_time

                        if result:
                            prediction_label = result["predicted_label"]
                            confidence = result["confidence"]

                            # Display results with improved styling
                            if prediction_label == "OSCC":
                                st.markdown(
                                    '<div class="danger-box">', unsafe_allow_html=True
                                )
                                st.markdown("### ⚠️ OSCC Detected")
                                st.markdown(f"**Confidence Level:** {confidence:.1%}")
                                st.markdown(
                                    "**⚠️ Recommendation:** Immediate consultation with an oncologist required"
                                )
                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    '<div class="success-box">', unsafe_allow_html=True
                                )
                                st.markdown("### ✅ Normal Tissue")
                                st.markdown(f"**Confidence Level:** {confidence:.1%}")
                                st.markdown(
                                    "**✅ Observation:** No cancerous abnormalities detected"
                                )
                                st.markdown("</div>", unsafe_allow_html=True)

                            # Detailed probability breakdown
                            st.markdown("### 📊 Detailed Analysis")

                            normal_prob = result["probabilities"]["normal"]
                            oscc_prob = result["probabilities"]["oscc"]

                            # Visual probability bars
                            col3, col4 = st.columns(2)
                            with col3:
                                st.metric(
                                    "Normal Tissue", f"{normal_prob:.1%}", delta=None
                                )
                                st.progress(normal_prob)

                            with col4:
                                st.metric("OSCC", f"{oscc_prob:.1%}", delta=None)
                                st.progress(oscc_prob)

                            # Performance metrics
                            st.markdown("### ⚡ Performance")
                            perf_col1, perf_col2, perf_col3 = st.columns(3)

                            with perf_col1:
                                st.metric("Inference Time", f"{inference_time:.3f}s")
                            with perf_col2:
                                st.metric("Model Confidence", f"{confidence:.1%}")
                            with perf_col3:
                                certainty = (
                                    "High"
                                    if confidence > 0.8
                                    else "Medium" if confidence > 0.6 else "Low"
                                )
                                st.metric("Certainty Level", certainty)

                            # Additional information based on prediction
                            with st.expander("🔬 Understanding the Results"):
                                if prediction_label == "OSCC":
                                    st.markdown(
                                        """
                                    **Oral Squamous Cell Carcinoma (OSCC)** characteristics detected:
                                    - Abnormal cellular architecture
                                    - Potential malignant transformation
                                    - Requires immediate medical attention
                                    
                                    **Next Steps:**
                                    - Consult with an oncologist immediately
                                    - Additional histopathological review
                                    - Consider staging and treatment planning
                                    """
                                    )
                                else:
                                    st.markdown(
                                        """
                                    **Normal tissue characteristics observed:**
                                    - Regular cellular organization
                                    - No obvious malignant features
                                    - Healthy epithelial structure
                                    
                                    **Recommendation:**
                                    - Continue routine monitoring
                                    - Follow standard oral health practices
                                    - Regular dental checkups recommended
                                    """
                                    )

                        else:
                            st.error(
                                "❌ Analysis failed. Please try uploading a different image or check the model."
                            )

            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                st.info(
                    "💡 Please ensure the uploaded file is a valid image in supported format."
                )

    # Medical disclaimer - prominently displayed
    st.markdown("---")
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(
        """
    ### ⚠️ IMPORTANT MEDICAL DISCLAIMER
    
    **This is a demonstration tool for educational and research purposes only.**
    
    🚫 **NOT for clinical diagnosis or treatment decisions**
    
    - This AI model is trained on limited datasets
    - Results require validation by qualified medical professionals
    - Always consult licensed healthcare providers for medical advice
    - False positives and false negatives are possible
    - This tool does not replace expert histopathological examination
    
    **For medical emergencies, contact your healthcare provider immediately.**
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🔬 <strong>Oral Cancer Detection System</strong></p>
        <p>Built with Streamlit • PyTorch • MobileNetV3 • TIMM</p>
        <p>For research and educational purposes</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
