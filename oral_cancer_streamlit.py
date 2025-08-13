#!/usr/bin/env python3
"""
OncoVision: Oral Cancer Detection 
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from PIL import Image
import numpy as np
import time
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="OncoVision | Oral Cancer AI",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="auto",
)

# Clean custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background: #d4f6d4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .danger-box {
        background: #ffe6e6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .result-normal {
        background: #d4f6d4;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    .result-oscc {
        background: #ffe6e6;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class OralCancerPredictor:
    """Simplified Oral cancer detection predictor"""

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = None
        self.transform = None
        self.class_names = ["Normal", "OSCC"]
        self.loaded = False
        self.val_acc = "Unknown"

        # Model paths
        self.model_paths = [
            "oral_cancer_model.pth",
            "model/oral_cancer_model.pth",
            "models/oral_cancer_model.pth",
            "../model/oral_cancer_model.pth",
            "./model/oral_cancer_model.pth",
            "/app/model/oral_cancer_model.pth",
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
        """Load the trained model"""
        try:
            if model_path is None:
                model_path = self.find_model_path()
                if model_path is None:
                    return False, "❌ Model file not found. Please upload manually."

            if not os.path.exists(model_path):
                return False, f"❌ Model file not found at: {model_path}"

            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            if file_size < 1:
                return False, f"❌ Model file too small ({file_size:.1f}MB)."

            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            if "model_state_dict" not in checkpoint:
                return False, "❌ Invalid model file format."

            config = checkpoint.get("config", {})
            model_name = config.get("MODEL_NAME", "mobilenetv3_small_100")
            num_classes = config.get("NUM_CLASSES", 2)
            self.class_names = config.get("CLASS_NAMES", ["Normal", "OSCC"])
            self.val_acc = checkpoint.get("val_acc", "Unknown")

            # Create model
            self.model = timm.create_model(model_name, pretrained=False)
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
            success_msg = f"✅ Model loaded successfully! (Size: {file_size:.1f}MB)"
            if self.val_acc != "Unknown":
                success_msg += f"\n🎯 Validation Accuracy: {self.val_acc:.1f}%"

            return True, success_msg

        except Exception as e:
            return False, f"❌ Error loading model: {str(e)}"

    def predict(self, image):
        """Make prediction on uploaded image"""
        if not self.loaded:
            return None

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            image_tensor = self.transform(image).unsqueeze(0)

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
            st.error(f"❌ Prediction error: {str(e)}")
            return None


@st.cache_resource
def load_predictor():
    """Load predictor with caching"""
    predictor = OralCancerPredictor()
    success, message = predictor.load_model()
    return predictor, success, message


def main():
    """Main Streamlit application - simplified"""

    # Header
    st.markdown('<h1 class="main-header">🔬 OncoVision</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Powered Oral Cancer Detection from Histopathology Images</p>',
        unsafe_allow_html=True,
    )

    # Load model
    with st.spinner("🚀 Loading AI model..."):
        predictor, model_loaded, load_message = load_predictor()

    # Model status
    if model_loaded:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### ✅ AI Model Ready")
        st.markdown(load_message.replace("\n", "  \n"))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="danger-box">', unsafe_allow_html=True)
        st.markdown("### ❌ Model Not Found")
        st.markdown(load_message)
        st.markdown("</div>", unsafe_allow_html=True)

        # Manual upload option
        st.markdown("### 📁 Upload Model File")
        uploaded_model = st.file_uploader(
            "Choose model file (oral_cancer_model.pth)",
            type=["pth"],
            help="Upload your trained model file",
        )

        if uploaded_model is not None:
            with st.spinner("📤 Loading uploaded model..."):
                temp_path = f"temp_model_{int(time.time())}.pth"
                try:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_model.read())

                    success, message = predictor.load_model(temp_path)
                    if success:
                        st.success("✅ Model uploaded successfully!")
                        model_loaded = True
                        st.rerun()
                    else:
                        st.error(f"Failed to load model: {message}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass

    # Sidebar - simplified
    with st.sidebar:
        st.markdown("## 📋 Quick Info")

        st.markdown("### 🎯 What it does")
        st.markdown(
            "Analyzes histopathology images to detect Oral Squamous Cell Carcinoma (OSCC)"
        )

        st.markdown("### 🏷️ Classes")
        st.markdown("• **Normal**: Healthy tissue")
        st.markdown("• **OSCC**: Cancer detected")

        st.markdown("### 📤 Upload Requirements")
        st.markdown("• **Format**: JPG, PNG, TIFF")
        st.markdown("• **Type**: H&E stained slides")
        st.markdown("• **Size**: 224x224+ pixels")

        st.markdown("### ⚠️ Important")
        st.markdown("• For **research only**")
        st.markdown("• **Not for diagnosis**")
        st.markdown("• Consult medical experts")

    # Main interface
    if model_loaded:
        st.markdown("---")
        st.markdown("## 📤 Upload & Analyze")

        uploaded_file = st.file_uploader(
            "Choose a histopathology image",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
            help="Upload H&E stained oral tissue image",
        )

        if uploaded_file is not None:
            # Display image and analysis side by side
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 📷 Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

                # Basic image info
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.caption(
                    f"Size: {image.size[0]}x{image.size[1]} pixels | {file_size:.1f}MB"
                )

            with col2:
                st.markdown("### 🧠 Analysis")

                if st.button(
                    "🔍 Analyze Image", type="primary", use_container_width=True
                ):
                    with st.spinner("Analyzing..."):
                        start_time = time.time()
                        result = predictor.predict(image)
                        inference_time = time.time() - start_time

                    if result:
                        prediction_label = result["predicted_label"]
                        confidence = result["confidence"]

                        # Results
                        if prediction_label == "OSCC":
                            st.markdown(
                                '<div class="result-oscc">', unsafe_allow_html=True
                            )
                            st.markdown(f"### ⚠️ OSCC Detected")
                            st.markdown(f"**Confidence:** {confidence:.1%}")
                            st.markdown("**Action:** Consult oncologist immediately")
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(
                                '<div class="result-normal">', unsafe_allow_html=True
                            )
                            st.markdown(f"### ✅ Normal Tissue")
                            st.markdown(f"**Confidence:** {confidence:.1%}")
                            st.markdown("**Status:** No cancer detected")
                            st.markdown("</div>", unsafe_allow_html=True)

                        # Quick stats
                        st.markdown("### 📊 Details")

                        col3, col4 = st.columns(2)
                        with col3:
                            st.metric(
                                "Normal", f"{result['probabilities']['normal']:.1%}"
                            )
                        with col4:
                            st.metric("OSCC", f"{result['probabilities']['oscc']:.1%}")

                        st.metric("Processing Time", f"{inference_time:.2f}s")

                        # Additional info
                        with st.expander("📝 Understanding Results"):
                            if prediction_label == "OSCC":
                                st.markdown(
                                    """
                                **OSCC (Oral Squamous Cell Carcinoma)** detected:
                                - Most common type of oral cancer
                                - Requires immediate medical attention
                                - Early detection improves outcomes
                                
                                **Next steps:**
                                - Contact your doctor immediately
                                - Schedule oncology consultation
                                - Get second opinion if needed
                                """
                                )
                            else:
                                st.markdown(
                                    """
                                **Normal tissue** characteristics:
                                - Healthy cellular structure observed
                                - No obvious cancer signs detected
                                - Regular monitoring recommended
                                
                                **Recommendations:**
                                - Continue routine checkups
                                - Maintain good oral hygiene
                                - Annual oral cancer screenings
                                """
                                )

                    else:
                        st.error("❌ Analysis failed. Try a different image.")

    # Medical disclaimer - simplified but prominent
    st.markdown("---")
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(
        """
    ### ⚠️ MEDICAL DISCLAIMER
    
    **This is an AI research tool - NOT for medical diagnosis.**
    
    🚫 **Do not use for clinical decisions**  
    👩‍⚕️ **Always consult qualified doctors**  
    🔬 **For research and education only**
    
    False results are possible. Medical professionals must validate all findings.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Simple footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p><strong>OncoVision</strong> | Built with Streamlit & PyTorch</p>
            <p>For research and educational purposes</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
