# OncoVision

**OncoVision** is a deep learning-based tool for analyzing histopathological images to detect **Oral Squamous Cell Carcinoma (OSCC)**. It leverages state-of-the-art convolutional neural networks to provide accurate, fast, and interpretable predictions from H&E stained oral tissue slides.

## Live Demo

Access the interactive Streamlit app here: [**OncoVision App**](https://oralcancerapp-eawaw4yueu6qoppbnpfdapp.streamlit.app/)  

> Upload images, analyze predictions, and explore model performance directly in your browser.

## About the Project

Oral cancer is among the most common cancers worldwide, and early detection is crucial for effective treatment. Histopathological image analysis is the gold standard, but manual diagnosis is time-consuming and requires expert pathologists.  

**OncoVision** automates this process using a deep learning pipeline that:

1. Preprocesses histopathology images for input  
2. Applies a **MobileNetV3-Small** model trained on thousands of images  
3. Classifies images into **Normal** or **OSCC**  
4. Provides confidence scores for each prediction  

This tool is intended for **research, education, and experimental purposes**, not for clinical diagnosis.

## Dataset

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/ashenafifasilkebede/dataset)  
- **Data Type**: H&E stained oral histopathology slides  
- **Magnification**: 100x and 400x  
- **Classes**:  
  - Normal – 2435 images  
  - OSCC – 2511 images  
- **Train/Validation Split**: 80/20 (3956 train, 990 validation)  
- **Test Set**: 126 images (Normal: 31, OSCC: 95)  

## Model Architecture

- **Backbone**: MobileNetV3-Small (pretrained on ImageNet)  
- **Classifier**: Linear layer with dropout  
- **Optimizer**: Adam with learning rate scheduling  
- **Data Augmentation**: Random rotations, flips, color jitter, affine transforms  
- **Input Size**: 224x224 RGB  
- **Parameters**: ~1.5 million  
- **Model Size**: ~5.8 MB  

## Training Details

- **Hardware**: GPU T4 (14.7 GB), 4 CPU cores  
- **Epochs**: 15  
- **Batch Size**: 32  
- **Training Accuracy**: 99.45%  
- **Validation Accuracy**: 98.48%  

**Validation Classification Metrics:**

| Class  | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| Normal | 0.98      | 0.99   | 0.98     |
| OSCC   | 0.99      | 0.98   | 0.98     |

**Confusion Matrix (Validation):**

| Actual \ Predicted | Normal | OSCC |
|-------------------|--------|------|
| Normal            | 480    | 7    |
| OSCC              | 9      | 494  |

## Testing Results

- **Test Accuracy**: 88.89%  
- **Test Dataset**: 126 images (Normal: 31, OSCC: 95)  

**Test Classification Metrics:**

| Class  | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| Normal | 0.77      | 0.77   | 0.77     |
| OSCC   | 0.93      | 0.93   | 0.93     |

**Confusion Matrix (Test Set):**

| Actual \ Predicted | Normal | OSCC |
|-------------------|--------|------|
| Normal            | 24     | 7    |
| OSCC              | 7      | 88   |

**Observations:**  
- The model performs extremely well on validation, with high precision and recall for both classes.  
- Test accuracy is slightly lower due to dataset variability but remains strong, especially for OSCC detection.  
- Lightweight architecture ensures fast inference (~0.05s per image).

## How to Use

1. Visit the [Streamlit App](https://oralcancerapp-eawaw4yueu6qoppbnpfdapp.streamlit.app/)  
2. Upload a histopathological image (JPEG/PNG)  
3. Click **Analyze Image**  
4. View predicted class and confidence score  

## Medical Disclaimer

**IMPORTANT**: OncoVision is for **research and educational purposes only**.  
- **Not for clinical diagnosis**  
- Always consult qualified medical professionals  
- Results are based on limited datasets  

## Technologies

- **PyTorch** – Deep learning framework  
- **timm** – Pre-trained models  
- **Streamlit** – Web interface  
- **PIL & NumPy** – Image processing and numerical computations  

## Acknowledgments

- Kaggle dataset contributors  
- Pathologists and medical experts for annotations  
- Open-source deep learning community for tools and libraries  
