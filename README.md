# Bournemouth University Campus Pathfinder

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg)
![YOLOv5](https://img.shields.io/badge/YOLO-v5-%2300A98F)
![License](https://img.shields.io/badge/License-MIT-green)

A computer vision project aimed at detecting and classifying buildings on Bournemouth University's Talbot Campus. This repository contains the code for a two-step ResNet-based classifier/detector and a state-of-the-art YOLOv5 object detection model, trained to recognize eight key buildings under varying conditions.

**Live Demo (Google Colab):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your_username/bu-campus-pathfinder/blob/main/Bournemouth_University_Campus_Pathfinder.ipynb)

---

## 📖 Project Overview

Navigating a university campus can be challenging for new students and visitors. Static maps lack the dynamic, context-aware capabilities needed for intuitive wayfinding. This project lays the groundwork for an AI-powered "Campus Pathfinder" application that could use a smartphone's camera and GPS to identify nearby buildings and provide navigation cues.

The core objective was to develop a robust model capable of detecting and classifying the following eight buildings with high accuracy:
- Poole House
- The Sir Michael Cobham Library
- Kimmeridge House
- Dorset House
- Weymouth House
- The Fusion Building
- Christchurch House
- Poole Gateway Building

---

## 🧪 Technical Approach

We implemented and compared two distinct deep-learning paradigms:

### 1. Two-Step Classification & Detection (ResNet-Based)
- **Step 1: Image Classification:** Benchmarking of multiple pre-trained architectures (ResNet18, 34, 50, 101, EfficientNet-B0) to identify the most effective backbone for our dataset.
- **Step 2: Bounding Box Regression:** Extending the best classifier (ResNet18) with a custom detection head to perform simultaneous classification and bounding box prediction. The dataset was converted into a Pascal VOC-like format for this task.

### 2. End-to-End Object Detection (YOLOv5)
- Fine-tuning the pre-trained `YOLOv5s` model on our custom dataset.
- Two training regimens were compared:
  - **Base Model:** Trained on the original annotated dataset.
  - **Augmented Model:** Trained on a 3x augmented dataset (using Roboflow) with transformations like shear, rotation, and brightness adjustments to improve robustness to real-world variance.

---

## 🗃️ Dataset

The project utilized a meticulously curated dataset:
- **Total Images:** 256
- **Class Balance:** 32 images per building (16 daytime, 16 nighttime)
- **Challenges:** Includes occlusions (trees, structures) and varying viewpoints to enhance model robustness.
- **Annotation:** Annotations were created using the VGG Image Annotator (VIA), initially including occlusions but finalized with a focus on the primary building subject.

*Example of a night image with occlusion:*
![Sir Michael Cobham Library at Night](images/lib_night_occluded.jpg)

---

## 🚀 Key Results & Findings

| Model | Key Strength | Key Weakness | Best Use Case |
| :--- | :--- | :--- | :--- |
| **ResNet-18 Classifier** | High validation accuracy; good generalizability on isolated building images. | Treats classification as mutually exclusive; cannot detect multiple buildings. | Simple, fast classification where the subject building is the clear focus. |
| **Custom ResNet-18 Detector** | Successful proof-of-concept for a unified classification and detection pipeline. | Poor convergence on bounding box regression; high and unstable validation loss. | Academic exploration of multi-task learning. Not suitable for deployment. |
| **YOLOv5 (Base)** | High precision (>99.5% on several classes); fast inference. | Struggled with heavy occlusions and night images. | Scenarios with clear, unobstructed views of buildings. |
| **YOLOv5 (Augmented)** | **Superior overall performance.** Excellent robustness to night conditions and occlusions. | Slight confusion between architecturally similar buildings (e.g., Fusion & Weymouth). | **Real-world deployment.** Handles the varied conditions of a live campus environment. |

**Conclusion:** The **YOLOv5 model trained on augmented data** was identified as the most viable model for a future pathfinding application due to its high accuracy, robustness, and native efficiency in object detection.

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- Google Colab (recommended for easy replication) or a local GPU environment.

### Quick Start (Using Google Colab)
The entire project pipeline is contained within a single Jupyter notebook.

1.  Clone the repository and open the notebook in Colab:
    ```bash
    git clone https://github.com/your_username/bu-campus-pathfinder.git
    cd bu-campus-pathfinder
    ```
2.  Open `Bournemouth_University_Campus_Pathfinder.ipynb` in **Google Colab**.
3.  Run the cells sequentially. The notebook will handle dependency installation, data downloading, and model training.

### Running Inference with the Trained YOLO Model
To test the best-performing model on your own image:
```python
import torch

# Load the custom trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best_augmented_weights.pt')

# Perform inference
results = model('path/to/your/test_image.jpg')

# Show results
results.show()
results.pandas().xyxy[0]  # Pandas DataFrame of results
