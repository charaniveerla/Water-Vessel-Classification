
# 🚢 Iv3-Xc: Hybrid Deep Learning Model for Water Vessel Classification

A hybrid deep learning approach that integrates **InceptionV3** and **Xception** architectures to accurately classify water vessels. Designed for marine safety, defense, and naval surveillance, this ensemble model leverages transfer learning to achieve **93.04% accuracy** on the Analytics Vidhya dataset.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)
- [Credits](#credits)

---

## 🧠 Overview

This project proposes a **transfer learning-based ensemble classification model (Iv3-Xc)** by combining InceptionV3 and Xception architectures. The hybrid model efficiently classifies water vessels into five categories:

- Cargo
- Military
- Carrier
- Cruise
- Tanker

The aim is to improve classification accuracy, reduce overfitting, and handle vanishing gradient issues prevalent in traditional CNNs.

---

## ✨ Features

- Transfer Learning using **InceptionV3** and **Xception**
- **Ensemble Fusion** using `GlobalMaxPooling` and `Concatenate` layers
- Advanced data augmentation for robustness
- High accuracy with reduced training cost
- Evaluation with precision, recall, F1-score, and confusion matrix

---

## 🗂 Dataset

- Source: [Analytics Vidhya – Game of Deep Learning](https://datahack.analyticsvidhya.com/contest/game-of-deep-learning/)
- Total Images: **6,252**
- Classes: Cargo, Military, Carrier, Cruise, Tanker
- Format: JPEG/PNG images resized to `150x150x3`

---

## 🏗️ Model Architecture

> **Iv3-Xc (InceptionV3 + Xception Ensemble)**

1. **Input Layer**: `150x150x3`
2. **InceptionV3** + **Xception** base models (frozen)
3. `GlobalMaxPooling2D` + `Dense(128, relu)` for each
4. `Concatenate` layers
5. `Dense(5, softmax)` for output classification

> Optimizer: **Nadam**  
> Loss: **Categorical Crossentropy**  
> Epochs: **30**  
> Batch Size: **32**

---

## 🛠 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/iv3-xc-vessel-classifier.git
   cd iv3-xc-vessel-classifier
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and organize it:
   ```
   └── dataset/
       ├── train/
       ├── test/
       └── validation/
   ```

---

## ▶️ Usage

Open the notebooks in order:

1. `Ensemble Iv3Xc Model.ipynb` – model creation, training, and saving weights  
2. `EvaluationMetrics.ipynb` – test predictions, confusion matrix, precision, recall, F1 score

To classify a new image:
```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('iv3_xc_model.h5')
image = Image.open("sample.jpg").resize((150, 150))
image = np.array(image) / 255.0
image = image.reshape(1, 150, 150, 3)

prediction = model.predict(image)
predicted_class = np.argmax(prediction)
print(predicted_class)
```

---

## 📊 Evaluation Metrics

| Metric      | Value     |
|-------------|-----------|
| Accuracy    | 93.04%    |
| Precision   | 94%       |
| Recall      | 93%       |
| F1-Score    | 93%       |

---

## 📈 Results

- Confusion matrix and classification report included in the notebook
- Iv3-Xc outperformed other models:

| Model          | Accuracy |
|----------------|----------|
| VGG16          | 83.41%   |
| ResNet50       | 89.03%   |
| EfficientNetB0 | 90.22%   |
| InceptionV3    | 91.00%   |
| Xception       | 92.00%   |
| **Iv3-Xc**     | **93.04%**   |

---

## 🔮 Future Work

- Incorporate **acoustic sound classification** for ships using radiated noise
- Improve classification under occlusion (e.g., fog)
- Real-time classification pipeline for marine surveillance systems

---

## 👥 Credits

**Authors:**

- Charani Sri Veerla – `charaniveerla@gmail.com`
- Twinkle Mounami Budithi – `twinklemounami@gmail.com`
- Srinivas Kudipudi – Professor, VRSEC

Affiliation: Department of Computer Science Engineering, VR Siddhartha Engineering College

---

## 📄 License

This project is for academic and research purposes only. Contact the authors for permission if you intend to use it commercially.

---
