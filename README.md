
# 🚢 Iv3-Xc: Hybrid Deep Learning Model for Water Vessel Classification

This project presents a **two-phase approach** to water vessel classification using state-of-the-art deep learning techniques.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Phase 1: Transfer Learning Model Evaluation](#phase-1-transfer-learning-model-evaluation)
- [Phase 2: Hybrid Iv3-Xc Ensemble Model](#phase-2-hybrid-iv3-xc-ensemble-model)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)
- [Credits](#credits)
- [License](#license)

---

## 📖 Project Overview

This project targets the classification of marine water vessels (Cargo, Military, Carrier, Cruise, Tanker) using deep learning, with applications in **maritime security, naval defense**, and **coastal surveillance**.

We began by experimenting with **five pre-trained transfer learning models**, then proceeded to create a hybrid ensemble model combining the two best performers — **InceptionV3** and **Xception** — into the final `Iv3-Xc` architecture.

---

## 🔍 Phase 1: Transfer Learning Model Evaluation

We fine-tuned and tested the following models on a labeled dataset from Analytics Vidhya:

- VGG16
- ResNet50
- EfficientNetB0
- InceptionV3
- Xception

After evaluating performance based on Accuracy, Precision, Recall, and F1-Score, the **top two models** identified were:

✅ InceptionV3  
✅ Xception

---

## 🚀 Phase 2: Hybrid Iv3-Xc Ensemble Model

### Key Design:
- Combined **InceptionV3** and **Xception** using a concatenated architecture
- Integrated custom top layers using `GlobalMaxPooling2D` and `Dense` layers
- Final classifier: `Dense(5, softmax)` for multi-class output

### Training Details:
- Image Input Size: `150x150x3`
- Epochs: 30
- Batch Size: 32
- Optimizer: Nadam
- Loss Function: Categorical Crossentropy

---

## 🗂 Dataset

- Source: [Analytics Vidhya – Game of Deep Learning](https://datahack.analyticsvidhya.com/contest/game-of-deep-learning/)
- Total Images: **6,252**
- Classes: Cargo, Military, Carrier, Cruise, Tanker
- Format: JPEG/PNG, resized to `150x150`

---

## ✨ Features

- ✅ Transfer learning with five CNNs
- ✅ Final hybrid ensemble: Iv3-Xc
- ✅ Data Augmentation using `ImageDataGenerator`
- ✅ Confusion Matrix & Classification Report
- ✅ Notebook-based reproducibility

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/iv3-xc-vessel-classifier.git
cd iv3-xc-vessel-classifier
pip install -r requirements.txt
```

Organize your dataset like:

```
dataset/
├── train/
├── test/
└── validation/
```

---

## ▶️ Usage

Open the following notebooks in sequence:

1. `Ensemble Iv3Xc Model.ipynb` – model definition, training, and saving weights
2. `EvaluationMetrics.ipynb` – test predictions, confusion matrix, and performance evaluation

Example inference code:
```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('iv3_xc_model.h5')
image = Image.open("sample.jpg").resize((150, 150))
image = np.array(image) / 255.0
image = image.reshape(1, 150, 150, 3)
pred = model.predict(image)
print("Predicted Class:", np.argmax(pred))
```

---

## 📊 Evaluation Metrics

| Model          | Accuracy |
|----------------|----------|
| VGG16          | 83.41%   |
| ResNet50       | 89.03%   |
| EfficientNetB0 | 90.22%   |
| InceptionV3    | 91.00%   |
| Xception       | 92.00%   |
| **Iv3-Xc**     | **93.04%** |

Additional Metrics for Iv3-Xc:
- Precision: 94%
- Recall: 93%
- F1-Score: 93%

---

## 📈 Results

- The hybrid Iv3-Xc model **outperformed all standalone models**
- Confusion matrix and classification report included in the notebook
- Visual plots for training and validation accuracy/loss tracked over 30 epochs

---

## 🔮 Future Work

- Integrate **acoustic signal-based classification** using radiated marine vessel noise
- Improve classification under low-visibility conditions (fog, night)
- Real-time deployment for coastal monitoring systems

---

## 👥 Credits

**Contributors:**

- Charani Sri Veerla – `charaniveerla@gmail.com`
- Twinkle Mounami Budithi – `twinklemounami@gmail.com`
- Dr. Srinivas Kudipudi – Professor, VR Siddhartha Engineering College

---

## 📄 License

This project is intended for **academic and research purposes** only. Contact the authors for any commercial use or redistribution.

---
