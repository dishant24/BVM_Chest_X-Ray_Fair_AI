<<<<<<< HEAD
# Improving Generalization and Robustness of ChestX-ray AI Models: Preprocessing method to mitigate Racial/Ethnic Bias

This project focuses on improving the **fairness** and **robustness** of AI models used in **medical chest X-ray (CXR) diagnosis**. The primary objective is mitigate **racial demographic biases** that can arise in deep learning models trained on medical imaging CXR datasets.

---

## 🚀 Objective

Build processing method for an AI model that is:
- **Accurate**: High performance on core classification tasks (e.g., disease detection)
- **Fair**: Ensures consistent performance across different racial, gender, and age groups
- **Robust**: Maintains reliability under various distribution shifts and real-world scenarios

---


## 📦 Installation

```bash
git clone git@gitlab.fme.lan:sutariya/cxr_preprocessing.git
cd cxr_preprocessing
pip install -r requirements.txt
```

## ⚙️ Methods and Approach

1. **Model Training**:
   - Reproduce inital experiments of reference papers and bulit DenseNet model
   - Train model on diagnosis label and use same embedding to train model on race (Use transfer learning for race classification)
   - Evaluate model performance across demographic slices

2. **Bias Mitigation**:
   - Used lung segmentation for model only focus on relevant clinical feature 
   - Used CLAHE histogram Equalization method to prevent image from noise and give better constast


## 🗂️ Dataset(s)

This project uses publicly available chest X-ray datasets such as:
- **MIMIC-CXR**
- **CheXpert**

These datasets include metadata for demographic attributes.

---
=======
# BVM_Chest_X-Ray_Fair_AI
This project focuses on improving the fairness and robustness of AI models used in medical chest X-ray (CXR) diagnosis. The primary objective is mitigate racial demographic biases that can arise in deep learning models trained on medical imaging CXR datasets.
>>>>>>> ff21d3f9cdf57b3b023492718ea065cafc25db92
