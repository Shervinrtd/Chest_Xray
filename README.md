# Pneumonia Detection from Chest X-Ray Images Using Deep Learning

## Overview

This project focuses on detecting pneumonia from chest X-ray images using deep learning techniques. The main objective is to develop and compare different convolutional neural network (CNN) architectures for binary image classification (NORMAL vs PNEUMONIA).

The project includes:

* A baseline CNN model built from scratch
* Class imbalance handling using weighted loss functions
* Transfer learning with ResNet50
* Fine-tuning pretrained models
* Model evaluation using multiple performance metrics
* Grad-CAM visualization for explainability

The implementation is developed using **PyTorch** and trained in **Google Colab** with GPU acceleration.

---

# Dataset

The project uses the publicly available Chest X-ray dataset from Kaggle:

Dataset:
Chest X-Ray Images (Pneumonia)

Classes:

* NORMAL
* PNEUMONIA

Dataset Structure:

```text
train/
    NORMAL/
    PNEUMONIA/

val/
    NORMAL/
    PNEUMONIA/

test/
    NORMAL/
    PNEUMONIA/
```

---

# Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Seaborn
* OpenCV
* Scikit-learn
* Google Colab

---

# Project Pipeline

## 1. Data Preprocessing

* Image resizing
* Normalization
* Data augmentation
* Grayscale to RGB conversion

## 2. Baseline CNN

A custom convolutional neural network was implemented to establish baseline performance.

## 3. Weighted CNN

Class imbalance was handled using weighted cross-entropy loss to improve balanced classification performance.

## 4. Transfer Learning

A pretrained ResNet50 model was used for transfer learning and fine-tuning.

## 5. Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## 6. Explainability

Grad-CAM was implemented to visualize the regions influencing the model predictions.

---

# Results Summary

| Model               | Accuracy | Weighted F1 Score |
| ------------------- | -------- | ----------------- |
| Baseline CNN        | 76%      | 0.72              |
| Weighted CNN        | 83%      | 0.83              |
| Fine-Tuned ResNet50 | 82%      | 0.80              |

The weighted CNN achieved the best balanced performance on the test set.

---

# Sample Features

* Custom CNN architecture
* Weighted loss for class imbalance
* Transfer learning with ResNet50
* Fine-tuning pretrained networks
* Grad-CAM heatmap visualization
* Training and validation plots
* Confusion matrix analysis

---

# How to Run


## Install Dependencies

```bash
pip install torch torchvision matplotlib seaborn scikit-learn opencv-python
```

## Run the Notebook

Open the Jupyter Notebook or Google Colab notebook and execute the cells sequentially.

---

# Project Structure

```text
├── data/
├── notebooks/
├── models/
├── outputs/
│   ├── plots/
│   ├── heatmaps/
│   └── confusion_matrices/
├── README.md
└── requirements.txt
```

---

# Future Improvements

Possible future enhancements include:

* Using larger medical datasets
* Trying EfficientNet or DenseNet architectures
* Hyperparameter optimization
* Ensemble learning
* Lung segmentation before classification
* Clinical deployment optimization

---

# Acknowledgements

* Kaggle Chest X-ray Dataset
* PyTorch Documentation
* Google Colab GPU Resources

---

# License

This project is intended for educational and research purposes.
