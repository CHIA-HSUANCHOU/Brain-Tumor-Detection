# Deep Learning in Medical Image Analysis: Homework 3

**Author**: 313657003 周佳萱  
**Course**: Deep Learning in Medical Image Analysis  
**Date**: 2024/11/18  

---

## Overview

This project addresses two tasks in medical image analysis:  
1. **Parkinson's Disease (PD) Classification**: Using ResNet50 to classify SPECT images into six PD stages.  
2. **Brain Tumor Detection**: Using 3D MRI images to classify normal participants and patients with brain tumors.  
Both tasks aim to address challenges such as imbalanced data and small sample sizes by employing advanced techniques and pre-trained models.

---

## Requirements

### Part 1: SPECT Image Analysis (PD Classification)
1. **Dataset**:
   - Provided SPECT images classified into six PD stages.
   - Training data distribution is imbalanced.

2. **Objective**:
   - Address imbalanced data and small sample size issues.
   - Train a ResNet50 model to classify six PD stages.
   - Predict probabilities and disease stages for test images.

---

### Part 2: 3D MRI Image Analysis (Brain Tumor Detection)
1. **Dataset**:
   - 120 3D MRI images: 60 normal participants and 60 patients with brain tumors.
   - Each participant has T1- and T2-weighted 2D MRI slices.

2. **Preprocessing**:
   - Normalize pixel values using WW (window width) and WC (window center).
   - Unify 3D MRI dimensions to `(512 × 512 × 22)` using trilinear interpolation.

3. **Models**:
   - Perform four analyses: single slice, late fusion, early fusion, and 3D CNNs.
   - Utilize transfer learning and pre-trained models for improved accuracy.
   - For pre-trained models, represent three channels as T1 grayscale, T2 grayscale, and their average.

---

## Implementation

### Parkinson's Disease Classification:
- **Approach**: 
  - ResNet50 architecture fine-tuned on the provided dataset.
  - Applied data augmentation and oversampling techniques to address class imbalance.
  - Predicted probabilities for six stages and saved results in `ResNet50_6c.csv`.

---

### Brain Tumor Detection:
1. **Single Slice**:
   - Used VGG16 with a modified fully connected layer of size 7×7.
   - Applied `softmax` on logits and averaged predictions across samples.

2. **Late Fusion**:
   - Used VGG16 with a 3×3 AvgPool layer.
   - Combined features from multiple slices before flattening and classification.

3. **Early Fusion**:
   - Used VGG16, modified the first convolutional layer to accept 66 input channels.
   - Flattened extracted features into a one-dimensional vector for classification.

4. **3D CNNs**:
   - Used ResNet50, modified for 3D inputs with 3D convolutional layers.
   - Added batch normalization, ReLU activation, and max-pooling.
   - Consisted of three layers, each with three bottleneck blocks.

---

## Challenges and Solutions

1. **Imbalanced Data**:
   - Oversampling and data augmentation were applied to mitigate class imbalance.
   - Ensured all classes were sufficiently represented during training.

2. **Limited Sample Size**:
   - Utilized transfer learning to leverage pre-trained weights from larger datasets.
   - Adjusted learning rate dynamically with `OneCycleLR` for better performance on small datasets.

3. **Training Difficulties**:
   - Experimented with different pooling sizes (2×2, 3×3, 4×4, 5×5) in VGG16 for late fusion.
   - Opted for 3×3 AvgPool for a balance between parameter count and accuracy.

---

## Results

1. **Selected Model**: Late Fusion with VGG16  
   - **Parameter Count**: 14,917,442  
   - Achieved robust performance on test data.

---

