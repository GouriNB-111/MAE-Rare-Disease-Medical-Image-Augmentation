# MAE-based Data Augmentation for Rare Disease Classification

This repository implements a Transformer-based Masked Autoencoder (MAE) for synthetic image generation and class balancing on the DermaMNIST dataset.

This work was presented at an International Conference in Bangkok, Thailand.

---

## Overview

Rare dermatological datasets often suffer from:

- Severe class imbalance  
- Limited annotated samples  
- Poor generalization in deep learning models  

This project addresses these challenges using a Masked Autoencoder (MAE) to generate synthetic minority-class samples and improve classification performance.

---

## Methodology

The pipeline consists of four phases:

1. MAE Pretraining  
   - 75% patch masking  
   - Transformer encoder–decoder architecture  
   - Reconstruction using MSE loss  

2. Synthetic Data Augmentation  
   - Class-wise sample balancing  
   - Minority class expansion using MAE reconstructions  

3. Classifier Training  
   - DeiT-Tiny Vision Transformer  
   - Cross-entropy loss  
   - Adam optimizer  

4. Evaluation  
   - Accuracy  
   - Macro-F1  
   - Weighted-F1  

---

## Results

The MAE-augmented dataset improves minority class recognition while maintaining overall accuracy.

- Baseline Accuracy: 0.757  
- MAE-Augmented Accuracy: 0.762  
- Macro-F1 improved from 0.54 to 0.57  
- Weighted-F1 improved from 0.76 to 0.77  

---

## Requirements

Install dependencies using:

pip install -r requirements.txt

---

## Code Structure

- ml_dataaugmentation.py — Complete training and augmentation pipeline

---

## Dataset

DermaMNIST (MedMNIST Benchmark)  
