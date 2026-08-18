# Parkinson Disease AI Prediction Using RNA-Seq

## Overview

This project develops a machine learning pipeline for Parkinson's disease classification using transcriptomic gene expression data.

The goal is to identify a robust gene signature and evaluate machine learning models for Parkinson's disease prediction.

---

## Datasets

### Training Dataset

GSE99039

- Platform: Gene expression profiling
- Used for:
  - Feature selection
  - Biomarker discovery
  - Model training


### External Validation Dataset

GSE165082

- Independent dataset
- Used for external validation

---

# Pipeline

The workflow consists of:

## 1. Data preprocessing

- Gene expression preparation
- Annotation
- Quality checking


## 2. Differential Expression Analysis

Identification of genes associated with Parkinson's disease.


## 3. Feature Selection

Methods:

- Machine learning importance
- Differential expression
- PPI network analysis
- Stability selection


## 4. Gene Signature Construction

Final locked signature:

---

# Machine Learning Models

Evaluated models:

- Logistic Regression
- Support Vector Machine
- Random Forest


---

# Results

## Training Dataset (GSE99039)

Best model:

Logistic Regression


Accuracy:

72.7%


AUC:

0.776


---

## External Validation

Dataset:

GSE165082


Accuracy:

53.8%


AUC:

0.625


---

# Project Structure

---

# Future Work

Planned extensions:

- Larger datasets (PPMI)
- RNA-Seq + MRI multimodal learning
- Deep learning models
- Explainable AI
- Clinical validation
