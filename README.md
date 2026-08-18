# Parkinson Disease Prediction Using RNA-Seq and Machine Learning

## Overview

This project develops a reproducible machine learning pipeline for Parkinson's disease (PD) prediction using transcriptomic gene expression data.

The main objective is to identify robust gene expression signatures and evaluate machine learning models for Parkinson's disease classification.

The workflow includes:

- Gene expression preprocessing
- Differential expression analysis
- Feature selection
- Biomarker discovery
- Machine learning classification
- External validation


# Research Workflow

RNA-Seq / Gene Expression Data

↓

Data Preprocessing

↓

Differential Expression Analysis

↓

Feature Selection

↓

Gene Signature Construction

↓

Machine Learning Classification

↓

External Validation


# Datasets

## Training Dataset

### GSE99039

Source:

NCBI Gene Expression Omnibus (GEO)


Used for:

- Feature selection
- Biomarker discovery
- Model training


## External Validation Dataset

### GSE165082

Independent GEO dataset used for evaluating model generalization.


Used for:

- External validation
- Performance evaluation


# Computational Pipeline

## 1. Data Preprocessing

Steps:

- Gene expression matrix preparation
- Quality control
- Gene annotation
- Expression normalization


## 2. Differential Expression Analysis

Identification of genes associated with Parkinson's disease by comparing Parkinson's disease and control samples.


## 3. Feature Selection

Integrated approaches:

- Differential expression analysis
- Machine learning feature importance
- Stability selection
- Biological filtering


## 4. Machine Learning Models

Models evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest


# Model Performance

## Training Dataset: GSE99039


| Model | Features | Accuracy | AUC |
|---|---|---|---|
| Logistic Regression | 14 genes | 72.7% | 0.776 |
| Support Vector Machine | 14 genes | 75.0% | 0.756 |
| Random Forest | Selected genes | Evaluated | Evaluated |


## External Validation: GSE165082


| Metric | Value |
|---|---|
| Accuracy | 53.8% |
| AUC | 0.625 |


# Final Gene Signature

The final gene signature was selected using stability-based feature selection and machine learning optimization.

Selected genes:

- PTGDS
- KIR2DL1
- KIR2DL3
- LILRB1
- KIAA0319L
- PPP4C
- TYROBP
- MBOAT7
- MMP9
- HLA-C
- LRRC25
- KIR3DL1
- BCL2
- RHOG


# Project Structure

```text
Parkinson-RNAseq-AI/

├── README.md
├── requirements.txt
├── data/
├── preprocessing/
├── pipeline/
├── models/
├── results/
└── docs/
```


# Installation

Clone repository:

```bash
git clone https://github.com/Ai-Hosseini-Ali/Parkinson-RNAseq-AI.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```


# Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Bioinformatics tools
- GEO datasets


# Future Work

Planned extensions:

- Validation on larger cohorts (PPMI)
- RNA-seq + MRI multimodal learning
- Deep learning models
- Explainable AI (XAI)
- Clinical validation


# Citation

Ali Hoseini (2026)

**Parkinson Disease Prediction Using RNA-Seq and Machine Learning**

GitHub Repository:

https://github.com/Ai-Hosseini-Ali/Parkinson-RNAseq-AI
