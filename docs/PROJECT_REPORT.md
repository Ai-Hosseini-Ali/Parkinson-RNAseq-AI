# Parkinson Disease Prediction Using RNA-Seq Gene Expression and Machine Learning

## Project Overview

This project aims to develop an Artificial Intelligence based diagnostic model for Parkinson's Disease using RNA-Seq gene expression data.

The workflow combines:
- Differential Expression Analysis (DE)
- Machine Learning based feature selection
- Stability selection
- Protein-Protein Interaction (PPI) network analysis
- Multi-gene signature construction
- External validation on independent GEO dataset


---

# Dataset

## Training Dataset

GEO Accession:
GSE99039

Platform:
RNA-Seq gene expression dataset

Purpose:
Model training and biomarker discovery.


## External Validation Dataset

GEO Accession:
GSE165082

Purpose:
Independent validation of the discovered gene signature.


---

# Computational Pipeline

## 1. Data Processing

Steps:

- Load RNA-Seq expression matrix
- Quality checking
- Gene filtering
- Label preparation
- Dataset normalization


## 2. Biomarker Discovery

Multiple approaches were combined:

### Differential Expression Analysis

Identification of genes showing significant expression differences between Parkinson's Disease and Control groups.


### Machine Learning Feature Importance

Machine learning models were used to rank informative genes.


### Stability Selection

Repeated sampling was performed to identify robust genes.

Parameters:

- 100 iterations
- Top 15 genes selected in each iteration


### Network Analysis

STRING PPI network information was incorporated to evaluate biological importance.


---

# Final Gene Signature

The final locked signature contains 14 genes:

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


---

# Machine Learning Models

Three classifiers were evaluated:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest


## Internal Validation Results

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 0.727 | 0.776 |
| SVM | 0.659 | 0.697 |
| Random Forest | 0.648 | 0.716 |


Best model:

Logistic Regression


---

# External Validation

Independent validation:

Dataset:
GSE165082


Result:

Accuracy:
0.538


AUC:
0.625


Additional normalization strategies improved performance:

Rank normalization:

Accuracy:
0.731

AUC:
0.708


---

# Cross Platform Analysis

Gene distribution differences between datasets were investigated.

Large expression shifts were observed in immune-related genes:

- KIR2DL1
- KIR2DL3
- KIR3DL1
- TYROBP
- MMP9


This indicates platform and batch effects between datasets.


---

# Biological Analysis

Performed:

- GO enrichment analysis
- KEGG pathway analysis
- STRING PPI analysis


The selected genes are mainly associated with:

- Immune regulation
- Inflammatory pathways
- Cell signaling
- Neurodegenerative mechanisms


---

# Project Structure


---

# Reproducibility

Main execution pipeline:



Run:


---

# Model Files

Saved models:

- final_parkinson_logistic_model.pkl
- final_15gene_model.pkl
- final_scaler.pkl


---

# Current Status

Version:

v1-final-14gene


The current model represents the first stable Parkinson RNA-Seq AI signature.

Future work:

- Additional GEO datasets
- PPMI validation
- Deep Learning approaches
- Multi-modal integration with MRI and clinical data


---

# Author

Ali Hoseini

Biomedical Engineering - AI in Biomedical Applications