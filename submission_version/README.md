# Parkinson AI - RNA-Seq Machine Learning Model

## Project Overview

This project develops an AI-based Parkinson's disease classification model using gene expression data.

The main objective is to identify Parkinson's disease-associated molecular signatures and build a machine learning classifier.

---

## Dataset

Training Dataset:
- GEO Dataset: GSE99039
- Platform: RNA expression dataset
- Samples: 438
- Features: 21756 genes

External Validation Dataset:
- GEO Dataset: GSE165082
- Samples: 26
- Platform: RNA-Seq

---

## Machine Learning Pipeline

Pipeline:

1. Data preprocessing
2. Gene expression filtering
3. Differential expression analysis
4. Feature selection
5. Locked gene signature extraction
6. Machine learning model training
7. External validation


---

## Selected Gene Signature

Final 14 genes:

PTGDS
KIR2DL1
KIR2DL3
LILRB1
KIAA0319L
PPP4C
TYROBP
MBOAT7
MMP9
HLA-C
LRRC25
KIR3DL1
BCL2
RHOG


---

## Models Tested

Models:

- Logistic Regression
- Support Vector Machine
- Random Forest


Best Model:

Logistic Regression


Internal Validation:

Accuracy:
72.7%

AUC:
0.776


External Validation:

Dataset:
GSE165082

Accuracy:
53.8%

AUC:
0.625


---

## Project Structure

data/
    Raw and processed datasets

pipeline/
    Analysis and training scripts

model/
    Saved machine learning models

results/
    Evaluation results and figures

figures/
    Visualization outputs


---

## Main Files

Training:
pipeline/FINAL_MODEL_BENCHMARK.py

External Validation:
pipeline/07_external_validation_final.py

Probability Analysis:
pipeline/08_check_external_probability.py

Distribution Analysis:
pipeline/09_external_distribution_analysis.py


---

## Environment

Python:
3.x

Main Libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- scipy


---

## Author

Ali Hosseini

Biomedical Engineering
AI in Biomedical Applications