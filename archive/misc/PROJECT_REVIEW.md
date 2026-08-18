 Section 1: Project Overview



  Project: Parkinson's Disease AI Biomarker Discovery & Risk Prediction



  Location: C:\Users\alihoseini\Desktop\Parkinson_AI

  

  Goal: Identify a gene expression signature that distinguishes Parkinson's

  disease patients from healthy controls, and deploy a patient-level risk

  predictor via a Streamlit web app.



  Datasets used:

  ┌───────────┬──────────────────┬─────────────┬──────────────────────────┐

  │  Dataset  │     Platform     │    Type     │           Role           │

  ├───────────┼──────────────────┼─────────────┼──────────────────────────┤

  │ GSE99039  │ GPL570           │ Microarray  │ Primary                  │

  │           │ (Affymetrix)     │             │ training/discovery       │

  ├───────────┼──────────────────┼─────────────┼──────────────────────────┤

  │ GSE165082 │ RNA-seq          │ Counts      │ External validation      │

  ├───────────┼──────────────────┼─────────────┼──────────────────────────┤

  │ GSE49036  │ Microarray       │ Series      │ Third validation         │

  │           │                  │ matrix      │ (incomplete)             │

  └───────────┴──────────────────┴─────────────┴──────────────────────────┘



  Pipeline stages (70+ scripts in preprocessing/):

  1. Differential expression analysis (Welch's t-test + BH FDR)

  2. Feature selection (top DE genes + RF importance)

  3. ML model comparison (LR, SVM, RF, MLP, Gradient Boosting)

  4. PPI network analysis (STRING API)

  5. Composite biomarker scoring (ML + DE + PPI + bootstrap stability)

  6. Signature optimization (tested 5–100 gene panels)

  7. External validation on GSE165082

  8. Deployment via Streamlit app with 14-gene Logistic Regression model



  Final deliverable: A 14-gene signature with a serialized Logistic Regression  model (model/final_parkinson_logistic_model.pkl) and a Streamlit app for

  patient CSV upload.



  ---

  Ready for Section 2 (Data Leakage & Methodological Issues) when you say the

  word.



✻ Worked for 32s



❯ Summarize the audit findings in small sections.                             

  Start with Section 1 only.                                                  

  Do not analyze the project again.                                           



● Section 1: Project Overview



  Project: Parkinson's Disease AI — Biomarker Discovery & Risk Prediction

  Location: C:\Users\alihoseini\Desktop\Parkinson_AI

  Objective: Discover a blood-based gene expression signature that separates

  Parkinson's disease (PD) patients from healthy controls, then deploy a

  patient-level risk predictor.



  Datasets:



  ┌───────────┬─────────────────┬──────────────┬──────────────────────────┐

  │  Dataset  │    Platform     │     Type     │           Role           │

  ├───────────┼─────────────────┼──────────────┼──────────────────────────┤

  │ GSE99039  │ GPL570          │ Microarray   │ Primary training &       │

  │           │ Affymetrix      │              │ discovery                │

  ├───────────┼─────────────────┼──────────────┼──────────────────────────┤

  │ GSE165082 │ RNA-seq         │ Counts       │ External validation      │

  ├───────────┼─────────────────┼──────────────┼──────────────────────────┤

  │ GSE49036  │ Microarray      │ Series       │ Third validation         │

  │           │                 │ matrix       │ (incomplete)             │

  └───────────┴─────────────────┴──────────────┴──────────────────────────┘



  Pipeline (70+ scripts across preprocessing/, models/, app/):



  1. Differential expression — Welch's t-test with Benjamini-Hochberg FDR

  2. Feature selection — top DE genes intersected with Random Forest

  importance

  3. Model comparison — Logistic Regression, SVM, Random Forest, MLP, Gradient  Boosting

  4. Protein-protein interaction — STRING API, hub gene identification

  5. Composite scoring — weighted ML (0.25) + DE (0.25) + PPI (0.25) +

  stability (0.25)

  6. Signature size optimization — panels of 5, 10, 15, 20, 30, 50, 100 genes

  tested

  7. External validation — GSE165082 with batch mean-centering

  8. Deployment — Streamlit app accepting patient CSV uploads

  │  Dataset  │    Platform     │     Type     │           Role           │

  ├───────────┼─────────────────┼──────────────┼──────────────────────────┤

  │ GSE99039  │ GPL570          │ Microarray   │ Primary training &       │

  │           │ Affymetrix      │              │ discovery                │

  ├───────────┼─────────────────┼──────────────┼──────────────────────────┤

  │ GSE165082 │ RNA-seq         │ Counts       │ External validation      │

  ├───────────┼─────────────────┼──────────────┼──────────────────────────┤

  │ GSE49036  │ Microarray      │ Series       │ Third validation         │

  │           │                 │ matrix       │ (incomplete)             │

  └───────────┴─────────────────┴──────────────┴──────────────────────────┘



  Pipeline (70+ scripts across preprocessing/, models/, app/):



  1. Differential expression — Welch's t-test with Benjamini-Hochberg FDR

  2. Feature selection — top DE genes intersected with Random Forest

  importance

  3. Model comparison — Logistic Regression, SVM, Random Forest, MLP, Gradient  Boosting

  4. Protein-protein interaction — STRING API, hub gene identification

  5. Composite scoring — weighted ML (0.25) + DE (0.25) + PPI (0.25) +

  stability (0.25)

  6. Signature size optimization — panels of 5, 10, 15, 20, 30, 50, 100 genes

  tested

  7. External validation — GSE165082 with batch mean-centering

  8. Deployment — Streamlit app accepting patient CSV uploads



  Final deliverable: 14-gene Logistic Regression model serialized to

  model/final_parkinson_logistic_model.pkl, served by a Streamlit web app.
  
● Section 2: Critical Methodological Issues

  2.1 Data Leakage — Feature Selection Before Split

  In select_features.py, the top 500 genes are selected using all samples (PD  + Control combined) before any train/test split occurs. Every subsequent
  cross-validation result on these 500 genes is optimistically biased — the  model has already "seen" the test fold during feature selection.

  Impact: Reported accuracy (92% with 500 genes) is inflated and not
  reproducible on truly unseen data.

  2.2 Data Leakage — Scaler Fit on Full Data

  In mlp_backprop_GSE99039.py, StandardScaler.fit_transform(X) is applied to
  the entire dataset before passing it to cross_val_score. The scaler's mean
  and variance incorporate test-fold samples.

  2.3 Training AUC Reported as Performance

  In evaluate_final_signature.py, a Random Forest is trained and evaluated on
  the same data (lines 184–206). The AUC printed there is a training metric,
  not a generalization estimate. No held-out set is used.

  2.4 Silent Zero-Imputation for Missing Genes

  In predictor.py (line 106), any gene absent from a patient's CSV is silently  filled with 0. For a gene whose normal expression is, say, 8.5 on a log2
  scale, injecting 0 creates an extreme outlier that will skew the prediction
  — with no warning to the user.

  2.5 Inadequate Batch Correction

  Cross-platform validation (microarray → RNA-seq) uses simple per-gene mean
  centering. Established methods like ComBat or limma's removeBatchEffect
  account for higher-order distributional differences. Mean centering alone is  insufficient when the platforms have different dynamic ranges and noise
  profiles.

  2.6 Incomplete Third-Dataset Validation

  validate_GSE49036.py is truncated at 50 lines — it loads data but contains
  no model evaluation code. The third-dataset validation was never completed.

  
● Section 3: Gene Signature Inconsistencies

  3.1 The "15-Gene" vs 14-Gene Contradiction

  The project describes itself as a 15-gene biomarker signature, but the  actual gene count varies across files:
  ┌──────────────────────────────────────┬────────────┬───────────────┐
  │                Source                │ Gene Count │ DGKK Present? │
  ├──────────────────────────────────────┼────────────┼───────────────┤
  │ model/final_genes.txt                │ 14         │ ❌ No         │
  ├──────────────────────────────────────┼────────────┼───────────────┤
  │ test_patient.csv                     │ 14         │ ❌ No         │
  ├──────────────────────────────────────┼────────────┼───────────────┤
  │ train_final_model.py                 │ 14         │ ❌ No         │
  ├──────────────────────────────────────┼────────────┼───────────────┤
  │ predictor.py                         │ 14         │ ❌ No         │
  ├──────────────────────────────────────┼────────────┼───────────────┤
  │ external_validation_GSE165082.py     │ 15         │ ✅ Yes        │
  ├──────────────────────────────────────┼────────────┼───────────────┤
  │ batch_correct_external_validation.py │ 14         │ ❌ No         │
  └──────────────────────────────────────┴────────────┴───────────────┘

  DGKK was apparently dropped at some point during development, but the
  external validation script still references the original 15-gene list. This
  means the external validation was run on a different gene set than the
  deployed model uses.

  3.2 The Final 14-Gene Signature

  From model/final_genes.txt:

  PTGDS, KIR2DL1, KIR2DL3, LILRB1, KIAA0319L, PPP4C, TYROBP,
  MBOAT7, MMP9, HLA-C, LRRC25, KIR3DL1, BCL2, RHOG

  3.3 Biological Plausibility Concern

  The KEGG pathway enrichment results tell a striking story:

  - Top hit: Graft-versus-host disease (P=0.0003, Adj.P=0.07)
  - Other top pathways: Antigen processing, NK cell cytotoxicity, NF-κB
  signaling, osteoclast differentiation
  - "Parkinson disease" pathway: Ranked last (row 235 of 235, Adj.P=0.996,
  only 1 gene: NDUFA5)

  The signature is dominated by KIR (killer-cell immunoglobulin-like receptor)  and HLA genes — immune/inflammatory markers, not
  neurodegeneration-specific. This may reflect a real peripheral immune signal  in PD blood, but it could also reflect confounding by age, medication, or
  comorbidities in the GSE99039 cohort. The project does not discuss or
  control for these confounders.

  ## Section 4: Code Quality & Reproducibility

(متن کامل Section 4)
Say next for Section 4                                                      

● Section 4: Code Quality & Reproducibility

  4.1 Empty Placeholder Files (7 total)

  ┌────────────────────────────────────┬────────────────┐  │                File                │    Location    │
  ├────────────────────────────────────┼────────────────┤  │ filtering.py                       │ preprocessing/ │
  ├────────────────────────────────────┼────────────────┤
  │ normalization.py                   │ preprocessing/ │
  ├────────────────────────────────────┼────────────────┤
  │ robust_gene_mapping_compare.py     │ preprocessing/ │
  ├────────────────────────────────────┼────────────────┤
  │ consensus_two_dataset_signature.py │ preprocessing/ │
  ├────────────────────────────────────┼────────────────┤
  │ random_forest.py                   │ models/        │
  ├────────────────────────────────────┼────────────────┤
  │ xgboost_model.py                   │ models/        │
  ├────────────────────────────────────┼────────────────┤
  │ neural_network.py                  │ models/        │
  └────────────────────────────────────┴────────────────┘

  These suggest planned work that was never implemented. The models/
  placeholders indicate XGBoost and a dedicated neural network architecture
  were intended but abandoned.

  4.2 Massive Code Duplication

  Nearly identical data-loading and preprocessing blocks are copy-pasted
  across 50+ scripts. There is no shared utility module for common operations
  like:
  - Loading and filtering expression matrices
  - Label extraction
  - Normalization (log2(x+1))
  - Train/test splitting with stratification

  A single change to preprocessing logic would require editing dozens of
  files.

  4.3 Hardcoded Fragilities

  ┌──────────────┬────────────────────────────────────────────────────────┐
  │    Issue     │                        Example                         │
  ├──────────────┼────────────────────────────────────────────────────────┤
  │ Hardcoded    │ "data/GSE99039_series_matrix.txt" — all relative, no   │
  │ paths        │ config file                                            │
  ├──────────────┼────────────────────────────────────────────────────────┤
  │ Hardcoded    │ create_labels_GSE99039.py reads disease labels from    │
  │ line number  │ line 40 of the series matrix                           │
  ├──────────────┼────────────────────────────────────────────────────────┤
  │ Hardcoded    │ figure2_model_comparison.py plots a bar chart with     │
  │ accuracies   │ manually typed values (50%, 92%)                       │
  ├──────────────┼────────────────────────────────────────────────────────┤
  │ Wrong        │ volcano_GSE99039.py loads                              │
  │ relative     │ "differential_expression_GSE99039_FDR.csv" without the │
  │ path         │  data/ prefix                                          │
  └──────────────┴────────────────────────────────────────────────────────┘

  4.4 No Dependency Management

  No requirements.txt, setup.py, pyproject.toml, or environment.yml.
  Dependencies must be reverse-engineered from imports across all files. Key
  libraries used: scikit-learn, pandas, numpy, matplotlib, seaborn,
  statsmodels, gseapy, mygene, networkx, joblib, streamlit.

  4.5 Comments in Persian (Farsi)

  All inline comments are written in Farsi. This limits international
  reproducibility and peer review.

  4.6 Interactive CLI in Production Code

  predict_patient.py uses input() prompts to collect gene values one by one —
  completely unsuitable for automated pipelines, testing, or integration.

  4.7 No Logging, No Error Handling, No Tests

  - Zero logging calls across the entire project
  - Exception handling limited to a bare except Exception as e in the
  │ Wrong        │ volcano_GSE99039.py loads                              │
  │ relative     │ "differential_expression_GSE99039_FDR.csv" without the │
  │ path         │  data/ prefix                                          │
  └──────────────┴────────────────────────────────────────────────────────┘

  4.4 No Dependency Management

  No requirements.txt, setup.py, pyproject.toml, or environment.yml.
  Dependencies must be reverse-engineered from imports across all files. Key
  libraries used: scikit-learn, pandas, numpy, matplotlib, seaborn,
  statsmodels, gseapy, mygene, networkx, joblib, streamlit.

  4.5 Comments in Persian (Farsi)

  All inline comments are written in Farsi. This limits international
  reproducibility and peer review.

  4.6 Interactive CLI in Production Code

  predict_patient.py uses input() prompts to collect gene values one by one —
  completely unsuitable for automated pipelines, testing, or integration.

  4.7 No Logging, No Error Handling, No Tests

  - Zero logging calls across the entire project
  - Exception handling limited to a bare except Exception as e in the
  Streamlit app
  - No unit tests, no integration tests, no test directory
## Section 5: Deployment & App Assessment

(متن کامل Section 5)

● Section 5: Deployment & App Assessment

  5.1 Streamlit App (app/app.py)

  The app provides a simple interface:  1. User uploads a CSV with gene expression values
  2. App calls preprocessing.predictor.predict_patient()  3. Displays Parkinson Risk % and a binary prediction label

  5.2 Model Artifacts (model/)

  ┌────────────────────────────────────┬───────────────────────────────┐
  │                File                │           Contents            │
  ├────────────────────────────────────┼───────────────────────────────┤
  │ final_parkinson_logistic_model.pkl │ Serialized LogisticRegression │
  ├────────────────────────────────────┼───────────────────────────────┤
  │ final_scaler.pkl                   │ Serialized StandardScaler     │
  ├────────────────────────────────────┼───────────────────────────────┤
  │ final_genes.txt                    │ 14 gene symbols (plain text)  │
  └────────────────────────────────────┴───────────────────────────────┘

  5.3 Deployment Issues

  No input validation. The app does not verify that the uploaded CSV contains
  the expected gene columns before calling the predictor. A CSV with
  completely wrong columns will either crash or silently produce a meaningless  prediction (due to zero-imputation — see §2.4).

  No model versioning. There is no metadata recording which training data,
  hyperparameters, or gene list produced the serialized model. If the .pkl
  files are replaced, there is no way to trace provenance.

  No confidence calibration. The "Risk %" displayed is a raw predict_proba
  output from Logistic Regression. This has not been calibrated (e.g., via
  Platt scaling or isotonic regression). A displayed "73% risk" does not mean
  a 73% probability of PD — it is an uncalibrated score.

  Silent gene imputation in production. The predictor reports "Genes found:
  X/14" but does not flag which genes are missing or how much the prediction
  degrades. A patient CSV with only 8 of 14 genes will still return a
  confident-looking risk percentage.

  Import hack. The app uses sys.path.append(BASE_DIR) to reach the
  preprocessing package — fragile and breaks if the directory structure
  changes.

  No rate limiting, authentication, or audit logging. The Streamlit app is a
  bare prototype with no access controls.

  5.4 Duplicate Model Storage

  Serialized model files exist in both model/ and data/:
  - model/final_parkinson_logistic_model.pkl
  - data/final_parkinson_logistic_model.pkl

  hyperparameters, or gene list produced the serialized model. If the .pkl
  files are replaced, there is no way to trace provenance.

  No confidence calibration. The "Risk %" displayed is a raw predict_proba
  output from Logistic Regression. This has not been calibrated (e.g., via
  Platt scaling or isotonic regression). A displayed "73% risk" does not mean
  a 73% probability of PD — it is an uncalibrated score.

  Silent gene imputation in production. The predictor reports "Genes found:
  X/14" but does not flag which genes are missing or how much the prediction
  degrades. A patient CSV with only 8 of 14 genes will still return a
  confident-looking risk percentage.

  Import hack. The app uses sys.path.append(BASE_DIR) to reach the
  preprocessing package — fragile and breaks if the directory structure
  changes.

  No rate limiting, authentication, or audit logging. The Streamlit app is a
  bare prototype with no access controls.

  5.4 Duplicate Model Storage

  Serialized model files exist in both model/ and data/:
  - model/final_parkinson_logistic_model.pkl
  - data/final_parkinson_logistic_model.pkl

  It is unclear which location is authoritative. The predictor loads from
  model/, but train_final_model.py saves to data/.
                                    

● Section 6: Summary Scorecard & Recommendations

  6.1 Scorecard

  ┌───────────────────┬──────────┬───────────────────────────────────────┐  │     Dimension     │  Rating  │              Key Finding              │
  ├───────────────────┼──────────┼───────────────────────────────────────┤  │ Scientific Rigor  │ 🔴 Weak  │ Feature selection before CV splits;   │
  │                   │          │ training AUC reported as performance  │
  ├───────────────────┼──────────┼───────────────────────────────────────┤
  │                   │          │ No dependency file, hardcoded         │
  │ Reproducibility   │ 🔴 Weak  │ paths/line numbers, Farsi-only        │
  │                   │          │ comments                              │
  ├───────────────────┼──────────┼───────────────────────────────────────┤
  │ Statistical       │ 🟡       │ Proper FDR correction used, but       │
  │ Validity          │ Moderate │ leakage inflates downstream results   │
  ├───────────────────┼──────────┼───────────────────────────────────────┤
  │ Biological        │ 🟡       │ Immune signature may be real, but PD  │
  │ Plausibility      │ Moderate │ pathway ranks last in KEGG; no        │
  │                   │          │ confounder analysis                   │
  ├───────────────────┼──────────┼───────────────────────────────────────┤
  │ Code Quality      │ 🔴 Weak  │ 7 empty files, massive duplication,   │
  │                   │          │ no tests, no logging                  │
  ├───────────────────┼──────────┼───────────────────────────────────────┤
  │ External          │ 🟡       │ GSE165082 attempted but on wrong gene │
  │ Validation        │ Moderate │  set (15 vs 14); GSE49036 incomplete  │
  ├───────────────────┼──────────┼───────────────────────────────────────┤
  │ Deployment        │ 🔴 Weak  │ No input validation, uncalibrated     │
  │ Readiness         │          │ probabilities, silent zero-imputation │
  ├───────────────────┼──────────┼───────────────────────────────────────┤
  │ Documentation     │ 🔴 Weak  │ No README, no docstrings, no pipeline │
  │                   │          │  orchestration, no CHANGELOG          │
  └───────────────────┴──────────┴───────────────────────────────────────┘

  6.2 Recommendations (Priority Order)

  P0 — Must fix before any results are publishable:

  1. Redo feature selection inside CV folds. Wrap gene filtering + model
  training in a single pipeline so test folds are never seen during selection.  2. Fix the 14/15-gene inconsistency. Pick one definitive gene list. Re-run
  external validation with the same list the deployed model uses.
  3. Fit scalers inside CV loops. Use sklearn.pipeline.Pipeline to prevent
  leakage from StandardScaler.
  4. Replace training AUC with held-out AUC in all evaluation scripts.

  P1 — Required for reproducibility:

  5. Create requirements.txt with pinned versions for all dependencies.
  6. Extract shared utilities — one module for data loading, normalization,
  and splitting.
  7. Replace hardcoded line numbers with metadata parsing (search for
  !Sample_characteristics dynamically).
  8. Translate comments to English or add bilingual annotations.
  9. Delete empty placeholder files or implement them.

  P2 — Required for deployment:

  10. Calibrate predict_proba using Platt scaling or isotonic regression on a
  held-out set.
  11. Add input validation in the Streamlit app — reject CSVs missing critical  genes rather than imputing zeros.
  12. Add model version metadata — save training date, gene list hash, dataset  ID, and hyperparameters alongside the .pkl files.
  13. Add logging and error handling throughout the pipeline.

  P3 — Nice to have:

  14. Complete GSE49036 validation for a true three-dataset generalization
  claim.
  15. Apply ComBat or limma batch correction instead of mean centering for
  cross-platform validation.
  16. Investigate confounders — age, sex, medication status — in the GSE99039
  cohort.
  17. Add unit tests for the predictor, gene mapper, and preprocessing
  6. Extract shared utilities — one module for data loading, normalization,
  and splitting.
  7. Replace hardcoded line numbers with metadata parsing (search for
  !Sample_characteristics dynamically).
  8. Translate comments to English or add bilingual annotations.
  9. Delete empty placeholder files or implement them.

  P2 — Required for deployment:

  10. Calibrate predict_proba using Platt scaling or isotonic regression on a
  held-out set.
  11. Add input validation in the Streamlit app — reject CSVs missing critical  genes rather than imputing zeros.
  12. Add model version metadata — save training date, gene list hash, dataset  ID, and hyperparameters alongside the .pkl files.
  13. Add logging and error handling throughout the pipeline.

  P3 — Nice to have:

  14. Complete GSE49036 validation for a true three-dataset generalization
  claim.
  15. Apply ComBat or limma batch correction instead of mean centering for
  cross-platform validation.
  16. Investigate confounders — age, sex, medication status — in the GSE99039
  cohort.
  17. Add unit tests for the predictor, gene mapper, and preprocessing
  utilities.