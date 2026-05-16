# Explainable Machine Learning for Parkinson's Disease Classification from IMU Gait Signals

This repository contains the source code and supporting artifacts for our study on **Parkinson's Disease (PD) versus Healthy Control classification from inertial measurement unit (IMU) gait recordings**, with model interpretability provided through **SHAP** and **LIME**.

It accompanies the associated research paper and is intended to allow the editor, reviewers, and any interested reader to reproduce the experiments, inspect the generated figures and tables, and verify the explainability analyses reported in the paper.

---

## 1. Overview

We perform binary classification (Control vs. PD) on the **ParkinSenseDB** multimodal gait dataset, using only its IMU (accelerometer, gyroscope, magnetometer) channels. The pipeline:

1. Loads precomputed trial-level features (`trial_level_features.csv`).
2. Evaluates six strongly tuned classical machine-learning classifiers under **repeated subject-aware Stratified Group K-Fold cross-validation** with **nested hyperparameter tuning**, so that no subject ever appears in both the training and test folds.
3. Applies a fold-safe **explainability suite** built around **SHAP** (global and local) and **LIME** (TP / TN / FP / FN cases) on the three top-performing models (SVM-RBF, Logistic Regression, XGBoost).
4. Produces publication-ready figures and tables, including ROC, precision–recall, calibration, confusion-matrix, train/test-gap, methodology-flow, SHAP global/beeswarm/dependence/waterfall/force plots, and curated LIME panels.

The headline cross-validated results (mean ± std over 15 outer folds = 5 folds × 3 repeats) are:

| Model                          | Test AUC          | Test Accuracy     |
|--------------------------------|-------------------|-------------------|
| SVM (RBF)                      | 0.857 ± 0.131     | 0.805 ± 0.101     |
| Logistic Regression (tuned)    | 0.871 ± 0.104     | 0.793 ± 0.086     |
| XGBoost (tuned)                | 0.875 ± 0.090     | 0.802 ± 0.102     |

---

## 2. Repository Contents

```
.
├── README.md                              # This file
├── dataset_description.md                 # Detailed description of ParkinSenseDB and its file layout
├── comprehensive_model_evaluation.py      # Tuned classical-ML evaluation under nested grouped CV
├── final_explainability_suite.py          # SHAP + LIME explainability + performance plots
├── publication_polish_postprocess.py      # Adds publication-quality artifacts on top of the suite outputs
└── publication_polish_outputs/            # Curated outputs used in the paper
    ├── publication_polish_report.md
    ├── tables/                            # Extended metric summary, statistical comparisons, feature aliases, etc.
    └── figures/                           # Methodology, performance, LIME panels, SHAP feature panels
```

The full ParkinSenseDB raw IMU recordings, demographics, and annotations are **not** redistributed in this repository (see Section 7 for access). All preprocessing steps required to convert raw IMU CSV files into `trial_level_features.csv` are described in `dataset_description.md` and the paper.

---

## 3. Dataset

We use the public **ParkinSenseDB** multimodal gait dataset. Only the IMU channels are used here; motion-capture and EMG data are out of scope.

| Property              | Value                                                  |
|-----------------------|--------------------------------------------------------|
| Subjects              | 53 (29 PD, 23 Control, 1 mislabeled record cleaned)    |
| Total gait trials     | 747                                                    |
| Test sessions         | TEST A, TEST B, TEST C, TEST D                         |
| Trials per subject    | 12–16                                                  |
| Age range             | 36–76 years                                            |
| IMU channels per file | 10 (Frame + 3-axis Magnetometer + 3-axis Gyroscope + 3-axis Accelerometer) |
| Sampling              | Native IMU sensor rate                                 |
| Trial length          | Typically 1500–4000 frames (variable)                  |

A full description of the file layout, joins between `Annotation.csv` and `Demographic_Information.xlsx`, IMU channel units, and known data-quality notes (e.g. the `Controllo` typo on subject 013) is in **[dataset_description.md](dataset_description.md)**.

The classification label (`Label`) is derived from the `Status` column: `Control = 0`, `PD = 1`. All splits are performed at the **subject** level (`SubjectID` as the group) to prevent any leakage between training and test data.

---

## 4. Pipeline and Reproducibility

### 4.1 Input data file

Both scripts read from a precomputed trial-level feature table:

```
trial_level_features.csv
```

Each row corresponds to one gait trial (subject × test × gait), with metadata columns (`SubjectID`, `Test`, `Gait`, `TrialLength`, `Sex`, `Age`, `Status`, `Label`) and a large number of statistical/spectral features extracted from the IMU signals. The exact feature-extraction step is described in the paper; the columns excluded from `X` are listed in `comprehensive_model_evaluation.py` and `final_explainability_suite.py`.

### 4.2 Step 1 — Model evaluation: `comprehensive_model_evaluation.py`

Performs **nested, grouped, repeated cross-validation** on six classical ML models:

- SVM with RBF kernel
- Logistic Regression (L1 / L2 / ElasticNet)
- Random Forest
- Gradient Boosting
- k-NN (baseline)
- XGBoost (if `xgboost` is installed)

Key design choices (defined as constants at the top of the file):

- `RANDOM_STATE = 42`
- Outer CV: **5-fold StratifiedGroupKFold × 3 repeats** = **15 outer folds**
- Inner CV: **3-fold StratifiedGroupKFold** for hyperparameter search
- Inner search: `RandomizedSearchCV` with `scoring = "balanced_accuracy"`
- All folds are sanity-checked for subject overlap between train and test (printed in the log)
- Missing values, if any, are filled with column medians

Outputs:

- `model_comparison_results.csv` — ranked summary table of all models
- `model_detailed_results.json` — per-fold metrics and the **best hyperparameters chosen per outer fold** (consumed by the explainability suite)

Run:

```bash
python comprehensive_model_evaluation.py
```

### 4.3 Step 2 — Explainability and performance plots: `final_explainability_suite.py`

This is the explainability code referenced in the paper. It uses **SHAP** and **LIME** in a fold-safe way: explanations are produced only on the **outer-test rows** of each fold, never on training rows the model has already seen.

For each of the three top models (**SVM-RBF**, **Logistic Regression**, **XGBoost**):

- Refits the model per outer fold using the **per-fold best hyperparameters** loaded from `model_detailed_results.json`.
- Records per-fold train/test accuracy, AUC, F1, and confusion-matrix predictions.
- Computes a fold-level **outer ROC** on the held-out test rows and an **inner out-of-fold (OOF) ROC** on the outer-train rows, then plots their mean curves with ±1 std bands.
- Computes **SHAP** values appropriate to the model family:
    - `shap.KernelExplainer` for SVM (RBF), with a k-means / sampled background of size `--background-size`.
    - `shap.LinearExplainer` for Logistic Regression (on standardized features).
    - `shap.TreeExplainer` for XGBoost.
- Saves SHAP **global importance bars**, **beeswarm summary plots**, **dependence plots** for the top features, and per-sample **waterfall / force plots**.
- Computes **LIME** explanations on selected **TP / TN / FP / FN** cases per fold using `LimeTabularExplainer`, saving each as an interactive HTML report and high-resolution PNG / PDF / SVG.
- Aggregates outer-test predictions across all folds to plot an **overall confusion matrix** and **calibration curve** per model.
- Draws a **methodology flow diagram** (publication quality) describing the nested grouped CV with SHAP and LIME branches.
- Saves a **combined multi-model ROC comparison** figure.

Default outputs go to `final_explainability_outputs/` and are organized as:

```
final_explainability_outputs/
├── figures/
│   ├── common/                            # Methodology flow, combined ROC
│   ├── svm/                               # SVM SHAP + LIME + performance plots
│   ├── lr/                                # Logistic Regression SHAP + LIME + performance plots
│   └── xgb/                               # XGBoost SHAP + LIME + performance plots
├── tables/                                # Per-model fold metrics, test predictions, LIME case reports, SHAP importances
└── logs/
```

Every figure is saved in **PNG (600 dpi), PDF, and SVG** for print-quality reuse.

Run:

```bash
python final_explainability_suite.py \
    --data trial_level_features.csv \
    --detailed-results model_detailed_results.json \
    --output-dir final_explainability_outputs
```

Useful flags (all have defaults; see `parse_args` in the script):

| Flag                      | Default | Meaning                                              |
|---------------------------|---------|------------------------------------------------------|
| `--n-splits`              | 5       | Outer CV folds                                       |
| `--n-repeats`             | 3       | Outer CV repeats                                     |
| `--inner-splits`          | 3       | Inner CV folds for inner-ROC OOF                     |
| `--max-shap-samples`      | 30      | Max SHAP samples per fold                            |
| `--background-size`       | 60      | SHAP background size                                 |
| `--svm-kernel-nsamples`   | 150     | KernelSHAP `nsamples`                                |
| `--lime-per-fold-per-type`| 1       | LIME cases per TP/TN/FP/FN per fold                  |
| `--lime-num-features`     | 10      | Features per LIME explanation                        |
| `--top-features`          | 15      | Features in SHAP global plots                        |

The script is written to run on **Google Colab** as well as a local machine.

### 4.4 Step 3 — Publication polish: `publication_polish_postprocess.py`

Reads the outputs of `final_explainability_suite.py` and adds manuscript-grade artifacts:

- A detailed methodology flow diagram annotated with dataset / split numbers.
- Outer-test metric **boxplots** across folds.
- **Train–test gap** summary with std error bars.
- **Mean inner-vs-outer AUC** bar chart.
- **Mean Precision–Recall curves** (combined + per model).
- A curated **LIME TP/TN/FP/FN panel** per model.
- **SHAP top-feature bar charts** with human-readable alias labels.
- Tables: `extended_metric_summary.csv`, `pairwise_statistical_comparisons.csv`, `feature_alias_mapping.csv`, `curated_lime_cases.csv`.

The curated outputs are already provided in **[publication_polish_outputs/](publication_polish_outputs/)** so reviewers can inspect them without rerunning the pipeline. A short summary of these artifacts is in [publication_polish_outputs/publication_polish_report.md](publication_polish_outputs/publication_polish_report.md).

Run:

```bash
python publication_polish_postprocess.py \
    --base-dir final_explainability_outputs \
    --output-dir publication_polish_outputs
```

---

## 5. Requirements

The code targets **Python 3.10+** and the following packages:

```
numpy
pandas
scikit-learn
xgboost
shap
lime
matplotlib
openpyxl     # only if regenerating features from the original .xlsx demographics
```

A minimal installation:

```bash
pip install numpy pandas scikit-learn xgboost shap lime matplotlib openpyxl
```

If `xgboost` is not available, `comprehensive_model_evaluation.py` will skip the XGBoost model and `final_explainability_suite.py` will skip the XGBoost branch. SVM and Logistic Regression results will be produced regardless.

---

## 6. Reproducing the Results in the Paper

A full end-to-end run, assuming `trial_level_features.csv` is in the working directory:

```bash
# 1. Tuned classical-ML evaluation (writes model_detailed_results.json + model_comparison_results.csv)
python comprehensive_model_evaluation.py

# 2. Fold-safe SHAP + LIME explainability suite
python final_explainability_suite.py

# 3. Publication-quality post-processing on top of suite outputs
python publication_polish_postprocess.py
```

All randomness is seeded with `RANDOM_STATE = 42`. With identical input data and library versions, the reported metrics and figures are reproducible up to the small variability introduced by parallel hyperparameter search.

---

## 7. Data Availability

The ParkinSenseDB dataset is distributed by its original authors; please refer to the dataset description and the references listed in the paper for access details. The features file `trial_level_features.csv` used as direct input to the scripts is derived deterministically from the public ParkinSenseDB IMU recordings using the preprocessing described in `dataset_description.md` and the methods section of the paper.

---

## 8. Citation

If you use this code or build on the experiments in this repository, please cite our paper. Full citation information will be provided once the paper is published.

---

## 9. License and Contact

The code in this repository is released for **academic research and review** purposes accompanying the paper. For questions or to report an issue with reproducing the results, please open an issue on this repository or contact the corresponding author listed in the paper.
