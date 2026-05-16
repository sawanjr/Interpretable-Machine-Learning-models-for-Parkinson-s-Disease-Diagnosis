# Publication Polish Add-ons

This folder adds manuscript-focused artifacts on top of `final_explainability_outputs`.

## Added Figures
- Detailed methodology flow diagram with dataset and CV numbers
- Test metric boxplots across outer folds
- Train-test gap summary with std error bars
- Mean inner vs outer AUC bar chart
- Mean PR curves (combined and per model)
- Curated LIME panel (TP/TN/FP/FN) per model
- SHAP top-feature bar charts with alias labels

## Added Tables
- extended_metric_summary.csv
- pairwise_statistical_comparisons.csv
- feature_alias_mapping.csv
- curated_lime_cases.csv

## Current Core Performance
- SVM (RBF): test AUC 0.857 +/- 0.131, test accuracy 0.805 +/- 0.101
- Logistic Regression (tuned): test AUC 0.871 +/- 0.104, test accuracy 0.793 +/- 0.086
- XGBoost (tuned): test AUC 0.875 +/- 0.090, test accuracy 0.802 +/- 0.102
