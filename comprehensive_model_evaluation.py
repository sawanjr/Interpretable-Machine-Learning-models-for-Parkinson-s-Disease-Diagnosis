"""
ParkinSenseDB Tuned Traditional Model Evaluation
================================================

What this script does:
- Evaluates strongly tuned traditional ML models on trial_level_features.csv
- Uses repeated StratifiedGroupKFold on SubjectID (no subject leakage)
- Uses nested CV (inner tuning only on training fold)
- Reports robust aggregate metrics across all outer folds

Models included:
- SVM (RBF)
- Logistic Regression (L1/L2/ElasticNet)
- Random Forest
- Gradient Boosting
- k-NN (baseline)
- XGBoost (if installed)

This version intentionally focuses on classical tabular models first.
"""

import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3
INNER_SPLITS = 3
SCORING = "balanced_accuracy"


def to_serializable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def build_outer_splits(X_data, y_data, groups, n_splits, n_repeats, random_state):
    outer_splits = []
    for repeat_idx in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + repeat_idx,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(
            splitter.split(X_data, y_data, groups), start=1
        ):
            outer_splits.append(
                {
                    "repeat": repeat_idx + 1,
                    "fold": fold_idx,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
    return outer_splits


def print_split_sanity(outer_splits, groups):
    print("\nLeakage verification across repeated outer CV:")
    for split in outer_splits:
        train_subjects = set(groups[split["train_idx"]])
        test_subjects = set(groups[split["test_idx"]])
        overlap = train_subjects.intersection(test_subjects)
        label = f"Repeat {split['repeat']}, Fold {split['fold']}"
        if overlap:
            print(f"  [FAIL] {label}: overlap found -> {sorted(list(overlap))}")
        else:
            print(
                f"  [OK] {label}: Train={len(split['train_idx'])}, "
                f"Test={len(split['test_idx'])}, no overlap"
            )


def evaluate_model(
    model_name,
    estimator,
    param_distributions,
    n_iter,
    X_data,
    y_data,
    groups,
    outer_splits,
):
    print("\n" + "=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    fold_records = []
    model_start = time.time()

    for outer_idx, split in enumerate(outer_splits, start=1):
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]

        X_train = X_data[train_idx]
        X_test = X_data[test_idx]
        y_train = y_data[train_idx]
        y_test = y_data[test_idx]
        groups_train = groups[train_idx]

        inner_cv = StratifiedGroupKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + split["repeat"] * 100 + split["fold"],
        )

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=SCORING,
            cv=inner_cv,
            refit=True,
            random_state=RANDOM_STATE + split["repeat"] * 100 + split["fold"],
            n_jobs=-1,
            verbose=0,
        )

        search.fit(X_train, y_train, groups=groups_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)[:, 1]
        else:
            y_score = best_model.decision_function(X_test)

        fold_acc = accuracy_score(y_test, y_pred)
        fold_prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        fold_rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        fold_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        fold_auc = safe_auc(y_test, y_score)

        fold_records.append(
            {
                "repeat": split["repeat"],
                "fold": split["fold"],
                "global_fold": outer_idx,
                "accuracy": fold_acc,
                "precision": fold_prec,
                "recall": fold_rec,
                "f1": fold_f1,
                "auc": fold_auc,
                "best_params": to_serializable(search.best_params_),
                "best_inner_score": float(search.best_score_),
            }
        )

        print(
            f"  Outer {outer_idx:02d}/{len(outer_splits)} "
            f"(R{split['repeat']} F{split['fold']}): "
            f"Acc={fold_acc:.4f}, F1={fold_f1:.4f}, AUC={fold_auc:.4f}"
        )

    elapsed = time.time() - model_start

    accuracies = np.array([r["accuracy"] for r in fold_records], dtype=float)
    precisions = np.array([r["precision"] for r in fold_records], dtype=float)
    recalls = np.array([r["recall"] for r in fold_records], dtype=float)
    f1s = np.array([r["f1"] for r in fold_records], dtype=float)
    aucs = np.array([r["auc"] for r in fold_records], dtype=float)

    result = {
        "Model": model_name,
        "Accuracy_Mean": float(np.mean(accuracies)),
        "Accuracy_Std": float(np.std(accuracies)),
        "Precision_Mean": float(np.mean(precisions)),
        "Recall_Mean": float(np.mean(recalls)),
        "F1_Mean": float(np.mean(f1s)),
        "AUC_Mean": float(np.nanmean(aucs)),
        "Time_sec": float(elapsed),
        "Fold_Details": fold_records,
    }

    print(
        f"  -> {model_name}: "
        f"Acc={result['Accuracy_Mean']:.4f} +/- {result['Accuracy_Std']:.4f}, "
        f"F1={result['F1_Mean']:.4f}, AUC={result['AUC_Mean']:.4f}, "
        f"Time={elapsed:.1f}s"
    )

    return result


def main():
    np.random.seed(RANDOM_STATE)

    print("=" * 80)
    print("ParkinSenseDB Tuned Traditional Model Evaluation")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Outer CV: {N_SPLITS}-fold StratifiedGroupKFold x {N_REPEATS} repeats")
    print(f"Inner CV: {INNER_SPLITS}-fold StratifiedGroupKFold")
    print(f"Hyperparameter search scoring: {SCORING}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("STEP 1: Loading and validating data")
    print("=" * 80)

    try:
        df = pd.read_csv("trial_level_features.csv")
    except FileNotFoundError:
        print("[ERROR] trial_level_features.csv not found.")
        print("Run preprocess_parkinsense.py first.")
        return

    print(f"[OK] Loaded trial_level_features.csv")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Subjects: {df['SubjectID'].nunique()}")
    print(f"  Trials: {len(df)}")
    print(f"  PD subjects: {df[df['Label'] == 1]['SubjectID'].nunique()}")
    print(f"  Control subjects: {df[df['Label'] == 0]['SubjectID'].nunique()}")
    print(f"  Class distribution: {df['Label'].value_counts().to_dict()}")

    missing_count = int(df.isnull().sum().sum())
    if missing_count > 0:
        print(f"[WARN] Missing values found: {missing_count}; filling with column medians")
        numeric_medians = df.median(numeric_only=True)
        df = df.fillna(numeric_medians)
    else:
        print("[OK] No missing values found")

    print("\n" + "=" * 80)
    print("STEP 2: Feature matrix and groups")
    print("=" * 80)

    exclude_cols = ["SubjectID", "Test", "Gait", "TrialLength", "Sex", "Age", "Status", "Label"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X_data = df[feature_cols].values
    y_data = df["Label"].values
    groups = df["SubjectID"].values

    print(f"[OK] Feature columns: {len(feature_cols)}")
    print(f"  Example features: {feature_cols[:5]}")
    print(f"[OK] X shape: {X_data.shape}")
    print(f"[OK] y shape: {y_data.shape}")
    print(f"[OK] Groups (SubjectID): {len(np.unique(groups))}")

    print("\nGroup distribution:")
    for label in [0, 1]:
        mask = y_data == label
        label_subjects = len(np.unique(groups[mask]))
        label_trials = int(mask.sum())
        label_name = "Control" if label == 0 else "PD"
        print(f"  Label {label} ({label_name}): {label_subjects} subjects, {label_trials} trials")

    print("\n" + "=" * 80)
    print("STEP 3: Build repeated grouped CV")
    print("=" * 80)

    outer_splits = build_outer_splits(
        X_data=X_data,
        y_data=y_data,
        groups=groups,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    print(f"[OK] Total outer folds: {len(outer_splits)}")
    print_split_sanity(outer_splits, groups)

    # Build model configs
    print("\n" + "=" * 80)
    print("STEP 4: Model training with stronger tuning")
    print("=" * 80)

    model_configs = []

    svm_estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]
    )
    svm_params = {
        "clf__C": np.logspace(-3, 3, 40).tolist(),
        "clf__gamma": ["scale"] + np.logspace(-4, 0, 30).tolist(),
        "clf__class_weight": [None, "balanced"],
    }
    model_configs.append(("SVM (RBF)", svm_estimator, svm_params, 24))

    lr_estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    max_iter=6000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    lr_params = [
        {
            "clf__penalty": ["l1"],
            "clf__C": np.logspace(-4, 2, 40).tolist(),
            "clf__class_weight": [None, "balanced"],
        },
        {
            "clf__penalty": ["l2"],
            "clf__C": np.logspace(-4, 2, 40).tolist(),
            "clf__class_weight": [None, "balanced"],
        },
        {
            "clf__penalty": ["elasticnet"],
            "clf__C": np.logspace(-4, 2, 30).tolist(),
            "clf__l1_ratio": np.linspace(0.1, 0.9, 9).tolist(),
            "clf__class_weight": [None, "balanced"],
        },
    ]
    model_configs.append(("Logistic Regression (tuned)", lr_estimator, lr_params, 24))

    rf_estimator = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_params = {
        "n_estimators": [200, 300, 400, 600, 800, 1000],
        "max_depth": [None, 4, 6, 8, 12, 16, 24],
        "min_samples_split": [2, 3, 5, 8, 12],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_features": ["sqrt", "log2", 0.4, 0.6, 0.8],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    model_configs.append(("Random Forest (tuned)", rf_estimator, rf_params, 24))

    gb_estimator = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_params = {
        "n_estimators": [100, 200, 300, 400, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.6, 0.75, 0.9, 1.0],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    model_configs.append(("Gradient Boosting (tuned)", gb_estimator, gb_params, 20))

    knn_estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]
    )
    knn_params = {
        "clf__n_neighbors": list(range(3, 32, 2)),
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    }
    model_configs.append(("k-NN (baseline tuned)", knn_estimator, knn_params, 15))

    if XGBOOST_AVAILABLE:
        class_counts = np.bincount(y_data.astype(int))
        if len(class_counts) == 2 and class_counts[1] > 0:
            pos_weight = class_counts[0] / class_counts[1]
        else:
            pos_weight = 1.0

        xgb_estimator = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb_params = {
            "n_estimators": [200, 300, 500, 700, 900],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
            "subsample": [0.6, 0.75, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0.0, 0.1, 0.3, 0.5, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [1.0, 3.0, 5.0, 10.0, 20.0],
            "scale_pos_weight": [1.0, pos_weight],
        }
        model_configs.append(("XGBoost (tuned)", xgb_estimator, xgb_params, 24))
    else:
        print("[INFO] xgboost not installed; XGBoost model will be skipped.")

    results = []
    for model_name, estimator, params, n_iter in model_configs:
        result = evaluate_model(
            model_name=model_name,
            estimator=estimator,
            param_distributions=params,
            n_iter=n_iter,
            X_data=X_data,
            y_data=y_data,
            groups=groups,
            outer_splits=outer_splits,
        )
        results.append(result)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    results_df = pd.DataFrame(
        [
            {
                "Model": r["Model"],
                "Accuracy": f"{r['Accuracy_Mean']:.4f} +/- {r['Accuracy_Std']:.4f}",
                "Accuracy_Mean": r["Accuracy_Mean"],
                "Accuracy_Std": r["Accuracy_Std"],
                "F1_Score": f"{r['F1_Mean']:.4f}",
                "AUC": f"{r['AUC_Mean']:.4f}",
                "Time_sec": f"{r['Time_sec']:.1f}",
            }
            for r in results
        ]
    )

    results_df = results_df.sort_values("Accuracy_Mean", ascending=False)

    print("\nMODEL RANKING (by repeated grouped CV accuracy):")
    print(results_df[["Model", "Accuracy", "F1_Score", "AUC", "Time_sec"]].to_string(index=False))

    results_df.to_csv("model_comparison_results.csv", index=False)
    with open("model_detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(results), f, indent=2)

    print("\n[OK] Results saved to: model_comparison_results.csv")
    print("[OK] Detailed results saved to: model_detailed_results.json")

    best_model = results_df.iloc[0]
    total_outer_folds = len(outer_splits)

    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    print(f"Best model: {best_model['Model']}")
    print(f"Accuracy: {best_model['Accuracy']}")
    print(f"F1: {best_model['F1_Score']}")
    print(f"AUC: {best_model['AUC']}")

    print("\n95% confidence intervals for top models:")
    for _, row in results_df.iterrows():
        mean = row["Accuracy_Mean"]
        std = row["Accuracy_Std"]
        ci = 1.96 * (std / np.sqrt(total_outer_folds))
        print(f"  {row['Model']:<30}: {mean:.4f} +/- {ci:.4f} [{mean - ci:.4f}, {mean + ci:.4f}]")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Models evaluated: {len(results)}")
    print(f"Outer CV folds: {N_SPLITS} x {N_REPEATS} = {total_outer_folds}")
    print(f"Data: trial_level_features.csv ({len(df)} trials, {df['SubjectID'].nunique()} subjects)")
    print("Notes:")
    print("  1. Group-aware repeated CV was used in all outer evaluations.")
    print("  2. Hyperparameter tuning stayed strictly inside each training fold.")
    print("  3. This setup is conservative and clinically more realistic.")


if __name__ == "__main__":
    main()
