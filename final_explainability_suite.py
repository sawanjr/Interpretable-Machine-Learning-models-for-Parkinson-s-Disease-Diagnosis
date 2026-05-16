"""
Final explainability and visualization suite for SVM, Logistic Regression, XGBoost.

Designed for Google Colab.

Features:
- Fold-safe explainability (outer test fold only)
- SHAP: global importance, beeswarm, dependence, waterfall, force
- LIME: TP/TN/FP/FN local explanations (HTML + HD PNG/PDF/SVG)
- Mean outer and inner ROC-AUC curves
- Training/testing accuracy and AUC across repeats with std error bars
- Aggregated confusion matrix and calibration plot
- Combined multi-model ROC comparison
- Methodology flow diagram (publication quality)
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def import_runtime_packages() -> tuple[Any, Any, Any, Any, Any, bool]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    try:
        import xgboost as xgb

        xgb_available = True
    except Exception:
        xgb = None
        xgb_available = False

    return plt, FancyBboxPatch, shap, LimeTabularExplainer, xgb, xgb_available


RANDOM_STATE = 42


@dataclass
class ModelSpec:
    key: str
    pretty_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final explainability suite")
    parser.add_argument("--data", default="trial_level_features.csv", help="Input trial-level CSV")
    parser.add_argument(
        "--detailed-results",
        default="model_detailed_results.json",
        help="JSON from tuned grouped CV run",
    )
    parser.add_argument("--output-dir", default="final_explainability_outputs", help="Output directory")
    parser.add_argument("--n-splits", type=int, default=5, help="Outer CV splits")
    parser.add_argument("--n-repeats", type=int, default=3, help="Outer repeats")
    parser.add_argument("--inner-splits", type=int, default=3, help="Inner splits for inner-ROC OOF")
    parser.add_argument("--max-shap-samples", type=int, default=30, help="Max SHAP test samples per fold")
    parser.add_argument("--background-size", type=int, default=60, help="SHAP background size")
    parser.add_argument("--svm-kernel-nsamples", type=int, default=150, help="Kernel SHAP nsamples")
    parser.add_argument("--lime-per-fold-per-type", type=int, default=1, help="LIME cases per TP/TN/FP/FN per fold")
    parser.add_argument("--lime-num-features", type=int, default=10, help="Top features in each LIME explanation")
    parser.add_argument("--top-features", type=int, default=15, help="Top features in SHAP global plots")
    return parser.parse_args()


def set_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.figsize": (14, 8),
        }
    )


def save_figure_all_formats(fig: Any, base_path: str, dpi_png: int = 600) -> None:
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig.savefig(base_path + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    fig.savefig(base_path + ".svg", bbox_inches="tight")


def ensure_output_dirs(base_dir: str, model_keys: list[str]) -> None:
    for d in ["tables", "figures", "logs"]:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

    for mk in model_keys:
        os.makedirs(os.path.join(base_dir, "tables", mk), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "figures", mk, "shap"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "figures", mk, "lime"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "figures", mk, "performance"), exist_ok=True)

    os.makedirs(os.path.join(base_dir, "figures", "common"), exist_ok=True)


def load_data(data_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(data_path)
    exclude_cols = ["SubjectID", "Test", "Gait", "TrialLength", "Sex", "Age", "Status", "Label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values
    y = df["Label"].values.astype(int)
    groups = df["SubjectID"].values
    return df, X, y, groups, feature_cols


def build_outer_splits(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, n_repeats: int
) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = []
    for repeat in range(n_repeats):
        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )
        for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
            splits.append({"repeat": repeat + 1, "fold": fold, "train_idx": tr, "test_idx": te})
    return splits


def load_best_params(json_path: str) -> dict[str, dict[int, dict[str, Any]]]:
    raw = json.loads(open(json_path, "r", encoding="utf-8").read())
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for model in raw:
        model_name = model["Model"]
        result[model_name] = {}
        for fd in model["Fold_Details"]:
            result[model_name][int(fd["global_fold"])] = fd.get("best_params", {})
    return result


def sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(n), size=k, replace=False))


def build_model(model_name: str, best_params: dict[str, Any], xgb: Any, xgb_available: bool) -> Any:
    if model_name == "SVM (RBF)":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]
        )
        if best_params:
            model.set_params(**best_params)
        return model

    if model_name == "Logistic Regression (tuned)":
        model = Pipeline(
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
        if best_params:
            model.set_params(**best_params)
        return model

    if model_name == "XGBoost (tuned)":
        if not xgb_available:
            raise RuntimeError("XGBoost is not available. Install xgboost in Colab.")
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
        if best_params:
            model.set_params(**best_params)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def get_prob_scores(model: Any, X_part: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_part)[:, 1]
    raise ValueError("Model does not support predict_proba")


def normalize_shap_output(shap_values: Any, expected_value: Any) -> tuple[np.ndarray, float]:
    values = shap_values
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    base = expected_value
    if isinstance(base, (list, np.ndarray)):
        base_arr = np.asarray(base).flatten()
        base = base_arr[1] if base_arr.size > 1 else base_arr[0]
    base = float(base)

    return values, base


def select_case_indices(y_true: np.ndarray, y_pred: np.ndarray, k_per_type: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    tn = np.where((y_true == 0) & (y_pred == 0))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]

    def pick(arr: np.ndarray) -> np.ndarray:
        if len(arr) <= k_per_type:
            return arr
        return np.sort(rng.choice(arr, size=k_per_type, replace=False))

    return {"TP": pick(tp), "TN": pick(tn), "FP": pick(fp), "FN": pick(fn)}


def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
    }


def mean_roc_stats(roc_records: list[dict[str, Any]], n_grid: int = 300) -> dict[str, Any]:
    mean_fpr = np.linspace(0.0, 1.0, n_grid)
    tprs = []
    aucs = []
    for rec in roc_records:
        fpr = np.asarray(rec["fpr"], dtype=float)
        tpr = np.asarray(rec["tpr"], dtype=float)
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)
        aucs.append(float(rec["auc"]))

    tprs_arr = np.asarray(tprs, dtype=float)
    mean_tpr = tprs_arr.mean(axis=0)
    std_tpr = tprs_arr.std(axis=0)
    mean_tpr[-1] = 1.0

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
    }


def plot_mean_inner_outer_roc(
    plt: Any,
    model_title: str,
    outer_stats: dict[str, Any],
    inner_stats: dict[str, Any],
    out_base: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        outer_stats["mean_fpr"],
        outer_stats["mean_tpr"],
        color="#1f77b4",
        lw=2.5,
        label=f"Outer ROC (AUC={outer_stats['mean_auc']:.3f} +/- {outer_stats['std_auc']:.3f})",
    )
    ax.fill_between(
        outer_stats["mean_fpr"],
        np.maximum(outer_stats["mean_tpr"] - outer_stats["std_tpr"], 0),
        np.minimum(outer_stats["mean_tpr"] + outer_stats["std_tpr"], 1),
        color="#1f77b4",
        alpha=0.15,
    )

    ax.plot(
        inner_stats["mean_fpr"],
        inner_stats["mean_tpr"],
        color="#2ca02c",
        lw=2.5,
        label=f"Inner ROC (AUC={inner_stats['mean_auc']:.3f} +/- {inner_stats['std_auc']:.3f})",
    )
    ax.fill_between(
        inner_stats["mean_fpr"],
        np.maximum(inner_stats["mean_tpr"] - inner_stats["std_tpr"], 0),
        np.minimum(inner_stats["mean_tpr"] + inner_stats["std_tpr"], 1),
        color="#2ca02c",
        alpha=0.12,
    )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_title}: Mean Outer and Mean Inner ROC-AUC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    save_figure_all_formats(fig, out_base)
    plt.close(fig)


def plot_repeat_train_test_bars(
    plt: Any,
    fold_df: pd.DataFrame,
    metric_col_train: str,
    metric_col_test: str,
    title: str,
    y_label: str,
    out_base: str,
) -> None:
    grouped = fold_df.groupby("repeat")
    repeats = sorted(fold_df["repeat"].unique())

    train_mean = grouped[metric_col_train].mean().reindex(repeats).values
    train_std = grouped[metric_col_train].std().reindex(repeats).fillna(0).values
    test_mean = grouped[metric_col_test].mean().reindex(repeats).values
    test_std = grouped[metric_col_test].std().reindex(repeats).fillna(0).values

    x = np.arange(len(repeats))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width / 2, train_mean, width, yerr=train_std, capsize=5, label="Train", color="#4c78a8")
    ax.bar(x + width / 2, test_mean, width, yerr=test_std, capsize=5, label="Test", color="#f58518")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Exp {r}" for r in repeats])
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure_all_formats(fig, out_base)
    plt.close(fig)


def plot_confusion_matrix(
    plt: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_base: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Control", "Parkinson"])
    ax.set_yticklabels(["Control", "Parkinson"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            txt = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure_all_formats(fig, out_base)
    plt.close(fig)


def plot_calibration(
    plt: Any,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    out_base: str,
) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2.2, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    save_figure_all_formats(fig, out_base)
    plt.close(fig)


def save_shap_global_plots(
    plt: Any,
    shap: Any,
    model_key: str,
    feature_names: list[str],
    shap_values_all: np.ndarray,
    X_all: np.ndarray,
    top_features: int,
    out_dir: str,
) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(shap_values_all), axis=0)
    imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )

    top_df = imp_df.head(top_features).copy()
    top_df["feature_wrapped"] = top_df["feature"].apply(lambda x: "\n".join(textwrap.wrap(str(x), 28)))

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.barh(top_df["feature_wrapped"][::-1], top_df["mean_abs_shap"][::-1], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{model_key.upper()} - SHAP Global Feature Importance (Top {top_features})")
    save_figure_all_formats(fig, os.path.join(out_dir, "figures", model_key, "shap", "global_importance_top_features"))
    plt.close(fig)

    plt.figure(figsize=(14, 9))
    shap.summary_plot(shap_values_all, X_all, feature_names=feature_names, max_display=top_features, show=False)
    plt.title(f"{model_key.upper()} - SHAP Summary (Beeswarm)")
    plt.tight_layout()
    fig = plt.gcf()
    save_figure_all_formats(fig, os.path.join(out_dir, "figures", model_key, "shap", "summary_beeswarm"))
    plt.close(fig)

    for feature in imp_df.head(3)["feature"].tolist():
        plt.figure(figsize=(12, 7))
        shap.dependence_plot(feature, shap_values_all, X_all, feature_names=feature_names, show=False)
        plt.title(f"{model_key.upper()} - SHAP Dependence: {feature}")
        plt.tight_layout()
        fig = plt.gcf()
        safe = feature.replace("/", "_").replace(" ", "_")[:90]
        save_figure_all_formats(fig, os.path.join(out_dir, "figures", model_key, "shap", f"dependence_{safe}"))
        plt.close(fig)

    imp_df.to_csv(os.path.join(out_dir, "tables", model_key, "shap_global_importance.csv"), index=False)
    return imp_df


def save_shap_local_plots(
    plt: Any,
    shap: Any,
    model_key: str,
    base_value: float,
    shap_vec: np.ndarray,
    x_vec: np.ndarray,
    feature_names: list[str],
    out_dir: str,
    suffix: str,
) -> None:
    try:
        exp = shap.Explanation(
            values=shap_vec,
            base_values=base_value,
            data=x_vec,
            feature_names=feature_names,
        )
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(exp, max_display=15, show=False)
        fig = plt.gcf()
        save_figure_all_formats(
            fig,
            os.path.join(out_dir, "figures", model_key, "shap", f"waterfall_{suffix}"),
        )
        plt.close(fig)
    except Exception:
        pass

    try:
        plt.figure(figsize=(14, 4.5))
        shap.force_plot(base_value, shap_vec, x_vec, feature_names=feature_names, matplotlib=True, show=False)
        fig = plt.gcf()
        save_figure_all_formats(
            fig,
            os.path.join(out_dir, "figures", model_key, "shap", f"force_{suffix}"),
        )
        plt.close(fig)
    except Exception:
        plt.close("all")


def draw_methodology_flow_diagram(plt: Any, FancyBboxPatch: Any, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, txt: str, color: str) -> tuple[float, float, float, float]:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#4a4a4a",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=11)
        return (x, y, w, h)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#222222"),
        )

    b1 = box(3.2, 22.2, 3.6, 1.0, "Dataset selection", "#dbe9f6")
    b2 = box(2.6, 20.6, 4.8, 1.1, "Preprocessing and\nnormalization", "#dbe9f6")
    b3 = box(2.7, 19.0, 4.6, 1.1, "Feature extraction\nand selection", "#dbe9f6")
    b4 = box(2.2, 17.4, 5.6, 1.2, "Nested grouped cross-validation\nfor classification", "#e9def7")
    b5 = box(2.8, 15.8, 4.4, 1.0, "Outer cross-validation fold", "#e9def7")

    b6 = box(0.6, 14.2, 2.6, 1.0, "Training set", "#efe7fb")
    b7 = box(0.4, 12.6, 3.0, 1.1, "Inner cross-validation", "#efe7fb")
    b8 = box(0.2, 11.0, 2.4, 1.0, "Validation set", "#efe7fb")
    b9 = box(2.9, 11.0, 2.8, 1.0, "Hyperparameter\ntuning", "#efe7fb")

    b10 = box(6.7, 12.6, 2.8, 1.1, "Final model\nselection", "#efe7fb")
    b11 = box(4.2, 9.2, 2.8, 1.1, "Final model\nevaluation", "#efe7fb")
    b12 = box(2.6, 7.3, 5.4, 1.2, "Explainable AI for\ninterpretability", "#deefd9")
    b13 = box(2.7, 5.5, 2.3, 1.0, "SHAP", "#f7d9e8")
    b14 = box(5.7, 5.5, 2.3, 1.0, "LIME", "#f7d9e8")

    arrow(5.0, 22.2, 5.0, 21.7)
    arrow(5.0, 20.6, 5.0, 20.1)
    arrow(5.0, 19.0, 5.0, 18.6)
    arrow(5.0, 17.4, 5.0, 16.8)
    arrow(5.0, 15.8, 2.0, 15.2)
    arrow(5.0, 15.8, 8.1, 13.7)

    arrow(2.0, 14.2, 2.0, 13.7)
    arrow(2.0, 12.6, 1.4, 12.0)
    arrow(2.0, 12.6, 4.3, 12.0)
    arrow(4.3, 11.0, 5.4, 10.3)
    arrow(8.1, 12.6, 6.1, 10.3)
    arrow(5.6, 9.2, 5.3, 8.5)
    arrow(5.3, 7.3, 3.9, 6.5)
    arrow(5.3, 7.3, 6.9, 6.5)

    ax.set_title("Proposed Methodology Flow Diagram", fontsize=20, pad=16)
    save_figure_all_formats(fig, os.path.join(out_dir, "figures", "common", "methodology_flow_diagram"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt, FancyBboxPatch, shap, LimeTabularExplainer, xgb, xgb_available = import_runtime_packages()
    set_style(plt)

    model_specs = [
        ModelSpec(key="svm", pretty_name="SVM (RBF)"),
        ModelSpec(key="lr", pretty_name="Logistic Regression (tuned)"),
        ModelSpec(key="xgb", pretty_name="XGBoost (tuned)"),
    ]
    if not xgb_available:
        model_specs = [m for m in model_specs if m.pretty_name != "XGBoost (tuned)"]

    ensure_output_dirs(args.output_dir, [m.key for m in model_specs])

    df, X, y, groups, feature_names = load_data(args.data)
    outer_splits = build_outer_splits(X, y, groups, args.n_splits, args.n_repeats)
    best_params_map = load_best_params(args.detailed_results)

    log_lines = [
        "Final Explainability Suite",
        f"Samples: {len(df)}",
        f"Features: {len(feature_names)}",
        f"Subjects: {df['SubjectID'].nunique()}",
        f"Outer folds: {len(outer_splits)} ({args.n_splits} x {args.n_repeats})",
        f"Models: {', '.join([m.pretty_name for m in model_specs])}",
    ]

    draw_methodology_flow_diagram(plt, FancyBboxPatch, args.output_dir)

    all_model_roc_for_comparison: list[dict[str, Any]] = []
    summary_rows = []

    for model_spec in model_specs:
        model_name = model_spec.pretty_name
        model_key = model_spec.key

        print("\n" + "=" * 96)
        print(f"Processing {model_name}")
        print("=" * 96)

        fold_rows = []
        pred_rows = []
        lime_rows = []

        shap_values_all_folds: list[np.ndarray] = []
        shap_data_all_folds: list[np.ndarray] = []

        outer_roc_records: list[dict[str, Any]] = []
        inner_roc_records: list[dict[str, Any]] = []

        local_shap_saved = False

        for outer_idx, split in enumerate(outer_splits, start=1):
            tr_idx = split["train_idx"]
            te_idx = split["test_idx"]

            X_train, X_test = X[tr_idx], X[te_idx]
            y_train, y_test = y[tr_idx], y[te_idx]
            g_train = groups[tr_idx]

            fold_params = best_params_map.get(model_name, {}).get(outer_idx, {})
            model = build_model(model_name, fold_params, xgb, xgb_available)
            model.fit(X_train, y_train)

            y_prob_test = get_prob_scores(model, X_test)
            y_pred_test = (y_prob_test >= 0.5).astype(int)
            y_prob_train = get_prob_scores(model, X_train)
            y_pred_train = (y_prob_train >= 0.5).astype(int)

            train_metrics = compute_fold_metrics(y_train, y_pred_train, y_prob_train)
            test_metrics = compute_fold_metrics(y_test, y_pred_test, y_prob_test)

            # Outer ROC
            fpr_o, tpr_o, _ = roc_curve(y_test, y_prob_test)
            outer_roc_records.append(
                {
                    "global_fold": outer_idx,
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "fpr": fpr_o.tolist(),
                    "tpr": tpr_o.tolist(),
                    "auc": test_metrics["auc"],
                }
            )

            # Inner OOF ROC on outer-train
            inner_cv = StratifiedGroupKFold(
                n_splits=args.inner_splits,
                shuffle=True,
                random_state=RANDOM_STATE + outer_idx,
            )
            oof_prob = np.full(y_train.shape[0], np.nan, dtype=float)

            for in_tr, in_val in inner_cv.split(X_train, y_train, g_train):
                in_model = clone(model)
                in_model.fit(X_train[in_tr], y_train[in_tr])
                oof_prob[in_val] = get_prob_scores(in_model, X_train[in_val])

            valid = ~np.isnan(oof_prob)
            y_train_oof = y_train[valid]
            p_train_oof = oof_prob[valid]
            fpr_i, tpr_i, _ = roc_curve(y_train_oof, p_train_oof)
            auc_i = auc(fpr_i, tpr_i)
            inner_roc_records.append(
                {
                    "global_fold": outer_idx,
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "fpr": fpr_i.tolist(),
                    "tpr": tpr_i.tolist(),
                    "auc": float(auc_i),
                }
            )

            fold_rows.append(
                {
                    "model": model_name,
                    "repeat": split["repeat"],
                    "fold": split["fold"],
                    "global_fold": outer_idx,
                    "train_accuracy": train_metrics["accuracy"],
                    "test_accuracy": test_metrics["accuracy"],
                    "train_auc": train_metrics["auc"],
                    "test_auc": test_metrics["auc"],
                    "train_f1": train_metrics["f1"],
                    "test_f1": test_metrics["f1"],
                    "inner_auc_oof": float(auc_i),
                }
            )

            for li, gi in enumerate(te_idx):
                pred_rows.append(
                    {
                        "model": model_name,
                        "repeat": split["repeat"],
                        "fold": split["fold"],
                        "global_fold": outer_idx,
                        "test_row_index": int(gi),
                        "SubjectID": df.iloc[gi]["SubjectID"],
                        "y_true": int(y_test[li]),
                        "y_pred": int(y_pred_test[li]),
                        "y_prob": float(y_prob_test[li]),
                    }
                )

            # SHAP per fold on sampled outer-test rows
            test_take = sample_indices(len(X_test), args.max_shap_samples, RANDOM_STATE + outer_idx)

            if model_name == "SVM (RBF)":
                bg_take = sample_indices(len(X_train), args.background_size, RANDOM_STATE + outer_idx + 1000)
                bg = X_train[bg_take]
                try:
                    if hasattr(shap, "kmeans"):
                        bg = shap.kmeans(bg, min(args.background_size, len(bg)))
                except Exception:
                    if hasattr(shap, "sample"):
                        bg = shap.sample(bg, min(args.background_size, len(bg)), random_state=RANDOM_STATE + outer_idx)

                X_sub = X_test[test_take]
                explainer = shap.KernelExplainer(lambda z: model.predict_proba(z)[:, 1], bg)
                shap_values_raw = explainer.shap_values(X_sub, nsamples=args.svm_kernel_nsamples)
                shap_values, shap_base = normalize_shap_output(shap_values_raw, explainer.expected_value)
                shap_data = X_sub

            elif model_name == "Logistic Regression (tuned)":
                scaler = model.named_steps["scaler"]
                clf = model.named_steps["clf"]
                X_train_scaled = scaler.transform(X_train)
                X_sub_scaled = scaler.transform(X_test[test_take])

                if hasattr(shap, "sample"):
                    bg = shap.sample(
                        X_train_scaled,
                        min(args.background_size, len(X_train_scaled)),
                        random_state=RANDOM_STATE + outer_idx,
                    )
                else:
                    bg_take = sample_indices(len(X_train_scaled), args.background_size, RANDOM_STATE + outer_idx + 2000)
                    bg = X_train_scaled[bg_take]

                explainer = shap.LinearExplainer(clf, bg)
                shap_values_raw = explainer.shap_values(X_sub_scaled)
                shap_values, shap_base = normalize_shap_output(shap_values_raw, explainer.expected_value)
                shap_data = X_sub_scaled

            else:
                X_sub = X_test[test_take]
                explainer = shap.TreeExplainer(model)
                shap_values_raw = explainer.shap_values(X_sub)
                shap_values, shap_base = normalize_shap_output(shap_values_raw, explainer.expected_value)
                shap_data = X_sub

            if shap_values.shape[1] == len(feature_names):
                shap_values_all_folds.append(shap_values)
                shap_data_all_folds.append(shap_data)

            if not local_shap_saved and shap_values.shape[0] > 0:
                local_idx = 0
                suffix = f"fold{outer_idx:02d}_sample0"
                save_shap_local_plots(
                    plt=plt,
                    shap=shap,
                    model_key=model_key,
                    base_value=shap_base,
                    shap_vec=shap_values[local_idx],
                    x_vec=shap_data[local_idx],
                    feature_names=feature_names,
                    out_dir=args.output_dir,
                    suffix=suffix,
                )
                local_shap_saved = True

            # LIME on TP/TN/FP/FN
            lime_explainer = LimeTabularExplainer(
                training_data=X_train,
                feature_names=feature_names,
                class_names=["Control", "Parkinson"],
                mode="classification",
                discretize_continuous=True,
                random_state=RANDOM_STATE + outer_idx,
            )

            selected_cases = select_case_indices(
                y_true=y_test,
                y_pred=y_pred_test,
                k_per_type=args.lime_per_fold_per_type,
                seed=RANDOM_STATE + outer_idx,
            )

            for case_type, idx_arr in selected_cases.items():
                for idx_local in idx_arr:
                    exp = lime_explainer.explain_instance(
                        data_row=X_test[idx_local],
                        predict_fn=model.predict_proba,
                        labels=(1,),
                        num_features=args.lime_num_features,
                    )

                    label_for_plot = 1 if 1 in exp.local_exp else list(exp.local_exp.keys())[0]

                    base_name = f"fold{outer_idx:02d}_{case_type}_row{int(te_idx[idx_local])}"
                    html_path = os.path.join(args.output_dir, "figures", model_key, "lime", f"{base_name}.html")
                    exp.save_to_file(html_path)

                    fig = exp.as_pyplot_figure(label=label_for_plot)
                    fig.set_size_inches(11, 7)
                    fig.tight_layout()
                    save_figure_all_formats(
                        fig,
                        os.path.join(args.output_dir, "figures", model_key, "lime", base_name),
                    )
                    plt.close(fig)

                    for feat, weight in exp.as_list(label=label_for_plot):
                        lime_rows.append(
                            {
                                "model": model_name,
                                "repeat": split["repeat"],
                                "fold": split["fold"],
                                "global_fold": outer_idx,
                                "case_type": case_type,
                                "test_row_index": int(te_idx[idx_local]),
                                "SubjectID": df.iloc[te_idx[idx_local]]["SubjectID"],
                                "y_true": int(y_test[idx_local]),
                                "y_pred": int(y_pred_test[idx_local]),
                                "y_prob": float(y_prob_test[idx_local]),
                                "feature_rule": feat,
                                "lime_weight": float(weight),
                            }
                        )

            print(
                f"  Fold {outer_idx:02d}/{len(outer_splits)} (R{split['repeat']} F{split['fold']}): "
                f"TestAcc={test_metrics['accuracy']:.4f}, TestAUC={test_metrics['auc']:.4f}, "
                f"InnerAUC={auc_i:.4f}"
            )

        # Save tables
        fold_df = pd.DataFrame(fold_rows)
        pred_df = pd.DataFrame(pred_rows)
        lime_df = pd.DataFrame(lime_rows)

        fold_df.to_csv(os.path.join(args.output_dir, "tables", model_key, "fold_metrics.csv"), index=False)
        pred_df.to_csv(os.path.join(args.output_dir, "tables", model_key, "fold_test_predictions.csv"), index=False)
        lime_df.to_csv(os.path.join(args.output_dir, "tables", model_key, "lime_case_reports.csv"), index=False)

        # ROC plots
        outer_stats = mean_roc_stats(outer_roc_records)
        inner_stats = mean_roc_stats(inner_roc_records)

        plot_mean_inner_outer_roc(
            plt=plt,
            model_title=model_name,
            outer_stats=outer_stats,
            inner_stats=inner_stats,
            out_base=os.path.join(args.output_dir, "figures", model_key, "performance", "mean_outer_inner_roc"),
        )

        all_model_roc_for_comparison.append(
            {
                "model": model_name,
                "mean_fpr": outer_stats["mean_fpr"],
                "mean_tpr": outer_stats["mean_tpr"],
                "mean_auc": outer_stats["mean_auc"],
                "std_auc": outer_stats["std_auc"],
            }
        )

        # Accuracy and AUC bars with std across folds per repeat
        plot_repeat_train_test_bars(
            plt=plt,
            fold_df=fold_df,
            metric_col_train="train_accuracy",
            metric_col_test="test_accuracy",
            title=f"{model_name}: Training vs Testing Accuracy Across Experiments",
            y_label="Accuracy",
            out_base=os.path.join(args.output_dir, "figures", model_key, "performance", "train_test_accuracy_by_experiment"),
        )

        plot_repeat_train_test_bars(
            plt=plt,
            fold_df=fold_df,
            metric_col_train="train_auc",
            metric_col_test="test_auc",
            title=f"{model_name}: Training vs Testing AUC Across Experiments",
            y_label="AUC",
            out_base=os.path.join(args.output_dir, "figures", model_key, "performance", "train_test_auc_by_experiment"),
        )

        # Confusion matrix + calibration on aggregated outer-test predictions
        y_true_all = pred_df["y_true"].values
        y_pred_all = pred_df["y_pred"].values
        y_prob_all = pred_df["y_prob"].values

        plot_confusion_matrix(
            plt=plt,
            y_true=y_true_all,
            y_pred=y_pred_all,
            title=f"{model_name}: Aggregated Confusion Matrix (Outer Test Folds)",
            out_base=os.path.join(args.output_dir, "figures", model_key, "performance", "confusion_matrix_aggregated"),
        )

        plot_calibration(
            plt=plt,
            y_true=y_true_all,
            y_prob=y_prob_all,
            title=f"{model_name}: Calibration Curve (Outer Test Folds)",
            out_base=os.path.join(args.output_dir, "figures", model_key, "performance", "calibration_curve"),
        )

        # SHAP global plots
        if len(shap_values_all_folds) > 0:
            shap_values_all = np.vstack(shap_values_all_folds)
            shap_data_all = np.vstack(shap_data_all_folds)
            imp_df = save_shap_global_plots(
                plt=plt,
                shap=shap,
                model_key=model_key,
                feature_names=feature_names,
                shap_values_all=shap_values_all,
                X_all=shap_data_all,
                top_features=args.top_features,
                out_dir=args.output_dir,
            )
            top1, top2, top3 = imp_df.iloc[0]["feature"], imp_df.iloc[1]["feature"], imp_df.iloc[2]["feature"]
        else:
            top1, top2, top3 = "NA", "NA", "NA"

        summary_rows.append(
            {
                "model": model_name,
                "mean_test_accuracy": float(fold_df["test_accuracy"].mean()),
                "std_test_accuracy": float(fold_df["test_accuracy"].std()),
                "mean_test_auc": float(fold_df["test_auc"].mean()),
                "std_test_auc": float(fold_df["test_auc"].std()),
                "mean_inner_auc": float(fold_df["inner_auc_oof"].mean()),
                "std_inner_auc": float(fold_df["inner_auc_oof"].std()),
                "n_outer_predictions": int(len(pred_df)),
                "n_lime_rows": int(len(lime_df)),
                "top_feature_1": top1,
                "top_feature_2": top2,
                "top_feature_3": top3,
            }
        )

    # Combined ROC figure for all models
    fig, ax = plt.subplots(figsize=(12, 9))
    for rec in all_model_roc_for_comparison:
        ax.plot(
            rec["mean_fpr"],
            rec["mean_tpr"],
            lw=2.5,
            label=f"{rec['model']} (AUC={rec['mean_auc']:.3f} +/- {rec['std_auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Mean Outer ROC Curves Across Final Models")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    save_figure_all_formats(fig, os.path.join(args.output_dir, "figures", "common", "all_models_mean_outer_roc"))
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, "tables", "final_explainability_summary.csv"), index=False)

    with open(os.path.join(args.output_dir, "logs", "run_log.txt"), "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")
        f.write("\nCompleted final explainability suite.\n")

    print("\nDone. Final outputs saved at:", args.output_dir)
    print("- figures/common: methodology + combined ROC")
    print("- figures/{svm,lr,xgb}: SHAP/LIME/performance")
    print("- tables/{svm,lr,xgb}: fold metrics, predictions, LIME, SHAP importance")
    print("- tables/final_explainability_summary.csv")


if __name__ == "__main__":
    main()
