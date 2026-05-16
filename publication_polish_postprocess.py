"""
Publication polish post-processing for final explainability outputs.

This script reads existing outputs from final_explainability_suite.py and adds:
- Detailed methodology flow diagram with dataset/split numbers
- Extra performance plots (boxplots, train-test gap, inner-vs-outer AUC)
- Mean Precision-Recall curves (per model + combined)
- Pairwise statistical comparison tables
- Curated LIME panels (TP/TN/FP/FN) per model for manuscript main text
- Feature alias mapping table for cleaner figure labels

Run (Colab):
python publication_polish_postprocess.py \
  --base-dir final_explainability_outputs \
  --output-dir publication_polish_outputs
"""

from __future__ import annotations

import argparse
import itertools
import os
import textwrap
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, precision_recall_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publication polish post-processing")
    parser.add_argument("--base-dir", default="final_explainability_outputs", help="Base directory of final suite outputs")
    parser.add_argument("--output-dir", default="publication_polish_outputs", help="Directory for polished artifacts")
    parser.add_argument("--top-features", type=int, default=20, help="Top features per model for alias mapping")
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
            "figure.figsize": (12, 8),
        }
    )


def save_fig_all(fig: Any, base_path: str) -> None:
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig.savefig(base_path + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(base_path + ".pdf", bbox_inches="tight")
    fig.savefig(base_path + ".svg", bbox_inches="tight")


def ensure_dirs(out_dir: str) -> None:
    for p in [
        "tables",
        "figures/common",
        "figures/performance",
        "figures/lime_panels",
        "figures/features",
        "logs",
    ]:
        os.makedirs(os.path.join(out_dir, p), exist_ok=True)


def load_model_key_to_name(summary_df: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    for name in summary_df["model"].tolist():
        if "SVM" in name:
            mapping["svm"] = name
        elif "Logistic" in name:
            mapping["lr"] = name
        elif "XGBoost" in name:
            mapping["xgb"] = name
    return mapping


def metric_summary(series: pd.Series) -> dict[str, float]:
    arr = series.values.astype(float)
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def draw_detailed_flow_diagram(plt: Any, FancyBboxPatch: Any, out_dir: str, summary_df: pd.DataFrame, log_lines: list[str]) -> None:
    stats = {k: v for k, v in [line.split(":", 1) for line in log_lines if ":" in line]}
    samples = stats.get("Samples", "NA").strip()
    features = stats.get("Features", "NA").strip()
    subjects = stats.get("Subjects", "NA").strip()
    outer = stats.get("Outer folds", "NA").strip()

    model_lines = []
    for _, row in summary_df.iterrows():
        model_lines.append(f"{row['model']}: Acc {row['mean_test_accuracy']:.3f}, AUC {row['mean_test_auc']:.3f}")

    fig, ax = plt.subplots(figsize=(10, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 28)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, txt: str, color: str) -> tuple[float, float, float, float]:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#3d3d3d",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10.5)
        return (x, y, w, h)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#222222"),
        )

    b1 = box(2.2, 25.8, 5.6, 1.3, f"Dataset\nSamples={samples}, Subjects={subjects}, Features={features}", "#dbe9f6")
    b2 = box(2.2, 24.0, 5.6, 1.2, "Preprocessing\n(signal cleaning, feature engineering, missing handling)", "#dbe9f6")
    b3 = box(2.2, 22.3, 5.6, 1.2, "Feature matrix + labels + SubjectID grouping", "#dbe9f6")
    b4 = box(1.7, 20.4, 6.6, 1.5, f"Outer grouped CV\n{outer}\n(no subject overlap)", "#e9def7")
    b5 = box(0.5, 18.5, 3.2, 1.1, "Outer train split", "#efe7fb")
    b6 = box(6.3, 18.5, 3.2, 1.1, "Outer test split", "#efe7fb")
    b7 = box(0.5, 16.8, 3.2, 1.1, "Inner grouped CV\n(3-fold)", "#efe7fb")
    b8 = box(0.5, 15.1, 3.2, 1.1, "Hyperparameter tuning", "#efe7fb")
    b9 = box(3.9, 15.1, 2.2, 1.1, "Best model", "#efe7fb")
    b10 = box(6.3, 16.8, 3.2, 1.1, "Final evaluation\non outer test", "#efe7fb")
    b11 = box(1.6, 13.0, 6.8, 1.6, "Explainability\nSHAP (global + local) and LIME (TP/TN/FP/FN)", "#deefd9")
    b12 = box(1.0, 9.8, 8.0, 2.6, "\n".join(model_lines), "#f9f1db")
    b13 = box(1.0, 7.8, 8.0, 1.4, "Publication outputs\nROC/PR, calibration, confusion matrix, statistical tests", "#f7d9e8")

    arrow(5.0, 25.8, 5.0, 25.2)
    arrow(5.0, 24.0, 5.0, 23.5)
    arrow(5.0, 22.3, 5.0, 21.9)
    arrow(5.0, 20.4, 2.1, 19.6)
    arrow(5.0, 20.4, 7.9, 19.6)
    arrow(2.1, 18.5, 2.1, 17.9)
    arrow(2.1, 16.8, 2.1, 16.2)
    arrow(2.1, 15.1, 5.0, 15.1)
    arrow(7.9, 18.5, 7.9, 17.9)
    arrow(7.9, 16.8, 5.0, 14.6)
    arrow(5.0, 13.0, 5.0, 12.4)
    arrow(5.0, 9.8, 5.0, 9.2)

    ax.set_title("Detailed Methodology Flow (with dataset and split numbers)", fontsize=20, pad=16)
    save_fig_all(fig, os.path.join(out_dir, "figures", "common", "detailed_methodology_flow"))
    plt.close(fig)


def plot_metric_boxplots(plt: Any, all_fold_df: pd.DataFrame, out_dir: str) -> None:
    metrics = ["test_accuracy", "test_auc", "test_f1"]
    titles = ["Test Accuracy", "Test AUC", "Test F1"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric, title in zip(axes, metrics, titles):
        data = [all_fold_df.loc[all_fold_df["model"] == m, metric].values for m in sorted(all_fold_df["model"].unique())]
        ax.boxplot(data, labels=sorted(all_fold_df["model"].unique()), showmeans=True)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Distribution of Test Metrics Across Outer Folds", fontsize=18)
    fig.tight_layout()
    save_fig_all(fig, os.path.join(out_dir, "figures", "performance", "test_metric_boxplots"))
    plt.close(fig)


def plot_train_test_gap(plt: Any, summary_gap_df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(summary_gap_df))
    axes[0].bar(x, summary_gap_df["acc_gap_mean"], yerr=summary_gap_df["acc_gap_std"], capsize=5, color="#4c78a8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_gap_df["model"], rotation=20)
    axes[0].set_title("Train-Test Accuracy Gap")
    axes[0].set_ylabel("Train - Test")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, summary_gap_df["auc_gap_mean"], yerr=summary_gap_df["auc_gap_std"], capsize=5, color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary_gap_df["model"], rotation=20)
    axes[1].set_title("Train-Test AUC Gap")
    axes[1].set_ylabel("Train - Test")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Overfitting Gap Summary (with std error bars)", fontsize=18)
    fig.tight_layout()
    save_fig_all(fig, os.path.join(out_dir, "figures", "performance", "train_test_gap_summary"))
    plt.close(fig)


def plot_inner_outer_auc_bars(plt: Any, summary_df: pd.DataFrame, out_dir: str) -> None:
    x = np.arange(len(summary_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(
        x - width / 2,
        summary_df["mean_inner_auc"],
        width,
        yerr=summary_df["std_inner_auc"],
        capsize=5,
        label="Mean Inner AUC",
        color="#2ca02c",
    )
    ax.bar(
        x + width / 2,
        summary_df["mean_test_auc"],
        width,
        yerr=summary_df["std_test_auc"],
        capsize=5,
        label="Mean Outer AUC",
        color="#1f77b4",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"], rotation=20)
    ax.set_ylabel("AUC")
    ax.set_title("Mean Inner vs Outer AUC-ROC by Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save_fig_all(fig, os.path.join(out_dir, "figures", "performance", "inner_vs_outer_auc_bar"))
    plt.close(fig)


def compute_mean_pr(pred_df: pd.DataFrame, n_grid: int = 400) -> dict[str, Any]:
    recall_grid = np.linspace(0.0, 1.0, n_grid)
    precision_curves = []
    aps = []

    for fold in sorted(pred_df["global_fold"].unique()):
        d = pred_df[pred_df["global_fold"] == fold]
        y_true = d["y_true"].values
        y_prob = d["y_prob"].values
        if len(np.unique(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]

        interp_precision = np.interp(recall_grid, recall_sorted, precision_sorted)
        precision_curves.append(interp_precision)
        aps.append(ap)

    curves = np.array(precision_curves)
    mean_precision = curves.mean(axis=0)
    std_precision = curves.std(axis=0)

    return {
        "recall_grid": recall_grid,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_ap": float(np.mean(aps)),
        "std_ap": float(np.std(aps)),
    }


def plot_pr_curves(plt: Any, model_pr_stats: dict[str, dict[str, Any]], out_dir: str) -> None:
    # Combined
    fig, ax = plt.subplots(figsize=(11, 8))
    for model_name, stats in model_pr_stats.items():
        ax.plot(
            stats["recall_grid"],
            stats["mean_precision"],
            lw=2.5,
            label=f"{model_name} (AP={stats['mean_ap']:.3f} +/- {stats['std_ap']:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Mean Precision-Recall Curves (Outer Folds)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    save_fig_all(fig, os.path.join(out_dir, "figures", "performance", "combined_mean_pr_curves"))
    plt.close(fig)

    # Per model
    for model_name, stats in model_pr_stats.items():
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(stats["recall_grid"], stats["mean_precision"], color="#1f77b4", lw=2.5)
        ax.fill_between(
            stats["recall_grid"],
            np.maximum(stats["mean_precision"] - stats["std_precision"], 0),
            np.minimum(stats["mean_precision"] + stats["std_precision"], 1),
            color="#1f77b4",
            alpha=0.15,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{model_name}: Mean PR Curve (AP={stats['mean_ap']:.3f} +/- {stats['std_ap']:.3f})")
        ax.grid(alpha=0.25)
        safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        save_fig_all(fig, os.path.join(out_dir, "figures", "performance", f"mean_pr_{safe}"))
        plt.close(fig)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    gt = 0
    lt = 0
    for a in x:
        gt += int(np.sum(a > y))
        lt += int(np.sum(a < y))
    n = len(x) * len(y)
    if n == 0:
        return 0.0
    return (gt - lt) / n


def wilcoxon_or_sign_test(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(p), "wilcoxon"
    except Exception:
        diff = x - y
        diff = diff[diff != 0]
        if len(diff) == 0:
            return 1.0, "sign_test"
        n_pos = int(np.sum(diff > 0))
        n = len(diff)
        k = min(n_pos, n - n_pos)
        p = 2 * sum(
            [
                (np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))) * (0.5 ** n)
                for i in range(k + 1)
            ]
        )
        return float(min(1.0, p)), "sign_test"


def build_stat_tables(all_fold_df: pd.DataFrame, out_dir: str) -> None:
    models = sorted(all_fold_df["model"].unique())
    metrics = ["test_auc", "test_accuracy", "test_f1"]

    rows = []
    for metric in metrics:
        for m1, m2 in itertools.combinations(models, 2):
            d1 = all_fold_df[all_fold_df["model"] == m1].sort_values("global_fold")
            d2 = all_fold_df[all_fold_df["model"] == m2].sort_values("global_fold")
            merged = d1[["global_fold", metric]].merge(
                d2[["global_fold", metric]], on="global_fold", suffixes=("_m1", "_m2")
            )
            x = merged[f"{metric}_m1"].values
            y = merged[f"{metric}_m2"].values

            p_val, test_name = wilcoxon_or_sign_test(x, y)
            delta = cliffs_delta(x, y)

            rows.append(
                {
                    "metric": metric,
                    "model_1": m1,
                    "model_2": m2,
                    "mean_1": float(np.mean(x)),
                    "mean_2": float(np.mean(y)),
                    "mean_diff_1_minus_2": float(np.mean(x - y)),
                    "p_value": p_val,
                    "effect_size_cliffs_delta": delta,
                    "test_used": test_name,
                }
            )

    stat_df = pd.DataFrame(rows)
    stat_df.to_csv(os.path.join(out_dir, "tables", "pairwise_statistical_comparisons.csv"), index=False)


def curate_lime_cases(lime_df: pd.DataFrame) -> pd.DataFrame:
    case_cols = [
        "global_fold",
        "repeat",
        "fold",
        "case_type",
        "test_row_index",
        "SubjectID",
        "y_true",
        "y_pred",
        "y_prob",
    ]
    case_df = lime_df[case_cols].drop_duplicates()

    selected_rows = []
    for case_type in ["TP", "TN", "FP", "FN"]:
        d = case_df[case_df["case_type"] == case_type].copy()
        if d.empty:
            continue
        if case_type == "TP":
            pick = d.sort_values("y_prob", ascending=False).iloc[0]
        elif case_type == "TN":
            pick = d.sort_values("y_prob", ascending=True).iloc[0]
        elif case_type == "FP":
            pick = d.sort_values("y_prob", ascending=False).iloc[0]
        else:  # FN
            pick = d.sort_values("y_prob", ascending=True).iloc[0]
        selected_rows.append(pick)

    if len(selected_rows) == 0:
        return pd.DataFrame(columns=case_cols)
    return pd.DataFrame(selected_rows)


def build_lime_panel(plt: Any, base_dir: str, out_dir: str, model_key: str, model_name: str, curated_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    layout = [(0, 0, "TP"), (0, 1, "TN"), (1, 0, "FP"), (1, 1, "FN")]

    for r, c, case_type in layout:
        ax = axes[r, c]
        row = curated_df[curated_df["case_type"] == case_type]
        if row.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{case_type}: not available", ha="center", va="center")
            continue

        row = row.iloc[0]
        img_name = f"fold{int(row['global_fold']):02d}_{case_type}_row{int(row['test_row_index'])}.png"
        img_path = os.path.join(base_dir, "figures", model_key, "lime", img_name)

        if os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(
                f"{case_type} | Fold {int(row['global_fold'])} | "
                f"P(PD)={row['y_prob']:.3f} | Subj {row['SubjectID']}"
            )
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Missing image:\n{img_name}", ha="center", va="center")

    fig.suptitle(f"{model_name}: Curated LIME Cases (TP/TN/FP/FN)", fontsize=18)
    fig.tight_layout()
    save_fig_all(fig, os.path.join(out_dir, "figures", "lime_panels", f"{model_key}_curated_lime_panel"))
    plt.close(fig)


def build_feature_alias_mapping(base_dir: str, out_dir: str, model_key_to_name: dict[str, str], top_features: int) -> pd.DataFrame:
    pool = []
    for mk in model_key_to_name:
        p = os.path.join(base_dir, "tables", mk, "shap_global_importance.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p).head(top_features)
        d["model_key"] = mk
        pool.append(d)
    if len(pool) == 0:
        return pd.DataFrame(columns=["alias", "feature"])

    allf = pd.concat(pool, ignore_index=True)
    uniq = allf[["feature"]].drop_duplicates().reset_index(drop=True)
    uniq["alias"] = [f"F{idx:02d}" for idx in range(1, len(uniq) + 1)]
    uniq = uniq[["alias", "feature"]]
    uniq.to_csv(os.path.join(out_dir, "tables", "feature_alias_mapping.csv"), index=False)

    # Also save per-model top feature table with aliases
    for mk, model_name in model_key_to_name.items():
        p = os.path.join(base_dir, "tables", mk, "shap_global_importance.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p).head(top_features)
        d = d.merge(uniq, on="feature", how="left")
        d["model"] = model_name
        d = d[["model", "alias", "feature", "mean_abs_shap"]]
        d.to_csv(os.path.join(out_dir, "tables", f"{mk}_top_features_with_alias.csv"), index=False)

    return uniq


def plot_feature_alias_bars(plt: Any, base_dir: str, out_dir: str, model_key_to_name: dict[str, str], alias_df: pd.DataFrame, top_n: int = 12) -> None:
    alias_map = dict(zip(alias_df["feature"], alias_df["alias"]))

    for mk, model_name in model_key_to_name.items():
        p = os.path.join(base_dir, "tables", mk, "shap_global_importance.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p).head(top_n).copy()
        d["alias"] = d["feature"].map(alias_map)
        d["label"] = d["alias"] + " - " + d["feature"].apply(lambda s: "\n".join(textwrap.wrap(str(s), 20)))

        fig, ax = plt.subplots(figsize=(13, 9))
        ax.barh(d["label"][::-1], d["mean_abs_shap"][::-1], color="#4c78a8")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{model_name}: Top SHAP Features with Alias Labels")
        ax.grid(axis="x", alpha=0.2)
        save_fig_all(fig, os.path.join(out_dir, "figures", "features", f"{mk}_alias_shap_bar"))
        plt.close(fig)


def write_report_md(out_dir: str, summary_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Publication Polish Add-ons")
    lines.append("")
    lines.append("This folder adds manuscript-focused artifacts on top of `final_explainability_outputs`.")
    lines.append("")
    lines.append("## Added Figures")
    lines.append("- Detailed methodology flow diagram with dataset and CV numbers")
    lines.append("- Test metric boxplots across outer folds")
    lines.append("- Train-test gap summary with std error bars")
    lines.append("- Mean inner vs outer AUC bar chart")
    lines.append("- Mean PR curves (combined and per model)")
    lines.append("- Curated LIME panel (TP/TN/FP/FN) per model")
    lines.append("- SHAP top-feature bar charts with alias labels")
    lines.append("")
    lines.append("## Added Tables")
    lines.append("- extended_metric_summary.csv")
    lines.append("- pairwise_statistical_comparisons.csv")
    lines.append("- feature_alias_mapping.csv")
    lines.append("- curated_lime_cases.csv")
    lines.append("")
    lines.append("## Current Core Performance")
    for _, r in summary_df.iterrows():
        lines.append(f"- {r['model']}: test AUC {r['mean_test_auc']:.3f} +/- {r['std_test_auc']:.3f}, "
                     f"test accuracy {r['mean_test_accuracy']:.3f} +/- {r['std_test_accuracy']:.3f}")
    lines.append("")
    with open(os.path.join(out_dir, "publication_polish_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    set_style(plt)
    ensure_dirs(args.output_dir)

    base = args.base_dir
    summary_path = os.path.join(base, "tables", "final_explainability_summary.csv")
    log_path = os.path.join(base, "logs", "run_log.txt")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    log_lines = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log_lines = [line.strip() for line in f if line.strip()]

    model_key_to_name = load_model_key_to_name(summary_df)

    # Load per-model fold and prediction tables
    fold_tables = []
    pred_tables = {}
    lime_tables = {}

    for mk, model_name in model_key_to_name.items():
        fpath = os.path.join(base, "tables", mk, "fold_metrics.csv")
        ppath = os.path.join(base, "tables", mk, "fold_test_predictions.csv")
        lpath = os.path.join(base, "tables", mk, "lime_case_reports.csv")

        if os.path.exists(fpath):
            d = pd.read_csv(fpath)
            d["model"] = model_name
            fold_tables.append(d)
        if os.path.exists(ppath):
            pred_tables[model_name] = pd.read_csv(ppath)
        if os.path.exists(lpath):
            lime_tables[model_name] = pd.read_csv(lpath)

    all_fold_df = pd.concat(fold_tables, ignore_index=True)

    # Extended metric summary table
    ext_rows = []
    gap_rows = []
    for model_name in sorted(all_fold_df["model"].unique()):
        d = all_fold_df[all_fold_df["model"] == model_name]
        for metric in ["test_accuracy", "test_auc", "test_f1", "inner_auc_oof"]:
            s = metric_summary(d[metric])
            ext_rows.append({"model": model_name, "metric": metric, **s})

        acc_gap = d["train_accuracy"].values - d["test_accuracy"].values
        auc_gap = d["train_auc"].values - d["test_auc"].values
        gap_rows.append(
            {
                "model": model_name,
                "acc_gap_mean": float(np.mean(acc_gap)),
                "acc_gap_std": float(np.std(acc_gap, ddof=1)),
                "auc_gap_mean": float(np.mean(auc_gap)),
                "auc_gap_std": float(np.std(auc_gap, ddof=1)),
            }
        )

    ext_df = pd.DataFrame(ext_rows)
    gap_df = pd.DataFrame(gap_rows)
    ext_df.to_csv(os.path.join(args.output_dir, "tables", "extended_metric_summary.csv"), index=False)
    gap_df.to_csv(os.path.join(args.output_dir, "tables", "train_test_gap_summary.csv"), index=False)

    # Draw detailed flow
    draw_detailed_flow_diagram(plt, FancyBboxPatch, args.output_dir, summary_df, log_lines)

    # Performance visuals
    plot_metric_boxplots(plt, all_fold_df, args.output_dir)
    plot_train_test_gap(plt, gap_df, args.output_dir)
    plot_inner_outer_auc_bars(plt, summary_df, args.output_dir)

    # PR curves
    pr_stats = {}
    for model_name, d in pred_tables.items():
        pr_stats[model_name] = compute_mean_pr(d)
    plot_pr_curves(plt, pr_stats, args.output_dir)

    # Statistical comparisons
    build_stat_tables(all_fold_df, args.output_dir)

    # Curated LIME panel + table
    curated_rows = []
    for mk, model_name in model_key_to_name.items():
        if model_name not in lime_tables:
            continue
        cdf = curate_lime_cases(lime_tables[model_name])
        if len(cdf) > 0:
            cdf["model"] = model_name
            curated_rows.append(cdf)
        build_lime_panel(plt, base, args.output_dir, mk, model_name, cdf)

    if len(curated_rows) > 0:
        curated_all = pd.concat(curated_rows, ignore_index=True)
    else:
        curated_all = pd.DataFrame()
    curated_all.to_csv(os.path.join(args.output_dir, "tables", "curated_lime_cases.csv"), index=False)

    # Feature alias mapping + bars
    alias_df = build_feature_alias_mapping(base, args.output_dir, model_key_to_name, args.top_features)
    if len(alias_df) > 0:
        plot_feature_alias_bars(plt, base, args.output_dir, model_key_to_name, alias_df, top_n=min(12, args.top_features))

    # Final report
    write_report_md(args.output_dir, summary_df)

    with open(os.path.join(args.output_dir, "logs", "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("Publication polish post-processing completed successfully.\n")
        f.write(f"Base dir: {args.base_dir}\n")
        f.write(f"Output dir: {args.output_dir}\n")

    print("Done. Publication polish artifacts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
