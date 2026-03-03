"""
src/evaluation.py
=================
Unified evaluation utilities: metrics computation, confusion matrices,
bar chart comparisons, feature importance plots, and Excel export.

Merges functionality from:
  - scripts/evaluate_models.py
  - scripts/export_metrics_excel.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation (from train_all results)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_results(
    train_results: list[dict],
    label_encoder,
    dataset_name: str,
    snp_ids: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all trained models, save plots, return summary DataFrame.
    """
    rows = []
    target_name = train_results[0]["target"]

    for r in train_results:
        model   = r["fitted_model"]
        X_test  = r["X_test"]
        y_test  = r["y_test"]
        name    = r["model_name"]

        y_pred  = model.predict(X_test)

        acc      = accuracy_score(y_test, y_pred)
        bal_acc  = balanced_accuracy_score(y_test, y_pred)
        f1_mac   = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wt    = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        mcc      = matthews_corrcoef(y_test, y_pred)

        # ROC-AUC
        try:
            y_prob  = model.predict_proba(X_test)
            n_cls   = r["n_classes"]
            roc_auc = (roc_auc_score(y_test, y_prob[:, 1]) if n_cls == 2
                       else roc_auc_score(y_test, y_prob,
                                          multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = float("nan")

        row = {
            "dataset":              dataset_name,
            "target":               target_name,
            "model":                name,
            "test_accuracy":        round(acc,     4),
            "test_balanced_acc":    round(bal_acc, 4),
            "test_mcc":             round(mcc,     4),
            "test_f1_macro":        round(f1_mac,  4),
            "test_f1_weighted":     round(f1_wt,   4),
            "test_roc_auc_macro":   round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
            "cv_accuracy_mean":     round(r["cv_accuracy_mean"],    4),
            "cv_accuracy_std":      round(r["cv_accuracy_std"],     4),
            "cv_f1_mean":           round(r["cv_f1_mean"],          4),
            "cv_balanced_acc_mean": round(r["cv_balanced_acc_mean"],4),
            "fit_time_sec":         round(r["fit_time_sec"],        2),
            "best_params":          str(r.get("best_params", {})),
        }
        rows.append(row)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Model: {name}  |  Dataset: {dataset_name}  |  Target: {target_name}")
            print(f"  Test Accuracy       : {acc:.4f}")
            print(f"  Balanced Accuracy   : {bal_acc:.4f}")
            print(f"  F1 Macro            : {f1_mac:.4f}")
            print(f"  MCC                 : {mcc:.4f}")
            auc_str = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A"
            print(f"  ROC-AUC (macro ovr) : {auc_str}")
            print(classification_report(
                y_test, y_pred,
                target_names=label_encoder.classes_,
                zero_division=0,
            ))

        # ── Confusion matrix ─────────────────────────────────────────────
        plot_confusion_matrix(
            y_test, y_pred,
            class_names=label_encoder.classes_,
            title=f"{name} | {dataset_name} | {target_name}",
            save_path=RESULTS_DIR / f"confusion_{dataset_name}_{target_name}_{name}.png",
        )

        # ── Feature importance ────────────────────────────────────────────
        if snp_ids and name in ("RandomForest", "GradientBoosting", "XGBoost"):
            plot_feature_importance(
                model, snp_ids, name, dataset_name, target_name,
            )

    df_summary = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    csv_path = RESULTS_DIR / f"model_comparison_{dataset_name}_{target_name}.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n[evaluate] Saved summary → {csv_path}")

    # Bar charts
    for metric, label in [
        ("test_accuracy",    "Accuracy"),
        ("test_f1_macro",    "F1 Macro"),
        ("cv_accuracy_mean", "CV Accuracy (mean)"),
    ]:
        plot_bar_comparison(df_summary, metric, dataset_name, target_name, label)

    return df_summary


# ─────────────────────────────────────────────────────────────────────────────
# Per-label metrics (for Excel export)
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_label_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob,
    classes: list[str],
    stage: str,
) -> list[dict]:
    """Per-class precision, recall, F1, MCC, AUC."""
    n_classes = len(classes)
    rows = []
    for i, cls_name in enumerate(classes):
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)

        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        mcc  = matthews_corrcoef(y_true_bin, y_pred_bin)

        if y_prob is not None:
            try:
                auc = roc_auc_score(y_true_bin, y_prob[:, i])
            except Exception:
                auc = float("nan")
        else:
            auc = float("nan")

        rows.append({
            "stage":     stage,
            "model":     model_name,
            "label":     cls_name,
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1_score":  round(f1,   4),
            "mcc":       round(mcc,  4),
            "auc_roc":   round(auc,  4) if not np.isnan(auc) else "N/A",
            "support":   int((y_true == i).sum()),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, subtitle in zip(
        axes, [cm, cm_norm], ["d", ".2f"], ["Counts", "Normalised (row %)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    cmap="Blues", linewidths=0.5)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.set_title(subtitle,     fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_bar_comparison(df, metric, dataset, target, metric_label):
    fig, ax = plt.subplots(figsize=(9, 5))
    models = df["model"].tolist()
    values = df[metric].tolist()
    colors = sns.color_palette("viridis", len(models))
    bars = ax.barh(models, values, color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlim(0, min(1.05, max(values) * 1.12))
    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_title(f"{metric_label} — {dataset} | target: {target}", fontsize=13)
    ax.invert_yaxis()
    fig.tight_layout()
    save_path = RESULTS_DIR / f"comparison_{metric}_{dataset}_{target}.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, snp_ids, model_name, dataset, target, top_n=20):
    clf = model
    if hasattr(model, "named_steps"):
        clf = list(model.named_steps.values())[-1]

    if not hasattr(clf, "feature_importances_"):
        return

    importances = clf.feature_importances_
    if len(importances) != len(snp_ids):
        return

    idx = np.argsort(importances)[::-1][:top_n]
    top_snps = [snp_ids[i] for i in idx]
    top_imps = importances[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top_imps, y=top_snps, orient="h", ax=ax,
                hue=top_snps, palette="magma_r", legend=False)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Top-{top_n} Feature Importances\n"
                 f"{model_name} | {dataset} | {target}", fontsize=12)
    fig.tight_layout()
    save_path = RESULTS_DIR / f"feature_importance_{dataset}_{target}_{model_name}.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Saved feature importance → {save_path}")
