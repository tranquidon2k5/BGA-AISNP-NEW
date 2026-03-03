"""
scripts/export_excel.py
=======================
Re-train both stages and export comprehensive per-label + summary metrics
to results/metrics_per_label.xlsx.

Usage:
    python scripts/export_excel.py
"""

from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_continental, load_eas, load_dataset, EAS_CSV
from src.training      import train_all
from src.evaluation    import evaluate_results, compute_per_label_metrics

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_stage_for_export(X, y, le, snp_ids, stage_tag, stage_title):
    """Train, evaluate, collect per-label + summary metrics."""
    classes = le.classes_

    print(f"\n{'#'*70}")
    print(f"  {stage_title}")
    print(f"  Classes ({len(classes)}): {list(classes)}")
    print(f"{'#'*70}")

    results = train_all(X, y, stage_tag)

    per_label_rows = []
    summary_rows   = []

    for r in results:
        model      = r["fitted_model"]
        X_test     = r["X_test"]
        y_test     = r["y_test"]
        model_name = r["model_name"]

        y_pred = model.predict(X_test)
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            y_prob = None

        per_label_rows.extend(
            compute_per_label_metrics(
                model_name, y_test, y_pred, y_prob, classes, stage_tag
            )
        )

        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, matthews_corrcoef,
            f1_score, roc_auc_score,
        )
        acc     = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        mcc     = matthews_corrcoef(y_test, y_pred)
        f1_mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wt   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        try:
            if y_prob is not None:
                n_cls = r["n_classes"]
                auc = (roc_auc_score(y_test, y_prob[:, 1]) if n_cls == 2
                       else roc_auc_score(y_test, y_prob,
                                          multi_class="ovr", average="macro"))
            else:
                auc = float("nan")
        except Exception:
            auc = float("nan")

        summary_rows.append({
            "stage":              stage_tag,
            "model":              model_name,
            "test_accuracy":      round(acc,     4),
            "test_balanced_acc":  round(bal_acc, 4),
            "test_mcc":           round(mcc,     4),
            "test_f1_macro":      round(f1_mac,  4),
            "test_f1_weighted":   round(f1_wt,   4),
            "test_auc_roc_macro": round(auc, 4) if not np.isnan(auc) else "N/A",
            "cv_accuracy_mean":   round(r["cv_accuracy_mean"],    4),
            "cv_f1_mean":         round(r["cv_f1_mean"],          4),
            "cv_balanced_acc":    round(r["cv_balanced_acc_mean"],4),
            "fit_time_sec":       round(r["fit_time_sec"],        2),
            "best_params":        str(r.get("best_params", {})),
        })

    return pd.DataFrame(per_label_rows), pd.DataFrame(summary_rows).sort_values(
        "test_accuracy", ascending=False)


def export_to_excel(s1_per, s1_sum, s2_per, s2_sum, out_path):
    """Write all sheets to Excel."""
    all_summary = pd.concat([s1_sum, s2_sum], ignore_index=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        s1_per.to_excel(writer, sheet_name="Stage1_per_label", index=False)
        s1_sum.to_excel(writer, sheet_name="Stage1_summary",   index=False)
        s2_per.to_excel(writer, sheet_name="Stage2_per_label", index=False)
        s2_sum.to_excel(writer, sheet_name="Stage2_summary",   index=False)
        all_summary.to_excel(writer, sheet_name="ALL_summary", index=False)

        # Pivot tables
        for stage_df, prefix in [(s1_per, "S1"), (s2_per, "S2")]:
            for metric in ["f1_score", "mcc", "auc_roc", "recall"]:
                if metric in stage_df.columns:
                    pivot = stage_df.pivot_table(
                        index="label", columns="model",
                        values=metric, aggfunc="first",
                    ).reset_index()
                    pivot.to_excel(writer,
                                   sheet_name=f"{prefix}_pivot_{metric}",
                                   index=False)

    print(f"\n[export] Excel saved → {out_path}")


def main():
    # Stage 1
    X1, y_super, _, le1, snp_ids1, _ = load_continental()
    s1_per, s1_sum = run_stage_for_export(
        X1, y_super, le1["super_pop"], snp_ids1,
        "stage1_super_pop",
        "STAGE 1 · Continental super-population",
    )

    # Stage 2
    _, _, y_pop, le2, snp_ids2, _ = load_eas()
    X2, _, _, _, _, _ = load_dataset(EAS_CSV, verbose=False)
    s2_per, s2_sum = run_stage_for_export(
        X2, y_pop, le2["pop"], snp_ids2,
        "stage2_EAS_pop",
        "STAGE 2 · EAS sub-population",
    )

    out_path = RESULTS_DIR / "metrics_per_label.xlsx"
    export_to_excel(s1_per, s1_sum, s2_per, s2_sum, out_path)


if __name__ == "__main__":
    main()
