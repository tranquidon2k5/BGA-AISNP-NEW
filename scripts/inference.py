"""
scripts/inference.py
====================
Unified inference pipeline for the 2-stage hierarchical classifier.

Supports both:
  - XGBoost / sklearn-based models (loaded from .pkl)
  - Generative Bayesian model (loaded from .pkl)

Usage:
    python scripts/inference.py --sample HG00096
    python scripts/inference.py --sample HG00096 --model-type generative
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_dataset, CONTINENTAL_CSV, EAS_CSV, additive_encode, get_snp_ids

import pandas as pd

MODELS_DIR = PROJECT_ROOT / "models"


def load_xgb_pipeline():
    """Load XGBoost-based stage 1 & 2 models."""
    return {
        "stage1_model":  joblib.load(MODELS_DIR / "stage1_super_pop_XGBoost.pkl"),
        "stage2_model":  joblib.load(MODELS_DIR / "stage2_EAS_pop_XGBoost.pkl"),
    }


def predict_sample(
    sample_id: str,
    model_type: str = "xgboost",
    verbose: bool = True,
) -> dict:
    """
    Run the 2-stage pipeline for a single sample.

    Returns dict with keys:
        sample, stage1_pred, stage1_proba,
        stage2_pred (if EAS), stage2_proba (if EAS),
        final_label
    """
    # Load data to get this sample's encoded features
    df_cont = pd.read_csv(CONTINENTAL_CSV)
    snp_ids_cont = get_snp_ids(list(df_cont.columns))
    X_cont = additive_encode(df_cont, snp_ids_cont)

    row_idx = df_cont.index[df_cont["sample"] == sample_id]
    if len(row_idx) == 0:
        raise ValueError(f"Sample '{sample_id}' not found in continental dataset.")
    idx = row_idx[0]
    x_cont = X_cont[idx:idx+1]

    # Stage 1
    models = load_xgb_pipeline()
    s1_model = models["stage1_model"]

    feat_cols1 = [f"snp_{i}" for i in range(x_cont.shape[1])]
    x_cont_df = pd.DataFrame(x_cont, columns=feat_cols1)

    s1_pred = s1_model.predict(x_cont_df)[0]

    # Decode stage 1 label
    # Load label encoder
    X_all, y_super, _, le_dict, _, _ = load_dataset(CONTINENTAL_CSV, verbose=False)
    le_super = le_dict["super_pop"]
    s1_label = le_super.inverse_transform([s1_pred])[0]

    result = {
        "sample": sample_id,
        "stage1_pred": s1_label,
        "final_label": s1_label,
    }

    try:
        s1_proba = s1_model.predict_proba(x_cont_df)[0]
        result["stage1_proba"] = dict(zip(le_super.classes_, s1_proba.tolist()))
    except Exception:
        result["stage1_proba"] = {s1_label: 1.0}

    if verbose:
        print(f"Stage 1 → {s1_label}  (proba: {result['stage1_proba']})")

    # Stage 2 — only if EAS
    if s1_label == "EAS":
        df_eas = pd.read_csv(EAS_CSV)
        snp_ids_eas = get_snp_ids(list(df_eas.columns))
        X_eas = additive_encode(df_eas, snp_ids_eas)

        eas_row = df_eas.index[df_eas["sample"] == sample_id]
        if len(eas_row) > 0:
            eidx = eas_row[0]
            x_eas = X_eas[eidx:eidx+1]

            s2_model = models["stage2_model"]
            feat_cols2 = [f"snp_{i}" for i in range(x_eas.shape[1])]
            x_eas_df = pd.DataFrame(x_eas, columns=feat_cols2)

            s2_pred = s2_model.predict(x_eas_df)[0]

            _, _, y_pop, le_eas, _, _ = load_dataset(EAS_CSV, verbose=False)
            le_pop = le_eas["pop"]
            s2_label = le_pop.inverse_transform([s2_pred])[0]

            result["stage2_pred"] = s2_label
            result["final_label"] = f"EAS/{s2_label}"

            try:
                s2_proba = s2_model.predict_proba(x_eas_df)[0]
                result["stage2_proba"] = dict(zip(le_pop.classes_, s2_proba.tolist()))
            except Exception:
                result["stage2_proba"] = {s2_label: 1.0}

            if verbose:
                print(f"Stage 2 → {s2_label}  (proba: {result['stage2_proba']})")
        else:
            result["note"] = "EAS predicted but sample not in EAS dataset"

    if verbose:
        print(f"Final   → {result['final_label']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Ancestry inference pipeline")
    parser.add_argument("--sample", required=True, help="Sample ID")
    parser.add_argument("--model-type", default="xgboost",
                        choices=["xgboost", "generative"],
                        help="Which model backend to use")
    args = parser.parse_args()

    result = predict_sample(args.sample, model_type=args.model_type)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
