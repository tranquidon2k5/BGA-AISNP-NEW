"""
scripts/run_pipeline.py
==========================
Main entry point — 2-Stage Hierarchical Ancestry Prediction Pipeline.

  Stage 1: Continental super_pop (AFR | AMR | EAS | EUR | SAS)
  Stage 2: East-Asian sub-pop   (CDX | CHB | CHS | JPT | KHV)

Usage:
    python scripts/run_pipeline.py                          # all models, both stages
    python scripts/run_pipeline.py --stage 1                # stage 1 only
    python scripts/run_pipeline.py --models XGBoost         # 1 model only
    python scripts/run_pipeline.py --models XGBoost LightGBM  # 2 models
    python scripts/run_pipeline.py --list-models            # show available models

Outputs → results/
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_continental, load_eas, load_dataset, EAS_CSV
from src.training      import train_all
from src.evaluation    import evaluate_results
from src.model_registry import list_available_models, ALL_MODEL_NAMES

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_stage(X, y, label_encoder, snp_ids, stage_tag, stage_title,
              model_names=None, verbose=True):
    """Train models, evaluate, return summary DataFrame."""
    n_classes = len(np.unique(y))
    print(f"\n{'#' * 70}")
    print(f"  {stage_title}")
    print(f"  Classes ({n_classes}): {list(label_encoder.classes_)}")
    if model_names:
        print(f"  Models: {model_names}")
    else:
        print(f"  Models: ALL ({len(ALL_MODEL_NAMES)})")
    print(f"{'#' * 70}")

    train_results = train_all(X, y, target_name=stage_tag,
                              model_names=model_names, verbose=verbose)

    df_summary = evaluate_results(
        train_results,
        label_encoder=label_encoder,
        dataset_name=stage_tag,
        snp_ids=snp_ids,
        verbose=verbose,
    )

    print(f"\n{'─' * 60}")
    print(f"Ranking — {stage_title}")
    print(df_summary[["model", "test_accuracy", "test_f1_macro",
                       "test_balanced_acc", "cv_accuracy_mean",
                       "fit_time_sec"]].to_string(index=False))
    return df_summary


def run_stage1(model_names=None):
    print("\n" + "=" * 70)
    print("  STAGE 1  ▶  Continental super-population classification")
    print("=" * 70)

    X, y_super, _, le, snp_ids, _ = load_continental()

    df = run_stage(
        X=X, y=y_super,
        label_encoder=le["super_pop"],
        snp_ids=snp_ids,
        stage_tag="stage1_super_pop",
        stage_title="STAGE 1 · Continental super-population  (5 classes)",
        model_names=model_names,
    )

    csv_path = RESULTS_DIR / "stage1_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Stage 1] Summary → {csv_path}")
    return df


def run_stage2(model_names=None):
    print("\n" + "=" * 70)
    print("  STAGE 2  ▶  East-Asian sub-population classification")
    print("=" * 70)

    _, _, y_pop, le, snp_ids, _ = load_eas()
    X, _, _, _, _, _ = load_dataset(EAS_CSV, verbose=False)

    df = run_stage(
        X=X, y=y_pop,
        label_encoder=le["pop"],
        snp_ids=snp_ids,
        stage_tag="stage2_EAS_pop",
        stage_title="STAGE 2 · East-Asian sub-population  (5 classes)",
        model_names=model_names,
    )

    csv_path = RESULTS_DIR / "stage2_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Stage 2] Summary → {csv_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="2-Stage Hierarchical Ancestry Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available models: {', '.join(ALL_MODEL_NAMES)}",
    )
    parser.add_argument(
        "--stage", default="both",
        choices=["1", "2", "both"],
        help="1=super_pop only, 2=EAS_pop only, both=all (default: both)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help="Run only specific model(s). E.g. --models XGBoost LightGBM",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print available model names and exit",
    )
    args = parser.parse_args()

    # --list-models
    if args.list_models:
        print("Available models:")
        for m in ALL_MODEL_NAMES:
            print(f"  • {m}")
        return

    # Validate model names
    model_names = args.models
    if model_names:
        invalid = [m for m in model_names if m not in ALL_MODEL_NAMES]
        if invalid:
            parser.error(
                f"Unknown model(s): {invalid}\n"
                f"Available: {ALL_MODEL_NAMES}"
            )

    summaries = []
    if args.stage in ("1", "both"):
        summaries.append(run_stage1(model_names=model_names))
    if args.stage in ("2", "both"):
        summaries.append(run_stage2(model_names=model_names))

    if len(summaries) > 1:
        combined = pd.concat(summaries, ignore_index=True)
        out = RESULTS_DIR / "ALL_model_comparison.csv"
        combined.to_csv(out, index=False)
        print(f"\n[pipeline] ALL results → {out}")

    print("\n[pipeline] Done. Check the results/ folder.\n")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
