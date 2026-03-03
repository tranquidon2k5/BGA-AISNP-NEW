"""
scripts/export_models.py (refactored)
=====================================
Train both stages, then export all fitted models as .pkl files
along with a models_metadata.csv.

Usage:
    python scripts/export_models.py
    python scripts/export_models.py --out-dir /path/to/models
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_continental, load_eas, load_dataset, EAS_CSV
from src.training      import train_all


def export(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []

    # ── Stage 1 ───────────────────────────────────────────────────────────
    X1, y_super, _, le1, snp_ids1, _ = load_continental()
    results1 = train_all(X1, y_super, "stage1_super_pop")

    for r in results1:
        name = r["model_name"]
        tag  = f"stage1_super_pop_{name}"
        path = out_dir / f"{tag}.pkl"
        joblib.dump(r["fitted_model"], path)
        meta_rows.append({
            "stage":     "stage1_super_pop",
            "model":     name,
            "path":      str(path),
            "n_classes":       r["n_classes"],
            "cv_accuracy_mean": r["cv_accuracy_mean"],
            "cv_f1_mean":       r["cv_f1_mean"],
            "best_params":      str(r["best_params"]),
            "exported_at":      datetime.datetime.now().isoformat(),
        })
        print(f"  [export] {tag} → {path}")

    # ── Stage 2 ───────────────────────────────────────────────────────────
    _, _, y_pop, le2, snp_ids2, _ = load_eas()
    X2, _, _, _, _, _ = load_dataset(EAS_CSV, verbose=False)
    results2 = train_all(X2, y_pop, "stage2_EAS_pop")

    for r in results2:
        name = r["model_name"]
        tag  = f"stage2_EAS_pop_{name}"
        path = out_dir / f"{tag}.pkl"
        joblib.dump(r["fitted_model"], path)
        meta_rows.append({
            "stage":     "stage2_EAS_pop",
            "model":     name,
            "path":      str(path),
            "n_classes":       r["n_classes"],
            "cv_accuracy_mean": r["cv_accuracy_mean"],
            "cv_f1_mean":       r["cv_f1_mean"],
            "best_params":      str(r["best_params"]),
            "exported_at":      datetime.datetime.now().isoformat(),
        })
        print(f"  [export] {tag} → {path}")

    # ── Metadata CSV ──────────────────────────────────────────────────────
    meta_df = pd.DataFrame(meta_rows)
    meta_path = out_dir / "models_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\n[export] Metadata → {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Export trained models")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "models"),
                        help="Directory to save models")
    args = parser.parse_args()

    export(Path(args.out_dir))


if __name__ == "__main__":
    main()
