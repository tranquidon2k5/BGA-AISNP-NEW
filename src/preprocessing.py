"""
src/preprocessing.py
====================
Unified data loading and genotype encoding for the BGA-AISNP project.

Merges functionality from the two previous implementations:
  - src/data_utils.py   (developer A — Counter-based major-allele encoding)
  - scripts/preprocessing.py (developer B — minor-allele additive encoding)

The canonical pipeline is:
    load_dataset() → returns (X, y_super_pop, y_pop, label_encoders, snp_ids, df_meta)

Encoding
--------
For each biallelic SNP (columns rsXXXX_1 / rsXXXX_2):
    1. Infer minor allele from allele frequency across all samples.
    2. Additive coding: count of minor alleles per sample → 0, 1, or 2.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Paths  (relative to project root)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "DATA"

CONTINENTAL_CSV = DATA_DIR / "AISNP_by_sample_continental.csv"
EAS_CSV         = DATA_DIR / "AISNP_by_sample_EAS_only.csv"

META_COLS = ["sample", "pop", "super_pop"]


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_snp_ids(columns: list[str]) -> list[str]:
    """Return sorted list of unique SNP IDs from column names ending in _1/_2."""
    seen: set[str] = set()
    snps: list[str] = []
    for c in columns:
        if c.endswith("_1") or c.endswith("_2"):
            snp_id = c[:-2]
            if snp_id not in seen:
                snps.append(snp_id)
                seen.add(snp_id)
    return sorted(snps)


def additive_encode(df: pd.DataFrame, snp_ids: list[str]) -> np.ndarray:
    """
    Additive encoding: for each SNP, infer the minor allele then count it.

    Returns float32 matrix of shape (n_samples, n_snps).
    Missing / monomorphic SNPs get value 0.
    """
    X = np.zeros((len(df), len(snp_ids)), dtype=np.float32)
    for j, snp in enumerate(snp_ids):
        col1 = f"{snp}_1"
        col2 = f"{snp}_2"
        if col1 not in df.columns or col2 not in df.columns:
            continue

        a1 = df[col1].astype(str).str.upper().values
        a2 = df[col2].astype(str).str.upper().values

        # Combine alleles, filter out missing
        all_alleles = np.concatenate([a1, a2])
        mask_valid = ~np.isin(all_alleles, ["NAN", "NONE", "0", ""])
        valid_alleles = all_alleles[mask_valid]

        if len(valid_alleles) == 0:
            continue

        vals, counts = np.unique(valid_alleles, return_counts=True)
        if len(vals) < 2:
            continue  # monomorphic

        minor = vals[np.argmin(counts)]
        X[:, j] = (a1 == minor).astype(np.float32) + (a2 == minor).astype(np.float32)

    return X


# ─────────────────────────────────────────────────────────────────────────────
# Main loading functions
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    csv_path: str | Path,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, list[str], pd.DataFrame]:
    """
    Load a single AISNP CSV, encode genotypes, and return everything needed.

    Returns
    -------
    X           : np.ndarray (n_samples, n_snps) — additive-encoded features
    y_super_pop : np.ndarray of int
    y_pop       : np.ndarray of int
    le_dict     : {"super_pop": LabelEncoder, "pop": LabelEncoder}
    snp_ids     : list[str]
    df_meta     : pd.DataFrame with columns [sample, pop, super_pop]
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if verbose:
        print(f"[preprocess] Loaded {csv_path.name}  ->  "
              f"{df.shape[0]} samples, {df.shape[1]} columns")

    snp_ids = get_snp_ids(list(df.columns))
    if verbose:
        print(f"[preprocess] Found {len(snp_ids)} SNPs")

    X = additive_encode(df, snp_ids)

    le_super = LabelEncoder().fit(df["super_pop"])
    le_pop   = LabelEncoder().fit(df["pop"])

    y_super_pop = le_super.transform(df["super_pop"])
    y_pop       = le_pop.transform(df["pop"])

    df_meta = df[META_COLS].copy()

    if verbose:
        print(f"[preprocess] super_pop classes: {list(le_super.classes_)}")
        print(f"[preprocess] pop       classes: {list(le_pop.classes_)}")

    return X, y_super_pop, y_pop, {"super_pop": le_super, "pop": le_pop}, snp_ids, df_meta


def load_continental(verbose: bool = True):
    """Convenience: load the continental dataset."""
    return load_dataset(CONTINENTAL_CSV, verbose=verbose)


def load_eas(verbose: bool = True):
    """Convenience: load the East-Asian-only dataset."""
    return load_dataset(EAS_CSV, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def split_xy(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Stratified train/test split."""
    from sklearn.model_selection import train_test_split
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CONTINENTAL dataset")
    print("=" * 60)
    X, ys, yp, le, snps, meta = load_continental()
    print(f"X shape: {X.shape}, super_pop unique: {np.unique(ys)}, "
          f"pop unique: {np.unique(yp)}")

    print()
    print("=" * 60)
    print("EAS-only dataset")
    print("=" * 60)
    X2, ys2, yp2, le2, snps2, meta2 = load_eas()
    print(f"X shape: {X2.shape}, super_pop unique: {np.unique(ys2)}, "
          f"pop unique: {np.unique(yp2)}")
