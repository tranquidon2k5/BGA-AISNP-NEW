# scripts/train_with_paper_params.py
"""
Train XGBoost models với tham số từ paper:
- Continental: learning_rate=0.1, max_depth=5, n_estimators=200
- East Asian: learning_rate=0.1, max_depth=7, n_estimators=200
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import (
    load_continental_csv,
    load_eastasian_csv,
    encode_genotypes,
    split_xy,
    stratified_train_test,
    encode_labels,
)
from src.models import make_xgb_multiclass, train_and_eval
import joblib


# ============================================================================
# Paper Parameters (from Supplementary Table 6)
# ============================================================================

PAPER_PARAMS_CONTINENTAL = {
    "learning_rate": 0.1,
    "max_depth": 5,
    "n_estimators": 200,
}

PAPER_PARAMS_EASTASIAN = {
    "learning_rate": 0.1,
    "max_depth": 7,
    "n_estimators": 200,
}


def train_continental_with_paper_params():
    """Train Continental model với tham số từ paper"""
    print("=" * 70)
    print("TRAINING CONTINENTAL XGBOOST WITH PAPER PARAMETERS")
    print("=" * 70)
    print(f"Parameters: {PAPER_PARAMS_CONTINENTAL}")
    print()

    DATA_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")
    df_raw = load_continental_csv(DATA_PATH)
    print(f"Loaded continental data: {df_raw.shape}")

    df_encoded, snp_names = encode_genotypes(df_raw)
    print(f"Encoded continental data: {df_encoded.shape}")
    print(f"Number of SNPs (continental panel): {len(snp_names)}")

    X, y = split_xy(df_encoded, snp_names, label_col="super_pop")
    y_int, le = encode_labels(y)

    X_train, X_test, y_train, y_test = stratified_train_test(X, y_int)

    # Train với tham số từ paper
    model = make_xgb_multiclass(
        num_classes=len(le.classes_),
        **PAPER_PARAMS_CONTINENTAL
    )

    acc, _ = train_and_eval(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        label_encoder=le,
        title="XGBoost Continental (Paper Parameters)",
    )

    # Train lại trên full data để deploy
    model.fit(X, y_int)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        model, os.path.join("models", "continent_xgb_paper.pkl")
    )
    joblib.dump(
        le, os.path.join("models", "continent_label_encoder_paper.pkl")
    )
    joblib.dump(
        snp_names, os.path.join("models", "continent_snp_names_paper.pkl")
    )

    print(f"\n[OK] Saved Continental model (paper params, acc={acc:.4f}) to models/")
    return acc


def train_eastasian_with_paper_params():
    """Train East Asian model với tham số từ paper"""
    print("\n" + "=" * 70)
    print("TRAINING EAST ASIAN XGBOOST WITH PAPER PARAMETERS")
    print("=" * 70)
    print(f"Parameters: {PAPER_PARAMS_EASTASIAN}")
    print()

    DATA_PATH = os.path.join("data", "AISNP_by_sample_eastasian.csv")
    df_raw = load_eastasian_csv(DATA_PATH)
    print(f"Loaded eastasian data: {df_raw.shape}")

    df_encoded, snp_names = encode_genotypes(df_raw)
    print(f"Encoded eastasian data: {df_encoded.shape}")
    print(f"Number of SNPs (East Asia panel): {len(snp_names)}")

    # Lọc chỉ EAS
    df_eas = df_encoded[df_encoded["super_pop"] == "EAS"].copy()
    print(f"East Asian subset: {df_eas.shape}")
    print("East Asian populations:", df_eas["pop"].value_counts())

    X, y = split_xy(df_eas, snp_names, label_col="pop")
    y_int, le = encode_labels(y)

    X_train, X_test, y_train, y_test = stratified_train_test(X, y_int)

    # Train với tham số từ paper
    model = make_xgb_multiclass(
        num_classes=len(le.classes_),
        **PAPER_PARAMS_EASTASIAN
    )

    acc, _ = train_and_eval(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        label_encoder=le,
        title="XGBoost East Asian (Paper Parameters)",
    )

    # Train lại trên full data để deploy
    model.fit(X, y_int)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        model, os.path.join("models", "eastasia_xgb_paper.pkl")
    )
    joblib.dump(
        le, os.path.join("models", "eastasia_label_encoder_paper.pkl")
    )
    joblib.dump(
        snp_names, os.path.join("models", "eastasia_snp_names_paper.pkl")
    )

    print(f"\n[OK] Saved East Asian model (paper params, acc={acc:.4f}) to models/")
    return acc


def compare_with_baseline():
    """So sánh kết quả với baseline models nếu có"""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    try:
        # Load baseline models nếu có
        baseline_cont = joblib.load("models/continent_xgb.pkl")
        baseline_eas = joblib.load("models/eastasia_xgb_baseline.pkl")

        print("\n[INFO] Continental Model Comparison:")
        print(f"  Baseline params: max_depth={baseline_cont.max_depth}, "
              f"learning_rate={baseline_cont.learning_rate}, "
              f"n_estimators={baseline_cont.n_estimators}")
        print(f"  Paper params: {PAPER_PARAMS_CONTINENTAL}")

        print("\n[INFO] East Asian Model Comparison:")
        print(f"  Baseline params: max_depth={baseline_eas.max_depth}, "
              f"learning_rate={baseline_eas.learning_rate}, "
              f"n_estimators={baseline_eas.n_estimators}")
        print(f"  Paper params: {PAPER_PARAMS_EASTASIAN}")

    except FileNotFoundError:
        print("[WARNING] Baseline models not found. Run baseline training first.")


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("TRAINING XGBOOST MODELS WITH PAPER PARAMETERS")
    print("=" * 70)
    print("\nPaper: Supplementary Table 6 - Suitable parameters")
    print("Continental: learning_rate=0.1, max_depth=5, n_estimators=200")
    print("East Asian: learning_rate=0.1, max_depth=7, n_estimators=200")
    print()

    # Train Continental
    acc_cont = train_continental_with_paper_params()

    # Train East Asian
    acc_eas = train_eastasian_with_paper_params()

    # Compare
    compare_with_baseline()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Continental accuracy (paper params): {acc_cont:.4f}")
    print(f"East Asian accuracy (paper params): {acc_eas:.4f}")
    print("\n[OK] Training completed! Models saved with '_paper' suffix.")


if __name__ == "__main__":
    main()

