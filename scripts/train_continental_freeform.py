"""
Train FREEFORM-style stacking classifier cho phân loại châu lục (Continental tier).
Pattern tương tự train_continental_tabpfn.py.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib

from src.data_utils import (
    load_continental_csv,
    encode_genotypes,
    split_xy,
    stratified_train_test,
    encode_labels,
)
from src.xgboost_models import train_and_eval
from src.freeform_model import FreeformBGAClassifier


DATA_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")


def main():
    df_raw = load_continental_csv(DATA_PATH)
    print(f"Loaded continental data: {df_raw.shape}")

    df_encoded, snp_names = encode_genotypes(df_raw)
    print(f"Encoded continental data: {df_encoded.shape}")
    print(f"Number of SNPs (continental panel): {len(snp_names)}")

    X, y = split_xy(df_encoded, snp_names, label_col="super_pop")

    y_int, le = encode_labels(y)
    X_train, X_test, y_train, y_test = stratified_train_test(X, y_int)

    model = FreeformBGAClassifier()
    model.fit(X_train, y_train, snp_names=snp_names)

    acc, _ = train_and_eval(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        label_encoder=le,
        title="FREEFORM Continental Ancestry",
    )

    # Train on full data for deployment
    model_full = FreeformBGAClassifier()
    model_full.fit(X, y_int, snp_names=snp_names)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_full, os.path.join("models", "continent_freeform.pkl"))
    joblib.dump(le, os.path.join("models", "continent_freeform_label_encoder.pkl"))
    joblib.dump(snp_names, os.path.join("models", "continent_freeform_snp_names.pkl"))

    print(f"Saved continental FREEFORM model (acc on test={acc:.4f}) to models/.")


if __name__ == "__main__":
    main()
