"""
Train FREEFORM-style stacking classifier cho phân loại quần thể Đông Á (East Asian tier).
Pattern tương tự train_eastasian_tabpfn.py.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib

from src.data_utils import (
    load_eastasian_csv,
    encode_genotypes,
    split_xy,
    stratified_train_test,
    encode_labels,
)
from src.xgboost_models import train_and_eval
from src.freeform_model import FreeformBGAClassifier


DATA_PATH = os.path.join("data", "AISNP_by_sample_eastasian.csv")


def main():
    df_raw = load_eastasian_csv(DATA_PATH)
    print(f"Loaded eastasian data: {df_raw.shape}")

    df_encoded, snp_names = encode_genotypes(df_raw)
    print(f"Encoded eastasian data: {df_encoded.shape}")
    print(f"Number of SNPs (East Asia panel): {len(snp_names)}")

    df_eas = df_encoded[df_encoded["super_pop"] == "EAS"].copy()
    print(f"East Asian subset: {df_eas.shape}")
    print("East Asian populations:", df_eas["pop"].value_counts())

    X, y = split_xy(df_eas, snp_names, label_col="pop")

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
        title="FREEFORM East Asian Subpopulation",
    )

    # Train on full data for deployment
    model_full = FreeformBGAClassifier()
    model_full.fit(X, y_int, snp_names=snp_names)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_full, os.path.join("models", "eastasia_freeform.pkl"))
    joblib.dump(le, os.path.join("models", "eastasia_freeform_label_encoder.pkl"))
    joblib.dump(snp_names, os.path.join("models", "eastasia_freeform_snp_names.pkl"))

    print(f"Saved East Asian FREEFORM model (acc on test={acc:.4f}) to models/.")


if __name__ == "__main__":
    main()
