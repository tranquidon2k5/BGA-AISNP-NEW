"""
Train TabPFN cho phân loại quần thể Đông Á (East Asian tier).
Cùng pattern với train_eastasian_xgb.py.
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
from src.models import train_and_eval
from src.tabpfn_model import make_tabpfn_classifier


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

    model = make_tabpfn_classifier(device="cpu")

    acc, _ = train_and_eval(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        label_encoder=le,
        title="TabPFN East Asian Subpopulation",
    )

    # Train lại trên full data để deploy
    model_full = make_tabpfn_classifier(device="cpu")
    model_full.fit(X, y_int)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_full, os.path.join("models", "eastasia_tabpfn.pkl"))
    joblib.dump(le, os.path.join("models", "eastasia_tabpfn_label_encoder.pkl"))
    joblib.dump(snp_names, os.path.join("models", "eastasia_tabpfn_snp_names.pkl"))

    print(f"Saved East Asian TabPFN model (acc on test={acc:.4f}) to models/.")


if __name__ == "__main__":
    main()
