import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import (
    load_eastasian_csv,
    encode_genotypes,
    split_xy,
    stratified_train_test,
    encode_labels,
)
from src.models import make_xgb_multiclass, train_and_eval


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

    model = make_xgb_multiclass(num_classes=len(le.classes_))

    acc, _ = train_and_eval(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        label_encoder=le,
        title="XGBoost East Asian Subpopulation (baseline)",
    )

    # train lại full để deploy baseline nếu muốn
    model.fit(X, y_int)

    import joblib

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "eastasia_xgb_baseline.pkl"))
    joblib.dump(le, os.path.join("models", "eastasia_label_encoder_baseline.pkl"))
    joblib.dump(snp_names, os.path.join("models", "eastasia_snp_names.pkl"))

    print(f"Saved baseline EastAsia model (acc on test={acc:.4f}) to models/.")


if __name__ == "__main__":
    main()
