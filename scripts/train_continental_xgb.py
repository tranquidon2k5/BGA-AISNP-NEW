import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import (
    load_continental_csv,
    encode_genotypes,
    split_xy,
    stratified_train_test,
    encode_labels,
)
from src.models import make_xgb_multiclass, train_and_eval


DATA_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")


def main():
    df_raw = load_continental_csv(DATA_PATH)
    print(f"Loaded continental data: {df_raw.shape}")

    df_encoded, snp_names = encode_genotypes(df_raw)
    print(f"Encoded continental data: {df_encoded.shape}")
    print(f"Number of SNPs (continental panel): {len(snp_names)}")

    X, y = split_xy(df_encoded, snp_names, label_col="super_pop")

    y_int, le = encode_labels(y)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = stratified_train_test(X, y_int)

    model = make_xgb_multiclass(num_classes=len(le.classes_))

    acc, _ = train_and_eval(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        label_encoder=le,
        title="XGBoost Continental Ancestry",
    )

    # Train lại trên full data để deploy
    model.fit(X, y_int)

    # Save model + encoder + danh sách SNP
    import joblib

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "continent_xgb.pkl"))
    joblib.dump(le, os.path.join("models", "continent_label_encoder.pkl"))
    joblib.dump(snp_names, os.path.join("models", "continent_snp_names.pkl"))

    print(f"Saved continental model (acc on test={acc:.4f}) to models/.")


if __name__ == "__main__":
    main()
