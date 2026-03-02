# scripts/tune_eastasian_xgb.py
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import (
    load_eastasian_csv,
    encode_genotypes,
)
from src.data_utils import split_xy, encode_labels
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from src.models import make_xgb_multiclass


DATA_PATH = os.path.join("data", "AISNP_by_sample_eastasian.csv")


def main():
    df_raw = load_eastasian_csv(DATA_PATH)
    print(f"Loaded eastasian data: {df_raw.shape}")

    df_encoded, snp_names = encode_genotypes(df_raw)
    print(f"Encoded eastasian data: {df_encoded.shape}")
    print(f"Number of SNPs (East Asia panel): {len(snp_names)}")

    # lọc chỉ EAS
    df_eas = df_encoded[df_encoded["super_pop"] == "EAS"].copy()
    print(f"East Asian subset: {df_eas.shape}")
    print("East Asian populations:", df_eas["pop"].value_counts())

    X, y = split_xy(df_eas, snp_names, label_col="pop")
    y_int, le = encode_labels(y)

    num_classes = len(le.classes_)
    print("Classes:", le.classes_)

    base_model = make_xgb_multiclass(num_classes=num_classes)

    # space hyperparam
    param_dist = {
        "n_estimators": [150, 200, 300, 400, 500],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 3, 5],
        "gamma": [0, 0.1, 0.3, 0.5],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
    }

    # macro F1 cho multi-class
    macro_f1 = make_scorer(f1_score, average="macro")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=40,              # bạn có thể tăng/giảm tùy thời gian
        scoring=macro_f1,
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42,
    )

    search.fit(X, y_int)

    print("=== Best params (macro F1) ===")
    print(search.best_params_)
    print(f"Best CV macro F1: {search.best_score_:.4f}")

    # train final model với best params trên toàn bộ dữ liệu East Asia
    best_params = search.best_params_
    final_model = make_xgb_multiclass(num_classes=num_classes, **best_params)
    final_model.fit(X, y_int)

    # lưu model + label encoder
    import joblib

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, os.path.join("models", "eastasia_xgb.pkl"))
    joblib.dump(le, os.path.join("models", "eastasia_label_encoder.pkl"))

    print("Saved tuned EastAsia model + label encoder into models/ folder.")


if __name__ == "__main__":
    main()
