# scripts/eval_generative_uncertainty.py
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.data_utils import (
    load_continental_csv,
    load_eastasian_csv,
    encode_genotypes,
    split_xy,
)
from src.generative_model import GenerativeBGAModel


DATA_CONT_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")
DATA_EAS_PATH = os.path.join("data", "AISNP_by_sample_eastasian.csv")

MODELS_DIR = "models"
CONT_GEN_PATH = os.path.join(MODELS_DIR, "continent_gen_model.pkl")
EAS_GEN_PATH = os.path.join(MODELS_DIR, "eastasia_gen_model.pkl")


def eval_uncertainty_task(name, model, X, y, thresholds):
    """
    X, y: full dataset cho task (continental hoặc EastAsia).
    Ta chia train/test như trong train_generative_bga.py rồi evaluate trên test.
    """
    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\n=== {name} ===")
    print(f"Test size: {len(y_test)} samples")

    # model đã được train trên full data (đã load từ .pkl), nên ta dùng luôn
    for thr in thresholds:
        labels, max_probs, _ = model.predict_with_uncertainty(X_test, threshold=thr)
        known_mask = labels != "UNKNOWN"

        coverage = known_mask.mean()
        if coverage > 0:
            acc = accuracy_score(y_test[known_mask], labels[known_mask])
        else:
            acc = np.nan

        print(f"Threshold={thr:.2f} | coverage={coverage:.3f} | acc_known={acc:.3f}")


def main():
    # ===== Load models =====
    cont_model: GenerativeBGAModel = joblib.load(CONT_GEN_PATH)
    eas_model: GenerativeBGAModel = joblib.load(EAS_GEN_PATH)

    # ===== Continental data =====
    df_cont_raw = load_continental_csv(DATA_CONT_PATH)
    df_cont_enc, cont_snp_names = encode_genotypes(df_cont_raw)
    Xc, yc = split_xy(df_cont_enc, cont_snp_names, label_col="super_pop")

    # ===== EastAsia data (chỉ EAS) =====
    df_eas_raw = load_eastasian_csv(DATA_EAS_PATH)
    df_eas_enc, eas_snp_names = encode_genotypes(df_eas_raw)
    df_eas = df_eas_enc[df_eas_enc["super_pop"] == "EAS"].copy()
    Xe, ye = split_xy(df_eas, eas_snp_names, label_col="pop")

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    eval_uncertainty_task(
        "Generative Continental BGA (uncertainty)",
        cont_model,
        Xc,
        yc,
        thresholds,
    )

    eval_uncertainty_task(
        "Generative EastAsia BGA (uncertainty)",
        eas_model,
        Xe,
        ye,
        thresholds,
    )


if __name__ == "__main__":
    main()

