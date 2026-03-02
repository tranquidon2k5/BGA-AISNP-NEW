# scripts/feature_importance.py
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np

from src.data_utils import (
    load_continental_csv,
    load_eastasian_csv,
    encode_genotypes,
)


MODELS_DIR = "models"

CONT_MODEL_PATH = os.path.join(MODELS_DIR, "continent_xgb.pkl")
CONT_LE_PATH = os.path.join(MODELS_DIR, "continent_label_encoder.pkl")
CONT_SNPS_PATH = os.path.join(MODELS_DIR, "continent_snp_names.pkl")

EAS_MODEL_PATH = os.path.join(MODELS_DIR, "eastasia_xgb.pkl")
EAS_LE_PATH = os.path.join(MODELS_DIR, "eastasia_label_encoder.pkl")

CONT_DATA_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")
EAS_DATA_PATH = os.path.join("data", "AISNP_by_sample_eastasian.csv")


def map_importance_to_snp(model, snp_names):
    """
    XGBoost booster feature names là f0, f1, ..., f{n-1}
    ta map theo index vào snp_names.
    """
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")  # hoặc "weight", "total_gain", ...

    # score là dict: {"f0": value, "f1": value, ...}
    snp_importance = []
    for fname, val in score.items():
        # fname dạng "f0", "f1", ...
        idx = int(fname[1:])
        if idx < len(snp_names):
            snp_importance.append((snp_names[idx], val))

    # sort giảm dần theo importance
    snp_importance.sort(key=lambda x: x[1], reverse=True)
    return snp_importance


def main():
    # === Continental ===
    cont_model = joblib.load(CONT_MODEL_PATH)
    cont_le = joblib.load(CONT_LE_PATH)
    cont_snp_names = joblib.load(CONT_SNPS_PATH)

    print("Classes (continent):", cont_le.classes_)
    cont_importance = map_importance_to_snp(cont_model, cont_snp_names)

    print("\n=== Top 20 SNP quan trọng nhất (Continental) ===")
    for snp, val in cont_importance[:20]:
        print(f"{snp:15s}  importance={val:.4f}")

    # === East Asia ===
    # Với Đông Á, ta lấy lại danh sách SNP bằng cách encode từ file eastasian
    df_eas_raw = load_eastasian_csv(EAS_DATA_PATH)
    df_eas_enc, eas_snp_names = encode_genotypes(df_eas_raw)

    eas_model = joblib.load(EAS_MODEL_PATH)
    eas_le = joblib.load(EAS_LE_PATH)

    print("\nClasses (EastAsia):", eas_le.classes_)
    eas_importance = map_importance_to_snp(eas_model, eas_snp_names)

    print("\n=== Top 20 SNP quan trọng nhất (East Asia) ===")
    for snp, val in eas_importance[:20]:
        print(f"{snp:15s}  importance={val:.4f}")


if __name__ == "__main__":
    main()
