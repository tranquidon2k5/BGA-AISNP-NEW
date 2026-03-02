"""
Pipeline inference hai tầng sử dụng TabPFN.
Tương tự inference_pipeline.py (XGBoost) nhưng dùng TabPFN models.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
import pandas as pd

from src.data_utils import load_continental_csv, load_eastasian_csv, encode_genotypes

# --- Đường dẫn mặc định ---
DATA_CONT_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")
DATA_EAS_PATH = os.path.join("data", "AISNP_by_sample_eastasian.csv")

MODELS_DIR = "models"


def _load_models():
    """Nạp TabPFN models đã train cho cả hai tầng."""
    cont_model = joblib.load(os.path.join(MODELS_DIR, "continent_tabpfn.pkl"))
    cont_le = joblib.load(os.path.join(MODELS_DIR, "continent_tabpfn_label_encoder.pkl"))
    cont_snps = joblib.load(os.path.join(MODELS_DIR, "continent_tabpfn_snp_names.pkl"))

    eas_model = joblib.load(os.path.join(MODELS_DIR, "eastasia_tabpfn.pkl"))
    eas_le = joblib.load(os.path.join(MODELS_DIR, "eastasia_tabpfn_label_encoder.pkl"))
    eas_snps = joblib.load(os.path.join(MODELS_DIR, "eastasia_tabpfn_snp_names.pkl"))

    return cont_model, cont_le, cont_snps, eas_model, eas_le, eas_snps


def _get_sample_vector(sample_id: str, df_encoded: pd.DataFrame, snp_names: list):
    """Lấy vector genotype của một sample từ DataFrame đã encode."""
    row = df_encoded[df_encoded["sample"] == sample_id]
    if row.empty:
        raise ValueError(f"Sample '{sample_id}' không tìm thấy trong dữ liệu.")
    return row[snp_names].values.astype(float)


def predict_sample(sample_id: str) -> dict:
    """
    Dự đoán nguồn gốc địa lý cho một sample sử dụng pipeline hai tầng TabPFN.

    Args:
        sample_id: ID của mẫu (ví dụ 'HG01168').

    Returns:
        dict với các khoá:
          - 'sample': ID mẫu
          - 'continent_pred': châu lục dự đoán
          - 'continent_probs': dict {label: probability}
          - 'eastasia_subpop_pred': quần thể Đông Á (chỉ có nếu continent_pred == 'EAS')
          - 'eastasia_probs': dict {label: probability} (chỉ có nếu EAS)
    """
    cont_model, cont_le, cont_snps, eas_model, eas_le, eas_snps = _load_models()

    # --- Tầng 1: phân loại châu lục ---
    df_cont_raw = load_continental_csv(DATA_CONT_PATH)
    df_cont_enc, _ = encode_genotypes(df_cont_raw)
    X_cont = _get_sample_vector(sample_id, df_cont_enc, cont_snps)

    cont_proba = cont_model.predict_proba(X_cont)[0]
    cont_pred_idx = int(np.argmax(cont_proba))
    cont_pred = cont_le.inverse_transform([cont_pred_idx])[0]
    cont_probs_dict = {label: float(p) for label, p in zip(cont_le.classes_, cont_proba)}

    result = {
        "sample": sample_id,
        "continent_pred": cont_pred,
        "continent_probs": cont_probs_dict,
    }

    # --- Tầng 2: phân loại Đông Á (chỉ khi được dự đoán là EAS) ---
    if cont_pred == "EAS":
        df_eas_raw = load_eastasian_csv(DATA_EAS_PATH)
        df_eas_enc, _ = encode_genotypes(df_eas_raw)
        try:
            X_eas = _get_sample_vector(sample_id, df_eas_enc, eas_snps)
            eas_proba = eas_model.predict_proba(X_eas)[0]
            eas_pred_idx = int(np.argmax(eas_proba))
            eas_pred = eas_le.inverse_transform([eas_pred_idx])[0]
            eas_probs_dict = {label: float(p) for label, p in zip(eas_le.classes_, eas_proba)}
            result["eastasia_subpop_pred"] = eas_pred
            result["eastasia_probs"] = eas_probs_dict
        except ValueError:
            result["eastasia_subpop_pred"] = None
            result["eastasia_probs"] = {}

    return result


def main():
    """Demo: dự đoán một vài samples."""
    sample_ids = ["HG01168", "NA12878", "HG00403"]
    for sid in sample_ids:
        try:
            result = predict_sample(sid)
            print(f"\n=== {result['sample']} ===")
            print(f"  Continent: {result['continent_pred']}")
            print(f"  Continent probs: { {k: f'{v:.3f}' for k, v in result['continent_probs'].items()} }")
            if "eastasia_subpop_pred" in result:
                print(f"  East Asia subpop: {result['eastasia_subpop_pred']}")
                print(f"  East Asia probs: { {k: f'{v:.3f}' for k, v in result['eastasia_probs'].items()} }")
        except ValueError as e:
            print(f"[WARN] {e}")


if __name__ == "__main__":
    main()
