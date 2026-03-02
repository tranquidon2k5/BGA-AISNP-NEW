# scripts/inference_pipeline.py
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import (
    load_continental_csv,
    load_eastasian_csv,
    encode_genotypes,
)
import joblib


# === ĐƯỜNG DẪN FILE ===
CONT_DATA_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")
EAS_DATA_PATH = os.path.join("data", "AISNP_by_sample_eastasian.csv")

MODELS_DIR = "models"
CONT_MODEL_PATH = os.path.join(MODELS_DIR, "continent_xgb.pkl")
CONT_LE_PATH = os.path.join(MODELS_DIR, "continent_label_encoder.pkl")
CONT_SNPS_PATH = os.path.join(MODELS_DIR, "continent_snp_names.pkl")

# dùng model đã tune
EAS_MODEL_PATH = os.path.join(MODELS_DIR, "eastasia_xgb.pkl")
EAS_LE_PATH = os.path.join(MODELS_DIR, "eastasia_label_encoder.pkl")
EAS_SNPS_PATH = os.path.join(MODELS_DIR, "eastasia_snp_names.pkl")


def load_all_models():
    """Load model + encoder + danh sách SNP cho 2 tầng."""
    cont_model = joblib.load(CONT_MODEL_PATH)
    cont_le = joblib.load(CONT_LE_PATH)
    cont_snp_names = joblib.load(CONT_SNPS_PATH)

    eas_model = joblib.load(EAS_MODEL_PATH)
    eas_le = joblib.load(EAS_LE_PATH)
    eas_snp_names = joblib.load(EAS_SNPS_PATH)

    return cont_model, cont_le, cont_snp_names, eas_model, eas_le, eas_snp_names


def load_and_encode_data():
    """
    Đọc lại 2 file CSV raw và encode genotype 0/1/2.
    Làm y hệt như lúc train để đảm bảo mapping giống nhau.
    """
    df_cont_raw = load_continental_csv(CONT_DATA_PATH)
    df_cont_enc, cont_snp_names_from_data = encode_genotypes(df_cont_raw)

    df_eas_raw = load_eastasian_csv(EAS_DATA_PATH)
    df_eas_enc, eas_snp_names_from_data = encode_genotypes(df_eas_raw)

    # kiểm tra cho chắc: danh sách SNP trong model khớp với encode hiện tại
    return df_cont_enc, df_eas_enc


def predict_sample(sample_id: str):
    """
    Pipeline 2 tầng cho 1 sample:
      1) Dùng model continent → dự đoán châu lục.
      2) Nếu dự đoán là 'EAS' → dùng model Đông Á để dự đoán subpopulation.
    """
    (
        cont_model,
        cont_le,
        cont_snp_names,
        eas_model,
        eas_le,
        eas_snp_names,
    ) = load_all_models()

    df_cont_enc, df_eas_enc = load_and_encode_data()

    # lấy dòng có sample_id tương ứng
    row_cont = df_cont_enc[df_cont_enc["sample"] == sample_id]
    if row_cont.empty:
        raise ValueError(f"Sample {sample_id} không tồn tại trong continental CSV.")

    # continent prediction
    X_cont = row_cont[cont_snp_names].values.astype(float)
    cont_probs = cont_model.predict_proba(X_cont)[0]
    cont_idx = int(np.argmax(cont_probs))
    cont_label = cont_le.inverse_transform([cont_idx])[0]

    result = {
        "sample": sample_id,
        "continent_pred": cont_label,
        "continent_probs": {
            cls: float(p) for cls, p in zip(cont_le.classes_, cont_probs)
        },
    }

    # nếu không phải EAS thì dừng ở đây
    if cont_label != "EAS":
        return result

    # nếu là EAS → dự đoán tiếp subpopulation
    row_eas = df_eas_enc[df_eas_enc["sample"] == sample_id]
    if row_eas.empty:
        raise ValueError(f"Sample {sample_id} không tồn tại trong eastasian CSV.")

    X_eas = row_eas[eas_snp_names].values.astype(float)
    eas_probs = eas_model.predict_proba(X_eas)[0]
    eas_idx = int(np.argmax(eas_probs))
    eas_label = eas_le.inverse_transform([eas_idx])[0]

    result["eastasia_subpop_pred"] = eas_label
    result["eastasia_probs"] = {
        cls: float(p) for cls, p in zip(eas_le.classes_, eas_probs)
    }

    return result


if __name__ == "__main__":
    # Ví dụ test nhanh với 1 sample có trong data
    # Bạn thay "HG00096" bằng 1 sample_id thật từ CSV của bạn.
    test_sample = "HG01168"
    out = predict_sample(test_sample)
    print(out)
