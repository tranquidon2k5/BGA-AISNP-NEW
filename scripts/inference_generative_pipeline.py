# scripts/inference_generative_pipeline.py
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
from src.data_utils import (
    load_continental_csv,
    load_eastasian_csv,
    encode_genotypes,
)
from src.generative_model import GenerativeBGAModel


# ===== ĐƯỜNG DẪN =====
DATA_CONT_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")
DATA_EAS_PATH = os.path.join("data", "AISNP_by_sample_eastasian.csv")

MODELS_DIR = "models"
CONT_GEN_PATH = os.path.join(MODELS_DIR, "continent_gen_model.pkl")
EAS_GEN_PATH = os.path.join(MODELS_DIR, "eastasia_gen_model.pkl")


def load_models_and_data():
    # Load model đã train
    cont_model: GenerativeBGAModel = joblib.load(CONT_GEN_PATH)
    eas_model: GenerativeBGAModel = joblib.load(EAS_GEN_PATH)

    # Encode lại data để có X đúng thứ tự SNP
    df_cont_raw = load_continental_csv(DATA_CONT_PATH)
    df_cont_enc, cont_snp_names = encode_genotypes(df_cont_raw)

    df_eas_raw = load_eastasian_csv(DATA_EAS_PATH)
    df_eas_enc, eas_snp_names = encode_genotypes(df_eas_raw)

    return cont_model, eas_model, df_cont_enc, df_eas_enc, cont_snp_names, eas_snp_names


def predict_sample_generative(
    sample_id: str,
    threshold_cont: float = 0.7,
    threshold_eas: float = 0.7,
):
    """
    Pipeline phân cấp cho 1 sample:

    1) Generative Continental → label_cont, prob_cont, nếu max_prob<threshold_cont -> 'UNKNOWN_CONT'.
    2) Nếu label_cont == 'EAS' → Generative EastAsia → subpop hoặc 'UNKNOWN_EAS'.
    """
    (
        cont_model,
        eas_model,
        df_cont_enc,
        df_eas_enc,
        cont_snp_names,
        eas_snp_names,
    ) = load_models_and_data()

    # ===== Tầng 1: continent =====
    row_cont = df_cont_enc[df_cont_enc["sample"] == sample_id]
    if row_cont.empty:
        raise ValueError(f"Sample {sample_id} không tồn tại trong continental CSV.")

    Xc = row_cont[cont_snp_names].values.astype(float)
    labels_cont, max_probs_cont, proba_cont = cont_model.predict_with_uncertainty(
        Xc, threshold=threshold_cont
    )

    label_cont = labels_cont[0]
    maxp_cont = float(max_probs_cont[0])
    probs_cont = proba_cont[0]

    result = {
        "sample": sample_id,
        "continent_pred": label_cont,
        "continent_max_prob": maxp_cont,
        "continent_probs": {
            pop: float(p) for pop, p in zip(cont_model.pop_labels, probs_cont)
        },
    }

    # Nếu mô hình thấy không chắc chắn ở tầng continent:
    if label_cont == "UNKNOWN":
        result["note"] = "Low confidence at continental level"
        return result

    # Nếu không phải EAS thì dừng ở đây
    if label_cont != "EAS":
        return result

    # ===== Tầng 2: EastAsia subpopulation =====
    row_eas = df_eas_enc[df_eas_enc["sample"] == sample_id]
    if row_eas.empty:
        # trường hợp hiếm: sample EAS nhưng không có trong eastasian CSV
        result["note"] = "EAS continent but no EastAsia SNP record"
        return result

    Xe = row_eas[eas_snp_names].values.astype(float)
    labels_eas, max_probs_eas, proba_eas = eas_model.predict_with_uncertainty(
        Xe, threshold=threshold_eas
    )

    label_eas = labels_eas[0]
    maxp_eas = float(max_probs_eas[0])
    probs_eas = proba_eas[0]

    result["eastasia_subpop_pred"] = label_eas
    result["eastasia_max_prob"] = maxp_eas
    result["eastasia_probs"] = {
        pop: float(p) for pop, p in zip(eas_model.pop_labels, probs_eas)
    }

    if label_eas == "UNKNOWN":
        result["note"] = "Low confidence at EastAsia subpopulation level"

    return result


if __name__ == "__main__":
    # TODO: thay bằng 1 sample EAS thật, ví dụ lấy từ csv (CHB, CHS, ...).
    test_sample = "HG00096"
    out = predict_sample_generative(test_sample, threshold_cont=0.7, threshold_eas=0.7)
    print(out)
