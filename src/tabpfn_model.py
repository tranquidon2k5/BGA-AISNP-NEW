# src/tabpfn_model.py
"""
Wrapper cho TabPFN classifier.
TabPFN là một pre-trained transformer cho bài toán phân loại tabular,
hoạt động theo cơ chế in-context learning (không cần hyperparameter tuning).

API tương thích scikit-learn: fit, predict, predict_proba.
"""

import os
import pathlib

# --- Tự động cấu hình môi trường HuggingFace ---

# Cho phép chạy với dataset lớn trên CPU (TabPFN v2 có guard mặc định)
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

# Tắt cảnh báo symlinks trên Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Nếu chưa có HF_TOKEN trong env, thử đọc từ cache HuggingFace
if not os.environ.get("HF_TOKEN"):
    _token_path = pathlib.Path.home() / ".cache" / "huggingface" / "token"
    if _token_path.exists():
        os.environ["HF_TOKEN"] = _token_path.read_text(encoding="utf-8").strip()

from tabpfn import TabPFNClassifier


def make_tabpfn_classifier(device: str = "cpu", **kwargs):
    """
    Tạo TabPFNClassifier đã cấu hình sẵn.

    Args:
        device: 'cpu' hoặc 'cuda' (dùng GPU nếu có).
        **kwargs: Các tham số bổ sung truyền thẳng vào TabPFNClassifier.

    Returns:
        TabPFNClassifier instance.
    """
    default_params = dict(
        device=device,
        ignore_pretraining_limits=True,  # cho phép dataset lớn hơn giới hạn mặc định
    )
    default_params.update(kwargs)
    model = TabPFNClassifier(**default_params)
    return model

