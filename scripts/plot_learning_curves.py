# scripts/plot_learning_curves.py
"""
Vẽ Learning Curves cho XGBoost và Generative Model.
Learning curve cho thấy hiệu suất model thay đổi theo kích thước tập training.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no GUI window
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import (
    load_continental_csv,
    encode_genotypes,
    split_xy,
    encode_labels,
)
from src.models import make_xgb_multiclass
from src.generative_model import GenerativeBGAModel


# -------------------------------------------------------------------------
# Generative model wrapper (sklearn-compatible để dùng với learning_curve)
# -------------------------------------------------------------------------
class GenerativeModelWrapper:
    """Wrapper cho GenerativeBGAModel để tương thích với sklearn learning_curve."""

    def __init__(self, snp_names, smoothing_alpha=1.0):
        self.snp_names = snp_names
        self.smoothing_alpha = smoothing_alpha
        self.model = None

    def fit(self, X, y):
        self.model = GenerativeBGAModel(smoothing_alpha=self.smoothing_alpha)
        self.model.fit(X, y, self.snp_names)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self, deep=True):
        return {"snp_names": self.snp_names, "smoothing_alpha": self.smoothing_alpha}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def plot_learning_curves_manual(X, y, snp_names, title_suffix="", save_path=None):
    """
    Vẽ learning curves thủ công cho cả XGBoost và Generative model.
    Cách này cho phép kiểm soát tốt hơn.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    # Các tỷ lệ training data
    train_sizes = np.linspace(0.1, 1.0, 10)
    n_splits = 5  # Cross-validation splits

    # Lưu kết quả
    results = {
        "XGBoost": {"train_sizes": [], "train_scores": [], "test_scores": []},
        "Generative": {"train_sizes": [], "train_scores": [], "test_scores": []},
    }

    # Số classes
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {le.classes_}")
    print(f"Total samples: {len(y)}")

    # Với mỗi train_size
    for train_size in train_sizes:
        print(f"\n--- Training size: {train_size*100:.0f}% ---")

        xgb_train_scores = []
        xgb_test_scores = []
        gen_train_scores = []
        gen_test_scores = []

        # Cross-validation
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(sss.split(X, y_encoded)):
            # Lấy subset theo train_size
            n_train = int(len(train_idx) * train_size)
            if n_train < num_classes:
                n_train = num_classes  # Đảm bảo có đủ mẫu cho mỗi class
            train_idx_subset = train_idx[:n_train]

            X_train = X[train_idx_subset]
            y_train = y_encoded[train_idx_subset]
            X_test = X[test_idx]
            y_test = y_encoded[test_idx]

            # --- XGBoost ---
            xgb_model = make_xgb_multiclass(num_classes=num_classes)
            xgb_model.fit(X_train, y_train)
            xgb_train_score = xgb_model.score(X_train, y_train)
            xgb_test_score = xgb_model.score(X_test, y_test)
            xgb_train_scores.append(xgb_train_score)
            xgb_test_scores.append(xgb_test_score)

            # --- Generative ---
            gen_model = GenerativeBGAModel(smoothing_alpha=1.0)
            y_train_str = le.inverse_transform(y_train)
            y_test_str = le.inverse_transform(y_test)
            gen_model.fit(X_train, y_train_str, snp_names)
            
            gen_train_pred = gen_model.predict(X_train)
            gen_test_pred = gen_model.predict(X_test)
            gen_train_score = np.mean(gen_train_pred == y_train_str)
            gen_test_score = np.mean(gen_test_pred == y_test_str)
            gen_train_scores.append(gen_train_score)
            gen_test_scores.append(gen_test_score)

        # Tính mean và std
        n_samples = int(len(train_idx) * train_size)
        results["XGBoost"]["train_sizes"].append(n_samples)
        results["XGBoost"]["train_scores"].append((np.mean(xgb_train_scores), np.std(xgb_train_scores)))
        results["XGBoost"]["test_scores"].append((np.mean(xgb_test_scores), np.std(xgb_test_scores)))

        results["Generative"]["train_sizes"].append(n_samples)
        results["Generative"]["train_scores"].append((np.mean(gen_train_scores), np.std(gen_train_scores)))
        results["Generative"]["test_scores"].append((np.mean(gen_test_scores), np.std(gen_test_scores)))

        print(f"  XGBoost    - Train: {np.mean(xgb_train_scores):.4f}, Test: {np.mean(xgb_test_scores):.4f}")
        print(f"  Generative - Train: {np.mean(gen_train_scores):.4f}, Test: {np.mean(gen_test_scores):.4f}")

    # --- Vẽ Learning Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "XGBoost": {"train": "#1f77b4", "test": "#ff7f0e"},
        "Generative": {"train": "#2ca02c", "test": "#d62728"},
    }

    for idx, model_name in enumerate(["XGBoost", "Generative"]):
        ax = axes[idx]
        data = results[model_name]
        
        train_sizes = np.array(data["train_sizes"])
        train_means = np.array([s[0] for s in data["train_scores"]])
        train_stds = np.array([s[1] for s in data["train_scores"]])
        test_means = np.array([s[0] for s in data["test_scores"]])
        test_stds = np.array([s[1] for s in data["test_scores"]])

        # Plot với fill_between cho std
        ax.fill_between(train_sizes, train_means - train_stds, train_means + train_stds,
                        alpha=0.2, color=colors[model_name]["train"])
        ax.fill_between(train_sizes, test_means - test_stds, test_means + test_stds,
                        alpha=0.2, color=colors[model_name]["test"])
        
        ax.plot(train_sizes, train_means, 'o-', color=colors[model_name]["train"],
                label='Training score', linewidth=2, markersize=6)
        ax.plot(train_sizes, test_means, 'o-', color=colors[model_name]["test"],
                label='Validation score', linewidth=2, markersize=6)

        ax.set_xlabel('Number of training samples', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Learning Curve - {model_name} {title_suffix}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved learning curves to: {save_path}")

    plt.close(fig)  # Close figure to free memory
    return results


def plot_combined_learning_curves(X, y, snp_names, title_suffix="", save_path=None):
    """
    Vẽ learning curves của cả 2 model trên cùng 1 biểu đồ để so sánh.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder

    train_sizes = np.linspace(0.1, 1.0, 10)
    n_splits = 5

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    results = {
        "XGBoost": {"train_sizes": [], "train_scores": [], "test_scores": []},
        "Generative": {"train_sizes": [], "train_scores": [], "test_scores": []},
    }

    print(f"\n{'='*60}")
    print(f"LEARNING CURVES {title_suffix}")
    print(f"{'='*60}")
    print(f"Number of classes: {num_classes}, Total samples: {len(y)}")

    for train_size in train_sizes:
        xgb_train_scores, xgb_test_scores = [], []
        gen_train_scores, gen_test_scores = [], []

        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(sss.split(X, y_encoded)):
            n_train = max(int(len(train_idx) * train_size), num_classes)
            train_idx_subset = train_idx[:n_train]

            X_train, y_train = X[train_idx_subset], y_encoded[train_idx_subset]
            X_test, y_test = X[test_idx], y_encoded[test_idx]

            # XGBoost
            xgb_model = make_xgb_multiclass(num_classes=num_classes)
            xgb_model.fit(X_train, y_train)
            xgb_train_scores.append(xgb_model.score(X_train, y_train))
            xgb_test_scores.append(xgb_model.score(X_test, y_test))

            # Generative
            gen_model = GenerativeBGAModel(smoothing_alpha=1.0)
            y_train_str = le.inverse_transform(y_train)
            y_test_str = le.inverse_transform(y_test)
            gen_model.fit(X_train, y_train_str, snp_names)
            gen_train_scores.append(np.mean(gen_model.predict(X_train) == y_train_str))
            gen_test_scores.append(np.mean(gen_model.predict(X_test) == y_test_str))

        n_samples = int(len(train_idx) * train_size)
        for name, train_s, test_s in [
            ("XGBoost", xgb_train_scores, xgb_test_scores),
            ("Generative", gen_train_scores, gen_test_scores),
        ]:
            results[name]["train_sizes"].append(n_samples)
            results[name]["train_scores"].append((np.mean(train_s), np.std(train_s)))
            results[name]["test_scores"].append((np.mean(test_s), np.std(test_s)))

        print(f"Train {train_size*100:5.1f}% ({n_samples:4d} samples) | "
              f"XGB: {np.mean(xgb_test_scores):.4f} | Gen: {np.mean(gen_test_scores):.4f}")

    # Vẽ biểu đồ so sánh
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {
        "XGBoost": {"color": "#2563eb", "marker": "o"},
        "Generative": {"color": "#dc2626", "marker": "s"},
    }

    for model_name in ["XGBoost", "Generative"]:
        data = results[model_name]
        train_sizes = np.array(data["train_sizes"])
        test_means = np.array([s[0] for s in data["test_scores"]])
        test_stds = np.array([s[1] for s in data["test_scores"]])

        style = styles[model_name]
        ax.fill_between(train_sizes, test_means - test_stds, test_means + test_stds,
                        alpha=0.15, color=style["color"])
        ax.plot(train_sizes, test_means, marker=style["marker"], color=style["color"],
                label=model_name, linewidth=2.5, markersize=8)

    ax.set_xlabel('Number of Training Samples', fontsize=13)
    ax.set_ylabel('Validation Accuracy', fontsize=13)
    ax.set_title(f'Learning Curves Comparison {title_suffix}', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")

    plt.close(fig)  # Close figure to free memory
    return results


def main():
    # Load dữ liệu Continental
    continental_path = os.path.join("data", "AISNP_by_sample_continental.csv")
    df_raw = load_continental_csv(continental_path)
    df_encoded, snp_names = encode_genotypes(df_raw)
    X, y = split_xy(df_encoded, snp_names, label_col="super_pop")

    print("="*60)
    print("LEARNING CURVES - CONTINENTAL ANCESTRY")
    print("="*60)

    # Vẽ learning curves riêng
    results_separate = plot_learning_curves_manual(
        X, y, snp_names,
        title_suffix="(Continental)",
        save_path="learning_curves_continental_separate.png"
    )

    # Vẽ learning curves so sánh
    results_combined = plot_combined_learning_curves(
        X, y, snp_names,
        title_suffix="(Continental)",
        save_path="learning_curves_continental_combined.png"
    )

    # Load dữ liệu East Asian (nếu có)
    eastasian_path = os.path.join("data", "AISNP_by_sample_eastasian.csv")
    if os.path.exists(eastasian_path):
        from src.data_utils import load_eastasian_csv
        
        print("\n" + "="*60)
        print("LEARNING CURVES - EAST ASIAN SUBPOPULATION")
        print("="*60)

        df_raw_ea = load_eastasian_csv(eastasian_path)
        df_encoded_ea, snp_names_ea = encode_genotypes(df_raw_ea)
        X_ea, y_ea = split_xy(df_encoded_ea, snp_names_ea, label_col="pop")

        # Vẽ learning curves riêng
        plot_learning_curves_manual(
            X_ea, y_ea, snp_names_ea,
            title_suffix="(East Asian)",
            save_path="learning_curves_eastasian_separate.png"
        )

        # Vẽ learning curves so sánh
        plot_combined_learning_curves(
            X_ea, y_ea, snp_names_ea,
            title_suffix="(East Asian)",
            save_path="learning_curves_eastasian_combined.png"
        )


if __name__ == "__main__":
    main()

