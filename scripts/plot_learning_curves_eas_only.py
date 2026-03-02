# scripts/plot_learning_curves_eas_only.py
"""
Ve Learning Curves cho XGBoost va Generative Model tren EAS-only data.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import encode_genotypes, split_xy
from src.models import make_xgb_multiclass
from src.generative_model import GenerativeBGAModel

# Path to EAS-only data
DATA_PATH = os.path.join("data", "AISNP_by_sample_EAS_only.csv")

# Paper parameters
PAPER_PARAMS_XGB = {
    "learning_rate": 0.1,
    "max_depth": 7,
    "n_estimators": 200,
}


def load_eas_only_csv(path: str) -> pd.DataFrame:
    """Load EAS-only CSV file"""
    df = pd.read_csv(path)
    return df


def plot_learning_curves(X, y, snp_names, save_path=None):
    """
    Ve learning curves cho XGBoost va Generative model.
    """
    train_sizes = np.linspace(0.1, 1.0, 10)
    n_splits = 5

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    results = {
        "XGBoost": {"train_sizes": [], "train_scores": [], "test_scores": []},
        "Generative": {"train_sizes": [], "train_scores": [], "test_scores": []},
    }

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {le.classes_}")
    print(f"Total samples: {len(y)}")
    print(f"\nXGBoost params: {PAPER_PARAMS_XGB}")
    print(f"Generative smoothing_alpha: 1.0")

    for train_size in train_sizes:
        print(f"\n--- Training size: {train_size*100:.0f}% ---")

        xgb_train_scores = []
        xgb_test_scores = []
        gen_train_scores = []
        gen_test_scores = []

        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(sss.split(X, y_encoded)):
            n_train = max(int(len(train_idx) * train_size), num_classes)
            train_idx_subset = train_idx[:n_train]

            X_train = X[train_idx_subset]
            y_train = y_encoded[train_idx_subset]
            X_test = X[test_idx]
            y_test = y_encoded[test_idx]

            # --- XGBoost (paper params) ---
            xgb_model = make_xgb_multiclass(num_classes=num_classes, **PAPER_PARAMS_XGB)
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

        n_samples = int(len(train_idx) * train_size)
        results["XGBoost"]["train_sizes"].append(n_samples)
        results["XGBoost"]["train_scores"].append((np.mean(xgb_train_scores), np.std(xgb_train_scores)))
        results["XGBoost"]["test_scores"].append((np.mean(xgb_test_scores), np.std(xgb_test_scores)))

        results["Generative"]["train_sizes"].append(n_samples)
        results["Generative"]["train_scores"].append((np.mean(gen_train_scores), np.std(gen_train_scores)))
        results["Generative"]["test_scores"].append((np.mean(gen_test_scores), np.std(gen_test_scores)))

        print(f"  XGBoost    - Train: {np.mean(xgb_train_scores):.4f}, Test: {np.mean(xgb_test_scores):.4f}")
        print(f"  Generative - Train: {np.mean(gen_train_scores):.4f}, Test: {np.mean(gen_test_scores):.4f}")

    # --- Ve Learning Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "XGBoost": {"train": "#1f77b4", "test": "#ff7f0e"},
        "Generative": {"train": "#2ca02c", "test": "#d62728"},
    }

    for idx, model_name in enumerate(["XGBoost", "Generative"]):
        ax = axes[idx]
        data = results[model_name]

        train_sizes_arr = np.array(data["train_sizes"])
        train_means = np.array([s[0] for s in data["train_scores"]])
        train_stds = np.array([s[1] for s in data["train_scores"]])
        test_means = np.array([s[0] for s in data["test_scores"]])
        test_stds = np.array([s[1] for s in data["test_scores"]])

        ax.fill_between(train_sizes_arr, train_means - train_stds, train_means + train_stds,
                        alpha=0.2, color=colors[model_name]["train"])
        ax.fill_between(train_sizes_arr, test_means - test_stds, test_means + test_stds,
                        alpha=0.2, color=colors[model_name]["test"])

        ax.plot(train_sizes_arr, train_means, 'o-', color=colors[model_name]["train"],
                label='Training score', linewidth=2, markersize=6)
        ax.plot(train_sizes_arr, test_means, 'o-', color=colors[model_name]["test"],
                label='Validation score', linewidth=2, markersize=6)

        ax.set_xlabel('Number of training samples', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Learning Curve - {model_name} (EAS-only)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved learning curves to: {save_path}")

    plt.close(fig)
    return results


def plot_combined_learning_curves(X, y, snp_names, save_path=None):
    """
    Ve learning curves cua ca 2 model tren cung 1 bieu do.
    """
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
    print("LEARNING CURVES - EAS-ONLY (COMBINED)")
    print(f"{'='*60}")
    print(f"Classes: {num_classes}, Samples: {len(y)}")

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
            xgb_model = make_xgb_multiclass(num_classes=num_classes, **PAPER_PARAMS_XGB)
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

        print(f"Train {train_size*100:5.1f}% ({n_samples:3d} samples) | "
              f"XGB: {np.mean(xgb_test_scores):.4f} | Gen: {np.mean(gen_test_scores):.4f}")

    # Ve bieu do so sanh
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {
        "XGBoost": {"color": "#2563eb", "marker": "o"},
        "Generative": {"color": "#dc2626", "marker": "s"},
    }

    for model_name in ["XGBoost", "Generative"]:
        data = results[model_name]
        train_sizes_arr = np.array(data["train_sizes"])
        test_means = np.array([s[0] for s in data["test_scores"]])
        test_stds = np.array([s[1] for s in data["test_scores"]])

        style = styles[model_name]
        ax.fill_between(train_sizes_arr, test_means - test_stds, test_means + test_stds,
                        alpha=0.15, color=style["color"])
        ax.plot(train_sizes_arr, test_means, marker=style["marker"], color=style["color"],
                label=model_name, linewidth=2.5, markersize=8)

    ax.set_xlabel('Number of Training Samples', fontsize=13)
    ax.set_ylabel('Validation Accuracy', fontsize=13)
    ax.set_title('Learning Curves Comparison (EAS-only, 504 samples)', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")

    plt.close(fig)
    return results


def main():
    print("=" * 60)
    print("LEARNING CURVES - EAS-ONLY DATA")
    print("=" * 60)

    # Load data
    df_raw = load_eas_only_csv(DATA_PATH)
    print(f"\nLoaded EAS-only data: {df_raw.shape}")
    print(f"Populations: {df_raw['pop'].value_counts().to_dict()}")

    df_encoded, snp_names = encode_genotypes(df_raw)
    X, y = split_xy(df_encoded, snp_names, label_col="pop")
    X = X.astype(float)

    # Ve learning curves rieng
    print("\n" + "=" * 60)
    print("PLOTTING SEPARATE LEARNING CURVES")
    print("=" * 60)
    results_separate = plot_learning_curves(
        X, y, snp_names,
        save_path="learning_curves_eas_only_separate.png"
    )

    # Ve learning curves combined
    results_combined = plot_combined_learning_curves(
        X, y, snp_names,
        save_path="learning_curves_eas_only_combined.png"
    )

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print("\nSaved files:")
    print("  - learning_curves_eas_only_separate.png")
    print("  - learning_curves_eas_only_combined.png")


if __name__ == "__main__":
    main()

