"""
scripts/plot_learning_curves.py (refactored)
============================================
Plot learning curves for models across different training set sizes.

Usage:
    python scripts/plot_learning_curves.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing   import LabelEncoder

from src.preprocessing     import load_continental, load_eas
from src.model_registry    import build_models
from src.generative_model  import GenerativeBGAModel

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def plot_learning_curves(
    X, y, snp_names,
    model_configs: dict,
    title_suffix: str = "",
    save_path: str | Path | None = None,
    n_splits: int = 5,
):
    """
    Plot learning curves for multiple models.

    model_configs: dict mapping model_name → callable(X_train, y_train) → fitted model
    """
    train_sizes_frac = np.linspace(0.1, 1.0, 10)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    results = {name: {"sizes": [], "train": [], "test": []}
               for name in model_configs}

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

    for frac in train_sizes_frac:
        print(f"  Training size: {frac*100:.0f}%")
        scores = {name: {"train": [], "test": []} for name in model_configs}

        for train_idx, test_idx in sss.split(X, y_encoded):
            n_train = max(int(len(train_idx) * frac), num_classes)
            train_sub = train_idx[:n_train]

            X_tr, y_tr = X[train_sub], y_encoded[train_sub]
            X_te, y_te = X[test_idx],  y_encoded[test_idx]

            for name, train_fn in model_configs.items():
                model = train_fn(X_tr, y_tr, le)
                y_tr_str = le.inverse_transform(y_tr)
                y_te_str = le.inverse_transform(y_te)

                tr_pred = model.predict(X_tr) if not isinstance(model, GenerativeBGAModel) \
                    else model.predict(X_tr)
                te_pred = model.predict(X_te) if not isinstance(model, GenerativeBGAModel) \
                    else model.predict(X_te)

                if isinstance(model, GenerativeBGAModel):
                    tr_score = np.mean(tr_pred == y_tr_str)
                    te_score = np.mean(te_pred == y_te_str)
                else:
                    tr_score = model.score(X_tr, y_tr)
                    te_score = model.score(X_te, y_te)

                scores[name]["train"].append(tr_score)
                scores[name]["test"].append(te_score)

        n_samples = int(len(train_idx) * frac)
        for name in model_configs:
            results[name]["sizes"].append(n_samples)
            results[name]["train"].append(
                (np.mean(scores[name]["train"]), np.std(scores[name]["train"])))
            results[name]["test"].append(
                (np.mean(scores[name]["test"]),  np.std(scores[name]["test"])))

    # Plot
    n_models = len(model_configs)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, model_configs):
        data = results[name]
        sizes = np.array(data["sizes"])
        tr_m  = np.array([s[0] for s in data["train"]])
        tr_s  = np.array([s[1] for s in data["train"]])
        te_m  = np.array([s[0] for s in data["test"]])
        te_s  = np.array([s[1] for s in data["test"]])

        ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.2)
        ax.fill_between(sizes, te_m - te_s, te_m + te_s, alpha=0.2)
        ax.plot(sizes, tr_m, "o-", label="Training", linewidth=2)
        ax.plot(sizes, te_m, "o-", label="Validation", linewidth=2)
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning Curve — {name} {title_suffix}", fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.close(fig)


def main():
    from src.model_registry import build_models as _build

    # Continental
    X, y_super, _, le, snp_ids, _ = load_continental()

    def make_xgb(X_tr, y_tr, le):
        m = _build(len(le.classes_))["XGBoost"]
        m.fit(X_tr, y_tr)
        return m

    def make_gen(X_tr, y_tr, le):
        m = GenerativeBGAModel(smoothing_alpha=1.0)
        m.fit(X_tr, le.inverse_transform(y_tr), snp_ids)
        return m

    plot_learning_curves(
        X, y_super, snp_ids,
        {"XGBoost": make_xgb, "Generative": make_gen},
        title_suffix="(Continental)",
        save_path=RESULTS_DIR / "learning_curves_continental.png",
    )

    # EAS
    X2, _, y_pop, le2, snp_ids2, _ = load_eas()
    from src.preprocessing import load_dataset, EAS_CSV
    X2, _, _, _, _, _ = load_dataset(EAS_CSV, verbose=False)

    def make_xgb2(X_tr, y_tr, le):
        m = _build(len(le.classes_))["XGBoost"]
        m.fit(X_tr, y_tr)
        return m

    def make_gen2(X_tr, y_tr, le):
        m = GenerativeBGAModel(smoothing_alpha=1.0)
        m.fit(X_tr, le.inverse_transform(y_tr), snp_ids2)
        return m

    plot_learning_curves(
        X2, y_pop, snp_ids2,
        {"XGBoost": make_xgb2, "Generative": make_gen2},
        title_suffix="(EAS only)",
        save_path=RESULTS_DIR / "learning_curves_eas.png",
    )


if __name__ == "__main__":
    main()
