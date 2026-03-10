#!/usr/bin/env python
"""
Standalone runner for the 5 classic comparison models:
  1. GA-SVM
  2. SVD-MLP-Adv
  3. Diet Networks
  4. popVAE
  5. Federated MLP

Trains each model on both stages (Continental + East Asian),
computes ACC, MCC, F1, ROC-AUC, saves results CSV and confusion matrices.
"""

import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from src.preprocessing import load_continental, load_eas

# Import classic models
from src.ga_svm_model import GASVMClassifier

try:
    from src.svd_mlp_adv_model import SVDMLPAdvClassifier
    HAS_SVD_MLP = True
except ImportError:
    HAS_SVD_MLP = False

try:
    from src.diet_networks_model import DietNetworkClassifier
    HAS_DIET = True
except ImportError:
    HAS_DIET = False

try:
    from src.popvae_model import PopVAEClassifier
    HAS_POPVAE = True
except ImportError:
    HAS_POPVAE = False

try:
    from src.federated_mlp_model import FederatedMLPClassifier
    HAS_FED = True
except ImportError:
    HAS_FED = False


OUTPUT_DIR = os.path.join("results", "classic_models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")


def slugify(text: str) -> str:
    return text.lower().replace(" ", "_")


def compute_metrics(
    method: str,
    dataset: str,
    class_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    fit_time: float,
    plot_path: str,
) -> Dict:
    """Compute metrics and save confusion matrix heatmap."""
    metrics = {
        "dataset": dataset,
        "method": method,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "fit_time_sec": fit_time,
    }

    # Per-class F1
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, digits=4
    )
    for cls in class_names:
        metrics[f"f1_{cls}"] = report[cls]["f1-score"]

    # ROC-AUC
    if y_proba is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
            if y_true_bin.shape[1] >= 2:
                auc_values = roc_auc_score(
                    y_true_bin, y_proba, average=None, multi_class="ovr"
                )
                metrics["roc_auc_macro"] = float(np.mean(auc_values))
                for cls, auc in zip(class_names, auc_values):
                    metrics[f"auc_{cls}"] = float(auc)
        except ValueError:
            metrics["roc_auc_macro"] = float("nan")
    else:
        metrics["roc_auc_macro"] = float("nan")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f"{dataset} - {method}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return metrics


def train_and_evaluate(model, model_name, dataset_name, class_names,
                       X_train, X_test, y_train, y_test, snp_names=None):
    """Train a model and compute all metrics."""
    print(f"  Training {model_name}...", end=" ", flush=True)
    t0 = time.time()
    model.fit(X_train, y_train, snp_names=snp_names)
    fit_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    plot_path = os.path.join(
        PLOTS_DIR, f"{slugify(dataset_name)}_{slugify(model_name)}_confusion.png"
    )

    metrics = compute_metrics(
        method=model_name,
        dataset=dataset_name,
        class_names=class_names,
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        fit_time=fit_time,
        plot_path=plot_path,
    )
    print(f"ACC={metrics['accuracy']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"({fit_time:.1f}s)")
    return metrics


def run_stage(stage_name, X, y, snp_ids):
    """Run all classic models on a single dataset stage."""
    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]) if isinstance(y_train[0], str) else y)
    if isinstance(y_train[0], str):
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
    else:
        y_train_enc = y_train
        y_test_enc = y_test
    class_names = list(le.classes_) if hasattr(le.classes_[0], '__len__') else [str(c) for c in le.classes_]

    results = []

    # 1. GA-SVM
    try:
        model = GASVMClassifier(
            pop_size=50, n_generations=40, tournament_size=3,
            mutation_prob=0.03, svm_C=10.0, cv_folds=3,
        )
        metrics = train_and_evaluate(
            model, "GA-SVM", stage_name, class_names,
            X_train, X_test, y_train_enc, y_test_enc, snp_names=snp_ids,
        )
        results.append(metrics)
    except Exception as e:
        print(f"  [WARN] GA-SVM skipped: {e}")

    # 2. SVD-MLP-Adv
    if HAS_SVD_MLP:
        try:
            model = SVDMLPAdvClassifier(
                n_components=20, hidden_sizes=(128, 64),
                epsilon=0.05, alpha=0.3, epochs=200, patience=20,
            )
            metrics = train_and_evaluate(
                model, "SVD-MLP-Adv", stage_name, class_names,
                X_train, X_test, y_train_enc, y_test_enc,
            )
            results.append(metrics)
        except Exception as e:
            print(f"  [WARN] SVD-MLP-Adv skipped: {e}")
    else:
        print("  [WARN] SVD-MLP-Adv skipped: PyTorch not installed")

    # 3. Diet Networks
    if HAS_DIET:
        try:
            model = DietNetworkClassifier(
                embed_dim=64, aux_hidden=64, clf_hidden=32,
                epochs=200, patience=20,
            )
            metrics = train_and_evaluate(
                model, "DietNetworks", stage_name, class_names,
                X_train, X_test, y_train_enc, y_test_enc,
            )
            results.append(metrics)
        except Exception as e:
            print(f"  [WARN] DietNetworks skipped: {e}")
    else:
        print("  [WARN] DietNetworks skipped: PyTorch not installed")

    # 4. popVAE
    if HAS_POPVAE:
        try:
            model = PopVAEClassifier(
                latent_dim=10, enc_hidden=(128, 64),
                beta=1.0, gamma=10.0, epochs=200, patience=20,
            )
            metrics = train_and_evaluate(
                model, "popVAE", stage_name, class_names,
                X_train, X_test, y_train_enc, y_test_enc,
            )
            results.append(metrics)
        except Exception as e:
            print(f"  [WARN] popVAE skipped: {e}")
    else:
        print("  [WARN] popVAE skipped: PyTorch not installed")

    # 5. Federated MLP
    if HAS_FED:
        try:
            model = FederatedMLPClassifier(
                n_clients=5, hidden_sizes=(128, 64),
                n_rounds=20, local_epochs=5, patience=8,
            )
            metrics = train_and_evaluate(
                model, "FederatedMLP", stage_name, class_names,
                X_train, X_test, y_train_enc, y_test_enc,
            )
            results.append(metrics)
        except Exception as e:
            print(f"  [WARN] FederatedMLP skipped: {e}")
    else:
        print("  [WARN] FederatedMLP skipped: PyTorch not installed")

    return results


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_results = []

    # Stage 1: Continental super-populations
    print("Loading Continental dataset...")
    X1, y_super, y_pop, le_dict, snp_ids, df_meta = load_continental()
    all_results.extend(run_stage("Continental (super_pop)", X1, y_super, snp_ids))

    # Stage 2: East Asian sub-populations
    print("\nLoading East Asian dataset...")
    X2, y_super2, y_pop2, le_dict2, snp_ids2, df_meta2 = load_eas()
    all_results.extend(run_stage("EastAsia (pop)", X2, y_pop2, snp_ids2))

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)

        # Print summary table
        print(f"\n{'='*60}")
        print("  RESULTS SUMMARY")
        print(f"{'='*60}")
        summary_cols = ["dataset", "method", "accuracy", "mcc", "f1_macro",
                        "balanced_accuracy", "roc_auc_macro", "fit_time_sec"]
        available_cols = [c for c in summary_cols if c in df.columns]
        print(df[available_cols].to_string(index=False, float_format="%.4f"))

        # Save CSV
        csv_path = os.path.join(OUTPUT_DIR, "classic_models_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        print(f"Confusion matrices saved to: {PLOTS_DIR}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
