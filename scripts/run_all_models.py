#!/usr/bin/env python
"""
Chạy toàn bộ XGBoost + Generative Bayesian Model + TabPFN cho hai tầng
(continental và East Asia), tổng hợp metric thành Excel và vẽ heatmap
confusion matrix.
"""
import os
import sys
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from src.data_utils import (
    load_continental_csv,
    load_eastasian_csv,
    encode_genotypes,
)
from src.xgboost_models import make_xgb_multiclass
from src.generative_model import GenerativeBGAModel
from src.tabpfn_model import make_tabpfn_classifier
from src.freeform_model import FreeformBGAClassifier

try:
    from src.mlp_geo_model import MLPGeoModel
    HAS_MLP_GEO = True
except ImportError:
    HAS_MLP_GEO = False

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


DATA_CONT_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")
DATA_EAS_PATH = os.path.join("data", "AISNP_by_sample_eastasian.csv")
OUTPUT_DIR = os.path.join("reports", "aggregated_results")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")
EXCEL_PATH = os.path.join(OUTPUT_DIR, "model_metrics.xlsx")


def slugify(text: str) -> str:
    return text.lower().replace(" ", "_")


def prepare_dataframe(
    loader: Callable[[str], pd.DataFrame],
    path: str,
    snp_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> tuple[pd.DataFrame, List[str]]:
    df_raw = loader(path)
    df_encoded, snp_names = encode_genotypes(df_raw)
    if snp_filter is not None:
        df_encoded = snp_filter(df_encoded)
    return df_encoded, snp_names


def compute_metrics(
    method: str,
    dataset: str,
    class_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    plot_path: str,
) -> Dict[str, float]:
    """Tính các metric và vẽ heatmap confusion matrix."""
    metrics: Dict[str, float] = {
        "dataset": dataset,
        "method": method,
    }

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    metrics["accuracy"] = acc
    metrics["mcc"] = mcc

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4,
    )

    for cls in class_names:
        metrics[f"f1_{cls}"] = report[cls]["f1-score"]

    if y_proba is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
            # roc_auc_score yêu cầu ít nhất 2 lớp
            if y_true_bin.shape[1] >= 2:
                auc_values = roc_auc_score(
                    y_true_bin,
                    y_proba,
                    average=None,
                    multi_class="ovr",
                )
                for cls, auc in zip(class_names, auc_values):
                    metrics[f"auc_{cls}"] = float(auc)
            else:
                metrics[f"auc_{class_names[0]}"] = float("nan")
        except ValueError:
            for cls in class_names:
                metrics[f"auc_{cls}"] = float("nan")
    else:
        for cls in class_names:
            metrics[f"auc_{cls}"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{dataset} - {method}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    metrics["confusion_matrix_plot"] = plot_path
    return metrics


def evaluate_xgboost(
    dataset: str,
    class_names: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
) -> Dict[str, float]:
    model = make_xgb_multiclass(num_classes=len(class_names))
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    plot_path = os.path.join(
        PLOTS_DIR, f"{slugify(dataset)}_xgboost_confusion.png"
    )
    return compute_metrics(
        method="XGBoost",
        dataset=dataset,
        class_names=class_names,
        y_true=y_test_enc,
        y_pred=y_pred,
        y_proba=y_proba,
        plot_path=plot_path,
    )


def evaluate_generative(
    dataset: str,
    class_names: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_str: np.ndarray,
    y_test_enc: np.ndarray,
    snp_names: List[str],
) -> Dict[str, float]:
    gen_model = GenerativeBGAModel(smoothing_alpha=1.0)
    gen_model.fit(X_train, y_train_str, snp_names=snp_names)

    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y_pred_labels = gen_model.predict(X_test)
    y_pred = np.array([label_to_idx[label] for label in y_pred_labels])

    proba = gen_model.predict_proba(X_test)
    reorder = [list(gen_model.pop_labels).index(cls) for cls in class_names]
    proba = proba[:, reorder]

    plot_path = os.path.join(
        PLOTS_DIR, f"{slugify(dataset)}_generative_confusion.png"
    )
    return compute_metrics(
        method="GenerativeBGA",
        dataset=dataset,
        class_names=class_names,
        y_true=y_test_enc,
        y_pred=y_pred,
        y_proba=proba,
        plot_path=plot_path,
    )


def evaluate_tabpfn(
    dataset: str,
    class_names: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
) -> Dict[str, float]:
    model = make_tabpfn_classifier(device="cpu")
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    plot_path = os.path.join(
        PLOTS_DIR, f"{slugify(dataset)}_tabpfn_confusion.png"
    )
    return compute_metrics(
        method="TabPFN",
        dataset=dataset,
        class_names=class_names,
        y_true=y_test_enc,
        y_pred=y_pred,
        y_proba=y_proba,
        plot_path=plot_path,
    )


def evaluate_freeform(
    dataset: str,
    class_names: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
    snp_names: List[str],
) -> Dict[str, float]:
    model = FreeformBGAClassifier()
    model.fit(X_train, y_train_enc, snp_names=snp_names)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    plot_path = os.path.join(
        PLOTS_DIR, f"{slugify(dataset)}_freeform_confusion.png"
    )
    return compute_metrics(
        method="FREEFORM",
        dataset=dataset,
        class_names=class_names,
        y_true=y_test_enc,
        y_pred=y_pred,
        y_proba=y_proba,
        plot_path=plot_path,
    )


def evaluate_mlp_geo(
    dataset: str,
    class_names: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
    pop_train: np.ndarray,
) -> Dict[str, float]:
    model = MLPGeoModel()
    model.fit(X_train, y_train_enc, pop_labels=pop_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    plot_path = os.path.join(
        PLOTS_DIR, f"{slugify(dataset)}_mlp_geo_confusion.png"
    )
    return compute_metrics(
        method="MLP-Geo",
        dataset=dataset,
        class_names=class_names,
        y_true=y_test_enc,
        y_pred=y_pred,
        y_proba=y_proba,
        plot_path=plot_path,
    )


def evaluate_ga_svm(
    dataset: str,
    class_names: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
    snp_names: List[str],
) -> Dict[str, float]:
    model = GASVMClassifier(
        pop_size=50, n_generations=40, tournament_size=3, mutation_prob=0.03,
    )
    model.fit(X_train, y_train_enc, snp_names=snp_names)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    plot_path = os.path.join(
        PLOTS_DIR, f"{slugify(dataset)}_ga_svm_confusion.png"
    )
    return compute_metrics(
        method="GA-SVM",
        dataset=dataset,
        class_names=class_names,
        y_true=y_test_enc,
        y_pred=y_pred,
        y_proba=y_proba,
        plot_path=plot_path,
    )


def evaluate_svd_mlp_adv(
    dataset: str,
    class_names: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
) -> Dict[str, float]:
    model = SVDMLPAdvClassifier(
        n_components=20, hidden_sizes=(128, 64),
        epsilon=0.05, alpha=0.3, epochs=200, patience=20,
    )
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    plot_path = os.path.join(
        PLOTS_DIR, f"{slugify(dataset)}_svd_mlp_adv_confusion.png"
    )
    return compute_metrics(
        method="SVD-MLP-Adv",
        dataset=dataset,
        class_names=class_names,
        y_true=y_test_enc,
        y_pred=y_pred,
        y_proba=y_proba,
        plot_path=plot_path,
    )


def evaluate_diet_networks(
    dataset: str,
    class_names: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
) -> Dict[str, float]:
    model = DietNetworkClassifier(
        embed_dim=64, aux_hidden=64, clf_hidden=32,
        epochs=200, patience=20,
    )
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    plot_path = os.path.join(
        PLOTS_DIR, f"{slugify(dataset)}_diet_networks_confusion.png"
    )
    return compute_metrics(
        method="DietNetworks",
        dataset=dataset,
        class_names=class_names,
        y_true=y_test_enc,
        y_pred=y_pred,
        y_proba=y_proba,
        plot_path=plot_path,
    )


def evaluate_popvae(
    dataset: str,
    class_names: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
) -> Dict[str, float]:
    model = PopVAEClassifier(
        latent_dim=10, enc_hidden=(128, 64),
        beta=1.0, gamma=10.0, epochs=200, patience=20,
    )
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    plot_path = os.path.join(
        PLOTS_DIR, f"{slugify(dataset)}_popvae_confusion.png"
    )
    return compute_metrics(
        method="popVAE",
        dataset=dataset,
        class_names=class_names,
        y_true=y_test_enc,
        y_pred=y_pred,
        y_proba=y_proba,
        plot_path=plot_path,
    )


def evaluate_federated_mlp(
    dataset: str,
    class_names: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
) -> Dict[str, float]:
    model = FederatedMLPClassifier(
        n_clients=5, hidden_sizes=(128, 64),
        n_rounds=20, local_epochs=5, patience=8,
    )
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    plot_path = os.path.join(
        PLOTS_DIR, f"{slugify(dataset)}_federated_mlp_confusion.png"
    )
    return compute_metrics(
        method="FederatedMLP",
        dataset=dataset,
        class_names=class_names,
        y_true=y_test_enc,
        y_pred=y_pred,
        y_proba=y_proba,
        plot_path=plot_path,
    )


def run_dataset(
    dataset_name: str,
    loader: Callable[[str], pd.DataFrame],
    path: str,
    label_col: str,
    snp_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> List[Dict[str, float]]:
    df_encoded, snp_names = prepare_dataframe(loader, path, snp_filter)
    X = df_encoded[snp_names].values.astype(float)
    y = df_encoded[label_col].values
    pop_labels = df_encoded["pop"].values

    X_train, X_test, y_train, y_test, pop_train, pop_test = train_test_split(
        X,
        y,
        pop_labels,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    le = LabelEncoder()
    le.fit(y)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    class_names = list(le.classes_)

    results = []
    results.append(
        evaluate_xgboost(
            dataset=dataset_name,
            class_names=class_names,
            X_train=X_train,
            X_test=X_test,
            y_train_enc=y_train_enc,
            y_test_enc=y_test_enc,
        )
    )
    results.append(
        evaluate_generative(
            dataset=dataset_name,
            class_names=class_names,
            X_train=X_train,
            X_test=X_test,
            y_train_str=y_train,
            y_test_enc=y_test_enc,
            snp_names=snp_names,
        )
    )
    results.append(
        evaluate_tabpfn(
            dataset=dataset_name,
            class_names=class_names,
            X_train=X_train,
            X_test=X_test,
            y_train_enc=y_train_enc,
            y_test_enc=y_test_enc,
        )
    )

    # FREEFORM model
    try:
        results.append(
            evaluate_freeform(
                dataset=dataset_name,
                class_names=class_names,
                X_train=X_train,
                X_test=X_test,
                y_train_enc=y_train_enc,
                y_test_enc=y_test_enc,
                snp_names=snp_names,
            )
        )
    except Exception as e:
        print(f"[WARN] FREEFORM skipped for {dataset_name}: {e}")

    # MLP-Geo model
    if HAS_MLP_GEO:
        try:
            results.append(
                evaluate_mlp_geo(
                    dataset=dataset_name,
                    class_names=class_names,
                    X_train=X_train,
                    X_test=X_test,
                    y_train_enc=y_train_enc,
                    y_test_enc=y_test_enc,
                    pop_train=pop_train,
                )
            )
        except Exception as e:
            print(f"[WARN] MLP-Geo skipped for {dataset_name}: {e}")
    else:
        print(f"[WARN] MLP-Geo skipped for {dataset_name}: PyTorch not installed")

    # GA-SVM
    try:
        results.append(
            evaluate_ga_svm(
                dataset=dataset_name,
                class_names=class_names,
                X_train=X_train,
                X_test=X_test,
                y_train_enc=y_train_enc,
                y_test_enc=y_test_enc,
                snp_names=snp_names,
            )
        )
    except Exception as e:
        print(f"[WARN] GA-SVM skipped for {dataset_name}: {e}")

    # SVD-MLP-Adv
    if HAS_SVD_MLP:
        try:
            results.append(
                evaluate_svd_mlp_adv(
                    dataset=dataset_name,
                    class_names=class_names,
                    X_train=X_train,
                    X_test=X_test,
                    y_train_enc=y_train_enc,
                    y_test_enc=y_test_enc,
                )
            )
        except Exception as e:
            print(f"[WARN] SVD-MLP-Adv skipped for {dataset_name}: {e}")
    else:
        print(f"[WARN] SVD-MLP-Adv skipped for {dataset_name}: PyTorch not installed")

    # Diet Networks
    if HAS_DIET:
        try:
            results.append(
                evaluate_diet_networks(
                    dataset=dataset_name,
                    class_names=class_names,
                    X_train=X_train,
                    X_test=X_test,
                    y_train_enc=y_train_enc,
                    y_test_enc=y_test_enc,
                )
            )
        except Exception as e:
            print(f"[WARN] DietNetworks skipped for {dataset_name}: {e}")
    else:
        print(f"[WARN] DietNetworks skipped for {dataset_name}: PyTorch not installed")

    # popVAE
    if HAS_POPVAE:
        try:
            results.append(
                evaluate_popvae(
                    dataset=dataset_name,
                    class_names=class_names,
                    X_train=X_train,
                    X_test=X_test,
                    y_train_enc=y_train_enc,
                    y_test_enc=y_test_enc,
                )
            )
        except Exception as e:
            print(f"[WARN] popVAE skipped for {dataset_name}: {e}")
    else:
        print(f"[WARN] popVAE skipped for {dataset_name}: PyTorch not installed")

    # Federated MLP
    if HAS_FED:
        try:
            results.append(
                evaluate_federated_mlp(
                    dataset=dataset_name,
                    class_names=class_names,
                    X_train=X_train,
                    X_test=X_test,
                    y_train_enc=y_train_enc,
                    y_test_enc=y_test_enc,
                )
            )
        except Exception as e:
            print(f"[WARN] FederatedMLP skipped for {dataset_name}: {e}")
    else:
        print(f"[WARN] FederatedMLP skipped for {dataset_name}: PyTorch not installed")

    return results


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_results: List[Dict[str, float]] = []

    all_results.extend(
        run_dataset(
            dataset_name="Continental",
            loader=load_continental_csv,
            path=DATA_CONT_PATH,
            label_col="super_pop",
        )
    )

    all_results.extend(
        run_dataset(
            dataset_name="East Asia",
            loader=load_eastasian_csv,
            path=DATA_EAS_PATH,
            label_col="pop",
            snp_filter=lambda df: df[df["super_pop"] == "EAS"].copy(),
        )
    )

    df_metrics = pd.DataFrame(all_results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_metrics.to_excel(EXCEL_PATH, index=False)

    print("Saved aggregated metrics to:", EXCEL_PATH)
    print("Saved confusion matrix heatmaps to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
