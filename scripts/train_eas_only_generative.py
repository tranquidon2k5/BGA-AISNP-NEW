# scripts/train_eas_only_generative.py
"""
Train Generative model tren file AISNP_by_sample_EAS_only.csv
So sanh voi XGBoost
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

from src.data_utils import encode_genotypes, split_xy
from src.generative_model import GenerativeBGAModel

# Path to new EAS-only data
DATA_PATH = os.path.join("data", "AISNP_by_sample_EAS_only.csv")


def load_eas_only_csv(path: str) -> pd.DataFrame:
    """Load EAS-only CSV file"""
    df = pd.read_csv(path)
    return df


def train_and_evaluate():
    """Train va evaluate Generative model tren EAS-only data"""
    
    print("=" * 70)
    print("TRAINING GENERATIVE MODEL ON EAS-ONLY DATA")
    print("=" * 70)
    print(f"\nData file: {DATA_PATH}")
    
    # Load data
    df_raw = load_eas_only_csv(DATA_PATH)
    print(f"\nLoaded EAS-only data: {df_raw.shape}")
    
    # Check population distribution
    print(f"\nPopulation distribution:")
    print(df_raw["pop"].value_counts())
    
    # Encode genotypes
    df_encoded, snp_names = encode_genotypes(df_raw)
    print(f"\nEncoded data shape: {df_encoded.shape}")
    print(f"Number of SNPs: {len(snp_names)}")
    
    # Split X, y
    X, y = split_xy(df_encoded, snp_names, label_col="pop")
    X = X.astype(float)
    
    # Encode labels for evaluation
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    print(f"\nClasses: {le.classes_}")
    print(f"Number of classes: {num_classes}")
    
    # =========================================================================
    # Method 1: Single train/test split (80/20)
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 1: SINGLE TRAIN/TEST SPLIT (80/20)")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {len(y_train)}")
    print(f"Test size: {len(y_test)}")
    
    # Train Generative model
    model = GenerativeBGAModel(smoothing_alpha=1.0)
    model.fit(X_train, y_train, snp_names)
    
    # Evaluate on train
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # Evaluate on test
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Gap (overfit indicator): {train_acc - test_acc:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
    
    print("Confusion Matrix (Test Set):")
    # Convert to integer encoding for confusion matrix
    y_test_int = le.transform(y_test)
    y_pred_int = le.transform(y_pred)
    print(confusion_matrix(y_test_int, y_pred_int))
    
    # =========================================================================
    # Method 2: 5-Fold Cross Validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 2: 5-FOLD CROSS VALIDATION")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
        
        fold_model = GenerativeBGAModel(smoothing_alpha=1.0)
        fold_model.fit(X_train_cv, y_train_cv, snp_names)
        
        y_pred_cv = fold_model.predict(X_test_cv)
        fold_acc = accuracy_score(y_test_cv, y_pred_cv)
        cv_scores.append(fold_acc)
    
    cv_scores = np.array(cv_scores)
    
    print(f"\nCV Accuracy scores: {cv_scores}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # =========================================================================
    # Train final model on full data
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON FULL DATA")
    print("=" * 70)
    
    final_model = GenerativeBGAModel(smoothing_alpha=1.0)
    final_model.fit(X, y, snp_names)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, os.path.join("models", "eastasia_gen_eas_only.pkl"))
    
    print("\n[OK] Saved model to:")
    print("  - models/eastasia_gen_eas_only.pkl")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY - GENERATIVE MODEL")
    print("=" * 70)
    print(f"\nDataset: EAS-only ({len(y)} samples, {num_classes} classes)")
    print(f"SNPs: {len(snp_names)}")
    print(f"Smoothing alpha: 1.0")
    
    print(f"\nResults:")
    print(f"  Single Split Test Accuracy: {test_acc:.4f}")
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return {
        "test_acc": test_acc,
        "cv_acc_mean": cv_scores.mean(),
        "cv_acc_std": cv_scores.std(),
    }


if __name__ == "__main__":
    train_and_evaluate()

