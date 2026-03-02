# scripts/train_eas_only.py
"""
Train XGBoost trên file AISNP_by_sample_EAS_only.csv (da loc san EAS)
Su dung tham so tu paper: max_depth=7, learning_rate=0.1, n_estimators=200
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

from src.data_utils import encode_genotypes, split_xy
from src.models import make_xgb_multiclass

# Path to new EAS-only data
DATA_PATH = os.path.join("data", "AISNP_by_sample_EAS_only.csv")

# Paper parameters for East Asian
PAPER_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 7,
    "n_estimators": 200,
}


def load_eas_only_csv(path: str) -> pd.DataFrame:
    """Load EAS-only CSV file"""
    df = pd.read_csv(path)
    return df


def train_and_evaluate():
    """Train va evaluate XGBoost tren EAS-only data"""
    
    print("=" * 70)
    print("TRAINING XGBOOST ON EAS-ONLY DATA")
    print("=" * 70)
    print(f"\nData file: {DATA_PATH}")
    print(f"Paper parameters: {PAPER_PARAMS}")
    
    # Load data
    df_raw = load_eas_only_csv(DATA_PATH)
    print(f"\nLoaded EAS-only data: {df_raw.shape}")
    
    # Check super_pop distribution
    print(f"\nSuper population distribution:")
    print(df_raw["super_pop"].value_counts())
    
    # Check population distribution
    print(f"\nPopulation distribution:")
    print(df_raw["pop"].value_counts())
    
    # Encode genotypes
    df_encoded, snp_names = encode_genotypes(df_raw)
    print(f"\nEncoded data shape: {df_encoded.shape}")
    print(f"Number of SNPs: {len(snp_names)}")
    
    # Split X, y
    X, y = split_xy(df_encoded, snp_names, label_col="pop")
    
    # Encode labels
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
        X, y_int, test_size=0.2, random_state=42, stratify=y_int
    )
    
    print(f"\nTrain size: {len(y_train)}")
    print(f"Test size: {len(y_test)}")
    
    # Train with paper params
    model = make_xgb_multiclass(num_classes=num_classes, **PAPER_PARAMS)
    model.fit(X_train, y_train)
    
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
    print(confusion_matrix(y_test, y_pred))
    
    # =========================================================================
    # Method 2: 5-Fold Cross Validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 2: 5-FOLD CROSS VALIDATION")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create fresh model for CV
    cv_model = make_xgb_multiclass(num_classes=num_classes, **PAPER_PARAMS)
    
    cv_scores = cross_val_score(cv_model, X, y_int, cv=cv, scoring='accuracy')
    cv_f1_scores = cross_val_score(cv_model, X, y_int, cv=cv, scoring='f1_macro')
    
    print(f"\nCV Accuracy scores: {cv_scores}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    print(f"\nCV F1 Macro scores: {cv_f1_scores}")
    print(f"CV F1 Macro: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std()*2:.4f})")
    
    # =========================================================================
    # Train final model on full data
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON FULL DATA")
    print("=" * 70)
    
    final_model = make_xgb_multiclass(num_classes=num_classes, **PAPER_PARAMS)
    final_model.fit(X, y_int)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, os.path.join("models", "eastasia_xgb_eas_only.pkl"))
    joblib.dump(le, os.path.join("models", "eastasia_label_encoder_eas_only.pkl"))
    joblib.dump(snp_names, os.path.join("models", "eastasia_snp_names_eas_only.pkl"))
    
    print("\n[OK] Saved models to:")
    print("  - models/eastasia_xgb_eas_only.pkl")
    print("  - models/eastasia_label_encoder_eas_only.pkl")
    print("  - models/eastasia_snp_names_eas_only.pkl")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nDataset: EAS-only ({len(y)} samples, {num_classes} classes)")
    print(f"SNPs: {len(snp_names)}")
    print(f"\nPaper Parameters:")
    for k, v in PAPER_PARAMS.items():
        print(f"  {k}: {v}")
    
    print(f"\nResults:")
    print(f"  Single Split Test Accuracy: {test_acc:.4f}")
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"  5-Fold CV F1 Macro: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std()*2:.4f})")
    
    return {
        "test_acc": test_acc,
        "cv_acc_mean": cv_scores.mean(),
        "cv_acc_std": cv_scores.std(),
        "cv_f1_mean": cv_f1_scores.mean(),
        "cv_f1_std": cv_f1_scores.std(),
    }


if __name__ == "__main__":
    train_and_evaluate()

