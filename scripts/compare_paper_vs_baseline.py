# scripts/compare_paper_vs_baseline.py
"""
So sánh chi tiết giữa models với tham số từ paper và baseline
"""

import os
import sys
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import (
    load_continental_csv,
    load_eastasian_csv,
    encode_genotypes,
    split_xy,
    stratified_train_test,
    encode_labels,
)


def evaluate_model(model, X_test, y_test, le, title):
    """Evaluate model và trả về metrics"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
    
    return acc, y_pred


def compare_continental():
    """So sánh Continental models"""
    print("\n" + "="*70)
    print("CONTINENTAL MODEL COMPARISON")
    print("="*70)
    
    # Load data
    DATA_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")
    df_raw = load_continental_csv(DATA_PATH)
    df_encoded, snp_names = encode_genotypes(df_raw)
    X, y = split_xy(df_encoded, snp_names, label_col="super_pop")
    y_int, le = encode_labels(y)
    X_train, X_test, y_train, y_test = stratified_train_test(X, y_int)
    
    results = {}
    
    # Baseline model
    try:
        baseline_model = joblib.load("models/continent_xgb.pkl")
        acc_baseline, _ = evaluate_model(
            baseline_model, X_test, y_test, le, 
            "Baseline Model (max_depth=4)"
        )
        results['baseline'] = acc_baseline
        print(f"\nBaseline Parameters:")
        print(f"  max_depth: {baseline_model.max_depth}")
        print(f"  learning_rate: {baseline_model.learning_rate}")
        print(f"  n_estimators: {baseline_model.n_estimators}")
    except FileNotFoundError:
        print("\n[WARNING] Baseline model not found")
        results['baseline'] = None
    
    # Paper params model
    try:
        paper_model = joblib.load("models/continent_xgb_paper.pkl")
        acc_paper, _ = evaluate_model(
            paper_model, X_test, y_test, le,
            "Paper Parameters Model (max_depth=5)"
        )
        results['paper'] = acc_paper
        print(f"\nPaper Parameters:")
        print(f"  max_depth: {paper_model.max_depth}")
        print(f"  learning_rate: {paper_model.learning_rate}")
        print(f"  n_estimators: {paper_model.n_estimators}")
    except FileNotFoundError:
        print("\n[WARNING] Paper params model not found")
        results['paper'] = None
    
    # Summary
    print(f"\n{'='*70}")
    print("CONTINENTAL SUMMARY")
    print(f"{'='*70}")
    if results['baseline'] and results['paper']:
        diff = results['paper'] - results['baseline']
        print(f"Baseline accuracy:  {results['baseline']:.4f}")
        print(f"Paper params accuracy: {results['paper']:.4f}")
        print(f"Difference: {diff:+.4f} ({diff*100:+.2f}%)")
        if diff > 0:
            print("[OK] Paper parameters perform BETTER")
        elif diff < 0:
            print("[INFO] Baseline performs better")
        else:
            print("[INFO] Similar performance")
    
    return results


def compare_eastasian():
    """So sánh East Asian models"""
    print("\n" + "="*70)
    print("EAST ASIAN MODEL COMPARISON")
    print("="*70)
    
    # Load data
    DATA_PATH = os.path.join("data", "AISNP_by_sample_eastasian.csv")
    df_raw = load_eastasian_csv(DATA_PATH)
    df_encoded, snp_names = encode_genotypes(df_raw)
    df_eas = df_encoded[df_encoded["super_pop"] == "EAS"].copy()
    X, y = split_xy(df_eas, snp_names, label_col="pop")
    y_int, le = encode_labels(y)
    X_train, X_test, y_train, y_test = stratified_train_test(X, y_int)
    
    results = {}
    
    # Baseline model
    try:
        baseline_model = joblib.load("models/eastasia_xgb_baseline.pkl")
        acc_baseline, _ = evaluate_model(
            baseline_model, X_test, y_test, le,
            "Baseline Model (max_depth=4)"
        )
        results['baseline'] = acc_baseline
        print(f"\nBaseline Parameters:")
        print(f"  max_depth: {baseline_model.max_depth}")
        print(f"  learning_rate: {baseline_model.learning_rate}")
        print(f"  n_estimators: {baseline_model.n_estimators}")
    except FileNotFoundError:
        print("\n[WARNING] Baseline model not found")
        results['baseline'] = None
    
    # Paper params model
    try:
        paper_model = joblib.load("models/eastasia_xgb_paper.pkl")
        acc_paper, _ = evaluate_model(
            paper_model, X_test, y_test, le,
            "Paper Parameters Model (max_depth=7)"
        )
        results['paper'] = acc_paper
        print(f"\nPaper Parameters:")
        print(f"  max_depth: {paper_model.max_depth}")
        print(f"  learning_rate: {paper_model.learning_rate}")
        print(f"  n_estimators: {paper_model.n_estimators}")
    except FileNotFoundError:
        print("\n[WARNING] Paper params model not found")
        results['paper'] = None
    
    # Summary
    print(f"\n{'='*70}")
    print("EAST ASIAN SUMMARY")
    print(f"{'='*70}")
    if results['baseline'] and results['paper']:
        diff = results['paper'] - results['baseline']
        print(f"Baseline accuracy:  {results['baseline']:.4f}")
        print(f"Paper params accuracy: {results['paper']:.4f}")
        print(f"Difference: {diff:+.4f} ({diff*100:+.2f}%)")
        if diff > 0.1:
            print("[OK] Paper parameters perform MUCH BETTER!")
        elif diff > 0:
            print("[OK] Paper parameters perform BETTER")
        elif diff < 0:
            print("[INFO] Baseline performs better")
        else:
            print("[INFO] Similar performance")
    
    return results


def main():
    """Main comparison function"""
    print("="*70)
    print("COMPARING PAPER PARAMETERS vs BASELINE")
    print("="*70)
    print("\nPaper Parameters:")
    print("  Continental: max_depth=5, learning_rate=0.1, n_estimators=200")
    print("  East Asian: max_depth=7, learning_rate=0.1, n_estimators=200")
    print("\nBaseline Parameters:")
    print("  Both: max_depth=4, learning_rate=0.1, n_estimators=200")
    
    # Compare Continental
    cont_results = compare_continental()
    
    # Compare East Asian
    eas_results = compare_eastasian()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print("\nContinental:")
    if cont_results.get('baseline') and cont_results.get('paper'):
        print(f"  Baseline: {cont_results['baseline']:.4f}")
        print(f"  Paper:    {cont_results['paper']:.4f}")
        print(f"  Change:   {cont_results['paper'] - cont_results['baseline']:+.4f}")
    
    print("\nEast Asian:")
    if eas_results.get('baseline') and eas_results.get('paper'):
        print(f"  Baseline: {eas_results['baseline']:.4f}")
        print(f"  Paper:    {eas_results['paper']:.4f}")
        print(f"  Change:   {eas_results['paper'] - eas_results['baseline']:+.4f}")


if __name__ == "__main__":
    main()

