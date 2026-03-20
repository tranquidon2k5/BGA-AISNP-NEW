"""
src/training.py
===============
Unified training engine.

Trains all registered models (from model_registry) on a given dataset,
with RandomizedSearchCV hyperparameter tuning and cross-validation.

Usage:
    from src.training import train_all
    results = train_all(X, y, target_name="super_pop")
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split,
)

from src.model_registry import (
    build_models,
    get_param_grids,
    tune_model,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    N_ITER_SEARCH,
)


def train_all(
    X: np.ndarray,
    y: np.ndarray,
    target_name: str,
    model_names: list[str] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Train (with tuning) a suite of models on the given data.

    Parameters
    ----------
    X           : feature matrix  (n_samples, n_features)
    y           : label vector    (integer-encoded)
    target_name : human label, e.g. "super_pop" or "pop"
    model_names : subset of models to train (default: all)
    verbose     : print progress

    Returns
    -------
    results : list[dict] — one entry per model, containing:
        model_name, target, fitted_model,
        X_test, y_test,
        cv_accuracy_mean, cv_accuracy_std,
        cv_f1_mean, cv_f1_std,
        cv_balanced_acc_mean, cv_balanced_acc_std,
        best_params, fit_time_sec, n_classes
    """
    n_classes   = len(np.unique(y))
    base_models = build_models(n_classes)
    param_grids = get_param_grids(n_classes)

    if model_names is not None:
        base_models = {k: v for k, v in base_models.items() if k in model_names}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    # DataFrame wrapper so LightGBM gets valid feature names
    n_feats   = X_train.shape[1]
    feat_cols = [f"snp_{i}" for i in range(n_feats)]
    X_train_df = pd.DataFrame(X_train, columns=feat_cols)
    X_test_df  = pd.DataFrame(X_test,  columns=feat_cols)

    skf     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
    scoring = ["accuracy", "f1_macro", "balanced_accuracy"]

    results = []
    for name, base_model in base_models.items():
        if verbose:
            print(f"  Tuning {name} for target={target_name} "
                  f"(n_iter={N_ITER_SEARCH}) …", end=" ", flush=True)
        t0 = time.time()

        param_dist = param_grids.get(name, {})

        # ── Tune ──────────────────────────────────────────────────────────
        best_model, best_params, best_tune_score = tune_model(
            name, base_model, param_dist, X_train_df, y_train, skf,
            verbose=verbose,
        )

        # ── CV evaluation ─────────────────────────────────────────────────
        try:
            cv_res = cross_validate(
                best_model, X_train_df, y_train,
                cv=skf, scoring=scoring, n_jobs=1,
            )
            cv_acc_mean  = float(cv_res["test_accuracy"].mean())
            cv_acc_std   = float(cv_res["test_accuracy"].std())
            cv_f1_mean   = float(cv_res["test_f1_macro"].mean())
            cv_f1_std    = float(cv_res["test_f1_macro"].std())
            cv_bacc_mean = float(cv_res["test_balanced_accuracy"].mean())
            cv_bacc_std  = float(cv_res["test_balanced_accuracy"].std())
        except Exception as cv_err:
            # Some estimators (e.g. CatBoost) can't be cloned by sklearn
            if verbose:
                print(f"[CV fallback: {cv_err.__class__.__name__}]", end=" ")
            cv_acc_mean = cv_acc_std = 0.0
            cv_f1_mean = cv_f1_std = 0.0
            cv_bacc_mean = best_tune_score
            cv_bacc_std = 0.0

        elapsed = time.time() - t0

        if verbose:
            print(f"done in {elapsed:.1f}s  |  CV acc={cv_acc_mean:.4f}  "
                  f"[tune bal_acc={best_tune_score:.4f}]")

        results.append({
            "model_name":           name,
            "target":               target_name,
            "fitted_model":         best_model,
            "X_test":               X_test_df,
            "y_test":               y_test,
            "cv_accuracy_mean":     cv_acc_mean,
            "cv_accuracy_std":      cv_acc_std,
            "cv_f1_mean":           cv_f1_mean,
            "cv_f1_std":            cv_f1_std,
            "cv_balanced_acc_mean": cv_bacc_mean,
            "cv_balanced_acc_std":  cv_bacc_std,
            "best_params":          best_params,
            "fit_time_sec":         elapsed,
            "n_classes":            n_classes,
        })

    return results
