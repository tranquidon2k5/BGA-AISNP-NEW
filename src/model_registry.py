"""
src/model_registry.py
=====================
Unified model registry for the BGA-AISNP project.

This module is the SINGLE source of truth for all classifier definitions,
hyperparameter search spaces, and the NGBoost wrapper.

To add a new model:
    1. Add its default constructor in build_models()
    2. Add its param grid in get_param_grids()
    3. (Optional) If it needs a custom sklearn wrapper, add it here.
    4. Update structure.md

See structure.md for full instructions.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import randint, uniform, loguniform

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.base           import BaseEstimator, ClassifierMixin
from xgboost                import XGBClassifier
from lightgbm               import LGBMClassifier
from catboost               import CatBoostClassifier
from ngboost                import NGBClassifier
from ngboost.distns          import k_categorical
from src.generative_model   import GenerativeBGAModel


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5
N_ITER_SEARCH = 20   # RandomizedSearchCV iterations

# All available model keys (for CLI help / validation)
ALL_MODEL_NAMES = [
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "NGBoost",
    "GenerativeNaiveBayes",
]


def list_available_models() -> list[str]:
    """Return the list of registered model names."""
    return list(ALL_MODEL_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# NGBoost sklearn-compatible wrapper
# ─────────────────────────────────────────────────────────────────────────────

class NGBoostWrapper(BaseEstimator, ClassifierMixin):
    """Thin sklearn wrapper around NGBClassifier for multi-class problems."""

    def __init__(self, n_classes: int, n_estimators: int = 400,
                 learning_rate: float = 0.05, random_state: int = 42):
        self.n_classes     = n_classes
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.random_state  = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._ngb = NGBClassifier(
            Dist=k_categorical(self.n_classes),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=False,
        )
        self._ngb.fit(X, y)
        return self

    def predict(self, X):
        return self._ngb.predict(X)

    def predict_proba(self, X):
        dist = self._ngb.pred_dist(X)
        return dist.prob


# ─────────────────────────────────────────────────────────────────────────────
# GenerativeBGA sklearn-compatible wrapper
# ─────────────────────────────────────────────────────────────────────────────

class GenerativeBGAWrapper(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper around GenerativeBGAModel (Bayesian generative).

    Accepts feature names via the `feature_names_in_` convention so it
    integrates seamlessly with the training engine (which passes DataFrames).
    """

    def __init__(self, smoothing_alpha: float = 1.0):
        self.smoothing_alpha = smoothing_alpha

    def fit(self, X, y):
        import pandas as pd
        self.classes_ = np.unique(y)
        # Extract feature/SNP names from DataFrame columns if available
        if hasattr(X, "columns"):
            snp_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X, dtype=float)
            snp_names = [f"snp_{i}" for i in range(X_arr.shape[1])]

        self.feature_names_in_ = snp_names
        self._model = GenerativeBGAModel(smoothing_alpha=self.smoothing_alpha)
        self._model.fit(X_arr, y, snp_names=snp_names)
        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.predict(X_arr)

    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.predict_proba(X_arr)


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — default estimators
# ─────────────────────────────────────────────────────────────────────────────

def build_models(n_classes: int) -> dict[str, BaseEstimator]:
    """
    Return dict mapping model_name → default (un-tuned) estimator.

    HOW TO ADD A NEW MODEL
    ----------------------
    1. Import the classifier at the top of this file.
    2. Add a key-value entry below with sensible defaults.
    3. Add corresponding param grid in get_param_grids().
    """
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, random_state=RANDOM_STATE,
                solver="lbfgs", C=1.0,
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            eval_metric="mlogloss", random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.1, num_leaves=63,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, learning_rate=0.1, depth=6,
            random_seed=RANDOM_STATE, thread_count=-1, verbose=0,
        ),
        "NGBoost": NGBoostWrapper(
            n_classes=n_classes, n_estimators=400,
            learning_rate=0.05, random_state=RANDOM_STATE,
        ),
        "GenerativeNaiveBayes": GenerativeBGAWrapper(smoothing_alpha=1.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search spaces
# ─────────────────────────────────────────────────────────────────────────────

def get_param_grids(n_classes: int) -> dict[str, dict]:
    """
    Return per-model param distributions for RandomizedSearchCV.

    HOW TO ADD A GRID FOR A NEW MODEL
    ----------------------------------
    Add a key matching the name used in build_models(), with values being
    scipy distributions or lists of candidates.
    """
    return {
        "LogisticRegression": {
            "clf__C":        loguniform(0.01, 100),
            "clf__solver":   ["lbfgs", "saga"],
            "clf__max_iter": [2000],
        },
        "RandomForest": {
            "n_estimators":      randint(200, 600),
            "max_depth":         [None, 8, 12, 16, 20],
            "min_samples_leaf":  randint(1, 6),
            "max_features":      ["sqrt", "log2", 0.5, 0.7],
            "min_samples_split": randint(2, 10),
        },
        "GradientBoosting": {
            "n_estimators":  randint(200, 500),
            "learning_rate": loguniform(0.02, 0.3),
            "max_depth":     randint(3, 8),
            "subsample":     uniform(0.7, 0.3),
            "max_features":  ["sqrt", "log2", None],
        },
        "XGBoost": {
            "n_estimators":     randint(200, 600),
            "learning_rate":    loguniform(0.02, 0.3),
            "max_depth":        randint(3, 8),
            "subsample":        uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha":        loguniform(1e-3, 10),
            "reg_lambda":       loguniform(1e-3, 10),
            "min_child_weight": randint(1, 10),
        },
        "LightGBM": {
            "n_estimators":      randint(200, 600),
            "learning_rate":     loguniform(0.02, 0.3),
            "num_leaves":        randint(20, 120),
            "max_depth":         [-1, 6, 8, 10, 12],
            "subsample":         uniform(0.7, 0.3),
            "colsample_bytree":  uniform(0.6, 0.4),
            "reg_alpha":         loguniform(1e-3, 10),
            "reg_lambda":        loguniform(1e-3, 10),
            "min_child_samples": randint(5, 30),
        },
        "CatBoost": {
            "iterations":           randint(200, 600),
            "learning_rate":        loguniform(0.02, 0.3),
            "depth":                randint(4, 10),
            "l2_leaf_reg":          loguniform(1, 20),
            "border_count":         randint(32, 128),
            "bagging_temperature":  uniform(0, 1),
        },
        "NGBoost": {
            "n_estimators":  randint(300, 600),
            "learning_rate": loguniform(0.01, 0.1),
        },
        "GenerativeNaiveBayes": {
            "smoothing_alpha": loguniform(0.1, 10),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tuning helper
# ─────────────────────────────────────────────────────────────────────────────

def tune_model(
    name: str,
    base_model: BaseEstimator,
    param_dist: dict,
    X_train,
    y_train,
    cv=None,
    n_iter: int = N_ITER_SEARCH,
    verbose: bool = True,
) -> tuple:
    """
    Run RandomizedSearchCV for one model.
    Returns (best_estimator, best_params, best_cv_score).
    Falls back to base_model.fit() if tuning fails.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)

    search_jobs = 1 if name == "CatBoost" else -1

    rscv = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=search_jobs,
        refit=True,
        verbose=0,
        error_score="raise",
    )
    try:
        rscv.fit(X_train, y_train)
        return rscv.best_estimator_, rscv.best_params_, rscv.best_score_
    except Exception as e:
        if verbose:
            print(f"    [WARN] Tuning failed for {name}: {e}. Using base model.")
        base_model.fit(X_train, y_train)
        return base_model, {}, 0.0
