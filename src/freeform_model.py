"""
FREEFORM-style BGA classifier.

Feature engineering + stacking ensemble inspired by LLM-based FREEFORM pipeline
(Borisov et al., 2024). Generates pairwise interactions, aggregate statistics,
and centroid distance features from 58 AISNPs, then uses a stacking classifier
with gradient boosting, extra trees, and logistic regression.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from sklearn.ensemble import StackingClassifier

    HAS_STACKING = True
except ImportError:
    HAS_STACKING = False


class SNPFeatureEngineer:
    """
    Feature engineering that mimics an LLM-based FREEFORM pipeline:
    - Selects top-K important SNPs via RandomForest
    - Creates pairwise interaction features (abs diff + product)
    - Computes aggregate statistics (mean, var, homozygous/heterozygous counts)
    - Computes Euclidean distance to each class centroid
    """

    def __init__(self, top_k: int = 15, n_interaction_pairs: int = 45):
        self.top_k = top_k
        self.n_interaction_pairs = n_interaction_pairs
        self.top_indices_ = None
        self.centroids_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, snp_names=None):
        """
        Fit feature engineer: select top-K SNPs and compute class centroids.

        Parameters
        ----------
        X : array of shape (n_samples, n_snps), values 0/1/2/NaN
        y : array of encoded integer labels
        snp_names : optional list of SNP names (unused, kept for API compat)
        """
        self.n_features_in_ = X.shape[1]

        # Impute NaN for RF fitting
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)

        # Select top-K features by RF importance
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        )
        rf.fit(X_imp, y)
        importances = rf.feature_importances_
        k = min(self.top_k, len(importances))
        self.top_indices_ = np.argsort(importances)[-k:][::-1]

        # Compute class centroids from imputed data
        self.classes_ = np.unique(y)
        self.centroids_ = {}
        for cls in self.classes_:
            mask = y == cls
            self.centroids_[cls] = np.nanmean(X_imp[mask], axis=0)

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform raw SNP genotypes into engineered feature matrix.

        Returns concatenation of:
        1. Original SNP features
        2. Pairwise interactions (abs diff + product) for top-K pairs
        3. Aggregate statistics (mean, var, hom count, het count, minor allele prop)
        4. Centroid distances
        """
        if not self._fitted:
            raise RuntimeError("SNPFeatureEngineer has not been fitted yet.")

        n_samples = X.shape[0]

        # Impute for transformation
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)

        parts = []

        # 1. Original features
        parts.append(X_imp)

        # 2. Pairwise interactions for top-K SNPs
        top_idx = self.top_indices_
        interaction_feats = []
        count = 0
        for i in range(len(top_idx)):
            for j in range(i + 1, len(top_idx)):
                if count >= self.n_interaction_pairs:
                    break
                xi = X_imp[:, top_idx[i]]
                xj = X_imp[:, top_idx[j]]
                interaction_feats.append(np.abs(xi - xj))
                interaction_feats.append(xi * xj)
                count += 1
            if count >= self.n_interaction_pairs:
                break

        if interaction_feats:
            parts.append(np.column_stack(interaction_feats))

        # 3. Aggregate statistics
        agg_feats = []
        agg_feats.append(np.nanmean(X_imp, axis=1, keepdims=True))
        agg_feats.append(np.nanvar(X_imp, axis=1, keepdims=True))

        # Count homozygous (0 or 2) and heterozygous (1)
        hom_count = np.sum((X_imp == 0) | (X_imp == 2), axis=1, keepdims=True)
        het_count = np.sum(X_imp == 1, axis=1, keepdims=True)
        agg_feats.append(hom_count.astype(float))
        agg_feats.append(het_count.astype(float))

        # Minor allele proportion (fraction of non-zero genotypes)
        minor_prop = np.sum(X_imp > 0, axis=1, keepdims=True) / max(
            X_imp.shape[1], 1
        )
        agg_feats.append(minor_prop)

        parts.append(np.hstack(agg_feats))

        # 4. Centroid distances
        centroid_dists = []
        for cls in sorted(self.centroids_.keys()):
            c = self.centroids_[cls]
            dist = np.sqrt(np.sum((X_imp - c) ** 2, axis=1, keepdims=True))
            centroid_dists.append(dist)
        parts.append(np.hstack(centroid_dists))

        return np.hstack(parts).astype(np.float64)


class FreeformBGAClassifier(BaseEstimator, ClassifierMixin):
    """
    FREEFORM-style BGA classifier with sklearn-compatible API.

    Pipeline: SNP Feature Engineering -> StandardScaler -> Imputer -> StackingClassifier
    Base learners: LGBMClassifier (or GBM fallback), ExtraTreesClassifier, LogisticRegression
    Meta learner: LogisticRegression with 5-fold CV
    """

    def __init__(self, top_k: int = 15, n_interaction_pairs: int = 45):
        self.top_k = top_k
        self.n_interaction_pairs = n_interaction_pairs
        self.engineer_ = None
        self.scaler_ = None
        self.imputer_ = None
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y, snp_names=None):
        """
        Fit the FREEFORM pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_snps)
        y : array-like of encoded integer labels
        snp_names : optional list of SNP names
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # Feature engineering
        self.engineer_ = SNPFeatureEngineer(
            top_k=self.top_k,
            n_interaction_pairs=self.n_interaction_pairs,
        )
        self.engineer_.fit(X, y, snp_names=snp_names)
        X_eng = self.engineer_.transform(X)

        # Impute any remaining NaN
        self.imputer_ = SimpleImputer(strategy="median")
        X_eng = self.imputer_.fit_transform(X_eng)

        # Scale
        self.scaler_ = StandardScaler()
        X_eng = self.scaler_.fit_transform(X_eng)

        # Build stacking classifier
        if HAS_LGBM:
            boosting = LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        else:
            boosting = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )

        extra_trees = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )

        lr_base = LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, solver="lbfgs"
        )

        estimators = [
            ("boosting", boosting),
            ("extra_trees", extra_trees),
            ("lr", lr_base),
        ]

        meta_lr = LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, solver="lbfgs"
        )

        if HAS_STACKING:
            self.model_ = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_lr,
                cv=5,
                stack_method="predict_proba",
                n_jobs=-1,
            )
        else:
            # Fallback: just use the boosting model
            self.model_ = boosting

        self.model_.fit(X_eng, y)
        return self

    def _transform(self, X):
        """Apply full preprocessing pipeline."""
        X = np.asarray(X, dtype=float)
        X_eng = self.engineer_.transform(X)
        X_eng = self.imputer_.transform(X_eng)
        X_eng = self.scaler_.transform(X_eng)
        return X_eng

    def predict(self, X):
        X_eng = self._transform(X)
        return self.model_.predict(X_eng)

    def predict_proba(self, X):
        X_eng = self._transform(X)
        return self.model_.predict_proba(X_eng)
