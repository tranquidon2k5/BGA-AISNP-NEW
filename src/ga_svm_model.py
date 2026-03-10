"""
GA-SVM: Genetic Algorithm feature selection + SVM classifier.

Uses a simple genetic algorithm to evolve binary feature masks,
evaluating fitness via SVM cross-validation balanced accuracy.
The final model trains an RBF-SVM on the selected feature subset.

Pure sklearn implementation (no PyTorch required).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class GASVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Genetic Algorithm feature selection + RBF-SVM classifier.

    Parameters
    ----------
    pop_size : int
        GA population size.
    n_generations : int
        Number of GA generations.
    tournament_size : int
        Tournament selection size.
    crossover_prob : float
        Probability of single-point crossover.
    mutation_prob : float
        Per-gene mutation probability.
    svm_C : float
        SVM regularization parameter.
    svm_gamma : str or float
        SVM kernel coefficient.
    cv_folds : int
        Cross-validation folds for fitness evaluation.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        pop_size: int = 50,
        n_generations: int = 40,
        tournament_size: int = 3,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.03,
        svm_C: float = 10.0,
        svm_gamma: str = "scale",
        cv_folds: int = 3,
        random_state: int = 42,
    ):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.svm_C = svm_C
        self.svm_gamma = svm_gamma
        self.cv_folds = cv_folds
        self.random_state = random_state

    def fit(self, X, y, snp_names=None):
        """
        Run GA feature selection and train final SVM.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of integer labels
        snp_names : optional list of feature names
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.snp_names_ = snp_names

        # Impute and scale
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imp)

        n_features = X_scaled.shape[1]
        rng = np.random.RandomState(self.random_state)

        # --- Genetic Algorithm ---
        # Initialize population: binary masks with ~50% features on
        population = rng.randint(0, 2, size=(self.pop_size, n_features)).astype(bool)
        # Ensure at least 1 feature per individual
        for i in range(self.pop_size):
            if not population[i].any():
                population[i, rng.randint(n_features)] = True

        best_mask = population[0].copy()
        best_fitness = -1.0

        for gen in range(self.n_generations):
            # Evaluate fitness
            fitness = np.array([
                self._evaluate_fitness(X_scaled, y, mask) for mask in population
            ])

            # Track best
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_mask = population[gen_best_idx].copy()

            # Selection + crossover + mutation -> next generation
            new_pop = [best_mask.copy()]  # elitism: keep best

            while len(new_pop) < self.pop_size:
                # Tournament selection
                p1 = self._tournament_select(population, fitness, rng)
                p2 = self._tournament_select(population, fitness, rng)

                # Crossover
                if rng.random() < self.crossover_prob:
                    c1, c2 = self._crossover(p1, p2, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                c1 = self._mutate(c1, rng)
                c2 = self._mutate(c2, rng)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            population = np.array(new_pop[:self.pop_size])

        self.selected_features_ = best_mask
        n_selected = int(best_mask.sum())

        # If GA selected no features, use all
        if n_selected == 0:
            self.selected_features_ = np.ones(n_features, dtype=bool)

        # Train final SVM on selected features
        X_sel = X_scaled[:, self.selected_features_]
        self.svm_ = SVC(
            C=self.svm_C,
            kernel="rbf",
            gamma=self.svm_gamma,
            probability=True,
            random_state=self.random_state,
        )
        self.svm_.fit(X_sel, y)

        self._fitted = True
        return self

    def predict(self, X):
        X_proc = self._preprocess(X)
        return self.svm_.predict(X_proc)

    def predict_proba(self, X):
        X_proc = self._preprocess(X)
        return self.svm_.predict_proba(X_proc)

    def _preprocess(self, X):
        """Impute, scale, and select features."""
        X = np.asarray(X, dtype=float)
        X_imp = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imp)
        return X_scaled[:, self.selected_features_]

    def _evaluate_fitness(self, X_scaled, y, mask):
        """Evaluate a feature mask via SVM cross-validation balanced accuracy."""
        if not mask.any():
            return 0.0

        X_sel = X_scaled[:, mask]
        svm = SVC(
            C=self.svm_C,
            kernel="rbf",
            gamma=self.svm_gamma,
            random_state=self.random_state,
        )
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        try:
            scores = cross_val_score(
                svm, X_sel, y, cv=cv, scoring="balanced_accuracy", n_jobs=1
            )
            return scores.mean()
        except Exception:
            return 0.0

    def _tournament_select(self, population, fitness, rng):
        """Tournament selection."""
        indices = rng.choice(len(population), size=self.tournament_size, replace=False)
        best = indices[np.argmax(fitness[indices])]
        return population[best].copy()

    def _crossover(self, p1, p2, rng):
        """Single-point crossover."""
        point = rng.randint(1, len(p1))
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        # Ensure at least 1 feature
        if not c1.any():
            c1[rng.randint(len(c1))] = True
        if not c2.any():
            c2[rng.randint(len(c2))] = True
        return c1, c2

    def _mutate(self, individual, rng):
        """Bit-flip mutation."""
        flip = rng.random(len(individual)) < self.mutation_prob
        individual = individual.copy()
        individual[flip] = ~individual[flip]
        if not individual.any():
            individual[rng.randint(len(individual))] = True
        return individual

    def get_selected_feature_names(self):
        """Return names of GA-selected features (if snp_names were provided)."""
        if self.snp_names_ is not None and hasattr(self, "selected_features_"):
            return [
                name
                for name, sel in zip(self.snp_names_, self.selected_features_)
                if sel
            ]
        return None
