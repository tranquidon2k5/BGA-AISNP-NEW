"""
SVD-MLP with Adversarial Training (FGSM).

Step 1: TruncatedSVD reduces feature dimensionality.
Step 2: MLP with BatchNorm + Dropout for classification.
Adversarial training via FGSM perturbation of inputs.
Loss = (1 - alpha) * loss_clean + alpha * loss_adversarial.

Requires PyTorch.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class _AdvMLP(nn.Module):
        """MLP with BatchNorm and Dropout for classification."""

        def __init__(self, n_input: int, n_classes: int, hidden_sizes=(128, 64),
                     dropout: float = 0.3):
            super().__init__()
            layers = []
            prev = n_input
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


class SVDMLPAdvClassifier(BaseEstimator, ClassifierMixin):
    """
    SVD dimensionality reduction + MLP with FGSM adversarial training.

    Parameters
    ----------
    n_components : int
        Number of SVD components.
    hidden_sizes : tuple
        MLP hidden layer sizes.
    dropout : float
        Dropout rate.
    epsilon : float
        FGSM perturbation magnitude.
    alpha : float
        Adversarial loss weight: loss = (1-alpha)*clean + alpha*adv.
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.
    batch_size : int
        Mini-batch size.
    patience : int
        Early stopping patience.
    random_state : int
        Random seed.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        n_components: int = 20,
        hidden_sizes: tuple = (128, 64),
        dropout: float = 0.3,
        epsilon: float = 0.05,
        alpha: float = 0.3,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        patience: int = 20,
        random_state: int = 42,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for SVDMLPAdvClassifier. "
                "Install with: pip install torch"
            )
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.epsilon = epsilon
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def fit(self, X, y, snp_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Preprocessing
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imp)

        # SVD reduction
        n_comp = min(self.n_components, X_scaled.shape[1])
        self.svd_ = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        X_svd = self.svd_.fit_transform(X_scaled)

        # Validation split
        rng = np.random.RandomState(self.random_state)
        n = len(X_svd)
        n_val = max(int(n * 0.15), 1)
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_tr, y_tr = X_svd[train_idx], y[train_idx]
        X_vl, y_vl = X_svd[val_idx], y[val_idx]

        # Build model
        torch.manual_seed(self.random_state)
        self.model_ = _AdvMLP(
            n_input=n_comp,
            n_classes=n_classes,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
        )

        # Data loaders
        X_tr_t = torch.FloatTensor(X_tr).to(self.device)
        y_tr_t = torch.LongTensor(y_tr).to(self.device)
        train_ds = TensorDataset(X_tr_t, y_tr_t)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        X_vl_t = torch.FloatTensor(X_vl).to(self.device)
        y_vl_t = torch.LongTensor(y_vl).to(self.device)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in train_loader:
                # Clean forward pass
                xb.requires_grad_(True)
                logits_clean = self.model_(xb)
                loss_clean = criterion(logits_clean, yb)

                # FGSM adversarial examples
                loss_clean.backward(retain_graph=True)
                grad = xb.grad.data
                x_adv = xb + self.epsilon * grad.sign()
                xb.requires_grad_(False)

                logits_adv = self.model_(x_adv.detach())
                loss_adv = criterion(logits_adv, yb)

                # Combined loss
                loss = (1 - self.alpha) * loss_clean + self.alpha * loss_adv

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_vl_t)
                val_loss = criterion(val_logits, y_vl_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
            self.model_ = self.model_.to(self.device)

        self.model_.eval()
        self._fitted = True
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        X_proc = self._preprocess(X)
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_proc).to(self.device)
            logits = self.model_(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def _preprocess(self, X):
        X = np.asarray(X, dtype=float)
        X_imp = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imp)
        return self.svd_.transform(X_scaled)
