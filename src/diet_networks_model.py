"""
Diet Networks for ancestry classification from SNPs.

The auxiliary network predicts the main network's embedding weights from
per-SNP statistics (mean, var, frac_0, frac_1, frac_2), dramatically
reducing the number of free parameters when n_features >> n_samples.

Reference: Romero et al., "Diet Networks: Thin Parameters for Fat Genomics", ICLR 2017.

Requires PyTorch.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
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

    class _AuxNet(nn.Module):
        """
        Auxiliary network: predicts embedding weight matrix from SNP statistics.

        Input:  (n_features, stat_dim) — per-SNP stats
        Output: (n_features, embed_dim) — weight matrix W_e
        """

        def __init__(self, stat_dim: int = 5, hidden: int = 64, embed_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(stat_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, embed_dim),
            )

        def forward(self, snp_stats):
            # snp_stats: (n_features, stat_dim) -> W_e: (n_features, embed_dim)
            return self.net(snp_stats)

    class _DietClassifier(nn.Module):
        """
        Main classifier using auxiliary-generated embedding weights.

        Forward:  x @ W_e -> ReLU/BN/Dropout -> hidden -> n_classes
        """

        def __init__(self, embed_dim: int = 64, hidden: int = 32,
                     n_classes: int = 5, dropout: float = 0.3):
            super().__init__()
            self.bn = nn.BatchNorm1d(embed_dim)
            self.drop = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, x, W_e):
            # x: (batch, n_features), W_e: (n_features, embed_dim)
            h = x @ W_e  # (batch, embed_dim)
            h = torch.relu(h)
            h = self.bn(h)
            h = self.drop(h)
            return self.classifier(h)


class DietNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    Diet Networks classifier with sklearn-compatible API.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (output of auxiliary network).
    aux_hidden : int
        Hidden size of auxiliary network.
    clf_hidden : int
        Hidden size of main classifier.
    dropout : float
        Dropout rate.
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
        embed_dim: int = 64,
        aux_hidden: int = 64,
        clf_hidden: int = 32,
        dropout: float = 0.3,
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
                "PyTorch is required for DietNetworkClassifier. "
                "Install with: pip install torch"
            )
        self.embed_dim = embed_dim
        self.aux_hidden = aux_hidden
        self.clf_hidden = clf_hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def _compute_snp_stats(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-SNP statistics: [mean, var, frac_0, frac_1, frac_2].

        Parameters
        ----------
        X : array of shape (n_samples, n_features), values 0/1/2

        Returns
        -------
        stats : array of shape (n_features, 5)
        """
        n = X.shape[0]
        mean = np.nanmean(X, axis=0)
        var = np.nanvar(X, axis=0)
        frac_0 = np.sum(X == 0, axis=0) / n
        frac_1 = np.sum(X == 1, axis=0) / n
        frac_2 = np.sum(X == 2, axis=0) / n
        return np.column_stack([mean, var, frac_0, frac_1, frac_2]).astype(np.float32)

    def fit(self, X, y, snp_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Impute and scale
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imp)

        # Compute SNP statistics (from imputed data before scaling)
        self.snp_stats_ = self._compute_snp_stats(X_imp)
        snp_stats_t = torch.FloatTensor(self.snp_stats_).to(self.device)

        # Validation split
        rng = np.random.RandomState(self.random_state)
        n = len(X_scaled)
        n_val = max(int(n * 0.15), 1)
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_tr, y_tr = X_scaled[train_idx], y[train_idx]
        X_vl, y_vl = X_scaled[val_idx], y[val_idx]

        # Build networks
        torch.manual_seed(self.random_state)
        self.aux_net_ = _AuxNet(
            stat_dim=5, hidden=self.aux_hidden, embed_dim=self.embed_dim
        ).to(self.device)
        self.clf_net_ = _DietClassifier(
            embed_dim=self.embed_dim,
            hidden=self.clf_hidden,
            n_classes=n_classes,
            dropout=self.dropout,
        ).to(self.device)

        params = list(self.aux_net_.parameters()) + list(self.clf_net_.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
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
            self.aux_net_.train()
            self.clf_net_.train()

            for xb, yb in train_loader:
                W_e = self.aux_net_(snp_stats_t)
                logits = self.clf_net_(xb, W_e)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            self.aux_net_.eval()
            self.clf_net_.eval()
            with torch.no_grad():
                W_e = self.aux_net_(snp_stats_t)
                val_logits = self.clf_net_(X_vl_t, W_e)
                val_loss = criterion(val_logits, y_vl_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "aux": {k: v.cpu().clone() for k, v in self.aux_net_.state_dict().items()},
                    "clf": {k: v.cpu().clone() for k, v in self.clf_net_.state_dict().items()},
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        if best_state is not None:
            self.aux_net_.load_state_dict(best_state["aux"])
            self.clf_net_.load_state_dict(best_state["clf"])
            self.aux_net_ = self.aux_net_.to(self.device)
            self.clf_net_ = self.clf_net_.to(self.device)

        self.aux_net_.eval()
        self.clf_net_.eval()
        self._fitted = True
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        X_proc = self._preprocess(X)
        snp_stats_t = torch.FloatTensor(self.snp_stats_).to(self.device)

        self.aux_net_.eval()
        self.clf_net_.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_proc).to(self.device)
            W_e = self.aux_net_(snp_stats_t)
            logits = self.clf_net_(X_t, W_e)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def _preprocess(self, X):
        X = np.asarray(X, dtype=float)
        X_imp = self.imputer_.transform(X)
        return self.scaler_.transform(X_imp)
