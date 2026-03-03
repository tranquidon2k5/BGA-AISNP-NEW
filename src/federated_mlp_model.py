"""
Federated MLP: FedAvg simulation for ancestry classification.

Simulates federated learning with K clients, IID stratified data partition,
and a shared MLP architecture. Each round, clients train locally for a few
epochs, then the server averages their weights (FedAvg).

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017.

Requires PyTorch.
"""

from __future__ import annotations

import copy
from collections import OrderedDict

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

    class _FedMLP(nn.Module):
        """Simple MLP for federated classification."""

        def __init__(self, n_input: int, n_classes: int,
                     hidden_sizes: tuple = (128, 64), dropout: float = 0.3):
            super().__init__()
            layers = []
            prev = n_input
            for h in hidden_sizes:
                layers.extend([
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev = h
            layers.append(nn.Linear(prev, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


def _average_weights(weight_list: list[OrderedDict]) -> OrderedDict:
    """Average model weights from multiple clients (FedAvg)."""
    avg = OrderedDict()
    for key in weight_list[0]:
        avg[key] = torch.stack([w[key].float() for w in weight_list]).mean(dim=0)
    return avg


class FederatedMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Federated Learning (FedAvg) MLP classifier with sklearn-compatible API.

    Parameters
    ----------
    n_clients : int
        Number of federated clients.
    hidden_sizes : tuple
        MLP hidden layer sizes (shared architecture).
    dropout : float
        Dropout rate.
    n_rounds : int
        Number of communication rounds.
    local_epochs : int
        Local training epochs per client per round.
    lr : float
        Client learning rate.
    weight_decay : float
        L2 regularization.
    batch_size : int
        Client mini-batch size.
    patience : int
        Early stopping patience on global validation loss.
    random_state : int
        Random seed.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        n_clients: int = 5,
        hidden_sizes: tuple = (128, 64),
        dropout: float = 0.3,
        n_rounds: int = 20,
        local_epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        patience: int = 8,
        random_state: int = 42,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for FederatedMLPClassifier. "
                "Install with: pip install torch"
            )
        self.n_clients = n_clients
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.n_rounds = n_rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def _partition_data(self, X, y, rng):
        """IID stratified partition of data into K clients."""
        n = len(X)
        classes = np.unique(y)
        client_indices = [[] for _ in range(self.n_clients)]

        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            splits = np.array_split(cls_idx, self.n_clients)
            for k, split in enumerate(splits):
                client_indices[k].extend(split.tolist())

        # Shuffle each client's data
        for k in range(self.n_clients):
            rng.shuffle(client_indices[k])

        return client_indices

    def _train_local(self, model, X_local, y_local, criterion):
        """Train a client model locally for local_epochs."""
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        X_t = torch.FloatTensor(X_local).to(self.device)
        y_t = torch.LongTensor(y_local).to(self.device)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        for _ in range(self.local_epochs):
            for xb, yb in loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model.state_dict()

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

        # Validation split
        rng = np.random.RandomState(self.random_state)
        n = len(X_scaled)
        n_val = max(int(n * 0.15), 1)
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_tr, y_tr = X_scaled[train_idx], y[train_idx]
        X_vl, y_vl = X_scaled[val_idx], y[val_idx]

        X_vl_t = torch.FloatTensor(X_vl).to(self.device)
        y_vl_t = torch.LongTensor(y_vl).to(self.device)

        # Partition training data among clients
        client_indices = self._partition_data(X_tr, y_tr, rng)

        # Initialize global model
        torch.manual_seed(self.random_state)
        self.model_ = _FedMLP(
            n_input=n_features,
            n_classes=n_classes,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        best_state = None
        rounds_no_improve = 0

        for rnd in range(self.n_rounds):
            global_weights = copy.deepcopy(self.model_.state_dict())
            local_weights = []

            for k in range(self.n_clients):
                # Create local copy
                local_model = _FedMLP(
                    n_input=n_features,
                    n_classes=n_classes,
                    hidden_sizes=self.hidden_sizes,
                    dropout=self.dropout,
                ).to(self.device)
                local_model.load_state_dict(global_weights)

                ci = client_indices[k]
                if len(ci) == 0:
                    local_weights.append(global_weights)
                    continue

                w = self._train_local(
                    local_model, X_tr[ci], y_tr[ci], criterion
                )
                local_weights.append(w)

            # FedAvg: average client weights
            avg_weights = _average_weights(local_weights)
            self.model_.load_state_dict(avg_weights)

            # Global validation
            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_vl_t)
                val_loss = criterion(val_logits, y_vl_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                rounds_no_improve = 0
            else:
                rounds_no_improve += 1

            if rounds_no_improve >= self.patience:
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
        return self.scaler_.transform(X_imp)
