"""
Deep MLP-Geo model for biogeographical ancestry prediction.

Locator/GeoGenIE-style lat/lon regression using a multi-layer perceptron.
Predicts geographic coordinates from 58 AISNPs, then maps to nearest
population centroid for classification.

Requires PyTorch. Falls back gracefully if unavailable.
"""

import math
import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# 1KGP population sampling coordinates (lat, lon)
# ---------------------------------------------------------------------------
POP_COORDINATES = {
    # AFR
    "ACB": (13.1, -59.6),
    "ASW": (36.1, -86.8),
    "ESN": (6.5, 3.4),
    "GWD": (13.5, -15.4),
    "LWK": (-0.1, 34.8),
    "MSL": (8.5, -13.2),
    "YRI": (7.4, 3.9),
    # AMR
    "CLM": (4.7, -74.1),
    "MXL": (23.6, -102.5),
    "PEL": (-12.0, -77.0),
    "PUR": (18.2, -66.6),
    # EAS
    "CDX": (22.0, 100.8),
    "CHB": (39.9, 116.4),
    "CHS": (23.1, 113.3),
    "JPT": (35.7, 139.7),
    "KHV": (21.0, 105.8),
    # EUR
    "CEU": (40.8, -111.9),
    "FIN": (60.2, 24.9),
    "GBR": (51.5, -0.1),
    "IBS": (40.4, -3.7),
    "TSI": (43.8, 11.3),
    # SAS
    "BEB": (23.7, 90.4),
    "GIH": (23.0, 72.6),
    "ITU": (15.5, 78.5),
    "PJL": (31.5, 74.3),
    "STU": (7.9, 80.7),
}

# Super-population centroids (average of member populations)
SUPERPOP_COORDINATES = {
    "AFR": (12.1, -18.9),
    "AMR": (8.6, -80.1),
    "EAS": (28.3, 115.2),
    "EUR": (47.3, -15.9),
    "SAS": (20.3, 79.3),
}


def _pop_to_coords(pop_label: str) -> tuple:
    """Look up coordinates for a population or super-population label."""
    if pop_label in POP_COORDINATES:
        return POP_COORDINATES[pop_label]
    if pop_label in SUPERPOP_COORDINATES:
        return SUPERPOP_COORDINATES[pop_label]
    raise KeyError(
        f"Unknown population label '{pop_label}'. "
        f"Known: {sorted(POP_COORDINATES.keys())} + {sorted(SUPERPOP_COORDINATES.keys())}"
    )


# ---------------------------------------------------------------------------
# PyTorch model components
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class GeoMLP(nn.Module):
        """
        Multi-layer perceptron for geographic coordinate regression.

        Architecture:
            BatchNorm -> [Linear+ReLU+BN+Dropout] x 3 -> Linear(2)
        Output: (lat, lon)
        """

        def __init__(
            self,
            n_input: int,
            hidden_sizes=(512, 256, 128),
            dropout: float = 0.3,
        ):
            super().__init__()
            self.input_bn = nn.BatchNorm1d(n_input)

            layers = []
            prev = n_input
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.Dropout(dropout))
                prev = h

            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(prev, 2)  # (lat, lon)

        def forward(self, x):
            x = self.input_bn(x)
            x = self.backbone(x)
            return self.head(x)

    def haversine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Haversine distance loss on Earth surface (km).

        Parameters
        ----------
        pred : tensor of shape (n, 2) with (lat, lon) in degrees
        target : tensor of shape (n, 2) with (lat, lon) in degrees

        Returns
        -------
        Mean haversine distance in km.
        """
        R = 6371.0  # Earth radius in km

        lat1 = torch.deg2rad(pred[:, 0])
        lon1 = torch.deg2rad(pred[:, 1])
        lat2 = torch.deg2rad(target[:, 0])
        lon2 = torch.deg2rad(target[:, 1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        )
        # Clamp for numerical stability
        a = torch.clamp(a, 0.0, 1.0)
        c = 2 * torch.asin(torch.sqrt(a))

        return (R * c).mean()

    def train_geo_mlp(
        model: GeoMLP,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 300,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        device: str = "cpu",
    ) -> GeoMLP:
        """
        Train GeoMLP with AdamW, ReduceLROnPlateau, and early stopping.

        Parameters
        ----------
        model : GeoMLP instance
        X_train, y_train : training data (features, coords)
        X_val, y_val : optional validation data for early stopping
        epochs : max training epochs
        batch_size : mini-batch size
        lr : initial learning rate
        weight_decay : L2 regularization
        patience : early stopping patience
        device : 'cpu' or 'cuda'

        Returns
        -------
        Trained GeoMLP (best checkpoint by validation loss).
        """
        model = model.to(device)

        X_t = torch.FloatTensor(X_train).to(device)
        y_t = torch.FloatTensor(y_train).to(device)
        train_ds = TensorDataset(X_t, y_t)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = torch.FloatTensor(X_val).to(device)
            y_v = torch.FloatTensor(y_val).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = haversine_loss(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)

            # Validation
            if has_val:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_v)
                    val_loss = haversine_loss(val_pred, y_v).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break
            else:
                scheduler.step(avg_train)

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(device)

        model.eval()
        return model


class MLPGeoModel:
    """
    Top-level API for geographic MLP ancestry prediction.

    Trains a GeoMLP to predict (lat, lon) from SNP genotypes,
    then maps predictions to nearest population centroid for classification.

    Parameters
    ----------
    hidden_sizes : tuple of hidden layer sizes
    dropout : dropout rate
    epochs : max training epochs
    lr : learning rate
    weight_decay : L2 regularization
    patience : early stopping patience
    temperature : softmax temperature for pseudo-probabilities
    device : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        hidden_sizes=(256, 128, 64),
        dropout: float = 0.5,
        epochs: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        patience: int = 20,
        temperature: float = 500.0,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for MLPGeoModel. Install with: pip install torch"
            )

        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.temperature = temperature
        self.device = device

        self.model_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.centroids_ = None  # {encoded_label: (lat, lon)}
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X, y_enc, pop_labels):
        """
        Fit the MLP-Geo model.

        Parameters
        ----------
        X : array of shape (n_samples, n_snps), genotype values 0/1/2/NaN
        y_enc : array of encoded integer labels (used for centroid mapping)
        pop_labels : array of string population labels (e.g., 'GBR', 'EAS')
            Used to look up geographic coordinates.
        """
        X = np.asarray(X, dtype=float)
        y_enc = np.asarray(y_enc)
        pop_labels = np.asarray(pop_labels)

        # Map population labels to coordinates
        y_coords = np.array([_pop_to_coords(p) for p in pop_labels], dtype=float)

        # Compute class centroids
        self.classes_ = np.unique(y_enc)
        self.n_classes_ = len(self.classes_)
        self.centroids_ = {}
        for cls in self.classes_:
            mask = y_enc == cls
            self.centroids_[cls] = y_coords[mask].mean(axis=0)

        # Impute and scale
        self.imputer_ = SimpleImputer(strategy="constant", fill_value=0)
        X_imp = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imp)

        # Validation split (10%)
        n = len(X_scaled)
        n_val = max(int(n * 0.1), 1)
        indices = np.random.RandomState(42).permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_tr = X_scaled[train_idx]
        y_tr = y_coords[train_idx]
        X_vl = X_scaled[val_idx]
        y_vl = y_coords[val_idx]

        # Build and train model
        n_features = X_scaled.shape[1]
        self.model_ = GeoMLP(
            n_input=n_features,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        )

        self.model_ = train_geo_mlp(
            model=self.model_,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_vl,
            y_val=y_vl,
            epochs=self.epochs,
            batch_size=min(64, max(16, n // 10)),
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            device=self.device,
        )

        return self

    def _preprocess(self, X):
        """Impute and scale input features."""
        X = np.asarray(X, dtype=float)
        X_imp = self.imputer_.transform(X)
        return self.scaler_.transform(X_imp)

    def predict_coordinates(self, X) -> np.ndarray:
        """
        Predict geographic coordinates (lat, lon) for each sample.

        Returns
        -------
        coords : array of shape (n_samples, 2)
        """
        X_scaled = self._preprocess(X)

        self.model_.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_scaled).to(self.device)
            coords = self.model_(X_t).cpu().numpy()

        return coords

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels by nearest centroid classification.

        Returns
        -------
        y_pred : array of integer labels
        """
        coords = self.predict_coordinates(X)
        centroid_keys = sorted(self.centroids_.keys())
        centroid_arr = np.array([self.centroids_[k] for k in centroid_keys])

        # Euclidean distance to each centroid
        dists = np.sqrt(
            ((coords[:, None, :] - centroid_arr[None, :, :]) ** 2).sum(axis=2)
        )
        nearest = np.argmin(dists, axis=1)
        return np.array([centroid_keys[i] for i in nearest])

    def predict_proba(self, X) -> np.ndarray:
        """
        Compute pseudo-probabilities via softmax(-distance / temperature).

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
        """
        coords = self.predict_coordinates(X)
        centroid_keys = sorted(self.centroids_.keys())
        centroid_arr = np.array([self.centroids_[k] for k in centroid_keys])

        dists = np.sqrt(
            ((coords[:, None, :] - centroid_arr[None, :, :]) ** 2).sum(axis=2)
        )

        # Softmax over negative distances
        logits = -dists / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return proba
