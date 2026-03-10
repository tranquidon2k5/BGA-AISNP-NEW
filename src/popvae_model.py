"""
popVAE: Semi-supervised Variational Autoencoder for population classification.

Encoder maps SNP genotypes to a latent space, decoder reconstructs genotypes,
and a classification head on z_mu predicts ancestry labels.

Loss = MSE_recon + beta * KL_divergence + gamma * CrossEntropy

Reference: Battey et al., "Predicting geographic location from genetic variation
with deep neural networks", eLife 2020.

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

    class _PopVAENet(nn.Module):
        """
        VAE with encoder, decoder, and classification head.

        Encoder:  n_input -> [128, 64] -> (z_mu, z_logvar) of dim latent_dim
        Decoder:  latent_dim -> [64, 128] -> n_input (sigmoid * 2 for [0,2] range)
        Classifier: z_mu -> 32 -> n_classes
        """

        def __init__(self, n_input: int, n_classes: int, latent_dim: int = 10,
                     enc_hidden: tuple = (128, 64), dropout: float = 0.2):
            super().__init__()

            # Encoder
            enc_layers = []
            prev = n_input
            for h in enc_hidden:
                enc_layers.extend([
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev = h
            self.encoder = nn.Sequential(*enc_layers)
            self.fc_mu = nn.Linear(prev, latent_dim)
            self.fc_logvar = nn.Linear(prev, latent_dim)

            # Decoder (reverse of encoder)
            dec_hidden = list(reversed(enc_hidden))
            dec_layers = []
            prev = latent_dim
            for h in dec_hidden:
                dec_layers.extend([
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev = h
            dec_layers.append(nn.Linear(prev, n_input))
            dec_layers.append(nn.Sigmoid())  # output in [0, 1], scale by 2
            self.decoder = nn.Sequential(*dec_layers)

            # Classification head on z_mu
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_classes),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.decoder(z) * 2.0  # scale to [0, 2] range

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            class_logits = self.classifier(mu)
            return recon, mu, logvar, class_logits


class PopVAEClassifier(BaseEstimator, ClassifierMixin):
    """
    Semi-supervised popVAE classifier with sklearn-compatible API.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    enc_hidden : tuple
        Encoder hidden layer sizes.
    dropout : float
        Dropout rate.
    beta : float
        KL divergence weight.
    gamma : float
        Classification loss weight.
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
        latent_dim: int = 10,
        enc_hidden: tuple = (128, 64),
        dropout: float = 0.2,
        beta: float = 1.0,
        gamma: float = 10.0,
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
                "PyTorch is required for PopVAEClassifier. "
                "Install with: pip install torch"
            )
        self.latent_dim = latent_dim
        self.enc_hidden = enc_hidden
        self.dropout = dropout
        self.beta = beta
        self.gamma = gamma
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

        # Build model
        torch.manual_seed(self.random_state)
        self.model_ = _PopVAENet(
            n_input=n_features,
            n_classes=n_classes,
            latent_dim=self.latent_dim,
            enc_hidden=self.enc_hidden,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        ce_criterion = nn.CrossEntropyLoss()
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
                recon, mu, logvar, class_logits = self.model_(xb)

                # Reconstruction loss (MSE)
                recon_loss = nn.functional.mse_loss(recon, xb)

                # KL divergence
                kl_loss = -0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )

                # Classification loss
                cls_loss = ce_criterion(class_logits, yb)

                loss = recon_loss + self.beta * kl_loss + self.gamma * cls_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            self.model_.eval()
            with torch.no_grad():
                recon_v, mu_v, logvar_v, cls_v = self.model_(X_vl_t)
                recon_l = nn.functional.mse_loss(recon_v, X_vl_t)
                kl_l = -0.5 * torch.mean(1 + logvar_v - mu_v.pow(2) - logvar_v.exp())
                cls_l = ce_criterion(cls_v, y_vl_t)
                val_loss = (recon_l + self.beta * kl_l + self.gamma * cls_l).item()

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
            mu, _ = self.model_.encode(X_t)
            logits = self.model_.classifier(mu)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def encode_latent(self, X):
        """Return latent z_mu for visualization/analysis."""
        X_proc = self._preprocess(X)
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_proc).to(self.device)
            mu, _ = self.model_.encode(X_t)
        return mu.cpu().numpy()

    def _preprocess(self, X):
        X = np.asarray(X, dtype=float)
        X_imp = self.imputer_.transform(X)
        return self.scaler_.transform(X_imp)
