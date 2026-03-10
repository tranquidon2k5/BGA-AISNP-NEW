"""
Train MLP-Geo coordinate regression model cho phân loại châu lục (Continental tier).
Dự đoán tọa độ (lat, lon) từ 58 AISNPs, rồi ánh xạ về super_pop qua nearest centroid.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_utils import (
    load_continental_csv,
    encode_genotypes,
    split_xy,
    encode_labels,
)
from sklearn.model_selection import train_test_split
from src.mlp_geo_model import MLPGeoModel


DATA_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")


def main():
    df_raw = load_continental_csv(DATA_PATH)
    print(f"Loaded continental data: {df_raw.shape}")

    df_encoded, snp_names = encode_genotypes(df_raw)
    print(f"Encoded continental data: {df_encoded.shape}")
    print(f"Number of SNPs (continental panel): {len(snp_names)}")

    X = df_encoded[snp_names].values.astype(float)
    y = df_encoded["super_pop"].values
    pop_labels = df_encoded["pop"].values

    y_int, le = encode_labels(y)

    X_train, X_test, y_train, y_test, pop_train, pop_test = train_test_split(
        X, y_int, pop_labels,
        test_size=0.2, random_state=42, stratify=y_int,
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = MLPGeoModel(
        hidden_sizes=(256, 128, 64),
        dropout=0.3,
        epochs=300,
        lr=1e-3,
        weight_decay=1e-3,
        patience=20,
    )
    model.fit(X_train, y_train, pop_labels=pop_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== MLP-Geo Continental Ancestry ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Predict coordinates on test set
    coords = model.predict_coordinates(X_test)
    print(f"\nPredicted coordinate range: lat [{coords[:,0].min():.1f}, {coords[:,0].max():.1f}], "
          f"lon [{coords[:,1].min():.1f}, {coords[:,1].max():.1f}]")

    # Train on full data for deployment
    model_full = MLPGeoModel(
        hidden_sizes=(256, 128, 64),
        dropout=0.3,
        epochs=300,
        lr=1e-3,
        weight_decay=1e-3,
        patience=20,
    )
    model_full.fit(X, y_int, pop_labels=pop_labels)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_full, os.path.join("models", "continent_mlp_geo.pkl"))
    joblib.dump(le, os.path.join("models", "continent_mlp_geo_label_encoder.pkl"))
    joblib.dump(snp_names, os.path.join("models", "continent_mlp_geo_snp_names.pkl"))

    print(f"\nSaved continental MLP-Geo model (acc on test={acc:.4f}) to models/.")


if __name__ == "__main__":
    main()
