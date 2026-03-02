#!/usr/bin/env python3
"""
Generate 10 stratified test samples: 5 from train set, 5 from test set
for model validation and blind testing
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data_path = os.path.join('data', 'merged_matrix_201plus58_standardized.csv')
df = pd.read_csv(data_path, index_col=0)

print(f"Dataset loaded: {df.shape}")
print(f"Total samples: {len(df)}")

# Get SNP columns (all except last two)
snp_names = df.columns[:-2].tolist()
X = df[snp_names].values.astype(np.float32)
y_super_pop = df['super_pop'].values

# Encode labels to stratify properly
le_super_pop = LabelEncoder()
y_super_pop_encoded = le_super_pop.fit_transform(y_super_pop)

# Do stratified train/test split with same random_state as notebook
X_train, X_test, indices_train, indices_test, y_train, y_test = train_test_split(
    X,
    np.arange(len(df)),  # Keep track of original indices
    y_super_pop_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_super_pop_encoded
)

print(f"Train set: {len(indices_train)} samples")
print(f"Test set: {len(indices_test)} samples")

# Sample 5 from train and 5 from test
np.random.seed(42)
sampled_train_idx = np.random.choice(len(indices_train), size=5, replace=False)
sampled_test_idx = np.random.choice(len(indices_test), size=5, replace=False)

# Get original indices
original_train_indices = indices_train[sampled_train_idx]
original_test_indices = indices_test[sampled_test_idx]

# Combine and create dataframe
combined_indices = np.concatenate([original_train_indices, original_test_indices])
sample_df = df.iloc[combined_indices].copy()

# Add a column to mark source (train or test)
source = ['Train'] * 5 + ['Test'] * 5
sample_df['source'] = source

print(f"\n✓ Sampled 10 test samples:")
print(f"  - 5 from training set")
print(f"  - 5 from test set")
print(f"\nSamples:")
print(sample_df[['pop', 'super_pop', 'source']])

# Save full samples (with all SNP columns)
output_path = os.path.join('data', 'stratified_samples_10.csv')
sample_df.to_csv(output_path)
print(f"\n✓ Full samples saved to: {output_path}")
print(f"Shape: {sample_df.shape}")

# Save SNP-only version
snp_cols = snp_names
snp_only_df = sample_df[snp_cols].copy()
snp_output_path = os.path.join('data', 'stratified_samples_10_snps_only.csv')
snp_only_df.to_csv(snp_output_path)
print(f"✓ SNP-only samples saved to: {snp_output_path}")
print(f"Shape: {snp_only_df.shape}")

# Print summary
print(f"\n{'='*70}")
print("STRATIFIED SAMPLES SUMMARY:")
print(f"{'='*70}")
print(f"\nTrain set samples:")
for idx in original_train_indices:
    row = df.iloc[idx]
    print(f"  {idx:5d}: {row['pop']:5s} ({row['super_pop']})")

print(f"\nTest set samples:")
for idx in original_test_indices:
    row = df.iloc[idx]
    print(f"  {idx:5d}: {row['pop']:5s} ({row['super_pop']})")

print(f"\n{'='*70}")
print("Continental distribution:")
print(sample_df['super_pop'].value_counts().to_string())
print(f"\nPopulation distribution:")
print(sample_df['pop'].value_counts().to_string())
