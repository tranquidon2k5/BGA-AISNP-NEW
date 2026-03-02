#!/usr/bin/env python3
"""
Generate 10 test samples from the main dataset for UI testing
These samples will be used to test the ancestry prediction interface
"""

import os
import pandas as pd
import numpy as np

# Load the main dataset
data_path = os.path.join('data', 'merged_matrix_201plus58_standardized.csv')
df = pd.read_csv(data_path, index_col=0)

print(f"Dataset loaded: {df.shape}")
print(f"Total samples: {len(df)}")

# Randomly select 10 samples
np.random.seed(42)
test_samples = df.sample(n=10, random_state=42)

# Save test samples
output_path = os.path.join('data', 'test_samples_10.csv')
test_samples.to_csv(output_path)

print(f"\n✓ Test samples saved to: {output_path}")
print(f"\nTest samples:")
print(test_samples[['pop', 'super_pop']])

# Also create version without labels for UI testing (only SNP columns)
snp_cols = test_samples.columns[:-2]
test_samples_snps_only = test_samples[snp_cols].copy()
output_snps_only = os.path.join('data', 'test_samples_10_snps_only.csv')
test_samples_snps_only.to_csv(output_snps_only)

print(f"\n✓ Test samples (SNPs only) saved to: {output_snps_only}")
print(f"  Shape: {test_samples_snps_only.shape}")
print(f"  (Perfect for testing prediction UI without knowing true labels)")
