# src/data_utils.py
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

META_COLS = ["sample", "pop", "super_pop"]


def load_continental_csv(path: str) -> pd.DataFrame:
    """
    Đọc CSV continental, trả về DataFrame raw (vẫn còn _1/_2).
    """
    df = pd.read_csv(path)
    return df


def load_eastasian_csv(path: str) -> pd.DataFrame:
    """
    Đọc CSV eastasian, trả về DataFrame raw (vẫn còn _1/_2).
    """
    df = pd.read_csv(path)
    return df


def encode_genotypes(df: pd.DataFrame):
    """
    Input: df với cột:
        sample, pop, super_pop, rsXXXX_1, rsXXXX_2, ...

    Output:
        encoded_df: DataFrame gồm meta + genotype 0/1/2 cho mỗi SNP
        snp_names: list tên SNP (không có _1/_2)
    """
    # Tìm danh sách SNP từ các cột kết thúc bằng _1
    snp_names = sorted(set(col[:-2] for col in df.columns if col.endswith("_1")))

    geno_data = {}

    for snp in snp_names:
        a1 = df[f"{snp}_1"].astype(str)
        a2 = df[f"{snp}_2"].astype(str)

        # Đếm tần suất allele để chọn major allele
        alleles = list(a1) + list(a2)
        counts = Counter(alleles)

        # Loại bỏ các giá trị thiếu nếu tồn tại
        for bad in ["nan", "NaN", "None", "0"]:
            counts.pop(bad, None)

        if len(counts) == 0:
            major = None
        else:
            major = counts.most_common(1)[0][0]

        genotypes = []
        for x, y in zip(a1, a2):
            if (
                major is None
                or x in ["nan", "NaN", "None", "0"]
                or y in ["nan", "NaN", "None", "0"]
            ):
                genotypes.append(np.nan)  # nếu muốn có thể impute sau
            else:
                # Đếm số allele KHÁC major (0,1,2)
                cnt_non_major = (x != major) + (y != major)
                genotypes.append(cnt_non_major)

        geno_data[snp] = genotypes

    geno_df = pd.DataFrame(geno_data)
    meta_df = df[META_COLS].reset_index(drop=True)
    encoded_df = pd.concat([meta_df, geno_df], axis=1)

    return encoded_df, snp_names


def split_xy(df_encoded: pd.DataFrame, snp_names, label_col: str):
    """
    Tách X, y từ DataFrame đã encode genotype.
    """
    X = df_encoded[snp_names].values
    y = df_encoded[label_col].values
    return X, y


def stratified_train_test(X, y, test_size=0.2, random_state=42):
    """
    Stratified split cho classification.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def encode_labels(y):
    """
    Dùng LabelEncoder để encode nhãn thành số nguyên (0..K-1)
    """
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    return y_int, le
