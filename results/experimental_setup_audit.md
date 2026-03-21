# Báo cáo Kiểm tra Thiết lập Thí nghiệm
## BGA-AISNP: Đánh giá tính nhất quán giữa các models

**Ngày:** 2026-03-20
**Mục tiêu:** Kiểm tra xem tất cả models có chạy cùng một thiết lập thí nghiệm (experimental setup) để đảm bảo so sánh công bằng (fair comparison).

---

## 1. Tổng quan các models

Dự án có **14 models** chia làm 2 nhóm, chạy bởi **2 pipeline riêng biệt**:

| Nhóm | Pipeline | Models |
|------|----------|--------|
| Main pipeline | `scripts/run_pipeline.py` | LogisticRegression, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost, NGBoost, GenerativeNaiveBayes, GA-SVM |
| Classic models | `scripts/run_classic_models.py` | GA-SVM, SVD-MLP-Adv, DietNetworks, popVAE, FederatedMLP |

> **Lưu ý:** GA-SVM xuất hiện ở CẢ HAI pipeline nhưng với thiết lập khác nhau.

---

## 2. So sánh thiết lập giữa 2 pipeline

### 2.1. Các thiết lập GIỐNG nhau

| Thiết lập | Main Pipeline | Classic Pipeline | Khớp? |
|-----------|---------------|------------------|-------|
| Data source | `load_continental()` / `load_eas()` | `load_continental()` / `load_eas()` | OK |
| Genotype encoding | Additive (0/1/2) | Additive (0/1/2) | OK |
| Train/Test ratio | 80% / 20% | 80% / 20% | OK |
| Stratified split | `stratify=y` | `stratify=y` | OK |
| Random seed (split) | 42 | 42 | OK |
| Test metrics | ACC, Balanced ACC, MCC, F1 macro, F1 weighted, ROC-AUC | ACC, Balanced ACC, MCC, F1 macro, F1 weighted, ROC-AUC | OK |

### 2.2. Các thiết lập KHÁC nhau (VẤN ĐỀ CHÍNH)

| Thiết lập | Main Pipeline | Classic Pipeline | Mức ảnh hưởng |
|-----------|---------------|------------------|---------------|
| **Hyperparameter tuning** | `RandomizedSearchCV` (20 iter, 5-fold) | **KHÔNG CÓ** — dùng default params | **NGHIÊM TRỌNG** |
| **Cross-validation evaluation** | 5-fold StratifiedKFold sau tuning | **KHÔNG CÓ** | **NGHIÊM TRỌNG** |
| **CV metrics (báo cáo)** | cv_accuracy_mean, cv_f1_mean, cv_balanced_acc_mean | **Không có** | Trung bình |
| **Tuning scoring** | `balanced_accuracy` | N/A | N/A |
| **Best params tracking** | Có (lưu trong CSV) | Không | Nhẹ |

---

## 3. Phân tích chi tiết Main Pipeline (9 models)

### 3.1. Thiết lập chung (nhất quán)

Tất cả 9 models trong main pipeline chia sẻ:

```
RANDOM_STATE  = 42
TEST_SIZE     = 0.20          # stratified split
CV_FOLDS      = 5             # StratifiedKFold(shuffle=True, random_state=42)
N_ITER_SEARCH = 20            # RandomizedSearchCV iterations
scoring       = "balanced_accuracy"   # tuning objective
```

- **Train/test split:** Gọi 1 lần duy nhất trong `training.py:train_all()`, tất cả models dùng chung `X_train, X_test, y_train, y_test`.
- **Tuning:** Mỗi model đều qua `RandomizedSearchCV` với cùng `n_iter=20`, `cv=StratifiedKFold(5)`, `scoring=balanced_accuracy`.
- **Post-tuning CV:** Sau khi tune, tất cả models được đánh giá lại bằng `cross_validate()` với cùng 5-fold split và `scoring=[accuracy, f1_macro, balanced_accuracy]`.
- **Test evaluation:** Cùng `X_test, y_test`, cùng bộ metrics.

### 3.2. Khác biệt nhỏ giữa các models (chấp nhận được)

#### a) Feature Scaling

| Model | Scaling | Lý do |
|-------|---------|-------|
| LogisticRegression | `StandardScaler` trong Pipeline | Linear model cần scaling |
| GA-SVM | `StandardScaler` + `SimpleImputer` bên trong `fit()` | SVM kernel RBF cần scaling |
| 7 models còn lại | Không scaling | Tree-based/probabilistic không cần scaling |

**Đánh giá:** Hợp lý. Tree-based models (RF, GB, XGB, LGBM, CatBoost) và NGBoost không cần scaling. GenerativeNaiveBayes hoạt động trên allele counts (0/1/2) nên cũng không cần.

#### b) GA-SVM internal CV

GA-SVM sử dụng `cv_folds=3` cho fitness evaluation bên trong GA (nested), khác với outer CV 5-fold. Đây là thiết kế hợp lý vì:
- GA đã rất tốn computation (50 cá thể × 40 thế hệ × 3-fold = 6,000 SVM fits mỗi tuning iteration)
- Inner CV là phần của model, không phải protocol đánh giá

#### c) CatBoost tuning parallelism

```python
search_jobs = 1 if name == "CatBoost" else -1  # line 294
```

CatBoost tuning chạy single-thread do CatBoost tự parallel hóa bên trong. **Không ảnh hưởng kết quả**, chỉ tốc độ.

#### d) Random seed naming

| Model | Parameter name | Value |
|-------|---------------|-------|
| CatBoost | `random_seed` | 42 |
| Tất cả models khác | `random_state` | 42 |

Chỉ khác tên API, giá trị giống nhau. **Không ảnh hưởng.**

---

## 4. Phân tích Classic Pipeline (5 models)

### 4.1. Thiết lập

```python
# run_classic_models.py:run_stage()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- Cùng split ratio và seed → **cùng train/test partition** với main pipeline.
- **NHƯNG:** không có hyperparameter tuning, không có cross-validation.
- Models chạy với **fixed hyperparameters** (hardcoded):

| Model | Key Hyperparameters (fixed) |
|-------|----------------------------|
| GA-SVM | `pop_size=50, n_generations=40, svm_C=10.0, cv_folds=3` |
| SVD-MLP-Adv | `n_components=20, hidden_sizes=(128,64), epochs=200, patience=20` |
| DietNetworks | `embed_dim=64, aux_hidden=64, epochs=200, patience=20` |
| popVAE | `latent_dim=10, enc_hidden=(128,64), epochs=200, patience=20` |
| FederatedMLP | `n_clients=5, hidden_sizes=(128,64), n_rounds=20, local_epochs=5` |

### 4.2. Vấn đề so sánh

So sánh trực tiếp kết quả giữa 2 pipeline là **không công bằng** vì:

1. **Main pipeline models được tối ưu hyperparameters** (20 iterations RandomizedSearchCV) → có lợi thế
2. **Classic models dùng default/fixed params** → có thể chưa đạt hiệu suất tốt nhất
3. **Không có CV metrics** cho classic models → không thể so sánh variance/stability

---

## 5. Kết quả thực tế (hiện có)

### Stage 1 — Continental super-population (5 classes)

| Model | Pipeline | Test ACC | Balanced ACC | F1 Macro | Tuned? |
|-------|----------|----------|--------------|----------|--------|
| LogisticRegression | Main | **0.9741** | 0.9676 | 0.9685 | Yes |
| popVAE | Classic | 0.9701 | 0.9627 | 0.9635 | No |
| SVD-MLP-Adv | Classic | 0.9701 | 0.9626 | 0.9636 | No |
| GradientBoosting | Main | 0.9681 | 0.9611 | 0.9615 | Yes |
| LightGBM | Main | 0.9681 | 0.9594 | 0.9609 | Yes |
| FederatedMLP | Classic | 0.9681 | 0.9594 | 0.9610 | No |
| XGBoost | Main | 0.9661 | 0.9583 | 0.9591 | Yes |
| RandomForest | Main | 0.9621 | 0.9485 | 0.9519 | Yes |
| CatBoost | Main | 0.9621 | 0.9516 | 0.9535 | Yes |
| GA-SVM | Classic | 0.9541 | 0.9444 | 0.9448 | No |
| NGBoost | Main | 0.9361 | 0.9248 | 0.9237 | Yes |
| DietNetworks | Classic | 0.9281 | 0.9045 | 0.9070 | No |

### Stage 2 — East Asian sub-population (5 classes)

| Model | Pipeline | Test ACC | Balanced ACC | F1 Macro | Tuned? |
|-------|----------|----------|--------------|----------|--------|
| LogisticRegression | Main | **0.7723** | 0.7730 | 0.7645 | Yes |
| popVAE | Classic | 0.7624 | 0.7625 | 0.7486 | No |
| FederatedMLP | Classic | 0.7624 | 0.7610 | 0.7527 | No |
| CatBoost | Main | 0.7525 | 0.7540 | 0.7439 | Yes |
| GradientBoosting | Main | 0.7426 | 0.7429 | 0.7356 | Yes |
| RandomForest | Main | 0.7228 | 0.7222 | 0.7139 | Yes |
| SVD-MLP-Adv | Classic | 0.7030 | 0.7049 | 0.6849 | No |
| XGBoost | Main | 0.6931 | 0.6932 | 0.6846 | Yes |
| LightGBM | Main | 0.6832 | 0.6821 | 0.6779 | Yes |
| GA-SVM | Classic | 0.6535 | 0.6551 | 0.6462 | No |
| NGBoost | Main | 0.5644 | 0.5675 | 0.5516 | Yes |
| DietNetworks | Classic | 0.5545 | 0.5583 | 0.5539 | No |

### Models chưa có kết quả

| Model | Pipeline | Ghi chú |
|-------|----------|---------|
| GenerativeNaiveBayes | Main | Không xuất hiện trong results — có thể lỗi hoặc chưa chạy |
| GA-SVM | Main (có đăng ký) | Chỉ có kết quả từ classic pipeline |

---

## 6. Kết luận và Khuyến nghị

### 6.1. Kết luận

| Phạm vi | Kết luận |
|---------|----------|
| **Trong main pipeline** (7 models đã chạy) | **NHẤT QUÁN** — cùng split, cùng tuning, cùng CV, cùng metrics |
| **Trong classic pipeline** (5 models) | **NHẤT QUÁN** nội bộ — cùng split, cùng metrics (nhưng không có tuning/CV) |
| **Giữa 2 pipeline** | **KHÔNG NHẤT QUÁN** — main có tuning + CV, classic không có |

### 6.2. Khuyến nghị

#### Ưu tiên cao (ảnh hưởng tính hợp lệ của so sánh)

1. **Thống nhất protocol đánh giá:** Nếu muốn so sánh trực tiếp tất cả models, classic models cũng cần được đưa qua `RandomizedSearchCV` hoặc ít nhất cross-validation evaluation.

2. **Chạy lại GenerativeNaiveBayes và GA-SVM** trong main pipeline để có kết quả đầy đủ 9 models.

3. **Thêm CV evaluation cho classic models:** Ít nhất chạy `cross_validate()` sau khi train để có CV metrics tương đương.

#### Ưu tiên thấp (cải thiện thêm)

4. **Ghi rõ trong paper/report** rằng classic models dùng fixed hyperparameters (nếu giữ nguyên setup hiện tại), kèm lý do (ví dụ: "tuning không khả thi cho deep models với dataset nhỏ").

5. **Xem xét early stopping** cho neural network models (SVD-MLP-Adv, DietNetworks, popVAE, FederatedMLP) như một hình thức implicit regularization tương đương tuning.

---

## 7. Tham chiếu Code

| File | Vai trò |
|------|---------|
| `src/model_registry.py` | Định nghĩa tất cả models, param grids, constants |
| `src/training.py` | Training engine (tuning + CV) |
| `src/evaluation.py` | Metrics computation + plots |
| `src/preprocessing.py` | Data loading + genotype encoding |
| `scripts/run_pipeline.py` | Main pipeline entry point |
| `scripts/run_classic_models.py` | Classic models entry point |
