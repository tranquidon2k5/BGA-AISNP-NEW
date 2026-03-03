# 📁 BGA-AISNP Project Structure

> **Mục đích**: Tài liệu chuẩn hóa cấu trúc dự án, hướng dẫn thêm model mới, và tránh trùng lặp code khi có nhiều người cùng phát triển.

---

## 1. Tổng quan kiến trúc

```
BGA-AISNP-NEW/
│
├── DATA/                          # 📊 Dữ liệu gốc (KHÔNG sửa đổi)
│   ├── AISNP_by_sample_continental.csv   # Stage 1: 2504 samples × 5 super_pop
│   └── AISNP_by_sample_EAS_only.csv      # Stage 2: 504 EAS samples × 5 pop
│
├── src/                           # 🧠 Core library (import bởi tất cả scripts)
│   ├── __init__.py
│   ├── preprocessing.py           # ✅ Unified data loading & genotype encoding
│   ├── model_registry.py          # ✅ Unified model definitions & param grids
│   ├── training.py                # ✅ Unified training engine (train_all)
│   ├── evaluation.py              # ✅ Metrics, plots, confusion matrices
│   ├── generative_model.py        # Bayesian generative BGA classifier
│   └── tabpfn_model.py            # TabPFN wrapper
│
├── scripts/                       # 🚀 Runnable scripts
│   ├── run_pipeline.py         # ✅ MAIN: 2-stage pipeline (--models, --stage)
│   ├── inference.py            # ✅ Inference cho single sample
│   ├── export_excel.py            # ✅ Export metrics ra Excel
│   ├── export_models.py        # ✅ Export trained models (.pkl)
│   └── plot_learning_curves.py # ✅ Vẽ learning curves
│
├── models/                        # 💾 Trained model artifacts (.pkl)
│   ├── models_metadata.csv
│   ├── stage1_super_pop_*.pkl
│   └── stage2_EAS_pop_*.pkl
│
├── results/                       # 📈 Output: CSVs, plots, Excel
│   ├── ALL_model_comparison.csv
│   ├── stage1_model_comparison.csv
│   ├── stage2_model_comparison.csv
│   ├── confusion_*.png
│   ├── comparison_*.png
│   └── feature_importance_*.png
│
├── structure.md                   # 📖 (File này) Hướng dẫn cấu trúc
└── README.md
```

---

## 2. Pipeline hoạt động như thế nào

```
┌──────────────────────────────────────────────────────────┐
│  STAGE 1 — Continental-level classification              │
│  Data   : AISNP_by_sample_continental.csv                │
│  Target : super_pop  (AFR | AMR | EAS | EUR | SAS)       │
│  Models : 7 algorithms (LR, RF, GB, XGB, LGBM, CB, NGB) │
└───────────────────────┬──────────────────────────────────┘
                        │ Nếu predicted = EAS
                        ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 2 — East-Asian sub-population classification      │
│  Data   : AISNP_by_sample_EAS_only.csv                   │
│  Target : pop  (CDX | CHB | CHS | JPT | KHV)             │
│  Models : 7 algorithms (tương tự Stage 1)                 │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Các module trong `src/` — Vai trò & API

### 3.1 `src/preprocessing.py` — Data loading

| Function | Mô tả |
|----------|--------|
| `load_dataset(csv_path)` | Load CSV, encode genotype → `(X, y_super, y_pop, le_dict, snp_ids, meta)` |
| `load_continental()` | Shortcut cho continental CSV |
| `load_eas()` | Shortcut cho EAS CSV |
| `get_snp_ids(columns)` | Lấy danh sách SNP từ tên cột |
| `additive_encode(df, snp_ids)` | Encode genotype _1/_2 → 0/1/2 |
| `split_xy(X, y)` | Stratified train/test split |

### 3.2 `src/model_registry.py` — Model definitions

| Function/Class | Mô tả |
|----------------|--------|
| `build_models(n_classes)` | Dict `{model_name: estimator}` với default params |
| `get_param_grids(n_classes)` | Dict `{model_name: param_distributions}` cho tuning |
| `tune_model(name, model, ...)` | RandomizedSearchCV wrapper |
| `NGBoostWrapper` | sklearn-compatible wrapper cho NGBClassifier |

**Constants**: `RANDOM_STATE=42`, `TEST_SIZE=0.20`, `CV_FOLDS=5`, `N_ITER_SEARCH=20`

### 3.3 `src/training.py` — Training engine

| Function | Mô tả |
|----------|--------|
| `train_all(X, y, target_name)` | Train + tune tất cả models, return `list[dict]` |

### 3.4 `src/evaluation.py` — Evaluation & Visualization

| Function | Mô tả |
|----------|--------|
| `evaluate_results(results, le, ...)` | Evaluate tất cả, save plots, return DataFrame |
| `compute_per_label_metrics(...)` | Per-class P/R/F1/MCC/AUC |
| `plot_confusion_matrix(...)` | Heatmap confusion matrix |
| `plot_bar_comparison(...)` | So sánh models bằng bar chart |
| `plot_feature_importance(...)` | Top-N feature importance |

### 3.5 `src/generative_model.py` — Bayesian Generative Model

| Method | Mô tả |
|--------|--------|
| `fit(X, y, snp_names)` | Ước lượng allele frequencies per population |
| `predict(X)` | Dự đoán label |
| `predict_proba(X)` | Posterior probabilities |
| `predict_with_uncertainty(X, threshold)` | Predict + UNKNOWN nếu low confidence |

### 3.6 `src/tabpfn_model.py` — TabPFN Wrapper

| Function | Mô tả |
|----------|--------|
| `make_tabpfn_classifier(device)` | Tạo TabPFNClassifier instance |

---

## 4. ⭐ Hướng dẫn thêm Model mới

### Bước 1: Đăng ký model trong `src/model_registry.py`

Mở file `src/model_registry.py`, thêm vào 2 chỗ:

#### a) Thêm default estimator trong `build_models()`

```python
def build_models(n_classes: int) -> dict[str, BaseEstimator]:
    return {
        # ... models hiện tại ...
        
        # ✅ THÊM MODEL MỚI Ở ĐÂY
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
    }
```

#### b) Thêm param grid trong `get_param_grids()`

```python
def get_param_grids(n_classes: int) -> dict[str, dict]:
    return {
        # ... grids hiện tại ...
        
        # ✅ THÊM GRID CHO MODEL MỚI
        "SVM": {
            "clf__C":      loguniform(0.01, 100),
            "clf__gamma":  loguniform(1e-4, 1),
            "clf__kernel": ["rbf", "poly"],
        },
    }
```

> **Lưu ý**: Nếu model cần `Pipeline` (ví dụ cần scaling), prefix param key bằng `clf__`.

### Bước 2: Import thư viện (nếu cần)

Thêm import ở đầu file `src/model_registry.py`:

```python
from sklearn.svm import SVC
```

### Bước 3: Test

```bash
# Chạy pipeline để test model mới
python scripts/run_pipeline.py --stage 1
```

### Bước 4: (Tùy chọn) Model cần custom wrapper

Nếu model không compatible với sklearn API, tạo wrapper class giống `NGBoostWrapper`:

```python
class MyModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, ...):
        ...
    
    def fit(self, X, y):
        ...
        return self
    
    def predict(self, X):
        ...
    
    def predict_proba(self, X):
        ...
```

### Bước 5: Cập nhật tài liệu

Thêm dòng mô tả model mới vào phần "Algorithms" ở đầu `src/model_registry.py`.

---

## 5. Quy tắc chung khi phát triển

### ✅ NÊN làm

| Quy tắc | Giải thích |
|---------|------------|
| Import từ `src.*` | `from src.preprocessing import load_continental` |
| Dùng `Path(__file__).resolve().parent.parent` | Để xác định project root |
| Thêm model qua `model_registry.py` | Không tạo file train riêng cho mỗi model |
| Viết docstring cho function | Giúp đồng đội hiểu code |
| Đặt constant ở `model_registry.py` | `RANDOM_STATE`, `TEST_SIZE`, etc. |

### ❌ KHÔNG NÊN làm

| Quy tắc | Lý do |
|---------|-------|
| Tạo file `train_<model>_<dataset>.py` riêng | Gây duplicate — dùng `model_registry.py` |
| Hardcode path `"data/"` hay `"DATA/"` | Dùng constant từ `src/preprocessing.py` |
| Dùng `sys.path.append(...)` kiểu cũ | Dùng `PROJECT_ROOT` pattern thống nhất |
| Viết preprocessing mới | Dùng `src/preprocessing.py` đã có |
| Copy-paste code evaluate metrics | Dùng `src/evaluation.py` |

---

## 6. Cách chạy

### Pipeline chính (train + evaluate tất cả models)
```bash
python scripts/run_pipeline.py              # Cả 2 stages, tất cả models
python scripts/run_pipeline.py --stage 1    # Chỉ Stage 1
python scripts/run_pipeline.py --stage 2    # Chỉ Stage 2
```

### Chạy 1 model cụ thể
```bash
python scripts/run_pipeline.py --models XGBoost                # Chỉ XGBoost
python scripts/run_pipeline.py --models XGBoost LightGBM       # 2 models
python scripts/run_pipeline.py --stage 1 --models CatBoost     # Stage 1 + CatBoost only
```

### Xem danh sách models có sẵn
```bash
python scripts/run_pipeline.py --list-models
```

### Export metrics ra Excel
```bash
python scripts/export_excel.py
```

### Export models (.pkl)
```bash
python scripts/export_models.py
```

### Inference cho 1 sample
```bash
python scripts/inference.py --sample HG00096
```

### Vẽ learning curves
```bash
python scripts/plot_learning_curves.py
```

---

## 7. Danh sách Models hiện tại

| # | Model | Type | Cần scaling? | Notes |
|---|-------|------|-------------|-------|
| 1 | LogisticRegression | Linear | ✅ (Pipeline) | Baseline tuyến tính |
| 2 | RandomForest | Bagging | ❌ | Feature importance |
| 3 | GradientBoosting | Boosting (sklearn) | ❌ | Chậm hơn XGB/LGBM |
| 4 | XGBoost | Boosting | ❌ | Strong regularization |
| 5 | LightGBM | Boosting | ❌ | Leaf-wise, nhanh |
| 6 | CatBoost | Boosting | ❌ | Ordered boosting |
| 7 | NGBoost | Probabilistic boosting | ❌ | Natural gradient |
| + | GenerativeBGA | Bayesian | ❌ | Custom (not in registry) |
| + | TabPFN | Transformer | ❌ | Pre-trained, no tuning |

---

## 8. Checklist khi thêm model mới

- [ ] Import classifier trong `src/model_registry.py`
- [ ] Thêm tên vào `ALL_MODEL_NAMES` list
- [ ] Thêm entry trong `build_models()`
- [ ] Thêm param grid trong `get_param_grids()`
- [ ] Nếu cần wrapper → tạo class kế thừa `BaseEstimator, ClassifierMixin`
- [ ] Chạy test: `python scripts/run_pipeline.py --stage 1 --models TenModelMoi`
- [ ] Kiểm tra kết quả trong `results/`
- [ ] Cập nhật bảng models trong file này (Section 7)
- [ ] Commit & push

---

*Cập nhật lần cuối: 2026-03-03*
