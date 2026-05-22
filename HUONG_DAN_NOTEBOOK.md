# 📓 Hướng Dẫn Sử Dụng Notebook

## 🎯 Notebook: carbon24-anomaly-energy-prediction.ipynb

### Nội Dung

Notebook này bao gồm 3 phần chính:

1. **🔍 Phát Hiện Dị Biệt (Anomaly Detection)**
   - Isolation Forest
   - Local Outlier Factor (LOF)
   - One-Class SVM
   - Statistical Methods (Z-score)
   - Consensus anomalies

2. **⚡ Phân Loại Stability**
   - K-means clustering (k=3)
   - Phân loại: Highly Stable, Moderately Stable, Less Stable
   - PCA visualization

3. **🤖 Dự Đoán Energy per Atom**
   - Random Forest
   - Gradient Boosting
   - Ridge Regression
   - Lasso Regression
   - Feature importance analysis

---

## 🚀 Cách Sử Dụng

### Bước 1: Mở Notebook

```bash
jupyter notebook carbon24-anomaly-energy-prediction.ipynb
```

Hoặc với JupyterLab:

```bash
jupyter lab carbon24-anomaly-energy-prediction.ipynb
```

### Bước 2: Chạy Tất Cả Cells

**Cách 1: Chạy từng cell**
- Click vào cell
- Nhấn `Shift + Enter` để chạy và chuyển sang cell tiếp theo
- Hoặc nhấn `Ctrl + Enter` để chạy và ở lại cell hiện tại

**Cách 2: Chạy tất cả**
- Menu: `Cell` → `Run All`
- Hoặc: `Kernel` → `Restart & Run All`

### Bước 3: Xem Kết Quả

Notebook sẽ tạo ra:
- **Visualizations** - Hiển thị trực tiếp trong notebook
- **Statistics** - In ra console
- **CSV files** - Lưu trong folder `carbon24_anomaly_prediction_results/`

---

## 📊 Kết Quả Mong Đợi

### 1. Anomaly Detection

**Output:**
```
🌲 Isolation Forest
============================================================
Anomalies detected: 507 (5.00%)
Normal structures: 9,646

📍 Local Outlier Factor (LOF)
============================================================
Anomalies detected: 507 (5.00%)
Normal structures: 9,646

🎯 One-Class SVM
============================================================
Anomalies detected: 507 (5.00%)
Normal structures: 9,646

📊 Statistical Methods (Z-score)
============================================================
Anomalies detected (|z| > 3): ~300 (3.00%)
Normal structures: ~9,850

🎯 Consensus Anomaly Detection
============================================================
Consensus anomalies (≥2 methods): ~400 (4.00%)
```

**Visualizations:**
- Anomaly count distribution
- Energy distribution: Normal vs Anomaly
- Isolation Forest scores
- Method agreement comparison

### 2. Stability Classification

**Output:**
```
🔄 K-means Clustering
============================================================
✅ Clustering completed with k=3

Cluster distribution:
  Cluster 0: 3,409 (33.6%)
  Cluster 1: 2,229 (22.0%)
  Cluster 2: 4,515 (44.5%)

📊 Stability Analysis by Cluster
============================================================
         count    mean     std     min     max  median
cluster
0         3409 -0.4725  1.0173 -2.1775  1.3348 -0.4371
1         2229 -0.3771  1.1525 -1.7323  1.3348 -1.1366
2         4515  0.5429  0.5278 -1.4196  1.3348  0.6060

🎯 Stability Classification:
  Cluster 0: 🟢 Highly Stable
    - Structures: 3,409
    - Mean energy: -0.4725 eV/atom

  Cluster 1: 🟢 Highly Stable
    - Structures: 2,229
    - Mean energy: -0.3771 eV/atom

  Cluster 2: 🔴 Less Stable
    - Structures: 4,515
    - Mean energy: 0.5429 eV/atom
```

**Visualizations:**
- Energy distribution by cluster
- Box plot comparison
- Cluster size distribution
- PCA visualization

### 3. Energy Prediction

**Output:**
```
🤖 Training ML Models
============================================================

Training Random Forest...
  ✅ Test R²: 1.0000, Test MAE: 0.0014

Training Gradient Boosting...
  ✅ Test R²: 1.0000, Test MAE: 0.0042

Training Ridge...
  ✅ Test R²: 0.8988, Test MAE: 1.5101

Training Lasso...
  ✅ Test R²: 0.8297, Test MAE: 2.1678

============================================================
📊 MODEL COMPARISON
============================================================
            Model  Train R²  Test R²  Train MAE  Test MAE  Train RMSE  Test RMSE
    Random Forest  1.000000 1.000000   0.000536  0.001413    0.000991   0.002569
Gradient Boosting  0.999999 0.999999   0.004081  0.004187    0.005360   0.005549
 Ridge Regression  0.903917 0.898774   1.493332  1.510104    1.786694   1.806484
 Lasso Regression  0.836049 0.829748   2.148136  2.167753    2.333906   2.342797

🏆 Best Model: Random Forest
   Test R²: 1.0000
   Test MAE: 0.0014 eV/atom
```

**Visualizations:**
- Actual vs Predicted scatter plot
- Residual plot
- Error distribution
- Model comparison bar chart
- Feature importance (top 15)

---

## 📁 Output Files

Sau khi chạy notebook, các files sau sẽ được tạo trong `carbon24_anomaly_prediction_results/`:

```
carbon24_anomaly_prediction_results/
├── anomaly_detection_results.csv      # Kết quả phát hiện dị biệt
├── stability_classification.csv       # Phân loại stability
├── model_comparison.csv               # So sánh các mô hình
└── energy_predictions.csv             # Kết quả dự đoán energy
```

### File Descriptions

**1. anomaly_detection_results.csv**
```
Columns:
- row_index: ID của cấu trúc
- is_anomaly: 1 nếu là anomaly (consensus), 0 nếu normal
- anomaly_count: Số phương pháp đánh dấu là anomaly
- anomaly_iso_forest: Kết quả từ Isolation Forest (-1 = anomaly, 1 = normal)
- anomaly_lof: Kết quả từ LOF
- anomaly_ocsvm: Kết quả từ One-Class SVM
- anomaly_zscore: Kết quả từ Z-score
- relative_energy: Năng lượng tương đối
- energy_per_atom: Năng lượng trên mỗi nguyên tử
```

**2. stability_classification.csv**
```
Columns:
- row_index: ID của cấu trúc
- cluster: Cluster ID (0, 1, 2)
- relative_energy: Năng lượng tương đối
- energy_per_atom: Năng lượng trên mỗi nguyên tử
```

**3. model_comparison.csv**
```
Columns:
- Model: Tên mô hình
- Train R²: R² score trên training set
- Test R²: R² score trên test set
- Train MAE: MAE trên training set
- Test MAE: MAE trên test set
- Train RMSE: RMSE trên training set
- Test RMSE: RMSE trên test set
```

**4. energy_predictions.csv**
```
Columns:
- actual: Giá trị thực tế
- predicted: Giá trị dự đoán
- error: Sai số (actual - predicted)
- abs_error: Sai số tuyệt đối
```

---

## 💡 Tips & Tricks

### 1. Chạy Nhanh Hơn

Nếu muốn chạy nhanh hơn, giảm số lượng estimators:

```python
# Trong cell "Train Models"
models = {
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),  # Giảm từ 100
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
    ...
}
```

### 2. Thay Đổi Contamination Rate

Để phát hiện nhiều/ít anomalies hơn:

```python
# Trong cells anomaly detection
iso_forest = IsolationForest(
    contamination=0.10,  # Tăng lên 10% thay vì 5%
    ...
)
```

### 3. Thay Đổi Số Clusters

```python
# Trong cell "Clustering"
optimal_k = 4  # Thay vì 3
```

### 4. Xem Thêm Features

```python
# Trong cell "Feature Importance"
top_n = 20  # Thay vì 15
```

### 5. Export Figures

Thêm vào cuối mỗi cell visualization:

```python
plt.savefig('my_figure.png', dpi=300, bbox_inches='tight')
```

---

## 🔧 Troubleshooting

### Lỗi: Module not found

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Lỗi: File not found

Đảm bảo các file dữ liệu tồn tại:
- `carbon24_feature_selected/carbon24_feature_selected_standard.csv`
- `carbon24_features/carbon24_project/data/carbon24_features.csv`
- `carbon24_feature_selected/selected_features.json`

### Lỗi: Memory error

Giảm sample size hoặc số features:

```python
# Sample data
df_sample = df.sample(n=5000, random_state=42)
X_anomaly = df_sample[features_for_anomaly].values
```

### Kernel bị treo

- `Kernel` → `Interrupt`
- Hoặc `Kernel` → `Restart`

---

## 📊 Phân Tích Kết Quả

### 1. Anomalies

**Câu hỏi:** Cấu trúc nào là anomaly?

```python
# Trong notebook, thêm cell mới:
anomalies = df[df['is_anomaly'] == 1]
print(f"Found {len(anomalies)} anomalies")
print(anomalies[['row_index', 'relative_energy', 'energy_per_atom']].head(10))
```

**Phân tích:**
- Anomalies thường có năng lượng rất cao hoặc rất thấp
- Có thể là cấu trúc đặc biệt hoặc lỗi dữ liệu
- Cần kiểm tra thêm để xác định nguyên nhân

### 2. Stability Groups

**Câu hỏi:** Cluster nào ổn định nhất?

```python
# Xem mean energy của mỗi cluster
df.groupby('cluster')['relative_energy'].mean().sort_values()
```

**Phân tích:**
- Cluster có mean energy thấp nhất là ổn định nhất
- Clusters 0 & 1: Highly Stable (phù hợp cho ứng dụng)
- Cluster 2: Less Stable (năng lượng cao hơn)

### 3. Energy Prediction

**Câu hỏi:** Feature nào quan trọng nhất?

```python
# Xem feature importance
feature_importance_df.head(10)
```

**Phân tích:**
- `num_atoms` là quan trọng nhất (99.99%)
- Các features về bond length cũng quan trọng
- Có thể dự đoán energy từ cấu trúc hình học

---

## 🎓 Học Thêm

### Anomaly Detection
- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [LOF Algorithm](https://en.wikipedia.org/wiki/Local_outlier_factor)
- [One-Class SVM](https://scikit-learn.org/stable/modules/svm.html#svm-outlier-detection)

### Machine Learning
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [Feature Importance](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

### Clustering
- [K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)

---

## 📞 Support

Nếu gặp vấn đề:

1. Kiểm tra output của mỗi cell
2. Xem error messages
3. Restart kernel và chạy lại
4. Kiểm tra data files

---

## 🎉 Kết Luận

Notebook này cung cấp:

✅ **Phát hiện dị biệt** với 4 phương pháp  
✅ **Phân loại stability** với clustering  
✅ **Dự đoán energy** với ML (R² = 1.0000)  
✅ **Visualizations** đẹp và dễ hiểu  
✅ **Export results** sang CSV  

**Chúc bạn phân tích thành công! 🎊**

---

**Quick Start:**
```bash
jupyter notebook carbon24-anomaly-energy-prediction.ipynb
```

**Sau đó:** `Cell` → `Run All`
