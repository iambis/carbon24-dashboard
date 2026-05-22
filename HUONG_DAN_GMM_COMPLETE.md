# 📊 Hướng Dẫn GMM Clustering - Phiên Bản Hoàn Chỉnh

## 🎯 Notebook: carbon24-gmm-clustering.ipynb

### ✨ Điểm Mới So Với Phiên Bản Cũ

**Phiên bản cũ thiếu:**
- ❌ Không hiển thị kết quả phân cụm cụ thể (số lượng mẫu trong mỗi cluster)
- ❌ Không có trực quan hóa PCA 3D
- ❌ Thiếu phân tích chi tiết các clusters

**Phiên bản mới có:**
- ✅ **Kết quả phân cụm cụ thể** với số lượng mẫu và phần trăm
- ✅ **Trực quan hóa PCA 3D tương tác** (có thể xoay, zoom, hover)
- ✅ **Phân tích uncertainty** (mẫu không chắc chắn)
- ✅ **Cluster profiles** chi tiết
- ✅ **Quality metrics** đầy đủ
- ✅ **Export kết quả** hoàn chỉnh

---

## 🚀 Quick Start

```bash
jupyter notebook carbon24-gmm-clustering.ipynb
```

Sau đó: **Cell → Run All**

**Thời gian chạy:** ~3-5 phút (tùy máy)

---

## 📊 Nội Dung Notebook (15 Cells)

### 1. 📂 Load Data & Preparation (2 cells)
- Import thư viện (pandas, sklearn, plotly)
- Load dữ liệu features đã chuẩn hóa
- Chuẩn bị dữ liệu cho clustering

### 2. 🔍 GMM Model Selection (3 cells)
- Chạy GMM với n_components từ 2 đến 10
- Tính AIC và BIC cho mỗi model
- Chọn số clusters tối ưu (thường theo BIC)
- **Visualization:** Biểu đồ AIC vs BIC

### 3. 🎯 Clustering Results (2 cells)
- Predict clusters với model tối ưu
- **✨ KẾT QUẢ PHÂN CỤM CỤ THỂ:**
  - Số lượng mẫu trong mỗi cluster
  - Phần trăm phân bố
  - Năng lượng trung bình của mỗi cluster
  - Xác suất trung bình
- **Visualization:** Bar chart + Pie chart

### 4. 📊 PCA 3D Visualization (3 cells)
- Thực hiện PCA 3 chiều
- **✨ TRỰC QUAN HÓA 3D TƯƠNG TÁC:**
  - 3D scatter plot theo clusters
  - 3D scatter plot theo năng lượng
  - Có thể xoay, zoom, hover để xem chi tiết
  - Hiển thị variance explained của mỗi PC

### 5. 📈 Clustering Quality Metrics (1 cell)
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Interpretation tự động

### 6. 🔍 Uncertainty Analysis (1 cell)
- Phân tích mẫu có xác suất thấp (< 0.7)
- Histogram phân bố xác suất
- Boxplot xác suất theo cluster

### 7. 📋 Cluster Profiles (1 cell)
- Bảng đặc điểm của từng cluster
- Xác định cluster ổn định nhất

### 8. 💾 Export Results (1 cell)
- Lưu dữ liệu đã phân cụm
- Lưu cluster profiles
- Lưu uncertain samples
- Lưu clustering report (JSON)

---

## 📁 Input Files

```
carbon24_feature_selected/
├── carbon24_feature_selected_standard.csv  # Dữ liệu đã chuẩn hóa
└── selected_features.json                   # Danh sách features
```

---

## 📊 Output Files

Sau khi chạy, các files sẽ được tạo trong `carbon24_gmm_results/`:

```
carbon24_gmm_results/
├── results/
│   └── carbon24_gmm_results.csv           # Dữ liệu + cluster labels + PCA
├── tables/
│   ├── gmm_cluster_profile.csv            # Đặc điểm các clusters
│   └── gmm_uncertain_samples.csv          # Mẫu không chắc chắn
└── gmm_clustering_report.json             # Báo cáo tổng hợp
```

---

## 🎯 Kết Quả Mong Đợi

### Cluster Distribution (Ví dụ với n=3)

```
Cluster 0:
  Số lượng mẫu: 3,500 (34.47%)
  Năng lượng trung bình: 0.2500 ± 0.0800 eV/atom
  Xác suất trung bình: 0.9200

Cluster 1:
  Số lượng mẫu: 4,200 (41.37%)
  Năng lượng trung bình: 0.3200 ± 0.1200 eV/atom
  Xác suất trung bình: 0.8800

Cluster 2:
  Số lượng mẫu: 2,453 (24.16%)
  Năng lượng trung bình: 0.4500 ± 0.1500 eV/atom
  Xác suất trung bình: 0.8500
```

### Quality Metrics (Ví dụ)

```
Silhouette Score:      0.2500  (Weak separation)
Davies-Bouldin Index:  1.4500  (Acceptable)
Calinski-Harabasz:     2400.50 (Excellent)
```

### PCA Variance Explained

```
PC1: 24.57% variance
PC2: 15.38% variance
PC3: 10.25% variance
Total: 50.20% variance explained
```

---

## 💡 Cách Sử Dụng Trực Quan Hóa 3D

### Interactive 3D Plot Features:

1. **Xoay (Rotate):**
   - Click và kéo chuột để xoay
   - Xem clusters từ nhiều góc độ

2. **Zoom:**
   - Scroll chuột để zoom in/out
   - Hoặc dùng nút zoom trên toolbar

3. **Hover:**
   - Di chuột qua điểm để xem thông tin:
     - PCA coordinates
     - Cluster assignment
     - Probability
     - Relative energy
     - Material ID

4. **Pan:**
   - Shift + Click và kéo để di chuyển

5. **Reset:**
   - Double-click để reset về view ban đầu

6. **Save:**
   - Nút camera để lưu ảnh PNG

---

## 🔧 Customization

### 1. Thay Đổi Số Clusters

```python
# Trong cell "Select best model"
optimal_n = 5  # Thay vì dùng optimal_n_bic
best_gmm = models[optimal_n]
```

### 2. Thay Đổi Uncertainty Threshold

```python
# Trong cell "Select best model"
df['Is_Uncertain'] = max_probs < 0.6  # Thay vì 0.7
```

### 3. Thay Đổi PCA Components

```python
# Trong cell "PCA for visualization"
pca_3d = PCA(n_components=4, random_state=42)  # Thêm PC4
```

### 4. Thay Đổi Color Scheme

```python
# Trong cell "Interactive 3D scatter plot"
color_discrete_sequence=px.colors.qualitative.Pastel  # Thay vì Set3
```

---

## 📊 So Sánh Với Các Phương Pháp Khác

### GMM vs K-means

**GMM:**
- ✅ Soft clustering (xác suất)
- ✅ Có thể model elliptical clusters
- ✅ Uncertainty quantification
- ❌ Chậm hơn K-means
- ❌ Phức tạp hơn

**K-means:**
- ✅ Nhanh và đơn giản
- ✅ Dễ hiểu và implement
- ❌ Hard clustering only
- ❌ Chỉ model spherical clusters

### Khi Nào Dùng GMM?

✅ **Nên dùng GMM khi:**
- Cần xác suất phân cụm (soft clustering)
- Clusters có hình dạng elliptical
- Cần uncertainty quantification
- Có overlapping clusters
- Muốn model phức tạp hơn

❌ **Không nên dùng GMM khi:**
- Dataset rất lớn (> 100,000 samples)
- Cần tốc độ nhanh
- Clusters rõ ràng và spherical
- Chỉ cần hard clustering

---

## 🔍 Troubleshooting

### Issue 1: Notebook chạy chậm

**Giải pháp:**
```python
# Giảm sample size cho Silhouette Score
if len(X) > 3000:
    sample_idx = np.random.choice(len(X), 3000, replace=False)
```

### Issue 2: 3D plot không hiển thị

**Giải pháp:**
1. Cài đặt plotly: `pip install plotly`
2. Restart kernel
3. Chạy lại cell

### Issue 3: Memory error

**Giải pháp:**
```python
# Giảm số components test
n_components_range = range(2, 6)  # Thay vì 2-10
```

### Issue 4: Convergence warning

**Giải pháp:**
```python
# Tăng max_iter
gmm = GaussianMixture(n_components=n, max_iter=500)  # Thay vì 200
```

---

## 📚 Hiểu Về GMM

### Gaussian Mixture Model là gì?

GMM giả định rằng dữ liệu được tạo ra từ một mixture của nhiều Gaussian distributions.

**Công thức:**
```
P(x) = Σ πk * N(x | μk, Σk)
```

Trong đó:
- `πk`: Mixing coefficient (weight) của component k
- `N(x | μk, Σk)`: Gaussian distribution với mean μk và covariance Σk

### AIC vs BIC

**AIC (Akaike Information Criterion):**
- Đo độ fit của model
- Penalty nhẹ hơn cho complexity
- Thường chọn nhiều clusters hơn

**BIC (Bayesian Information Criterion):**
- Đo độ fit của model
- Penalty nặng hơn cho complexity
- Thường chọn ít clusters hơn (conservative)
- **Khuyến nghị:** Dùng BIC cho production

### Soft vs Hard Clustering

**Soft Clustering (GMM):**
- Mỗi điểm có xác suất thuộc về mỗi cluster
- Ví dụ: Point A: 70% Cluster 0, 20% Cluster 1, 10% Cluster 2

**Hard Clustering (K-means):**
- Mỗi điểm chỉ thuộc về 1 cluster
- Ví dụ: Point A: 100% Cluster 0

---

## 🎓 Phân Tích Kết Quả

### 1. Cluster Distribution

**Câu hỏi:**
- Clusters có cân bằng không?
- Có cluster nào quá nhỏ/lớn?

**Hành động:**
- Nếu có cluster < 5%: Xem xét giảm số clusters
- Nếu clusters cân bằng: Good sign!

### 2. Uncertainty Analysis

**Câu hỏi:**
- Bao nhiêu % mẫu uncertain?
- Uncertain samples tập trung ở cluster nào?

**Hành động:**
- Nếu > 20% uncertain: Xem xét tăng số clusters
- Nếu < 10% uncertain: Good confidence!

### 3. Energy Distribution

**Câu hỏi:**
- Cluster nào có năng lượng thấp nhất?
- Có correlation giữa cluster và energy?

**Hành động:**
- Cluster năng lượng thấp = Stable structures
- Sử dụng cho material discovery

### 4. Quality Metrics

**Câu hỏi:**
- Metrics có tốt không?
- So với các phương pháp khác?

**Hành động:**
- So sánh với K-means, Hierarchical, HDBSCAN
- Chọn phương pháp tốt nhất cho use case

---

## 🎉 Next Steps

### Sau Khi Chạy Notebook:

1. **So Sánh Phương Pháp:**
   - Chạy `carbon24-clustering-comparison-evaluation.ipynb`
   - So sánh GMM với K-means, Hierarchical, HDBSCAN

2. **Stability Analysis:**
   - Phân tích stable vs unstable structures
   - Sử dụng cluster labels

3. **Energy Prediction:**
   - Train model dự đoán năng lượng
   - Sử dụng cluster features

4. **Material Discovery:**
   - Tìm structures mới trong stable clusters
   - Optimize properties

---

## 💾 Export & Share

### Export to HTML
```bash
jupyter nbconvert --to html carbon24-gmm-clustering.ipynb
```

### Export to PDF
```bash
jupyter nbconvert --to pdf carbon24-gmm-clustering.ipynb
```

### Share 3D Plots
```python
# Trong notebook, sau khi tạo fig
fig.write_html('gmm_3d_clusters.html')
```

---

## 🎯 Tóm Tắt

### ✅ Notebook Này Cung Cấp:

1. **Kết quả phân cụm cụ thể** - Số lượng mẫu trong mỗi cluster
2. **Trực quan hóa 3D tương tác** - PCA 3D với Plotly
3. **Phân tích uncertainty** - Mẫu không chắc chắn
4. **Cluster profiles** - Đặc điểm chi tiết
5. **Quality metrics** - Đánh giá chất lượng
6. **Export đầy đủ** - Tất cả kết quả

### 📊 Key Features:

- ✨ **Interactive 3D visualization** - Xoay, zoom, hover
- ✨ **Detailed cluster results** - Số lượng, phần trăm, năng lượng
- ✨ **Uncertainty quantification** - Soft clustering advantage
- ✨ **Complete export** - CSV, JSON, profiles

### 🚀 Ready to Use!

```bash
jupyter notebook carbon24-gmm-clustering.ipynb
```

**Chúc bạn phân tích thành công! 🎉**

---

**Quick Reference:**
- **Notebook:** `carbon24-gmm-clustering.ipynb`
- **Output:** `carbon24_gmm_results/`
- **Runtime:** ~3-5 minutes
- **Cells:** 15 cells total
- **Key Features:** Clustering results + 3D PCA visualization
