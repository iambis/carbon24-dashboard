# 📊 Hướng Dẫn So Sánh Các Phương Pháp Clustering

## 🎯 Notebook: carbon24-clustering-comparison-evaluation.ipynb

### Mục Đích

So sánh và đánh giá 4 phương pháp clustering đã thực hiện:
1. **K-means** - Phân cụm dựa trên centroid
2. **GMM** - Gaussian Mixture Model
3. **Hierarchical** - Phân cụm phân cấp
4. **HDBSCAN** - Phân cụm dựa trên mật độ

---

## 🚀 Quick Start

```bash
jupyter notebook carbon24-clustering-comparison-evaluation.ipynb
```

Sau đó: `Cell` → `Run All`

---

## 📊 Nội Dung Notebook (21 Cells)

### 1. 📂 Load Clustering Results (3 cells)
- Load kết quả từ 4 phương pháp
- Kiểm tra data integrity
- Overview số lượng samples và clusters

### 2. 📈 Cluster Distribution Analysis (3 cells)
- Phân tích phân bố clusters
- So sánh kích thước clusters
- Visualizations: Bar charts cho mỗi phương pháp

### 3. 🎯 Clustering Quality Metrics (4 cells)
- **Silhouette Score** (↑ higher is better)
- **Davies-Bouldin Index** (↓ lower is better)
- **Calinski-Harabasz Score** (↑ higher is better)
- Visualizations: 3 bar charts so sánh

### 4. 🏆 Method Ranking (2 cells)
- Ranking theo từng metric
- Overall ranking (rank-based scoring)
- Xác định phương pháp tốt nhất

### 5. ⚡ Energy Analysis (3 cells)
- Phân tích năng lượng theo clusters
- Tìm cluster ổn định nhất
- Visualizations: Energy distributions

### 6. 📊 Summary & Recommendations (3 cells)
- Tổng kết findings
- Recommendations cho từng use case
- Export results to CSV

### 7. 💾 Export Results (1 cell)
- Save comparison tables
- Save metrics
- Save rankings

---

## 📁 Input Files

Notebook sẽ load từ các folders:

```
carbon24_kmeans_results/
├── carbon24_clustered.csv
└── clustering_report.json

carbon24_gmm_results/
└── results/*.csv

carbon24_hierarchical_baseline/
└── results/*.csv

hdbscan_phuc/
├── hdbscan_results.csv
└── hdbscan_cluster_profile.csv
```

---

## 📊 Output Files

Sau khi chạy, các files sẽ được tạo trong `carbon24_clustering_comparison_results/`:

```
carbon24_clustering_comparison_results/
├── methods_overview.csv          # Tổng quan các phương pháp
├── quality_metrics.csv            # Metrics của từng phương pháp
└── method_ranking.csv             # Ranking tổng hợp
```

---

## 🎯 Metrics Explained

### Silhouette Score (↑ higher is better)
- **Range:** [-1, 1]
- **Ý nghĩa:** Đo độ tách biệt giữa các clusters
- **Good:** > 0.5
- **Excellent:** > 0.7

### Davies-Bouldin Index (↓ lower is better)
- **Range:** [0, ∞)
- **Ý nghĩa:** Đo độ compact và separation
- **Good:** < 1.0
- **Excellent:** < 0.5

### Calinski-Harabasz Score (↑ higher is better)
- **Range:** [0, ∞)
- **Ý nghĩa:** Tỷ lệ variance giữa/trong clusters
- **Good:** > 100
- **Excellent:** > 1000

---

## 📈 Kết Quả Mong Đợi

### Overview Table
```
Method         Samples  Clusters  Noise Points  Noise %
K-means        10,153   3         0             0.00%
GMM            10,153   3         0             0.00%
Hierarchical   10,153   3         0             0.00%
HDBSCAN        10,153   5-10      ~500          ~5.00%
```

### Metrics Comparison
```
Method         Silhouette  Davies-Bouldin  Calinski-Harabasz
K-means        0.45-0.55   0.8-1.2         800-1200
GMM            0.40-0.50   0.9-1.3         700-1100
Hierarchical   0.42-0.52   0.85-1.25       750-1150
HDBSCAN        0.35-0.45   1.0-1.5         600-1000
```

### Ranking
```
🥇 1. K-means/GMM/Hierarchical (tùy dataset)
🥈 2. ...
🥉 3. ...
   4. HDBSCAN (thường thấp hơn do noise points)
```

---

## 💡 Interpretation Guide

### Nếu K-means thắng:
- ✅ Clusters có hình dạng spherical
- ✅ Kích thước clusters tương đối đồng đều
- ✅ Số clusters k=3 phù hợp
- 📌 **Recommendation:** Sử dụng K-means cho production

### Nếu GMM thắng:
- ✅ Clusters có hình dạng elliptical
- ✅ Cần probabilistic assignments
- ✅ Overlapping clusters
- 📌 **Recommendation:** Sử dụng GMM khi cần soft clustering

### Nếu Hierarchical thắng:
- ✅ Có cấu trúc phân cấp trong data
- ✅ Cần dendrogram để phân tích
- ✅ Muốn explore nhiều k values
- 📌 **Recommendation:** Sử dụng Hierarchical cho exploratory analysis

### Nếu HDBSCAN thắng:
- ✅ Clusters có mật độ khác nhau
- ✅ Có nhiều noise/outliers
- ✅ Không biết trước số clusters
- 📌 **Recommendation:** Sử dụng HDBSCAN khi có outliers

---

## 🔧 Customization

### Thay đổi sample size cho metrics
```python
# Trong cell "compute-metrics"
if len(X_filtered) > 5000:
    sample_idx = np.random.choice(len(X_filtered), 10000, replace=False)  # Tăng lên 10000
```

### Thêm metrics khác
```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Nếu có ground truth labels
ari = adjusted_rand_score(true_labels, predicted_labels)
nmi = normalized_mutual_info_score(true_labels, predicted_labels)
```

### Thay đổi visualization style
```python
plt.style.use('ggplot')  # Thay vì 'seaborn-v0_8-darkgrid'
sns.set_palette('Set2')  # Thay vì 'husl'
```

---

## 📊 Visualizations

### 1. Cluster Distribution (4 plots)
- Bar charts cho mỗi phương pháp
- Hiển thị số lượng samples trong mỗi cluster
- Highlight noise points (nếu có)

### 2. Quality Metrics (3 plots)
- Silhouette Score comparison
- Davies-Bouldin Index comparison
- Calinski-Harabasz Score comparison
- Best method highlighted in gold

### 3. Energy Distribution (4 plots)
- Histograms cho mỗi phương pháp
- Energy distribution per cluster
- Identify most stable clusters

---

## 🎓 Understanding Results

### Scenario 1: All methods similar scores
**Interpretation:** Data có cấu trúc clustering rõ ràng, bất kỳ phương pháp nào cũng work well

**Action:** Chọn phương pháp đơn giản nhất (K-means) hoặc nhanh nhất

### Scenario 2: One method significantly better
**Interpretation:** Data có đặc điểm phù hợp với phương pháp đó

**Action:** Sử dụng phương pháp thắng cuộc, phân tích tại sao nó tốt hơn

### Scenario 3: HDBSCAN có nhiều noise
**Interpretation:** Data có nhiều outliers hoặc clusters không rõ ràng

**Action:** 
- Nếu outliers là quan trọng → Sử dụng HDBSCAN
- Nếu muốn cluster tất cả → Sử dụng K-means/GMM

### Scenario 4: Low scores across all methods
**Interpretation:** Data không có cấu trúc clustering rõ ràng

**Action:**
- Thử feature engineering
- Thử dimensionality reduction (PCA, t-SNE)
- Xem xét lại số clusters k

---

## 🔍 Troubleshooting

### Lỗi: File not found
**Giải pháp:**
- Kiểm tra các folders kết quả tồn tại
- Đảm bảo đã chạy các notebooks clustering trước

### Lỗi: Memory error
**Giải pháp:**
```python
# Giảm sample size
sample_idx = np.random.choice(len(X_filtered), 1000, replace=False)
```

### Metrics không tính được
**Giải pháp:**
- Kiểm tra có ít nhất 2 clusters
- Kiểm tra cluster labels hợp lệ
- Filter out noise points cho HDBSCAN

---

## 📚 References

### Metrics
- [Silhouette Score](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)
- [Davies-Bouldin Index](https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index)
- [Calinski-Harabasz Score](https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index)

### Methods
- [K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [GMM](https://scikit-learn.org/stable/modules/mixture.html)
- [Hierarchical](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [HDBSCAN](https://hdbscan.readthedocs.io/)

---

## 🎯 Next Steps

### After Running Notebook:

1. **Analyze Results**
   - Xem ranking table
   - So sánh metrics
   - Đọc recommendations

2. **Choose Best Method**
   - Dựa trên metrics
   - Dựa trên use case
   - Dựa trên computational cost

3. **Use for Further Analysis**
   - Stability analysis
   - Energy prediction
   - Material discovery

4. **Document Findings**
   - Save comparison results
   - Note best method and why
   - Plan next experiments

---

## 💾 Export & Share

### Export to HTML
```bash
jupyter nbconvert --to html carbon24-clustering-comparison-evaluation.ipynb
```

### Export to PDF
```bash
jupyter nbconvert --to pdf carbon24-clustering-comparison-evaluation.ipynb
```

### Share Results
- CSV files in `carbon24_clustering_comparison_results/`
- Visualizations (screenshot or save figures)
- Summary table

---

## 🎉 Conclusion

Notebook này cung cấp:

✅ **Comprehensive comparison** của 4 phương pháp  
✅ **Quantitative metrics** để đánh giá  
✅ **Visual comparisons** dễ hiểu  
✅ **Clear recommendations** cho từng use case  
✅ **Export results** ready to use  

**Chúc bạn phân tích thành công! 🚀**

---

**Quick Start:**
```bash
jupyter notebook carbon24-clustering-comparison-evaluation.ipynb
```

**Sau đó:** `Cell` → `Run All` → Đợi ~2-3 phút → Xem kết quả!
