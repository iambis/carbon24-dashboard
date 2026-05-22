# 🎉 Dashboard Đã Được Cập Nhật!

## ✨ Tính Năng Mới

Dashboard Carbon-24 đã được cập nhật với **6 trang phân cụm** đầy đủ:

### 📊 Các Trang Phân Cụm

1. **🎯 Phân cụm K-means** (Đã có)
   - Cluster overview & distribution
   - PCA 2D/3D visualization
   - Cluster analysis & feature comparison

2. **🎲 Phân cụm GMM** (MỚI!)
   - Gaussian Mixture Model clustering
   - Uncertainty analysis với probability scores
   - AIC/BIC model selection
   - Cluster profiles với soft assignments

3. **🌳 Phân cụm Hierarchical** (MỚI!)
   - Agglomerative clustering với Ward linkage
   - Hierarchical structure visualization
   - Cluster interpretation
   - Energy analysis by cluster

4. **🔍 Phân cụm HDBSCAN** (MỚI!)
   - Density-based clustering
   - Automatic cluster detection
   - Noise/outlier analysis
   - Membership probability visualization

5. **📊 So sánh thuật toán** (Đã có)
   - Bảng xếp hạng 4 thuật toán
   - So sánh chỉ số chất lượng
   - Chi tiết từng thuật toán
   - Visualization đa chiều

## 🚀 Cách Chạy Dashboard

```bash
streamlit run carbon24_dashboard.py
```

Hoặc sử dụng file batch:
```bash
run_dashboard.bat
```

## 📁 Cấu Trúc Dữ Liệu Cần Thiết

Dashboard sẽ tự động load dữ liệu từ các thư mục sau:

### K-means
```
carbon24_kmeans_results/
├── carbon24_clustered.csv
└── clustering_report.json
```

### GMM
```
carbon24_gmm_results/
├── results/
│   └── carbon24_gmm_results.csv
├── tables/
│   └── gmm_cluster_profile.csv
└── gmm_clustering_report.json
```

### Hierarchical
```
carbon24_hierarchical_baseline/
├── results/
│   └── carbon24_hierarchical_results.csv
└── tables/
    └── hierarchical_cluster_interpretation.csv
```

### HDBSCAN
```
hdbscan_phuc/
├── hdbscan_results.csv
├── hdbscan_cluster_profile.csv
├── hdbscan_energy_summary.csv
└── hdbscan_noise_outliers.csv
```

### So sánh thuật toán
```
carbon24_clustering_comparison_results/
├── methods_overview.csv
├── quality_metrics.csv
└── method_ranking.csv
```

## 🎯 Tính Năng Chi Tiết Từng Trang

### 🎲 Trang GMM

**Tab 1: Cluster Overview**
- Phân bố clusters với bar chart & pie chart
- AIC/BIC scores cho model selection
- Thông tin số clusters tối ưu

**Tab 2: Visualization**
- PCA 2D scatter plots
- Color by: Cluster, Max Probability, Relative Energy
- Interactive plotly charts

**Tab 3: Uncertainty Analysis**
- Distribution of max probabilities
- Uncertain samples identification
- Threshold slider để filter samples
- Box plot probability by cluster

**Tab 4: Cluster Profiles**
- Cluster profile table
- Energy distribution by cluster
- Statistical summary

### 🌳 Trang Hierarchical

**Tab 1: Cluster Overview**
- Cluster size distribution
- Pie chart visualization
- Thông tin về Ward linkage method

**Tab 2: Visualization**
- PCA 2D scatter plots
- Color by: Cluster, Relative Energy
- Interactive visualization

**Tab 3: Cluster Interpretation**
- Cluster interpretation table
- Energy analysis by cluster
- Detailed cluster statistics
- Individual cluster analysis với selector

### 🔍 Trang HDBSCAN

**Tab 1: Cluster Overview**
- Cluster distribution (excluding noise)
- Pie chart with noise points
- Thông tin về density-based clustering

**Tab 2: Visualization**
- PCA 2D scatter plots
- Color by: Cluster, Membership Probability, Relative Energy
- Noise points visualization

**Tab 3: Noise Analysis**
- Noise points statistics
- Top noise/outlier samples table
- Energy comparison: Noise vs Clusters
- Box plot visualization

**Tab 4: Cluster Profiles**
- Cluster profile table (excluding noise)
- Energy summary by cluster
- Individual cluster analysis
- Membership probability distribution

### 📊 Trang So Sánh Thuật Toán

**Tab 1: Xếp hạng**
- Bảng xếp hạng với medals 🥇🥈🥉
- Bar chart điểm tổng
- Radar chart top 2 methods
- Giải thích cách tính điểm

**Tab 2: Chỉ số chất lượng**
- Bảng metrics chi tiết
- 3 bar charts riêng cho từng metric
- Giải thích ý nghĩa các chỉ số

**Tab 3: Chi tiết thuật toán**
- Selector để chọn thuật toán
- Thông tin cơ bản & metrics
- So sánh với các thuật toán khác
- Đặc điểm: Ưu điểm, Nhược điểm, Use cases

**Tab 4: So sánh trực quan**
- Parallel coordinates plot
- Scatter plots đa chiều
- Noise points distribution
- Heatmap normalized metrics

## 💡 Tips Sử Dụng

1. **Navigation**: Sử dụng sidebar để chuyển giữa các trang
2. **Interactive Charts**: Hover để xem chi tiết, click legend để hide/show
3. **3D Visualization**: Drag để rotate, scroll để zoom
4. **Filters**: Sử dụng sliders và selectors để filter data
5. **Export**: Click camera icon trên charts để download images

## 🔧 Troubleshooting

### Dashboard không load được dữ liệu?
- Kiểm tra các thư mục kết quả đã tồn tại chưa
- Chạy các notebook tương ứng để tạo kết quả
- Kiểm tra tên file và đường dẫn

### Charts không hiển thị?
- Kiểm tra có PCA components trong data không
- Refresh browser (Ctrl + F5)
- Xóa cache: Settings > Clear cache

### Lỗi import?
- Cài đặt dependencies: `pip install -r requirements_dashboard.txt`
- Kiểm tra version của streamlit và plotly

## 📝 Ghi Chú

- Dashboard tự động cache data để tăng tốc độ
- Nếu cập nhật data, click "Clear cache" trong sidebar
- Tất cả visualizations đều interactive với Plotly
- Dashboard responsive, hoạt động tốt trên mobile

## 🎨 Customization

Để customize dashboard:
1. Sửa CSS trong phần `st.markdown()` ở đầu file
2. Thay đổi color schemes trong plotly charts
3. Thêm metrics mới trong các tab
4. Tạo thêm visualizations theo nhu cầu

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. Console logs trong terminal
2. Browser developer console (F12)
3. Streamlit documentation: https://docs.streamlit.io

---

**Phiên bản:** 2.0  
**Cập nhật:** Thêm 3 trang phân cụm mới (GMM, Hierarchical, HDBSCAN)  
**Tác giả:** Carbon-24 Data Mining Team
