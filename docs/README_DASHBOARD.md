# Carbon-24 Data Mining Dashboard

## 📊 Tổng Quan

Dashboard interactive để demo dự án "Khai thác dữ liệu cấu trúc Carbon-24" với các tính năng:

- 🏠 **Tổng quan dự án**
- 🔍 **Khảo sát dữ liệu** (phân phối, tương quan, crystal systems)
- 🎯 **Phân cụm K-means** (metrics, visualization, analysis)
- ⚠️ **Phát hiện dị biệt** (đang phát triển)
- 📈 **Dự đoán năng lượng** (đang phát triển)

## 🚀 Cài Đặt

### Bước 1: Giải phóng dung lượng ổ đĩa

Dashboard cần ~50MB để cài đặt. Hãy xóa các file không cần thiết:

```bash
# Xóa cache pip
pip cache purge

# Xóa các file tạm
del /q %TEMP%\*
```

### Bước 2: Cài đặt dependencies

```bash
pip install streamlit plotly
```

Hoặc từ file requirements:

```bash
pip install -r requirements_dashboard.txt
```

## 📝 Cấu Trúc Files

```
.
├── carbon24_dashboard.py                    # Main dashboard
├── carbon24_preprocessing_results/          # Dữ liệu đã preprocessing
│   ├── carbon24_feature_selected.csv
│   ├── carbon24_feature_selected_standard.csv
│   ├── carbon24_feature_selected_robust.csv
│   └── selected_features.json
├── carbon24_kmeans_results/                 # Kết quả clustering (optional)
│   ├── carbon24_clustered.csv
│   ├── cluster_centers.csv
│   └── clustering_report.json
└── README_DASHBOARD.md                      # File này
```

## 🎮 Chạy Dashboard

### Cách 1: Sử dụng batch file

```bash
run_dashboard.bat
```

### Cách 2: Command line

```bash
streamlit run carbon24_dashboard.py
```

Dashboard sẽ mở tại: **http://localhost:8501**

## 📚 Hướng Dẫn Sử Dụng

### 1. Trang Tổng Quan
- Xem metrics tổng quan (số mẫu, features, clusters)
- Quy trình xử lý dữ liệu
- Tiến độ dự án
- Preview dataset

### 2. Khảo Sát Dữ Liệu
- **Tab Phân phối**: Histogram và boxplot cho từng feature
- **Tab Tương quan**: Ma trận tương quan giữa các features
- **Tab Crystal Systems**: Phân bố các hệ tinh thể

### 3. Phân Cụm K-means
- **Tab Cluster Overview**: Phân bố clusters, metrics
- **Tab Visualization**: PCA 2D plots (by cluster & energy)
- **Tab Analysis**: Phân tích chi tiết từng cluster

### 4. Phát Hiện Dị Biệt (Coming Soon)
- Isolation Forest
- Local Outlier Factor (LOF)
- Statistical methods

### 5. Dự Đoán Năng Lượng (Coming Soon)
- Linear Regression
- Random Forest
- Gradient Boosting
- Neural Networks

## 🎨 Features Dashboard

### ✅ Đã hoàn thành:
- [x] Load và hiển thị dữ liệu
- [x] Khảo sát phân phối features
- [x] Ma trận tương quan
- [x] Phân tích crystal systems
- [x] Hiển thị kết quả clustering
- [x] PCA visualization
- [x] Cluster analysis

### 🚧 Đang phát triển:
- [ ] t-SNE visualization
- [ ] Anomaly detection
- [ ] Energy prediction
- [ ] 3D visualization
- [ ] Export reports

## 🛠️ Troubleshooting

### Lỗi: "No module named 'streamlit'"
```bash
pip install streamlit plotly
```

### Lỗi: "FileNotFoundError"
Đảm bảo bạn đã chạy:
1. `carbon24-preprocessing.ipynb` - Tạo preprocessed data
2. `carbon24-kmeans-clustering.ipynb` - Tạo clustering results

### Dashboard chạy chậm
- Giảm số lượng điểm trong scatter plots
- Sử dụng sampling cho t-SNE
- Đóng các ứng dụng khác

## 📊 Dataset Requirements

Dashboard cần các files sau:

**Bắt buộc:**
- `carbon24_preprocessing_results/carbon24_feature_selected_standard.csv`
- `carbon24_preprocessing_results/selected_features.json`

**Tùy chọn (cho clustering):**
- `carbon24_kmeans_results/carbon24_clustered.csv`
- `carbon24_kmeans_results/clustering_report.json`

## 💡 Tips

1. **Chạy preprocessing trước**: Dashboard sẽ tự động load dữ liệu đã xử lý
2. **Chạy clustering**: Để xem phần phân cụm, chạy notebook K-means trước
3. **Sử dụng sidebar**: Navigate giữa các trang
4. **Interactive plots**: Hover để xem chi tiết, zoom, pan

## 🎯 Demo Presentation

### Luồng demo đề xuất:

1. **Giới thiệu** (Trang Tổng quan)
   - Số liệu dataset
   - Quy trình xử lý

2. **Khảo sát dữ liệu**
   - Phân phối features quan trọng
   - Tương quan giữa features
   - Phân bố crystal systems

3. **Kết quả phân cụm**
   - Metrics đánh giá
   - Visualization PCA
   - Phân tích từng cluster

4. **Kế hoạch tiếp theo**
   - Anomaly detection
   - Energy prediction

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. Python version >= 3.8
2. Đã cài đặt đầy đủ dependencies
3. Files dữ liệu tồn tại
4. Port 8501 không bị chiếm

## 🎓 Credits

**Dự án:** Khai thác dữ liệu cấu trúc Carbon-24  
**Phương pháp:** K-means, Isolation Forest, Regression  
**Tools:** Python, Streamlit, Plotly, Scikit-learn
