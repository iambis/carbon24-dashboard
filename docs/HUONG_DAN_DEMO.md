# 🎯 HƯỚNG DẪN DEMO DỰ ÁN CARBON-24

## ✅ TRẠNG THÁI DỰ ÁN

### Đã Hoàn Thành 100%:
- ✅ **Tiền xử lý dữ liệu** (Preprocessing)
- ✅ **Lựa chọn đặc trưng** (Feature Selection) 
- ✅ **Phân cụm K-means** (Clustering)
- ✅ **Visualization 2D & 3D** (PCA)
- ✅ **Dashboard Interactive** (Streamlit)

### Đang Phát Triển:
- 🚧 **Phát hiện dị biệt** (Anomaly Detection)
- 🚧 **Dự đoán năng lượng** (Energy Prediction)

---

## 🚀 CÁCH CHẠY DASHBOARD

### Cách 1: Sử dụng Batch File (Đơn giản nhất)

```bash
run_dashboard.bat
```

### Cách 2: Command Line

```bash
streamlit run carbon24_dashboard.py
```

Dashboard sẽ tự động mở tại: **http://localhost:8501**

---

## 📊 CẤU TRÚC DASHBOARD

### 1. 🏠 Trang Tổng Quan
**Nội dung:**
- Metrics tổng quan: 10,153 samples, 31 features, 5 clusters
- Quy trình xử lý dữ liệu (workflow diagram)
- Tiến độ dự án (progress bars)
- Preview dataset

**Demo tips:**
- Bắt đầu từ đây để giới thiệu dự án
- Highlight số liệu quan trọng
- Giải thích workflow từ raw data → clustering

---

### 2. 🔍 Trang Khảo Sát Dữ Liệu

#### Tab "📊 Phân phối"
- Histogram và Box plot cho từng feature
- Thống kê mô tả (mean, std, min, max, median)
- **Demo:** Chọn features quan trọng như `relative_energy`, `density`, `mean_coordination`

#### Tab "🔥 Tương quan"
- Ma trận tương quan (correlation heatmap)
- Hiển thị top 15 features
- **Demo:** Chỉ ra các cặp features có correlation cao/thấp

#### Tab "💎 Crystal Systems"
- Bar chart và Pie chart phân bố 7 crystal systems
- Bảng chi tiết với count và percentage
- **Demo:** Giải thích sự phân bố không đều giữa các systems

---

### 3. 🎯 Trang Phân Cụm K-means

#### Tab "📈 Cluster Overview"
- **Metrics chính:**
  - Silhouette Score (càng cao càng tốt, max = 1)
  - Davies-Bouldin Index (càng thấp càng tốt)
  - Calinski-Harabasz Score (càng cao càng tốt)
  - Inertia (tổng khoảng cách trong cluster)

- **Phân bố clusters:**
  - Bar chart: Số lượng samples trong mỗi cluster
  - Pie chart: Tỷ lệ phần trăm

**Demo tips:**
- Giải thích ý nghĩa các metrics
- So sánh kích thước các clusters

#### Tab "🎨 Visualization"

##### **PCA 2D** (Nhanh, dễ hiểu)
- Toggle: Chọn "2D"
- Color options:
  - **By Cluster**: Thấy rõ sự phân tách giữa các clusters
  - **By Relative Energy**: Gradient màu theo năng lượng

**Demo tips:**
- Bắt đầu với 2D để dễ hiểu
- Chỉ ra clusters tách biệt rõ ràng
- Highlight vùng năng lượng thấp/cao

##### **PCA 3D** (Chi tiết, ấn tượng)
- Toggle: Chọn "3D"
- Color options:
  - **By Cluster**: Mỗi cluster có màu riêng
  - **By Relative Energy**: Viridis colorscale (xanh → vàng)
  - **By Crystal System**: 7 màu cho 7 systems

**3D Controls:**
- 🖱️ **Rotate**: Click and drag
- 🔍 **Zoom**: Scroll wheel
- ✋ **Pan**: Right-click and drag
- 🔄 **Reset**: Double-click

**Demo tips:**
- Rotate từ từ để show các góc khác nhau
- Zoom vào clusters cụ thể
- Hover để xem chi tiết từng điểm
- Chuyển đổi giữa 3 color modes để highlight insights khác nhau

**Variance Explained:**
- PC1: 24.57% - Chiều biến thiên lớn nhất
- PC2: 15.38% - Chiều biến thiên thứ hai
- PC3: 10.59% - Chiều biến thiên thứ ba
- **Total: 50.54%** - Giữ lại hơn 50% thông tin từ 22 features

#### Tab "🔬 Analysis"
- Chọn cluster cụ thể để phân tích
- Metrics: Số lượng, tỷ lệ, mean/std energy
- Box plot so sánh features giữa các clusters

**Demo tips:**
- Chọn cluster lớn nhất/nhỏ nhất
- So sánh energy giữa các clusters
- Highlight features phân biệt clusters

---

### 4. ⚠️ Trang Phát Hiện Dị Biệt (Coming Soon)
- Placeholder với phương pháp sẽ áp dụng:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - Statistical methods (Z-score, IQR)

---

### 5. 📈 Trang Dự Đoán Năng Lượng (Coming Soon)
- Placeholder với models sẽ áp dụng:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting (XGBoost/LightGBM)
  - Neural Networks

---

## 🎬 KỊCH BẢN DEMO (15-20 phút)

### Phần 1: Giới Thiệu (2-3 phút)
**Trang: 🏠 Tổng quan**

```
"Xin chào, hôm nay tôi xin giới thiệu dự án Khai thác dữ liệu cấu trúc Carbon-24.

Dataset của chúng tôi có 10,153 samples với 31 features sau khi feature selection.

Quy trình xử lý gồm 5 bước chính:
1. Preprocessing - Làm sạch và chuẩn hóa dữ liệu
2. Feature Engineering - Tạo 4 features mới
3. Feature Selection - Loại bỏ 14 features có multicollinearity cao
4. Normalization - Chuẩn hóa với StandardScaler
5. K-means Clustering - Phân thành 5 clusters

Hiện tại chúng tôi đã hoàn thành phần Clustering, đang phát triển 
Anomaly Detection và Energy Prediction."
```

---

### Phần 2: Khảo Sát Dữ Liệu (3-4 phút)
**Trang: 🔍 Khảo sát dữ liệu**

#### Tab Phân phối:
```
"Hãy xem phân phối của feature quan trọng nhất: relative_energy.

[Chọn relative_energy trong dropdown]

Từ histogram, ta thấy phân phối lệch phải, với phần lớn samples 
có năng lượng thấp. Box plot cho thấy có một số outliers ở năng lượng cao.

Mean: [đọc số], Std: [đọc số], cho thấy độ biến thiên khá lớn."
```

#### Tab Tương quan:
```
"Ma trận tương quan cho thấy mối quan hệ giữa các features.

[Chỉ vào heatmap]

Các ô màu đỏ đậm là correlation dương mạnh, xanh đậm là âm mạnh.
Sau feature selection, chúng tôi đã loại bỏ các cặp có |r| > 0.95 
để tránh multicollinearity."
```

#### Tab Crystal Systems:
```
"Dataset có 7 crystal systems khác nhau.

[Chỉ vào charts]

Monoclinic chiếm đa số với [X]%, tiếp theo là Triclinic với [Y]%.
Sự phân bố không đều này sẽ ảnh hưởng đến clustering."
```

---

### Phần 3: Kết Quả Phân Cụm (8-10 phút) ⭐ PHẦN QUAN TRỌNG NHẤT
**Trang: 🎯 Phân cụm K-means**

#### Tab Cluster Overview:
```
"Chúng tôi đã áp dụng K-means với k=5 clusters.

Các metrics đánh giá:
- Silhouette Score: [X] - Cho thấy clusters tách biệt [tốt/khá tốt]
- Davies-Bouldin: [X] - [Thấp là tốt]
- Calinski-Harabasz: [X] - [Cao là tốt]

[Chỉ vào bar chart]

Cluster 0 có [X] samples, chiếm [Y]% dataset.
Cluster 1 có [X] samples, chiếm [Y]% dataset.
..."
```

#### Tab Visualization - 2D:
```
"Đầu tiên, hãy xem visualization 2D với PCA.

[Chọn 2D, Color by Cluster]

PCA giảm 22 features xuống 2 chiều để dễ visualize.
Ta thấy các clusters tách biệt khá rõ ràng, đặc biệt là Cluster [X] và [Y].

[Chuyển sang Color by Relative Energy]

Khi color theo năng lượng, ta thấy vùng màu xanh (năng lượng thấp) 
tập trung ở [vị trí], còn vùng vàng (năng lượng cao) ở [vị trí].
Điều này cho thấy clustering có capture được thông tin về năng lượng."
```

#### Tab Visualization - 3D: ⭐ HIGHLIGHT
```
"Bây giờ, hãy xem trong không gian 3D để có cái nhìn chi tiết hơn.

[Chọn 3D, Color by Cluster]

[Rotate từ từ]

Trong không gian 3D, ta thấy rõ hơn cấu trúc của các clusters.
Cluster [X] nằm ở góc này, tách biệt hoàn toàn với các clusters khác.
Cluster [Y] và [Z] có phần overlap nhẹ ở vùng này.

[Zoom vào một cluster]

Đây là Cluster [X], có [N] samples, khá compact và tách biệt.

[Chuyển sang Color by Relative Energy]

Khi color theo năng lượng, ta thấy gradient rõ ràng:
- Vùng xanh (năng lượng thấp) tập trung ở đây
- Vùng vàng (năng lượng cao) ở phía kia
- Có một số điểm năng lượng rất cao (màu vàng sáng) - đây có thể là 
  candidates cho anomaly detection.

[Chuyển sang Color by Crystal System]

Cuối cùng, color theo crystal system:
- Monoclinic (màu [X]) chiếm đa số
- Triclinic (màu [Y]) phân bố ở vùng này
- Các systems khác phân tán

Điều thú vị là các crystal systems không tập trung hoàn toàn trong 
một cluster, cho thấy clustering dựa trên geometry và energy, 
không chỉ crystal structure."
```

**Demo tips cho 3D:**
- Rotate chậm, dừng lại ở các góc đẹp
- Zoom vào clusters cụ thể để show chi tiết
- Hover vào một số điểm để show tooltip
- Chuyển đổi giữa 3 color modes để highlight insights khác nhau
- Nhấn mạnh variance explained: 50.54%

#### Tab Analysis:
```
"Hãy phân tích chi tiết Cluster [X].

[Chọn cluster trong dropdown]

Cluster này có [N] samples, chiếm [Y]% dataset.
Mean energy: [X], Std: [Y] - cho thấy [đồng nhất/đa dạng].

[Chọn feature để compare]

Box plot cho thấy Cluster [X] có [feature] [cao/thấp] hơn các clusters khác.
Điều này giúp characterize cluster này là [đặc điểm]."
```

---

### Phần 4: Kế Hoạch Tiếp Theo (2-3 phút)

#### Trang Phát hiện dị biệt:
```
"Tiếp theo, chúng tôi sẽ implement Anomaly Detection với:
- Isolation Forest: Phát hiện outliers dựa trên isolation
- LOF: Phát hiện local anomalies dựa trên density
- Statistical methods: Z-score và IQR

Mục tiêu là tìm các cấu trúc Carbon bất thường, có thể có tính chất đặc biệt."
```

#### Trang Dự đoán năng lượng:
```
"Cuối cùng, chúng tôi sẽ build models để predict relative_energy:
- Linear Regression: Baseline
- Random Forest: Ensemble method với feature importance
- Gradient Boosting: XGBoost/LightGBM cho performance cao
- Neural Networks: Deep learning cho complex patterns

Mục tiêu là predict năng lượng từ geometry features, giúp screening 
các cấu trúc mới mà không cần tính toán DFT tốn kém."
```

---

### Phần 5: Kết Luận (1-2 phút)
```
"Tóm lại, dự án đã hoàn thành:
✅ Preprocessing và Feature Selection: 10,153 samples, 31 features
✅ K-means Clustering: 5 clusters với metrics tốt
✅ Visualization 2D & 3D: PCA giữ lại 50.54% variance
✅ Interactive Dashboard: Streamlit với Plotly

Đang phát triển:
🚧 Anomaly Detection
🚧 Energy Prediction

Cảm ơn các bạn đã theo dõi!"
```

---

## 💡 TIPS DEMO HIỆU QUẢ

### Trước Demo:
1. ✅ Test dashboard trước: `streamlit run carbon24_dashboard.py`
2. ✅ Mở sẵn các HTML files 3D trong browser tabs
3. ✅ Chuẩn bị notes về metrics và insights
4. ✅ Luyện tập rotate 3D plot mượt mà
5. ✅ Kiểm tra internet connection (nếu cần)

### Trong Demo:
1. 🎯 **Bắt đầu với overview** - Tạo context
2. 📊 **Show data exploration** - Chứng minh hiểu dữ liệu
3. ⭐ **Focus vào clustering** - Phần quan trọng nhất
4. 🎨 **Highlight 3D visualization** - Ấn tượng nhất
5. 🚀 **Kết thúc với future work** - Show ambition

### Kỹ Thuật Trình Bày:
- ✅ Nói chậm, rõ ràng
- ✅ Pause sau mỗi insight quan trọng
- ✅ Point vào screen khi giải thích
- ✅ Tương tác với plots (hover, zoom, rotate)
- ✅ Trả lời câu hỏi tự tin

### Xử Lý Tình Huống:
- ❌ **Dashboard lag**: Giảm số điểm trong plot, hoặc dùng HTML files
- ❌ **Câu hỏi khó**: "Đó là hướng phát triển tiếp theo của chúng tôi"
- ❌ **Technical issue**: Có backup HTML files sẵn sàng

---

## 📁 FILES QUAN TRỌNG

### Data Files:
```
carbon24_kmeans_results/
├── carbon24_clustered.csv              # Data với clusters và PCA 3D
├── clustering_report.json              # Metrics
└── pca_3d_*.html                       # 3 HTML files (backup)

carbon24_preprocessing_results/
├── carbon24_feature_selected.csv       # Data sau feature selection
└── selected_features.json              # Feature list
```

### Code Files:
```
carbon24_dashboard.py                   # Main dashboard
carbon24-preprocessing.ipynb            # Preprocessing notebook
carbon24-kmeans-clustering.ipynb        # Clustering notebook
add_pca_3d_to_notebook.py              # Script tạo PCA 3D
```

### Documentation:
```
README_DASHBOARD.md                     # Dashboard guide
README_PCA_3D.md                        # PCA 3D guide
HUONG_DAN_DEMO.md                       # File này
```

---

## 🎓 INSIGHTS QUAN TRỌNG CẦN NHỚ

### Về Dataset:
- 10,153 samples Carbon-24 structures
- 31 features sau feature selection (từ 41 ban đầu)
- 7 crystal systems, Monoclinic chiếm đa số
- Loại bỏ 14 features có multicollinearity > 0.95

### Về Clustering:
- K-means với k=5 clusters (chọn bằng Elbow method)
- Silhouette score: [Xem trong dashboard]
- Clusters có kích thước khác nhau
- Có correlation giữa clusters và energy levels

### Về PCA:
- **2D PCA**: ~40% variance, dễ visualize
- **3D PCA**: 50.54% variance (PC1: 24.57%, PC2: 15.38%, PC3: 10.59%)
- 3D cho thấy structure rõ hơn 2D
- Clusters tách biệt khá tốt trong PCA space

### Về Visualization:
- Interactive với Plotly
- 3 color modes: Cluster, Energy, Crystal System
- Hover để xem details
- Export được ảnh PNG

---

## 🐛 TROUBLESHOOTING

### Dashboard không chạy:
```bash
# Check dependencies
pip list | Select-String -Pattern "streamlit|plotly"

# Reinstall nếu cần
pip install streamlit plotly

# Run lại
streamlit run carbon24_dashboard.py
```

### Port 8501 bị chiếm:
```bash
# Dùng port khác
streamlit run carbon24_dashboard.py --server.port 8502
```

### Dashboard chậm:
- Giảm số điểm trong scatter plots
- Dùng sampling cho large datasets
- Close các tabs không cần thiết

### HTML files không mở:
```bash
# Mở bằng browser cụ thể
start chrome carbon24_kmeans_results/pca_3d_clusters.html
```

---

## 📞 SUPPORT

### Nếu gặp vấn đề:
1. Check Python version >= 3.8
2. Check dependencies installed
3. Check data files exist
4. Check port 8501 available
5. Restart terminal và thử lại

### Resources:
- [Streamlit Docs](https://docs.streamlit.io/)
- [Plotly Docs](https://plotly.com/python/)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)

---

## ✨ GOOD LUCK!

**Remember:**
- 🎯 Focus on insights, not just code
- 📊 Tell a story with data
- 🎨 Make it visual and interactive
- 💡 Show you understand the domain
- 🚀 Demonstrate future potential

**You got this! 💪**

---

**Created:** 2026-05-20  
**Version:** 1.0  
**Status:** Ready for Demo ✅
