# 📊 Hướng dẫn sử dụng Anomaly Detection trên Dashboard

## ✅ Đã sửa

Dashboard đã được cập nhật để hiển thị phần **Phát hiện Dị biệt (Anomaly Detection)**.

### Thay đổi:
1. ✅ Thêm functions `load_anomaly_detection_results()` và `render_anomaly_detection_tab()`
2. ✅ Di chuyển functions lên đúng vị trí (trước phần xử lý pages)
3. ✅ Page "Phát hiện dị biệt" giờ gọi `render_anomaly_detection_tab()` thay vì hiển thị "đang phát triển"

## 🚀 Cách sử dụng

### Bước 1: Chạy module Anomaly Detection

Trước tiên, bạn cần tạo dữ liệu anomaly detection:

```bash
python carbon24_anomaly_detection.py
```

**Output mong đợi:**
```
================================================================================
CARBON-24 ANOMALY DETECTION
================================================================================

✅ Đã tạo thư mục output: carbon24_anomaly_detection
✅ Đã load HDBSCAN results: (10153, 33)

================================================================================
PHƯƠNG PHÁP 1: HDBSCAN NOISE POINTS
================================================================================
📊 Số điểm noise: 786 / 10,153 (7.74%)

... (các phương pháp khác)

✅ HOÀN TẤT ANOMALY DETECTION
```

**Files được tạo:**
```
carbon24_anomaly_detection/
├── anomaly_detection_results.csv      # Full results
├── anomaly_summary.csv                # Summary statistics
├── anomaly_method_comparison.csv      # Method comparison
├── anomaly_details.csv                # Consensus anomalies
└── figures/
    ├── anomaly_methods_comparison.png
    ├── anomaly_vote_distribution.png
    ├── anomaly_energy_distribution.png
    └── isolation_forest_score_distribution.png
```

### Bước 2: Chạy Dashboard

```bash
streamlit run carbon24_dashboard.py
```

### Bước 3: Xem Anomaly Detection

1. Dashboard sẽ mở trong browser
2. Trong **Sidebar**, chọn **"🔍 Phát hiện dị biệt"**
3. Dashboard sẽ hiển thị:
   - 📊 Tổng quan (metrics)
   - 🔬 So sánh các phương pháp
   - 🗺️ Phân bố không gian (PCA)
   - ⚡ Phân tích năng lượng
   - 📋 Chi tiết consensus anomalies
   - 💡 Khuyến nghị sử dụng

## 📊 Nội dung Dashboard

### 1. Tổng quan (Metrics)
- **Tổng số mẫu**: 10,153
- **Consensus Anomalies**: Số anomalies được ≥2 phương pháp đồng ý
- **High Confidence**: Số anomalies được cả 3 phương pháp đồng ý
- **Any Method**: Số anomalies được ≥1 phương pháp phát hiện

### 2. So sánh các phương pháp
- **Bar chart**: Tỷ lệ anomaly của từng phương pháp
- **Vote distribution**: Phân bố số phương pháp đồng ý (0, 1, 2, 3)

### 3. Phân bố không gian (PCA)
- Scatter plot hiển thị anomalies trong không gian PCA
- Có thể chọn phương pháp:
  - HDBSCAN Noise
  - Low Probability
  - Isolation Forest
  - Consensus (≥2)
  - All 3 methods

### 4. Phân tích năng lượng
- **Table**: So sánh năng lượng trung bình (Anomaly vs Normal)
- **Histogram**: Phân bố relative energy

### 5. Chi tiết Consensus Anomalies
- Table hiển thị danh sách anomalies
- Filter theo vote count (2 hoặc 3)
- Slider để chọn số lượng hiển thị
- Download button để tải CSV

### 6. Khuyến nghị sử dụng
- 🎯 **Phân tích sâu**: Dùng `is_anomaly_all` (high precision)
- ⚖️ **Lọc dữ liệu**: Dùng `is_anomaly_consensus` (balanced)
- 🔍 **Khám phá**: Dùng `is_anomaly_any` (high recall)

## ⚠️ Troubleshooting

### Lỗi: "Chưa có dữ liệu Anomaly Detection"

**Nguyên nhân**: Chưa chạy module anomaly detection

**Giải pháp**:
```bash
python carbon24_anomaly_detection.py
```

### Lỗi: "Không tìm thấy file HDBSCAN results"

**Nguyên nhân**: Chưa chạy HDBSCAN clustering

**Giải pháp**:
```bash
jupyter notebook HDBSCAN.ipynb
# Chạy toàn bộ notebook
```

### Dashboard không hiển thị page Anomaly Detection

**Kiểm tra**:
1. Sidebar có option "🔍 Phát hiện dị biệt" không?
2. Nếu không, restart dashboard: `Ctrl+C` rồi chạy lại

### Lỗi khi load dữ liệu

**Kiểm tra**:
```bash
# Kiểm tra files tồn tại
ls carbon24_anomaly_detection/
```

**Expected output:**
```
anomaly_detection_results.csv
anomaly_summary.csv
anomaly_method_comparison.csv
anomaly_details.csv
figures/
```

## 🔄 Workflow hoàn chỉnh

```
1. HDBSCAN Clustering
   └─> jupyter notebook HDBSCAN.ipynb
       └─> hdbscan_phuc/hdbscan_results.csv

2. Anomaly Detection
   └─> python carbon24_anomaly_detection.py
       └─> carbon24_anomaly_detection/*.csv

3. Dashboard
   └─> streamlit run carbon24_dashboard.py
       └─> Chọn "Phát hiện dị biệt" trong sidebar
```

## 📝 Lưu ý

1. **Phải chạy HDBSCAN trước**: Anomaly detection dựa trên kết quả HDBSCAN
2. **Dữ liệu được cache**: Nếu cập nhật dữ liệu, restart dashboard
3. **Interactive**: Có thể chọn phương pháp, filter, và download dữ liệu
4. **Responsive**: Dashboard tự động điều chỉnh theo kích thước màn hình

## 🎨 Screenshots

### Tổng quan
![Overview](Metrics hiển thị tổng số mẫu và anomalies)

### So sánh phương pháp
![Comparison](Bar chart và vote distribution)

### PCA Visualization
![PCA](Scatter plot với normal và anomaly points)

### Energy Analysis
![Energy](Table và histogram so sánh năng lượng)

## 💡 Tips

1. **Chọn phương pháp phù hợp**:
   - Nghiên cứu → `All 3 methods`
   - Lọc dữ liệu → `Consensus (≥2)`
   - Khám phá → `Any method`

2. **Download dữ liệu**:
   - Click button "📥 Download Anomaly Details (CSV)"
   - Sử dụng cho phân tích ngoài dashboard

3. **Refresh dữ liệu**:
   - Nếu chạy lại anomaly detection
   - Restart dashboard hoặc clear cache

## 🔗 Liên kết

- **Module code**: `carbon24_anomaly_detection.py`
- **Notebook demo**: `carbon24-anomaly-detection.ipynb`
- **Documentation**: `ANOMALY_DETECTION_README.md`
- **Guide**: `ANOMALY_DETECTION_GUIDE.md`

---

**Version**: 1.0.0  
**Last Updated**: 2026-05-21  
**Status**: ✅ Working
