# Carbon-24 Anomaly Detection Module

## 📋 Tổng quan

Module phát hiện dị biệt (Anomaly Detection) cho dữ liệu Carbon-24 dựa trên kết quả phân cụm HDBSCAN.

## 🎯 Mục tiêu

Phát hiện các cấu trúc Carbon-24 bất thường (anomalies/outliers) sử dụng nhiều phương pháp khác nhau và kết hợp kết quả để có độ tin cậy cao.

## 🔬 Phương pháp

Module sử dụng **3 phương pháp** phát hiện dị biệt:

### 1. HDBSCAN Noise Points
- **Nguyên lý**: Các điểm được HDBSCAN gán nhãn `-1` (noise/outlier)
- **Ưu điểm**: Dựa trên mật độ, phát hiện điểm nằm ở vùng thưa
- **Đặc điểm**: Các điểm không thuộc bất kỳ cụm nào

### 2. HDBSCAN Low Probability Members
- **Nguyên lý**: Các điểm có xác suất thuộc cụm thấp (< 0.5)
- **Ưu điểm**: Phát hiện điểm nằm ở biên cụm, không chắc chắn
- **Đặc điểm**: Có thể thuộc cụm nhưng membership yếu

### 3. Isolation Forest
- **Nguyên lý**: Thuật toán ensemble dựa trên cây quyết định
- **Ưu điểm**: Hiệu quả với dữ liệu nhiều chiều, không phụ thuộc vào phân cụm
- **Đặc điểm**: Phát hiện điểm dễ bị "cô lập" trong feature space

## 📊 Kết hợp phương pháp

Module cung cấp nhiều cách kết hợp:

- **`is_anomaly_any`**: Ít nhất 1 phương pháp đánh dấu (high recall)
- **`is_anomaly_consensus`**: Ít nhất 2/3 phương pháp đồng ý (balanced)
- **`is_anomaly_all`**: Cả 3 phương pháp đồng ý (high precision)

## 🚀 Cách sử dụng

### Chạy script Python

```bash
python carbon24_anomaly_detection.py
```

### Chạy Jupyter Notebook

```bash
jupyter notebook carbon24-anomaly-detection.ipynb
```

## 📁 Cấu trúc Input/Output

### Input (Required)
```
hdbscan_phuc/
├── hdbscan_results.csv          # Kết quả HDBSCAN clustering
└── hdbscan_noise_outliers.csv   # Noise points từ HDBSCAN
```

### Output
```
carbon24_anomaly_detection/
├── anomaly_detection_results.csv      # Full results với tất cả cột
├── anomaly_summary.csv                # Thống kê tổng hợp
├── anomaly_method_comparison.csv      # So sánh các phương pháp
├── anomaly_details.csv                # Chi tiết consensus anomalies
└── figures/
    ├── anomaly_methods_comparison.png         # So sánh 4 phương pháp (PCA)
    ├── anomaly_vote_distribution.png          # Phân bố vote count
    ├── anomaly_energy_distribution.png        # Phân bố năng lượng
    └── isolation_forest_score_distribution.png # Phân bố anomaly score
```

## 📈 Kết quả chính

### 1. Anomaly Detection Results (`anomaly_detection_results.csv`)

Chứa toàn bộ dữ liệu với các cột mới:

- `is_hdbscan_noise`: 1 nếu là HDBSCAN noise, 0 nếu không
- `is_low_probability`: 1 nếu probability < 0.5, 0 nếu không
- `is_isolation_forest_anomaly`: 1 nếu Isolation Forest phát hiện, 0 nếu không
- `isolation_forest_score`: Anomaly score từ Isolation Forest (càng âm càng bất thường)
- `anomaly_vote_count`: Số phương pháp đánh dấu là anomaly (0-3)
- `is_anomaly_consensus`: 1 nếu ≥2 phương pháp đồng ý, 0 nếu không
- `is_anomaly_any`: 1 nếu ≥1 phương pháp đánh dấu, 0 nếu không
- `is_anomaly_all`: 1 nếu cả 3 phương pháp đồng ý, 0 nếu không

### 2. Method Comparison (`anomaly_method_comparison.csv`)

So sánh số lượng và tỷ lệ anomalies của từng phương pháp:

| method | n_anomalies | anomaly_ratio | n_normal | normal_ratio |
|--------|-------------|---------------|----------|--------------|
| is_hdbscan_noise | ... | ... | ... | ... |
| is_low_probability | ... | ... | ... | ... |
| is_isolation_forest_anomaly | ... | ... | ... | ... |
| is_anomaly_consensus | ... | ... | ... | ... |

### 3. Anomaly Summary (`anomaly_summary.csv`)

Phân tích đặc điểm năng lượng của anomalies vs normal:

- `anomaly_energy_mean`: Năng lượng trung bình của anomalies
- `normal_energy_mean`: Năng lượng trung bình của normal
- `energy_diff_mean`: Chênh lệch năng lượng

### 4. Anomaly Details (`anomaly_details.csv`)

Danh sách chi tiết các consensus anomalies (≥2 phương pháp đồng ý), sorted theo vote count.

## 🎨 Visualizations

### 1. Methods Comparison (PCA Space)
4 subplot so sánh phân bố anomalies trong không gian PCA:
- HDBSCAN Noise
- Low Probability
- Isolation Forest
- Consensus (≥2 methods)

### 2. Vote Distribution
Bar chart phân bố số phương pháp đồng ý (0, 1, 2, 3)

### 3. Energy Distribution
Histogram so sánh phân bố năng lượng giữa anomaly và normal cho từng phương pháp

### 4. Isolation Forest Score Distribution
Histogram phân bố anomaly score từ Isolation Forest

## ⚙️ Configuration

Các tham số có thể điều chỉnh trong `carbon24_anomaly_detection.py`:

```python
HDBSCAN_PROBABILITY_THRESHOLD = 0.5  # Ngưỡng xác suất thấp
ISOLATION_FOREST_CONTAMINATION = 0.1  # Tỷ lệ dị biệt dự kiến
RANDOM_STATE = 42                     # Random seed
```

## 📊 Ví dụ kết quả

### Typical Output:
```
📊 TỔNG QUAN:
   - Tổng số mẫu: 10,153
   - HDBSCAN Noise: 786 (7.74%)
   - Low Probability: 1,234 (12.15%)
   - Isolation Forest: 1,015 (10.00%)

📊 KẾT HỢP PHƯƠNG PHÁP:
   - Consensus (≥2 methods): 456 (4.49%)
   - All 3 methods agree: 123 (1.21%)
   - Any method (≥1): 2,345 (23.09%)
```

## 🔍 Phân tích sâu

### Overlap Analysis
Module tính toán Jaccard similarity giữa các phương pháp để đánh giá mức độ đồng thuận.

### Energy Characteristics
Phân tích xem anomalies có năng lượng cao hơn (kém ổn định) hay thấp hơn (ổn định bất thường) so với normal.

## 💡 Khuyến nghị sử dụng

### Cho phân tích sâu:
Sử dụng `is_anomaly_all` (cả 3 phương pháp đồng ý)
- High precision, low false positive
- Các điểm thực sự bất thường

### Cho lọc dữ liệu:
Sử dụng `is_anomaly_consensus` (≥2 phương pháp)
- Balanced precision/recall
- Phù hợp cho hầu hết ứng dụng

### Cho khám phá:
Sử dụng `is_anomaly_any` (≥1 phương pháp)
- High recall, có thể có false positive
- Phát hiện nhiều điểm tiềm năng

## 🔗 Liên kết với các module khác

### Input từ:
- **HDBSCAN Clustering**: `hdbscan_phuc/hdbscan_results.csv`

### Output có thể dùng cho:
- **Energy Prediction**: Loại bỏ anomalies trước khi train model
- **Stability Analysis**: Phân tích đặc điểm cấu trúc bất ổn định
- **Data Cleaning**: Lọc dữ liệu chất lượng cao

## 📚 Dependencies

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## 🐛 Troubleshooting

### Lỗi: "Không tìm thấy file HDBSCAN results"
**Giải pháp**: Chạy HDBSCAN notebook trước:
```bash
jupyter notebook HDBSCAN.ipynb
```

### Lỗi: "Thiếu cột cluster/probability"
**Giải pháp**: Kiểm tra file HDBSCAN results có đúng format không

### Warning: "Thiếu features"
**Giải pháp**: Một số features có thể không có trong dữ liệu, module sẽ tự động bỏ qua

## 📝 Notes

- Module hoạt động độc lập, không cần retrain HDBSCAN
- Kết quả có thể tái tạo với `RANDOM_STATE` cố định
- Isolation Forest có thể mất vài phút với dữ liệu lớn
- Các biểu đồ được lưu tự động, không cần hiển thị interactive

## 👤 Author

Hoàng Phúc

## 📅 Version

1.0.0 - Initial release

---

**Lưu ý**: Module này là phần mở rộng của HDBSCAN clustering analysis. Để có kết quả tốt nhất, nên chạy HDBSCAN với tham số tối ưu trước khi chạy anomaly detection.
