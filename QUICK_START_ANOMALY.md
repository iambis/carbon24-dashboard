# 🚀 Quick Start: Anomaly Detection trên Dashboard

## ✅ Đã hoàn tất

Module Anomaly Detection đã chạy thành công và tạo ra các files:

```
carbon24_anomaly_detection/
├── anomaly_detection_results.csv      (2.0 MB) ✅
├── anomaly_summary.csv                (2.7 KB) ✅
├── anomaly_method_comparison.csv      (466 B)  ✅
├── anomaly_details.csv                (69 KB)  ✅
└── figures/
    ├── anomaly_methods_comparison.png         ✅
    ├── anomaly_vote_distribution.png          ✅
    ├── anomaly_energy_distribution.png        ✅
    └── isolation_forest_score_distribution.png ✅
```

## 📊 Kết quả

### Tổng quan:
- **Tổng số mẫu**: 10,153
- **HDBSCAN Noise**: 786 (7.74%)
- **Low Probability**: 800 (7.88%)
- **Isolation Forest**: 1,016 (10.01%)
- **Consensus (≥2 methods)**: 791 (7.79%)
- **All 3 methods**: 180 (1.77%)

### Features sử dụng:
Do file HDBSCAN không có features cấu trúc chi tiết, module đã tự động sử dụng:
- `pca1`, `pca2` (từ PCA)
- `relative_energy`
- `energy`

## 🎯 Xem trên Dashboard

### Bước 1: Chạy Dashboard
```bash
streamlit run carbon24_dashboard.py
```

### Bước 2: Chọn page
1. Dashboard mở trong browser
2. **Sidebar** → Chọn **"🔍 Phát hiện dị biệt"**

### Bước 3: Khám phá dữ liệu

Dashboard sẽ hiển thị:

#### 📊 Tổng quan (4 metrics)
- Tổng số mẫu: 10,153
- Consensus Anomalies: 791 (7.79%)
- High Confidence: 180 (1.77%)
- Any Method: 1,631 (16.06%)

#### 🔬 So sánh phương pháp
- **Bar chart**: Tỷ lệ anomaly của từng phương pháp
- **Vote distribution**: 
  - 0 methods: 8,522 (83.94%)
  - 1 method: 840 (8.27%)
  - 2 methods: 611 (6.02%)
  - 3 methods: 180 (1.77%)

#### 🗺️ PCA Visualization
- Scatter plot interactive
- Chọn phương pháp:
  - HDBSCAN Noise
  - Low Probability
  - Isolation Forest
  - Consensus (≥2) ← **Khuyến nghị**
  - All 3 methods

#### ⚡ Energy Analysis
- **Table**: So sánh năng lượng
  - Anomalies có năng lượng cao hơn → Kém ổn định
  - Consensus: +0.0717 eV/atom
  
- **Histogram**: Phân bố relative energy
  - Normal mean: ~0.301 eV/atom
  - Anomaly mean: ~0.373 eV/atom

#### 📋 Consensus Anomalies
- 791 anomalies được ≥2 phương pháp đồng ý
- Filter theo vote count (2 hoặc 3)
- Download CSV

#### 💡 Khuyến nghị
- **Phân tích sâu**: Dùng `is_anomaly_all` (180 samples)
- **Lọc dữ liệu**: Dùng `is_anomaly_consensus` (791 samples) ← **Khuyến nghị**
- **Khám phá**: Dùng `is_anomaly_any` (1,631 samples)

## 🔍 Phân tích chi tiết

### Đặc điểm Anomalies:

1. **HDBSCAN Noise (786)**:
   - Điểm không thuộc cụm nào
   - Energy cao hơn: +0.0735 eV/atom
   - Cấu trúc thưa, không đủ mật độ

2. **Low Probability (800)**:
   - Membership yếu (< 0.5)
   - Energy cao hơn: +0.0716 eV/atom
   - Nằm ở biên cụm

3. **Isolation Forest (1,016)**:
   - Dễ bị "cô lập" trong feature space
   - Energy **thấp hơn**: -0.1577 eV/atom ⚠️
   - Có thể là cấu trúc đặc biệt

### Consensus Anomalies (791):
- Được ≥2 phương pháp đồng ý
- Energy cao hơn: +0.0718 eV/atom
- **Khuyến nghị sử dụng** cho lọc dữ liệu

### High Confidence (180):
- Cả 3 phương pháp đồng ý
- Energy cao hơn: +0.0296 eV/atom
- Thực sự bất thường

## 💾 Sử dụng dữ liệu

### Load trong Python:
```python
import pandas as pd

# Load full results
df = pd.read_csv('carbon24_anomaly_detection/anomaly_detection_results.csv')

# Filter consensus anomalies
anomalies = df[df['is_anomaly_consensus'] == 1]
normal = df[df['is_anomaly_consensus'] == 0]

print(f"Anomalies: {len(anomalies):,}")
print(f"Normal: {len(normal):,}")

# Phân tích năng lượng
print(f"\nAnomaly energy: {anomalies['relative_energy'].mean():.4f}")
print(f"Normal energy: {normal['relative_energy'].mean():.4f}")
```

### Download từ Dashboard:
1. Vào page "Phát hiện dị biệt"
2. Scroll xuống "Chi tiết Consensus Anomalies"
3. Click **"📥 Download Anomaly Details (CSV)"**

## 🔄 Cập nhật dữ liệu

Nếu muốn chạy lại với tham số khác:

### 1. Sửa tham số trong `carbon24_anomaly_detection.py`:
```python
HDBSCAN_PROBABILITY_THRESHOLD = 0.5  # Thay đổi ngưỡng
ISOLATION_FOREST_CONTAMINATION = 0.1  # Thay đổi tỷ lệ
```

### 2. Chạy lại:
```bash
python carbon24_anomaly_detection.py
```

### 3. Refresh Dashboard:
- Restart dashboard: `Ctrl+C` → `streamlit run carbon24_dashboard.py`
- Hoặc clear cache trong dashboard

## 📈 Use Cases

### 1. Lọc dữ liệu cho training:
```python
# Loại bỏ consensus anomalies
clean_data = df[df['is_anomaly_consensus'] == 0]
clean_data.to_csv('carbon24_clean.csv', index=False)
```

### 2. Phân tích cấu trúc bất thường:
```python
# Xem high confidence anomalies
high_conf = df[df['is_anomaly_all'] == 1]
print(high_conf[['material_id', 'relative_energy', 'crystal_system']])
```

### 3. So sánh phương pháp:
```python
# Overlap giữa các phương pháp
hdbscan_noise = set(df[df['is_hdbscan_noise'] == 1].index)
iso_forest = set(df[df['is_isolation_forest_anomaly'] == 1].index)

overlap = hdbscan_noise & iso_forest
print(f"Overlap: {len(overlap)} samples")
```

## ⚠️ Lưu ý

1. **Isolation Forest có kết quả khác**:
   - Phát hiện anomalies có energy **thấp hơn**
   - Có thể là cấu trúc ổn định bất thường
   - Cần phân tích thêm

2. **Consensus là an toàn nhất**:
   - Ít nhất 2/3 phương pháp đồng ý
   - Giảm false positive
   - Khuyến nghị cho production

3. **Features hạn chế**:
   - Chỉ dùng PCA + energy
   - Nếu có features cấu trúc đầy đủ, kết quả sẽ tốt hơn

## 🎉 Hoàn tất!

Bây giờ bạn có thể:
- ✅ Xem Anomaly Detection trên Dashboard
- ✅ Phân tích 791 consensus anomalies
- ✅ Download dữ liệu để sử dụng
- ✅ Lọc dữ liệu cho các bước tiếp theo

---

**Ngày tạo**: 2026-05-21  
**Status**: ✅ Working  
**Files**: 4 CSV + 4 PNG
