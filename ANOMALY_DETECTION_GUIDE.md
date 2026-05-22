# 🔍 Hướng dẫn sử dụng Module Anomaly Detection

## 📚 Tổng quan

Module **Anomaly Detection** phát hiện các cấu trúc Carbon-24 bất thường dựa trên kết quả phân cụm HDBSCAN. Module sử dụng 3 phương pháp độc lập và kết hợp kết quả để tăng độ tin cậy.

## 🎯 Mục đích

1. **Phát hiện outliers**: Các cấu trúc không thuộc bất kỳ cụm nào
2. **Phát hiện weak members**: Các cấu trúc thuộc cụm nhưng membership yếu
3. **Phát hiện structural anomalies**: Các cấu trúc có đặc điểm bất thường trong feature space
4. **Consensus detection**: Kết hợp nhiều phương pháp để tăng độ tin cậy

## 📋 Yêu cầu

### Prerequisites
- Python 3.8+
- Đã chạy HDBSCAN clustering (file `hdbscan_phuc/hdbscan_results.csv` phải tồn tại)

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 🚀 Cách sử dụng

### Bước 1: Chạy HDBSCAN (nếu chưa có)

```bash
jupyter notebook HDBSCAN.ipynb
```

Hoặc nếu đã có file kết quả, bỏ qua bước này.

### Bước 2: Chạy Anomaly Detection

#### Option A: Chạy script Python

```bash
python carbon24_anomaly_detection.py
```

**Output:**
```
================================================================================
CARBON-24 ANOMALY DETECTION
================================================================================
Dựa trên kết quả HDBSCAN clustering
Input: hdbscan_phuc\hdbscan_results.csv
Output: carbon24_anomaly_detection

✅ Đã tạo thư mục output: carbon24_anomaly_detection

✅ Đã load HDBSCAN results: (10153, 33)
   Columns: ['material_id', 'hdbscan_cluster', 'hdbscan_probability', ...]

================================================================================
PHƯƠNG PHÁP 1: HDBSCAN NOISE POINTS
================================================================================
📊 Số điểm noise: 786 / 10,153 (7.74%)

================================================================================
PHƯƠNG PHÁP 2: HDBSCAN LOW PROBABILITY (< 0.5)
================================================================================
📊 Số điểm low probability: 1,234 / 10,153 (12.15%)

================================================================================
PHƯƠNG PHÁP 3: ISOLATION FOREST (contamination=0.1)
================================================================================
📊 Sử dụng 19 features
📊 Số điểm anomaly: 1,015 / 10,153 (10.00%)

================================================================================
KẾT HỢP CÁC PHƯƠNG PHÁP
================================================================================
📊 Kết quả tổng hợp:
   - Anomaly (Any method):       2,345 (23.09%)
   - Anomaly (Consensus ≥2):     456 (4.49%)
   - Anomaly (All 3 methods):    123 (1.21%)

✅ HOÀN TẤT ANOMALY DETECTION
```

#### Option B: Chạy Jupyter Notebook

```bash
jupyter notebook carbon24-anomaly-detection.ipynb
```

Notebook cung cấp phân tích chi tiết và interactive visualizations.

### Bước 3: Kiểm tra kết quả

```bash
python test_anomaly_detection.py
```

## 📊 Hiểu kết quả

### 1. Các cột trong `anomaly_detection_results.csv`

| Cột | Ý nghĩa | Giá trị |
|-----|---------|---------|
| `is_hdbscan_noise` | HDBSCAN noise point | 0 hoặc 1 |
| `is_low_probability` | Probability < 0.5 | 0 hoặc 1 |
| `is_isolation_forest_anomaly` | Isolation Forest phát hiện | 0 hoặc 1 |
| `isolation_forest_score` | Anomaly score | Số thực (càng âm càng bất thường) |
| `anomaly_vote_count` | Số phương pháp đồng ý | 0, 1, 2, hoặc 3 |
| `is_anomaly_consensus` | ≥2 phương pháp đồng ý | 0 hoặc 1 |
| `is_anomaly_any` | ≥1 phương pháp phát hiện | 0 hoặc 1 |
| `is_anomaly_all` | Cả 3 phương pháp đồng ý | 0 hoặc 1 |

### 2. Cách chọn anomalies

#### Cho phân tích sâu (High Precision)
```python
df = pd.read_csv('carbon24_anomaly_detection/anomaly_detection_results.csv')
high_confidence_anomalies = df[df['is_anomaly_all'] == 1]
```
- Cả 3 phương pháp đồng ý
- Độ tin cậy cao nhất
- Ít false positive
- **Use case**: Nghiên cứu chi tiết các cấu trúc thực sự bất thường

#### Cho lọc dữ liệu (Balanced)
```python
consensus_anomalies = df[df['is_anomaly_consensus'] == 1]
```
- Ít nhất 2/3 phương pháp đồng ý
- Cân bằng precision/recall
- **Use case**: Loại bỏ anomalies trước khi train model

#### Cho khám phá (High Recall)
```python
potential_anomalies = df[df['is_anomaly_any'] == 1]
```
- Ít nhất 1 phương pháp phát hiện
- Phát hiện nhiều nhất
- Có thể có false positive
- **Use case**: Khám phá các điểm tiềm năng bất thường

### 3. Phân tích năng lượng

```python
# So sánh năng lượng
anomalies = df[df['is_anomaly_consensus'] == 1]
normal = df[df['is_anomaly_consensus'] == 0]

print(f"Anomaly energy: {anomalies['relative_energy'].mean():.4f}")
print(f"Normal energy: {normal['relative_energy'].mean():.4f}")
print(f"Difference: {anomalies['relative_energy'].mean() - normal['relative_energy'].mean():.4f}")
```

**Giải thích:**
- Nếu anomalies có năng lượng **cao hơn** → Cấu trúc kém ổn định
- Nếu anomalies có năng lượng **thấp hơn** → Cấu trúc ổn định bất thường (có thể là cấu trúc đặc biệt)

## 📈 Visualizations

Module tự động tạo 4 biểu đồ:

### 1. `anomaly_methods_comparison.png`
- 4 subplot so sánh các phương pháp trong không gian PCA
- Màu xanh: Normal
- Màu đỏ: Anomaly

### 2. `anomaly_vote_distribution.png`
- Bar chart phân bố vote count (0, 1, 2, 3)
- Cho biết mức độ đồng thuận giữa các phương pháp

### 3. `anomaly_energy_distribution.png`
- Histogram so sánh phân bố năng lượng
- Anomaly vs Normal cho từng phương pháp

### 4. `isolation_forest_score_distribution.png`
- Phân bố anomaly score từ Isolation Forest
- Càng âm càng bất thường

## 🔧 Tùy chỉnh

### Thay đổi ngưỡng probability

Trong `carbon24_anomaly_detection.py`:

```python
HDBSCAN_PROBABILITY_THRESHOLD = 0.5  # Mặc định
# Giảm xuống 0.3 để phát hiện nhiều hơn
# Tăng lên 0.7 để chặt chẽ hơn
```

### Thay đổi contamination của Isolation Forest

```python
ISOLATION_FOREST_CONTAMINATION = 0.1  # Mặc định (10%)
# Tăng lên 0.15 nếu nghi ngờ có nhiều anomalies
# Giảm xuống 0.05 nếu dữ liệu sạch
```

### Thêm/bớt features

```python
ANOMALY_FEATURES = [
    "num_atoms",
    "a", "b", "c",
    # Thêm features khác nếu cần
]
```

## 🔗 Tích hợp với Dashboard

### Thêm vào Streamlit Dashboard

```bash
python add_anomaly_to_dashboard.py
```

Sau đó trong `carbon24_dashboard.py`, thêm tab:

```python
tabs = st.tabs([
    "Overview",
    "K-Means",
    "GMM",
    "Hierarchical",
    "HDBSCAN",
    "Anomaly Detection",  # ← Thêm tab mới
    "Comparison"
])

with tabs[5]:  # Anomaly Detection tab
    render_anomaly_detection_tab()
```

## 📊 Use Cases

### 1. Data Cleaning
```python
# Loại bỏ anomalies trước khi train model
df = pd.read_csv('carbon24_anomaly_detection/anomaly_detection_results.csv')
clean_data = df[df['is_anomaly_consensus'] == 0]
clean_data.to_csv('carbon24_clean.csv', index=False)
```

### 2. Focused Analysis
```python
# Phân tích chi tiết các anomalies
anomalies = df[df['is_anomaly_all'] == 1]

# Xem đặc điểm cấu trúc
print(anomalies[['num_atoms', 'volume', 'density', 'relative_energy']].describe())

# Xem phân bố crystal system
print(anomalies['crystal_system'].value_counts())
```

### 3. Energy Prediction
```python
# Train model chỉ trên normal data
normal_data = df[df['is_anomaly_consensus'] == 0]

from sklearn.ensemble import RandomForestRegressor
X = normal_data[ANOMALY_FEATURES]
y = normal_data['relative_energy']

model = RandomForestRegressor()
model.fit(X, y)
```

### 4. Stability Analysis
```python
# So sánh stability giữa anomalies và normal
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(normal['relative_energy'], bins=50, alpha=0.5, label='Normal', density=True)
ax.hist(anomalies['relative_energy'], bins=50, alpha=0.5, label='Anomaly', density=True)

ax.set_xlabel('Relative Energy (eV/atom)')
ax.set_ylabel('Density')
ax.legend()
plt.show()
```

## 🐛 Troubleshooting

### Lỗi: "Không tìm thấy file HDBSCAN results"

**Nguyên nhân**: Chưa chạy HDBSCAN clustering

**Giải pháp**:
```bash
jupyter notebook HDBSCAN.ipynb
# Chạy toàn bộ notebook
```

### Lỗi: "Thiếu cột cluster/probability"

**Nguyên nhân**: File HDBSCAN results không đúng format

**Giải pháp**: Kiểm tra file có các cột:
- `hdbscan_cluster` hoặc tương tự
- `hdbscan_probability` hoặc tương tự

### Warning: "Thiếu features"

**Nguyên nhân**: Một số features không có trong dữ liệu

**Giải pháp**: Module sẽ tự động bỏ qua, không ảnh hưởng kết quả

### Isolation Forest chạy chậm

**Nguyên nhân**: Dữ liệu lớn

**Giải pháp**: Giảm `n_estimators` trong code:
```python
iso_forest = IsolationForest(
    n_estimators=50,  # Giảm từ 100 xuống 50
    ...
)
```

## 📚 Tài liệu tham khảo

### HDBSCAN
- Paper: "Density-Based Clustering Based on Hierarchical Density Estimates"
- Ưu điểm: Tự động phát hiện noise, không cần chỉ định số cụm

### Isolation Forest
- Paper: "Isolation Forest" (Liu et al., 2008)
- Ưu điểm: Hiệu quả với high-dimensional data, không cần giả định phân phối

### Consensus Methods
- Kết hợp nhiều phương pháp tăng độ tin cậy
- Giảm false positive rate

## 💡 Best Practices

1. **Luôn kiểm tra kết quả trực quan**: Xem biểu đồ PCA để hiểu phân bố
2. **So sánh năng lượng**: Anomalies có năng lượng khác biệt không?
3. **Phân tích vote count**: Nhiều phương pháp đồng ý = tin cậy cao
4. **Sử dụng consensus**: Cân bằng giữa precision và recall
5. **Validate manually**: Kiểm tra một số anomalies bằng tay để xác nhận

## 🎓 Ví dụ workflow hoàn chỉnh

```python
# 1. Load data
import pandas as pd
df = pd.read_csv('carbon24_anomaly_detection/anomaly_detection_results.csv')

# 2. Chọn consensus anomalies
anomalies = df[df['is_anomaly_consensus'] == 1]
normal = df[df['is_anomaly_consensus'] == 0]

print(f"Anomalies: {len(anomalies):,} ({len(anomalies)/len(df)*100:.2f}%)")
print(f"Normal: {len(normal):,} ({len(normal)/len(df)*100:.2f}%)")

# 3. Phân tích năng lượng
print(f"\nEnergy comparison:")
print(f"Anomaly mean: {anomalies['relative_energy'].mean():.4f}")
print(f"Normal mean: {normal['relative_energy'].mean():.4f}")
print(f"Difference: {anomalies['relative_energy'].mean() - normal['relative_energy'].mean():.4f}")

# 4. Phân tích cấu trúc
print(f"\nCrystal system distribution:")
print(anomalies['crystal_system'].value_counts())

# 5. Export cho phân tích tiếp
anomalies.to_csv('anomalies_for_analysis.csv', index=False)
normal.to_csv('normal_for_training.csv', index=False)

# 6. Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Energy distribution
axes[0].hist(normal['relative_energy'], bins=50, alpha=0.5, label='Normal')
axes[0].hist(anomalies['relative_energy'], bins=50, alpha=0.5, label='Anomaly')
axes[0].set_xlabel('Relative Energy (eV/atom)')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].set_title('Energy Distribution')

# Vote count
vote_counts = df['anomaly_vote_count'].value_counts().sort_index()
axes[1].bar(vote_counts.index, vote_counts.values)
axes[1].set_xlabel('Vote Count')
axes[1].set_ylabel('Number of Samples')
axes[1].set_title('Anomaly Vote Distribution')

plt.tight_layout()
plt.savefig('my_anomaly_analysis.png', dpi=300)
plt.show()
```

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. File README: `ANOMALY_DETECTION_README.md`
2. Test script: `python test_anomaly_detection.py`
3. Notebook demo: `carbon24-anomaly-detection.ipynb`

---

**Version**: 1.0.0  
**Author**: Hoàng Phúc  
**Last Updated**: 2026-05-21
