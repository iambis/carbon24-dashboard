# 🔬 Hướng Dẫn Phân Tích Stability & Dự Đoán Energy - Carbon-24

## 📋 Tổng Quan

Dự án này mở rộng từ phân tích clustering ban đầu để thêm:

1. **Phát hiện nhóm cấu trúc ổn định/kém ổn định** - Phân loại các cluster theo mức độ ổn định năng lượng
2. **Dự đoán energy_per_atom** - Sử dụng Machine Learning để dự đoán năng lượng
3. **Trực quan hóa kết quả** - Dashboard tương tác và visualizations chi tiết

## 🎯 Kết Quả Chính

### Phân Loại Stability

Dựa trên phân tích clustering với k=3, các cấu trúc Carbon được phân thành 3 nhóm:

- **Cluster 0: Highly Stable** (33.6% - 3,409 cấu trúc)
  - Mean relative energy: -0.4725 eV/atom
  - Đây là nhóm cấu trúc ổn định nhất

- **Cluster 1: Highly Stable** (22.0% - 2,229 cấu trúc)
  - Mean relative energy: -0.3771 eV/atom
  - Nhóm ổn định thứ hai

- **Cluster 2: Less Stable** (44.5% - 4,515 cấu trúc)
  - Mean relative energy: 0.5429 eV/atom
  - Nhóm kém ổn định hơn

### Dự Đoán Energy

Các mô hình Machine Learning được thử nghiệm:

| Model | Test R² | Test MAE (eV/atom) | Test RMSE (eV/atom) |
|-------|---------|-------------------|---------------------|
| **Random Forest** | **1.0000** | **0.0014** | **0.0026** |
| Gradient Boosting | 1.0000 | 0.0042 | 0.0055 |
| Ridge Regression | 0.8988 | 1.5101 | 1.8065 |
| Lasso Regression | 0.8297 | 2.1678 | 2.3428 |

**🏆 Best Model: Random Forest**
- Độ chính xác gần như hoàn hảo (R² = 1.0000)
- Sai số trung bình chỉ 0.0014 eV/atom

### Feature Importance

Top 5 features quan trọng nhất cho dự đoán energy:

1. **num_atoms** - Số lượng nguyên tử (quan trọng nhất)
2. bond_length_range - Khoảng độ dài liên kết
3. bond_complexity - Độ phức tạp liên kết
4. volume_ratio - Tỷ lệ thể tích
5. min_bond_length - Độ dài liên kết tối thiểu

## 📁 Cấu Trúc Files

```
carbon24_stability_analysis/
├── cluster_stability_classification.csv    # Phân loại stability của các cluster
├── prediction_model_comparison.csv         # So sánh hiệu suất các mô hình
├── best_model_predictions.csv             # Kết quả dự đoán của mô hình tốt nhất
├── feature_importance.csv                 # Độ quan trọng của các features
├── ANALYSIS_REPORT.txt                    # Báo cáo tổng hợp
└── figures/
    ├── stability_analysis_overview.png    # Tổng quan phân tích
    ├── cluster_stability_details.png      # Chi tiết stability theo cluster
    ├── feature_importance.png             # Biểu đồ feature importance
    ├── prediction_error_analysis.png      # Phân tích lỗi dự đoán
    └── pca_stability_visualization.png    # Trực quan hóa PCA
```

## 🚀 Cách Sử Dụng

### 1. Chạy Phân Tích

```bash
python carbon24-stability-analysis.py
```

Script này sẽ:
- Load dữ liệu và thực hiện clustering
- Phân loại stability cho từng cluster
- Train và đánh giá các mô hình ML
- Tạo visualizations
- Lưu kết quả và báo cáo

**Thời gian chạy:** ~15-20 giây

### 2. Xem Dashboard Tương Tác

```bash
streamlit run carbon24_interactive_dashboard.py
```

Dashboard sẽ mở tại: `http://localhost:8501`

#### Các trang trong Dashboard:

**📋 Overview**
- Tổng quan về dự án
- Metrics chính
- Key findings
- Quick visualizations

**⚡ Stability Analysis**
- Phân tích chi tiết về stability
- Phân bố relative energy theo cluster
- So sánh stability giữa các cluster
- Bảng thống kê chi tiết

**🤖 Energy Prediction**
- So sánh hiệu suất các mô hình
- Biểu đồ Actual vs Predicted
- Phân tích lỗi dự đoán
- Feature importance

**🔍 Cluster Explorer**
- Khám phá từng cluster chi tiết
- Phân bố features trong cluster
- So sánh features giữa các cluster
- Scatter plots tương tác

**📊 Data Explorer**
- Lọc và tìm kiếm dữ liệu
- Xem bảng dữ liệu
- Download filtered data
- Summary statistics

## 📊 Visualizations

### 1. Stability Analysis Overview
![Stability Overview](carbon24_stability_analysis/figures/stability_analysis_overview.png)

Bao gồm:
- Phân bố relative energy theo cluster
- Box plot so sánh stability
- So sánh hiệu suất các mô hình
- Actual vs Predicted energy

### 2. Cluster Stability Details
![Cluster Details](carbon24_stability_analysis/figures/cluster_stability_details.png)

Hiển thị:
- Kích thước và phân loại stability của từng cluster
- Mean relative energy theo cluster

### 3. Feature Importance
![Feature Importance](carbon24_stability_analysis/figures/feature_importance.png)

Top features quan trọng nhất cho dự đoán energy

### 4. Prediction Error Analysis
![Error Analysis](carbon24_stability_analysis/figures/prediction_error_analysis.png)

Phân tích:
- Phân bố lỗi dự đoán
- Residual plot

### 5. PCA Visualization
![PCA](carbon24_stability_analysis/figures/pca_stability_visualization.png)

Trực quan hóa:
- Structures colored by stability
- Structures colored by relative energy

## 🔍 Phân Tích Chi Tiết

### Stability Classification

Các cluster được phân loại dựa trên mean relative energy:

- **Highly Stable**: mean_energy < 0.15 eV/atom
- **Moderately Stable**: 0.15 ≤ mean_energy < 0.30 eV/atom
- **Less Stable**: mean_energy ≥ 0.30 eV/atom

### Model Selection

**Tại sao chọn Random Forest?**

1. **Độ chính xác cao nhất**: R² = 1.0000
2. **Sai số thấp nhất**: MAE = 0.0014 eV/atom
3. **Robust**: Xử lý tốt với nhiều features
4. **Interpretable**: Có thể phân tích feature importance

### Feature Engineering

Các features được sử dụng:
- **Geometric features**: a, b, c, alpha, beta, gamma
- **Structural features**: num_atoms, bond lengths, coordination
- **Derived features**: volume_ratio, lattice_asymmetry, bond_complexity

## 📈 Insights

### 1. Phân Bố Stability

- **55.6%** cấu trúc thuộc nhóm Highly Stable (Clusters 0 & 1)
- **44.5%** cấu trúc thuộc nhóm Less Stable (Cluster 2)
- Có sự phân tách rõ ràng giữa các nhóm stability

### 2. Dự Đoán Energy

- Geometric features có thể dự đoán energy với độ chính xác rất cao
- **num_atoms** là feature quan trọng nhất (99.99% importance)
- Các features về bond length và complexity cũng đóng vai trò quan trọng

### 3. Cluster Characteristics

**Cluster 0 (Highly Stable):**
- Mean energy: -0.4725 eV/atom
- Cấu trúc compact, ổn định
- 33.6% tổng số cấu trúc

**Cluster 1 (Highly Stable):**
- Mean energy: -0.3771 eV/atom
- Median energy rất thấp: -1.1366 eV/atom
- 22.0% tổng số cấu trúc

**Cluster 2 (Less Stable):**
- Mean energy: 0.5429 eV/atom
- Năng lượng cao hơn đáng kể
- 44.5% tổng số cấu trúc

## 🎓 Ứng Dụng

### 1. Material Discovery
- Xác định các cấu trúc Carbon ổn định cho ứng dụng thực tế
- Tập trung nghiên cứu vào Clusters 0 & 1

### 2. Property Prediction
- Dự đoán energy cho cấu trúc mới
- Screening nhanh các cấu trúc tiềm năng

### 3. Structure-Property Relationships
- Hiểu mối quan hệ giữa cấu trúc hình học và stability
- Thiết kế cấu trúc với properties mong muốn

## 🔧 Customization

### Thay đổi số lượng clusters

Trong file `carbon24-stability-analysis.py`:

```python
# Thay đổi optimal_k
optimal_k = 4  # Thay vì 3
```

### Thử các mô hình khác

Thêm mô hình mới vào dictionary `models`:

```python
from sklearn.neural_network import MLPRegressor

models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
    # ... các mô hình khác
}
```

### Điều chỉnh stability thresholds

```python
if mean_energy < 0.10:  # Thay vì 0.15
    stability = "Highly Stable"
elif mean_energy < 0.25:  # Thay vì 0.30
    stability = "Moderately Stable"
```

## 📚 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly
```

## 🐛 Troubleshooting

### Lỗi: File not found

**Giải pháp:**
- Đảm bảo đã chạy `carbon24-stability-analysis.py` trước
- Kiểm tra đường dẫn đến các file dữ liệu

### Dashboard không hiển thị

**Giải pháp:**
```bash
# Cài đặt lại streamlit
pip install --upgrade streamlit

# Chạy với port khác
streamlit run carbon24_interactive_dashboard.py --server.port 8502
```

### Memory error

**Giải pháp:**
- Giảm số lượng samples trong visualization
- Sử dụng sampling cho các biểu đồ lớn

## 📞 Support

Nếu có vấn đề hoặc câu hỏi:
1. Kiểm tra file `ANALYSIS_REPORT.txt` để xem kết quả chi tiết
2. Xem lại các visualizations trong folder `figures/`
3. Kiểm tra console output khi chạy script

## 🎯 Next Steps

### Mở rộng thêm:

1. **Deep Learning Models**
   - Thử Graph Neural Networks cho dự đoán properties
   - CNN cho pattern recognition trong structures

2. **Advanced Analysis**
   - Time series analysis nếu có dữ liệu temporal
   - Anomaly detection cho unusual structures

3. **Interactive Tools**
   - 3D visualization của crystal structures
   - Real-time prediction tool

4. **Optimization**
   - Hyperparameter tuning cho models
   - Feature selection optimization
   - Ensemble methods

## 📝 Citation

Nếu sử dụng phân tích này trong nghiên cứu, vui lòng cite:

```
Carbon-24 Stability Analysis & Energy Prediction
Dataset: Carbon-24 Allotropes
Analysis Date: 2026
```

---

**Chúc bạn phân tích thành công! 🎉**
