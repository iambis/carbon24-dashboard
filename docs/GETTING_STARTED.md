# 🚀 Getting Started - Carbon-24 Stability Analysis

## 📋 Tổng Quan Nhanh

Dự án này mở rộng phân tích clustering Carbon-24 để thêm:
- ⚡ **Phát hiện nhóm ổn định/kém ổn định**
- 🤖 **Dự đoán energy_per_atom** (R² = 1.0000)
- 📊 **Dashboard tương tác** với Streamlit
- 📄 **Báo cáo PDF** chuyên nghiệp

## ⚡ Quick Start (3 bước)

### Bước 1: Demo Nhanh (30 giây)

```bash
python demo_stability_analysis.py
```

Xem ngay:
- Phân loại stability của 3 clusters
- Kết quả dự đoán energy
- Top 10 features quan trọng nhất

### Bước 2: Phân Tích Đầy Đủ (20 giây)

```bash
python carbon24-stability-analysis.py
```

Tạo ra:
- ✅ 5 CSV files với kết quả chi tiết
- ✅ 5 PNG visualizations
- ✅ Báo cáo text đầy đủ

### Bước 3: Dashboard Tương Tác

```bash
streamlit run carbon24_interactive_dashboard.py
```

Mở trình duyệt tại: **http://localhost:8501**

Khám phá:
- 📋 Overview - Tổng quan
- ⚡ Stability Analysis - Phân tích chi tiết
- 🤖 Energy Prediction - Dự đoán
- 🔍 Cluster Explorer - Khám phá clusters
- 📊 Data Explorer - Lọc dữ liệu

## 📊 Kết Quả Chính

### ⚡ Stability Classification

```
🟢 Cluster 0: Highly Stable
   - 3,409 structures (33.6%)
   - Mean energy: -0.4725 eV/atom

🟢 Cluster 1: Highly Stable
   - 2,229 structures (22.0%)
   - Mean energy: -0.3771 eV/atom

🔴 Cluster 2: Less Stable
   - 4,515 structures (44.5%)
   - Mean energy: 0.5429 eV/atom
```

### 🤖 Energy Prediction

```
⭐ Random Forest (Best Model)
   - Test R²: 1.0000
   - Test MAE: 0.0014 eV/atom
   - Độ chính xác gần hoàn hảo!
```

## 📁 Files Được Tạo Ra

```
carbon24_stability_analysis/
├── 📊 cluster_stability_classification.csv
├── 📈 prediction_model_comparison.csv
├── 🎯 best_model_predictions.csv
├── 🔍 feature_importance.csv
├── 📄 ANALYSIS_REPORT.txt
├── 📕 Carbon24_Stability_Report_YYYYMMDD.pdf
└── 📁 figures/
    ├── stability_analysis_overview.png
    ├── cluster_stability_details.png
    ├── feature_importance.png
    ├── prediction_error_analysis.png
    └── pca_stability_visualization.png
```

## 🎯 Use Cases

### 1. Nghiên Cứu Material Science
```python
# Tìm các cấu trúc ổn định nhất
import pandas as pd
df = pd.read_csv('carbon24_stability_analysis/cluster_stability_classification.csv')
highly_stable = df[df['stability'] == 'Highly Stable']
print(highly_stable)
```

### 2. Dự Đoán Energy Cho Cấu Trúc Mới
```python
# Load model và predict
from sklearn.ensemble import RandomForestRegressor
import joblib

# Train model (đã có trong analysis script)
# Sau đó dùng để predict cho cấu trúc mới
# new_structure_features = [...]
# predicted_energy = model.predict([new_structure_features])
```

### 3. Phân Tích Tương Quan
```python
# Xem mối quan hệ giữa features và energy
import pandas as pd
predictions = pd.read_csv('carbon24_stability_analysis/best_model_predictions.csv')
print(predictions.describe())
```

## 📚 Documentation

| File | Mô Tả |
|------|-------|
| **README_STABILITY_ANALYSIS.md** | Tổng quan nhanh |
| **HUONG_DAN_STABILITY_ANALYSIS.md** | Hướng dẫn chi tiết (tiếng Việt) |
| **GETTING_STARTED.md** | File này - Quick start |
| **ANALYSIS_REPORT.txt** | Báo cáo kết quả |

## 🔧 Customization

### Thay đổi số clusters

Trong `carbon24-stability-analysis.py`:
```python
optimal_k = 4  # Thay vì 3
```

### Thử model khác

```python
from sklearn.neural_network import MLPRegressor

models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
}
```

### Điều chỉnh stability thresholds

```python
if mean_energy < 0.10:  # Thay vì 0.15
    stability = "Highly Stable"
```

## 📊 Visualizations Preview

### 1. Stability Overview
- Phân bố energy theo cluster
- Box plot so sánh
- Model performance
- Actual vs Predicted

### 2. Cluster Details
- Kích thước clusters
- Mean energy comparison

### 3. Feature Importance
- Top features cho prediction
- Contribution analysis

### 4. Error Analysis
- Error distribution
- Residual plot

### 5. PCA Visualization
- 2D projection của structures
- Colored by stability/energy

## 🎓 Advanced Usage

### 1. Tạo Báo Cáo PDF

```bash
python generate_pdf_report.py
```

Output: `Carbon24_Stability_Report_YYYYMMDD.pdf`

### 2. Export Data Từ Dashboard

1. Mở dashboard
2. Vào "Data Explorer"
3. Apply filters
4. Click "Download Filtered Data as CSV"

### 3. Batch Processing

```python
# Process multiple datasets
datasets = ['dataset1.csv', 'dataset2.csv']
for dataset in datasets:
    # Run analysis
    pass
```

## 🐛 Troubleshooting

### Lỗi: Module not found

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly
```

### Dashboard không mở

```bash
# Thử port khác
streamlit run carbon24_interactive_dashboard.py --server.port 8502
```

### Memory error

```python
# Giảm sample size trong visualization
sample_size = min(1000, len(X))  # Thay vì 5000
```

## 📈 Performance

| Task | Time | Output |
|------|------|--------|
| Demo | ~5s | Console output |
| Full Analysis | ~20s | 5 CSVs + 5 PNGs + Report |
| Dashboard | ~3s | Interactive web app |
| PDF Report | ~10s | Professional PDF |

## 🎯 Next Steps

### Immediate:
1. ✅ Run demo
2. ✅ View dashboard
3. ✅ Generate PDF report

### Short-term:
- Explore different clusters
- Try different ML models
- Customize visualizations

### Long-term:
- Integrate with other analyses
- Build prediction API
- Create 3D visualizations

## 💡 Tips & Tricks

### Tip 1: Quick Comparison
```bash
# So sánh nhiều models nhanh
python carbon24-stability-analysis.py > results.txt
```

### Tip 2: Dashboard Shortcuts
- `Ctrl + R` - Refresh dashboard
- `Ctrl + C` - Stop server
- Sidebar - Quick navigation

### Tip 3: Export Figures
```python
# Trong dashboard, right-click on chart
# -> "Save image as..."
```

## 📞 Support & Resources

### Documentation
- 📖 HUONG_DAN_STABILITY_ANALYSIS.md - Chi tiết đầy đủ
- 📄 ANALYSIS_REPORT.txt - Kết quả phân tích
- 📕 PDF Report - Báo cáo chuyên nghiệp

### Quick Help
```bash
python demo_stability_analysis.py  # Xem tổng quan
```

### Check Results
```bash
ls carbon24_stability_analysis/  # Xem files đã tạo
```

## 🏆 Success Checklist

- [ ] Chạy demo thành công
- [ ] Xem được dashboard
- [ ] Hiểu kết quả stability classification
- [ ] Hiểu kết quả energy prediction
- [ ] Tạo được PDF report
- [ ] Đọc documentation

## 🎉 Congratulations!

Bạn đã sẵn sàng sử dụng hệ thống phân tích Carbon-24!

**Bắt đầu ngay:**
```bash
python demo_stability_analysis.py
streamlit run carbon24_interactive_dashboard.py
```

---

**Questions?** Đọc `HUONG_DAN_STABILITY_ANALYSIS.md` để biết thêm chi tiết!
