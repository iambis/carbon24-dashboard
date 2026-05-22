# 💎 Carbon-24 Stability Analysis & Energy Prediction

## 🎯 Mục Tiêu Dự Án

Mở rộng từ phân tích clustering ban đầu để:

1. **Phát hiện nhóm cấu trúc ổn định/kém ổn định** dựa trên relative energy
2. **Dự đoán energy_per_atom** sử dụng Machine Learning
3. **Trực quan hóa kết quả** với dashboard tương tác và báo cáo PDF

## 📊 Kết Quả Chính

### ⚡ Phân Loại Stability

| Cluster | Stability | Số Cấu Trúc | % | Mean Energy (eV/atom) |
|---------|-----------|-------------|---|-----------------------|
| 0 | Highly Stable | 3,409 | 33.6% | -0.4725 |
| 1 | Highly Stable | 2,229 | 22.0% | -0.3771 |
| 2 | Less Stable | 4,515 | 44.5% | 0.5429 |

**Insight:** 55.6% cấu trúc thuộc nhóm Highly Stable

### 🤖 Dự Đoán Energy

| Model | Test R² | Test MAE (eV/atom) | Test RMSE (eV/atom) |
|-------|---------|-------------------|---------------------|
| **Random Forest** ⭐ | **1.0000** | **0.0014** | **0.0026** |
| Gradient Boosting | 1.0000 | 0.0042 | 0.0055 |
| Ridge Regression | 0.8988 | 1.5101 | 1.8065 |
| Lasso Regression | 0.8297 | 2.1678 | 2.3428 |

**Best Model:** Random Forest với độ chính xác gần như hoàn hảo!

### 🔍 Feature Importance

Top 5 features quan trọng nhất:

1. **num_atoms** (99.99%) - Số lượng nguyên tử
2. bond_length_range - Khoảng độ dài liên kết
3. bond_complexity - Độ phức tạp liên kết
4. volume_ratio - Tỷ lệ thể tích
5. min_bond_length - Độ dài liên kết tối thiểu

## 🚀 Quick Start

### 1. Chạy Phân Tích

```bash
python carbon24-stability-analysis.py
```

**Output:**
- Cluster stability classification
- Model comparison results
- Feature importance analysis
- 5 visualization figures
- Detailed analysis report

**Thời gian:** ~15-20 giây

### 2. Xem Dashboard Tương Tác

```bash
streamlit run carbon24_interactive_dashboard.py
```

**Dashboard URL:** http://localhost:8501

**Tính năng:**
- 📋 Overview - Tổng quan dự án
- ⚡ Stability Analysis - Phân tích stability chi tiết
- 🤖 Energy Prediction - Kết quả dự đoán
- 🔍 Cluster Explorer - Khám phá từng cluster
- 📊 Data Explorer - Lọc và tìm kiếm dữ liệu

### 3. Tạo Báo Cáo PDF

```bash
python generate_pdf_report.py
```

**Output:** `Carbon24_Stability_Report_YYYYMMDD.pdf`

**Nội dung:**
- Title page với key metrics
- Executive summary
- Stability analysis visualizations
- Energy prediction results
- Feature importance
- Detailed statistics

## 📁 Cấu Trúc Files

```
📦 carbon24_stability_analysis/
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

## 📚 Documentation

- **HUONG_DAN_STABILITY_ANALYSIS.md** - Hướng dẫn chi tiết (tiếng Việt)
- **ANALYSIS_REPORT.txt** - Báo cáo tổng hợp
- **PDF Report** - Báo cáo chuyên nghiệp

## 🎓 Key Insights

### 1. Structure-Energy Relationships
- Geometric features có thể dự đoán energy với độ chính xác rất cao (R² = 1.0000)
- **num_atoms** là yếu tố quyết định nhất (99.99% importance)
- Các features về bond length và complexity cũng quan trọng

### 2. Stability Groups
- **Highly Stable** (Clusters 0 & 1): 55.6% cấu trúc
  - Mean energy: -0.47 to -0.38 eV/atom
  - Phù hợp cho ứng dụng thực tế
  
- **Less Stable** (Cluster 2): 44.5% cấu trúc
  - Mean energy: 0.54 eV/atom
  - Năng lượng cao hơn đáng kể

### 3. Prediction Performance
- Random Forest đạt độ chính xác gần hoàn hảo
- MAE chỉ 0.0014 eV/atom
- Có thể dự đoán energy cho cấu trúc mới với độ tin cậy cao

## 🔧 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly
```

## 💡 Use Cases

### 1. Material Discovery
- Xác định cấu trúc Carbon ổn định cho ứng dụng
- Screening nhanh các cấu trúc tiềm năng
- Tập trung nghiên cứu vào Highly Stable clusters

### 2. Property Prediction
- Dự đoán energy cho cấu trúc mới
- Ước lượng stability trước khi tổng hợp
- Tối ưu hóa thiết kế cấu trúc

### 3. Structure Analysis
- Hiểu mối quan hệ structure-property
- Phân loại và hệ thống hóa cấu trúc
- Phát hiện patterns và trends

## 📊 Visualizations

### Stability Analysis
![Stability Overview](carbon24_stability_analysis/figures/stability_analysis_overview.png)

### Energy Prediction
- Actual vs Predicted scatter plot
- Error distribution analysis
- Model performance comparison

### Feature Importance
- Top features for energy prediction
- Contribution analysis

### PCA Visualization
- Structures colored by stability
- Structures colored by energy

## 🎯 Next Steps

### Mở rộng thêm:

1. **Advanced ML Models**
   - Graph Neural Networks
   - Deep Learning architectures
   - Ensemble methods

2. **Additional Analysis**
   - Anomaly detection
   - Time series (if applicable)
   - Cross-validation studies

3. **Interactive Tools**
   - 3D structure visualization
   - Real-time prediction API
   - Web-based interface

4. **Optimization**
   - Hyperparameter tuning
   - Feature engineering
   - Model compression

## 📞 Support

Nếu gặp vấn đề:

1. Kiểm tra `ANALYSIS_REPORT.txt` cho kết quả chi tiết
2. Xem visualizations trong folder `figures/`
3. Đọc `HUONG_DAN_STABILITY_ANALYSIS.md` cho hướng dẫn đầy đủ

## 🏆 Achievements

✅ Phân loại thành công 10,153 cấu trúc Carbon  
✅ Xác định 3 nhóm stability rõ ràng  
✅ Đạt độ chính xác dự đoán R² = 1.0000  
✅ Tạo dashboard tương tác đầy đủ  
✅ Báo cáo PDF chuyên nghiệp  

## 📝 Citation

```
Carbon-24 Stability Analysis & Energy Prediction
Dataset: Carbon-24 Allotropes (10,153 structures)
Method: K-means Clustering + Random Forest Regression
Date: May 2026
```

---

**🎉 Chúc mừng! Bạn đã hoàn thành phân tích stability và energy prediction cho Carbon-24!**

Để bắt đầu, chạy:
```bash
python carbon24-stability-analysis.py
streamlit run carbon24_interactive_dashboard.py
```
