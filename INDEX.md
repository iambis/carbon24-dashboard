# 📚 Carbon-24 Project - Documentation Index

## 🎯 Bắt Đầu Nhanh

| File | Mô Tả | Thời Gian Đọc |
|------|-------|---------------|
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Hướng dẫn bắt đầu nhanh | 5 phút |
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | Tổng quan toàn bộ dự án | 10 phút |

## 📖 Documentation Chi Tiết

| File | Ngôn Ngữ | Nội Dung | Độ Dài |
|------|----------|----------|--------|
| **[README_STABILITY_ANALYSIS.md](README_STABILITY_ANALYSIS.md)** | English | Quick reference, key results | Medium |
| **[HUONG_DAN_STABILITY_ANALYSIS.md](HUONG_DAN_STABILITY_ANALYSIS.md)** | Tiếng Việt | Hướng dẫn chi tiết đầy đủ | Long |

## 🚀 Scripts & Tools

### Analysis Scripts

| Script | Mục Đích | Thời Gian Chạy |
|--------|----------|----------------|
| `demo_stability_analysis.py` | Demo nhanh kết quả | ~5s |
| `carbon24-stability-analysis.py` | Phân tích đầy đủ | ~20s |
| `generate_pdf_report.py` | Tạo báo cáo PDF | ~10s |

### Interactive Tools

| Tool | Mô Tả | Command |
|------|-------|---------|
| `carbon24_interactive_dashboard.py` | Dashboard tương tác | `streamlit run carbon24_interactive_dashboard.py` |

## 📊 Results & Outputs

### Generated Files

```
carbon24_stability_analysis/
├── 📊 Data Files
│   ├── cluster_stability_classification.csv
│   ├── prediction_model_comparison.csv
│   ├── best_model_predictions.csv
│   └── feature_importance.csv
│
├── 📄 Reports
│   ├── ANALYSIS_REPORT.txt
│   └── Carbon24_Stability_Report_YYYYMMDD.pdf
│
└── 📁 Visualizations
    ├── stability_analysis_overview.png
    ├── cluster_stability_details.png
    ├── feature_importance.png
    ├── prediction_error_analysis.png
    └── pca_stability_visualization.png
```

## 🎓 Learning Path

### Beginner
1. ✅ Đọc [GETTING_STARTED.md](GETTING_STARTED.md)
2. ✅ Chạy `python demo_stability_analysis.py`
3. ✅ Xem dashboard: `streamlit run carbon24_interactive_dashboard.py`

### Intermediate
1. ✅ Đọc [README_STABILITY_ANALYSIS.md](README_STABILITY_ANALYSIS.md)
2. ✅ Chạy full analysis: `python carbon24-stability-analysis.py`
3. ✅ Khám phá các CSV results

### Advanced
1. ✅ Đọc [HUONG_DAN_STABILITY_ANALYSIS.md](HUONG_DAN_STABILITY_ANALYSIS.md)
2. ✅ Customize scripts
3. ✅ Extend với ML models mới

## 📋 Quick Commands

### Essential Commands

```bash
# 1. Quick demo
python demo_stability_analysis.py

# 2. Full analysis
python carbon24-stability-analysis.py

# 3. Interactive dashboard
streamlit run carbon24_interactive_dashboard.py

# 4. Generate PDF report
python generate_pdf_report.py
```

### Check Results

```bash
# List generated files
ls carbon24_stability_analysis/

# View text report
cat carbon24_stability_analysis/ANALYSIS_REPORT.txt

# Open PDF (Windows)
start carbon24_stability_analysis/Carbon24_Stability_Report_*.pdf
```

## 🔍 Find Information

### Tìm Thông Tin Về...

| Topic | File | Section |
|-------|------|---------|
| **Kết quả clustering** | PROJECT_SUMMARY.md | Clustering Results |
| **Dự đoán energy** | README_STABILITY_ANALYSIS.md | Energy Prediction |
| **Feature importance** | HUONG_DAN_STABILITY_ANALYSIS.md | Feature Importance |
| **Cách sử dụng dashboard** | GETTING_STARTED.md | Dashboard Section |
| **Customization** | HUONG_DAN_STABILITY_ANALYSIS.md | Customization |
| **Troubleshooting** | GETTING_STARTED.md | Troubleshooting |

## 📊 Key Results Summary

### Stability Classification
- 🟢 **Highly Stable:** 55.5% (Clusters 0 & 1)
- 🔴 **Less Stable:** 44.5% (Cluster 2)

### Energy Prediction
- ⭐ **Best Model:** Random Forest
- 📈 **Test R²:** 1.0000
- 📉 **Test MAE:** 0.0014 eV/atom

### Top Features
1. num_atoms (99.99%)
2. bond_length_range
3. bond_complexity
4. volume_ratio
5. min_bond_length

## 🎯 Use Cases

| Use Case | Relevant Files | Tools |
|----------|---------------|-------|
| **Material Discovery** | cluster_stability_classification.csv | Dashboard → Cluster Explorer |
| **Energy Prediction** | best_model_predictions.csv | Dashboard → Energy Prediction |
| **Feature Analysis** | feature_importance.csv | Dashboard → Feature Importance |
| **Data Exploration** | All CSV files | Dashboard → Data Explorer |
| **Reporting** | PDF Report | generate_pdf_report.py |

## 🔧 Customization Guide

| What to Customize | File | Line/Section |
|-------------------|------|--------------|
| Number of clusters | carbon24-stability-analysis.py | `optimal_k = 3` |
| ML models | carbon24-stability-analysis.py | `models = {...}` |
| Stability thresholds | carbon24-stability-analysis.py | Stability classification |
| Dashboard pages | carbon24_interactive_dashboard.py | Page sections |
| PDF layout | generate_pdf_report.py | Page creation |

## 📞 Support & Help

### Quick Help

```bash
# Run demo for overview
python demo_stability_analysis.py

# Check if analysis completed
ls carbon24_stability_analysis/

# View analysis report
cat carbon24_stability_analysis/ANALYSIS_REPORT.txt
```

### Documentation

| Question | Answer In |
|----------|-----------|
| How to start? | GETTING_STARTED.md |
| What are the results? | PROJECT_SUMMARY.md |
| How to use dashboard? | README_STABILITY_ANALYSIS.md |
| Detailed explanation? | HUONG_DAN_STABILITY_ANALYSIS.md |
| Troubleshooting? | GETTING_STARTED.md → Troubleshooting |

## 🎓 Additional Resources

### Code Examples

```python
# Load results
import pandas as pd

# Stability classification
stability = pd.read_csv('carbon24_stability_analysis/cluster_stability_classification.csv')
print(stability)

# Predictions
predictions = pd.read_csv('carbon24_stability_analysis/best_model_predictions.csv')
print(predictions.describe())

# Feature importance
features = pd.read_csv('carbon24_stability_analysis/feature_importance.csv')
print(features.head(10))
```

### Visualization Examples

```python
import matplotlib.pyplot as plt
import pandas as pd

# Plot stability distribution
df = pd.read_csv('carbon24_feature_selected/carbon24_feature_selected_standard.csv')
df['relative_energy'].hist(bins=50)
plt.xlabel('Relative Energy (eV/atom)')
plt.ylabel('Frequency')
plt.title('Energy Distribution')
plt.show()
```

## 🏆 Project Checklist

### Setup
- [x] Install dependencies
- [x] Prepare data
- [x] Run preprocessing

### Analysis
- [x] Clustering analysis
- [x] Stability classification
- [x] Energy prediction
- [x] Feature importance

### Deliverables
- [x] Analysis scripts
- [x] Interactive dashboard
- [x] PDF report generator
- [x] Documentation (EN + VI)
- [x] Demo script

### Results
- [x] CSV data files
- [x] Visualization figures
- [x] Text reports
- [x] PDF report

## 🎉 Next Steps

1. **Explore Results**
   ```bash
   python demo_stability_analysis.py
   ```

2. **Interactive Analysis**
   ```bash
   streamlit run carbon24_interactive_dashboard.py
   ```

3. **Generate Report**
   ```bash
   python generate_pdf_report.py
   ```

4. **Read Documentation**
   - Start: GETTING_STARTED.md
   - Details: HUONG_DAN_STABILITY_ANALYSIS.md

---

## 📝 Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│  CARBON-24 STABILITY ANALYSIS - QUICK REFERENCE         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  📊 RESULTS                                              │
│  • Clusters: 3 (2 Stable, 1 Less Stable)                │
│  • Best Model: Random Forest (R² = 1.0000)              │
│  • Accuracy: MAE = 0.0014 eV/atom                       │
│                                                          │
│  🚀 COMMANDS                                             │
│  • Demo:      python demo_stability_analysis.py         │
│  • Analysis:  python carbon24-stability-analysis.py     │
│  • Dashboard: streamlit run carbon24_interactive_...    │
│  • Report:    python generate_pdf_report.py             │
│                                                          │
│  📁 OUTPUTS                                              │
│  • Location: carbon24_stability_analysis/               │
│  • Files: 4 CSVs + 5 PNGs + 1 TXT + 1 PDF              │
│                                                          │
│  📖 DOCS                                                 │
│  • Quick: GETTING_STARTED.md                            │
│  • Full:  HUONG_DAN_STABILITY_ANALYSIS.md              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

**🎯 Start Here:** [GETTING_STARTED.md](GETTING_STARTED.md)

**📚 Full Guide:** [HUONG_DAN_STABILITY_ANALYSIS.md](HUONG_DAN_STABILITY_ANALYSIS.md)

**📊 Summary:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
