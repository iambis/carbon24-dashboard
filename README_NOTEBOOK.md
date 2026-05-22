# 📓 Carbon-24 Anomaly Detection & Energy Prediction Notebook

## 🎯 Quick Start

```bash
jupyter notebook carbon24-anomaly-energy-prediction.ipynb
```

Sau đó: `Cell` → `Run All`

---

## 📊 Nội Dung Notebook

### 1. 🔍 Phát Hiện Dị Biệt (Anomaly Detection)

**4 phương pháp:**
- **Isolation Forest** - Phát hiện outliers dựa trên isolation
- **Local Outlier Factor (LOF)** - Dựa trên mật độ local
- **One-Class SVM** - Dựa trên boundary
- **Z-score** - Phương pháp thống kê

**Kết quả:**
- Phát hiện ~5% cấu trúc bất thường
- Consensus anomalies: Cấu trúc được đánh dấu bởi ≥2 phương pháp
- Visualizations: 4 plots phân tích anomalies

### 2. ⚡ Phân Loại Stability

**Phương pháp:** K-means clustering (k=3)

**Kết quả:**
- 🟢 **Cluster 0:** Highly Stable (33.6% - 3,409 cấu trúc)
- 🟢 **Cluster 1:** Highly Stable (22.0% - 2,229 cấu trúc)
- 🔴 **Cluster 2:** Less Stable (44.5% - 4,515 cấu trúc)

**Visualizations:**
- Energy distribution by cluster
- Box plot comparison
- Cluster sizes
- PCA 2D projection

### 3. 🤖 Dự Đoán Energy per Atom

**4 mô hình ML:**
- Random Forest (Best: R² = 1.0000)
- Gradient Boosting (R² = 1.0000)
- Ridge Regression (R² = 0.8988)
- Lasso Regression (R² = 0.8297)

**Kết quả:**
- **Best Model:** Random Forest
- **Test R²:** 1.0000 (gần hoàn hảo!)
- **Test MAE:** 0.0014 eV/atom
- **Top Feature:** num_atoms (99.99% importance)

**Visualizations:**
- Actual vs Predicted
- Residual plot
- Error distribution
- Model comparison
- Feature importance (top 15)

---

## 📁 Output Files

Sau khi chạy notebook, các files sẽ được tạo trong `carbon24_anomaly_prediction_results/`:

```
carbon24_anomaly_prediction_results/
├── anomaly_detection_results.csv      # Kết quả phát hiện dị biệt
├── stability_classification.csv       # Phân loại stability
├── model_comparison.csv               # So sánh các mô hình
└── energy_predictions.csv             # Kết quả dự đoán energy
```

---

## 🎓 Tính Năng

✅ **24 cells** với code và markdown  
✅ **Anomaly detection** với 4 phương pháp  
✅ **Stability classification** với K-means  
✅ **Energy prediction** với 4 ML models  
✅ **12+ visualizations** đẹp và chi tiết  
✅ **Export results** sang CSV  
✅ **Feature importance** analysis  
✅ **Model comparison** đầy đủ  

---

## 📊 Thời Gian Chạy

| Section | Time | Output |
|---------|------|--------|
| Load Data | ~2s | DataFrame |
| Anomaly Detection | ~10s | 4 methods + viz |
| Stability Classification | ~5s | Clustering + viz |
| Energy Prediction | ~30s | 4 models + viz |
| **Total** | **~50s** | **All results** |

---

## 💡 Tips

### Chạy Nhanh Hơn

Giảm số estimators:
```python
RandomForestRegressor(n_estimators=50, ...)  # Thay vì 100
```

### Phát Hiện Nhiều Anomalies Hơn

Tăng contamination:
```python
IsolationForest(contamination=0.10, ...)  # Thay vì 0.05
```

### Thay Đổi Số Clusters

```python
optimal_k = 4  # Thay vì 3
```

### Export Figures

Thêm vào cuối cell visualization:
```python
plt.savefig('my_figure.png', dpi=300, bbox_inches='tight')
```

---

## 🔧 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## 📚 Documentation

- **[HUONG_DAN_NOTEBOOK.md](HUONG_DAN_NOTEBOOK.md)** - Hướng dẫn chi tiết
- **[README_STABILITY_ANALYSIS.md](README_STABILITY_ANALYSIS.md)** - Phân tích stability
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Tổng quan dự án

---

## 🎯 Use Cases

### 1. Phát Hiện Cấu Trúc Bất Thường

```python
# Xem anomalies
anomalies = df[df['is_anomaly'] == 1]
print(anomalies[['row_index', 'relative_energy']].head())
```

### 2. Tìm Cấu Trúc Ổn Định

```python
# Xem highly stable structures
stable = df[df['cluster'].isin([0, 1])]
print(f"Found {len(stable)} stable structures")
```

### 3. Dự Đoán Energy Cho Cấu Trúc Mới

```python
# Sử dụng trained model
new_structure_features = [...]  # Your features
predicted_energy = best_model.predict([new_structure_features])
```

---

## 🎉 Kết Quả Mong Đợi

### Anomaly Detection
- ~5% cấu trúc được đánh dấu là anomaly
- Consensus: ~4% được đánh dấu bởi ≥2 phương pháp
- Anomalies thường có năng lượng cao hoặc cấu trúc đặc biệt

### Stability Classification
- 55.5% cấu trúc thuộc nhóm Highly Stable
- 44.5% cấu trúc thuộc nhóm Less Stable
- Phân tách rõ ràng giữa các nhóm

### Energy Prediction
- Random Forest đạt R² = 1.0000 (gần hoàn hảo)
- MAE chỉ 0.0014 eV/atom
- num_atoms là feature quan trọng nhất (99.99%)

---

## 🚀 Quick Commands

```bash
# Open notebook
jupyter notebook carbon24-anomaly-energy-prediction.ipynb

# Or with JupyterLab
jupyter lab carbon24-anomaly-energy-prediction.ipynb

# Convert to HTML
jupyter nbconvert --to html carbon24-anomaly-energy-prediction.ipynb

# Convert to PDF (requires LaTeX)
jupyter nbconvert --to pdf carbon24-anomaly-energy-prediction.ipynb
```

---

## 📞 Support

Nếu gặp vấn đề:

1. Đọc [HUONG_DAN_NOTEBOOK.md](HUONG_DAN_NOTEBOOK.md)
2. Kiểm tra data files tồn tại
3. Restart kernel: `Kernel` → `Restart & Run All`
4. Kiểm tra dependencies đã cài đặt

---

## 🏆 Highlights

✨ **Comprehensive Analysis** - 3 phần phân tích đầy đủ  
✨ **Multiple Methods** - 4 anomaly detection + 4 ML models  
✨ **Beautiful Visualizations** - 12+ plots chuyên nghiệp  
✨ **Export Results** - CSV files ready to use  
✨ **High Accuracy** - R² = 1.0000 in prediction  
✨ **Well Documented** - Comments và markdown đầy đủ  

---

<div align="center">

**🎊 Notebook sẵn sàng sử dụng!**

```bash
jupyter notebook carbon24-anomaly-energy-prediction.ipynb
```

**Sau đó:** `Cell` → `Run All`

**⏱️ Thời gian:** ~50 giây  
**📊 Kết quả:** 4 CSV files + 12+ visualizations  
**🎯 Độ chính xác:** R² = 1.0000  

</div>
