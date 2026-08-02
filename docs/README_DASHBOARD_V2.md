# 🎉 Carbon-24 Dashboard v2.0

## 🚀 Chạy Dashboard

```bash
streamlit run carbon24_dashboard.py
```

## ✨ Tính Năng Mới

### 3 Trang Phân Cụm Mới:

1. **🎲 GMM (Gaussian Mixture Model)**
   - Uncertainty analysis
   - Probability-based clustering
   - AIC/BIC model selection

2. **🌳 Hierarchical Clustering**
   - Ward linkage method
   - Cluster interpretation
   - Energy analysis

3. **🔍 HDBSCAN (Density-Based)**
   - Automatic cluster detection
   - Noise/outlier analysis
   - Membership probability

## 📊 Tổng Quan Dashboard

| Trang | Tabs | Visualizations | Status |
|-------|------|----------------|--------|
| K-means | 3 | 10+ | ✅ |
| GMM | 4 | 12+ | ✅ NEW |
| Hierarchical | 3 | 8+ | ✅ NEW |
| HDBSCAN | 4 | 12+ | ✅ NEW |
| So sánh | 4 | 15+ | ✅ |

## 📁 Cấu Trúc Dữ Liệu

```
carbon24_kmeans_results/
carbon24_gmm_results/
carbon24_hierarchical_baseline/
hdbscan_phuc/
carbon24_clustering_comparison_results/
```

## 🎯 Tính Năng Chính

### Mỗi Trang Clustering Có:
- ✅ Cluster overview & distribution
- ✅ PCA 2D visualization
- ✅ Energy analysis
- ✅ Cluster profiles
- ✅ Interactive charts

### Trang So Sánh Có:
- 🏆 Ranking 4 thuật toán
- 📊 Multi-metric comparison
- 🎯 Radar charts
- 💡 Recommendations

## 💡 Quick Tips

1. **Sidebar**: Chọn trang
2. **Tabs**: Khám phá các phân tích khác nhau
3. **Hover**: Xem chi tiết trên charts
4. **Selectors**: Filter theo cluster
5. **Sliders**: Điều chỉnh thresholds

## 📝 Xem Thêm

- `DASHBOARD_UPDATE_GUIDE.md` - Hướng dẫn chi tiết
- `DASHBOARD_SUMMARY.md` - Tóm tắt tính năng

---

**Version:** 2.0 | **Status:** ✅ Ready
