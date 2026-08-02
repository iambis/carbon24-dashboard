# 🔬 Carbon-24 Clustering Methods Comparison

## 📋 Project Overview

This project provides a comprehensive comparison of 4 different clustering methods applied to the Carbon-24 dataset containing 10,153 carbon allotrope structures.

### 🎯 Objectives
- Compare clustering quality across 4 methods
- Identify the best performing method
- Analyze energy distributions by cluster
- Provide actionable recommendations

---

## 🚀 Quick Start

```bash
# 1. Check requirements
python test_notebook_ready.py

# 2. Run the notebook
jupyter notebook carbon24-clustering-comparison-evaluation.ipynb

# 3. In Jupyter: Cell → Run All
```

**Expected runtime:** ~2-3 minutes

---

## 📊 Methods Compared

| Method | Type | Clusters | Noise Detection |
|--------|------|----------|-----------------|
| **K-means** | Centroid-based | 3 | No |
| **GMM** | Probabilistic | 3 | No |
| **Hierarchical** | Agglomerative | 3 | No |
| **HDBSCAN** | Density-based | 3-5 | Yes (~5%) |

---

## 📈 Evaluation Metrics

### 1. Silhouette Score (↑ higher is better)
- **Range:** [-1, 1]
- **Measures:** Cluster separation quality
- **Good:** > 0.5

### 2. Davies-Bouldin Index (↓ lower is better)
- **Range:** [0, ∞)
- **Measures:** Cluster compactness and separation
- **Good:** < 1.0

### 3. Calinski-Harabasz Score (↑ higher is better)
- **Range:** [0, ∞)
- **Measures:** Ratio of between/within cluster variance
- **Good:** > 1000

---

## 📁 Project Structure

```
📂 Project Root
│
├── 📓 carbon24-clustering-comparison-evaluation.ipynb  # Main notebook (21 cells)
│
├── 📂 carbon24_kmeans_results/                         # K-means results
│   ├── carbon24_clustered.csv
│   └── clustering_report.json
│
├── 📂 carbon24_gmm_results/                            # GMM results
│   └── results/
│       └── carbon24_gmm_results.csv
│
├── 📂 carbon24_hierarchical_baseline/                  # Hierarchical results
│   └── results/
│       └── carbon24_hierarchical_results.csv
│
├── 📂 hdbscan_phuc/                                    # HDBSCAN results
│   ├── hdbscan_results.csv
│   └── hdbscan_cluster_profile.csv
│
├── 📂 carbon24_feature_selected/                       # Feature data
│   ├── carbon24_feature_selected_standard.csv
│   └── selected_features.json
│
├── 📂 carbon24_clustering_comparison_results/          # Output (created by notebook)
│   ├── methods_overview.csv
│   ├── quality_metrics.csv
│   └── method_ranking.csv
│
├── 📄 HUONG_DAN_CLUSTERING_COMPARISON.md               # Vietnamese guide
├── 📄 CLUSTERING_COMPARISON_SUMMARY.md                 # English summary
├── 📄 TOM_TAT_DU_AN.md                                 # Vietnamese summary
├── 📄 README_CLUSTERING_COMPARISON.md                  # This file
│
└── 🔧 Scripts
    ├── test_notebook_ready.py                          # Check requirements
    ├── fix_hdbscan_column.py                           # Fix column names
    ├── create_clustering_comparison_nb.py              # Notebook generator
    ├── add_metrics_comparison.py                       # Add metrics section
    └── add_final_sections.py                           # Add final sections
```

---

## 📊 Notebook Structure (21 Cells)

### Part 1: Setup & Loading (3 cells)
1. Import libraries
2. Load clustering results from 4 methods
3. Create overview comparison table

### Part 2: Distribution Analysis (2 cells)
4. Analyze cluster distribution statistics
5. Visualize distributions (4 bar charts)

### Part 3: Quality Metrics (3 cells)
6. Load feature data
7. Calculate 3 quality metrics
8. Visualize metrics comparison (3 bar charts)

### Part 4: Method Ranking (2 cells)
9. Rank methods by each metric
10. Calculate overall ranking

### Part 5: Energy Analysis (2 cells)
11. Analyze energy distribution by cluster
12. Visualize energy distributions (4 histograms)

### Part 6: Summary & Recommendations (2 cells)
13. Method characteristics summary
14. Generate recommendations

### Part 7: Export Results (1 cell)
15. Export all results to CSV files

---

## 📊 Expected Results

### Typical Metrics Range

| Method | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|--------|------------|----------------|-------------------|
| K-means | 0.20-0.30 | 1.4-1.6 | 2000-2500 |
| GMM | 0.18-0.28 | 1.5-1.7 | 1800-2300 |
| Hierarchical | 0.19-0.29 | 1.45-1.65 | 1900-2400 |
| HDBSCAN | 0.15-0.25 | 1.6-1.9 | 1500-2000 |

### Ranking System

The overall ranking uses **rank-based scoring**:
- Each method gets points based on its rank in each metric
- Best rank = n points, worst rank = 1 point
- Total score = sum across all 3 metrics
- Maximum score = n × 3 (e.g., 4 methods × 3 metrics = 12 points)

---

## 🎓 Method Recommendations

### 🥇 K-means
**Best for:**
- Large datasets (fast)
- Spherical clusters
- Known number of clusters
- Production systems

**Metrics:** Usually best Silhouette and Calinski-Harabasz

### 🎲 GMM
**Best for:**
- Elliptical clusters
- Soft clustering (probabilities)
- Overlapping clusters
- Uncertainty quantification

**Metrics:** Good balance across all metrics

### 🌳 Hierarchical
**Best for:**
- Exploratory analysis
- Dendrogram visualization
- Nested structures
- Unknown k

**Metrics:** Similar to K-means, slightly slower

### 🔍 HDBSCAN
**Best for:**
- Varying density clusters
- Automatic k selection
- Noise detection
- Unknown structure

**Metrics:** Lower scores due to noise points, but better for outlier detection

---

## 🔧 Customization

### Change Sample Size
```python
# In Cell 7 (compute-metrics)
# Current: 5000 samples
sample_idx = np.random.choice(len(X_filtered), 5000, replace=False)

# Change to 10000 for more accuracy
sample_idx = np.random.choice(len(X_filtered), 10000, replace=False)
```

### Add More Metrics
```python
# In Cell 7
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(true_labels, predicted_labels)
nmi = normalized_mutual_info_score(true_labels, predicted_labels)
```

### Change Visualization Style
```python
# In Cell 1
plt.style.use('ggplot')  # Instead of 'seaborn-v0_8-darkgrid'
sns.set_palette('Set2')  # Instead of 'husl'
```

---

## 🔍 Troubleshooting

### Issue: File Not Found
**Solution:**
1. Run `python test_notebook_ready.py` to check requirements
2. Ensure all clustering notebooks have been run
3. Verify folder structure matches expected layout

### Issue: Memory Error
**Solution:**
```python
# Reduce sample size in Cell 7
sample_idx = np.random.choice(len(X_filtered), 1000, replace=False)
```

### Issue: No Metrics Calculated
**Solution:**
1. Check clustering results have 'cluster' column
2. Verify at least 2 clusters exist
3. For HDBSCAN, check non-noise points exist

### Issue: HDBSCAN Column Error
**Solution:**
✅ Already fixed! Notebook automatically renames `hdbscan_cluster` to `cluster`

---

## 📚 Documentation

| File | Language | Description |
|------|----------|-------------|
| `HUONG_DAN_CLUSTERING_COMPARISON.md` | Vietnamese | Detailed usage guide |
| `CLUSTERING_COMPARISON_SUMMARY.md` | English | Complete summary |
| `TOM_TAT_DU_AN.md` | Vietnamese | Project summary |
| `README_CLUSTERING_COMPARISON.md` | English | This file |

---

## 📊 Output Files

After running the notebook, the following files will be created:

```
📂 carbon24_clustering_comparison_results/
├── methods_overview.csv          # Overview of all methods
├── quality_metrics.csv            # Quality metrics for each method
└── method_ranking.csv             # Overall ranking
```

---

## 🎉 Next Steps

### After Running the Notebook:

1. **Review Results**
   - Check the overall winner
   - Compare metrics across methods
   - Analyze energy distributions

2. **Choose Best Method**
   - Based on metrics
   - Based on your use case
   - Based on computational constraints

3. **Use for Further Analysis**
   - Stability analysis
   - Energy prediction
   - Material discovery
   - Property prediction

4. **Document Findings**
   - Save comparison results
   - Note best method and why
   - Plan next experiments

---

## 📊 Export Options

### Export to HTML
```bash
jupyter nbconvert --to html carbon24-clustering-comparison-evaluation.ipynb
```

### Export to PDF
```bash
jupyter nbconvert --to pdf carbon24-clustering-comparison-evaluation.ipynb
```

### Share Results
- CSV files in `carbon24_clustering_comparison_results/`
- Screenshots of visualizations
- Summary tables

---

## ✅ Requirements Checklist

- [x] All 4 clustering methods completed
- [x] Result folders exist with correct structure
- [x] Feature selection completed
- [ ] Jupyter notebook installed
- [ ] Required libraries installed:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

### Quick Check
```bash
python test_notebook_ready.py
```

---

## 🚀 Command Reference

```bash
# Check requirements
python test_notebook_ready.py

# Start Jupyter
jupyter notebook

# Run specific notebook
jupyter notebook carbon24-clustering-comparison-evaluation.ipynb

# Export to HTML
jupyter nbconvert --to html carbon24-clustering-comparison-evaluation.ipynb

# Export to PDF
jupyter nbconvert --to pdf carbon24-clustering-comparison-evaluation.ipynb
```

---

## 📞 Support

For issues or questions:

1. Check the **Troubleshooting** section
2. Review detailed documentation files
3. Verify all input files exist
4. Check library installations

---

## 🎯 Summary

✅ **Status:** Ready to use  
✅ **Notebook:** 21 cells, fully functional  
✅ **Methods:** 4 clustering algorithms compared  
✅ **Metrics:** 3 quality metrics calculated  
✅ **Visualizations:** 11 charts total  
✅ **Output:** 3 CSV files with results  
✅ **Documentation:** Complete in English and Vietnamese  

**To get started:**
```bash
jupyter notebook carbon24-clustering-comparison-evaluation.ipynb
```

**Then:** Cell → Run All → Wait ~2-3 minutes → Review results!

---

## 📝 Version History

- **v1.0** (2026-05-21): Initial release
  - 21-cell notebook
  - 4 methods comparison
  - 3 quality metrics
  - Complete documentation
  - Fixed HDBSCAN column naming

---

## 📄 License

This project is part of the Carbon-24 materials discovery research.

---

## 🙏 Acknowledgments

- Carbon-24 dataset providers
- Scikit-learn for clustering algorithms
- Jupyter for interactive computing

---

**Happy Clustering! 🚀**

*For Vietnamese documentation, see `TOM_TAT_DU_AN.md` or `HUONG_DAN_CLUSTERING_COMPARISON.md`*
