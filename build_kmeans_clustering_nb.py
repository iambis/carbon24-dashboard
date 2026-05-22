"""
Script để tạo Jupyter Notebook cho K-means Clustering
Sử dụng dữ liệu đã preprocessing sẵn
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ============================================================================
# TITLE
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""# Carbon-24 K-means Clustering Analysis

**Dự án:** Phân cụm cấu trúc Carbon-24  
**Phương pháp:** K-means Clustering  
**Dữ liệu:** Sử dụng dữ liệu đã preprocessing

## Nội dung:
1. Load dữ liệu đã preprocessing
2. Khảo sát tổng quan
3. Xác định số cluster tối ưu (Elbow, Silhouette)
4. Thực hiện K-means clustering
5. Đánh giá kết quả
6. Phân tích clusters
7. Visualization (PCA, t-SNE)
8. Lưu kết quả
"""))

# ============================================================================
# IMPORTS
# ============================================================================
cells.append(nbf.v4.new_code_cell("""# Import libraries
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples
)

import warnings
warnings.filterwarnings('ignore')

# Settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("✓ Libraries imported successfully!")
"""))

# ============================================================================
# LOAD DATA
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 1. Load Dữ Liệu Đã Preprocessing

Sử dụng dữ liệu đã được làm sạch và loại bỏ multicollinearity.
"""))

cells.append(nbf.v4.new_code_cell("""# Load preprocessed data
data_path = 'carbon24_preprocessing_results/carbon24_feature_selected_standard.csv'
df = pd.read_csv(data_path)

print(f"✓ Loaded: {df.shape}")
print(f"✓ Samples: {len(df)}")
print(f"✓ Features: {len(df.columns)}")

df.head()
"""))

cells.append(nbf.v4.new_code_cell("""# Load feature list
with open('carbon24_preprocessing_results/selected_features.json', 'r') as f:
    feature_info = json.load(f)

numeric_features = feature_info['numeric_features']
categorical_features = feature_info['categorical_features']

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"\\nNumeric features for clustering:")
for feat in numeric_features:
    print(f"  - {feat}")
"""))

# ============================================================================
# DATA OVERVIEW
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Khảo Sát Tổng Quan"""))

cells.append(nbf.v4.new_code_cell("""# Dataset info
print("Dataset Information:")
print("="*80)
df.info()
"""))

cells.append(nbf.v4.new_code_cell("""# Descriptive statistics
df[numeric_features].describe()
"""))

cells.append(nbf.v4.new_code_cell("""# Check for any remaining issues
print("Missing values:", df[numeric_features].isnull().sum().sum())
print("Infinite values:", np.isinf(df[numeric_features]).sum().sum())
print("\\nData ready for clustering!" if df[numeric_features].isnull().sum().sum() == 0 else "⚠️ Data needs cleaning")
"""))

cells.append(nbf.v4.new_code_cell("""# Crystal system distribution
if 'crystal_system' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    crystal_counts = df['crystal_system'].value_counts()
    
    # Bar plot
    crystal_counts.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title('Crystal System Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Crystal System')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # Pie chart
    axes[1].pie(crystal_counts, labels=crystal_counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette('husl', len(crystal_counts)))
    axes[1].set_title('Crystal System Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
"""))

# ============================================================================
# PREPARE DATA
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Chuẩn Bị Dữ Liệu Cho Clustering"""))

cells.append(nbf.v4.new_code_cell("""# Select features for clustering
X = df[numeric_features].copy()

print(f"Clustering data shape: {X.shape}")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

# Verify data is normalized (should be around mean=0, std=1)
print(f"\\nData statistics:")
print(f"Mean: {X.mean().mean():.4f}")
print(f"Std: {X.std().mean():.4f}")
"""))

# ============================================================================
# ELBOW METHOD
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Xác Định Số Cluster Tối Ưu

### 4.1 Elbow Method
"""))

cells.append(nbf.v4.new_code_cell("""# Elbow method
k_range = range(2, 21)
inertias = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

print("Testing different k values...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(X, kmeans.labels_))
    calinski_harabasz_scores.append(calinski_harabasz_score(X, kmeans.labels_))
    
    if k % 5 == 0:
        print(f"  k={k}: Silhouette={silhouette_scores[-1]:.4f}")

print("✓ Completed!")
"""))

cells.append(nbf.v4.new_code_cell("""# Plot elbow curve
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Inertia (Elbow)
axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0, 0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
axes[0, 0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Silhouette Score (higher is better)
axes[0, 1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
axes[0, 1].set_title('Silhouette Score by k (Higher is Better)', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Davies-Bouldin Index (lower is better)
axes[1, 0].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1, 0].set_ylabel('Davies-Bouldin Index', fontsize=12)
axes[1, 0].set_title('Davies-Bouldin Index by k (Lower is Better)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Calinski-Harabasz Index (higher is better)
axes[1, 1].plot(k_range, calinski_harabasz_scores, 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1, 1].set_ylabel('Calinski-Harabasz Index', fontsize=12)
axes[1, 1].set_title('Calinski-Harabasz Index by k (Higher is Better)', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Find optimal k
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
best_k_davies = k_range[np.argmin(davies_bouldin_scores)]
best_k_calinski = k_range[np.argmax(calinski_harabasz_scores)]

print("Optimal k by different metrics:")
print(f"  - Silhouette Score: k = {best_k_silhouette} (score: {max(silhouette_scores):.4f})")
print(f"  - Davies-Bouldin: k = {best_k_davies} (score: {min(davies_bouldin_scores):.4f})")
print(f"  - Calinski-Harabasz: k = {best_k_calinski} (score: {max(calinski_harabasz_scores):.2f})")

# Recommend k
recommended_k = best_k_silhouette
print(f"\\n✓ Recommended k: {recommended_k}")
"""))

# ============================================================================
# KMEANS CLUSTERING
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Thực Hiện K-means Clustering"""))

cells.append(nbf.v4.new_code_cell("""# Perform K-means with optimal k
optimal_k = recommended_k  # You can change this value

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = kmeans_final.fit_predict(X)

# Add cluster labels to dataframe
df['cluster'] = cluster_labels

print(f"✓ K-means clustering completed with k={optimal_k}")
print(f"\\nCluster distribution:")
print(df['cluster'].value_counts().sort_index())
"""))

cells.append(nbf.v4.new_code_cell("""# Clustering metrics
silhouette_avg = silhouette_score(X, cluster_labels)
davies_bouldin = davies_bouldin_score(X, cluster_labels)
calinski_harabasz = calinski_harabasz_score(X, cluster_labels)

print("Clustering Quality Metrics:")
print("="*60)
print(f"Silhouette Score: {silhouette_avg:.4f} (range: [-1, 1], higher is better)")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f} (higher is better)")
print(f"Inertia: {kmeans_final.inertia_:.2f}")
"""))

# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Phân Tích Clusters"""))

cells.append(nbf.v4.new_code_cell("""# Cluster statistics
cluster_stats = df.groupby('cluster')[numeric_features].mean()

print("Cluster Centers (mean values):")
display(cluster_stats.round(4))
"""))

cells.append(nbf.v4.new_code_cell("""# Analyze key features by cluster
key_features_analysis = ['relative_energy', 'num_atoms', 'angle_std']

fig, axes = plt.subplots(1, len(key_features_analysis), figsize=(18, 5))

for idx, feature in enumerate(key_features_analysis):
    if feature in df.columns:
        df.boxplot(column=feature, by='cluster', ax=axes[idx])
        axes[idx].set_title(f'{feature} by Cluster')
        axes[idx].set_xlabel('Cluster')
        axes[idx].set_ylabel(feature)
        axes[idx].grid(True, alpha=0.3)

plt.suptitle('')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Crystal system distribution by cluster
if 'crystal_system' in df.columns:
    crystal_cluster = pd.crosstab(df['cluster'], df['crystal_system'])
    
    print("Crystal System Distribution by Cluster:")
    display(crystal_cluster)
    
    # Visualization
    crystal_cluster.plot(kind='bar', stacked=True, figsize=(12, 6), 
                         colormap='tab10', edgecolor='black')
    plt.title('Crystal System Distribution by Cluster', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Crystal System', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
"""))

# ============================================================================
# PCA VISUALIZATION
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 7. Visualization

### 7.1 PCA Visualization
"""))

cells.append(nbf.v4.new_code_cell("""# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

print(f"PCA explained variance ratio:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"  PC2: {pca.explained_variance_ratio_[0]:.4f}")
print(f"  Total: {pca.explained_variance_ratio_.sum():.4f}")

# Add PCA components to dataframe
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]
"""))

cells.append(nbf.v4.new_code_cell("""# Plot PCA with clusters
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# By cluster
scatter1 = axes[0].scatter(df['pca1'], df['pca2'], c=df['cluster'], 
                           cmap='tab10', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
axes[0].set_title('PCA: Clusters', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# By relative energy
if 'relative_energy' in df.columns:
    scatter2 = axes[1].scatter(df['pca1'], df['pca2'], c=df['relative_energy'],
                               cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    axes[1].set_title('PCA: Relative Energy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Relative Energy')

plt.tight_layout()
plt.show()
"""))

# ============================================================================
# TSNE VISUALIZATION
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""### 7.2 t-SNE Visualization"""))

cells.append(nbf.v4.new_code_cell("""# t-SNE (may take a few minutes)
print("Computing t-SNE... (this may take a while)")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

df['tsne1'] = X_tsne[:, 0]
df['tsne2'] = X_tsne[:, 1]

print("✓ t-SNE completed!")
"""))

cells.append(nbf.v4.new_code_cell("""# Plot t-SNE
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# By cluster
scatter1 = axes[0].scatter(df['tsne1'], df['tsne2'], c=df['cluster'],
                           cmap='tab10', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('t-SNE 1', fontsize=12)
axes[0].set_ylabel('t-SNE 2', fontsize=12)
axes[0].set_title('t-SNE: Clusters', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# By relative energy
if 'relative_energy' in df.columns:
    scatter2 = axes[1].scatter(df['tsne1'], df['tsne2'], c=df['relative_energy'],
                               cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('t-SNE 1', fontsize=12)
    axes[1].set_ylabel('t-SNE 2', fontsize=12)
    axes[1].set_title('t-SNE: Relative Energy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Relative Energy')

plt.tight_layout()
plt.show()
"""))

# ============================================================================
# SILHOUETTE ANALYSIS
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""### 7.3 Silhouette Analysis"""))

cells.append(nbf.v4.new_code_cell("""# Silhouette analysis
silhouette_vals = silhouette_samples(X, cluster_labels)

fig, ax = plt.subplots(figsize=(12, 8))

y_lower = 10
for i in range(optimal_k):
    cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = plt.cm.tab10(i / optimal_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.set_xlabel('Silhouette Coefficient', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)
ax.set_title(f'Silhouette Plot (k={optimal_k}, avg={silhouette_avg:.4f})', 
             fontsize=14, fontweight='bold')
ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2, label=f'Average: {silhouette_avg:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""))

# ============================================================================
# SAVE RESULTS
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 8. Lưu Kết Quả"""))

cells.append(nbf.v4.new_code_cell("""# Create output directory
output_dir = 'carbon24_kmeans_results'
os.makedirs(output_dir, exist_ok=True)

# Save clustered data
df.to_csv(f'{output_dir}/carbon24_clustered.csv', index=False)
print(f"✓ Saved: {output_dir}/carbon24_clustered.csv")

# Save cluster centers
cluster_centers_df = pd.DataFrame(kmeans_final.cluster_centers_, columns=numeric_features)
cluster_centers_df.to_csv(f'{output_dir}/cluster_centers.csv', index=False)
print(f"✓ Saved: {output_dir}/cluster_centers.csv")
"""))

cells.append(nbf.v4.new_code_cell("""# Save clustering report
report = {
    'optimal_k': int(optimal_k),
    'n_samples': len(df),
    'n_features': len(numeric_features),
    'metrics': {
        'silhouette_score': float(silhouette_avg),
        'davies_bouldin_index': float(davies_bouldin),
        'calinski_harabasz_index': float(calinski_harabasz),
        'inertia': float(kmeans_final.inertia_)
    },
    'cluster_sizes': df['cluster'].value_counts().sort_index().to_dict(),
    'pca_variance_explained': {
        'pc1': float(pca.explained_variance_ratio_[0]),
        'pc2': float(pca.explained_variance_ratio_[1]),
        'total': float(pca.explained_variance_ratio_.sum())
    }
}

with open(f'{output_dir}/clustering_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"✓ Saved: {output_dir}/clustering_report.json")
"""))

# ============================================================================
# SUMMARY
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 📊 Summary

### Clustering Results:
- **Optimal k:** {optimal_k}
- **Silhouette Score:** {silhouette_avg:.4f}
- **Davies-Bouldin Index:** {davies_bouldin:.4f}
- **Calinski-Harabasz Index:** {calinski_harabasz:.2f}

### Output Files:
- `carbon24_kmeans_results/carbon24_clustered.csv` - Data with cluster labels
- `carbon24_kmeans_results/cluster_centers.csv` - Cluster centers
- `carbon24_kmeans_results/clustering_report.json` - Detailed report

### Next Steps:
- 🔍 Anomaly Detection (Isolation Forest, LOF)
- 📈 Energy Prediction (Regression models)
- 🎨 Advanced Visualization (UMAP, 3D plots)
- 📊 Cluster Interpretation & Domain Analysis
"""))

# Add cells to notebook
nb['cells'] = cells

# Save notebook
with open('carbon24-kmeans-clustering.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("✓ Notebook created successfully: carbon24-kmeans-clustering.ipynb")
