"""
Carbon-24 Clustering Methods Comparison
Compare K-means, GMM, Hierarchical, and HDBSCAN
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

import warnings
warnings.filterwarnings('ignore')

# Settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("🔬 CARBON-24 CLUSTERING METHODS COMPARISON")
print("=" * 80)
print("\nComparing 4 methods:")
print("  1. K-means")
print("  2. Gaussian Mixture Model (GMM)")
print("  3. Hierarchical Clustering (Agglomerative)")
print("  4. HDBSCAN")
print()

# Load data
print("📂 Loading data...")
data_path = 'carbon24_preprocessing_results/carbon24_feature_selected_standard.csv'
df = pd.read_csv(data_path)

with open('carbon24_preprocessing_results/selected_features.json', 'r') as f:
    feature_info = json.load(f)

numeric_features = feature_info['numeric_features']
X = df[numeric_features].values

print(f"✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
print()

# Create output directory
os.makedirs('carbon24_clustering_comparison', exist_ok=True)
os.makedirs('carbon24_clustering_comparison/figures', exist_ok=True)

# Number of clusters to test
n_clusters = 3  # Based on Elbow method

print(f"🎯 Testing with k={n_clusters} clusters")
print("=" * 80)
print()

# Results storage
results = {}

# ============================================================================
# 1. K-MEANS
# ============================================================================
print("1️⃣  K-MEANS CLUSTERING")
print("-" * 80)

start_time = time()
kmeans = KMeans(
    n_clusters=n_clusters,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans_labels = kmeans.fit_predict(X)
kmeans_time = time() - start_time

# Sample for Silhouette (memory efficient)
sample_size = min(5000, len(X))
sample_indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[sample_indices]
labels_sample = kmeans_labels[sample_indices]

kmeans_silhouette = silhouette_score(X_sample, labels_sample)
kmeans_davies = davies_bouldin_score(X, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X, kmeans_labels)

results['K-means'] = {
    'labels': kmeans_labels,
    'silhouette': kmeans_silhouette,
    'davies_bouldin': kmeans_davies,
    'calinski_harabasz': kmeans_calinski,
    'time': kmeans_time,
    'n_clusters': n_clusters
}

print(f"  Silhouette Score:      {kmeans_silhouette:.4f}")
print(f"  Davies-Bouldin Index:  {kmeans_davies:.4f}")
print(f"  Calinski-Harabasz:     {kmeans_calinski:.2f}")
print(f"  Time:                  {kmeans_time:.2f}s")
print(f"  ✅ Completed")
print()

# ============================================================================
# 2. GAUSSIAN MIXTURE MODEL (GMM)
# ============================================================================
print("2️⃣  GAUSSIAN MIXTURE MODEL (GMM)")
print("-" * 80)

start_time = time()
gmm = GaussianMixture(
    n_components=n_clusters,
    covariance_type='full',
    n_init=10,
    max_iter=300,
    random_state=42
)
gmm_labels = gmm.fit_predict(X)
gmm_time = time() - start_time

labels_sample = gmm_labels[sample_indices]

gmm_silhouette = silhouette_score(X_sample, labels_sample)
gmm_davies = davies_bouldin_score(X, gmm_labels)
gmm_calinski = calinski_harabasz_score(X, gmm_labels)

results['GMM'] = {
    'labels': gmm_labels,
    'silhouette': gmm_silhouette,
    'davies_bouldin': gmm_davies,
    'calinski_harabasz': gmm_calinski,
    'time': gmm_time,
    'n_clusters': n_clusters,
    'bic': gmm.bic(X),
    'aic': gmm.aic(X)
}

print(f"  Silhouette Score:      {gmm_silhouette:.4f}")
print(f"  Davies-Bouldin Index:  {gmm_davies:.4f}")
print(f"  Calinski-Harabasz:     {gmm_calinski:.2f}")
print(f"  BIC:                   {gmm.bic(X):.2f}")
print(f"  AIC:                   {gmm.aic(X):.2f}")
print(f"  Time:                  {gmm_time:.2f}s")
print(f"  ✅ Completed")
print()

# ============================================================================
# 3. HIERARCHICAL CLUSTERING (AGGLOMERATIVE)
# ============================================================================
print("3️⃣  HIERARCHICAL CLUSTERING (Agglomerative)")
print("-" * 80)

start_time = time()
hierarchical = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward'
)
hierarchical_labels = hierarchical.fit_predict(X)
hierarchical_time = time() - start_time

labels_sample = hierarchical_labels[sample_indices]

hierarchical_silhouette = silhouette_score(X_sample, labels_sample)
hierarchical_davies = davies_bouldin_score(X, hierarchical_labels)
hierarchical_calinski = calinski_harabasz_score(X, hierarchical_labels)

results['Hierarchical'] = {
    'labels': hierarchical_labels,
    'silhouette': hierarchical_silhouette,
    'davies_bouldin': hierarchical_davies,
    'calinski_harabasz': hierarchical_calinski,
    'time': hierarchical_time,
    'n_clusters': n_clusters
}

print(f"  Silhouette Score:      {hierarchical_silhouette:.4f}")
print(f"  Davies-Bouldin Index:  {hierarchical_davies:.4f}")
print(f"  Calinski-Harabasz:     {hierarchical_calinski:.2f}")
print(f"  Time:                  {hierarchical_time:.2f}s")
print(f"  ✅ Completed")
print()

# ============================================================================
# 4. HDBSCAN (if available)
# ============================================================================
print("4️⃣  HDBSCAN")
print("-" * 80)

try:
    import hdbscan
    
    start_time = time()
    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        metric='euclidean'
    )
    hdbscan_labels = hdbscan_clusterer.fit_predict(X)
    hdbscan_time = time() - start_time
    
    # Check if clustering was successful
    n_clusters_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    n_noise = list(hdbscan_labels).count(-1)
    
    if n_clusters_hdbscan > 1:
        # Filter out noise points for metrics
        mask = hdbscan_labels != -1
        X_no_noise = X[mask]
        labels_no_noise = hdbscan_labels[mask]
        
        # Sample for Silhouette
        sample_size_hdb = min(5000, len(X_no_noise))
        sample_indices_hdb = np.random.choice(len(X_no_noise), sample_size_hdb, replace=False)
        
        hdbscan_silhouette = silhouette_score(X_no_noise[sample_indices_hdb], labels_no_noise[sample_indices_hdb])
        hdbscan_davies = davies_bouldin_score(X_no_noise, labels_no_noise)
        hdbscan_calinski = calinski_harabasz_score(X_no_noise, labels_no_noise)
        
        results['HDBSCAN'] = {
            'labels': hdbscan_labels,
            'silhouette': hdbscan_silhouette,
            'davies_bouldin': hdbscan_davies,
            'calinski_harabasz': hdbscan_calinski,
            'time': hdbscan_time,
            'n_clusters': n_clusters_hdbscan,
            'n_noise': n_noise
        }
        
        print(f"  Silhouette Score:      {hdbscan_silhouette:.4f}")
        print(f"  Davies-Bouldin Index:  {hdbscan_davies:.4f}")
        print(f"  Calinski-Harabasz:     {hdbscan_calinski:.2f}")
        print(f"  Clusters found:        {n_clusters_hdbscan}")
        print(f"  Noise points:          {n_noise} ({n_noise/len(X)*100:.1f}%)")
        print(f"  Time:                  {hdbscan_time:.2f}s")
        print(f"  ✅ Completed")
    else:
        print(f"  ⚠️  HDBSCAN found only {n_clusters_hdbscan} cluster(s)")
        print(f"  Noise points: {n_noise} ({n_noise/len(X)*100:.1f}%)")
        print(f"  ❌ Not suitable for this dataset")
        results['HDBSCAN'] = None
        
except ImportError:
    print("  ⚠️  HDBSCAN not installed")
    print("  Install with: pip install hdbscan")
    results['HDBSCAN'] = None

print()
print("=" * 80)

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n📊 COMPARISON TABLE")
print("=" * 80)

comparison_data = []
for method, result in results.items():
    if result is not None:
        comparison_data.append({
            'Method': method,
            'Silhouette ↑': f"{result['silhouette']:.4f}",
            'Davies-Bouldin ↓': f"{result['davies_bouldin']:.4f}",
            'Calinski-Harabasz ↑': f"{result['calinski_harabasz']:.2f}",
            'Clusters': result['n_clusters'],
            'Time (s)': f"{result['time']:.2f}"
        })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))
print()

# ============================================================================
# RANKING
# ============================================================================
print("\n🏆 RANKING BY METRICS")
print("=" * 80)

valid_methods = {k: v for k, v in results.items() if v is not None}

# Silhouette (higher is better)
silhouette_ranking = sorted(valid_methods.items(), key=lambda x: x[1]['silhouette'], reverse=True)
print("\n1. Silhouette Score (higher is better):")
for i, (method, result) in enumerate(silhouette_ranking, 1):
    print(f"   {i}. {method:15s}: {result['silhouette']:.4f}")

# Davies-Bouldin (lower is better)
davies_ranking = sorted(valid_methods.items(), key=lambda x: x[1]['davies_bouldin'])
print("\n2. Davies-Bouldin Index (lower is better):")
for i, (method, result) in enumerate(davies_ranking, 1):
    print(f"   {i}. {method:15s}: {result['davies_bouldin']:.4f}")

# Calinski-Harabasz (higher is better)
calinski_ranking = sorted(valid_methods.items(), key=lambda x: x[1]['calinski_harabasz'], reverse=True)
print("\n3. Calinski-Harabasz Score (higher is better):")
for i, (method, result) in enumerate(calinski_ranking, 1):
    print(f"   {i}. {method:15s}: {result['calinski_harabasz']:.2f}")

# ============================================================================
# OVERALL WINNER
# ============================================================================
print("\n" + "=" * 80)
print("🎯 OVERALL WINNER")
print("=" * 80)

# Calculate scores (rank-based)
scores = {method: 0 for method in valid_methods.keys()}

for i, (method, _) in enumerate(silhouette_ranking):
    scores[method] += (len(silhouette_ranking) - i)

for i, (method, _) in enumerate(davies_ranking):
    scores[method] += (len(davies_ranking) - i)

for i, (method, _) in enumerate(calinski_ranking):
    scores[method] += (len(calinski_ranking) - i)

winner = max(scores.items(), key=lambda x: x[1])

print(f"\n🥇 WINNER: {winner[0]}")
print(f"   Total Score: {winner[1]}/{len(valid_methods)*3}")
print()

for method, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"   {method:15s}: {score}/{len(valid_methods)*3} points")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n📈 Creating comparison visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Clustering Methods Comparison', fontsize=16, fontweight='bold')

# 1. Silhouette Score
ax = axes[0, 0]
methods = [m for m in valid_methods.keys()]
silhouette_scores = [valid_methods[m]['silhouette'] for m in methods]
bars = ax.barh(methods, silhouette_scores, color='skyblue')
ax.set_xlabel('Silhouette Score (higher is better)', fontweight='bold')
ax.set_title('Silhouette Score Comparison')
ax.grid(axis='x', alpha=0.3)
# Highlight best
best_idx = silhouette_scores.index(max(silhouette_scores))
bars[best_idx].set_color('gold')

# 2. Davies-Bouldin Index
ax = axes[0, 1]
davies_scores = [valid_methods[m]['davies_bouldin'] for m in methods]
bars = ax.barh(methods, davies_scores, color='lightcoral')
ax.set_xlabel('Davies-Bouldin Index (lower is better)', fontweight='bold')
ax.set_title('Davies-Bouldin Index Comparison')
ax.grid(axis='x', alpha=0.3)
# Highlight best
best_idx = davies_scores.index(min(davies_scores))
bars[best_idx].set_color('gold')

# 3. Calinski-Harabasz Score
ax = axes[1, 0]
calinski_scores = [valid_methods[m]['calinski_harabasz'] for m in methods]
bars = ax.barh(methods, calinski_scores, color='lightgreen')
ax.set_xlabel('Calinski-Harabasz Score (higher is better)', fontweight='bold')
ax.set_title('Calinski-Harabasz Score Comparison')
ax.grid(axis='x', alpha=0.3)
# Highlight best
best_idx = calinski_scores.index(max(calinski_scores))
bars[best_idx].set_color('gold')

# 4. Execution Time
ax = axes[1, 1]
times = [valid_methods[m]['time'] for m in methods]
bars = ax.barh(methods, times, color='plum')
ax.set_xlabel('Execution Time (seconds)', fontweight='bold')
ax.set_title('Execution Time Comparison')
ax.grid(axis='x', alpha=0.3)
# Highlight fastest
best_idx = times.index(min(times))
bars[best_idx].set_color('gold')

plt.tight_layout()
plt.savefig('carbon24_clustering_comparison/figures/methods_comparison.png', dpi=300, bbox_inches='tight')
print("💾 Saved: carbon24_clustering_comparison/figures/methods_comparison.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n💾 Saving results...")

# Save comparison table
comparison_df.to_csv('carbon24_clustering_comparison/comparison_table.csv', index=False)
print("✅ Saved: carbon24_clustering_comparison/comparison_table.csv")

# Save detailed results
results_summary = {
    'winner': winner[0],
    'n_clusters': n_clusters,
    'methods': {}
}

for method, result in results.items():
    if result is not None:
        results_summary['methods'][method] = {
            'silhouette_score': float(result['silhouette']),
            'davies_bouldin_index': float(result['davies_bouldin']),
            'calinski_harabasz_score': float(result['calinski_harabasz']),
            'n_clusters': int(result['n_clusters']),
            'execution_time': float(result['time'])
        }
        if 'n_noise' in result:
            results_summary['methods'][method]['n_noise'] = int(result['n_noise'])

with open('carbon24_clustering_comparison/comparison_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print("✅ Saved: carbon24_clustering_comparison/comparison_results.json")

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "=" * 80)
print("✅ COMPARISON COMPLETED!")
print("=" * 80)
print(f"\n🏆 Best method: {winner[0]}")
print(f"\n📁 Results saved to: carbon24_clustering_comparison/")
print("\nFiles created:")
print("  - comparison_table.csv")
print("  - comparison_results.json")
print("  - figures/methods_comparison.png")
print()
