"""
Carbon-24 Stability Analysis & Energy Prediction
=================================================
Mở rộng từ clustering analysis:
1. Phát hiện nhóm cấu trúc ổn định/kém ổn định
2. Dự đoán energy_per_atom bằng ML
3. Trực quan hóa kết quả phân tích
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Clustering
from sklearn.cluster import KMeans

# Settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("🔬 CARBON-24 STABILITY ANALYSIS & ENERGY PREDICTION")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD DATA & CLUSTERING RESULTS
# ============================================================================
print("📂 STEP 1: Loading data...")
print("-" * 80)

# Load features
data_path = 'carbon24_feature_selected/carbon24_feature_selected_standard.csv'
df = pd.read_csv(data_path)

with open('carbon24_feature_selected/selected_features.json', 'r') as f:
    feature_info = json.load(f)

numeric_features = feature_info['numeric_features']

# Remove relative_energy from features (we'll predict energy_per_atom)
features_for_clustering = [f for f in numeric_features if f != 'relative_energy']

print(f"✅ Loaded: {len(df)} samples")
print(f"   Features for clustering: {len(features_for_clustering)}")
print(f"   Numeric features: {len(numeric_features)}")
print()

# Check if we have energy_per_atom
# We need to load the original data with energy_per_atom
original_data_path = 'carbon24_features/carbon24_project/data/carbon24_features.csv'
df_original = pd.read_csv(original_data_path)

# Merge energy_per_atom into our dataframe
if 'energy_per_atom' not in df.columns and 'energy' in df_original.columns:
    # Calculate energy_per_atom from energy and num_atoms
    df_original['energy_per_atom'] = df_original['energy'] / df_original['num_atoms']
    df = df.merge(df_original[['row_index', 'energy_per_atom']], on='row_index', how='left')
    print("✅ Added energy_per_atom from original data")
elif 'energy' in df.columns and 'num_atoms' in df.columns:
    df['energy_per_atom'] = df['energy'] / df['num_atoms']
    print("✅ Calculated energy_per_atom from energy and num_atoms")

print()

# Create output directory
os.makedirs('carbon24_stability_analysis', exist_ok=True)
os.makedirs('carbon24_stability_analysis/figures', exist_ok=True)
os.makedirs('carbon24_stability_analysis/models', exist_ok=True)

# ============================================================================
# 2. CLUSTERING FOR STABILITY GROUPS
# ============================================================================
print("📊 STEP 2: Clustering for stability groups...")
print("-" * 80)

# Prepare data for clustering
X_cluster = df[features_for_clustering].values

# Optimal k from previous analysis
optimal_k = 3

# Perform K-means clustering
print(f"Running K-means with k={optimal_k}...")
kmeans = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)
df['cluster'] = kmeans.fit_predict(X_cluster)

print(f"✅ Clustering completed")
print(f"   Cluster distribution:")
for i in range(optimal_k):
    count = (df['cluster'] == i).sum()
    print(f"   Cluster {i}: {count} samples ({count/len(df)*100:.1f}%)")
print()

# ============================================================================
# 3. STABILITY ANALYSIS BY CLUSTER
# ============================================================================
print("⚡ STEP 3: Analyzing stability by cluster...")
print("-" * 80)

# Analyze relative_energy by cluster
stability_stats = df.groupby('cluster')['relative_energy'].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).round(4)

print("\nStability Statistics by Cluster:")
print(stability_stats)
print()

# Classify clusters by stability
cluster_stability = []
for i in range(optimal_k):
    mean_energy = stability_stats.loc[i, 'mean']
    median_energy = stability_stats.loc[i, 'median']
    
    if mean_energy < 0.15:
        stability = "Highly Stable"
        color = "green"
    elif mean_energy < 0.30:
        stability = "Moderately Stable"
        color = "orange"
    else:
        stability = "Less Stable"
        color = "red"
    
    cluster_stability.append({
        'cluster': i,
        'stability': stability,
        'mean_energy': mean_energy,
        'median_energy': median_energy,
        'color': color
    })

stability_df = pd.DataFrame(cluster_stability)
print("\n🎯 Cluster Stability Classification:")
print(stability_df.to_string(index=False))
print()

# Save stability classification
stability_df.to_csv('carbon24_stability_analysis/cluster_stability_classification.csv', index=False)

# ============================================================================
# 4. ENERGY PREDICTION MODELS
# ============================================================================
print("🤖 STEP 4: Building energy prediction models...")
print("-" * 80)

# Prepare data for prediction
# Use geometric features to predict energy_per_atom
prediction_features = [f for f in features_for_clustering if f in df.columns]
X_pred = df[prediction_features].values
y_pred = df['energy_per_atom'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, y_pred, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print()

# Define models to test
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
}

# Train and evaluate models
results = []
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    start_time = time()
    
    # Train
    model.fit(X_train, y_train)
    train_time = time() - start_time
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Time (s)': train_time
    })
    
    trained_models[name] = {
        'model': model,
        'y_test_pred': y_test_pred
    }
    
    print(f"  ✅ Test R²: {test_r2:.4f}, Test MAE: {test_mae:.4f}")

print()

# Results table
results_df = pd.DataFrame(results)
print("\n📊 MODEL COMPARISON:")
print("=" * 80)
print(results_df.to_string(index=False))
print()

# Find best model
best_model_name = results_df.loc[results_df['Test R²'].idxmax(), 'Model']
best_model = trained_models[best_model_name]['model']
print(f"🏆 Best Model: {best_model_name}")
print(f"   Test R²: {results_df.loc[results_df['Test R²'].idxmax(), 'Test R²']:.4f}")
print(f"   Test MAE: {results_df.loc[results_df['Test R²'].idxmax(), 'Test MAE']:.4f}")
print()

# Save results
results_df.to_csv('carbon24_stability_analysis/prediction_model_comparison.csv', index=False)

# Save best model predictions
best_predictions = trained_models[best_model_name]['y_test_pred']
prediction_results = pd.DataFrame({
    'actual': y_test,
    'predicted': best_predictions,
    'error': y_test - best_predictions,
    'abs_error': np.abs(y_test - best_predictions)
})
prediction_results.to_csv('carbon24_stability_analysis/best_model_predictions.csv', index=False)

# ============================================================================
# 5. FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("📈 STEP 5: Analyzing feature importance...")
    print("-" * 80)
    
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': prediction_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))
    print()
    
    feature_importance_df.to_csv('carbon24_stability_analysis/feature_importance.csv', index=False)

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("📊 STEP 6: Creating visualizations...")
print("-" * 80)

# 6.1 Stability Distribution by Cluster
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Carbon-24 Stability Analysis', fontsize=16, fontweight='bold')

# Plot 1: Relative Energy Distribution by Cluster
ax = axes[0, 0]
for i in range(optimal_k):
    cluster_data = df[df['cluster'] == i]['relative_energy']
    stability_label = stability_df[stability_df['cluster'] == i]['stability'].values[0]
    ax.hist(cluster_data, bins=30, alpha=0.6, label=f'Cluster {i} ({stability_label})')
ax.set_xlabel('Relative Energy (eV/atom)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Relative Energy Distribution by Cluster')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Box Plot of Relative Energy by Cluster
ax = axes[0, 1]
cluster_colors = [stability_df[stability_df['cluster'] == i]['color'].values[0] for i in range(optimal_k)]
bp = ax.boxplot([df[df['cluster'] == i]['relative_energy'] for i in range(optimal_k)],
                 labels=[f'Cluster {i}' for i in range(optimal_k)],
                 patch_artist=True)
for patch, color in zip(bp['boxes'], cluster_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Relative Energy (eV/atom)', fontweight='bold')
ax.set_title('Stability Comparison Across Clusters')
ax.grid(axis='y', alpha=0.3)

# Plot 3: Model Performance Comparison
ax = axes[1, 0]
x_pos = np.arange(len(results_df))
bars = ax.bar(x_pos, results_df['Test R²'], color='skyblue', alpha=0.7)
# Highlight best model
best_idx = results_df['Test R²'].idxmax()
bars[best_idx].set_color('gold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.set_ylabel('Test R² Score', fontweight='bold')
ax.set_title('Energy Prediction Model Performance')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

# Plot 4: Actual vs Predicted Energy
ax = axes[1, 1]
ax.scatter(y_test, best_predictions, alpha=0.5, s=20)
min_val = min(y_test.min(), best_predictions.min())
max_val = max(y_test.max(), best_predictions.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Energy per Atom (eV/atom)', fontweight='bold')
ax.set_ylabel('Predicted Energy per Atom (eV/atom)', fontweight='bold')
ax.set_title(f'Prediction Performance ({best_model_name})')
ax.legend()
ax.grid(alpha=0.3)

# Add R² text
test_r2 = results_df.loc[results_df['Model'] == best_model_name, 'Test R²'].values[0]
ax.text(0.05, 0.95, f'R² = {test_r2:.4f}', transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('carbon24_stability_analysis/figures/stability_analysis_overview.png', dpi=300, bbox_inches='tight')
print("💾 Saved: stability_analysis_overview.png")

# 6.2 Detailed Stability Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Detailed Stability Analysis by Cluster', fontsize=16, fontweight='bold')

# Plot 1: Cluster sizes with stability labels
ax = axes[0]
cluster_counts = df['cluster'].value_counts().sort_index()
colors = [stability_df[stability_df['cluster'] == i]['color'].values[0] for i in range(optimal_k)]
bars = ax.bar(range(optimal_k), cluster_counts.values, color=colors, alpha=0.7)
ax.set_xlabel('Cluster', fontweight='bold')
ax.set_ylabel('Number of Structures', fontweight='bold')
ax.set_title('Cluster Size Distribution')
ax.set_xticks(range(optimal_k))

# Add labels
for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
    stability_label = stability_df[stability_df['cluster'] == i]['stability'].values[0]
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{stability_label}\n({count})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Mean relative energy by cluster
ax = axes[1]
mean_energies = [stability_stats.loc[i, 'mean'] for i in range(optimal_k)]
bars = ax.bar(range(optimal_k), mean_energies, color=colors, alpha=0.7)
ax.set_xlabel('Cluster', fontweight='bold')
ax.set_ylabel('Mean Relative Energy (eV/atom)', fontweight='bold')
ax.set_title('Average Stability by Cluster')
ax.set_xticks(range(optimal_k))
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, energy) in enumerate(zip(bars, mean_energies)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{energy:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('carbon24_stability_analysis/figures/cluster_stability_details.png', dpi=300, bbox_inches='tight')
print("💾 Saved: cluster_stability_details.png")

# 6.3 Feature Importance (if available)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = 15
    top_features = feature_importance_df.head(top_n)
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'], color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontweight='bold')
    ax.set_title(f'Top {top_n} Features for Energy Prediction ({best_model_name})', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('carbon24_stability_analysis/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: feature_importance.png")

# 6.4 Prediction Error Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Prediction Error Analysis', fontsize=16, fontweight='bold')

# Plot 1: Error distribution
ax = axes[0]
errors = y_test - best_predictions
ax.hist(errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax.set_xlabel('Prediction Error (eV/atom)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Prediction Errors')
ax.legend()
ax.grid(alpha=0.3)

# Add statistics
mean_error = np.mean(errors)
std_error = np.std(errors)
ax.text(0.05, 0.95, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Residual plot
ax = axes[1]
ax.scatter(best_predictions, errors, alpha=0.5, s=20, color='purple')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Energy per Atom (eV/atom)', fontweight='bold')
ax.set_ylabel('Residual (Actual - Predicted)', fontweight='bold')
ax.set_title('Residual Plot')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('carbon24_stability_analysis/figures/prediction_error_analysis.png', dpi=300, bbox_inches='tight')
print("💾 Saved: prediction_error_analysis.png")

# 6.5 PCA Visualization with Stability
print("\nCreating PCA visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('PCA Visualization of Carbon Structures', fontsize=16, fontweight='bold')

# Plot 1: Colored by cluster/stability
ax = axes[0]
for i in range(optimal_k):
    mask = df['cluster'] == i
    stability_label = stability_df[stability_df['cluster'] == i]['stability'].values[0]
    color = stability_df[stability_df['cluster'] == i]['color'].values[0]
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
              label=f'Cluster {i}: {stability_label}',
              alpha=0.6, s=30, c=color)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
ax.set_title('Structures Colored by Stability')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Colored by relative energy
ax = axes[1]
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=df['relative_energy'], 
                     cmap='RdYlGn_r', alpha=0.6, s=30)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Relative Energy (eV/atom)', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
ax.set_title('Structures Colored by Relative Energy')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('carbon24_stability_analysis/figures/pca_stability_visualization.png', dpi=300, bbox_inches='tight')
print("💾 Saved: pca_stability_visualization.png")

# ============================================================================
# 7. SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("📋 GENERATING SUMMARY REPORT")
print("=" * 80)

summary_report = f"""
================================================================================
CARBON-24 STABILITY ANALYSIS & ENERGY PREDICTION - SUMMARY REPORT
================================================================================

DATASET INFORMATION:
-------------------
Total structures: {len(df)}
Features used: {len(prediction_features)}
Optimal clusters: {optimal_k}

STABILITY CLASSIFICATION:
------------------------
"""

for _, row in stability_df.iterrows():
    cluster_count = (df['cluster'] == row['cluster']).sum()
    summary_report += f"\nCluster {row['cluster']}: {row['stability']}"
    summary_report += f"\n  - Structures: {cluster_count} ({cluster_count/len(df)*100:.1f}%)"
    summary_report += f"\n  - Mean relative energy: {row['mean_energy']:.4f} eV/atom"
    summary_report += f"\n  - Median relative energy: {row['median_energy']:.4f} eV/atom"

summary_report += f"""

ENERGY PREDICTION MODELS:
-------------------------
Best Model: {best_model_name}
  - Test R²: {results_df.loc[results_df['Model'] == best_model_name, 'Test R²'].values[0]:.4f}
  - Test MAE: {results_df.loc[results_df['Model'] == best_model_name, 'Test MAE'].values[0]:.4f} eV/atom
  - Test RMSE: {results_df.loc[results_df['Model'] == best_model_name, 'Test RMSE'].values[0]:.4f} eV/atom

All Models Performance:
"""

for _, row in results_df.iterrows():
    summary_report += f"\n{row['Model']}:"
    summary_report += f"\n  - Test R²: {row['Test R²']:.4f}"
    summary_report += f"\n  - Test MAE: {row['Test MAE']:.4f} eV/atom"
    summary_report += f"\n  - Test RMSE: {row['Test RMSE']:.4f} eV/atom"

if best_model_name in ['Random Forest', 'Gradient Boosting']:
    summary_report += f"""

TOP 5 MOST IMPORTANT FEATURES:
------------------------------
"""
    for idx, row in feature_importance_df.head(5).iterrows():
        summary_report += f"\n{idx+1}. {row['feature']}: {row['importance']:.4f}"

summary_report += f"""

KEY FINDINGS:
------------
1. Structures are grouped into {optimal_k} distinct clusters based on geometric features
2. Stability varies significantly across clusters:
   - Most stable cluster has mean relative energy: {stability_stats['mean'].min():.4f} eV/atom
   - Least stable cluster has mean relative energy: {stability_stats['mean'].max():.4f} eV/atom
3. Energy prediction achieves R² = {results_df['Test R²'].max():.4f} using {best_model_name}
4. Geometric features successfully predict energy with MAE = {results_df['Test MAE'].min():.4f} eV/atom

OUTPUT FILES:
------------
📁 carbon24_stability_analysis/
  ├── cluster_stability_classification.csv
  ├── prediction_model_comparison.csv
  ├── best_model_predictions.csv
  ├── feature_importance.csv (if applicable)
  └── figures/
      ├── stability_analysis_overview.png
      ├── cluster_stability_details.png
      ├── feature_importance.png (if applicable)
      ├── prediction_error_analysis.png
      └── pca_stability_visualization.png

================================================================================
Analysis completed successfully!
================================================================================
"""

print(summary_report)

# Save report
with open('carbon24_stability_analysis/ANALYSIS_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print("\n💾 Saved: ANALYSIS_REPORT.txt")
print("\n✅ ALL ANALYSIS COMPLETED!")
print("=" * 80)
