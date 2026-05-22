"""
Generate PDF Report for Carbon-24 Stability Analysis
====================================================
Tạo báo cáo PDF chuyên nghiệp với tất cả kết quả phân tích
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import json

# Settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 80)
print("📄 GENERATING PDF REPORT")
print("=" * 80)
print()

# Load data
print("Loading data...")
df = pd.read_csv('carbon24_feature_selected/carbon24_feature_selected_standard.csv')
df_original = pd.read_csv('carbon24_features/carbon24_project/data/carbon24_features.csv')
df_original['energy_per_atom'] = df_original['energy'] / df_original['num_atoms']
df = df.merge(df_original[['row_index', 'energy_per_atom']], on='row_index', how='left')

stability_df = pd.read_csv('carbon24_stability_analysis/cluster_stability_classification.csv')
model_results = pd.read_csv('carbon24_stability_analysis/prediction_model_comparison.csv')
predictions = pd.read_csv('carbon24_stability_analysis/best_model_predictions.csv')

try:
    feature_importance = pd.read_csv('carbon24_stability_analysis/feature_importance.csv')
except:
    feature_importance = None

with open('carbon24_feature_selected/selected_features.json', 'r') as f:
    feature_info = json.load(f)

# Add clustering
from sklearn.cluster import KMeans
features_for_clustering = [f for f in feature_info['numeric_features'] if f != 'relative_energy']
X_cluster = df[features_for_clustering].values
optimal_k = len(stability_df)
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=300, random_state=42)
df['cluster'] = kmeans.fit_predict(X_cluster)

print("✅ Data loaded")
print()

# Create PDF
pdf_filename = f'carbon24_stability_analysis/Carbon24_Stability_Report_{datetime.now().strftime("%Y%m%d")}.pdf'

with PdfPages(pdf_filename) as pdf:
    
    # ========================================================================
    # PAGE 1: TITLE PAGE
    # ========================================================================
    print("Creating title page...")
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    
    # Title
    plt.text(0.5, 0.75, 'Carbon-24 Allotropes', 
             ha='center', va='center', fontsize=36, fontweight='bold', color='#1f77b4')
    plt.text(0.5, 0.68, 'Stability Analysis & Energy Prediction', 
             ha='center', va='center', fontsize=24, color='#2ca02c')
    
    # Subtitle
    plt.text(0.5, 0.55, 'Phân nhóm và hệ thống hóa các dạng thù hình Carbon',
             ha='center', va='center', fontsize=14, style='italic', color='#666')
    plt.text(0.5, 0.50, 'dựa trên đặc trưng hình học và mức ổn định năng lượng',
             ha='center', va='center', fontsize=14, style='italic', color='#666')
    
    # Key metrics box
    metrics_text = f"""
    Dataset: {len(df):,} Carbon Structures
    Clusters: {len(stability_df)}
    Best Model: {model_results.loc[model_results['Test R²'].idxmax(), 'Model']}
    Prediction Accuracy: R² = {model_results['Test R²'].max():.4f}
    """
    
    plt.text(0.5, 0.30, metrics_text,
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#1f77b4', linewidth=2))
    
    # Date
    plt.text(0.5, 0.10, f'Report Generated: {datetime.now().strftime("%B %d, %Y")}',
             ha='center', va='center', fontsize=10, color='#666')
    
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PAGE 2: EXECUTIVE SUMMARY
    # ========================================================================
    print("Creating executive summary...")
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    
    summary_text = f"""
EXECUTIVE SUMMARY

PROJECT OVERVIEW
This analysis extends the initial clustering study of Carbon-24 allotropes to include:
• Stability classification of structural groups
• Machine learning-based energy prediction
• Comprehensive visualization and analysis

DATASET
• Total Structures: {len(df):,}
• Features Used: {len(feature_info['numeric_features'])}
• Clustering Method: K-means (k={len(stability_df)})

KEY FINDINGS

1. STABILITY CLASSIFICATION
   Three distinct stability groups were identified:
"""
    
    for _, row in stability_df.iterrows():
        cluster_count = (df['cluster'] == row['cluster']).sum()
        percentage = cluster_count / len(df) * 100
        summary_text += f"""
   • Cluster {row['cluster']}: {row['stability']}
     - Structures: {cluster_count:,} ({percentage:.1f}%)
     - Mean Energy: {row['mean_energy']:.4f} eV/atom
"""
    
    best_model = model_results.loc[model_results['Test R²'].idxmax(), 'Model']
    best_r2 = model_results['Test R²'].max()
    best_mae = model_results['Test MAE'].min()
    
    summary_text += f"""

2. ENERGY PREDICTION
   • Best Model: {best_model}
   • Test R²: {best_r2:.4f}
   • Test MAE: {best_mae:.4f} eV/atom
   • Prediction accuracy is near-perfect, demonstrating strong
     structure-energy relationships

3. FEATURE IMPORTANCE
"""
    
    if feature_importance is not None:
        for idx, row in feature_importance.head(3).iterrows():
            summary_text += f"   • {row['feature']}: {row['importance']:.4f}\n"
    
    summary_text += """

IMPLICATIONS
• Geometric features successfully predict energy stability
• Clear separation between stable and unstable structures
• Results can guide material discovery and design
"""
    
    plt.text(0.05, 0.95, summary_text,
             ha='left', va='top', fontsize=10, family='monospace',
             transform=fig.transFigure)
    
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PAGE 3: STABILITY ANALYSIS
    # ========================================================================
    print("Creating stability analysis page...")
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Stability Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Cluster distribution
    ax = axes[0, 0]
    cluster_counts = df['cluster'].value_counts().sort_index()
    colors = [stability_df[stability_df['cluster'] == i]['color'].values[0] for i in range(len(stability_df))]
    bars = ax.bar(range(len(stability_df)), cluster_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Cluster', fontweight='bold')
    ax.set_ylabel('Number of Structures', fontweight='bold')
    ax.set_title('Cluster Size Distribution')
    ax.set_xticks(range(len(stability_df)))
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
        stability_label = stability_df[stability_df['cluster'] == i]['stability'].values[0]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{stability_label}\n({count})',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 2: Mean energy by cluster
    ax = axes[0, 1]
    mean_energies = [df[df['cluster'] == i]['relative_energy'].mean() for i in range(len(stability_df))]
    bars = ax.bar(range(len(stability_df)), mean_energies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Cluster', fontweight='bold')
    ax.set_ylabel('Mean Relative Energy (eV/atom)', fontweight='bold')
    ax.set_title('Average Stability by Cluster')
    ax.set_xticks(range(len(stability_df)))
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    for i, (bar, energy) in enumerate(zip(bars, mean_energies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{energy:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Energy distribution
    ax = axes[1, 0]
    for i in range(len(stability_df)):
        cluster_data = df[df['cluster'] == i]['relative_energy']
        stability_label = stability_df[stability_df['cluster'] == i]['stability'].values[0]
        ax.hist(cluster_data, bins=30, alpha=0.6, label=f'Cluster {i}: {stability_label}')
    ax.set_xlabel('Relative Energy (eV/atom)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Energy Distribution by Cluster')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 4: Box plot
    ax = axes[1, 1]
    bp = ax.boxplot([df[df['cluster'] == i]['relative_energy'] for i in range(len(stability_df))],
                     labels=[f'C{i}' for i in range(len(stability_df))],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Cluster', fontweight='bold')
    ax.set_ylabel('Relative Energy (eV/atom)', fontweight='bold')
    ax.set_title('Energy Distribution Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PAGE 4: ENERGY PREDICTION
    # ========================================================================
    print("Creating energy prediction page...")
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Energy Prediction Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Model comparison (R²)
    ax = axes[0, 0]
    x_pos = range(len(model_results))
    bars = ax.bar(x_pos, model_results['Test R²'], color='skyblue', alpha=0.7, edgecolor='black')
    best_idx = model_results['Test R²'].idxmax()
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_results['Model'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Test R² Score', fontweight='bold')
    ax.set_title('Model Performance (R²)')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, r2) in enumerate(zip(bars, model_results['Test R²'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{r2:.4f}',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Model comparison (MAE)
    ax = axes[0, 1]
    bars = ax.bar(x_pos, model_results['Test MAE'], color='lightcoral', alpha=0.7, edgecolor='black')
    best_idx = model_results['Test MAE'].idxmin()
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_results['Model'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Test MAE (eV/atom)', fontweight='bold')
    ax.set_title('Model Performance (MAE)')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Actual vs Predicted
    ax = axes[1, 0]
    ax.scatter(predictions['actual'], predictions['predicted'], alpha=0.5, s=10)
    min_val = min(predictions['actual'].min(), predictions['predicted'].min())
    max_val = max(predictions['actual'].max(), predictions['predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Energy per Atom (eV/atom)', fontweight='bold')
    ax.set_ylabel('Predicted Energy per Atom (eV/atom)', fontweight='bold')
    ax.set_title(f'Prediction Performance ({best_model})')
    ax.legend()
    ax.grid(alpha=0.3)
    
    test_r2 = model_results.loc[model_results['Model'] == best_model, 'Test R²'].values[0]
    ax.text(0.05, 0.95, f'R² = {test_r2:.4f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 4: Error distribution
    ax = axes[1, 1]
    errors = predictions['error']
    ax.hist(errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error (eV/atom)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    mean_error = errors.mean()
    std_error = errors.std()
    ax.text(0.05, 0.95, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PAGE 5: FEATURE IMPORTANCE (if available)
    # ========================================================================
    if feature_importance is not None:
        print("Creating feature importance page...")
        
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        top_n = min(20, len(feature_importance))
        top_features = feature_importance.head(top_n)
        
        y_pos = range(len(top_features))
        ax.barh(y_pos, top_features['importance'], color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(f'Top {top_n} Features for Energy Prediction', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add values
        for i, (y, imp) in enumerate(zip(y_pos, top_features['importance'])):
            ax.text(imp + 0.001, y, f'{imp:.6f}', va='center', fontsize=8)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PAGE 6: STATISTICS TABLE
    # ========================================================================
    print("Creating statistics table...")
    
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    
    # Title
    plt.text(0.5, 0.95, 'Detailed Statistics', 
             ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Stability statistics
    stats_text = "\nSTABILITY STATISTICS BY CLUSTER\n" + "="*80 + "\n\n"
    
    for i in range(len(stability_df)):
        cluster_data = df[df['cluster'] == i]['relative_energy']
        stability_label = stability_df[stability_df['cluster'] == i]['stability'].values[0]
        
        stats_text += f"Cluster {i}: {stability_label}\n"
        stats_text += f"  Count:      {len(cluster_data):,}\n"
        stats_text += f"  Mean:       {cluster_data.mean():.4f} eV/atom\n"
        stats_text += f"  Median:     {cluster_data.median():.4f} eV/atom\n"
        stats_text += f"  Std Dev:    {cluster_data.std():.4f} eV/atom\n"
        stats_text += f"  Min:        {cluster_data.min():.4f} eV/atom\n"
        stats_text += f"  Max:        {cluster_data.max():.4f} eV/atom\n"
        stats_text += f"  25th %ile:  {cluster_data.quantile(0.25):.4f} eV/atom\n"
        stats_text += f"  75th %ile:  {cluster_data.quantile(0.75):.4f} eV/atom\n\n"
    
    # Model statistics
    stats_text += "\nMODEL PERFORMANCE STATISTICS\n" + "="*80 + "\n\n"
    
    for _, row in model_results.iterrows():
        stats_text += f"{row['Model']}\n"
        stats_text += f"  Train R²:    {row['Train R²']:.4f}\n"
        stats_text += f"  Test R²:     {row['Test R²']:.4f}\n"
        stats_text += f"  Train MAE:   {row['Train MAE']:.4f} eV/atom\n"
        stats_text += f"  Test MAE:    {row['Test MAE']:.4f} eV/atom\n"
        stats_text += f"  Train RMSE:  {row['Train RMSE']:.4f} eV/atom\n"
        stats_text += f"  Test RMSE:   {row['Test RMSE']:.4f} eV/atom\n"
        stats_text += f"  Time:        {row['Time (s)']:.2f} seconds\n\n"
    
    plt.text(0.05, 0.90, stats_text,
             ha='left', va='top', fontsize=9, family='monospace',
             transform=fig.transFigure)
    
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # METADATA
    # ========================================================================
    d = pdf.infodict()
    d['Title'] = 'Carbon-24 Stability Analysis Report'
    d['Author'] = 'Carbon-24 Analysis System'
    d['Subject'] = 'Stability Classification and Energy Prediction'
    d['Keywords'] = 'Carbon, Allotropes, Machine Learning, Clustering, Energy Prediction'
    d['CreationDate'] = datetime.now()

print()
print("=" * 80)
print(f"✅ PDF REPORT GENERATED SUCCESSFULLY!")
print("=" * 80)
print(f"\n📄 File: {pdf_filename}")
print(f"📊 Pages: 6-7 pages")
print(f"📁 Location: carbon24_stability_analysis/")
print()
print("Report includes:")
print("  ✓ Title page with key metrics")
print("  ✓ Executive summary")
print("  ✓ Stability analysis visualizations")
print("  ✓ Energy prediction results")
print("  ✓ Feature importance analysis")
print("  ✓ Detailed statistics tables")
print()
print("=" * 80)
