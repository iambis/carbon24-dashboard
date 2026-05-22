"""
Add PCA 3D visualization to existing clustered data
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import json

print("="*80)
print("ADDING PCA 3D TO CLUSTERED DATA")
print("="*80)

# Load clustered data
try:
    df = pd.read_csv('carbon24_kmeans_results/carbon24_clustered.csv')
    print(f"\n✓ Loaded clustered data: {df.shape}")
except:
    print("\n❌ Error: Please run K-means clustering first!")
    exit()

# Load feature info
with open('carbon24_preprocessing_results/selected_features.json', 'r') as f:
    feature_info = json.load(f)

numeric_features = feature_info['numeric_features']
print(f"✓ Loaded {len(numeric_features)} numeric features")

# Prepare data
X = df[numeric_features].copy()

# PCA 3D
print("\n[1] Computing PCA 3D...")
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X)

print(f"✓ PCA 3D explained variance:")
print(f"  PC1: {pca_3d.explained_variance_ratio_[0]:.4f}")
print(f"  PC2: {pca_3d.explained_variance_ratio_[1]:.4f}")
print(f"  PC3: {pca_3d.explained_variance_ratio_[2]:.4f}")
print(f"  Total: {pca_3d.explained_variance_ratio_.sum():.4f}")

# Add to dataframe
df['pca1_3d'] = X_pca_3d[:, 0]
df['pca2_3d'] = X_pca_3d[:, 1]
df['pca3_3d'] = X_pca_3d[:, 2]

# Save updated data
df.to_csv('carbon24_kmeans_results/carbon24_clustered.csv', index=False)
print(f"\n✓ Saved updated data with PCA 3D components")

# Create 3D visualization
print("\n[2] Creating 3D visualizations...")

# Plot 1: By Cluster
fig1 = go.Figure()

for cluster in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster]
    
    fig1.add_trace(go.Scatter3d(
        x=cluster_data['pca1_3d'],
        y=cluster_data['pca2_3d'],
        z=cluster_data['pca3_3d'],
        mode='markers',
        name=f'Cluster {cluster}',
        marker=dict(
            size=3,
            opacity=0.7,
            line=dict(width=0.5, color='white')
        ),
        text=[f'Cluster: {cluster}<br>Energy: {e:.4f}' 
              for e in cluster_data['relative_energy']],
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
    ))

fig1.update_layout(
    title=dict(
        text=f'PCA 3D: Clusters (Variance: {pca_3d.explained_variance_ratio_.sum():.2%})',
        font=dict(size=16, family='Arial Black')
    ),
    scene=dict(
        xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})',
        yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})',
        zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        )
    ),
    width=1000,
    height=800,
    showlegend=True
)

fig1.write_html('carbon24_kmeans_results/pca_3d_clusters.html')
print("✓ Saved: carbon24_kmeans_results/pca_3d_clusters.html")

# Plot 2: By Relative Energy
if 'relative_energy' in df.columns:
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter3d(
        x=df['pca1_3d'],
        y=df['pca2_3d'],
        z=df['pca3_3d'],
        mode='markers',
        marker=dict(
            size=3,
            color=df['relative_energy'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Relative<br>Energy"),
            opacity=0.7,
            line=dict(width=0.5, color='white')
        ),
        text=[f'Cluster: {c}<br>Energy: {e:.4f}' 
              for c, e in zip(df['cluster'], df['relative_energy'])],
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
    ))
    
    fig2.update_layout(
        title=dict(
            text=f'PCA 3D: Relative Energy (Variance: {pca_3d.explained_variance_ratio_.sum():.2%})',
            font=dict(size=16, family='Arial Black')
        ),
        scene=dict(
            xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})',
            yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})',
            zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1000,
        height=800
    )
    
    fig2.write_html('carbon24_kmeans_results/pca_3d_energy.html')
    print("✓ Saved: carbon24_kmeans_results/pca_3d_energy.html")

# Plot 3: By Crystal System
if 'crystal_system' in df.columns:
    fig3 = go.Figure()
    
    for crystal in sorted(df['crystal_system'].unique()):
        crystal_data = df[df['crystal_system'] == crystal]
        
        fig3.add_trace(go.Scatter3d(
            x=crystal_data['pca1_3d'],
            y=crystal_data['pca2_3d'],
            z=crystal_data['pca3_3d'],
            mode='markers',
            name=crystal,
            marker=dict(
                size=3,
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=[f'Crystal: {crystal}<br>Cluster: {c}<br>Energy: {e:.4f}' 
                  for c, e in zip(crystal_data['cluster'], crystal_data['relative_energy'])],
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
        ))
    
    fig3.update_layout(
        title=dict(
            text=f'PCA 3D: Crystal Systems (Variance: {pca_3d.explained_variance_ratio_.sum():.2%})',
            font=dict(size=16, family='Arial Black')
        ),
        scene=dict(
            xaxis_title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})',
            yaxis_title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})',
            zaxis_title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1000,
        height=800,
        showlegend=True
    )
    
    fig3.write_html('carbon24_kmeans_results/pca_3d_crystal_systems.html')
    print("✓ Saved: carbon24_kmeans_results/pca_3d_crystal_systems.html")

print("\n" + "="*80)
print("✅ PCA 3D COMPLETED!")
print("="*80)
print("\nGenerated files:")
print("  - carbon24_kmeans_results/pca_3d_clusters.html")
print("  - carbon24_kmeans_results/pca_3d_energy.html")
print("  - carbon24_kmeans_results/pca_3d_crystal_systems.html")
print("\nOpen these HTML files in your browser to view interactive 3D plots!")
