"""
Create Clustering Methods Comparison Notebook
"""
import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Title
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "title",
    "metadata": {},
    "source": [
        "# 🔬 So Sánh Các Phương Pháp Clustering - Carbon-24\n",
        "\n",
        "**Mục tiêu:** Đánh giá và so sánh hiệu suất của 4 phương pháp clustering:\n",
        "1. **K-means** - Phân cụm dựa trên centroid\n",
        "2. **GMM (Gaussian Mixture Model)** - Phân cụm xác suất\n",
        "3. **Hierarchical** - Phân cụm phân cấp\n",
        "4. **HDBSCAN** - Phân cụm dựa trên mật độ\n",
        "\n",
        "**Dataset:** 10,153 cấu trúc Carbon-24"
    ]
})

# Imports
notebook["cells"].append({
    "cell_type": "code",
    "id": "imports",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Import libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import json\n",
        "import os\n",
        "from pathlib import Path\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Metrics\n",
        "from sklearn.metrics import (\n",
        "    silhouette_score,\n",
        "    davies_bouldin_score,\n",
        "    calinski_harabasz_score,\n",
        "    adjusted_rand_score,\n",
        "    normalized_mutual_info_score\n",
        ")\n",
        "\n",
        "# Settings\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette('husl')\n",
        "%matplotlib inline\n",
        "\n",
        "print('✅ Libraries imported successfully!')"
    ]
})

# Load Results
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "load-title",
    "metadata": {},
    "source": [
        "## 📂 1. Load Clustering Results"
    ]
})

notebook["cells"].append({
    "cell_type": "code",
    "id": "load-results",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Load results from each method\n",
        "results = {}\n",
        "\n",
        "# 1. K-means\n",
        "print('📊 Loading K-means results...')\n",
        "try:\n",
        "    kmeans_df = pd.read_csv('carbon24_kmeans_results/carbon24_clustered.csv')\n",
        "    with open('carbon24_kmeans_results/clustering_report.json', 'r') as f:\n",
        "        kmeans_report = json.load(f)\n",
        "    results['K-means'] = {\n",
        "        'data': kmeans_df,\n",
        "        'report': kmeans_report,\n",
        "        'n_clusters': len(kmeans_df['cluster'].unique())\n",
        "    }\n",
        "    print(f'  ✅ K-means: {len(kmeans_df)} samples, {results[\"K-means\"][\"n_clusters\"]} clusters')\n",
        "except Exception as e:\n",
        "    print(f'  ❌ Error loading K-means: {e}')\n",
        "\n",
        "# 2. GMM\n",
        "print('\\n📊 Loading GMM results...')\n",
        "try:\n",
        "    # Try to find GMM results\n",
        "    gmm_files = list(Path('carbon24_gmm_results/results').glob('*.csv'))\n",
        "    if gmm_files:\n",
        "        gmm_df = pd.read_csv(gmm_files[0])\n",
        "        results['GMM'] = {\n",
        "            'data': gmm_df,\n",
        "            'n_clusters': len(gmm_df['cluster'].unique()) if 'cluster' in gmm_df.columns else 'N/A'\n",
        "        }\n",
        "        print(f'  ✅ GMM: {len(gmm_df)} samples')\n",
        "    else:\n",
        "        print('  ⚠️  GMM results not found in expected location')\n",
        "except Exception as e:\n",
        "    print(f'  ❌ Error loading GMM: {e}')\n",
        "\n",
        "# 3. Hierarchical\n",
        "print('\\n📊 Loading Hierarchical results...')\n",
        "try:\n",
        "    hier_files = list(Path('carbon24_hierarchical_baseline/results').glob('*.csv'))\n",
        "    if hier_files:\n",
        "        hier_df = pd.read_csv(hier_files[0])\n",
        "        results['Hierarchical'] = {\n",
        "            'data': hier_df,\n",
        "            'n_clusters': len(hier_df['cluster'].unique()) if 'cluster' in hier_df.columns else 'N/A'\n",
        "        }\n",
        "        print(f'  ✅ Hierarchical: {len(hier_df)} samples')\n",
        "    else:\n",
        "        print('  ⚠️  Hierarchical results not found')\n",
        "except Exception as e:\n",
        "    print(f'  ❌ Error loading Hierarchical: {e}')\n",
        "\n",
        "# 4. HDBSCAN\n",
        "print('\\n📊 Loading HDBSCAN results...')\n",
        "try:\n",
        "    hdbscan_df = pd.read_csv('hdbscan_phuc/hdbscan_results.csv')\n",
        "    hdbscan_profile = pd.read_csv('hdbscan_phuc/hdbscan_cluster_profile.csv')\n",
        "    results['HDBSCAN'] = {\n",
        "        'data': hdbscan_df,\n",
        "        'profile': hdbscan_profile,\n",
        "        'n_clusters': len(hdbscan_df['cluster'].unique()) - (1 if -1 in hdbscan_df['cluster'].values else 0),\n",
        "        'n_noise': (hdbscan_df['cluster'] == -1).sum() if 'cluster' in hdbscan_df.columns else 0\n",
        "    }\n",
        "    print(f'  ✅ HDBSCAN: {len(hdbscan_df)} samples, {results[\"HDBSCAN\"][\"n_clusters\"]} clusters')\n",
        "    print(f'     Noise points: {results[\"HDBSCAN\"][\"n_noise\"]}')\n",
        "except Exception as e:\n",
        "    print(f'  ❌ Error loading HDBSCAN: {e}')\n",
        "\n",
        "print(f'\\n✅ Loaded {len(results)} clustering methods')"
    ]
})

# Overview
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "overview-title",
    "metadata": {},
    "source": [
        "## 📊 2. Overview Comparison"
    ]
})

notebook["cells"].append({
    "cell_type": "code",
    "id": "overview",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Create overview table\n",
        "overview_data = []\n",
        "\n",
        "for method, data in results.items():\n",
        "    row = {\n",
        "        'Method': method,\n",
        "        'Samples': len(data['data']),\n",
        "        'Clusters': data['n_clusters']\n",
        "    }\n",
        "    \n",
        "    if method == 'HDBSCAN':\n",
        "        row['Noise Points'] = data['n_noise']\n",
        "        row['Noise %'] = f\"{data['n_noise']/len(data['data'])*100:.2f}%\"\n",
        "    else:\n",
        "        row['Noise Points'] = 0\n",
        "        row['Noise %'] = '0.00%'\n",
        "    \n",
        "    overview_data.append(row)\n",
        "\n",
        "overview_df = pd.DataFrame(overview_data)\n",
        "\n",
        "print('=' * 80)\n",
        "print('📊 CLUSTERING METHODS OVERVIEW')\n",
        "print('=' * 80)\n",
        "print(overview_df.to_string(index=False))\n",
        "print('=' * 80)"
    ]
})

# Cluster Distribution
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "distribution-title",
    "metadata": {},
    "source": [
        "## 📈 3. Cluster Distribution Analysis"
    ]
})

notebook["cells"].append({
    "cell_type": "code",
    "id": "distribution",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Analyze cluster distribution for each method\n",
        "print('=' * 80)\n",
        "print('📈 CLUSTER DISTRIBUTION')\n",
        "print('=' * 80)\n",
        "\n",
        "for method, data in results.items():\n",
        "    print(f'\\n{method}:')\n",
        "    print('-' * 60)\n",
        "    \n",
        "    if 'cluster' in data['data'].columns:\n",
        "        cluster_counts = data['data']['cluster'].value_counts().sort_index()\n",
        "        \n",
        "        for cluster_id, count in cluster_counts.items():\n",
        "            percentage = count / len(data['data']) * 100\n",
        "            if cluster_id == -1:\n",
        "                print(f'  Noise: {count:,} ({percentage:.2f}%)')\n",
        "            else:\n",
        "                print(f'  Cluster {cluster_id}: {count:,} ({percentage:.2f}%)')\n",
        "    else:\n",
        "        print('  ⚠️  No cluster column found')"
    ]
})

# Visualize Distribution
notebook["cells"].append({
    "cell_type": "code",
    "id": "viz-distribution",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Visualize cluster distributions\n",
        "n_methods = len(results)\n",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
        "fig.suptitle('Cluster Distribution Comparison', fontsize=16, fontweight='bold')\n",
        "axes = axes.flatten()\n",
        "\n",
        "for idx, (method, data) in enumerate(results.items()):\n",
        "    if idx >= 4:\n",
        "        break\n",
        "    \n",
        "    ax = axes[idx]\n",
        "    \n",
        "    if 'cluster' in data['data'].columns:\n",
        "        cluster_counts = data['data']['cluster'].value_counts().sort_index()\n",
        "        \n",
        "        # Separate noise from clusters\n",
        "        if -1 in cluster_counts.index:\n",
        "            noise_count = cluster_counts[-1]\n",
        "            cluster_counts = cluster_counts.drop(-1)\n",
        "            labels = [f'C{i}' for i in cluster_counts.index] + ['Noise']\n",
        "            values = list(cluster_counts.values) + [noise_count]\n",
        "            colors = plt.cm.Set3(range(len(cluster_counts))) + [(0.5, 0.5, 0.5, 0.7)]\n",
        "        else:\n",
        "            labels = [f'C{i}' for i in cluster_counts.index]\n",
        "            values = cluster_counts.values\n",
        "            colors = plt.cm.Set3(range(len(cluster_counts)))\n",
        "        \n",
        "        bars = ax.bar(range(len(values)), values, color=colors, edgecolor='black')\n",
        "        ax.set_xticks(range(len(labels)))\n",
        "        ax.set_xticklabels(labels, rotation=45)\n",
        "        ax.set_ylabel('Number of Samples', fontweight='bold')\n",
        "        ax.set_title(f'{method} (n_clusters={data[\"n_clusters\"]})', fontweight='bold')\n",
        "        ax.grid(axis='y', alpha=0.3)\n",
        "        \n",
        "        # Add value labels\n",
        "        for bar, val in zip(bars, values):\n",
        "            height = bar.get_height()\n",
        "            ax.text(bar.get_x() + bar.get_width()/2., height,\n",
        "                   f'{int(val):,}',\n",
        "                   ha='center', va='bottom', fontsize=9)\n",
        "    else:\n",
        "        ax.text(0.5, 0.5, 'No cluster data', ha='center', va='center',\n",
        "               transform=ax.transAxes, fontsize=14)\n",
        "        ax.set_title(method, fontweight='bold')\n",
        "\n",
        "# Hide unused subplots\n",
        "for idx in range(n_methods, 4):\n",
        "    axes[idx].axis('off')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('✅ Distribution visualization complete!')"
    ]
})

# Save notebook
with open('carbon24-clustering-comparison-evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print('✅ Part 1 created: carbon24-clustering-comparison-evaluation.ipynb')
print('   Cells: Load data, Overview, Distribution analysis')
