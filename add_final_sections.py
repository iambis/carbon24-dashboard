"""
Add Energy Analysis and Summary to Notebook
"""
import json

# Load notebook
with open('carbon24-clustering-comparison-evaluation.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Energy Analysis
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "energy-title",
    "metadata": {},
    "source": [
        "## ⚡ 6. Energy Analysis by Clustering Method\n",
        "\n",
        "Phân tích phân bố năng lượng trong các clusters của mỗi phương pháp"
    ]
})

notebook["cells"].append({
    "cell_type": "code",
    "id": "energy-analysis",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Analyze energy distribution\n",
        "print('=' * 80)\n",
        "print('⚡ ENERGY ANALYSIS BY METHOD')\n",
        "print('=' * 80)\n",
        "\n",
        "for method, data in results.items():\n",
        "    print(f'\\n{method}:')\n",
        "    print('-' * 60)\n",
        "    \n",
        "    if 'cluster' not in data['data'].columns:\n",
        "        print('  ⚠️  No cluster data')\n",
        "        continue\n",
        "    \n",
        "    if 'relative_energy' not in data['data'].columns:\n",
        "        print('  ⚠️  No energy data')\n",
        "        continue\n",
        "    \n",
        "    # Group by cluster\n",
        "    energy_stats = data['data'].groupby('cluster')['relative_energy'].agg([\n",
        "        'count', 'mean', 'std', 'min', 'max', 'median'\n",
        "    ]).round(4)\n",
        "    \n",
        "    print(energy_stats)\n",
        "    \n",
        "    # Find most stable cluster\n",
        "    if len(energy_stats) > 0:\n",
        "        most_stable_cluster = energy_stats['mean'].idxmin()\n",
        "        print(f'\\n  🟢 Most stable cluster: {most_stable_cluster}')\n",
        "        print(f'     Mean energy: {energy_stats.loc[most_stable_cluster, \"mean\"]:.4f} eV/atom')"
    ]
})

# Energy Visualization
notebook["cells"].append({
    "cell_type": "code",
    "id": "viz-energy",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Visualize energy distributions\n",
        "n_methods = len([m for m in results.keys() if 'cluster' in results[m]['data'].columns and 'relative_energy' in results[m]['data'].columns])\n",
        "\n",
        "if n_methods > 0:\n",
        "    fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
        "    fig.suptitle('Energy Distribution by Clustering Method', fontsize=16, fontweight='bold')\n",
        "    axes = axes.flatten()\n",
        "    \n",
        "    plot_idx = 0\n",
        "    for method, data in results.items():\n",
        "        if plot_idx >= 4:\n",
        "            break\n",
        "        \n",
        "        if 'cluster' not in data['data'].columns or 'relative_energy' not in data['data'].columns:\n",
        "            continue\n",
        "        \n",
        "        ax = axes[plot_idx]\n",
        "        \n",
        "        # Plot energy distribution for each cluster\n",
        "        clusters = sorted(data['data']['cluster'].unique())\n",
        "        \n",
        "        for cluster_id in clusters:\n",
        "            if cluster_id == -1:\n",
        "                label = 'Noise'\n",
        "                color = 'gray'\n",
        "            else:\n",
        "                label = f'Cluster {cluster_id}'\n",
        "                color = None\n",
        "            \n",
        "            cluster_data = data['data'][data['data']['cluster'] == cluster_id]['relative_energy']\n",
        "            ax.hist(cluster_data, bins=30, alpha=0.6, label=label, color=color)\n",
        "        \n",
        "        ax.set_xlabel('Relative Energy (eV/atom)', fontweight='bold')\n",
        "        ax.set_ylabel('Frequency', fontweight='bold')\n",
        "        ax.set_title(method, fontweight='bold')\n",
        "        ax.legend(fontsize=8)\n",
        "        ax.grid(alpha=0.3)\n",
        "        \n",
        "        plot_idx += 1\n",
        "    \n",
        "    # Hide unused subplots\n",
        "    for idx in range(plot_idx, 4):\n",
        "        axes[idx].axis('off')\n",
        "    \n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "    \n",
        "    print('✅ Energy visualization complete!')\n",
        "else:\n",
        "    print('⚠️  No energy data available for visualization')"
    ]
})

# Summary
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "summary-title",
    "metadata": {},
    "source": [
        "## 📊 7. Summary & Recommendations\n",
        "\n",
        "### Key Findings\n",
        "\n",
        "#### Method Characteristics:\n",
        "\n",
        "**K-means:**\n",
        "- ✅ Fast and scalable\n",
        "- ✅ Works well with spherical clusters\n",
        "- ❌ Requires pre-specifying k\n",
        "- ❌ Sensitive to initialization\n",
        "\n",
        "**GMM (Gaussian Mixture Model):**\n",
        "- ✅ Probabilistic clustering\n",
        "- ✅ Can model elliptical clusters\n",
        "- ✅ Provides cluster probabilities\n",
        "- ❌ Computationally expensive\n",
        "\n",
        "**Hierarchical:**\n",
        "- ✅ No need to specify k beforehand\n",
        "- ✅ Produces dendrogram\n",
        "- ✅ Can capture nested structures\n",
        "- ❌ Computationally expensive for large datasets\n",
        "\n",
        "**HDBSCAN:**\n",
        "- ✅ Automatically determines number of clusters\n",
        "- ✅ Can find clusters of varying densities\n",
        "- ✅ Identifies noise points\n",
        "- ❌ May produce many noise points\n",
        "- ❌ Sensitive to parameters"
    ]
})

notebook["cells"].append({
    "cell_type": "code",
    "id": "recommendations",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Generate recommendations\n",
        "print('=' * 80)\n",
        "print('💡 RECOMMENDATIONS')\n",
        "print('=' * 80)\n",
        "\n",
        "if not metrics_df.empty:\n",
        "    best_method = sorted_scores[0][0]\n",
        "    \n",
        "    print(f'\\n🏆 Best Overall Method: {best_method}')\n",
        "    print(f'\\nReasons:')\n",
        "    \n",
        "    best_metrics = metrics_df[metrics_df['Method'] == best_method].iloc[0]\n",
        "    \n",
        "    # Check each metric\n",
        "    if best_metrics['Silhouette'] == metrics_df['Silhouette'].max():\n",
        "        print(f'  ✅ Best Silhouette Score: {best_metrics[\"Silhouette\"]:.4f}')\n",
        "    \n",
        "    if best_metrics['Davies-Bouldin'] == metrics_df['Davies-Bouldin'].min():\n",
        "        print(f'  ✅ Best Davies-Bouldin Index: {best_metrics[\"Davies-Bouldin\"]:.4f}')\n",
        "    \n",
        "    if best_metrics['Calinski-Harabasz'] == metrics_df['Calinski-Harabasz'].max():\n",
        "        print(f'  ✅ Best Calinski-Harabasz Score: {best_metrics[\"Calinski-Harabasz\"]:.2f}')\n",
        "    \n",
        "    print(f'\\n📌 Use Cases:')\n",
        "    \n",
        "    if best_method == 'K-means':\n",
        "        print('  • Fast clustering for large datasets')\n",
        "        print('  • When clusters are roughly spherical')\n",
        "        print('  • When you know the number of clusters')\n",
        "    elif best_method == 'GMM':\n",
        "        print('  • When you need probabilistic assignments')\n",
        "        print('  • For elliptical/elongated clusters')\n",
        "        print('  • When soft clustering is needed')\n",
        "    elif best_method == 'Hierarchical':\n",
        "        print('  • When you need a dendrogram')\n",
        "        print('  • For nested cluster structures')\n",
        "        print('  • When exploring different k values')\n",
        "    elif best_method == 'HDBSCAN':\n",
        "        print('  • When clusters have varying densities')\n",
        "        print('  • When you want automatic k selection')\n",
        "        print('  • When noise detection is important')\n",
        "else:\n",
        "    print('\\n⚠️  No metrics available for recommendations')\n",
        "\n",
        "print('\\n' + '=' * 80)\n",
        "print('✅ ANALYSIS COMPLETE!')\n",
        "print('=' * 80)"
    ]
})

# Export Results
notebook["cells"].append({
    "cell_type": "code",
    "id": "export",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Export comparison results\n",
        "import os\n",
        "\n",
        "output_dir = 'carbon24_clustering_comparison_results'\n",
        "os.makedirs(output_dir, exist_ok=True)\n",
        "\n",
        "# Save overview\n",
        "overview_df.to_csv(f'{output_dir}/methods_overview.csv', index=False)\n",
        "print(f'✅ Saved: {output_dir}/methods_overview.csv')\n",
        "\n",
        "# Save metrics\n",
        "if not metrics_df.empty:\n",
        "    metrics_df.to_csv(f'{output_dir}/quality_metrics.csv', index=False)\n",
        "    print(f'✅ Saved: {output_dir}/quality_metrics.csv')\n",
        "\n",
        "# Save ranking\n",
        "if not metrics_df.empty:\n",
        "    ranking_df = pd.DataFrame(sorted_scores, columns=['Method', 'Total_Score'])\n",
        "    ranking_df['Rank'] = range(1, len(ranking_df) + 1)\n",
        "    ranking_df = ranking_df[['Rank', 'Method', 'Total_Score']]\n",
        "    ranking_df.to_csv(f'{output_dir}/method_ranking.csv', index=False)\n",
        "    print(f'✅ Saved: {output_dir}/method_ranking.csv')\n",
        "\n",
        "print(f'\\n📁 All results saved to: {output_dir}/')"
    ]
})

# Save notebook
with open('carbon24-clustering-comparison-evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print('✅ Notebook completed!')
print(f'   Total cells: {len(notebook["cells"])}')
print('   File: carbon24-clustering-comparison-evaluation.ipynb')
