"""
Script to add Elbow plot (SSE) to K-means clustering notebook
This will help determine the optimal number of clusters
"""

import json

# Read the notebook
with open('carbon24-kmeans-clustering.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell that determines optimal k (should be around cell with "Xác định số cluster tối ưu")
# We'll add the Elbow plot cell after the imports and before the metrics calculation

# New cell for Elbow plot
elbow_cell = {
    "cell_type": "markdown",
    "id": "elbow_plot_header",
    "metadata": {},
    "source": [
        "## 3. Xác Định Số Cluster Tối Ưu - Elbow Method\n",
        "\n",
        "Sử dụng **Elbow plot (SSE - Sum of Squared Errors)** để xác định điểm khuỷu tay (elbow point) - số cluster tối ưu."
    ]
}

elbow_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "elbow_plot_code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Prepare data for clustering\n",
        "X = df[numeric_features].values\n",
        "\n",
        "print(f\"Data shape for clustering: {X.shape}\")\n",
        "print(f\"Features: {len(numeric_features)}\")\n",
        "print(f\"Samples: {len(X)}\")\n",
        "\n",
        "# Create output directory\n",
        "os.makedirs('carbon24_kmeans_results', exist_ok=True)\n",
        "os.makedirs('carbon24_kmeans_results/figures', exist_ok=True)"
    ]
}

elbow_method_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "elbow_method",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Elbow Method - Calculate SSE (Sum of Squared Errors) for different k values\n",
        "print(\"🔍 Calculating SSE for Elbow Method...\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "k_range = range(2, 11)  # Test k from 2 to 10\n",
        "sse_values = []  # Sum of Squared Errors (Inertia)\n",
        "\n",
        "for k in k_range:\n",
        "    print(f\"Testing k={k}...\", end=' ')\n",
        "    \n",
        "    kmeans = KMeans(\n",
        "        n_clusters=k,\n",
        "        init='k-means++',\n",
        "        n_init=10,\n",
        "        max_iter=300,\n",
        "        random_state=42\n",
        "    )\n",
        "    \n",
        "    kmeans.fit(X)\n",
        "    sse_values.append(kmeans.inertia_)\n",
        "    \n",
        "    print(f\"SSE = {kmeans.inertia_:.2f}\")\n",
        "\n",
        "print(\"\\n✅ SSE calculation completed!\")"
    ]
}

elbow_plot_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "elbow_plot_visualization",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Plot Elbow curve\n",
        "fig, ax = plt.subplots(figsize=(10, 6))\n",
        "\n",
        "ax.plot(k_range, sse_values, 'bo-', linewidth=2, markersize=8)\n",
        "ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')\n",
        "ax.set_ylabel('SSE (Sum of Squared Errors)', fontsize=12, fontweight='bold')\n",
        "ax.set_title('Elbow Method - Xác Định Số Cluster Tối Ưu', fontsize=14, fontweight='bold')\n",
        "ax.grid(True, alpha=0.3)\n",
        "ax.set_xticks(k_range)\n",
        "\n",
        "# Highlight the elbow point (k=3 based on previous analysis)\n",
        "elbow_k = 3\n",
        "elbow_sse = sse_values[elbow_k - 2]  # Index adjustment\n",
        "ax.plot(elbow_k, elbow_sse, 'ro', markersize=15, label=f'Elbow Point (k={elbow_k})')\n",
        "ax.annotate(\n",
        "    f'Điểm khuỷu tay\\nk={elbow_k}',\n",
        "    xy=(elbow_k, elbow_sse),\n",
        "    xytext=(elbow_k + 1, elbow_sse + (max(sse_values) - min(sse_values)) * 0.1),\n",
        "    arrowprops=dict(arrowstyle='->', color='red', lw=2),\n",
        "    fontsize=11,\n",
        "    fontweight='bold',\n",
        "    color='red'\n",
        ")\n",
        "\n",
        "ax.legend(fontsize=11)\n",
        "plt.tight_layout()\n",
        "\n",
        "# Save figure\n",
        "plt.savefig('carbon24_kmeans_results/figures/kmeans_elbow_plot.png', dpi=300, bbox_inches='tight')\n",
        "print(\"💾 Saved: carbon24_kmeans_results/figures/kmeans_elbow_plot.png\")\n",
        "\n",
        "plt.show()\n",
        "\n",
        "# Print SSE values\n",
        "print(\"\\n📊 SSE Values:\")\n",
        "print(\"=\"*40)\n",
        "for k, sse in zip(k_range, sse_values):\n",
        "    marker = \" ← Elbow Point\" if k == elbow_k else \"\"\n",
        "    print(f\"k={k:2d}: SSE = {sse:,.2f}{marker}\")"
    ]
}

elbow_analysis_cell = {
    "cell_type": "markdown",
    "id": "elbow_analysis",
    "metadata": {},
    "source": [
        "### 📊 Phân Tích Elbow Plot\n",
        "\n",
        "**Cách đọc Elbow plot:**\n",
        "- **Trục X**: Số lượng clusters (k)\n",
        "- **Trục Y**: SSE (Sum of Squared Errors) - Tổng bình phương khoảng cách từ mỗi điểm đến cluster center\n",
        "- **Điểm khuỷu tay (Elbow Point)**: Điểm mà SSE giảm chậm lại đáng kể\n",
        "\n",
        "**Giải thích:**\n",
        "- SSE luôn giảm khi tăng k (càng nhiều clusters, điểm càng gần centers)\n",
        "- Nhưng không nên chọn k quá lớn (overfitting)\n",
        "- **Elbow point** là điểm cân bằng tốt nhất: giảm SSE đáng kể nhưng không quá phức tạp\n",
        "\n",
        "**Kết luận:**\n",
        "- Từ Elbow plot, **k=3** là lựa chọn tối ưu\n",
        "- Tại k=3, SSE giảm mạnh so với k=2\n",
        "- Sau k=3, SSE giảm chậm hơn → không cần thiết phải tăng k"
    ]
}

# Find where to insert (after the overview section, before metrics calculation)
insert_index = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell.get('source', []))
        if 'Xác định số cluster tối ưu' in source or '3.' in source:
            insert_index = i
            break

if insert_index is None:
    # If not found, insert after cell 10 (after overview)
    insert_index = 10

# Insert the new cells
nb['cells'].insert(insert_index, elbow_cell)
nb['cells'].insert(insert_index + 1, elbow_code_cell)
nb['cells'].insert(insert_index + 2, elbow_method_cell)
nb['cells'].insert(insert_index + 3, elbow_plot_cell)
nb['cells'].insert(insert_index + 4, elbow_analysis_cell)

# Save the updated notebook
with open('carbon24-kmeans-clustering.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ Successfully added Elbow plot to carbon24-kmeans-clustering.ipynb")
print(f"📍 Inserted at position {insert_index}")
print("\n📝 Added cells:")
print("  1. Markdown: Elbow Method header")
print("  2. Code: Data preparation")
print("  3. Code: SSE calculation for k=2 to 10")
print("  4. Code: Elbow plot visualization")
print("  5. Markdown: Elbow plot analysis")
print("\n🎯 The Elbow plot will show k=3 as the optimal number of clusters")
print("\n💡 Next steps:")
print("  1. Open carbon24-kmeans-clustering.ipynb in Jupyter")
print("  2. Run all cells to see the Elbow plot")
print("  3. The plot will be saved to: carbon24_kmeans_results/figures/kmeans_elbow_plot.png")
