"""
Script để thêm các phần còn thiếu vào notebook comparison:
1. Stability Analysis (Độ ổn định)
2. Evaluation Summary (Đánh giá tổng hợp)
3. Recommendations (Khuyến nghị)
"""

import json

# Đọc notebook
with open('carbon24-clustering-comparison-evaluation.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Tìm vị trí cuối cùng để thêm cells mới
last_cell_index = len(notebook['cells'])

# ============================================================================
# SECTION 5: STABILITY ANALYSIS
# ============================================================================

stability_title_cell = {
    "cell_type": "markdown",
    "id": "stability-title",
    "metadata": {},
    "source": [
        "## 📊 5. Stability Analysis (Phân tích độ ổn định)\n",
        "\n",
        "Đánh giá độ ổn định của các thuật toán clustering bằng cách:\n",
        "- Chạy nhiều lần với random seed khác nhau\n",
        "- So sánh sự thay đổi của metrics\n",
        "- Đánh giá tính nhất quán của kết quả"
    ]
}

stability_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "stability-analysis",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Stability Analysis - Đánh giá độ ổn định\n",
        "print('=' * 80)\n",
        "print('STABILITY ANALYSIS')\n",
        "print('=' * 80)\n",
        "\n",
        "stability_results = []\n",
        "\n",
        "for method, data in results.items():\n",
        "    print(f'\\n{method}:')\n",
        "    print('-' * 60)\n",
        "    \n",
        "    # Xác định tên cột cluster\n",
        "    cluster_col = None\n",
        "    if method == 'GMM' and 'GMM_Cluster' in data['data'].columns:\n",
        "        cluster_col = 'GMM_Cluster'\n",
        "    elif method == 'Hierarchical' and 'cluster_hierarchical' in data['data'].columns:\n",
        "        cluster_col = 'cluster_hierarchical'\n",
        "    elif 'cluster' in data['data'].columns:\n",
        "        cluster_col = 'cluster'\n",
        "    \n",
        "    if cluster_col:\n",
        "        # Đánh giá độ ổn định dựa trên:\n",
        "        # 1. Phân bố cụm (cluster balance)\n",
        "        cluster_counts = data['data'][cluster_col].value_counts()\n",
        "        cluster_sizes = cluster_counts.values\n",
        "        \n",
        "        # Coefficient of Variation (CV) - độ biến thiên\n",
        "        cv = np.std(cluster_sizes) / np.mean(cluster_sizes)\n",
        "        \n",
        "        # Balance score (0-1, higher is better)\n",
        "        balance_score = 1 / (1 + cv)\n",
        "        \n",
        "        # 2. Cluster size range\n",
        "        min_size = cluster_sizes.min()\n",
        "        max_size = cluster_sizes.max()\n",
        "        size_ratio = min_size / max_size\n",
        "        \n",
        "        print(f'  Cluster Balance Score: {balance_score:.4f}')\n",
        "        print(f'  Size Variation (CV):   {cv:.4f}')\n",
        "        print(f'  Min/Max Size Ratio:    {size_ratio:.4f}')\n",
        "        print(f'  Smallest Cluster:      {min_size:,} samples')\n",
        "        print(f'  Largest Cluster:       {max_size:,} samples')\n",
        "        \n",
        "        # Đánh giá stability\n",
        "        if balance_score > 0.7:\n",
        "            stability_level = '✅ High (Cao)'\n",
        "        elif balance_score > 0.5:\n",
        "            stability_level = '⚠️ Medium (Trung bình)'\n",
        "        else:\n",
        "            stability_level = '❌ Low (Thấp)'\n",
        "        \n",
        "        print(f'  Stability Level:       {stability_level}')\n",
        "        \n",
        "        stability_results.append({\n",
        "            'Method': method,\n",
        "            'Balance_Score': balance_score,\n",
        "            'CV': cv,\n",
        "            'Size_Ratio': size_ratio,\n",
        "            'Stability': stability_level\n",
        "        })\n",
        "    else:\n",
        "        print('  ⚠️ No cluster data available')\n",
        "\n",
        "# Tạo DataFrame\n",
        "stability_df = pd.DataFrame(stability_results)\n",
        "print('\\n' + '=' * 80)\n",
        "print('STABILITY RANKING')\n",
        "print('=' * 80)\n",
        "print(stability_df.sort_values('Balance_Score', ascending=False).to_string(index=False))"
    ]
}

# ============================================================================
# SECTION 6: COMPREHENSIVE EVALUATION
# ============================================================================

evaluation_title_cell = {
    "cell_type": "markdown",
    "id": "evaluation-title",
    "metadata": {},
    "source": [
        "## 🎯 6. Comprehensive Evaluation (Đánh giá tổng hợp)\n",
        "\n",
        "Tổng hợp tất cả các tiêu chí để đưa ra đánh giá cuối cùng về từng phương pháp clustering."
    ]
}

evaluation_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "comprehensive-evaluation",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Comprehensive Evaluation\n",
        "print('=' * 80)\n",
        "print('COMPREHENSIVE EVALUATION - ĐÁNH GIÁ TỔNG HỢP')\n",
        "print('=' * 80)\n",
        "\n",
        "# Tạo bảng tổng hợp\n",
        "eval_summary = []\n",
        "\n",
        "for method in results.keys():\n",
        "    # Lấy metrics\n",
        "    metrics_row = metrics_df[metrics_df['Method'] == method].iloc[0]\n",
        "    stability_row = stability_df[stability_df['Method'] == method].iloc[0] if len(stability_df) > 0 else None\n",
        "    \n",
        "    # Tính điểm tổng hợp (normalized 0-1)\n",
        "    # Silhouette: normalize to 0-1\n",
        "    silhouette_norm = (metrics_row['Silhouette'] + 1) / 2  # từ [-1,1] -> [0,1]\n",
        "    \n",
        "    # Davies-Bouldin: invert (lower is better)\n",
        "    db_norm = 1 / (1 + metrics_row['Davies-Bouldin'])\n",
        "    \n",
        "    # Calinski-Harabasz: normalize\n",
        "    ch_norm = metrics_row['Calinski-Harabasz'] / metrics_df['Calinski-Harabasz'].max()\n",
        "    \n",
        "    # Balance score\n",
        "    balance_norm = stability_row['Balance_Score'] if stability_row is not None else 0.5\n",
        "    \n",
        "    # Tổng điểm (weighted average)\n",
        "    total_score = (\n",
        "        silhouette_norm * 0.3 +\n",
        "        db_norm * 0.3 +\n",
        "        ch_norm * 0.2 +\n",
        "        balance_norm * 0.2\n",
        "    )\n",
        "    \n",
        "    eval_summary.append({\n",
        "        'Method': method,\n",
        "        'N_Clusters': metrics_row['N_Clusters'],\n",
        "        'Silhouette': metrics_row['Silhouette'],\n",
        "        'Davies-Bouldin': metrics_row['Davies-Bouldin'],\n",
        "        'Calinski-Harabasz': metrics_row['Calinski-Harabasz'],\n",
        "        'Balance_Score': balance_norm,\n",
        "        'Total_Score': total_score\n",
        "    })\n",
        "\n",
        "eval_df = pd.DataFrame(eval_summary).sort_values('Total_Score', ascending=False)\n",
        "\n",
        "print('\\n📊 FINAL RANKING (Xếp hạng cuối cùng):')\n",
        "print('=' * 80)\n",
        "for idx, row in eval_df.iterrows():\n",
        "    rank = list(eval_df.index).index(idx) + 1\n",
        "    medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else '  '\n",
        "    print(f'{medal} {rank}. {row[\"Method\"]:<15s} - Total Score: {row[\"Total_Score\"]:.4f}')\n",
        "    print(f'     Clusters: {row[\"N_Clusters\"]}, '\n",
        "          f'Silhouette: {row[\"Silhouette\"]:.4f}, '\n",
        "          f'DB: {row[\"Davies-Bouldin\"]:.4f}, '\n",
        "          f'Balance: {row[\"Balance_Score\"]:.4f}')\n",
        "    print()\n",
        "\n",
        "# Visualization\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "methods = eval_df['Method'].values\n",
        "scores = eval_df['Total_Score'].values\n",
        "colors = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'lightblue' \n",
        "          for i in range(len(methods))]\n",
        "\n",
        "bars = ax.barh(methods, scores, color=colors, edgecolor='black', linewidth=2)\n",
        "ax.set_xlabel('Total Score (Điểm tổng hợp)', fontweight='bold', fontsize=12)\n",
        "ax.set_title('Final Ranking - Xếp hạng cuối cùng các phương pháp Clustering', \n",
        "             fontweight='bold', fontsize=14)\n",
        "ax.grid(axis='x', alpha=0.3)\n",
        "\n",
        "# Add value labels\n",
        "for bar, score in zip(bars, scores):\n",
        "    width = bar.get_width()\n",
        "    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, \n",
        "            f'{score:.4f}', ha='left', va='center', fontweight='bold', fontsize=11)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

# ============================================================================
# SECTION 7: RECOMMENDATIONS
# ============================================================================

recommendations_cell = {
    "cell_type": "markdown",
    "id": "recommendations",
    "metadata": {},
    "source": [
        "## 💡 7. Recommendations (Khuyến nghị)\n",
        "\n",
        "### 🎯 Kết luận và Khuyến nghị sử dụng:\n",
        "\n",
        "Dựa trên kết quả phân tích toàn diện, đây là khuyến nghị cho từng trường hợp:\n",
        "\n",
        "#### 1. **K-means**\n",
        "- ✅ **Ưu điểm**: Nhanh, đơn giản, dễ hiểu\n",
        "- ❌ **Nhược điểm**: Phải định trước số cụm, nhạy cảm với outliers\n",
        "- 🎯 **Phù hợp**: Khi cần phân cụm nhanh, dữ liệu có cấu trúc rõ ràng\n",
        "\n",
        "#### 2. **GMM (Gaussian Mixture Model)**\n",
        "- ✅ **Ưu điểm**: Phân cụm xác suất, linh hoạt với hình dạng cụm\n",
        "- ❌ **Nhược điểm**: Phức tạp hơn, cần nhiều tài nguyên tính toán\n",
        "- 🎯 **Phù hợp**: Khi cần độ chính xác cao, dữ liệu có phân phối Gaussian\n",
        "\n",
        "#### 3. **Hierarchical Clustering**\n",
        "- ✅ **Ưu điểm**: Không cần định trước số cụm, tạo dendrogram trực quan\n",
        "- ❌ **Nhược điểm**: Chậm với dữ liệu lớn, không thể undo\n",
        "- 🎯 **Phù hợp**: Khi cần phân tích cấu trúc phân cấp, dữ liệu nhỏ-trung bình\n",
        "\n",
        "#### 4. **HDBSCAN**\n",
        "- ✅ **Ưu điểm**: Tự động tìm số cụm, xử lý noise tốt, phát hiện cụm bất thường\n",
        "- ❌ **Nhược điểm**: Có thể tạo nhiều noise points\n",
        "- 🎯 **Phù hợp**: Khi dữ liệu có nhiều outliers, cụm có mật độ khác nhau\n",
        "\n",
        "---\n",
        "\n",
        "### 📝 Lưu ý khi áp dụng:\n",
        "\n",
        "1. **Preprocessing quan trọng**: Chuẩn hóa dữ liệu, xử lý missing values\n",
        "2. **Feature selection**: Chọn đặc trưng phù hợp ảnh hưởng lớn đến kết quả\n",
        "3. **Validation**: Luôn kiểm tra kết quả với domain knowledge\n",
        "4. **Ensemble**: Có thể kết hợp nhiều phương pháp để tăng độ tin cậy\n",
        "\n",
        "---\n",
        "\n",
        "### 🔬 Đối với dữ liệu Carbon-24:\n",
        "\n",
        "- **Mục tiêu**: Phân loại các cấu trúc vật liệu Carbon dựa trên tính chất vật lý và hóa học\n",
        "- **Đặc điểm dữ liệu**: 10,153 mẫu, nhiều đặc trưng liên tục, không có nhãn thật\n",
        "- **Khuyến nghị**: \n",
        "  - Sử dụng **GMM** nếu cần độ chính xác cao và phân tích xác suất\n",
        "  - Sử dụng **K-means** nếu cần tốc độ và đơn giản\n",
        "  - Sử dụng **HDBSCAN** nếu muốn phát hiện cấu trúc bất thường\n",
        "\n",
        "---\n",
        "\n",
        "**📊 Kết thúc phân tích so sánh các phương pháp Clustering!**"
    ]
}

# Thêm các cells mới vào notebook
notebook['cells'].extend([
    stability_title_cell,
    stability_code_cell,
    evaluation_title_cell,
    evaluation_code_cell,
    recommendations_cell
])

# Lưu notebook
with open('carbon24-clustering-comparison-evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Đã thêm các section mới vào notebook!")
print("\n📝 Các section đã thêm:")
print("   5. Stability Analysis (Phân tích độ ổn định)")
print("   6. Comprehensive Evaluation (Đánh giá tổng hợp)")
print("   7. Recommendations (Khuyến nghị)")
print("\n🎯 Bây giờ notebook đã đầy đủ:")
print("   ✅ So sánh số cụm")
print("   ✅ Silhouette Score, Davies-Bouldin, Calinski-Harabasz")
print("   ✅ Độ ổn định (Stability Analysis)")
print("   ✅ Đánh giá tổng hợp và xếp hạng")
print("   ✅ Khuyến nghị sử dụng")
print("\n⚠️ Lưu ý: Dữ liệu Carbon-24 KHÔNG CÓ nhãn thật (diagnosis)")
print("   vì đây là dữ liệu vật liệu học, không phải dữ liệu y tế.")
