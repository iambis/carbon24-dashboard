"""
Script sửa lỗi tên cột trong dashboard
"""

# Đọc file dashboard
with open('carbon24_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: GMM metrics - calinski_harabasz_index -> calinski_harabasz_score
content = content.replace(
    "col3.metric(\"Calinski-Harabasz\", f\"{metrics['calinski_harabasz_index']:.2f}\")",
    "col3.metric(\"Calinski-Harabasz\", f\"{metrics['calinski_harabasz_score']:.2f}\")"
)

# Fix 2: Hierarchical cluster column - hierarchical_cluster -> cluster_hierarchical
content = content.replace("'hierarchical_cluster'", "'cluster_hierarchical'")
content = content.replace('"hierarchical_cluster"', '"cluster_hierarchical"')

# Fix 3: HDBSCAN cluster profile - không có cột 'Cluster', dùng 'hdbscan_cluster'
# Tìm và sửa dòng filter cluster profile
old_line = "profile_display = hdbscan_cluster_profile[hdbscan_cluster_profile['Cluster'] != -1]"
new_line = "profile_display = hdbscan_cluster_profile[hdbscan_cluster_profile['hdbscan_cluster'] != -1]"
content = content.replace(old_line, new_line)

# Fix 4: GMM report keys
# Sửa n_clusters -> optimal_n_components
content = content.replace(
    "col4.metric(\"Số Clusters\", gmm_report['n_clusters'])",
    "col4.metric(\"Số Clusters\", gmm_report['optimal_n_components'])"
)

# Ghi lại file
with open('carbon24_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Đã sửa các lỗi tên cột:")
print("   1. GMM: calinski_harabasz_index -> calinski_harabasz_score")
print("   2. Hierarchical: hierarchical_cluster -> cluster_hierarchical")
print("   3. HDBSCAN: Cluster -> hdbscan_cluster")
print("   4. GMM: n_clusters -> optimal_n_components")
print("\n🚀 Dashboard đã sẵn sàng!")
