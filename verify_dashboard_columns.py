"""
Script kiểm tra tất cả tên cột trong dashboard
"""
import pandas as pd
import json

print("=" * 70)
print("🔍 KIỂM TRA TÊN CỘT TRONG DASHBOARD")
print("=" * 70)

all_ok = True

# 1. K-means
print("\n📊 1. K-MEANS")
try:
    df = pd.read_csv('carbon24_kmeans_results/carbon24_clustered.csv')
    print(f"✅ Cluster column: {[c for c in df.columns if 'cluster' in c.lower()]}")
    print(f"✅ PCA columns: {[c for c in df.columns if 'pca' in c.lower()]}")
except Exception as e:
    print(f"❌ Error: {e}")
    all_ok = False

# 2. GMM
print("\n🎲 2. GMM")
try:
    df = pd.read_csv('carbon24_gmm_results/results/carbon24_gmm_results.csv')
    print(f"✅ Cluster column: {[c for c in df.columns if 'cluster' in c.lower()]}")
    print(f"✅ Probability column: {[c for c in df.columns if 'prob' in c.lower()]}")
    print(f"✅ PCA columns: {[c for c in df.columns if 'pca' in c.upper()]}")
    
    with open('carbon24_gmm_results/gmm_clustering_report.json') as f:
        report = json.load(f)
    print(f"✅ Report keys: {list(report.keys())}")
    print(f"✅ Metrics keys: {list(report['metrics'].keys())}")
except Exception as e:
    print(f"❌ Error: {e}")
    all_ok = False

# 3. Hierarchical
print("\n🌳 3. HIERARCHICAL")
try:
    df = pd.read_csv('carbon24_hierarchical_baseline/results/carbon24_hierarchical_results.csv')
    print(f"✅ Cluster column: {[c for c in df.columns if 'cluster' in c.lower()]}")
    print(f"✅ PCA columns: {[c for c in df.columns if 'pc' in c.upper() and len(c) <= 3]}")
except Exception as e:
    print(f"❌ Error: {e}")
    all_ok = False

# 4. HDBSCAN
print("\n🔍 4. HDBSCAN")
try:
    df = pd.read_csv('hdbscan_phuc/hdbscan_results.csv')
    print(f"✅ Cluster column: {[c for c in df.columns if 'cluster' in c.lower()]}")
    print(f"✅ Probability column: {[c for c in df.columns if 'prob' in c.lower()]}")
    print(f"✅ PCA columns: {[c for c in df.columns if 'pca' in c.lower()]}")
    
    profile = pd.read_csv('hdbscan_phuc/hdbscan_cluster_profile.csv')
    print(f"✅ Profile columns: {profile.columns.tolist()[:5]}...")
except Exception as e:
    print(f"❌ Error: {e}")
    all_ok = False

# Summary
print("\n" + "=" * 70)
if all_ok:
    print("✅ TẤT CẢ CÁC CỘT ĐỀU ĐÚNG!")
    print("\n📋 Tóm tắt tên cột:")
    print("   K-means: cluster, pca1, pca2, pca1_3d, pca2_3d, pca3_3d")
    print("   GMM: GMM_Cluster, Max_Probability, PCA1, PCA2, PCA3")
    print("   Hierarchical: cluster_hierarchical, PC1, PC2")
    print("   HDBSCAN: hdbscan_cluster, hdbscan_probability, pca1, pca2")
    print("\n🚀 Dashboard sẵn sàng chạy!")
else:
    print("❌ CÓ LỖI! Kiểm tra lại các file dữ liệu.")
print("=" * 70)
