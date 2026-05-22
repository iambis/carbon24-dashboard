"""
Quick test to verify all required files exist for the clustering comparison notebook
"""
import os
from pathlib import Path

print("=" * 80)
print("🔍 CHECKING CLUSTERING COMPARISON NOTEBOOK REQUIREMENTS")
print("=" * 80)

# Required files and folders
requirements = {
    "Notebook": [
        "carbon24-clustering-comparison-evaluation.ipynb"
    ],
    "K-means Results": [
        "carbon24_kmeans_results/carbon24_clustered.csv",
        "carbon24_kmeans_results/clustering_report.json"
    ],
    "GMM Results": [
        "carbon24_gmm_results/results"
    ],
    "Hierarchical Results": [
        "carbon24_hierarchical_baseline/results"
    ],
    "HDBSCAN Results": [
        "hdbscan_phuc/hdbscan_results.csv",
        "hdbscan_phuc/hdbscan_cluster_profile.csv"
    ],
    "Feature Data": [
        "carbon24_feature_selected/carbon24_feature_selected_standard.csv",
        "carbon24_feature_selected/selected_features.json"
    ],
    "Documentation": [
        "HUONG_DAN_CLUSTERING_COMPARISON.md",
        "CLUSTERING_COMPARISON_SUMMARY.md"
    ]
}

all_ready = True
missing_files = []

for category, files in requirements.items():
    print(f"\n📂 {category}:")
    print("-" * 60)
    
    for file_path in files:
        path = Path(file_path)
        
        if path.exists():
            if path.is_dir():
                # Check if directory has any CSV files
                csv_files = list(path.glob("*.csv"))
                if csv_files:
                    print(f"  ✅ {file_path} ({len(csv_files)} CSV files)")
                else:
                    print(f"  ⚠️  {file_path} (directory exists but no CSV files)")
                    all_ready = False
                    missing_files.append(f"{file_path} (no CSV files)")
            else:
                # Check file size
                size = path.stat().st_size
                size_mb = size / (1024 * 1024)
                if size_mb > 1:
                    print(f"  ✅ {file_path} ({size_mb:.2f} MB)")
                else:
                    print(f"  ✅ {file_path} ({size / 1024:.2f} KB)")
        else:
            print(f"  ❌ {file_path} (NOT FOUND)")
            all_ready = False
            missing_files.append(file_path)

print("\n" + "=" * 80)
if all_ready:
    print("✅ ALL REQUIREMENTS MET!")
    print("=" * 80)
    print("\n🚀 Ready to run the notebook:")
    print("   jupyter notebook carbon24-clustering-comparison-evaluation.ipynb")
    print("\n📝 Then: Cell → Run All")
else:
    print("⚠️  SOME REQUIREMENTS MISSING")
    print("=" * 80)
    print("\n❌ Missing files/folders:")
    for item in missing_files:
        print(f"   - {item}")
    print("\n💡 Please ensure all clustering notebooks have been run first:")
    print("   1. carbon24-kmeans-clustering.ipynb")
    print("   2. carbon24-gmm-clustering.ipynb")
    print("   3. carbon24-hierarchical-clustering.ipynb (or similar)")
    print("   4. HDBSCAN notebook (in hdbscan_phuc folder)")

print("\n" + "=" * 80)
