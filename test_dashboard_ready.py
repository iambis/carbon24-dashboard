"""
Test script to verify dashboard is ready for demo
Run this before your demo to check everything is working
"""

import os
import sys

def test_file_exists(filepath, description):
    """Test if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def test_import(module_name):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {module_name} installed")
        return True
    except ImportError:
        print(f"❌ {module_name} NOT installed")
        return False

def test_data_file(filepath):
    """Test if data file can be loaded"""
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        print(f"✅ {filepath}: {len(df)} rows, {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"❌ {filepath}: Error - {e}")
        return False

def main():
    print("=" * 70)
    print("🎯 CARBON-24 DASHBOARD READINESS TEST")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Test 1: Check Python packages
    print("📦 Testing Python Packages...")
    print("-" * 70)
    all_passed &= test_import("streamlit")
    all_passed &= test_import("plotly")
    all_passed &= test_import("pandas")
    all_passed &= test_import("numpy")
    print()
    
    # Test 2: Check main files
    print("📁 Testing Main Files...")
    print("-" * 70)
    all_passed &= test_file_exists("carbon24_dashboard.py", "Dashboard script")
    all_passed &= test_file_exists("run_dashboard.bat", "Run script")
    all_passed &= test_file_exists("README_DASHBOARD.md", "Dashboard README")
    all_passed &= test_file_exists("HUONG_DAN_DEMO.md", "Demo guide")
    all_passed &= test_file_exists("DEMO_QUICK_REFERENCE.md", "Quick reference")
    all_passed &= test_file_exists("DEMO_METRICS_ACTUAL.md", "Actual metrics")
    print()
    
    # Test 3: Check data files
    print("💾 Testing Data Files...")
    print("-" * 70)
    all_passed &= test_file_exists(
        "carbon24_preprocessing_results/selected_features.json",
        "Feature info"
    )
    all_passed &= test_file_exists(
        "carbon24_kmeans_results/clustering_report.json",
        "Clustering report"
    )
    print()
    
    # Test 4: Load and check data
    print("📊 Testing Data Loading...")
    print("-" * 70)
    
    # Check clustered data
    clustered_path = "carbon24_kmeans_results/carbon24_clustered.csv"
    if os.path.exists(clustered_path):
        try:
            import pandas as pd
            df = pd.read_csv(clustered_path)
            
            print(f"✅ Clustered data loaded: {len(df)} samples")
            
            # Check required columns
            required_cols = ['cluster', 'pca1', 'pca2']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                all_passed = False
            else:
                print(f"✅ Required columns present: {required_cols}")
            
            # Check 3D PCA columns
            pca_3d_cols = ['pca1_3d', 'pca2_3d', 'pca3_3d']
            has_3d = all(col in df.columns for col in pca_3d_cols)
            
            if has_3d:
                print(f"✅ PCA 3D columns present: {pca_3d_cols}")
            else:
                print(f"⚠️  PCA 3D columns missing (run: python add_pca_3d_to_notebook.py)")
            
            # Check cluster info
            n_clusters = df['cluster'].nunique()
            print(f"✅ Number of clusters: {n_clusters}")
            
            cluster_sizes = df['cluster'].value_counts().sort_index()
            for cluster_id, size in cluster_sizes.items():
                pct = size / len(df) * 100
                print(f"   Cluster {cluster_id}: {size:,} samples ({pct:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error loading clustered data: {e}")
            all_passed = False
    else:
        print(f"❌ Clustered data not found: {clustered_path}")
        all_passed = False
    
    print()
    
    # Test 5: Check HTML files
    print("🎨 Testing HTML Visualization Files...")
    print("-" * 70)
    html_files = [
        "carbon24_kmeans_results/pca_3d_clusters.html",
        "carbon24_kmeans_results/pca_3d_energy.html",
        "carbon24_kmeans_results/pca_3d_crystal_systems.html"
    ]
    
    for html_file in html_files:
        exists = test_file_exists(html_file, "3D visualization")
        if not exists:
            print(f"   ⚠️  Run: python add_pca_3d_to_notebook.py")
    
    print()
    
    # Test 6: Check clustering metrics
    print("📈 Testing Clustering Metrics...")
    print("-" * 70)
    
    report_path = "carbon24_kmeans_results/clustering_report.json"
    if os.path.exists(report_path):
        try:
            import json
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            print(f"✅ Clustering report loaded")
            print(f"   Optimal k: {report['optimal_k']}")
            print(f"   Samples: {report['n_samples']:,}")
            print(f"   Features: {report['n_features']}")
            
            metrics = report['metrics']
            print(f"   Silhouette: {metrics['silhouette_score']:.4f}")
            print(f"   Davies-Bouldin: {metrics['davies_bouldin_index']:.4f}")
            print(f"   Calinski-Harabasz: {metrics['calinski_harabasz_index']:.2f}")
            
            pca = report.get('pca_variance_explained', {})
            if pca:
                print(f"   PCA 2D variance: {pca['total']*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error loading clustering report: {e}")
            all_passed = False
    else:
        print(f"❌ Clustering report not found: {report_path}")
        all_passed = False
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("✅ ALL TESTS PASSED! Dashboard is ready for demo! 🎉")
        print()
        print("🚀 To start dashboard, run:")
        print("   streamlit run carbon24_dashboard.py")
        print()
        print("📚 Before demo, read:")
        print("   - HUONG_DAN_DEMO.md (full guide)")
        print("   - DEMO_QUICK_REFERENCE.md (quick reference)")
        print("   - DEMO_METRICS_ACTUAL.md (actual numbers)")
        print()
        return 0
    else:
        print("❌ SOME TESTS FAILED. Please fix issues before demo.")
        print()
        print("Common fixes:")
        print("   - Install packages: pip install streamlit plotly")
        print("   - Run preprocessing: carbon24-preprocessing.ipynb")
        print("   - Run clustering: carbon24-kmeans-clustering.ipynb")
        print("   - Generate 3D: python add_pca_3d_to_notebook.py")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
