"""
Script kiểm tra dashboard đã sẵn sàng chưa
"""
import os
import sys

def check_file(path, description):
    """Kiểm tra file có tồn tại không"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_directory(path, description):
    """Kiểm tra thư mục có tồn tại không"""
    exists = os.path.isdir(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def main():
    print("=" * 70)
    print("🔍 KIỂM TRA DASHBOARD CARBON-24")
    print("=" * 70)
    
    all_ok = True
    
    # 1. Check main dashboard file
    print("\n📄 1. DASHBOARD FILE")
    all_ok &= check_file("carbon24_dashboard.py", "Dashboard file")
    
    # 2. Check data directories
    print("\n📁 2. DATA DIRECTORIES")
    all_ok &= check_directory("carbon24_kmeans_results", "K-means results")
    all_ok &= check_directory("carbon24_gmm_results", "GMM results")
    all_ok &= check_directory("carbon24_hierarchical_baseline", "Hierarchical results")
    all_ok &= check_directory("hdbscan_phuc", "HDBSCAN results")
    all_ok &= check_directory("carbon24_clustering_comparison_results", "Comparison results")
    
    # 3. Check K-means files
    print("\n📊 3. K-MEANS FILES")
    all_ok &= check_file("carbon24_kmeans_results/carbon24_clustered.csv", "K-means clustered data")
    all_ok &= check_file("carbon24_kmeans_results/clustering_report.json", "K-means report")
    
    # 4. Check GMM files
    print("\n🎲 4. GMM FILES")
    all_ok &= check_file("carbon24_gmm_results/results/carbon24_gmm_results.csv", "GMM results")
    all_ok &= check_file("carbon24_gmm_results/gmm_clustering_report.json", "GMM report")
    all_ok &= check_file("carbon24_gmm_results/tables/gmm_cluster_profile.csv", "GMM cluster profile")
    
    # 5. Check Hierarchical files
    print("\n🌳 5. HIERARCHICAL FILES")
    all_ok &= check_file("carbon24_hierarchical_baseline/results/carbon24_hierarchical_results.csv", "Hierarchical results")
    all_ok &= check_file("carbon24_hierarchical_baseline/tables/hierarchical_cluster_interpretation.csv", "Hierarchical interpretation")
    
    # 6. Check HDBSCAN files
    print("\n🔍 6. HDBSCAN FILES")
    all_ok &= check_file("hdbscan_phuc/hdbscan_results.csv", "HDBSCAN results")
    all_ok &= check_file("hdbscan_phuc/hdbscan_cluster_profile.csv", "HDBSCAN cluster profile")
    all_ok &= check_file("hdbscan_phuc/hdbscan_energy_summary.csv", "HDBSCAN energy summary")
    all_ok &= check_file("hdbscan_phuc/hdbscan_noise_outliers.csv", "HDBSCAN noise outliers")
    
    # 7. Check Comparison files
    print("\n📈 7. COMPARISON FILES")
    all_ok &= check_file("carbon24_clustering_comparison_results/methods_overview.csv", "Methods overview")
    all_ok &= check_file("carbon24_clustering_comparison_results/quality_metrics.csv", "Quality metrics")
    all_ok &= check_file("carbon24_clustering_comparison_results/method_ranking.csv", "Method ranking")
    
    # 8. Check dependencies
    print("\n📦 8. DEPENDENCIES")
    try:
        import streamlit
        print("✅ streamlit installed")
    except ImportError:
        print("❌ streamlit NOT installed")
        all_ok = False
    
    try:
        import plotly
        print("✅ plotly installed")
    except ImportError:
        print("❌ plotly NOT installed")
        all_ok = False
    
    try:
        import pandas
        print("✅ pandas installed")
    except ImportError:
        print("❌ pandas NOT installed")
        all_ok = False
    
    try:
        import numpy
        print("✅ numpy installed")
    except ImportError:
        print("❌ numpy NOT installed")
        all_ok = False
    
    try:
        from sklearn.preprocessing import MinMaxScaler
        print("✅ sklearn installed")
    except ImportError:
        print("❌ sklearn NOT installed")
        all_ok = False
    
    # 9. Check documentation
    print("\n📝 9. DOCUMENTATION")
    check_file("DASHBOARD_UPDATE_GUIDE.md", "Update guide")
    check_file("DASHBOARD_SUMMARY.md", "Summary")
    check_file("README_DASHBOARD_V2.md", "README")
    check_file("CAP_NHAT_DASHBOARD.md", "Cập nhật (Vietnamese)")
    
    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("✅ TẤT CẢ ĐỀU OK! DASHBOARD SẴN SÀNG!")
        print("\n🚀 Chạy dashboard:")
        print("   streamlit run carbon24_dashboard.py")
        print("\n📊 Dashboard có:")
        print("   - 9 trang (7 đã hoàn thành)")
        print("   - 4 thuật toán clustering")
        print("   - 25+ tabs")
        print("   - 50+ visualizations")
        return 0
    else:
        print("❌ CÓ VẤN ĐỀ! Kiểm tra các file bị thiếu ở trên.")
        print("\n💡 Giải pháp:")
        print("   1. Chạy các notebook để tạo kết quả:")
        print("      - carbon24-kmeans-clustering.ipynb")
        print("      - carbon24-gmm-clustering.ipynb")
        print("      - HDBSCAN.ipynb")
        print("      - carbon24-clustering-comparison-evaluation.ipynb")
        print("   2. Cài đặt dependencies:")
        print("      pip install -r requirements_dashboard.txt")
        return 1
    print("=" * 70)

if __name__ == "__main__":
    sys.exit(main())
