"""
Test script cho Anomaly Detection Module
Kiểm tra xem module có chạy được không và output có đúng format không
"""

from pathlib import Path
import pandas as pd

def test_anomaly_detection():
    """Test anomaly detection module"""
    
    print("="*80)
    print("TESTING ANOMALY DETECTION MODULE")
    print("="*80)
    
    # 1. Check input files
    print("\n1️⃣ CHECKING INPUT FILES...")
    
    hdbscan_results = Path("hdbscan_phuc/hdbscan_results.csv")
    if not hdbscan_results.exists():
        print(f"❌ FAILED: {hdbscan_results} not found")
        print("   Please run HDBSCAN notebook first!")
        return False
    else:
        print(f"✅ Found: {hdbscan_results}")
        df = pd.read_csv(hdbscan_results)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()[:10]}...")
    
    # 2. Check if module can be imported
    print("\n2️⃣ CHECKING MODULE IMPORT...")
    try:
        import carbon24_anomaly_detection
        print("✅ Module imported successfully")
    except Exception as e:
        print(f"❌ FAILED to import module: {e}")
        return False
    
    # 3. Run the module
    print("\n3️⃣ RUNNING ANOMALY DETECTION...")
    try:
        carbon24_anomaly_detection.main()
        print("✅ Module executed successfully")
    except Exception as e:
        print(f"❌ FAILED to execute module: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Check output files
    print("\n4️⃣ CHECKING OUTPUT FILES...")
    
    output_dir = Path("carbon24_anomaly_detection")
    expected_files = [
        "anomaly_detection_results.csv",
        "anomaly_summary.csv",
        "anomaly_method_comparison.csv",
        "anomaly_details.csv",
    ]
    
    all_files_ok = True
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"✅ {filename}: {df.shape}")
        else:
            print(f"❌ {filename}: NOT FOUND")
            all_files_ok = False
    
    # 5. Check figures
    print("\n5️⃣ CHECKING FIGURES...")
    
    figure_dir = output_dir / "figures"
    expected_figures = [
        "anomaly_methods_comparison.png",
        "anomaly_vote_distribution.png",
        "anomaly_energy_distribution.png",
        "isolation_forest_score_distribution.png",
    ]
    
    all_figures_ok = True
    for filename in expected_figures:
        filepath = figure_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"✅ {filename}: {size_kb:.1f} KB")
        else:
            print(f"❌ {filename}: NOT FOUND")
            all_figures_ok = False
    
    # 6. Validate results
    print("\n6️⃣ VALIDATING RESULTS...")
    
    results_path = output_dir / "anomaly_detection_results.csv"
    if results_path.exists():
        results_df = pd.read_csv(results_path)
        
        # Check required columns
        required_cols = [
            'is_hdbscan_noise',
            'is_low_probability',
            'is_isolation_forest_anomaly',
            'anomaly_vote_count',
            'is_anomaly_consensus',
            'is_anomaly_any',
            'is_anomaly_all'
        ]
        
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            all_files_ok = False
        else:
            print(f"✅ All required columns present")
        
        # Check vote count range
        vote_counts = results_df['anomaly_vote_count'].unique()
        if set(vote_counts).issubset({0, 1, 2, 3}):
            print(f"✅ Vote count valid: {sorted(vote_counts)}")
        else:
            print(f"❌ Invalid vote count: {sorted(vote_counts)}")
            all_files_ok = False
        
        # Print summary
        print(f"\n📊 SUMMARY:")
        print(f"   Total samples: {len(results_df):,}")
        print(f"   HDBSCAN Noise: {results_df['is_hdbscan_noise'].sum():,} ({results_df['is_hdbscan_noise'].mean():.2%})")
        print(f"   Low Probability: {results_df['is_low_probability'].sum():,} ({results_df['is_low_probability'].mean():.2%})")
        print(f"   Isolation Forest: {results_df['is_isolation_forest_anomaly'].sum():,} ({results_df['is_isolation_forest_anomaly'].mean():.2%})")
        print(f"   Consensus (≥2): {results_df['is_anomaly_consensus'].sum():,} ({results_df['is_anomaly_consensus'].mean():.2%})")
        print(f"   All 3 methods: {results_df['is_anomaly_all'].sum():,} ({results_df['is_anomaly_all'].mean():.2%})")
    
    # Final result
    print("\n" + "="*80)
    if all_files_ok and all_figures_ok:
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        return False


if __name__ == "__main__":
    success = test_anomaly_detection()
    exit(0 if success else 1)
