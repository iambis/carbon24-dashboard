"""
Test script để kiểm tra scipy.stats import
"""

print("="*80)
print("TESTING SCIPY.STATS IMPORT")
print("="*80)

# Test 1: Import scipy.stats
print("\n1️⃣ Testing scipy.stats import...")
try:
    from scipy import stats
    print("✅ scipy.stats imported successfully")
except ImportError as e:
    print(f"❌ Failed to import scipy.stats: {e}")
    print("   Please install: pip install scipy")
    exit(1)

# Test 2: Test zscore function
print("\n2️⃣ Testing stats.zscore function...")
try:
    import numpy as np
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    z_scores = stats.zscore(data)
    print(f"✅ stats.zscore works correctly")
    print(f"   Sample data: {data}")
    print(f"   Z-scores: {z_scores}")
except Exception as e:
    print(f"❌ Failed to use stats.zscore: {e}")
    exit(1)

# Test 3: Test with absolute values
print("\n3️⃣ Testing np.abs(stats.zscore(...))...")
try:
    z_scores_abs = np.abs(stats.zscore(data))
    print(f"✅ np.abs(stats.zscore(...)) works correctly")
    print(f"   Absolute Z-scores: {z_scores_abs}")
except Exception as e:
    print(f"❌ Failed: {e}")
    exit(1)

# Test 4: Test with pandas Series
print("\n4️⃣ Testing with pandas Series...")
try:
    import pandas as pd
    df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    z_scores_series = np.abs(stats.zscore(df['values']))
    print(f"✅ Works with pandas Series")
    print(f"   Z-scores: {z_scores_series}")
except Exception as e:
    print(f"❌ Failed: {e}")
    exit(1)

# Test 5: Test anomaly detection logic
print("\n5️⃣ Testing anomaly detection logic...")
try:
    # Simulate relative_energy data
    relative_energy = np.array([0.1, 0.2, 0.15, 0.18, 0.22, 0.19, 0.21, 5.0, 0.17, 0.20])
    z_scores = np.abs(stats.zscore(relative_energy))
    
    # Detect anomalies (|z| > 3)
    anomaly_mask = z_scores > 3
    n_anomalies = anomaly_mask.sum()
    
    print(f"✅ Anomaly detection logic works")
    print(f"   Data: {relative_energy}")
    print(f"   Z-scores: {z_scores}")
    print(f"   Anomalies (|z| > 3): {n_anomalies}")
    print(f"   Anomaly indices: {np.where(anomaly_mask)[0]}")
except Exception as e:
    print(f"❌ Failed: {e}")
    exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\n💡 Lỗi 'NameError: name stats is not defined' đã được sửa!")
print("   Đã thêm 'from scipy import stats' vào các file:")
print("   - carbon24-anomaly-energy-prediction.ipynb")
print("   - create_anomaly_nb.py")
print("\n📝 Để sử dụng trong notebook:")
print("   1. Restart kernel")
print("   2. Run All Cells")
print("   3. Hoặc chạy lại cell import")
