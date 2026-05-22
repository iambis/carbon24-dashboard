"""
Setup script để chuẩn bị dữ liệu cho K-means clustering
"""

import os
import shutil
import json

print("="*80)
print("SETUP K-MEANS CLUSTERING DATA")
print("="*80)

# Tạo thư mục nếu chưa có
os.makedirs('carbon24_preprocessing_results', exist_ok=True)

# Copy files từ feature_selected sang preprocessing_results
source_dir = 'carbon24_feature_selected'
target_dir = 'carbon24_preprocessing_results'

files_to_copy = [
    'selected_features.json',
    'carbon24_feature_selected.csv',
    'carbon24_feature_selected_standard.csv',
    'carbon24_feature_selected_robust.csv'
]

print("\nCopying files...")
for file in files_to_copy:
    source = os.path.join(source_dir, file)
    target = os.path.join(target_dir, file)
    
    if os.path.exists(source):
        shutil.copy2(source, target)
        print(f"✓ Copied: {file}")
    else:
        print(f"⚠️  Not found: {file}")

# Verify
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

json_path = os.path.join(target_dir, 'selected_features.json')
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        feature_info = json.load(f)
    
    print(f"✓ JSON file loaded successfully")
    print(f"  - Total features: {len(feature_info['all_features'])}")
    print(f"  - Numeric features: {len(feature_info['numeric_features'])}")
    print(f"  - Categorical features: {len(feature_info['categorical_features'])}")
    print(f"  - Dropped features: {len(feature_info['dropped_features'])}")
else:
    print("❌ JSON file not found!")

# Check CSV files
csv_files = [
    'carbon24_feature_selected_standard.csv',
    'carbon24_feature_selected_robust.csv',
    'carbon24_feature_selected.csv'
]

print("\nCSV files:")
for csv_file in csv_files:
    csv_path = os.path.join(target_dir, csv_file)
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"✓ {csv_file}: {df.shape}")
    else:
        print(f"❌ {csv_file}: Not found")

print("\n" + "="*80)
print("✅ SETUP COMPLETED!")
print("="*80)
print("\nYou can now run the K-means clustering notebook:")
print("  - carbon24-kmeans-clustering.ipynb")
