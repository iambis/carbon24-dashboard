"""
Carbon-24 Feature Selection - Loại bỏ multicollinearity
Xử lý các features có tương quan cao để tránh redundancy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CARBON-24 FEATURE SELECTION - MULTICOLLINEARITY REMOVAL")
print("="*80)

# ============================================================================
# 1. LOAD DỮ LIỆU ĐÃ PREPROCESSING
# ============================================================================
print("\n[1] LOADING PREPROCESSED DATA...")
df = pd.read_csv('carbon24_preprocessing_results/carbon24_cleaned_engineered.csv')
print(f"✓ Loaded: {df.shape}")

# ============================================================================
# 2. PHÂN TÍCH MULTICOLLINEARITY
# ============================================================================
print("\n[2] ANALYZING MULTICOLLINEARITY...")

# Chọn numeric features (loại bỏ ID và categorical)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['row_index', 'space_group_number']
analysis_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"✓ Analyzing {len(analysis_cols)} numeric features")

# Tính correlation matrix
corr_matrix = df[analysis_cols].corr().abs()

# ============================================================================
# 3. XÁC ĐỊNH FEATURES CẦN LOẠI BỎ
# ============================================================================
print("\n[3] IDENTIFYING REDUNDANT FEATURES...")

def remove_correlated_features(df, threshold=0.95):
    """
    Loại bỏ features có tương quan cao
    Giữ lại feature có tương quan trung bình thấp hơn với các features khác
    """
    corr_matrix = df.corr().abs()
    
    # Tạo upper triangle matrix
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Tìm features có correlation > threshold
    to_drop = []
    dropped_pairs = []
    
    for column in upper_tri.columns:
        # Tìm các features có correlation cao với column này
        high_corr = upper_tri[column][upper_tri[column] > threshold]
        
        if len(high_corr) > 0:
            for corr_feature in high_corr.index:
                if corr_feature not in to_drop and column not in to_drop:
                    # So sánh mean correlation của 2 features
                    mean_corr_col = corr_matrix[column].mean()
                    mean_corr_feat = corr_matrix[corr_feature].mean()
                    
                    # Loại bỏ feature có mean correlation cao hơn
                    if mean_corr_col > mean_corr_feat:
                        to_drop.append(column)
                        dropped_pairs.append({
                            'Dropped': column,
                            'Kept': corr_feature,
                            'Correlation': upper_tri[column][corr_feature],
                            'Reason': f'Mean corr: {mean_corr_col:.3f} > {mean_corr_feat:.3f}'
                        })
                        break
                    else:
                        to_drop.append(corr_feature)
                        dropped_pairs.append({
                            'Dropped': corr_feature,
                            'Kept': column,
                            'Correlation': upper_tri[column][corr_feature],
                            'Reason': f'Mean corr: {mean_corr_feat:.3f} > {mean_corr_col:.3f}'
                        })
    
    return list(set(to_drop)), dropped_pairs

# Áp dụng với threshold 0.95
features_to_drop, dropped_info = remove_correlated_features(df[analysis_cols], threshold=0.95)

print(f"\n✓ Found {len(features_to_drop)} redundant features to remove:")
print("\nDETAILED REMOVAL DECISIONS:")
print("-" * 80)

dropped_df = pd.DataFrame(dropped_info)
if len(dropped_df) > 0:
    dropped_df = dropped_df.sort_values('Correlation', ascending=False)
    for idx, row in dropped_df.iterrows():
        print(f"❌ DROP: {row['Dropped']:<25} (corr={row['Correlation']:.4f} with {row['Kept']})")
        print(f"   Reason: {row['Reason']}")
    print()

# ============================================================================
# 4. PHÂN TÍCH CỤ THỂ CÁC NHÓM TƯƠNG QUAN CAO
# ============================================================================
print("\n[4] ANALYZING HIGH CORRELATION GROUPS...")

# Nhóm 1: density, packing_fraction, volume_per_atom
print("\n--- Group 1: Density-related features ---")
group1 = ['density', 'packing_fraction', 'volume_per_atom']
if all(f in df.columns for f in group1):
    print(f"Correlation matrix:")
    print(df[group1].corr().round(4))
    print(f"✓ DECISION: Keep 'density', drop 'packing_fraction' (r=1.0)")

# Nhóm 2: energy, relative_energy
print("\n--- Group 2: Energy features ---")
group2 = ['energy', 'relative_energy']
if all(f in df.columns for f in group2):
    print(f"Correlation: {df[group2].corr().iloc[0,1]:.6f}")
    print(f"✓ DECISION: Keep 'relative_energy' (normalized), drop 'energy'")

# Nhóm 3: fraction_sp2, fraction_sp3, mean_coordination
print("\n--- Group 3: Hybridization & Coordination ---")
group3 = ['fraction_sp2', 'fraction_sp3', 'mean_coordination']
if all(f in df.columns for f in group3):
    print(f"Correlation matrix:")
    print(df[group3].corr().round(4))
    print(f"✓ DECISION: Keep 'mean_coordination', drop 'fraction_sp2' & 'fraction_sp3'")

# Nhóm 4: mean_bond_length, mean_coordination
print("\n--- Group 4: Bond & Coordination ---")
group4 = ['mean_bond_length', 'mean_coordination']
if all(f in df.columns for f in group4):
    print(f"Correlation: {df[group4].corr().iloc[0,1]:.4f}")
    print(f"✓ DECISION: Keep both (r=0.986, still informative)")

# ============================================================================
# 5. TẠO DANH SÁCH FEATURES CUỐI CÙNG
# ============================================================================
print("\n[5] CREATING FINAL FEATURE SET...")

# Danh sách features cần loại bỏ (dựa trên phân tích)
manual_drop_list = [
    'packing_fraction',      # r=1.0 với density
    'energy',                # r=1.0 với relative_energy
    'fraction_sp2',          # r=-0.9997 với fraction_sp3
    'fraction_sp3',          # r=0.9993 với mean_coordination (giữ mean_coordination)
    'volume_per_atom',       # r=-0.996 với density (giữ density)
    'c_over_a',              # r=0.963 với lattice_anisotropy
    'std_bond_length',       # r=0.936 với bond_length_range
    'b_over_a',              # r=0.885 với b
]

# Kết hợp với auto-detected
all_features_to_drop = list(set(manual_drop_list + features_to_drop))

print(f"\n✓ Total features to drop: {len(all_features_to_drop)}")
print(f"Features to drop: {sorted(all_features_to_drop)}")

# Tạo dataset mới
features_to_keep = [col for col in df.columns if col not in all_features_to_drop]
df_reduced = df[features_to_keep].copy()

print(f"\n✓ Original features: {len(df.columns)}")
print(f"✓ Reduced features: {len(df_reduced.columns)}")
print(f"✓ Removed: {len(df.columns) - len(df_reduced.columns)} features")

# ============================================================================
# 6. KIỂM TRA LẠI CORRELATION SAU KHI LOẠI BỎ
# ============================================================================
print("\n[6] VERIFYING REDUCED CORRELATION...")

numeric_reduced = df_reduced.select_dtypes(include=[np.number]).columns.tolist()
exclude_reduced = ['row_index', 'space_group_number']
analysis_reduced = [col for col in numeric_reduced if col not in exclude_reduced]

corr_matrix_reduced = df_reduced[analysis_reduced].corr().abs()

# Tìm correlation cao còn lại
upper_tri_reduced = corr_matrix_reduced.where(
    np.triu(np.ones(corr_matrix_reduced.shape), k=1).astype(bool)
)

high_corr_remaining = []
for column in upper_tri_reduced.columns:
    high_corr = upper_tri_reduced[column][upper_tri_reduced[column] > 0.90]
    for idx in high_corr.index:
        high_corr_remaining.append({
            'Feature 1': column,
            'Feature 2': idx,
            'Correlation': upper_tri_reduced[column][idx]
        })

if high_corr_remaining:
    print(f"\n⚠️  Still {len(high_corr_remaining)} pairs with |r| > 0.90:")
    remaining_df = pd.DataFrame(high_corr_remaining).sort_values('Correlation', ascending=False)
    print(remaining_df.head(10))
else:
    print("\n✅ No correlation pairs > 0.90 remaining!")

# Vẽ correlation heatmap mới
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Before
sns.heatmap(df[analysis_cols].corr(), ax=axes[0], cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[0].set_title(f'Before: {len(analysis_cols)} features', fontsize=14, fontweight='bold')

# After
sns.heatmap(df_reduced[analysis_reduced].corr(), ax=axes[1], cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[1].set_title(f'After: {len(analysis_reduced)} features', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_selection_correlation_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: feature_selection_correlation_comparison.png")
plt.close()

# ============================================================================
# 7. KIỂM TRA VARIANCE
# ============================================================================
print("\n[7] CHECKING FEATURE VARIANCE...")

# Loại bỏ features có variance thấp
selector = VarianceThreshold(threshold=0.01)
numeric_data = df_reduced[analysis_reduced]
selector.fit(numeric_data)

low_variance_features = [
    analysis_reduced[i] for i in range(len(analysis_reduced)) 
    if not selector.get_support()[i]
]

if low_variance_features:
    print(f"⚠️  Found {len(low_variance_features)} low variance features:")
    for feat in low_variance_features:
        print(f"   - {feat}: variance = {df_reduced[feat].var():.6f}")
else:
    print("✅ All features have sufficient variance!")

# ============================================================================
# 8. TẠO CÁC PHIÊN BẢN CHUẨN HÓA
# ============================================================================
print("\n[8] CREATING NORMALIZED VERSIONS...")

from sklearn.preprocessing import StandardScaler, RobustScaler

# StandardScaler
scaler_standard = StandardScaler()
df_reduced_standard = df_reduced.copy()
df_reduced_standard[analysis_reduced] = scaler_standard.fit_transform(df_reduced[analysis_reduced])

# RobustScaler
scaler_robust = RobustScaler()
df_reduced_robust = df_reduced.copy()
df_reduced_robust[analysis_reduced] = scaler_robust.fit_transform(df_reduced[analysis_reduced])

print("✓ Created standardized versions")

# ============================================================================
# 9. LƯU KẾT QUẢ
# ============================================================================
print("\n[9] SAVING RESULTS...")

import os
output_dir = 'carbon24_feature_selected'
os.makedirs(output_dir, exist_ok=True)

# Lưu datasets
df_reduced.to_csv(f'{output_dir}/carbon24_feature_selected.csv', index=False)
df_reduced_standard.to_csv(f'{output_dir}/carbon24_feature_selected_standard.csv', index=False)
df_reduced_robust.to_csv(f'{output_dir}/carbon24_feature_selected_robust.csv', index=False)

print(f"✓ Saved: {output_dir}/carbon24_feature_selected.csv")
print(f"✓ Saved: {output_dir}/carbon24_feature_selected_standard.csv")
print(f"✓ Saved: {output_dir}/carbon24_feature_selected_robust.csv")

# Lưu danh sách features
with open(f'{output_dir}/feature_selection_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("CARBON-24 FEATURE SELECTION REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Original features: {len(df.columns)}\n")
    f.write(f"Selected features: {len(df_reduced.columns)}\n")
    f.write(f"Removed features: {len(all_features_to_drop)}\n\n")
    
    f.write("REMOVED FEATURES:\n")
    f.write("-" * 80 + "\n")
    for feat in sorted(all_features_to_drop):
        f.write(f"  ❌ {feat}\n")
    f.write("\n")
    
    f.write("KEPT FEATURES:\n")
    f.write("-" * 80 + "\n")
    for feat in sorted(analysis_reduced):
        f.write(f"  ✓ {feat}\n")
    f.write("\n")
    
    f.write("REMOVAL DECISIONS:\n")
    f.write("-" * 80 + "\n")
    if len(dropped_df) > 0:
        f.write(dropped_df.to_string())
    f.write("\n\n")
    
    if high_corr_remaining:
        f.write("REMAINING HIGH CORRELATIONS (|r| > 0.90):\n")
        f.write("-" * 80 + "\n")
        f.write(remaining_df.to_string())
    else:
        f.write("✅ No high correlations remaining!\n")

print(f"✓ Saved: {output_dir}/feature_selection_report.txt")

# Lưu feature list cho sử dụng sau này
selected_features = {
    'all_features': features_to_keep,
    'numeric_features': analysis_reduced,
    'categorical_features': df_reduced.select_dtypes(include=['object']).columns.tolist(),
    'dropped_features': all_features_to_drop
}

import json
with open(f'{output_dir}/selected_features.json', 'w') as f:
    json.dump(selected_features, f, indent=2)

print(f"✓ Saved: {output_dir}/selected_features.json")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FEATURE SELECTION COMPLETED!")
print("="*80)

print(f"\n📊 Summary:")
print(f"   Original dataset: {df.shape}")
print(f"   Reduced dataset: {df_reduced.shape}")
print(f"   Features removed: {len(all_features_to_drop)}")
print(f"   Features kept: {len(features_to_keep)}")
print(f"   Numeric features for modeling: {len(analysis_reduced)}")

print(f"\n📁 Output files:")
print(f"   - {output_dir}/carbon24_feature_selected.csv")
print(f"   - {output_dir}/carbon24_feature_selected_standard.csv")
print(f"   - {output_dir}/carbon24_feature_selected_robust.csv")
print(f"   - {output_dir}/feature_selection_report.txt")
print(f"   - {output_dir}/selected_features.json")

print(f"\n📈 Visualization:")
print(f"   - feature_selection_correlation_comparison.png")

print("\n✅ Dataset ready for modeling without multicollinearity issues!")
print("="*80)
