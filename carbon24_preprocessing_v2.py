"""
Carbon-24 Data Preprocessing Pipeline
Khảo sát, tiền xử lý và làm sạch dữ liệu cho dự án phân cụm Carbon-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Thiết lập style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("CARBON-24 DATA PREPROCESSING PIPELINE")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA...")
data_path = 'carbon24_features_v2/carbon24_project_v2/data/carbon24_features_v2.csv'
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} samples with {len(df.columns)} features")
print(f"Shape: {df.shape}")

# ============================================================================
# 2. KHẢO SÁT DỮ LIỆU BAN ĐẦU
# ============================================================================
print("\n[2] EXPLORATORY DATA ANALYSIS...")

# 2.1 Thông tin cơ bản
print("\n--- 2.1 Dataset Info ---")
print(df.info())

# 2.2 Thống kê mô tả
print("\n--- 2.2 Descriptive Statistics ---")
print(df.describe())

# 2.3 Kiểm tra missing values
print("\n--- 2.3 Missing Values ---")
missing = df.isnull().sum()
missing_pct = 100 * df.isnull().sum() / len(df)
missing_table = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
missing_table = missing_table[missing_table['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
if len(missing_table) > 0:
    print(missing_table)
else:
    print("✓ No missing values found!")

# 2.4 Kiểm tra duplicates
print("\n--- 2.4 Duplicate Rows ---")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# 2.5 Phân loại features
print("\n--- 2.5 Feature Classification ---")
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features ({len(numeric_features)}): {numeric_features[:10]}...")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# ============================================================================
# 3. PHÂN TÍCH PHÂN PHỐI DỮ LIỆU
# ============================================================================
print("\n DISTRIBUTION ANALYSIS...")

# Chọn các features quan trọng để phân tích
key_features = [
    'volume_per_atom', 'density', 'mean_bond_length', 
    'mean_coordination', 'relative_energy',
    'lattice_anisotropy', 'angle_std', 'packing_fraction'
]

# 3.1 Vẽ histogram và boxplot
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    # Histogram
    axes[idx*2].hist(df[feature].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[idx*2].set_title(f'Distribution: {feature}')
    axes[idx*2].set_xlabel(feature)
    axes[idx*2].set_ylabel('Frequency')
    axes[idx*2].grid(True, alpha=0.3)
    
    # Boxplot
    axes[idx*2+1].boxplot(df[feature].dropna(), vert=True)
    axes[idx*2+1].set_title(f'Boxplot: {feature}')
    axes[idx*2+1].set_ylabel(feature)
    axes[idx*2+1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessing_distributions.png', dpi=300, bbox_inches='tight')
print(" Saved: preprocessing_distributions.png")
plt.close()

# 3.2 Kiểm tra skewness và kurtosis
print("\n--- 3.2 Skewness & Kurtosis ---")
skew_kurt = pd.DataFrame({
    'Feature': key_features,
    'Skewness': [df[f].skew() for f in key_features],
    'Kurtosis': [df[f].kurtosis() for f in key_features]
})
print(skew_kurt)

# ============================================================================
# 4. PHÁT HIỆN OUTLIERS
# ============================================================================
print("\n[4] OUTLIER DETECTION...")

def detect_outliers_iqr(data, feature):
    """Phát hiện outliers bằng IQR method"""
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

def detect_outliers_zscore(data, feature, threshold=3):
    """Phát hiện outliers bằng Z-score method"""
    z_scores = np.abs(stats.zscore(data[feature].dropna()))
    outliers = len(z_scores[z_scores > threshold])
    return outliers

# Phân tích outliers
outlier_summary = []
for feature in key_features:
    iqr_count, lower, upper = detect_outliers_iqr(df, feature)
    zscore_count = detect_outliers_zscore(df, feature)
    outlier_summary.append({
        'Feature': feature,
        'IQR_Outliers': iqr_count,
        'IQR_Percentage': f"{100*iqr_count/len(df):.2f}%",
        'ZScore_Outliers': zscore_count,
        'ZScore_Percentage': f"{100*zscore_count/len(df):.2f}%"
    })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df)

# ============================================================================
# 5. PHÂN TÍCH TƯƠNG QUAN
# ============================================================================
print("\n[5] CORRELATION ANALYSIS...")

# Chọn numeric features (loại bỏ ID và categorical)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['row_index']
analysis_cols = [col for col in numeric_cols if col not in exclude_cols]

# 5.1 Correlation matrix
correlation_matrix = df[analysis_cols].corr()

# Vẽ heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - All Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('preprocessing_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: preprocessing_correlation_heatmap.png")
plt.close()

# 5.2 Tìm các cặp features có tương quan cao
print("\n--- 5.2 High Correlation Pairs (|r| > 0.8) ---")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'Feature 1': correlation_matrix.columns[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', 
                                                               key=abs, 
                                                               ascending=False)
    print(high_corr_df)
else:
    print("No high correlation pairs found.")

# ============================================================================
# 6. XỬ LÝ OUTLIERS
# ============================================================================
print("\n[6] HANDLING OUTLIERS...")

df_cleaned = df.copy()

# Phương pháp: Winsorization (cap outliers at percentiles)
def winsorize_feature(data, feature, lower_percentile=0.01, upper_percentile=0.99):
    """Cap outliers at specified percentiles"""
    lower_bound = data[feature].quantile(lower_percentile)
    upper_bound = data[feature].quantile(upper_percentile)
    data[feature] = data[feature].clip(lower=lower_bound, upper=upper_bound)
    return data

# Áp dụng winsorization cho các features có nhiều outliers
features_to_winsorize = ['relative_energy', 'angle_std', 'lattice_anisotropy']

for feature in features_to_winsorize:
    if feature in df_cleaned.columns:
        df_cleaned = winsorize_feature(df_cleaned, feature, 0.01, 0.99)
        print(f"✓ Winsorized: {feature}")

print(f"✓ Cleaned dataset shape: {df_cleaned.shape}")

# ============================================================================
# 7. FEATURE ENGINEERING
# ============================================================================
print("\n[7] FEATURE ENGINEERING...")

# 7.1 Tạo features mới
df_engineered = df_cleaned.copy()

# Tỷ lệ thể tích
if 'volume' in df_engineered.columns and 'num_atoms' in df_engineered.columns:
    df_engineered['volume_ratio'] = df_engineered['volume'] / df_engineered['num_atoms']

# Độ bất đối xứng mạng tinh thể tổng hợp
if all(col in df_engineered.columns for col in ['b_over_a', 'c_over_a']):
    df_engineered['lattice_asymmetry'] = np.abs(df_engineered['b_over_a'] - 1) + \
                                          np.abs(df_engineered['c_over_a'] - 1)

# Độ phức tạp liên kết
if all(col in df_engineered.columns for col in ['std_bond_length', 'mean_bond_length']):
    df_engineered['bond_complexity'] = df_engineered['std_bond_length'] / \
                                        (df_engineered['mean_bond_length'] + 1e-10)

# Chỉ số hybridization
if all(col in df_engineered.columns for col in ['fraction_sp', 'fraction_sp2', 'fraction_sp3']):
    df_engineered['hybridization_diversity'] = -(
        df_engineered['fraction_sp'] * np.log(df_engineered['fraction_sp'] + 1e-10) +
        df_engineered['fraction_sp2'] * np.log(df_engineered['fraction_sp2'] + 1e-10) +
        df_engineered['fraction_sp3'] * np.log(df_engineered['fraction_sp3'] + 1e-10)
    )

print(f" Created {len(df_engineered.columns) - len(df_cleaned.columns)} new features")
print(f" New features: {[col for col in df_engineered.columns if col not in df_cleaned.columns]}")

# ============================================================================
# 8. CHUẨN HÓA DỮ LIỆU
# ============================================================================
print("\n[8] DATA NORMALIZATION...")

# Chọn features để chuẩn hóa (numeric, không phải ID hoặc categorical)
features_to_scale = [col for col in df_engineered.select_dtypes(include=[np.number]).columns 
                     if col not in ['row_index', 'space_group_number']]

# 8.1 StandardScaler (Z-score normalization)
scaler_standard = StandardScaler()
df_standard = df_engineered.copy()
df_standard[features_to_scale] = scaler_standard.fit_transform(df_engineered[features_to_scale])

print(f" StandardScaler applied to {len(features_to_scale)} features")

# 8.2 RobustScaler (robust to outliers)
scaler_robust = RobustScaler()
df_robust = df_engineered.copy()
df_robust[features_to_scale] = scaler_robust.fit_transform(df_engineered[features_to_scale])

print(f" RobustScaler applied to {len(features_to_scale)} features")

# So sánh phân phối trước và sau chuẩn hóa
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

sample_features = key_features[:3]
for idx, feature in enumerate(sample_features):
    # Original
    axes[idx*3].hist(df_engineered[feature], bins=50, edgecolor='black', alpha=0.7)
    axes[idx*3].set_title(f'Original: {feature}')
    axes[idx*3].set_xlabel(feature)
    axes[idx*3].grid(True, alpha=0.3)
    
    # StandardScaler
    axes[idx*3+1].hist(df_standard[feature], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[idx*3+1].set_title(f'StandardScaler: {feature}')
    axes[idx*3+1].set_xlabel(feature)
    axes[idx*3+1].grid(True, alpha=0.3)
    
    # RobustScaler
    axes[idx*3+2].hist(df_robust[feature], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[idx*3+2].set_title(f'RobustScaler: {feature}')
    axes[idx*3+2].set_xlabel(feature)
    axes[idx*3+2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessing_normalization_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: preprocessing_normalization_comparison.png")
plt.close()

# ============================================================================
# 9. PHÂN TÍCH THEO CRYSTAL SYSTEM
# ============================================================================
print("\n[9] CRYSTAL SYSTEM ANALYSIS...")

if 'crystal_system' in df_engineered.columns:
    print("\n--- 9.1 Crystal System Distribution ---")
    crystal_counts = df_engineered['crystal_system'].value_counts()
    print(crystal_counts)
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    crystal_counts.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title('Crystal System Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Crystal System')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # Pie chart
    axes[1].pie(crystal_counts, labels=crystal_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=sns.color_palette('husl', len(crystal_counts)))
    axes[1].set_title('Crystal System Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('preprocessing_crystal_system.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: preprocessing_crystal_system.png")
    plt.close()
    
    # Phân tích energy theo crystal system
    print("\n--- 9.2 Energy by Crystal System ---")
    if 'relative_energy' in df_engineered.columns:
        energy_by_crystal = df_engineered.groupby('crystal_system')['relative_energy'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        print(energy_by_crystal)

# ============================================================================
# 10. LƯU KẾT QUẢ
# ============================================================================
print("\n[10] SAVING RESULTS...")

# Tạo thư mục output
import os
output_dir = 'carbon24_preprocessing_results'
os.makedirs(output_dir, exist_ok=True)

# Lưu các dataset
df_engineered.to_csv(f'{output_dir}/carbon24_cleaned_engineered.csv', index=False)
print(f" Saved: {output_dir}/carbon24_cleaned_engineered.csv")

df_standard.to_csv(f'{output_dir}/carbon24_standardized.csv', index=False)
print(f" Saved: {output_dir}/carbon24_standardized.csv")

df_robust.to_csv(f'{output_dir}/carbon24_robust_scaled.csv', index=False)
print(f" Saved: {output_dir}/carbon24_robust_scaled.csv")

# Lưu preprocessing report
with open(f'{output_dir}/preprocessing_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("CARBON-24 PREPROCESSING REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Original dataset: {df.shape}\n")
    f.write(f"Cleaned dataset: {df_cleaned.shape}\n")
    f.write(f"Engineered dataset: {df_engineered.shape}\n\n")
    
    f.write("Missing Values: None\n")
    f.write(f"Duplicate Rows: {duplicates}\n\n")
    
    f.write("Outlier Summary:\n")
    f.write(outlier_df.to_string())
    f.write("\n\n")
    
    f.write("High Correlation Pairs:\n")
    if high_corr_pairs:
        f.write(high_corr_df.to_string())
    else:
        f.write("None found (threshold: |r| > 0.8)\n")
    f.write("\n\n")
    
    f.write("New Engineered Features:\n")
    new_features = [col for col in df_engineered.columns if col not in df_cleaned.columns]
    for feat in new_features:
        f.write(f"  - {feat}\n")
    f.write("\n")
    
    f.write("Normalization Methods Applied:\n")
    f.write("  - StandardScaler (Z-score)\n")
    f.write("  - RobustScaler (median and IQR)\n")

print(f"✓ Saved: {output_dir}/preprocessing_report.txt")

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\n Dataset Summary:")
print(f"   - Original samples: {len(df)}")
print(f"   - Final samples: {len(df_engineered)}")
print(f"   - Original features: {len(df.columns)}")
print(f"   - Final features: {len(df_engineered.columns)}")
print(f"   - New features created: {len(df_engineered.columns) - len(df.columns)}")

print(f"\n Output Files:")
print(f"   - {output_dir}/carbon24_cleaned_engineered.csv")
print(f"   - {output_dir}/carbon24_standardized.csv")
print(f"   - {output_dir}/carbon24_robust_scaled.csv")
print(f"   - {output_dir}/preprocessing_report.txt")

print(f"\n Visualizations:")
print(f"   - preprocessing_distributions.png")
print(f"   - preprocessing_correlation_heatmap.png")
print(f"   - preprocessing_normalization_comparison.png")
print(f"   - preprocessing_crystal_system.png")

print("\n Ready for clustering, anomaly detection, and energy prediction!")
print("="*80)
