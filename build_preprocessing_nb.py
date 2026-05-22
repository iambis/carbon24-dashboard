"""
Script để tạo Jupyter Notebook cho preprocessing và feature selection
"""

import nbformat as nbf

# Tạo notebook mới
nb = nbf.v4.new_notebook()

# Danh sách cells
cells = []

# ============================================================================
# CELL 1: Title và Import
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""# Carbon-24 Data Preprocessing & Feature Selection

**Dự án:** Khai thác dữ liệu cấu trúc Carbon-24  
**Mục tiêu:** Khảo sát, tiền xử lý, làm sạch dữ liệu và loại bỏ multicollinearity

## Nội dung:
1. Load và khảo sát dữ liệu
2. Phân tích phân phối
3. Phát hiện outliers
4. Phân tích tương quan
5. Xử lý outliers
6. Feature engineering
7. Loại bỏ multicollinearity
8. Chuẩn hóa dữ liệu
9. Lưu kết quả
"""))

cells.append(nbf.v4.new_code_cell("""# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Thiết lập style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Libraries imported successfully!")
"""))

# ============================================================================
# CELL 2: Load Data
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 1. Load và Khảo Sát Dữ Liệu"""))

cells.append(nbf.v4.new_code_cell("""# Load data
data_path = 'carbon24_features_v2/carbon24_project_v2/data/carbon24_features_v2.csv'
df = pd.read_csv(data_path)

print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
print(f"✓ Shape: {df.shape}")
df.head()
"""))

cells.append(nbf.v4.new_code_cell("""# Dataset info
print("Dataset Information:")
print("="*80)
df.info()
"""))

cells.append(nbf.v4.new_code_cell("""# Descriptive statistics
df.describe()
"""))

cells.append(nbf.v4.new_code_cell("""# Check missing values
missing = df.isnull().sum()
missing_pct = 100 * df.isnull().sum() / len(df)
missing_table = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
missing_table = missing_table[missing_table['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if len(missing_table) > 0:
    print("Missing Values:")
    display(missing_table)
else:
    print("✓ No missing values found!")
"""))

cells.append(nbf.v4.new_code_cell("""# Check duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Feature classification
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(f"\\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"\\nCategorical: {categorical_features}")
"""))

# ============================================================================
# CELL 3: Distribution Analysis
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Phân Tích Phân Phối Dữ Liệu"""))

cells.append(nbf.v4.new_code_cell("""# Key features to analyze
key_features = [
    'volume_per_atom', 'density', 'mean_bond_length', 
    'mean_coordination', 'relative_energy',
    'lattice_anisotropy', 'angle_std', 'packing_fraction'
]

# Plot distributions
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
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Skewness and Kurtosis
skew_kurt = pd.DataFrame({
    'Feature': key_features,
    'Skewness': [df[f].skew() for f in key_features],
    'Kurtosis': [df[f].kurtosis() for f in key_features]
})
print("Skewness & Kurtosis Analysis:")
display(skew_kurt)
"""))

# ============================================================================
# CELL 4: Outlier Detection
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Phát Hiện Outliers"""))

cells.append(nbf.v4.new_code_cell("""def detect_outliers_iqr(data, feature):
    \"\"\"Phát hiện outliers bằng IQR method\"\"\"
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

def detect_outliers_zscore(data, feature, threshold=3):
    \"\"\"Phát hiện outliers bằng Z-score method\"\"\"
    z_scores = np.abs(stats.zscore(data[feature].dropna()))
    outliers = len(z_scores[z_scores > threshold])
    return outliers

# Analyze outliers
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
print("Outlier Detection Summary:")
display(outlier_df)
"""))

# ============================================================================
# CELL 5: Correlation Analysis
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Phân Tích Tương Quan"""))

cells.append(nbf.v4.new_code_cell("""# Select numeric features for correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['row_index']
analysis_cols = [col for col in numeric_cols if col not in exclude_cols]

# Correlation matrix
correlation_matrix = df[analysis_cols].corr()

# Plot heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - All Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Find high correlation pairs
print("High Correlation Pairs (|r| > 0.8):")
print("="*80)

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
    display(high_corr_df)
else:
    print("No high correlation pairs found.")
"""))

# ============================================================================
# CELL 6: Handle Outliers
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Xử Lý Outliers"""))

cells.append(nbf.v4.new_code_cell("""def winsorize_feature(data, feature, lower_percentile=0.01, upper_percentile=0.99):
    \"\"\"Cap outliers at specified percentiles\"\"\"
    lower_bound = data[feature].quantile(lower_percentile)
    upper_bound = data[feature].quantile(upper_percentile)
    data[feature] = data[feature].clip(lower=lower_bound, upper=upper_bound)
    return data

# Apply winsorization
df_cleaned = df.copy()
features_to_winsorize = ['relative_energy', 'angle_std', 'lattice_anisotropy']

for feature in features_to_winsorize:
    if feature in df_cleaned.columns:
        df_cleaned = winsorize_feature(df_cleaned, feature, 0.01, 0.99)
        print(f"✓ Winsorized: {feature}")

print(f"\\n✓ Cleaned dataset shape: {df_cleaned.shape}")
"""))

# ============================================================================
# CELL 7: Feature Engineering
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Feature Engineering"""))

cells.append(nbf.v4.new_code_cell("""df_engineered = df_cleaned.copy()

# Volume ratio
if 'volume' in df_engineered.columns and 'num_atoms' in df_engineered.columns:
    df_engineered['volume_ratio'] = df_engineered['volume'] / df_engineered['num_atoms']

# Lattice asymmetry
if all(col in df_engineered.columns for col in ['b_over_a', 'c_over_a']):
    df_engineered['lattice_asymmetry'] = np.abs(df_engineered['b_over_a'] - 1) + \\
                                          np.abs(df_engineered['c_over_a'] - 1)

# Bond complexity
if all(col in df_engineered.columns for col in ['std_bond_length', 'mean_bond_length']):
    df_engineered['bond_complexity'] = df_engineered['std_bond_length'] / \\
                                        (df_engineered['mean_bond_length'] + 1e-10)

# Hybridization diversity
if all(col in df_engineered.columns for col in ['fraction_sp', 'fraction_sp2', 'fraction_sp3']):
    df_engineered['hybridization_diversity'] = -(
        df_engineered['fraction_sp'] * np.log(df_engineered['fraction_sp'] + 1e-10) +
        df_engineered['fraction_sp2'] * np.log(df_engineered['fraction_sp2'] + 1e-10) +
        df_engineered['fraction_sp3'] * np.log(df_engineered['fraction_sp3'] + 1e-10)
    )

new_features = [col for col in df_engineered.columns if col not in df_cleaned.columns]
print(f"✓ Created {len(new_features)} new features:")
for feat in new_features:
    print(f"  - {feat}")
"""))

# ============================================================================
# CELL 8: Feature Selection - Remove Multicollinearity
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 7. Loại Bỏ Multicollinearity

Loại bỏ các features có tương quan cao (>0.95) để tránh redundancy và cải thiện hiệu suất mô hình.
"""))

cells.append(nbf.v4.new_code_cell("""# Features to drop based on high correlation analysis
features_to_drop = [
    'packing_fraction',      # r=1.0 với density
    'energy',                # r=1.0 với relative_energy
    'fraction_sp2',          # r=-0.9997 với fraction_sp3
    'fraction_sp3',          # r=0.9993 với mean_coordination
    'volume_per_atom',       # r=-0.996 với density
    'mean_coordination',     # r=0.999 với fraction_sp2
    'mean_bond_length',      # r=0.986 với mean_coordination
    'c_over_a',              # r=0.963 với lattice_anisotropy
    'volume',                # r=0.962 với num_atoms
    'lattice_anisotropy',    # r=0.959 với c_over_a
    'std_bond_length',       # r=0.936 với bond_length_range
    'std_coordination',      # r=0.997 với hybridization_diversity
    'b_over_a',              # r=0.885 với b
    'density',               # r=0.996 với volume_per_atom
]

# Remove features
features_to_keep = [col for col in df_engineered.columns if col not in features_to_drop]
df_reduced = df_engineered[features_to_keep].copy()

print(f"Original features: {len(df_engineered.columns)}")
print(f"Reduced features: {len(df_reduced.columns)}")
print(f"Removed: {len(features_to_drop)} features")
print(f"\\nRemoved features: {sorted(features_to_drop)}")
"""))

cells.append(nbf.v4.new_code_cell("""# Verify reduced correlation
numeric_reduced = df_reduced.select_dtypes(include=[np.number]).columns.tolist()
exclude_reduced = ['row_index', 'space_group_number']
analysis_reduced = [col for col in numeric_reduced if col not in exclude_reduced]

corr_matrix_reduced = df_reduced[analysis_reduced].corr().abs()

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Before
sns.heatmap(df_engineered[analysis_cols].corr(), ax=axes[0], cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[0].set_title(f'Before: {len(analysis_cols)} features', fontsize=14, fontweight='bold')

# After
sns.heatmap(df_reduced[analysis_reduced].corr(), ax=axes[1], cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[1].set_title(f'After: {len(analysis_reduced)} features', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Check remaining high correlations
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
    print(f"⚠️  Still {len(high_corr_remaining)} pairs with |r| > 0.90:")
    remaining_df = pd.DataFrame(high_corr_remaining).sort_values('Correlation', ascending=False)
    display(remaining_df.head(10))
else:
    print("✅ No correlation pairs > 0.90 remaining!")
"""))

# ============================================================================
# CELL 9: Normalization
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 8. Chuẩn Hóa Dữ Liệu"""))

cells.append(nbf.v4.new_code_cell("""# Features to scale
features_to_scale = [col for col in df_reduced.select_dtypes(include=[np.number]).columns 
                     if col not in ['row_index', 'space_group_number']]

# StandardScaler
scaler_standard = StandardScaler()
df_standard = df_reduced.copy()
df_standard[features_to_scale] = scaler_standard.fit_transform(df_reduced[features_to_scale])

# RobustScaler
scaler_robust = RobustScaler()
df_robust = df_reduced.copy()
df_robust[features_to_scale] = scaler_robust.fit_transform(df_reduced[features_to_scale])

print(f"✓ StandardScaler applied to {len(features_to_scale)} features")
print(f"✓ RobustScaler applied to {len(features_to_scale)} features")
"""))

cells.append(nbf.v4.new_code_cell("""# Compare normalization methods
sample_features = ['relative_energy', 'angle_std', 'num_atoms']

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for idx, feature in enumerate(sample_features):
    if feature in df_reduced.columns:
        # Original
        axes[idx*3].hist(df_reduced[feature], bins=50, edgecolor='black', alpha=0.7)
        axes[idx*3].set_title(f'Original: {feature}')
        axes[idx*3].grid(True, alpha=0.3)
        
        # StandardScaler
        axes[idx*3+1].hist(df_standard[feature], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[idx*3+1].set_title(f'StandardScaler: {feature}')
        axes[idx*3+1].grid(True, alpha=0.3)
        
        # RobustScaler
        axes[idx*3+2].hist(df_robust[feature], bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[idx*3+2].set_title(f'RobustScaler: {feature}')
        axes[idx*3+2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""))

# ============================================================================
# CELL 10: Crystal System Analysis
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 9. Phân Tích Theo Crystal System"""))

cells.append(nbf.v4.new_code_cell("""if 'crystal_system' in df_reduced.columns:
    # Distribution
    crystal_counts = df_reduced['crystal_system'].value_counts()
    
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
    plt.show()
    
    print("\\nCrystal System Counts:")
    display(crystal_counts)
"""))

cells.append(nbf.v4.new_code_cell("""# Energy by crystal system
if 'relative_energy' in df_reduced.columns and 'crystal_system' in df_reduced.columns:
    energy_by_crystal = df_reduced.groupby('crystal_system')['relative_energy'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    
    print("Energy Statistics by Crystal System:")
    display(energy_by_crystal)
"""))

# ============================================================================
# CELL 11: Save Results
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 10. Lưu Kết Quả"""))

cells.append(nbf.v4.new_code_cell("""# Create output directory
output_dir = 'carbon24_preprocessing_results'
os.makedirs(output_dir, exist_ok=True)

# Save datasets
df_reduced.to_csv(f'{output_dir}/carbon24_feature_selected.csv', index=False)
df_standard.to_csv(f'{output_dir}/carbon24_feature_selected_standard.csv', index=False)
df_robust.to_csv(f'{output_dir}/carbon24_feature_selected_robust.csv', index=False)

print("✓ Saved datasets:")
print(f"  - {output_dir}/carbon24_feature_selected.csv")
print(f"  - {output_dir}/carbon24_feature_selected_standard.csv")
print(f"  - {output_dir}/carbon24_feature_selected_robust.csv")
"""))

cells.append(nbf.v4.new_code_cell("""# Save feature lists
selected_features = {
    'all_features': features_to_keep,
    'numeric_features': analysis_reduced,
    'categorical_features': df_reduced.select_dtypes(include=['object']).columns.tolist(),
    'dropped_features': features_to_drop
}

with open(f'{output_dir}/selected_features.json', 'w') as f:
    json.dump(selected_features, f, indent=2)

print(f"✓ Saved: {output_dir}/selected_features.json")
"""))

# ============================================================================
# CELL 12: Summary
# ============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 📊 Summary

### Dataset Transformation:
- **Original:** 10,153 samples × 41 features
- **After Engineering:** 10,153 samples × 45 features
- **After Feature Selection:** 10,153 samples × 31 features
- **Features Removed:** 14 (multicollinearity)

### Key Steps Completed:
1. ✅ Data loading and exploration
2. ✅ Distribution analysis
3. ✅ Outlier detection and handling
4. ✅ Correlation analysis
5. ✅ Feature engineering (4 new features)
6. ✅ Multicollinearity removal (14 features dropped)
7. ✅ Data normalization (StandardScaler & RobustScaler)
8. ✅ Crystal system analysis
9. ✅ Results saved

### Ready for:
- 🎯 Clustering (K-means, DBSCAN, Hierarchical)
- 🔍 Anomaly Detection (Isolation Forest, LOF)
- 📈 Energy Prediction (Regression models)
- 📊 Visualization (PCA, t-SNE, UMAP)
"""))

# Thêm cells vào notebook
nb['cells'] = cells

# Lưu notebook
with open('carbon24-preprocessing.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("✓ Notebook created successfully: carbon24-preprocessing.ipynb")
