"""
Create Anomaly Detection & Energy Prediction Notebook
"""
import json

# Create notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Cell 1: Title
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "title",
    "metadata": {},
    "source": [
        "# 🔬 Carbon-24: Phát Hiện Dị Biệt & Dự Đoán Năng Lượng\n",
        "\n",
        "**Mục tiêu:**\n",
        "1. Phát hiện cấu trúc dị biệt (anomaly detection)\n",
        "2. Phân loại nhóm ổn định/kém ổn định\n",
        "3. Dự đoán energy_per_atom với Machine Learning\n",
        "4. Trực quan hóa kết quả\n",
        "\n",
        "**Dataset:** 10,153 cấu trúc Carbon-24"
    ]
})

# Cell 2: Imports
notebook["cells"].append({
    "cell_type": "code",
    "id": "imports",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Import libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import json\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Machine Learning\n",
        "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest\n",
        "from sklearn.linear_model import Ridge, Lasso\n",
        "from sklearn.cluster import KMeans, DBSCAN\n",
        "from sklearn.decomposition import PCA\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
        "from sklearn.covariance import EllipticEnvelope\n",
        "from sklearn.svm import OneClassSVM\n",
        "from sklearn.neighbors import LocalOutlierFactor\n",
        "from scipy import stats\n",
        "\n",
        "# Settings\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette('husl')\n",
        "%matplotlib inline\n",
        "\n",
        "print('✅ Libraries imported successfully!')"
    ]
})

# Cell 3: Load Data
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "load-title",
    "metadata": {},
    "source": [
        "## 📂 1. Load Data"
    ]
})

notebook["cells"].append({
    "cell_type": "code",
    "id": "load-data",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Load preprocessed data\n",
        "df = pd.read_csv('carbon24_feature_selected/carbon24_feature_selected_standard.csv')\n",
        "\n",
        "# Load original data for energy_per_atom\n",
        "df_original = pd.read_csv('carbon24_features/carbon24_project/data/carbon24_features.csv')\n",
        "df_original['energy_per_atom'] = df_original['energy'] / df_original['num_atoms']\n",
        "\n",
        "# Merge\n",
        "df = df.merge(df_original[['row_index', 'energy_per_atom', 'energy']], on='row_index', how='left')\n",
        "\n",
        "# Load feature info\n",
        "with open('carbon24_feature_selected/selected_features.json', 'r') as f:\n",
        "    feature_info = json.load(f)\n",
        "\n",
        "numeric_features = feature_info['numeric_features']\n",
        "\n",
        "print(f'✅ Loaded {len(df):,} structures')\n",
        "print(f'   Features: {len(numeric_features)}')\n",
        "print(f'\\nFirst few rows:')\n",
        "df.head()"
    ]
})

# Cell 4: Data Overview
notebook["cells"].append({
    "cell_type": "code",
    "id": "data-overview",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Data info\n",
        "print('📊 Dataset Information:')\n",
        "print(f'   Shape: {df.shape}')\n",
        "print(f'   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')\n",
        "print(f'\\n📈 Energy Statistics:')\n",
        "print(df[['energy_per_atom', 'relative_energy']].describe())"
    ]
})

# Cell 5: Anomaly Detection Title
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "anomaly-title",
    "metadata": {},
    "source": [
        "## 🔍 2. Phát Hiện Dị Biệt (Anomaly Detection)\n",
        "\n",
        "Sử dụng nhiều phương pháp để phát hiện cấu trúc bất thường:\n",
        "- **Isolation Forest** - Phát hiện outliers dựa trên isolation\n",
        "- **Local Outlier Factor (LOF)** - Phát hiện dựa trên mật độ local\n",
        "- **One-Class SVM** - Phát hiện dựa trên boundary\n",
        "- **Elliptic Envelope** - Phát hiện dựa trên Gaussian distribution\n",
        "- **Statistical Methods** - Z-score và IQR"
    ]
})

# Cell 6: Prepare Data for Anomaly Detection
notebook["cells"].append({
    "cell_type": "code",
    "id": "prep-anomaly",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Prepare features for anomaly detection\n",
        "features_for_anomaly = [f for f in numeric_features if f in df.columns]\n",
        "X_anomaly = df[features_for_anomaly].values\n",
        "\n",
        "print(f'Features for anomaly detection: {len(features_for_anomaly)}')\n",
        "print(f'Data shape: {X_anomaly.shape}')"
    ]
})

# Cell 7: Isolation Forest
notebook["cells"].append({
    "cell_type": "code",
    "id": "isolation-forest",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Isolation Forest\n",
        "print('🌲 Isolation Forest')\n",
        "print('=' * 60)\n",
        "\n",
        "iso_forest = IsolationForest(\n",
        "    contamination=0.05,  # 5% outliers\n",
        "    random_state=42,\n",
        "    n_estimators=100\n",
        ")\n",
        "\n",
        "df['anomaly_iso_forest'] = iso_forest.fit_predict(X_anomaly)\n",
        "df['anomaly_score_iso'] = iso_forest.score_samples(X_anomaly)\n",
        "\n",
        "n_anomalies = (df['anomaly_iso_forest'] == -1).sum()\n",
        "print(f'Anomalies detected: {n_anomalies:,} ({n_anomalies/len(df)*100:.2f}%)')\n",
        "print(f'Normal structures: {(df[\"anomaly_iso_forest\"] == 1).sum():,}')\n",
        "print(f'\\nAnomaly score range: [{df[\"anomaly_score_iso\"].min():.4f}, {df[\"anomaly_score_iso\"].max():.4f}]')"
    ]
})

# Cell 8: LOF
notebook["cells"].append({
    "cell_type": "code",
    "id": "lof",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Local Outlier Factor\n",
        "print('📍 Local Outlier Factor (LOF)')\n",
        "print('=' * 60)\n",
        "\n",
        "lof = LocalOutlierFactor(\n",
        "    contamination=0.05,\n",
        "    n_neighbors=20\n",
        ")\n",
        "\n",
        "df['anomaly_lof'] = lof.fit_predict(X_anomaly)\n",
        "df['anomaly_score_lof'] = lof.negative_outlier_factor_\n",
        "\n",
        "n_anomalies = (df['anomaly_lof'] == -1).sum()\n",
        "print(f'Anomalies detected: {n_anomalies:,} ({n_anomalies/len(df)*100:.2f}%)')\n",
        "print(f'Normal structures: {(df[\"anomaly_lof\"] == 1).sum():,}')"
    ]
})

# Cell 9: One-Class SVM
notebook["cells"].append({
    "cell_type": "code",
    "id": "ocsvm",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# One-Class SVM\n",
        "print('🎯 One-Class SVM')\n",
        "print('=' * 60)\n",
        "\n",
        "ocsvm = OneClassSVM(\n",
        "    nu=0.05,  # 5% outliers\n",
        "    kernel='rbf',\n",
        "    gamma='auto'\n",
        ")\n",
        "\n",
        "df['anomaly_ocsvm'] = ocsvm.fit_predict(X_anomaly)\n",
        "\n",
        "n_anomalies = (df['anomaly_ocsvm'] == -1).sum()\n",
        "print(f'Anomalies detected: {n_anomalies:,} ({n_anomalies/len(df)*100:.2f}%)')\n",
        "print(f'Normal structures: {(df[\"anomaly_ocsvm\"] == 1).sum():,}')"
    ]
})

# Cell 10: Statistical Methods
notebook["cells"].append({
    "cell_type": "code",
    "id": "statistical",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Statistical Methods - Z-score\n",
        "print('📊 Statistical Methods (Z-score)')\n",
        "print('=' * 60)\n",
        "\n",
        "# Calculate Z-scores for energy\n",
        "z_scores = np.abs(stats.zscore(df['relative_energy']))\n",
        "df['anomaly_zscore'] = (z_scores > 3).astype(int) * -1 + (z_scores <= 3).astype(int)\n",
        "\n",
        "n_anomalies = (df['anomaly_zscore'] == -1).sum()\n",
        "print(f'Anomalies detected (|z| > 3): {n_anomalies:,} ({n_anomalies/len(df)*100:.2f}%)')\n",
        "print(f'Normal structures: {(df[\"anomaly_zscore\"] == 1).sum():,}')"
    ]
})

# Cell 11: Consensus Anomalies
notebook["cells"].append({
    "cell_type": "code",
    "id": "consensus",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Consensus: structures flagged by multiple methods\n",
        "print('🎯 Consensus Anomaly Detection')\n",
        "print('=' * 60)\n",
        "\n",
        "# Count how many methods flagged each structure\n",
        "anomaly_cols = ['anomaly_iso_forest', 'anomaly_lof', 'anomaly_ocsvm', 'anomaly_zscore']\n",
        "df['anomaly_count'] = (df[anomaly_cols] == -1).sum(axis=1)\n",
        "\n",
        "# Consensus: flagged by at least 2 methods\n",
        "df['is_anomaly'] = (df['anomaly_count'] >= 2).astype(int)\n",
        "\n",
        "print(f'\\nAnomaly Detection Summary:')\n",
        "for i in range(5):\n",
        "    count = (df['anomaly_count'] == i).sum()\n",
        "    print(f'  Flagged by {i} methods: {count:,} ({count/len(df)*100:.2f}%)')\n",
        "\n",
        "n_consensus = df['is_anomaly'].sum()\n",
        "print(f'\\n🚨 Consensus anomalies (≥2 methods): {n_consensus:,} ({n_consensus/len(df)*100:.2f}%)')"
    ]
})

# Cell 12: Visualize Anomalies
notebook["cells"].append({
    "cell_type": "code",
    "id": "viz-anomalies",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Visualize anomalies\n",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
        "fig.suptitle('Anomaly Detection Results', fontsize=16, fontweight='bold')\n",
        "\n",
        "# Plot 1: Anomaly counts\n",
        "ax = axes[0, 0]\n",
        "anomaly_counts = df['anomaly_count'].value_counts().sort_index()\n",
        "bars = ax.bar(anomaly_counts.index, anomaly_counts.values, color='skyblue', edgecolor='black')\n",
        "bars[-1].set_color('red') if len(bars) > 0 else None\n",
        "ax.set_xlabel('Number of Methods Flagging as Anomaly', fontweight='bold')\n",
        "ax.set_ylabel('Count', fontweight='bold')\n",
        "ax.set_title('Distribution of Anomaly Flags')\n",
        "ax.grid(axis='y', alpha=0.3)\n",
        "\n",
        "# Plot 2: Energy distribution by anomaly status\n",
        "ax = axes[0, 1]\n",
        "normal = df[df['is_anomaly'] == 0]['relative_energy']\n",
        "anomaly = df[df['is_anomaly'] == 1]['relative_energy']\n",
        "ax.hist(normal, bins=50, alpha=0.7, label='Normal', color='green')\n",
        "ax.hist(anomaly, bins=50, alpha=0.7, label='Anomaly', color='red')\n",
        "ax.set_xlabel('Relative Energy (eV/atom)', fontweight='bold')\n",
        "ax.set_ylabel('Frequency', fontweight='bold')\n",
        "ax.set_title('Energy Distribution: Normal vs Anomaly')\n",
        "ax.legend()\n",
        "ax.grid(alpha=0.3)\n",
        "\n",
        "# Plot 3: Isolation Forest scores\n",
        "ax = axes[1, 0]\n",
        "colors = ['red' if x == -1 else 'green' for x in df['anomaly_iso_forest']]\n",
        "ax.scatter(range(len(df)), df['anomaly_score_iso'], c=colors, alpha=0.3, s=1)\n",
        "ax.set_xlabel('Structure Index', fontweight='bold')\n",
        "ax.set_ylabel('Anomaly Score', fontweight='bold')\n",
        "ax.set_title('Isolation Forest Anomaly Scores')\n",
        "ax.grid(alpha=0.3)\n",
        "\n",
        "# Plot 4: Method agreement\n",
        "ax = axes[1, 1]\n",
        "method_names = ['Iso Forest', 'LOF', 'One-Class SVM', 'Z-score']\n",
        "method_counts = [(df[col] == -1).sum() for col in anomaly_cols]\n",
        "bars = ax.barh(method_names, method_counts, color='coral', edgecolor='black')\n",
        "ax.set_xlabel('Number of Anomalies Detected', fontweight='bold')\n",
        "ax.set_title('Anomalies Detected by Each Method')\n",
        "ax.grid(axis='x', alpha=0.3)\n",
        "\n",
        "for i, (bar, count) in enumerate(zip(bars, method_counts)):\n",
        "    ax.text(count + 10, i, f'{count:,}', va='center')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('✅ Visualization complete!')"
    ]
})

# Save notebook
with open('carbon24-anomaly-energy-prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print('✅ Notebook created: carbon24-anomaly-energy-prediction.ipynb')
print('   Run: jupyter notebook carbon24-anomaly-energy-prediction.ipynb')
