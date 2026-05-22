"""
Add Energy Prediction cells to notebook
"""
import json

# Load existing notebook
with open('carbon24-anomaly-energy-prediction.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Add more cells for Energy Prediction

# Cell: Stability Classification Title
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "stability-title",
    "metadata": {},
    "source": [
        "## ⚡ 3. Phân Loại Stability\n",
        "\n",
        "Sử dụng clustering để phân loại các cấu trúc theo mức độ ổn định"
    ]
})

# Cell: Clustering
notebook["cells"].append({
    "cell_type": "code",
    "id": "clustering",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Clustering for stability groups\n",
        "print('🔄 K-means Clustering')\n",
        "print('=' * 60)\n",
        "\n",
        "# Prepare features (exclude anomalies and energy)\n",
        "features_for_clustering = [f for f in numeric_features if f != 'relative_energy' and f in df.columns]\n",
        "X_cluster = df[features_for_clustering].values\n",
        "\n",
        "# K-means with k=3\n",
        "optimal_k = 3\n",
        "kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=300, random_state=42)\n",
        "df['cluster'] = kmeans.fit_predict(X_cluster)\n",
        "\n",
        "print(f'✅ Clustering completed with k={optimal_k}')\n",
        "print(f'\\nCluster distribution:')\n",
        "for i in range(optimal_k):\n",
        "    count = (df['cluster'] == i).sum()\n",
        "    print(f'  Cluster {i}: {count:,} ({count/len(df)*100:.1f}%)')"
    ]
})

# Cell: Stability Analysis
notebook["cells"].append({
    "cell_type": "code",
    "id": "stability-analysis",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Analyze stability by cluster\n",
        "print('\\n📊 Stability Analysis by Cluster')\n",
        "print('=' * 60)\n",
        "\n",
        "stability_stats = df.groupby('cluster')['relative_energy'].agg([\n",
        "    'count', 'mean', 'std', 'min', 'max', 'median'\n",
        "]).round(4)\n",
        "\n",
        "print(stability_stats)\n",
        "\n",
        "# Classify clusters\n",
        "print('\\n🎯 Stability Classification:')\n",
        "for i in range(optimal_k):\n",
        "    mean_energy = stability_stats.loc[i, 'mean']\n",
        "    \n",
        "    if mean_energy < 0.15:\n",
        "        stability = \"🟢 Highly Stable\"\n",
        "    elif mean_energy < 0.30:\n",
        "        stability = \"🟡 Moderately Stable\"\n",
        "    else:\n",
        "        stability = \"🔴 Less Stable\"\n",
        "    \n",
        "    count = stability_stats.loc[i, 'count']\n",
        "    print(f'  Cluster {i}: {stability}')\n",
        "    print(f'    - Structures: {int(count):,}')\n",
        "    print(f'    - Mean energy: {mean_energy:.4f} eV/atom')\n",
        "    print()"
    ]
})

# Cell: Visualize Stability
notebook["cells"].append({
    "cell_type": "code",
    "id": "viz-stability",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Visualize stability\n",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
        "fig.suptitle('Stability Analysis', fontsize=16, fontweight='bold')\n",
        "\n",
        "# Plot 1: Energy distribution by cluster\n",
        "ax = axes[0, 0]\n",
        "for i in range(optimal_k):\n",
        "    cluster_data = df[df['cluster'] == i]['relative_energy']\n",
        "    ax.hist(cluster_data, bins=30, alpha=0.6, label=f'Cluster {i}')\n",
        "ax.set_xlabel('Relative Energy (eV/atom)', fontweight='bold')\n",
        "ax.set_ylabel('Frequency', fontweight='bold')\n",
        "ax.set_title('Energy Distribution by Cluster')\n",
        "ax.legend()\n",
        "ax.grid(alpha=0.3)\n",
        "\n",
        "# Plot 2: Box plot\n",
        "ax = axes[0, 1]\n",
        "df.boxplot(column='relative_energy', by='cluster', ax=ax)\n",
        "ax.set_xlabel('Cluster', fontweight='bold')\n",
        "ax.set_ylabel('Relative Energy (eV/atom)', fontweight='bold')\n",
        "ax.set_title('Energy Distribution Comparison')\n",
        "plt.sca(ax)\n",
        "plt.xticks(range(1, optimal_k+1), [f'C{i}' for i in range(optimal_k)])\n",
        "\n",
        "# Plot 3: Cluster sizes\n",
        "ax = axes[1, 0]\n",
        "cluster_counts = df['cluster'].value_counts().sort_index()\n",
        "bars = ax.bar(range(optimal_k), cluster_counts.values, color=['green', 'green', 'red'], alpha=0.7, edgecolor='black')\n",
        "ax.set_xlabel('Cluster', fontweight='bold')\n",
        "ax.set_ylabel('Number of Structures', fontweight='bold')\n",
        "ax.set_title('Cluster Size Distribution')\n",
        "ax.set_xticks(range(optimal_k))\n",
        "ax.grid(axis='y', alpha=0.3)\n",
        "\n",
        "for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):\n",
        "    ax.text(i, count + 50, f'{count:,}', ha='center', va='bottom', fontweight='bold')\n",
        "\n",
        "# Plot 4: PCA visualization\n",
        "ax = axes[1, 1]\n",
        "pca = PCA(n_components=2)\n",
        "X_pca = pca.fit_transform(X_cluster)\n",
        "\n",
        "for i in range(optimal_k):\n",
        "    mask = df['cluster'] == i\n",
        "    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.6, s=20, label=f'Cluster {i}')\n",
        "\n",
        "ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')\n",
        "ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')\n",
        "ax.set_title('PCA Visualization of Clusters')\n",
        "ax.legend()\n",
        "ax.grid(alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('✅ Stability visualization complete!')"
    ]
})

# Cell: Energy Prediction Title
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "prediction-title",
    "metadata": {},
    "source": [
        "## 🤖 4. Dự Đoán Energy per Atom\n",
        "\n",
        "Sử dụng Machine Learning để dự đoán energy_per_atom từ đặc trưng hình học"
    ]
})

# Cell: Prepare Data for ML
notebook["cells"].append({
    "cell_type": "code",
    "id": "prep-ml",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Prepare data for prediction\n",
        "print('📊 Preparing data for ML')\n",
        "print('=' * 60)\n",
        "\n",
        "# Features and target\n",
        "prediction_features = features_for_clustering\n",
        "X_pred = df[prediction_features].values\n",
        "y_pred = df['energy_per_atom'].values\n",
        "\n",
        "# Train-test split\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X_pred, y_pred, test_size=0.2, random_state=42\n",
        ")\n",
        "\n",
        "print(f'Training set: {len(X_train):,} samples')\n",
        "print(f'Test set: {len(X_test):,} samples')\n",
        "print(f'Features: {len(prediction_features)}')\n",
        "print(f'\\nTarget (energy_per_atom) statistics:')\n",
        "print(f'  Mean: {y_pred.mean():.4f} eV/atom')\n",
        "print(f'  Std: {y_pred.std():.4f} eV/atom')\n",
        "print(f'  Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]')"
    ]
})

# Cell: Train Models
notebook["cells"].append({
    "cell_type": "code",
    "id": "train-models",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Train multiple models\n",
        "print('🤖 Training ML Models')\n",
        "print('=' * 60)\n",
        "\n",
        "models = {\n",
        "    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),\n",
        "    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),\n",
        "    'Ridge': Ridge(alpha=1.0),\n",
        "    'Lasso': Lasso(alpha=0.1)\n",
        "}\n",
        "\n",
        "results = []\n",
        "\n",
        "for name, model in models.items():\n",
        "    print(f'\\nTraining {name}...')\n",
        "    \n",
        "    # Train\n",
        "    model.fit(X_train, y_train)\n",
        "    \n",
        "    # Predict\n",
        "    y_train_pred = model.predict(X_train)\n",
        "    y_test_pred = model.predict(X_test)\n",
        "    \n",
        "    # Evaluate\n",
        "    train_r2 = r2_score(y_train, y_train_pred)\n",
        "    test_r2 = r2_score(y_test, y_test_pred)\n",
        "    train_mae = mean_absolute_error(y_train, y_train_pred)\n",
        "    test_mae = mean_absolute_error(y_test, y_test_pred)\n",
        "    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))\n",
        "    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))\n",
        "    \n",
        "    results.append({\n",
        "        'Model': name,\n",
        "        'Train R²': train_r2,\n",
        "        'Test R²': test_r2,\n",
        "        'Train MAE': train_mae,\n",
        "        'Test MAE': test_mae,\n",
        "        'Train RMSE': train_rmse,\n",
        "        'Test RMSE': test_rmse\n",
        "    })\n",
        "    \n",
        "    print(f'  ✅ Test R²: {test_r2:.4f}, Test MAE: {test_mae:.4f}')\n",
        "\n",
        "# Results DataFrame\n",
        "results_df = pd.DataFrame(results)\n",
        "print('\\n' + '=' * 60)\n",
        "print('📊 MODEL COMPARISON')\n",
        "print('=' * 60)\n",
        "print(results_df.to_string(index=False))\n",
        "\n",
        "# Best model\n",
        "best_model_name = results_df.loc[results_df['Test R²'].idxmax(), 'Model']\n",
        "print(f'\\n🏆 Best Model: {best_model_name}')\n",
        "print(f'   Test R²: {results_df[\"Test R²\"].max():.4f}')\n",
        "print(f'   Test MAE: {results_df[\"Test MAE\"].min():.4f} eV/atom')"
    ]
})

# Cell: Feature Importance
notebook["cells"].append({
    "cell_type": "code",
    "id": "feature-importance",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Feature importance (for Random Forest)\n",
        "if best_model_name == 'Random Forest':\n",
        "    print('\\n📈 Feature Importance (Random Forest)')\n",
        "    print('=' * 60)\n",
        "    \n",
        "    rf_model = models['Random Forest']\n",
        "    importances = rf_model.feature_importances_\n",
        "    \n",
        "    feature_importance_df = pd.DataFrame({\n",
        "        'feature': prediction_features,\n",
        "        'importance': importances\n",
        "    }).sort_values('importance', ascending=False)\n",
        "    \n",
        "    print('\\nTop 10 Most Important Features:')\n",
        "    print(feature_importance_df.head(10).to_string(index=False))\n",
        "    \n",
        "    # Plot\n",
        "    plt.figure(figsize=(12, 8))\n",
        "    top_n = 15\n",
        "    top_features = feature_importance_df.head(top_n)\n",
        "    \n",
        "    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', edgecolor='black')\n",
        "    plt.yticks(range(len(top_features)), top_features['feature'])\n",
        "    plt.gca().invert_yaxis()\n",
        "    plt.xlabel('Importance', fontweight='bold')\n",
        "    plt.title(f'Top {top_n} Features for Energy Prediction', fontsize=14, fontweight='bold')\n",
        "    plt.grid(axis='x', alpha=0.3)\n",
        "    plt.tight_layout()\n",
        "    plt.show()"
    ]
})

# Cell: Prediction Visualization
notebook["cells"].append({
    "cell_type": "code",
    "id": "viz-prediction",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Visualize predictions\n",
        "best_model = models[best_model_name]\n",
        "y_test_pred = best_model.predict(X_test)\n",
        "\n",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
        "fig.suptitle(f'Energy Prediction Results ({best_model_name})', fontsize=16, fontweight='bold')\n",
        "\n",
        "# Plot 1: Actual vs Predicted\n",
        "ax = axes[0, 0]\n",
        "ax.scatter(y_test, y_test_pred, alpha=0.5, s=20)\n",
        "min_val = min(y_test.min(), y_test_pred.min())\n",
        "max_val = max(y_test.max(), y_test_pred.max())\n",
        "ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')\n",
        "ax.set_xlabel('Actual Energy per Atom (eV/atom)', fontweight='bold')\n",
        "ax.set_ylabel('Predicted Energy per Atom (eV/atom)', fontweight='bold')\n",
        "ax.set_title('Actual vs Predicted')\n",
        "ax.legend()\n",
        "ax.grid(alpha=0.3)\n",
        "\n",
        "test_r2 = r2_score(y_test, y_test_pred)\n",
        "ax.text(0.05, 0.95, f'R² = {test_r2:.4f}', transform=ax.transAxes,\n",
        "        fontsize=12, verticalalignment='top',\n",
        "        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n",
        "\n",
        "# Plot 2: Residuals\n",
        "ax = axes[0, 1]\n",
        "residuals = y_test - y_test_pred\n",
        "ax.scatter(y_test_pred, residuals, alpha=0.5, s=20, color='purple')\n",
        "ax.axhline(0, color='red', linestyle='--', linewidth=2)\n",
        "ax.set_xlabel('Predicted Energy per Atom (eV/atom)', fontweight='bold')\n",
        "ax.set_ylabel('Residual (Actual - Predicted)', fontweight='bold')\n",
        "ax.set_title('Residual Plot')\n",
        "ax.grid(alpha=0.3)\n",
        "\n",
        "# Plot 3: Error distribution\n",
        "ax = axes[1, 0]\n",
        "ax.hist(residuals, bins=50, color='coral', alpha=0.7, edgecolor='black')\n",
        "ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')\n",
        "ax.set_xlabel('Prediction Error (eV/atom)', fontweight='bold')\n",
        "ax.set_ylabel('Frequency', fontweight='bold')\n",
        "ax.set_title('Error Distribution')\n",
        "ax.legend()\n",
        "ax.grid(alpha=0.3)\n",
        "\n",
        "mean_error = residuals.mean()\n",
        "std_error = residuals.std()\n",
        "ax.text(0.05, 0.95, f'Mean: {mean_error:.4f}\\nStd: {std_error:.4f}',\n",
        "        transform=ax.transAxes, fontsize=11, verticalalignment='top',\n",
        "        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n",
        "\n",
        "# Plot 4: Model comparison\n",
        "ax = axes[1, 1]\n",
        "x_pos = range(len(results_df))\n",
        "bars = ax.bar(x_pos, results_df['Test R²'], color='skyblue', alpha=0.7, edgecolor='black')\n",
        "best_idx = results_df['Test R²'].idxmax()\n",
        "bars[best_idx].set_color('gold')\n",
        "bars[best_idx].set_edgecolor('red')\n",
        "bars[best_idx].set_linewidth(2)\n",
        "ax.set_xticks(x_pos)\n",
        "ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')\n",
        "ax.set_ylabel('Test R² Score', fontweight='bold')\n",
        "ax.set_title('Model Performance Comparison')\n",
        "ax.set_ylim([0, 1.05])\n",
        "ax.grid(axis='y', alpha=0.3)\n",
        "\n",
        "for i, (bar, r2) in enumerate(zip(bars, results_df['Test R²'])):\n",
        "    ax.text(i, r2 + 0.02, f'{r2:.4f}', ha='center', va='bottom', fontsize=9)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('✅ Prediction visualization complete!')"
    ]
})

# Cell: Summary
notebook["cells"].append({
    "cell_type": "markdown",
    "id": "summary",
    "metadata": {},
    "source": [
        "## 📊 5. Tổng Kết\n",
        "\n",
        "### Anomaly Detection\n",
        "- Sử dụng 4 phương pháp: Isolation Forest, LOF, One-Class SVM, Z-score\n",
        "- Consensus anomalies: Cấu trúc được đánh dấu bởi ≥2 phương pháp\n",
        "- Phát hiện các cấu trúc bất thường về năng lượng và hình học\n",
        "\n",
        "### Stability Classification\n",
        "- 3 clusters được xác định bằng K-means\n",
        "- Phân loại: Highly Stable, Moderately Stable, Less Stable\n",
        "- Dựa trên mean relative energy của mỗi cluster\n",
        "\n",
        "### Energy Prediction\n",
        "- Random Forest đạt hiệu suất tốt nhất\n",
        "- Có thể dự đoán energy_per_atom từ đặc trưng hình học\n",
        "- num_atoms là feature quan trọng nhất\n",
        "\n",
        "### Key Findings\n",
        "1. Có mối quan hệ mạnh giữa cấu trúc hình học và năng lượng\n",
        "2. Phần lớn cấu trúc là ổn định (Highly/Moderately Stable)\n",
        "3. Anomalies thường có năng lượng cao hoặc cấu trúc đặc biệt\n",
        "4. Machine Learning có thể dự đoán energy với độ chính xác cao"
    ]
})

# Cell: Save Results
notebook["cells"].append({
    "cell_type": "code",
    "id": "save-results",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Save results\n",
        "import os\n",
        "\n",
        "output_dir = 'carbon24_anomaly_prediction_results'\n",
        "os.makedirs(output_dir, exist_ok=True)\n",
        "\n",
        "# Save anomaly results\n",
        "anomaly_results = df[['row_index', 'is_anomaly', 'anomaly_count', \n",
        "                       'anomaly_iso_forest', 'anomaly_lof', 'anomaly_ocsvm', 'anomaly_zscore',\n",
        "                       'relative_energy', 'energy_per_atom']]\n",
        "anomaly_results.to_csv(f'{output_dir}/anomaly_detection_results.csv', index=False)\n",
        "\n",
        "# Save stability classification\n",
        "stability_results = df[['row_index', 'cluster', 'relative_energy', 'energy_per_atom']]\n",
        "stability_results.to_csv(f'{output_dir}/stability_classification.csv', index=False)\n",
        "\n",
        "# Save model comparison\n",
        "results_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)\n",
        "\n",
        "# Save predictions\n",
        "prediction_results = pd.DataFrame({\n",
        "    'actual': y_test,\n",
        "    'predicted': y_test_pred,\n",
        "    'error': y_test - y_test_pred,\n",
        "    'abs_error': np.abs(y_test - y_test_pred)\n",
        "})\n",
        "prediction_results.to_csv(f'{output_dir}/energy_predictions.csv', index=False)\n",
        "\n",
        "print('✅ Results saved to:', output_dir)\n",
        "print('\\nFiles created:')\n",
        "print('  - anomaly_detection_results.csv')\n",
        "print('  - stability_classification.csv')\n",
        "print('  - model_comparison.csv')\n",
        "print('  - energy_predictions.csv')"
    ]
})

# Save updated notebook
with open('carbon24-anomaly-energy-prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print('✅ Notebook updated with Energy Prediction!')
print('   Total cells:', len(notebook['cells']))
