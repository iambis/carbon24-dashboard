"""
Fix HDBSCAN column name in clustering comparison notebook
"""
import json

# Read the notebook
with open('carbon24-clustering-comparison-evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the HDBSCAN loading cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and cell['id'] == 'load-results':
        # Get the source code
        source = ''.join(cell['source'])
        
        # Find the HDBSCAN section and add the rename logic
        old_hdbscan = """# 4. HDBSCAN
print('\\n📊 Loading HDBSCAN results...')
try:
    hdbscan_df = pd.read_csv('hdbscan_phuc/hdbscan_results.csv')
    hdbscan_profile = pd.read_csv('hdbscan_phuc/hdbscan_cluster_profile.csv')
    results['HDBSCAN'] = {
        'data': hdbscan_df,
        'profile': hdbscan_profile,
        'n_clusters': len(hdbscan_df['cluster'].unique()) - (1 if -1 in hdbscan_df['cluster'].values else 0),
        'n_noise': (hdbscan_df['cluster'] == -1).sum() if 'cluster' in hdbscan_df.columns else 0
    }
    print(f'  ✅ HDBSCAN: {len(hdbscan_df)} samples, {results["HDBSCAN"]["n_clusters"]} clusters')
    print(f'     Noise points: {results["HDBSCAN"]["n_noise"]}')
except Exception as e:
    print(f'  ❌ Error loading HDBSCAN: {e}')"""
        
        new_hdbscan = """# 4. HDBSCAN
print('\\n📊 Loading HDBSCAN results...')
try:
    hdbscan_df = pd.read_csv('hdbscan_phuc/hdbscan_results.csv')
    hdbscan_profile = pd.read_csv('hdbscan_phuc/hdbscan_cluster_profile.csv')
    
    # Rename hdbscan_cluster to cluster for consistency
    if 'hdbscan_cluster' in hdbscan_df.columns:
        hdbscan_df = hdbscan_df.rename(columns={'hdbscan_cluster': 'cluster'})
    
    results['HDBSCAN'] = {
        'data': hdbscan_df,
        'profile': hdbscan_profile,
        'n_clusters': len(hdbscan_df['cluster'].unique()) - (1 if -1 in hdbscan_df['cluster'].values else 0),
        'n_noise': (hdbscan_df['cluster'] == -1).sum() if 'cluster' in hdbscan_df.columns else 0
    }
    print(f'  ✅ HDBSCAN: {len(hdbscan_df)} samples, {results["HDBSCAN"]["n_clusters"]} clusters')
    print(f'     Noise points: {results["HDBSCAN"]["n_noise"]}')
except Exception as e:
    print(f'  ❌ Error loading HDBSCAN: {e}')"""
        
        # Replace the source
        source = source.replace(old_hdbscan, new_hdbscan)
        
        # Convert back to list of lines
        cell['source'] = source.split('\n')
        # Add newlines back except for the last line
        cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line 
                         for i, line in enumerate(cell['source'])]
        
        print("✅ Updated HDBSCAN loading cell")
        break

# Save the updated notebook
with open('carbon24-clustering-comparison-evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook updated successfully!")
print("\n📝 Changes made:")
print("  - Added column rename: hdbscan_cluster → cluster")
print("  - Ensures consistency across all clustering methods")
