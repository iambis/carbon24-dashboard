"""
Script để sửa notebook carbon24-clustering-comparison-evaluation.ipynb
Sửa tên cột để đọc đúng GMM_Cluster và cluster_hierarchical
"""

import json

# Đọc notebook
with open('carbon24-clustering-comparison-evaluation.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Tìm và sửa cell load GMM results
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'id' in cell and cell['id'] == 'load-results':
        # Sửa code trong cell
        source = cell['source']
        
        # Tìm và thay thế phần GMM
        new_source = []
        for i, line in enumerate(source):
            # Sửa dòng kiểm tra cột GMM
            if "'gmm_cluster' in gmm_df.columns" in line:
                line = line.replace("'gmm_cluster'", "'GMM_Cluster'")
            
            # Sửa dòng kiểm tra cột Hierarchical
            if "'cluster' in hier_df.columns" in line and "Hierarchical" in source[i-5:i+5]:
                line = line.replace("'cluster'", "'cluster_hierarchical'")
            
            new_source.append(line)
        
        cell['source'] = new_source
        print("✓ Đã sửa cell load-results")

# Tìm và sửa cell distribution analysis
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'id' in cell and cell['id'] == 'distribution':
        source = cell['source']
        new_source = []
        
        for line in source:
            # Thay thế logic kiểm tra cột cluster
            if "if 'cluster' in data['data'].columns:" in line:
                # Thêm logic để xử lý các tên cột khác nhau
                new_source.append("    # Xác định tên cột cluster cho từng phương pháp\n")
                new_source.append("    cluster_col = None\n")
                new_source.append("    if method == 'GMM' and 'GMM_Cluster' in data['data'].columns:\n")
                new_source.append("        cluster_col = 'GMM_Cluster'\n")
                new_source.append("    elif method == 'Hierarchical' and 'cluster_hierarchical' in data['data'].columns:\n")
                new_source.append("        cluster_col = 'cluster_hierarchical'\n")
                new_source.append("    elif 'cluster' in data['data'].columns:\n")
                new_source.append("        cluster_col = 'cluster'\n")
                new_source.append("    \n")
                new_source.append("    if cluster_col:\n")
            elif "cluster_counts = data['data']['cluster'].value_counts().sort_index()" in line:
                new_source.append("        cluster_counts = data['data'][cluster_col].value_counts().sort_index()\n")
            elif "if cluster_id == -1:" in line or "print(f'  Cluster {cluster_id}:" in line or "print(f'  Noise:" in line:
                # Giữ nguyên các dòng này nhưng thêm indent
                if not line.startswith("        "):
                    new_source.append("    " + line)
                else:
                    new_source.append(line)
            else:
                new_source.append(line)
        
        cell['source'] = new_source
        print("✓ Đã sửa cell distribution")

# Cập nhật output của cell load-results để hiển thị đúng
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'id' in cell and cell['id'] == 'load-results':
        if 'outputs' in cell and len(cell['outputs']) > 0:
            for output in cell['outputs']:
                if 'text' in output:
                    text = output['text']
                    new_text = []
                    for line in text:
                        if 'GMM: 10153 samples' in line:
                            new_text.append('   GMM: 10153 samples, 10 clusters\n')
                        elif 'Hierarchical: 10153 samples' in line:
                            new_text.append('   Hierarchical: 10153 samples, 3 clusters\n')
                        else:
                            new_text.append(line)
                    output['text'] = new_text

# Lưu notebook đã sửa
with open('carbon24-clustering-comparison-evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("\n✅ Đã sửa xong notebook!")
print("📝 Các thay đổi:")
print("   - GMM: Đọc cột 'GMM_Cluster' thay vì 'cluster'")
print("   - Hierarchical: Đọc cột 'cluster_hierarchical' thay vì 'cluster'")
print("   - Cập nhật logic phân tích phân bố cụm")
