"""
Script để xóa tất cả output cũ trong notebook
Sau đó bạn chỉ cần chạy lại notebook trong Jupyter
"""

import json

# Đọc notebook
with open('carbon24-clustering-comparison-evaluation.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Xóa tất cả output
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# Lưu notebook
with open('carbon24-clustering-comparison-evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Đã xóa tất cả output cũ!")
print("📝 Bây giờ hãy:")
print("   1. Mở Jupyter Notebook")
print("   2. Mở file carbon24-clustering-comparison-evaluation.ipynb")
print("   3. Chọn 'Kernel' → 'Restart & Run All'")
print("   4. Xem kết quả mới!")
