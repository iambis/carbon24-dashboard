"""
Script sửa lỗi tên cột GMM trong dashboard
"""

# Đọc file dashboard
with open('carbon24_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix GMM cluster column: gmm_cluster -> GMM_Cluster
content = content.replace("'gmm_cluster'", "'GMM_Cluster'")
content = content.replace('"gmm_cluster"', '"GMM_Cluster"')

# Fix GMM probability column: max_probability -> Max_Probability
content = content.replace("'max_probability'", "'Max_Probability'")
content = content.replace('"max_probability"', '"Max_Probability"')

# Ghi lại file
with open('carbon24_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Đã sửa các lỗi tên cột GMM:")
print("   1. gmm_cluster -> GMM_Cluster")
print("   2. max_probability -> Max_Probability")
print("\n🚀 Dashboard đã sẵn sàng!")
