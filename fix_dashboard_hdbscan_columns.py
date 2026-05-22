"""
Script sửa lỗi tên cột HDBSCAN trong dashboard
"""

# Đọc file dashboard
with open('carbon24_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix HDBSCAN probability column: membership_probability -> hdbscan_probability
content = content.replace("'membership_probability'", "'hdbscan_probability'")
content = content.replace('"membership_probability"', '"hdbscan_probability"')

# Ghi lại file
with open('carbon24_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Đã sửa lỗi tên cột HDBSCAN:")
print("   1. membership_probability -> hdbscan_probability")
print("\n🚀 Dashboard đã sẵn sàng!")
