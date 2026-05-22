# Hướng Dẫn Chạy Dashboard V2

## ✅ Vấn đề đã được sửa

File `carbon24_dashboard-v2.py` đã được sửa lỗi cú pháp. Vấn đề là có đoạn code thừa sau khi gọi `render_energy_prediction_tab()`.

## 🚀 Cách chạy Dashboard

### Bước 1: Kích hoạt môi trường ảo (nếu có)
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Bước 2: Chạy Dashboard
```bash
streamlit run carbon24_dashboard-v2.py
```

**LƯU Ý QUAN TRỌNG:** 
- ❌ KHÔNG chạy bằng `python carbon24_dashboard-v2.py`
- ✅ PHẢI chạy bằng `streamlit run carbon24_dashboard-v2.py`

### Bước 3: Mở trình duyệt
Dashboard sẽ tự động mở tại: `http://localhost:8501`

## 📋 Các trang có sẵn trong Dashboard V2

1. **🔬 Research Workflow** - Quy trình nghiên cứu 4 bước
2. **📊 Tổng quan** - Tổng quan dự án
3. **🔍 Khảo sát dữ liệu** - Phân tích dữ liệu
4. **🎯 Phân cụm K-means** - Kết quả K-means clustering
5. **🎲 Phân cụm GMM** - Kết quả GMM clustering
6. **🌳 Phân cụm Hierarchical** - Kết quả Hierarchical clustering
7. **🔵 Phân cụm HDBSCAN** - Kết quả HDBSCAN clustering
8. **📊 So sánh thuật toán** - So sánh các phương pháp clustering
9. **🔍 Phát hiện dị biệt** - Anomaly detection
10. **🔬 Pipeline 3 Tầng** - Pipeline 3 tầng
11. **🏷️ Ground-Truth Labeling** - Gán nhãn khoa học
12. **⚡ Dự đoán năng lượng** - Dự đoán năng lượng cấu trúc

## 🔧 Khắc phục sự cố

### Lỗi: "ModuleNotFoundError"
```bash
pip install streamlit pandas numpy plotly scikit-learn
```

### Lỗi: "FileNotFoundError" khi load dữ liệu
Đảm bảo các thư mục sau tồn tại và có dữ liệu:
- `carbon24_preprocessing_results/`
- `carbon24_kmeans_results/`
- `carbon24_gmm_results/`
- `carbon24_hierarchical_baseline/`
- `hdbscan_phuc/`
- `carbon24_clustering_comparison_results/`
- `carbon24_anomaly_detection/`
- `carbon24_pipeline_results/`
- `carbon24_energy_results/`

### Dashboard không hiển thị gì
1. Kiểm tra console có lỗi không
2. Thử refresh trình duyệt (Ctrl+F5)
3. Kiểm tra port 8501 có bị chiếm không:
   ```bash
   # Windows
   netstat -ano | findstr :8501
   
   # Linux/Mac
   lsof -i :8501
   ```

## 📝 Ghi chú

- Dashboard sử dụng Streamlit framework
- Dữ liệu được cache để tăng tốc độ load
- Các biểu đồ được tạo bằng Plotly (interactive)
- Hỗ trợ download kết quả dưới dạng CSV

## 🎯 Tính năng nổi bật của Dashboard V2

### Research Workflow (Trang mới)
- **Bước 1:** Overview & Anomaly Filter với PCA 3D
- **Bước 2:** Clustering & MP Matching (click vào cụm để xem chi tiết)
- **Bước 3:** Model Leaderboard + Feature Importance
- **Bước 4:** Live Prediction (nhập thông số → dự đoán năng lượng ngay)

### Các tính năng khác
- Phân tích chi tiết cho từng phương pháp clustering
- So sánh hiệu năng các thuật toán
- Phát hiện dị biệt với 3 phương pháp (HDBSCAN, Low Probability, Isolation Forest)
- Pipeline 3 tầng: Noise → K-means → GMM
- Ground-Truth Labeling: So khớp với Materials Project
- Dự đoán năng lượng với 4 mô hình ML

## 💡 Tips

1. Sử dụng sidebar để điều hướng giữa các trang
2. Trang "Research Workflow" là điểm khởi đầu tốt nhất
3. Mỗi trang có thể download kết quả dưới dạng CSV
4. Các biểu đồ có thể zoom, pan và hover để xem chi tiết
