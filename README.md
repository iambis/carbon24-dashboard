# Carbon-24 Dashboard

Dự án phân tích cấu trúc Carbon-24: tiền xử lý đặc trưng, phân cụm, phân tích độ ổn định, dự đoán năng lượng và trực quan hóa bằng Streamlit.

## Bắt đầu nhanh

Từ thư mục gốc của dự án, cài các phụ thuộc:

```bash
pip install -r requirements.txt
```

Chạy dashboard:

```bash
streamlit run scripts/carbon24_interactive_dashboard.py
```

Hoặc trên Windows, chạy `run_dashboard.bat`. Dashboard mở tại `http://localhost:8501`.

## Chạy phân tích độ ổn định

```bash
python scripts/carbon24-stability-analysis.py
```

Script dùng dữ liệu đặc trưng đã có trong dự án và cập nhật các đầu ra trong `carbon24_stability_analysis/`:

- `cluster_stability_classification.csv`
- `prediction_model_comparison.csv`
- `best_model_predictions.csv`
- `feature_importance.csv`
- `ANALYSIS_REPORT.txt`
- `figures/` (các biểu đồ PNG)

Để tạo báo cáo PDF từ các kết quả hiện có:

```bash
python scripts/generate_pdf_report.py
```

## Cấu trúc thư mục

```text
carbon24-dashboard/
├── scripts/                         # Script Python chính
│   ├── carbon24_interactive_dashboard.py
│   ├── carbon24-stability-analysis.py
│   ├── carbon24_anomaly_detection.py
│   ├── carbon24_energy_prediction.py
│   └── generate_pdf_report.py
├── notebooks/                       # Notebook thử nghiệm và phân tích
├── docs/                            # Hướng dẫn chi tiết
├── data/                            # Dữ liệu đầu vào, gồm carbon.csv
├── assets/images/                   # Ảnh minh họa và biểu đồ dùng lại
├── archives/                        # Các gói ZIP lưu trữ
├── web/                             # Giao diện HTML/CSS/JS tĩnh
├── results/
│   ├── csv/                         # Vị trí gom CSV khi cần xuất chung
│   └── png/                         # Vị trí gom PNG khi cần xuất chung
├── carbon24_stability_analysis/     # Kết quả phân tích độ ổn định
├── carbon24_anomaly_detection/      # Kết quả phát hiện bất thường
├── carbon24_energy_results/         # Kết quả dự đoán năng lượng
├── carbon24_gmm_results/            # Kết quả GMM
├── carbon24_kmeans_results/         # Kết quả K-means
├── carbon24_hierarchical_baseline/  # Kết quả Hierarchical Clustering
├── hdbscan_phuc/                    # Kết quả HDBSCAN
├── requirements.txt
├── requirements_dashboard.txt
└── run_dashboard.bat
```

## Dữ liệu và kết quả

Kết quả được lưu theo từng phương pháp để không trộn lẫn đầu ra:

| Khu vực | Nội dung |
| --- | --- |
| `carbon24_stability_analysis/` | Phân loại độ ổn định, dự đoán năng lượng và biểu đồ liên quan |
| `carbon24_anomaly_detection/` | Kết quả anomaly detection, bảng CSV và figures |
| `carbon24_energy_results/` | So sánh mô hình, dự đoán và feature importance |
| `carbon24_gmm_results/` | Kết quả, bảng và figures GMM |
| `carbon24_kmeans_results/` | Dữ liệu và trực quan hóa K-means |
| `carbon24_hierarchical_baseline/` | Bảng kết quả và figures Hierarchical |
| `hdbscan_phuc/` | Dữ liệu và profile HDBSCAN |

Thư mục `results/csv/` và `results/png/` là nơi dành cho việc gom các đầu ra đã chọn. Chúng không tự sao chép dữ liệu từ các thư mục kết quả chuyên biệt.

## Tài liệu

- [Mục lục tài liệu](docs/INDEX.md)
- [Hướng dẫn bắt đầu](docs/GETTING_STARTED.md)
- [Hướng dẫn phân tích độ ổn định](docs/HUONG_DAN_STABILITY_ANALYSIS.md)
- [Hướng dẫn anomaly detection](docs/ANOMALY_DETECTION_GUIDE.md)

Lưu ý: một số tài liệu cũ trong `docs/` có thể vẫn nêu tên script ở vị trí cũ; hãy ưu tiên các lệnh trong README này và dùng đường dẫn `scripts/...`.

## Các script chính

| Script | Mục đích |
| --- | --- |
| `scripts/carbon24_interactive_dashboard.py` | Dashboard Streamlit |
| `scripts/carbon24-stability-analysis.py` | Phân tích độ ổn định và mô hình dự đoán |
| `scripts/carbon24_anomaly_detection.py` | Phát hiện điểm bất thường |
| `scripts/carbon24_energy_prediction.py` | Dự đoán năng lượng |
| `scripts/carbon24_feature_selection.py` | Lựa chọn đặc trưng |
| `scripts/carbon24_pipeline_3tier.py` | Pipeline phân tích ba tầng |
| `scripts/prepare_cif_cache.py` | Tạo lại CIF cache khi cần cho chế độ xem 3D |

## Lưu ý vận hành

- Chạy lệnh từ thư mục gốc của dự án để các đường dẫn dữ liệu tương đối hoạt động đúng.
- Dashboard cần các CSV hiện có trong các thư mục `carbon24_*`; hãy chạy các phân tích tương ứng nếu chúng bị thiếu.
- File PDF và CIF cache không được lưu sẵn để giữ repository gọn nhẹ; chúng có thể được tạo lại bằng script phù hợp khi cần.
