# PCA 3D Visualization Guide

## ✅ Đã Hoàn Thành!

PCA 3D đã được thêm vào dự án với **50.54% variance explained** (PC1: 24.57%, PC2: 15.38%, PC3: 10.59%)

## 📁 Files Đã Tạo

### 1. Interactive HTML Files (Standalone)
Mở trực tiếp trong browser, không cần chạy code:

```
carbon24_kmeans_results/
├── pca_3d_clusters.html          # 3D plot colored by clusters
├── pca_3d_energy.html             # 3D plot colored by relative energy
└── pca_3d_crystal_systems.html    # 3D plot colored by crystal systems
```

**Cách xem:**
- Double-click file HTML
- Hoặc: Kéo thả vào browser
- Hoặc: Right-click → Open with → Chrome/Firefox

### 2. Updated Data
```
carbon24_kmeans_results/carbon24_clustered.csv
```
Đã thêm 3 columns mới:
- `pca1_3d` - Principal Component 1
- `pca2_3d` - Principal Component 2  
- `pca3_3d` - Principal Component 3

### 3. Dashboard Integration
Dashboard (`carbon24_dashboard.py`) đã được cập nhật với:
- Toggle 2D/3D visualization
- 3 color options: Cluster, Energy, Crystal System
- Interactive 3D controls

## 🎮 Cách Sử Dụng

### Option 1: Xem HTML Files (Đơn giản nhất)

```bash
# Mở trong browser
start carbon24_kmeans_results/pca_3d_clusters.html
start carbon24_kmeans_results/pca_3d_energy.html
start carbon24_kmeans_results/pca_3d_crystal_systems.html
```

**3D Controls:**
- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag
- **Reset view**: Double-click

### Option 2: Dashboard (Interactive)

```bash
streamlit run carbon24_dashboard.py
```

Trong dashboard:
1. Chọn "🎯 Phân cụm K-means"
2. Tab "🎨 Visualization"
3. Chọn "3D" radio button
4. Chọn color by: Cluster / Energy / Crystal System

### Option 3: Jupyter Notebook

Thêm cell này vào notebook:

```python
import plotly.express as px

# Load data with PCA 3D
df = pd.read_csv('carbon24_kmeans_results/carbon24_clustered.csv')

# Create 3D plot
fig = px.scatter_3d(df, 
                    x='pca1_3d', 
                    y='pca2_3d', 
                    z='pca3_3d',
                    color='cluster',
                    title='PCA 3D: Clusters',
                    opacity=0.7)

fig.update_traces(marker=dict(size=3))
fig.show()
```

## 📊 Thông Tin PCA 3D

### Variance Explained:
- **PC1**: 24.57% - Chiều biến thiên lớn nhất
- **PC2**: 15.38% - Chiều biến thiên thứ hai
- **PC3**: 10.59% - Chiều biến thiên thứ ba
- **Total**: 50.54% - Tổng variance được giữ lại

### Ý Nghĩa:
- 3 chiều này giữ lại **hơn 50%** thông tin từ 22 features gốc
- Giúp visualize clusters trong không gian 3D
- Dễ phát hiện patterns và outliers

## 🎨 3 Loại Visualization

### 1. By Clusters (`pca_3d_clusters.html`)
- Mỗi cluster có màu riêng
- Dễ thấy sự phân tách giữa các clusters
- Hover để xem cluster ID và energy

### 2. By Relative Energy (`pca_3d_energy.html`)
- Gradient màu theo năng lượng
- Viridis colorscale (xanh → vàng)
- Thấy rõ phân bố năng lượng trong không gian

### 3. By Crystal Systems (`pca_3d_crystal_systems.html`)
- Mỗi crystal system có màu riêng
- 7 systems: triclinic, monoclinic, orthorhombic, cubic, hexagonal, trigonal, tetragonal
- Thấy mối quan hệ giữa structure và PCA space

## 💡 Tips Visualization

### Góc nhìn tốt nhất:
```python
fig.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)  # Góc nhìn từ trên cao
        )
    )
)
```

### Tùy chỉnh marker:
```python
fig.update_traces(
    marker=dict(
        size=3,              # Kích thước điểm
        opacity=0.7,         # Độ trong suốt
        line=dict(
            width=0.5,       # Viền
            color='white'    # Màu viền
        )
    )
)
```

### Export ảnh:
- Click camera icon trong plot
- Chọn "Download plot as png"
- Hoặc code:
```python
fig.write_image("pca_3d.png", width=1200, height=900)
```

## 🔄 Regenerate PCA 3D

Nếu cần tạo lại (sau khi clustering lại):

```bash
python add_pca_3d_to_notebook.py
```

Script sẽ:
1. Load dữ liệu clustered
2. Tính PCA 3D
3. Thêm vào dataframe
4. Tạo 3 HTML files
5. Save updated CSV

## 📈 So Sánh 2D vs 3D

| Aspect | 2D PCA | 3D PCA |
|--------|--------|--------|
| Variance | ~40% | ~50% |
| Visualization | Dễ hơn | Chi tiết hơn |
| Interpretation | Đơn giản | Phức tạp hơn |
| Use case | Quick overview | Deep analysis |

**Khuyến nghị:**
- Dùng **2D** cho presentations, reports
- Dùng **3D** cho exploratory analysis, demos

## 🎯 Demo Presentation

### Luồng demo với PCA 3D:

1. **Giới thiệu** (2D PCA)
   - "Đây là visualization 2D của clusters"
   - Chỉ ra các clusters rõ ràng

2. **Chuyển sang 3D**
   - "Bây giờ xem trong không gian 3D"
   - Rotate để show các góc khác nhau

3. **Highlight insights**
   - "Cluster X tách biệt rõ ràng"
   - "Energy thấp tập trung ở vùng này"
   - "Crystal systems có patterns riêng"

4. **Interactive demo**
   - Cho audience thử rotate
   - Zoom vào clusters cụ thể
   - Hover để xem details

## 🐛 Troubleshooting

### HTML không mở được:
```bash
# Thử browser khác
start chrome carbon24_kmeans_results/pca_3d_clusters.html
start firefox carbon24_kmeans_results/pca_3d_clusters.html
```

### Dashboard không hiện 3D:
```bash
# Chạy lại script
python add_pca_3d_to_notebook.py

# Restart dashboard
streamlit run carbon24_dashboard.py
```

### Plot quá chậm:
```python
# Giảm số điểm
df_sample = df.sample(5000)
fig = px.scatter_3d(df_sample, ...)
```

## 📚 References

- [Plotly 3D Scatter](https://plotly.com/python/3d-scatter-plots/)
- [PCA Explained](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Interactive Visualization Best Practices](https://plotly.com/python/3d-camera-controls/)

## ✨ Next Steps

- [ ] Add animation (rotate automatically)
- [ ] Add cluster centroids in 3D
- [ ] Add convex hulls around clusters
- [ ] Export to PowerPoint with 3D views
- [ ] Add VR/AR support (optional)

---

**Created:** 2026-05-20  
**Tool:** Plotly + Scikit-learn  
**Variance Explained:** 50.54%
