# 🔧 Sửa lỗi: NameError: name 'stats' is not defined

## ❌ Lỗi gặp phải

```python
NameError: name 'stats' is not defined
```

Lỗi này xảy ra khi sử dụng `stats.zscore()` mà chưa import `scipy.stats`.

## ✅ Đã sửa

### Files đã được cập nhật:

1. **carbon24-anomaly-energy-prediction.ipynb**
   - Thêm `from scipy import stats` vào cell import

2. **.ipynb_checkpoints/carbon24-anomaly-energy-prediction-checkpoint.ipynb**
   - Thêm `from scipy import stats` vào cell import

3. **create_anomaly_nb.py**
   - Thêm `from scipy import stats` vào phần tạo cell import

### Code đã thêm:

```python
from scipy import stats
```

## 🚀 Cách sử dụng

### Trong Jupyter Notebook:

#### Option 1: Restart Kernel (Khuyến nghị)
1. Mở notebook: `carbon24-anomaly-energy-prediction.ipynb`
2. Click **Kernel** → **Restart & Run All**
3. Hoặc nhấn `00` (zero zero) để restart kernel

#### Option 2: Chạy lại cell import
1. Mở notebook
2. Tìm cell đầu tiên (cell import)
3. Chạy lại cell đó (Shift + Enter)
4. Tiếp tục chạy các cell sau

### Trong Python Script:

Nếu tạo notebook mới từ `create_anomaly_nb.py`:

```bash
python create_anomaly_nb.py
```

Notebook mới sẽ có đầy đủ import.

## 🧪 Kiểm tra

Chạy test script để xác nhận:

```bash
python test_scipy_import.py
```

**Expected output:**
```
================================================================================
TESTING SCIPY.STATS IMPORT
================================================================================

1️⃣ Testing scipy.stats import...
✅ scipy.stats imported successfully

2️⃣ Testing stats.zscore function...
✅ stats.zscore works correctly

3️⃣ Testing np.abs(stats.zscore(...))...
✅ np.abs(stats.zscore(...)) works correctly

4️⃣ Testing with pandas Series...
✅ Works with pandas Series

5️⃣ Testing anomaly detection logic...
✅ Anomaly detection logic works

================================================================================
✅ ALL TESTS PASSED!
================================================================================
```

## 📝 Giải thích

### Tại sao cần import scipy.stats?

`stats.zscore()` là một hàm từ thư viện `scipy.stats` dùng để tính Z-score:

```python
z = (x - mean) / std
```

Z-score cho biết một giá trị cách trung bình bao nhiêu độ lệch chuẩn.

### Sử dụng trong Anomaly Detection:

```python
from scipy import stats
import numpy as np

# Calculate Z-scores
z_scores = np.abs(stats.zscore(df['relative_energy']))

# Detect anomalies (|z| > 3)
df['anomaly_zscore'] = (z_scores > 3).astype(int) * -1 + (z_scores <= 3).astype(int)

# Count anomalies
n_anomalies = (df['anomaly_zscore'] == -1).sum()
```

**Ngưỡng |z| > 3:**
- Giá trị cách trung bình > 3 độ lệch chuẩn
- Xác suất xảy ra: ~0.3% (rất hiếm)
- Được coi là outlier/anomaly

## 🔍 Các file khác đã có import đúng

Các file sau **KHÔNG** cần sửa vì đã có import:

✅ `carbon24_preprocessing_v2.py`
✅ `carbon24-preprocessing.ipynb`
✅ `carbon24-processing.ipynb`
✅ `build_preprocessing_nb.py`

## 💡 Best Practice

### Khi tạo notebook mới:

Luôn include các import cần thiết trong cell đầu tiên:

```python
# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis
from scipy import stats

# Machine Learning
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
# ... other imports
```

### Khi gặp NameError:

1. **Kiểm tra import**: Xem biến/hàm có được import chưa
2. **Kiểm tra typo**: Đảm bảo tên biến đúng
3. **Restart kernel**: Đôi khi kernel cache cũ

## 🐛 Troubleshooting

### Lỗi: ModuleNotFoundError: No module named 'scipy'

**Giải pháp:**
```bash
pip install scipy
```

### Lỗi vẫn còn sau khi sửa

**Giải pháp:**
1. Restart Jupyter kernel
2. Clear output: Cell → All Output → Clear
3. Run All Cells

### Import thành công nhưng vẫn lỗi

**Kiểm tra:**
```python
# In notebook cell
import scipy
print(scipy.__version__)

from scipy import stats
print(dir(stats))  # Should see 'zscore' in list
```

## 📚 Tài liệu tham khảo

- [SciPy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Z-score Wikipedia](https://en.wikipedia.org/wiki/Standard_score)
- [Anomaly Detection with Z-score](https://towardsdatascience.com/anomaly-detection-with-z-score-b3a3f1e3b3f3)

## ✅ Checklist

Sau khi sửa, kiểm tra:

- [ ] File notebook đã có `from scipy import stats`
- [ ] Restart kernel
- [ ] Run All Cells thành công
- [ ] Cell sử dụng `stats.zscore()` chạy không lỗi
- [ ] Test script pass: `python test_scipy_import.py`

---

**Ngày sửa:** 2026-05-21  
**Người sửa:** Kiro AI Assistant  
**Status:** ✅ Fixed
