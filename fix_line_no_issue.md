# 修复线路名包含"/"的问题

## 问题描述
当线路名包含"/"（如"光达/会展中心"）时，会导致文件路径错误，因为"/"是路径分隔符。

## 解决方案

### 1. 添加清理函数
在 `enknn_model.py` 的 `DEFAULT_KNN_FACTORS` 之后添加：

```python
def sanitize_line_no(line_no: str) -> str:
    """
    清理线路名，将不适合作为文件名的字符替换为安全字符
    
    参数:
        line_no: 原始线路名或线路编号
        
    返回:
        清理后的线路名，适合用作文件名
    """
    if not isinstance(line_no, str):
        line_no = str(line_no)
    # 将路径分隔符和特殊字符替换为连字符
    sanitized = line_no.replace('/', '-').replace('\\', '-').replace(':', '-')
    sanitized = sanitized.replace('*', '-').replace('?', '-').replace('"', '-')
    sanitized = sanitized.replace('<', '-').replace('>', '-').replace('|', '-')
    # 去除首尾空格和点号
    sanitized = sanitized.strip('. ')
    # 如果清理后为空，使用默认值
    if not sanitized:
        sanitized = 'unknown'
    return sanitized
```

### 2. 修复所有使用线路名构建文件路径的地方

需要在以下位置添加 `safe_line_no = sanitize_line_no(line_no)` 并将路径中的 `{line_no}` 替换为 `{safe_line_no}`：

1. **train方法** (约348-349行)
   - 在 `version = model_version or self.version` 之后添加 `safe_line_no = sanitize_line_no(line_no)`
   - 将 `f"knn_line_{line_no}_daily` 改为 `f"knn_line_{safe_line_no}_daily`
   - 将 `f"knn_scaler_line_{line_no}_daily` 改为 `f"knn_scaler_line_{safe_line_no}_daily`

2. **_predict_knn方法** (约409-410行)
   - 在 `version = model_version or self.version` 之后添加 `safe_line_no = sanitize_line_no(line_no)`
   - 将 `f"knn_line_{line_no}_daily` 改为 `f"knn_line_{safe_line_no}_daily`
   - 将 `f"knn_scaler_line_{line_no}_daily` 改为 `f"knn_scaler_line_{safe_line_no}_daily`

3. **save_model_info方法** (约516行)
   - 在 `}` 之后、`info_path = ...` 之前添加 `safe_line_no = sanitize_line_no(line_no)`
   - 将 `f"model_info_line_{line_no}_daily` 改为 `f"model_info_line_{safe_line_no}_daily`

4. **diagnose_zero_predictions方法** (约554行)
   - 在 `version = self.version` 之后添加 `safe_line_no = sanitize_line_no(line_no)`
   - 将 `f"knn_line_{line_no}_daily` 改为 `f"knn_line_{safe_line_no}_daily`

## 测试
```python
from enknn_model import sanitize_line_no
assert sanitize_line_no('光达/会展中心') == '光达-会展中心'
assert sanitize_line_no('test/name') == 'test-name'
assert sanitize_line_no('正常线路') == '正常线路'
```

