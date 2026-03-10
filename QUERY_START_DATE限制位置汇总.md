# QUERY_START_DATE 限制位置汇总

## 说明
`QUERY_START_DATE` 默认值为 `20230101`，在多个数据库读取函数中被用作全局起始日期限制，可能导致数据输出变少。

**当前配置值：** `20230101` (定义在 `db_config.yaml` 或默认配置中，第55行)

---

## 所有使用 QUERY_START_DATE 限制的函数位置

### 1. `read_station_daily_flow_history()` - 车站日客流历史数据
**文件：** `db_utils.py`  
**行号：** 第 355 行  
**位置：** WHERE 子句中  
**代码：**
```python
WHERE 
    REPLACE(S.SQUAD_DATE, '-', '') >= '{QUERY_START_DATE}'
    {station_filter}
```
**影响：** 限制车站日客流历史数据的查询起始日期

---

### 2. `read_station_hourly_flow_history()` - 车站小时客流历史数据
**文件：** `db_utils.py`  
**行号：** 第 586 行  
**位置：** where_conditions 列表中添加  
**代码：**
```python
# 增加全局起始日期限制
where_conditions.append(f"S.SQUAD_DATE >= '{QUERY_START_DATE}'")
```
**影响：** 限制车站小时客流历史数据的查询起始日期

---

### 3. `read_station_hourly_flow_history_old()` - 车站小时客流历史数据（旧版本）
**文件：** `db_utils.py`  
**行号：** 第 771 行  
**位置：** where_conditions 列表中添加  
**代码：**
```python
# 增加全局起始日期限制
where_conditions.append(f"S.SQUAD_DATE >= '{QUERY_START_DATE}'")
```
**影响：** 限制车站小时客流历史数据的查询起始日期（旧版本函数）

---

### 4. `fetch_holiday_features()` - 节假日特征数据
**文件：** `db_utils.py`  
**行号：** 第 876 行  
**位置：** where_conditions 列表中添加  
**代码：**
```python
# 增加全局起始日期限制
where_conditions.append(f"CC.F_DATE >= {QUERY_START_DATE}")
```
**影响：** 限制节假日特征数据的查询起始日期

---

### 5. `read_line_daily_flow_history()` - 线路日客流历史数据
**文件：** `db_utils.py`  
**行号：** 第 999 行  
**位置：** WHERE 子句中  
**代码：**
```python
WHERE 
    L.CREATOR = 'chency' AND L.F_DATE >= {QUERY_START_DATE}
```
**影响：** 限制线路日客流历史数据的查询起始日期

---

## 未使用 QUERY_START_DATE 的函数

### `read_line_hourly_flow_history()` - 线路小时客流历史数据
**文件：** `db_utils.py`  
**行号：** 第 390 行开始  
**说明：** 此函数只使用 `query_start_date` 参数和 `CREATOR='chency'` 限制，**没有使用 QUERY_START_DATE 全局限制**

---

## 配置位置

### 定义位置
- **文件：** `db_utils.py`
- **行号：** 第 24 行（默认配置）、第 55 行（从配置加载）
- **代码：**
  ```python
  DEFAULT_CONFIG = {
      ...
      "QUERY_START_DATE": 20230101,
      ...
  }
  
  QUERY_START_DATE = _config.get("QUERY_START_DATE", 20230101)
  ```

### 配置文件
- **文件：** `db_config.yaml`
- **字段：** `QUERY_START_DATE`
- **默认值：** `20230101`

---

## 影响分析

如果 `QUERY_START_DATE` 设置为 `20230101`（2023年1月1日），那么：
1. 所有早于 2023-01-01 的历史数据都会被过滤掉
2. 如果需要查询更早的历史数据（如2022年、2021年），需要修改此配置值
3. 修改方式：
   - 方法1：修改 `db_config.yaml` 文件中的 `QUERY_START_DATE` 值
   - 方法2：修改 `db_utils.py` 中的 `DEFAULT_CONFIG` 默认值

---

## 建议

1. **如果需要查询更早的历史数据**：
   - 将 `QUERY_START_DATE` 设置为更早的日期（如 `20200101` 或更早）
   - 或者设置为 `None` 或 `0` 来移除限制（需要修改代码）

2. **如果只需要查询最近的数据**：
   - 保持当前配置，可以提高查询性能

3. **如果需要灵活控制**：
   - 考虑将 `QUERY_START_DATE` 改为函数参数，而不是全局配置

