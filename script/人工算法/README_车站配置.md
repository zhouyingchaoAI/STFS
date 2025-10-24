# 车站预测功能配置说明

## ✅ 已完成功能

1. **配置文件支持** (`config.yaml`)
   - 支持车站筛选，限制预测的车站数量
   - 支持数据库配置
   - 支持查询起始日期配置

2. **车站预测功能**
   - 完全实现车站客流预测逻辑
   - 支持多年历史数据对比
   - 支持预测准确率计算
   - 前端页面完整

3. **性能优化**
   - 通过车站过滤减少数据量
   - 避免查询所有143个车站导致超时

## 📝 配置文件说明

### 配置文件位置
`/STFS_V1/script/人工算法/config.yaml`

### 配置项说明

```yaml
# 数据查询起始日期
QUERY_START_DATE: "20230101"

# 车站筛选配置
# - 如果为空列表 [] 或注释掉，则查询所有车站（143个，可能较慢）
# - 建议：选择10-20个重点车站可大幅提升预测速度
# - 注意：车站名称必须与数据库中完全一致（包括空格、斜杠等）
STATION_FILTER_NAMES:
  - 五一广场
  - 侯家塘
  - 人民东路
  # 更多车站...

# 数据库配置
db:
  server: "10.1.6.230"
  port: 1433
  user: "sa"
  password: "YourStrong!Passw0rd"
  database: "master"
```

## 🔍 如何获取正确的车站名称

由于数据库中车站名称必须完全匹配，建议使用以下SQL查询获取实际的车站名称列表：

```python
from yunying import get_db_conn
import pandas as pd

conn = get_db_conn()
query = '''
SELECT DISTINCT TOP 50 STATION_NAME
FROM [StationFlowPredict].[dbo].[STATION_FLOW_HISTORY]
WHERE STATION_NAME IS NOT NULL
'''
df = pd.read_sql(query, conn)
conn.close()

print('数据库中的车站名称：')
for name in df['STATION_NAME'].tolist():
    print(f'  - {name}')
```

或者直接使用SQL Server Management Studio执行：

```sql
SELECT DISTINCT STATION_NAME
FROM [StationFlowPredict].[dbo].[STATION_FLOW_HISTORY]
WHERE STATION_NAME IS NOT NULL
ORDER BY STATION_NAME
```

## 🚀 使用方法

1. **编辑配置文件**
   ```bash
   vi /STFS_V1/script/人工算法/config.yaml
   ```

2. **添加要预测的车站**
   - 从数据库查询实际车站名称
   - 复制准确的名称到`STATION_FILTER_NAMES`列表
   - 建议选择10-20个重点车站

3. **重启Flask服务**
   ```bash
   cd /STFS_V1/script/人工算法
   pkill -f "python.*web_app"
   python3 web_app.py &
   ```

4. **访问Web界面**
   - 浏览器打开: http://localhost:4566
   - 切换到"🚉 车站预测"标签页
   - 选择日期范围、指标类型和历史年限
   - 点击"开始预测"

## 📊 性能建议

| 车站数量 | 预测天数 | 历史年限 | 预计耗时 |
|---------|---------|---------|---------|
| 10个车站 | 5天 | 2年 | ~10秒 |
| 20个车站 | 5天 | 2年 | ~20秒 |
| 50个车站 | 5天 | 2年 | ~1分钟 |
| 143个车站（全部） | 5天 | 2年 | ~3分钟+ |

**推荐配置**：10-20个重点车站 + 5天预测 + 2年历史

## ⚠️ 常见问题

### 1. 查询结果为0行
**原因**：车站名称与数据库中不匹配
**解决**：使用上述SQL查询获取准确的车站名称

### 2. 服务器超时/崩溃
**原因**：车站数量过多导致计算量大
**解决**：减少车站数量（建议10-20个）或减少预测天数

### 3. "Failed to fetch" 错误
**原因**：Flask服务未启动或崩溃
**解决**：
```bash
pkill -f "python.*web_app"
cd /STFS_V1/script/人工算法
python3 web_app.py &
```

## 📦 依赖检查

确保已安装必要的Python包：
```bash
pip install pyyaml pymssql pandas flask -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 🎯 功能验证

测试配置是否正确：

```bash
cd /STFS_V1/script/人工算法
python3 -c "
from yunying import STATION_FILTER_NAMES, DB_CONFIG
print('车站过滤:', STATION_FILTER_NAMES)
print('数据库:', DB_CONFIG['server'])
"
```

测试车站数据查询：

```bash
python3 -c "
from yunying import read_station_daily_flow_history
df = read_station_daily_flow_history('F_PKLCOUNT', '20241001', '20241005')
print(f'查询结果: {len(df)}行, {df[\"F_LINENO\"].nunique()}个车站')
"
```

## 📞 技术支持

如有问题，请检查：
1. `config.yaml` 文件格式是否正确（YAML语法）
2. 车站名称是否与数据库完全一致
3. Flask服务是否正常运行
4. 数据库连接是否正常

---
**最后更新**: 2025-10-23

