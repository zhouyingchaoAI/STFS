# 配置文件完整指南

本文档详细说明 STFS_V1 系统的所有配置文件及其参数。

## 配置文件列表

### 核心配置文件

| 文件名 | 用途 | 必需 |
|--------|------|------|
| `db_config.yaml` | 数据库连接配置 | ✅ 是 |
| `task_config.yaml` | 任务调度配置 | ✅ 是 |
| `task_all_config.yaml` | 全局任务配置 | ❌ 否 |
| `task_chezhan_config.yaml` | 车站任务配置 | ❌ 否 |

### 模型配置文件

模型配置文件命名规则：`model_config_{flow_type}_{granularity}_{metric_type}.yaml`

**已存在的配置文件：**

#### 线网线路（xianwangxianlu）
- `model_config_xianwangxianlu_daily_F_PKLCOUNT.yaml` - 日客运量
- `model_config_xianwangxianlu_daily_F_ENTRANCE.yaml` - 日进站量
- `model_config_xianwangxianlu_daily_F_EXIT.yaml` - 日出站量
- `model_config_xianwangxianlu_daily_F_TRANSFER.yaml` - 日换乘量
- `model_config_xianwangxianlu_daily_F_BOARD_ALIGHT.yaml` - 日乘降量
- `model_config_xianwangxianlu_hourly_F_PKLCOUNT.yaml` - 小时客运量
- `model_config_xianwangxianlu_hourly_F_ENTRANCE.yaml` - 小时进站量
- `model_config_xianwangxianlu_hourly_F_EXIT.yaml` - 小时出站量
- `model_config_xianwangxianlu_hourly_F_TRANSFER.yaml` - 小时换乘量
- `model_config_xianwangxianlu_hourly_F_BOARD_ALIGHT.yaml` - 小时乘降量

#### 车站（chezhan）
- `model_config_chezhan_daily_*.yaml` - 车站日预测配置
- `model_config_chezhan_hourly_*.yaml` - 车站小时预测配置

#### 换乘（huhuan）
- `model_config_huhuan_hourly_F_ENTRANCE.yaml` - 换乘小时进站量
- `model_config_huhuan_hourly_F_EXIT.yaml` - 换乘小时出站量
- `model_config_huhuan_hourly_F_TRANSFER.yaml` - 换乘小时换乘量

### 其他配置
- `config/line_weights.json` - 线路算法权重配置
- `stationid_stationname_to_lineid.yaml` - 站点线路映射
- `supervisord.conf` - Supervisor进程管理配置

---

## 配置文件详解

### 1. 数据库配置 (db_config.yaml)

```yaml
db:
  server: "10.1.6.230"           # 数据库服务器IP
  user: "sa"                     # 数据库用户名
  password: "YourPassword"       # 数据库密码
  database: "master"             # 数据库名称
  port: 1433                     # 数据库端口

QUERY_START_DATE: 20230101       # 历史数据查询起始日期（YYYYMMDD）

STATION_FILTER_NAMES:            # 车站过滤列表（为空则查询所有）
  - 五一广场
  - 碧沙湖
  - 橘子洲
```

**参数说明：**

- `server`: SQL Server 服务器地址
- `user`: 数据库登录用户名
- `password`: 数据库登录密码（建议使用环境变量）
- `database`: 目标数据库名称
- `port`: 数据库端口，默认 1433
- `QUERY_START_DATE`: 查询历史数据的起始日期，影响训练数据范围
- `STATION_FILTER_NAMES`: 车站名称过滤列表，用于限制车站客流预测范围

**安全提示：**
- ⚠️ 不要将包含真实密码的配置文件提交到版本控制
- ✅ 使用 `db_config.example.yaml` 作为模板
- ✅ 生产环境建议使用环境变量或密钥管理服务

---

### 2. 任务调度配置 (task_config.yaml)

```yaml
host: "127.0.0.1"                # API服务地址
port: 4566                       # API服务端口

# 训练调度时间（24小时制）
train_schedule_times:
  - "07:15"                      # 每天7:15执行训练

# 预测调度时间
predict_schedule_times:
  - "08:00"                      # 每天8:00执行预测

# 训练算法
train_algorithm: knn             # 默认使用knn算法

# 日预测指标类型
predict_daily_metric_types:
  - F_PKLCOUNT                   # 客运量
  - F_ENTRANCE                   # 进站量
  - F_EXIT                       # 出站量
  - F_TRANSFER                   # 换乘量

# 小时预测指标类型
predict_hourly_metric_types:
  - F_PKLCOUNT
  - F_ENTRANCE
  - F_EXIT
  - F_TRANSFER

# 训练指标类型
train_daily_metric_types:
  - F_PKLCOUNT
  - F_ENTRANCE
  - F_EXIT
  - F_TRANSFER

train_hourly_metric_types:
  - F_PKLCOUNT
  - F_ENTRANCE
  - F_EXIT
  - F_TRANSFER

# 线网线路特定配置
xianwangxianlu_predict_daily_metric_types:
  - F_PKLCOUNT
  - F_ENTRANCE
  - F_EXIT
  - F_TRANSFER
  - F_BOARD_ALIGHT               # 额外支持乘降量

xianwangxianlu_predict_hourly_metric_types:
  - F_PKLCOUNT
  - F_ENTRANCE
  - F_EXIT
  - F_TRANSFER
  - F_BOARD_ALIGHT
```

**参数说明：**

- `host`: API服务器地址
- `port`: API服务器端口
- `train_schedule_times`: 训练任务执行时间列表，支持多个时间点
- `predict_schedule_times`: 预测任务执行时间列表
- `train_algorithm`: 默认训练算法（knn/lstm/prophet等）
- `*_metric_types`: 各场景需要预测/训练的指标类型

**时间格式：**
- 使用24小时制，格式为 "HH:MM"
- 示例：`"07:15"`, `"18:30"`
- 支持多个时间点，任务会在每个时间点执行

---

### 3. 模型配置文件

#### 通用参数（所有模型配置文件）

```yaml
current_version: '20250916'      # 当前模型版本（YYYYMMDD）
default_algorithm: knn           # 默认算法
model_root_dir: models/xianwangxianlu/daily/F_PKLCOUNT  # 模型存储路径

# 训练参数
train_params:
  # KNN参数
  n_neighbors: 5                 # K近邻数量
  lookback_days: 365             # 回溯天数（日预测）
  lookback_hours: 72             # 回溯小时数（小时预测）
  
  # 深度学习参数（LSTM/Transformer）
  batch_size: 32                 # 批次大小
  epochs: 100                    # 训练轮数
  learning_rate: 0.001           # 学习率
  patience: 10                   # 早停耐心值
  hidden_size: 64                # 隐藏层大小
  num_layers: 2                  # 网络层数

# 算法权重配置（全局）
algorithm_weights:
  knn: 0.8                       # KNN算法权重
  last_year_offset: 0.2          # 去年偏移算法权重

# 每条线路独立权重（可选）
line_algorithm_weights:
  '01':                          # 线路编号
    knn: 0.7
    last_year_offset: 0.3
  '02':
    knn: 0.6
    last_year_offset: 0.4
  '31':                          # 支线，波动大
    knn: 0.1
    last_year_offset: 0.9

# 预测因子
factors:
  - F_WEEK                       # 周数
  - F_HOLIDAYTYPE                # 节假日类型
  - F_HOLIDAYDAYS                # 节假日天数
  - F_HOLIDAYWHICHDAY            # 节假日第几天
  - F_DAYOFWEEK                  # 星期几
  - WEATHER_TYPE                 # 天气类型
  - F_YEAR                       # 年份
  # 小时预测额外因子
  - F_HOUR                       # 小时（0-23）
  - F_DATEFEATURES               # 日期特征
  - F_ISHOLIDAY                  # 是否节假日
  - F_ISNONGLI                   # 是否农历节日
  - F_ISYANGLI                   # 是否阳历节日
  - F_NEXTDAY                    # 次日类型
  - F_HOLIDAYTHDAY               # 节假日第几天
  - IS_FIRST                     # 是否首日

# 早晨配置（仅小时预测）
early_morning_config:
  cutoff_hour: 6                 # 早晨时段截止小时
  pure_offset_weight:            # 早晨时段使用纯历史偏移
    knn: 0.0
    last_year_offset: 1.0
```

#### 日预测配置示例 (model_config_xianwangxianlu_daily_F_PKLCOUNT.yaml)

```yaml
current_version: '20250916'
default_algorithm: knn
model_root_dir: models/xianwangxianlu/daily/F_PKLCOUNT

train_params:
  n_neighbors: 5
  lookback_days: 365

algorithm_weights:
  knn: 0.8
  last_year_offset: 0.2

factors:
  - F_WEEK
  - F_HOLIDAYTYPE
  - F_HOLIDAYDAYS
  - F_HOLIDAYWHICHDAY
  - F_DAYOFWEEK
  - WEATHER_TYPE
  - F_YEAR
```

#### 小时预测配置示例 (model_config_xianwangxianlu_hourly_F_PKLCOUNT.yaml)

```yaml
current_version: '20250916'
default_algorithm: knn
model_root_dir: models/xianwangxianlu/hourly/F_PKLCOUNT

train_params:
  n_neighbors: 5
  lookback_hours: 72

algorithm_weights:
  knn: 0.2
  last_year_offset: 0.8

early_morning_config:
  cutoff_hour: 6
  pure_offset_weight:
    knn: 0.0
    last_year_offset: 1.0

factors:
  - F_WEEK
  - F_DATEFEATURES
  - F_HOLIDAYTYPE
  - F_ISHOLIDAY
  - F_HOUR
  - WEATHER_TYPE
```

---

### 4. 线路权重配置 (config/line_weights.json)

```json
{
  "default_weights": {
    "knn": 0.6,
    "last_year_offset": 0.4
  },
  "line_weights": {
    "01": {
      "knn": 0.7,
      "last_year_offset": 0.3,
      "comment": "1号线，客流稳定"
    },
    "02": {
      "knn": 0.6,
      "last_year_offset": 0.4,
      "comment": "2号线"
    },
    "31": {
      "knn": 0.1,
      "last_year_offset": 0.9,
      "comment": "支线，波动大，更依赖历史"
    }
  }
}
```

**参数说明：**

- `default_weights`: 默认权重，未单独配置的线路使用此权重
- `line_weights`: 每条线路的独立权重配置
  - 键为线路编号（字符串格式）
  - `knn`: KNN算法权重（0.0-1.0）
  - `last_year_offset`: 去年同期偏移权重（0.0-1.0）
  - `comment`: 注释说明（可选）

**权重说明：**
- 两个权重之和应为 1.0（系统会自动归一化）
- KNN权重高：更依赖相似日期的预测
- 偏移权重高：更依赖去年同期的历史数据
- 建议主线权重偏向KNN，支线偏向偏移

---

### 5. Supervisor配置 (supervisord.conf)

```ini
[supervisord]
nodaemon=false
logfile=/STFS_V1/logs/supervisord.log
pidfile=/var/run/supervisord.pid

[program:stfs_api]
command=uvicorn server:app --host 0.0.0.0 --port 4566
directory=/STFS_V1
autostart=true
autorestart=true
stdout_logfile=/STFS_V1/logs/api_stdout.log
stderr_logfile=/STFS_V1/logs/api_stderr.log

[program:stfs_ui]
command=streamlit run main.py --server.address 0.0.0.0 --server.port 4577
directory=/STFS_V1
autostart=true
autorestart=true
stdout_logfile=/STFS_V1/logs/ui_stdout.log
stderr_logfile=/STFS_V1/logs/ui_stderr.log

[program:stfs_task]
command=python task.py
directory=/STFS_V1
autostart=true
autorestart=true
stdout_logfile=/STFS_V1/logs/task_stdout.log
stderr_logfile=/STFS_V1/logs/task_stderr.log
```

---

## 配置优先级

系统加载配置的优先级（从高到低）：

1. **API请求参数** - 通过API传入的参数
2. **模型配置文件** - `model_config_*.yaml`
3. **线路权重配置** - `config/line_weights.json`
4. **全局任务配置** - `task_all_config.yaml`
5. **默认任务配置** - `task_config.yaml`
6. **代码默认值** - 硬编码的默认值

---

## 配置最佳实践

### 1. 版本管理
- ✅ 使用日期作为模型版本号（YYYYMMDD）
- ✅ 定期更新 `current_version`
- ✅ 保留历史版本的配置文件

### 2. 参数调优
- 📊 根据预测效果调整算法权重
- 📊 主线建议 KNN:0.6-0.8, 偏移:0.2-0.4
- 📊 支线建议 KNN:0.1-0.3, 偏移:0.7-0.9
- 📊 新线路先使用默认权重，逐步调优

### 3. 安全配置
- 🔐 不要在配置文件中硬编码密码
- 🔐 使用 `.example` 文件作为模板
- 🔐 将敏感配置加入 `.gitignore`
- 🔐 生产环境使用环境变量

### 4. 性能优化
- ⚡ 根据数据量调整 `lookback_days/hours`
- ⚡ 调整 `n_neighbors` 平衡精度和速度
- ⚡ 深度学习模型减少 `batch_size` 节省内存

### 5. 监控和日志
- 📝 定期检查日志文件
- 📝 配置日志轮转避免磁盘占满
- 📝 监控预测准确率，及时调整参数

---

## 常见配置问题

### Q1: 修改配置后不生效？
**A:** 需要重启相关服务：
```bash
supervisorctl restart stfs_api
supervisorctl restart stfs_ui
supervisorctl restart stfs_task
```

### Q2: 如何为新线路添加配置？
**A:** 在 `config/line_weights.json` 中添加：
```json
"新线路编号": {
  "knn": 0.6,
  "last_year_offset": 0.4
}
```

### Q3: 数据库连接失败？
**A:** 检查 `db_config.yaml` 中的配置：
- 确认服务器地址和端口正确
- 确认用户名密码正确
- 确认网络连通性
- 检查防火墙设置

### Q4: 预测准确率低？
**A:** 调整参数：
1. 增加 `n_neighbors` 值（5 → 7 → 10）
2. 调整算法权重
3. 增加 `lookback_days/hours`
4. 检查历史数据质量
5. 尝试其他算法

### Q5: 内存不足？
**A:** 优化配置：
- 减少 `lookback_days/hours`
- 减少 `batch_size`
- 使用更轻量的算法（KNN → LightGBM）

---

## 配置文件检查清单

部署前请确认：

- [ ] `db_config.yaml` 配置正确，数据库可连接
- [ ] `task_config.yaml` 调度时间合理
- [ ] 模型配置文件存在且参数合理
- [ ] 线路权重配置完整
- [ ] 日志目录有写入权限
- [ ] 模型目录有写入权限
- [ ] 敏感信息已脱敏
- [ ] 已创建 `.gitignore`

---

## 获取帮助

如有配置问题，请：
1. 查看 [README.md](README.md) 文档
2. 查看 [常见问题](README.md#常见问题)
3. 查看日志文件 `logs/*.log`
4. 提交 [Issue](https://github.com/your-repo/STFS_V1/issues)

---

**最后更新**: 2025-01-15

