# 项目架构文档

本文档详细描述 STFS_V1 系统的架构设计、模块划分和数据流。

## 目录

- [系统概览](#系统概览)
- [技术架构](#技术架构)
- [模块设计](#模块设计)
- [数据流图](#数据流图)
- [数据库设计](#数据库设计)
- [API设计](#api设计)
- [部署架构](#部署架构)

---

## 系统概览

STFS_V1 是一个基于微服务架构的地铁客流预测系统，采用前后端分离设计。

### 核心特性

- **模块化设计**：各功能模块独立，便于维护和扩展
- **多算法支持**：支持多种机器学习算法，可灵活切换
- **版本化管理**：模型按日期版本化，支持回溯和对比
- **分布式部署**：支持水平扩展，满足高并发需求
- **自动化运维**：支持自动训练、预测和数据更新

---

## 技术架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                          负载均衡层 (Nginx)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
┌───────────▼──────────┐        ┌──────────▼──────────┐
│   Web UI Service     │        │   API Service       │
│   (Streamlit: 4577)  │        │   (FastAPI: 4566)   │
│                      │        │                     │
│  - 交互式界面         │        │  - RESTful API      │
│  - 可视化展示         │        │  - Swagger文档      │
│  - 参数配置          │        │  - 数据验证         │
└───────────┬──────────┘        └──────────┬──────────┘
            │                              │
            └──────────┬───────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │     业务逻辑层 (Python)      │
        │                             │
        │  - 预测流程编排              │
        │  - 数据预处理               │
        │  - 模型调度                 │
        │  - 结果后处理               │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │     模型算法层 (ML/DL)       │
        │                             │
        │  ┌─────────┬─────────┬───┐ │
        │  │   KNN   │  LSTM   │...│ │
        │  ├─────────┼─────────┼───┤ │
        │  │ Prophet │ XGBoost │...│ │
        │  └─────────┴─────────┴───┘ │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │     数据访问层 (DAL)         │
        │                             │
        │  - 数据库操作               │
        │  - 缓存管理                 │
        │  - 文件存储                 │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │      数据存储层              │
        │                             │
        │  ┌──────────┬──────────┐   │
        │  │ SQL Server│  Redis  │   │
        │  ├──────────┼──────────┤   │
        │  │  Models  │  Plots   │   │
        │  └──────────┴──────────┘   │
        └─────────────────────────────┘
```

### 技术栈

#### 前端层
- **Streamlit** 3.x - Web UI框架
- **Matplotlib** - 图表可视化
- **CSS** - 样式定制

#### 后端层
- **FastAPI** 0.100+ - REST API框架
- **Uvicorn** - ASGI服务器
- **Pydantic** - 数据验证

#### 算法层
- **scikit-learn** - 传统机器学习
- **PyTorch** - 深度学习框架
- **Prophet** - 时间序列预测
- **XGBoost/LightGBM** - 梯度提升

#### 数据层
- **SQL Server** - 关系型数据库
- **Redis** (计划) - 缓存服务
- **File System** - 模型存储

#### 运维层
- **Docker** - 容器化
- **Supervisor** - 进程管理
- **Nginx** - 负载均衡/反向代理

---

## 模块设计

### 1. Web UI 模块 (streamlit_app.py)

**职责**：
- 提供用户交互界面
- 参数配置和输入
- 结果可视化展示

**主要组件**：
```python
streamlit_app.py         # 主入口，页面布局
├── streamlit_daily.py   # 日预测界面
└── streamlit_hourly.py  # 小时预测界面
```

**交互流程**：
```
用户输入参数 → 调用预测模块 → 展示结果 → 保存到数据库
```

### 2. API 服务模块 (server.py)

**职责**：
- 提供RESTful API接口
- 请求验证和处理
- 响应格式化

**端点分类**：
- **元数据端点**：健康检查、类型列表
- **配置端点**：获取/更新配置
- **训练端点**：模型训练
- **预测端点**：客流预测

**API设计模式**：
```
/api/v1/{resource}/{action}
```

### 3. 业务逻辑模块

#### 3.1 日预测模块 (predict_daily.py)

**核心函数**：
```python
def predict_and_plot_timeseries_flow_daily(
    file_path: str,
    predict_start_date: str,
    algorithm: str,
    retrain: bool,
    save_path: str,
    mode: str,
    days: int,
    config: Dict,
    model_version: Optional[str],
    model_save_dir: Optional[str],
    flow_type: Optional[str],
    metric_type: Optional[str]
) -> Dict:
    """日预测主流程"""
```

**处理流程**：
```
1. 加载配置和版本
2. 初始化预测器
3. 读取历史数据
4. 数据预处理
5. 特征工程
6. 模型训练/加载
7. 执行预测
8. 结果可视化
9. 保存到数据库
```

#### 3.2 小时预测模块 (predict_hourly.py)

**处理流程**：
```
1. 加载配置和版本
2. 初始化预测器
3. 读取历史小时数据
4. 数据预处理（包含F_HOUR）
5. 特征工程
6. 模型训练/加载
7. 24小时预测
8. 早晨时段特殊处理
9. 结果可视化
10. 保存到数据库
```

#### 3.3 任务调度模块 (task.py)

**功能**：
- 定时触发训练和预测
- 日志记录和监控
- 错误处理和重试

**调度逻辑**：
```python
while True:
    current_time = datetime.now().strftime("%H:%M")
    
    if current_time in train_schedule_times:
        execute_training()
    
    if current_time in predict_schedule_times:
        execute_prediction()
    
    sleep(60)  # 每分钟检查一次
```

### 4. 算法模块

#### 4.1 KNN 日预测 (enknn_model.py)

**特点**：
- 基于工作日特征的K近邻
- 混合预测（KNN + 去年偏移）
- 支持线路权重配置

**核心方法**：
```python
class KNNFlowPredictor:
    def prepare_data()          # 数据预处理
    def train()                 # 模型训练
    def predict()               # 执行预测
    def calculate_mixed()       # 混合预测
    def save_model_info()       # 保存元数据
```

#### 4.2 KNN 小时预测 (hourknn_model.py)

**特点**：
- 24小时预测
- 早晨时段特殊处理
- 历史参考数据回溯

**早晨策略**：
```python
if hour < 6:
    # 使用纯历史偏移
    weight = {"knn": 0.0, "offset": 1.0}
else:
    # 使用配置权重
    weight = config_weights
```

#### 4.3 其他算法

- **LSTM** (lstm_model.py)：深度学习序列预测
- **Prophet** (prophet_model.py)：Facebook时间序列
- **XGBoost** (xgboost_model.py)：梯度提升树
- **LightGBM** (lightgbm_model.py)：轻量级梯度提升
- **Transformer** (transformer_model.py)：注意力机制

### 5. 数据访问模块 (db_utils.py)

**主要函数**：
```python
# 数据读取
read_line_daily_flow_history()      # 线路日数据
read_line_hourly_flow_history()     # 线路小时数据
read_station_daily_flow_history()   # 车站日数据
read_station_hourly_flow_history()  # 车站小时数据

# 数据写入
upload_xianwangxianlu_daily_prediction_sample()   # 线路日预测
upload_xianwangxianlu_hourly_prediction_sample()  # 线路小时预测
upload_station_daily_prediction_sample()          # 车站日预测
upload_station_hourly_prediction_sample()         # 车站小时预测

# 辅助功能
fetch_holiday_features()            # 节假日特征
get_lineids_by_station()           # 站点线路映射
```

**数据库连接管理**：
```python
def get_db_conn():
    """获取数据库连接"""
    return pymssql.connect(
        server=config['server'],
        user=config['user'],
        password=config['password'],
        database=config['database'],
        port=config['port']
    )
```

### 6. 工具模块

#### 6.1 配置管理 (config_utils.py)

```python
def load_yaml_config()        # 加载配置
def save_yaml_config()        # 保存配置
def get_version_dir()         # 获取版本目录
def get_current_version()     # 获取当前版本
```

#### 6.2 可视化 (plot_utils.py)

```python
def plot_daily_predictions()   # 日预测图表
def plot_hourly_predictions()  # 小时预测图表
```

#### 6.3 字体管理 (font_utils.py)

```python
def get_chinese_font()        # 获取中文字体
def configure_fonts()         # 配置matplotlib字体
```

---

## 数据流图

### 训练流程

```
┌──────────┐
│ 用户请求  │
└────┬─────┘
     │
     ▼
┌────────────────┐
│ API/UI 接收请求 │
└────┬───────────┘
     │ 参数验证
     ▼
┌────────────────┐
│ 预测模块初始化  │
│ - 加载配置      │
│ - 初始化预测器  │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ 数据读取        │
│ - SQL查询      │
│ - 数据清洗     │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ 特征工程        │
│ - 节假日特征    │
│ - 天气特征     │
│ - 时间特征     │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ 模型训练        │
│ - 数据分割     │
│ - 模型拟合     │
│ - 评估指标     │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ 保存模型        │
│ - .pkl文件     │
│ - metadata.json│
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ 返回结果        │
│ - 训练指标     │
│ - 模型信息     │
└────────────────┘
```

### 预测流程

```
┌──────────┐
│ 用户请求  │
└────┬─────┘
     │
     ▼
┌────────────────┐
│ API/UI 接收请求 │
└────┬───────────┘
     │ 参数验证
     ▼
┌────────────────┐
│ 加载模型        │
│ - 读取.pkl     │
│ - 加载配置     │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ 数据准备        │
│ - 历史数据     │
│ - 特征生成     │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ 执行预测        │
│ - KNN预测      │
│ - 偏移预测     │
│ - 加权组合     │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ 结果处理        │
│ - 格式化       │
│ - 可视化       │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ 保存结果        │
│ - 数据库       │
│ - 图片文件     │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│ 返回响应        │
│ - 预测数据     │
│ - 图表URL      │
└────────────────┘
```

---

## 数据库设计

### 表结构

#### 1. 历史数据表

**LineHourlyFlowHistory** - 线路小时客流历史
```sql
F_DATE          VARCHAR(8)      -- 日期 (YYYYMMDD)
F_HOUR          INT             -- 小时 (0-23)
F_LINENO        VARCHAR(10)     -- 线路编号
F_LINENAME      VARCHAR(50)     -- 线路名称
F_KLCOUNT       DECIMAL(18,2)   -- 客运量
F_ENTRANCE      DECIMAL(18,2)   -- 进站量
F_EXIT          DECIMAL(18,2)   -- 出站量
F_TRANSFER      DECIMAL(18,2)   -- 换乘量
-- ... 其他字段
```

**LineDailyFlowHistory** - 线路日客流历史
```sql
F_DATE          VARCHAR(8)      -- 日期
F_LINENO        VARCHAR(10)     -- 线路编号
F_LINENAME      VARCHAR(50)     -- 线路名称
F_KLCOUNT       DECIMAL(18,2)   -- 客运量
F_WEEK          INT             -- 周数
F_HOLIDAYTYPE   VARCHAR(50)     -- 节假日类型
F_DAYOFWEEK     INT             -- 星期几
-- ... 其他字段
```

#### 2. 预测结果表

**xianwangxianlu_daily_prediction** - 线路日预测
```sql
ID              VARCHAR(36)     -- UUID
F_DATE          VARCHAR(8)      -- 预测日期
F_LINENO        VARCHAR(10)     -- 线路编号
F_LINENAME      VARCHAR(50)     -- 线路名称
PRED_VALUE      DECIMAL(18,2)   -- 预测值
ALGORITHM       VARCHAR(20)     -- 算法名称
MODEL_VERSION   VARCHAR(8)      -- 模型版本
METRIC_TYPE     VARCHAR(20)     -- 指标类型
CREATE_TIME     DATETIME        -- 创建时间
```

#### 3. 辅助数据表

**CalendarHistory** - 日历信息
```sql
F_DATE              VARCHAR(8)
F_YEAR              INT
F_DAYOFWEEK         INT
F_WEEK              INT
F_HOLIDAYTYPE       VARCHAR(50)
F_HOLIDAYDAYS       INT
F_HOLIDAYWHICHDAY   INT
```

**WeatherHistory** - 天气信息
```sql
F_DATE      VARCHAR(8)
F_WEATHER   VARCHAR(50)
F_TQQK      VARCHAR(50)
```

### 索引设计

```sql
-- 主键索引
PRIMARY KEY (ID)

-- 查询优化索引
INDEX idx_date_line (F_DATE, F_LINENO)
INDEX idx_date (F_DATE)
INDEX idx_line (F_LINENO)
INDEX idx_model_version (MODEL_VERSION)
```

---

## API设计

### RESTful API 规范

#### 基础URL
```
http://api.example.com/v1
```

#### 通用响应格式

**成功响应**：
```json
{
  "success": true,
  "message": "操作成功",
  "timestamp": "2025-01-15T10:00:00",
  "data": { ... }
}
```

**错误响应**：
```json
{
  "success": false,
  "message": "错误描述",
  "timestamp": "2025-01-15T10:00:00",
  "error_type": "ValidationError",
  "error_details": "详细错误信息"
}
```

#### 主要端点

详见 README.md 的 [API文档](README.md#api文档) 部分。

---

## 部署架构

### 单机部署

```
┌────────────────────────────────┐
│       Linux Server              │
│                                 │
│  ┌──────────────────────────┐  │
│  │   Nginx (80/443)         │  │
│  └───────┬──────────────────┘  │
│          │                      │
│  ┌───────▼────────┐ ┌─────────▼──┐
│  │ Streamlit:4577 │ │ FastAPI:4566│
│  └────────────────┘ └────────────┘
│                                 │
│  ┌──────────────────────────┐  │
│  │   SQL Server :1433       │  │
│  └──────────────────────────┘  │
│                                 │
│  ┌──────────────────────────┐  │
│  │   File Storage           │  │
│  │   - Models               │  │
│  │   - Plots                │  │
│  └──────────────────────────┘  │
└────────────────────────────────┘
```

### Docker部署

```
┌────────────────────────────────┐
│       Docker Host               │
│                                 │
│  ┌──────────────────────────┐  │
│  │  nginx-container         │  │
│  │  (80/443 → 4566/4577)    │  │
│  └───────┬──────────────────┘  │
│          │                      │
│  ┌───────▼────────┐ ┌─────────▼──┐
│  │ stfs-ui        │ │ stfs-api   │
│  │ container      │ │ container  │
│  └────────────────┘ └────────────┘
│          │                 │      │
│  ┌───────▼─────────────────▼───┐ │
│  │   Shared Volumes            │ │
│  │   - /models                 │ │
│  │   - /config                 │ │
│  └─────────────────────────────┘ │
└────────────────────────────────┘
```

### 高可用部署

```
┌────────────────────────────────────┐
│       Load Balancer (HAProxy)       │
└──────┬─────────────────────┬───────┘
       │                     │
   ┌───▼────┐           ┌───▼────┐
   │ Node 1 │           │ Node 2 │
   │        │           │        │
   │ UI/API │           │ UI/API │
   └───┬────┘           └───┬────┘
       │                    │
       └────────┬───────────┘
                │
      ┌─────────▼──────────┐
      │   SQL Server       │
      │   (Master-Slave)   │
      └────────────────────┘
                │
      ┌─────────▼──────────┐
      │   Shared Storage   │
      │   (NFS/GlusterFS)  │
      └────────────────────┘
```

---

## 性能优化

### 1. 数据库优化

- 使用索引加速查询
- 数据分区（按日期）
- 连接池管理
- 查询缓存

### 2. 应用优化

- 异步处理
- 批量操作
- 结果缓存（Redis）
- 模型预加载

### 3. 部署优化

- 负载均衡
- CDN加速（静态资源）
- 容器编排（Kubernetes）
- 监控和告警

---

## 安全设计

### 1. 数据安全

- 数据库连接加密
- 敏感信息加密存储
- 定期备份

### 2. 接口安全

- API认证（JWT）
- 请求限流
- 参数验证
- XSS/CSRF防护

### 3. 运维安全

- 最小权限原则
- 日志审计
- 安全更新

---

## 监控和日志

### 1. 应用监控

- 请求成功率
- 响应时间
- 错误率
- CPU/内存使用率

### 2. 业务监控

- 预测准确率
- 模型性能指标
- 任务执行状态

### 3. 日志管理

- 分级日志（DEBUG/INFO/WARNING/ERROR）
- 日志轮转
- 集中式日志收集

---

## 扩展性设计

### 1. 水平扩展

- 无状态服务设计
- 支持多实例部署
- 负载均衡

### 2. 功能扩展

- 插件化算法
- 配置化流程
- 标准化接口

### 3. 数据扩展

- 支持多数据源
- 数据格式抽象
- 灵活的特征工程

---

本文档持续更新中，如有疑问或建议，欢迎反馈。

