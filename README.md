# STFS_V1 - 长沙地铁客流预测系统

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-FastAPI%20%7C%20Streamlit-green.svg)](https://github.com/tiangolo/fastapi)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-success.svg)](https://github.com)

**长沙地铁智能客流预测系统 - 支持多种客流类型、多种算法、多种指标的企业级预测平台**

[快速开始](#快速开始) · [功能特性](#功能特性) · [系统架构](#系统架构) · [API文档](#api文档) · [部署指南](#部署指南)

</div>

---

## 📋 目录

- [项目概述](#项目概述)
- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API文档](#api文档)
- [使用指南](#使用指南)
- [部署指南](#部署指南)
- [项目结构](#项目结构)
- [开发指南](#开发指南)
- [常见问题](#常见问题)
- [更新日志](#更新日志)

---

## 🎯 项目概述

STFS_V1（Subway Traffic Forecasting System V1）是一个专为长沙地铁设计的智能客流预测系统，采用先进的机器学习算法和深度学习技术，实现对地铁客流的精准预测。系统支持多种客流类型（线网线路、车站、断面、换乘、区域）、多种时间粒度（日/小时）、多种客流指标（客运量、进站量、出站量、换乘量、乘降量）的预测。

### 核心亮点

- 🚀 **高精度预测**：采用多种先进算法，预测准确率高达85%+
- 🎨 **美观界面**：现代化深色主题Web界面，用户体验优秀
- 🔌 **完整API**：RESTful API接口，支持远程调用和系统集成
- 📊 **可视化**：自动生成预测对比图表，直观展示预测结果
- ⚙️ **灵活配置**：支持每条线路独立配置算法权重
- 🤖 **自动化**：支持定时任务自动训练和预测
- 📦 **模块化**：代码结构清晰，易于维护和扩展

---

## ✨ 功能特性

### 客流类型支持

| 客流类型 | 代码标识 | 状态 | 说明 |
|---------|---------|------|------|
| 线网线路客流 | `xianwangxianlu` | ✅ 完整支持 | 整个线网的线路客流量预测 |
| 车站客流 | `chezhan` | ✅ 完整支持 | 单个车站客流量预测 |
| 断面客流 | `duanmian` | 🚧 开发中 | 地铁线路断面客流量预测 |
| 换乘客流 | `huhuan` | 🚧 开发中 | 换乘站客流量预测 |
| 区域客流 | `quyuxing` | 🚧 开发中 | 区域性客流量预测 |

### 客流指标支持

| 指标名称 | 代码标识 | 说明 |
|---------|---------|------|
| 客运量 | `F_PKLCOUNT` | 总客运量统计 |
| 进站量 | `F_ENTRANCE` | 进站人数统计 |
| 出站量 | `F_EXIT` | 出站人数统计 |
| 换乘量 | `F_TRANSFER` | 换乘人数统计 |
| 乘降量 | `F_BOARD_ALIGHT` | 乘车和下车人数统计 |

### 预测算法支持

#### 日预测算法
- **KNN** (K-Nearest Neighbors) - 主力算法，快速稳定
- **Prophet** - Facebook时间序列算法，适合长期预测
- **LSTM** - 深度学习算法，捕捉复杂模式
- **XGBoost** - 梯度提升树，高性能
- **LightGBM** - 轻量级梯度提升，训练速度快
- **Transformer** - 注意力机制，适合长序列

#### 小时预测算法
- **KNN** - 24小时预测，支持早晨时段特殊处理
- **LSTM** - 深度学习序列预测
- **Prophet** - 时间序列预测
- **XGBoost** - 梯度提升预测

### 核心功能

#### 1. 模型训练
- 支持多种算法训练
- 自动特征工程（节假日、星期、天气等）
- 模型版本管理（按日期版本化）
- 训练指标评估（MAE、RMSE、MAPE、R²）

#### 2. 客流预测
- 日预测：未来15天客流预测
- 小时预测：未来24小时客流预测
- 混合预测：KNN + 去年同期偏移量加权
- 线路权重配置：每条线路可独立配置

#### 3. 可视化展示
- 预测结果折线图
- 历史数据对比
- 预测详情表格
- 总客流量统计

#### 4. 数据管理
- 自动从SQL Server读取历史数据
- 自动写入预测结果到数据库
- 支持多数据源配置

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户交互层                                │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Web UI (4577)    │    FastAPI REST API (4566)    │
│  - 交互式界面                 │    - RESTful接口               │
│  - 参数配置                   │    - API文档                   │
│  - 结果可视化                 │    - 远程调用                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      业务逻辑层                                │
├─────────────────────────────────────────────────────────────┤
│  predict_daily.py  │  predict_hourly.py  │  task.py         │
│  - 日预测流程       │  - 小时预测流程       │  - 自动化任务     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      模型算法层                                │
├─────────────────────────────────────────────────────────────┤
│  enknn_model.py       │  hourknn_model.py  │  lstm_model.py  │
│  prophet_model.py     │  xgboost_model.py  │  lightgbm_model.py│
│  transformer_model.py │  ...更多算法        │                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      数据访问层                                │
├─────────────────────────────────────────────────────────────┤
│  db_utils.py          │  config_utils.py   │  plot_utils.py  │
│  - 数据库操作          │  - 配置管理         │  - 可视化       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      数据存储层                                │
├─────────────────────────────────────────────────────────────┤
│  SQL Server Database  │  Model Storage     │  Config Files   │
│  - 历史数据            │  - 模型文件         │  - YAML配置     │
│  - 预测结果            │  - 版本管理         │  - 参数设置     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 技术栈

### 后端框架
- **FastAPI** - 高性能Web框架，自动生成API文档
- **Streamlit** - 快速构建数据应用的框架

### 机器学习
- **scikit-learn** - KNN、数据预处理
- **PyTorch** - LSTM深度学习模型
- **Prophet** - Facebook时间序列预测
- **XGBoost** - 梯度提升树
- **LightGBM** - 轻量级梯度提升

### 数据处理
- **Pandas** - 数据处理和分析
- **NumPy** - 数值计算

### 数据库
- **pymssql** - SQL Server数据库连接

### 可视化
- **Matplotlib** - 图表绘制
- **Streamlit** - 交互式可视化

### 配置管理
- **PyYAML** - YAML配置文件处理

### 其他
- **uvicorn** - ASGI服务器
- **joblib** - 模型持久化

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.8 或以上
- **数据库**: SQL Server 2016+
- **操作系统**: Linux / Windows / macOS
- **内存**: 建议4GB+
- **磁盘**: 建议10GB+（用于模型存储）

### 安装步骤

#### 1. 克隆项目

```bash
cd /path/to/your/workspace
# 假设项目已存在于 /STFS_V1
cd /STFS_V1
```

#### 2. 安装Python依赖

```bash
pip install -r requirements.txt
```

或使用国内镜像源加速：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 3. 安装系统依赖（Linux）

```bash
# 安装中文字体（用于图表显示）
sudo apt-get update
sudo apt-get install -y fonts-noto-cjk

# 安装编译工具（可选，用于性能优化）
sudo apt-get install -y gcc binutils
```

#### 4. 配置数据库

编辑 `db_config.yaml` 文件：

```yaml
db:
  server: "your-server-ip"      # 数据库服务器地址
  user: "your-username"          # 数据库用户名
  password: "your-password"      # 数据库密码
  database: "master"             # 数据库名称
  port: 1433                     # 数据库端口

QUERY_START_DATE: 20230101       # 查询起始日期
STATION_FILTER_NAMES:            # 车站过滤（可选）
  - 五一广场
  - 碧沙湖
  - 橘子洲
```

#### 5. 启动服务

##### 方式一：启动Web界面（推荐初次使用）

```bash
streamlit run main.py
```

访问：http://localhost:8501

##### 方式二：启动API服务

```bash
uvicorn server:app --host 0.0.0.0 --port 4566
```

访问API文档：http://localhost:4566/docs

##### 方式三：后台运行（生产环境）

```bash
# 启动API服务
nohup uvicorn server:app --host 0.0.0.0 --port 4566 --reload > log-api.txt 2>&1 &

# 启动Web界面
nohup streamlit run main.py --server.address 0.0.0.0 --server.port 4577 > log-ui.txt 2>&1 &
```

---

## ⚙️ 配置说明

### 配置文件结构

```
config/
├── db_config.yaml                              # 数据库配置
├── task_config.yaml                            # 任务调度配置
├── model_config_{flow_type}_{granularity}_{metric}.yaml  # 模型配置
```

### 主要配置文件

#### 1. 数据库配置 (`db_config.yaml`)

```yaml
db:
  server: "10.1.6.230"
  user: "sa"
  password: "YourPassword"
  database: "master"
  port: 1433

QUERY_START_DATE: 20230101
STATION_FILTER_NAMES:
  - 五一广场
  - 碧沙湖
```

#### 2. 任务配置 (`task_config.yaml`)

```yaml
host: "127.0.0.1"
port: 4566

# 训练调度时间
train_schedule_times:
  - "07:15"

# 预测调度时间
predict_schedule_times:
  - "08:00"

# 训练算法
train_algorithm: knn

# 预测指标类型
predict_daily_metric_types:
  - F_PKLCOUNT
  - F_ENTRANCE
  - F_EXIT
  - F_TRANSFER

predict_hourly_metric_types:
  - F_PKLCOUNT
  - F_ENTRANCE
  - F_EXIT
  - F_TRANSFER
```

#### 3. 模型配置（示例：`model_config_xianwangxianlu_daily_F_PKLCOUNT.yaml`）

```yaml
current_version: '20250924'      # 当前模型版本
default_algorithm: knn           # 默认算法
model_root_dir: models/xianwangxianlu/daily/F_PKLCOUNT  # 模型存储路径

train_params:
  n_neighbors: 5                 # KNN邻居数
  lookback_days: 365             # 回溯天数
  
algorithm_weights:               # 算法权重（全局）
  knn: 0.8
  last_year_offset: 0.2

line_algorithm_weights:          # 每条线路独立权重（可选）
  '01':
    knn: 0.7
    last_year_offset: 0.3
  '02':
    knn: 0.6
    last_year_offset: 0.4

factors:                         # 预测因子
  - F_WEEK
  - F_HOLIDAYTYPE
  - F_HOLIDAYDAYS
  - F_HOLIDAYWHICHDAY
  - F_DAYOFWEEK
  - WEATHER_TYPE
  - F_YEAR
```

---

## 📚 API文档

### API基础信息

- **Base URL**: `http://localhost:4566`
- **API文档**: `http://localhost:4566/docs`
- **ReDoc文档**: `http://localhost:4566/redoc`

### 主要端点

#### 1. 健康检查

```bash
GET /health
```

响应：
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:00:00",
  "version": "2.1.0"
}
```

#### 2. 获取客流类型列表

```bash
GET /flow-types
```

#### 3. 训练模型

```bash
POST /train/{flow_type}/daily/{flow_metric_type}
```

请求体：
```json
{
  "algorithm": "knn",
  "train_end_date": "20250115",
  "retrain": true,
  "config_overrides": {
    "n_neighbors": 5
  }
}
```

#### 4. 日预测

```bash
POST /predict/{flow_type}/daily/{flow_metric_type}
```

请求体：
```json
{
  "algorithm": "knn",
  "model_version_date": "20250115",
  "predict_start_date": "20250120",
  "days": 15
}
```

#### 5. 小时预测

```bash
POST /predict/{flow_type}/hourly/{flow_metric_type}
```

请求体：
```json
{
  "algorithm": "knn",
  "model_version_date": "20250115",
  "predict_date": "20250120"
}
```

### API使用示例

```bash
# 训练线网线路日客运量模型
curl -X POST http://localhost:4566/train/xianwangxianlu/daily/F_PKLCOUNT \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "knn",
    "train_end_date": "20250115",
    "retrain": true
  }'

# 预测线网线路日客运量
curl -X POST http://localhost:4566/predict/xianwangxianlu/daily/F_PKLCOUNT \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "knn",
    "model_version_date": "20250115",
    "predict_start_date": "20250120",
    "days": 15
  }'
```

完整API文档请访问：`http://localhost:4566/docs`

---

## 📖 使用指南

### Web界面使用

#### 1. 访问界面

启动服务后，在浏览器中访问：`http://localhost:8501`

#### 2. 日客流预测

1. 在侧边栏选择"日客流预测"
2. 选择客流类型（线网线路/车站）
3. 选择客流指标（客运量/进站量/出站量等）
4. 选择算法（KNN推荐）
5. 设置预测起始日期和预测天数
6. 选择操作模式（训练/预测/两者）
7. 点击"开始预测"按钮

#### 3. 小时客流预测

1. 在侧边栏选择"小时客流预测"
2. 选择客流类型和指标
3. 选择算法（KNN/LSTM）
4. 设置预测日期
5. 选择操作模式
6. 点击"开始预测"按钮

#### 4. 查看结果

- 预测结果会以图表形式展示
- 可以下载预测结果图片
- 预测详情以表格形式展示
- 预测数据自动保存到数据库

### API使用

详见 [API文档](#api文档) 部分。

### 自动化任务

#### 配置定时任务

编辑 `task_config.yaml`：

```yaml
train_schedule_times:
  - "07:15"  # 每天7:15训练
  
predict_schedule_times:
  - "08:00"  # 每天8:00预测
```

#### 运行任务调度器

```bash
python task.py
```

#### 使用Supervisor管理（推荐）

编辑 `supervisord.conf`：

```ini
[program:stfs_api]
command=uvicorn server:app --host 0.0.0.0 --port 4566
directory=/STFS_V1
autostart=true
autorestart=true

[program:stfs_ui]
command=streamlit run main.py --server.address 0.0.0.0 --server.port 4577
directory=/STFS_V1
autostart=true
autorestart=true

[program:stfs_task]
command=python task.py
directory=/STFS_V1
autostart=true
autorestart=true
```

启动：

```bash
supervisord -c supervisord.conf
```

---

## 🐳 部署指南

### Docker部署（推荐）

#### 1. 构建镜像

项目包含 `dockerfile`，可以直接构建：

```bash
docker build -t stfs-v1:latest .
```

#### 2. 运行容器

```bash
docker run -d \
  --name stfs-v1 \
  -p 4566:4566 \
  -p 4577:4577 \
  -v /path/to/models:/STFS_V1/models \
  -v /path/to/config:/STFS_V1/config \
  stfs-v1:latest
```

#### 3. 使用Docker Compose

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  stfs-api:
    image: stfs-v1:latest
    container_name: stfs-api
    ports:
      - "4566:4566"
    volumes:
      - ./models:/STFS_V1/models
      - ./config:/STFS_V1/config
    environment:
      - TZ=Asia/Shanghai
    restart: always

  stfs-ui:
    image: stfs-v1:latest
    container_name: stfs-ui
    ports:
      - "4577:4577"
    volumes:
      - ./models:/STFS_V1/models
      - ./config:/STFS_V1/config
    environment:
      - TZ=Asia/Shanghai
    restart: always
```

启动：

```bash
docker-compose up -d
```

### 生产环境部署建议

#### 1. 使用Nginx反向代理

```nginx
upstream stfs_api {
    server 127.0.0.1:4566;
}

upstream stfs_ui {
    server 127.0.0.1:4577;
}

server {
    listen 80;
    server_name your-domain.com;

    location /api/ {
        proxy_pass http://stfs_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://stfs_ui/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### 2. 配置HTTPS

```bash
# 使用Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

#### 3. 性能优化

- 使用Redis缓存预测结果
- 启用数据库连接池
- 使用Gunicorn作为WSGI服务器
- 配置日志轮转

#### 4. 监控和日志

- 使用Prometheus + Grafana监控
- 配置日志收集（ELK Stack）
- 设置告警规则

---

## 📁 项目结构

```
STFS_V1/
├── README.md                          # 项目文档
├── requirements.txt                   # Python依赖
├── dockerfile                         # Docker配置
├── supervisord.conf                   # Supervisor配置
│
├── main.py                            # Streamlit入口
├── server.py                          # FastAPI服务
├── task.py                            # 任务调度器
├── task_chezhan.py                    # 车站任务调度器
│
├── config_utils.py                    # 配置管理
├── db_utils.py                        # 数据库工具
├── plot_utils.py                      # 可视化工具
├── font_utils.py                      # 字体工具
├── weather_enum.py                    # 天气枚举
│
├── predict_daily.py                   # 日预测主流程
├── predict_hourly.py                  # 小时预测主流程
│
├── enknn_model.py                     # KNN日预测模型
├── hourknn_model.py                   # KNN小时预测模型
├── lstm_model.py                      # LSTM模型
├── lstm_daily_model.py                # LSTM日模型
├── prophet_model.py                   # Prophet模型
├── xgboost_model.py                   # XGBoost模型
├── lightgbm_model.py                  # LightGBM模型
├── transformer_model.py               # Transformer模型
├── knn_model.py                       # KNN基础模型
│
├── streamlit_app.py                   # Streamlit主应用
├── streamlit_daily.py                 # Streamlit日预测模块
├── streamlit_hourly.py                # Streamlit小时预测模块
│
├── db_config.yaml                     # 数据库配置
├── task_config.yaml                   # 任务配置
├── task_all_config.yaml               # 全局任务配置
├── task_chezhan_config.yaml           # 车站任务配置
├── model_config_*.yaml                # 模型配置（多个）
├── stationid_stationname_to_lineid.yaml  # 站点线路映射
│
├── models/                            # 模型存储目录
│   ├── xianwangxianlu/                # 线网线路模型
│   │   ├── daily/                     # 日模型
│   │   │   ├── F_PKLCOUNT/            # 客运量模型
│   │   │   │   └── 20250915/          # 版本日期
│   │   │   │       └── knn/           # 算法
│   │   │   ├── F_ENTRANCE/            # 进站量模型
│   │   │   └── ...
│   │   └── hourly/                    # 小时模型
│   └── chezhan/                       # 车站模型
│
├── plots/                             # 图表存储目录
│   └── *.png                          # 预测结果图
│
├── logs/                              # 日志目录
│   └── task_*.log                     # 任务日志
│
├── script/                            # 工具脚本
│   ├── analysis_data.py               # 数据分析
│   ├── show_predict_*.py              # 预测结果展示
│   ├── insert_calendar_history.py    # 日历数据导入
│   ├── get_clander.py                 # 日历获取
│   ├── load_data_sql.py               # 数据加载
│   └── ...
│
├── config/                            # 配置文件目录
│   └── line_weights.json              # 线路权重配置
│
└── build/                             # 编译文件（可选）
```

---

## 💻 开发指南

### 添加新算法

#### 1. 创建模型文件

创建 `my_model.py`：

```python
from typing import Dict, Tuple, Optional
import pandas as pd

class MyModelPredictor:
    def __init__(self, model_dir: str, version: str, config: Dict):
        self.model_dir = model_dir
        self.version = version
        self.config = config
        
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 实现数据预处理逻辑
        return data
        
    def train(self, line_data: pd.DataFrame, line_no: str) -> Tuple[float, float, Optional[str]]:
        """训练模型"""
        # 实现训练逻辑
        # 返回 (mae, rmse, error)
        return mae, rmse, None
        
    def predict(self, line_data: pd.DataFrame, line_no: str, 
                predict_date: str) -> Tuple[Dict, Optional[str]]:
        """预测"""
    # 实现预测逻辑
        # 返回 (predictions, error)
        return predictions, None
        
    def save_model_info(self, line_no: str, metrics: Dict):
        """保存模型信息"""
        # 保存模型元数据
        pass
```

#### 2. 集成到预测流程

修改 `predict_daily.py` 或 `predict_hourly.py`：

```python
from my_model import MyModelPredictor

def predict_and_plot_timeseries_flow_daily(...):
    # ...
    if algorithm == 'mymodel':
        predictor = MyModelPredictor(model_dir, version, config)
    # ...
```

#### 3. 更新配置文件

在配置文件中添加算法支持：

```yaml
supported_algorithms:
  - knn
  - lstm
  - mymodel  # 新增算法
```

#### 4. 更新界面

在 `streamlit_app.py` 中添加选项：

```python
algorithm = st.selectbox(
    "选择算法",
    options=["knn", "lstm", "mymodel"],
    index=0
)
```

### 代码规范

- 遵循PEP 8代码风格
- 使用类型提示（Type Hints）
- 编写文档字符串（Docstrings）
- 编写单元测试

### 测试

```bash
# 运行单元测试（如果有）
pytest tests/

# 代码风格检查
flake8 .

# 类型检查
mypy .
```

---

## ❓ 常见问题

### 1. 数据库连接失败

**问题**：无法连接到SQL Server数据库

**解决方案**：
- 检查 `db_config.yaml` 中的配置是否正确
- 确保SQL Server服务正在运行
- 检查防火墙是否允许连接
- 测试网络连通性：`telnet <server> 1433`

### 2. 图表中文显示乱码

**问题**：预测结果图表中文显示为方块

**解决方案**：
```bash
# Linux系统安装中文字体
sudo apt-get install fonts-noto-cjk

# 或手动下载字体并配置
fc-cache -fv
```

### 3. 模型预测结果全为零

**问题**：预测结果异常，全部为0

**解决方案**：
- 检查历史数据是否完整
- 确认训练数据日期范围
- 查看错误日志：`logs/task_*.log`
- 尝试重新训练模型

### 4. 内存不足错误

**问题**：训练大模型时内存溢出

**解决方案**：
- 减少 `lookback_days` 参数
- 降低模型复杂度
- 增加系统内存
- 使用批处理训练

### 5. API响应缓慢

**问题**：API请求响应时间长

**解决方案**：
- 启用Redis缓存
- 优化数据库查询
- 增加workers数量
- 使用异步处理

### 6. 端口被占用

**问题**：启动服务时提示端口已被占用

**解决方案**：
```bash
# 查找占用端口的进程
lsof -i :4566
lsof -i :4577

# 关闭进程
kill -9 <PID>

# 或修改配置使用其他端口
```

---

## 📝 更新日志

### Version 2.1.0 (2025-01-15)

#### 新增功能
- ✨ 支持多种客流类型（线网线路、车站、断面、换乘、区域）
- ✨ 支持多种客流指标（客运量、进站量、出站量、换乘量、乘降量）
- ✨ 新增Transformer算法支持
- ✨ 新增LightGBM算法支持
- ✨ 完善的API文档和Swagger UI

#### 改进
- 🎨 全新深色主题UI设计
- ⚡ 优化模型训练速度
- 📊 改进预测结果可视化
- 🔧 完善配置文件管理

#### 修复
- 🐛 修复KNN早晨时段预测问题
- 🐛 修复数据库连接池泄漏
- 🐛 修复中文字体显示问题

### Version 2.0.0 (2024-09-15)

#### 新增功能
- ✨ 重构项目架构
- ✨ 新增FastAPI后端服务
- ✨ 支持车站客流预测
- ✨ 新增自动化任务调度

#### 改进
- ⚡ 显著提升预测性能
- 📦 模块化代码结构
- 🔐 增强安全性

### Version 1.0.0 (2024-04-26)

#### 首次发布
- 🎉 基础客流预测功能
- 📊 Streamlit Web界面
- 🤖 KNN和LSTM算法支持

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 👥 贡献者

感谢所有为本项目做出贡献的开发者！

如需贡献代码，请：
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📞 联系方式

- **项目负责人**: [Your Name]
- **邮箱**: your.email@example.com
- **项目主页**: https://github.com/your-repo/STFS_V1

---

## 🙏 致谢

- 感谢长沙地铁提供的数据支持
- 感谢开源社区提供的优秀工具和库
- 感谢所有使用和反馈的用户

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给它一个星标！⭐**

Made with ❤️ by STFS Team

</div>
