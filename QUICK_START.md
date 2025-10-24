# 快速开始指南

5分钟快速上手 STFS_V1 地铁客流预测系统。

## 📋 前置要求

- Python 3.8+
- SQL Server 2016+
- 4GB+ 内存

## 🚀 快速安装

### 1. 安装依赖

```bash
cd /STFS_V1
pip install -r requirements.txt
```

### 2. 配置数据库

复制配置模板并填写数据库信息：

```bash
cp db_config.example.yaml db_config.yaml
```

编辑 `db_config.yaml`：

```yaml
db:
  server: "your-server-ip"
  user: "your-username"
  password: "your-password"
  database: "master"
  port: 1433
```

### 3. 启动服务

#### 方式一：Web界面（推荐新手）

```bash
streamlit run main.py
```

访问：http://localhost:8501

#### 方式二：API服务

```bash
uvicorn server:app --host 0.0.0.0 --port 4566
```

API文档：http://localhost:4566/docs

## 🎯 第一次预测

### 使用Web界面

1. 打开浏览器访问 http://localhost:8501
2. 在侧边栏选择"日客流预测"
3. 配置参数：
   - 客流类型：线网线路
   - 客流指标：客运量
   - 算法：KNN
   - 预测起始日期：选择明天的日期
   - 预测天数：7天
   - 操作模式：预测和训练
4. 点击"开始预测"
5. 等待几分钟，查看结果

### 使用API

```bash
# 训练模型
curl -X POST http://localhost:4566/train/xianwangxianlu/daily/F_PKLCOUNT \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "knn",
    "train_end_date": "20250115",
    "retrain": true
  }'

# 执行预测
curl -X POST http://localhost:4566/predict/xianwangxianlu/daily/F_PKLCOUNT \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "knn",
    "model_version_date": "20250115",
    "predict_start_date": "20250120",
    "days": 7
  }'
```

## 📊 查看结果

### Web界面
- 图表自动显示在页面上
- 预测详情以表格形式展示
- 可以下载预测图表

### API
- 响应包含预测数据JSON
- `plot_url` 字段包含图表URL
- 访问图表：`http://localhost:4566/plots/<filename>`

### 数据库
预测结果自动保存到数据库表：
- 线路日预测：`xianwangxianlu_daily_prediction`
- 线路小时预测：`xianwangxianlu_hourly_prediction`

## 🔧 常见问题

### 1. 启动失败："端口被占用"

```bash
# 查找占用端口的进程
lsof -i :8501  # Streamlit
lsof -i :4566  # API

# 关闭进程或使用其他端口
streamlit run main.py --server.port 8502
```

### 2. 数据库连接失败

```bash
# 测试连接
telnet your-server-ip 1433

# 检查配置
cat db_config.yaml
```

### 3. 没有历史数据

确保数据库中有以下表的数据：
- `LineDailyFlowHistory` - 日客流历史
- `LineHourlyFlowHistory` - 小时客流历史
- `CalendarHistory` - 日历信息
- `WeatherHistory` - 天气信息

### 4. 预测结果全为零

- 检查历史数据日期范围
- 确认训练数据不为空
- 查看日志文件：`logs/task_*.log`
- 尝试重新训练模型

## 📚 下一步

- 📖 阅读完整文档：[README.md](README.md)
- ⚙️ 配置参数优化：[CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- 🏗️ 了解架构设计：[ARCHITECTURE.md](ARCHITECTURE.md)
- 🤝 贡献代码：[CONTRIBUTING.md](CONTRIBUTING.md)

## 🆘 获取帮助

- 查看 [常见问题](README.md#常见问题)
- 查看 [更新日志](CHANGELOG.md)
- 提交 [Issue](https://github.com/your-repo/STFS_V1/issues)

---

**提示**: 首次使用建议使用"预测和训练"模式，系统会自动训练模型并预测。

