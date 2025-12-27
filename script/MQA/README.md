# 地铁客流智能问数系统

## 项目简介

地铁客流智能问数系统是一个基于自然语言处理（NLP）的智能数据查询系统，允许用户通过自然语言提问的方式查询地铁客流相关数据。

## 功能特性

- ✅ 自然语言查询：支持中文自然语言查询地铁客流数据
- ✅ 多维度查询：支持按线路、车站、日期、时间等维度查询
- ✅ 历史数据查询：查询历史客流数据
- ✅ 预测数据查询：查询客流预测数据
- ✅ 数据可视化：自动生成图表展示查询结果
- ✅ SQL直接查询：支持直接执行SQL查询
- ✅ API接口：提供RESTful API接口

## 技术栈

### 后端
- FastAPI：高性能Web框架
- pymssql：SQL Server数据库驱动
- jieba：中文分词
- ollama：本地大语言模型支持（可选）
- Redis：缓存（可选）

### 前端（待实现）
- Vue 3：前端框架
- Element Plus：UI组件库
- ECharts：图表库

## 项目结构

```
MQA/
├── backend/              # 后端服务
│   ├── app/              # 应用代码
│   │   ├── api/         # API路由
│   │   ├── core/        # 核心业务逻辑
│   │   ├── models/      # 数据模型
│   │   └── utils/       # 工具函数
│   └── requirements.txt # Python依赖
├── docs/                 # 文档
│   ├── 系统详细设计文档.md
│   └── 项目结构说明.md
└── README.md
```

## 快速开始

### 1. 环境要求

- Python 3.8+
- SQL Server数据库
- Redis（可选）

### 2. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2.1 使用Ollama（可选）

如果需要使用大语言模型增强查询能力：

1. 安装Ollama：https://ollama.com/download
2. 下载模型：`ollama pull qwen2.5:7b`
3. 在 `.env` 中启用：`LLM_ENABLED=True`

详细说明请参考 [Ollama使用说明.md](Ollama使用说明.md)

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

### 4. 启动服务

```bash
cd backend
python -m app.main
```

服务将在 `http://localhost:8000` 启动。

### 5. 访问API文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API使用示例

### 自然语言查询

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "查询1号线昨天的客流量"
  }'
```

### SQL直接查询

```bash
curl -X POST "http://localhost:8000/api/v1/sql" \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT * FROM LineDailyFlowHistory WHERE f_date = 20240101 LIMIT 10",
    "database": "master"
  }'
```

## 查询示例

系统支持以下类型的自然语言查询：

1. **线路客流查询**
   - "查询1号线昨天的客流量"
   - "2号线本周的平均客流量是多少"

2. **车站客流查询**
   - "五一广场站今天的进站量"
   - "查询所有车站的客流量排名"

3. **时间维度查询**
   - "查询最近7天的客流趋势"
   - "今年国庆节的客流量"

4. **预测数据查询**
   - "预测明天1号线的客流量"
   - "未来一周的客流预测"

## 开发计划

- [x] 系统架构设计
- [x] 核心模块实现（NL2SQL、查询执行）
- [ ] 前端界面开发
- [ ] 查询历史功能
- [ ] 数据导出功能
- [ ] 性能优化
- [ ] 安全加固

## 文档

- [系统详细设计文档](系统详细设计文档.md)
- [项目结构说明](项目结构说明.md)
- [Ollama使用说明](Ollama使用说明.md)
- [部署指南](部署指南.md)

## 许可证

MIT License

## 联系方式

如有问题或建议，请联系项目维护者。

