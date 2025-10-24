# 贡献指南

感谢您对 STFS_V1 项目的关注！本文档将指导您如何为项目做出贡献。

## 目录

- [代码规范](#代码规范)
- [开发流程](#开发流程)
- [提交规范](#提交规范)
- [测试要求](#测试要求)
- [文档编写](#文档编写)

---

## 代码规范

### Python代码风格

遵循 PEP 8 代码风格指南：

```python
# 好的示例
def calculate_passenger_flow(
    line_no: str,
    start_date: str,
    end_date: str,
    algorithm: str = 'knn'
) -> Dict[str, Any]:
    """
    计算指定线路的客流量
    
    Args:
        line_no: 线路编号
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        algorithm: 使用的算法，默认为 'knn'
        
    Returns:
        包含预测结果的字典
    """
    # 实现代码
    pass
```

### 命名规范

- **文件名**：小写字母+下划线，如 `db_utils.py`
- **类名**：驼峰命名法，如 `KNNFlowPredictor`
- **函数名**：小写字母+下划线，如 `read_line_daily_flow_history`
- **常量**：大写字母+下划线，如 `DEFAULT_KNN_FACTORS`
- **变量**：小写字母+下划线，如 `train_end_date`

### 类型提示

强烈建议使用类型提示：

```python
from typing import Dict, List, Optional, Tuple

def train_model(
    data: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[float, float, Optional[str]]:
    """训练模型并返回评估指标"""
    pass
```

### 文档字符串

使用 Google 风格的文档字符串：

```python
def predict_daily_flow(
    line_no: str,
    predict_date: str,
    days: int = 15
) -> Dict[str, Any]:
    """
    预测指定线路的日客流量
    
    Args:
        line_no: 线路编号，如 '01', '02'
        predict_date: 预测起始日期，格式为 YYYYMMDD
        days: 预测天数，默认为 15 天
        
    Returns:
        包含预测结果的字典，格式为:
        {
            'dates': ['20250101', '20250102', ...],
            'values': [10000, 12000, ...],
            'metrics': {'mae': 100.5, 'rmse': 150.2}
        }
        
    Raises:
        ValueError: 当 line_no 无效时
        DatabaseError: 当数据库连接失败时
        
    Examples:
        >>> result = predict_daily_flow('01', '20250101', 7)
        >>> print(result['dates'])
        ['20250101', '20250102', ...]
    """
    pass
```

---

## 开发流程

### 1. Fork 项目

在 GitHub 上 fork 本项目到您的账户。

### 2. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/STFS_V1.git
cd STFS_V1
```

### 3. 创建分支

```bash
# 功能开发
git checkout -b feature/your-feature-name

# Bug修复
git checkout -b fix/your-bug-fix

# 文档更新
git checkout -b docs/your-doc-update
```

### 4. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖（如果有）
pip install -r requirements-dev.txt
```

### 5. 开发和测试

```bash
# 运行测试
pytest tests/

# 代码风格检查
flake8 .

# 类型检查
mypy .

# 格式化代码
black .
```

### 6. 提交更改

```bash
git add .
git commit -m "feat: add new prediction algorithm"
```

### 7. 推送分支

```bash
git push origin feature/your-feature-name
```

### 8. 创建 Pull Request

在 GitHub 上创建 Pull Request，并填写详细的描述。

---

## 提交规范

### Commit Message 格式

使用约定式提交（Conventional Commits）：

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type 类型

- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整（不影响功能）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建或辅助工具变动

#### 示例

```bash
# 新功能
git commit -m "feat(prediction): add transformer algorithm support"

# Bug修复
git commit -m "fix(database): resolve connection pool leak"

# 文档更新
git commit -m "docs(readme): update installation guide"

# 重构
git commit -m "refactor(knn): optimize data preprocessing logic"
```

---

## 测试要求

### 单元测试

为新功能编写单元测试：

```python
# tests/test_knn_model.py
import pytest
from enknn_model import KNNFlowPredictor

def test_knn_predictor_initialization():
    """测试KNN预测器初始化"""
    config = {'n_neighbors': 5}
    predictor = KNNFlowPredictor('models/test', '20250101', config)
    assert predictor.version == '20250101'
    assert predictor.config['n_neighbors'] == 5

def test_knn_prediction():
    """测试KNN预测功能"""
    # 准备测试数据
    test_data = prepare_test_data()
    
    # 执行预测
    predictor = KNNFlowPredictor('models/test', '20250101', {})
    result, error = predictor.predict(test_data, '01', '20250101')
    
    # 验证结果
    assert error is None
    assert len(result) > 0
```

### 集成测试

测试完整的预测流程：

```python
def test_daily_prediction_flow():
    """测试日预测完整流程"""
    result = predict_and_plot_timeseries_flow_daily(
        file_path='',
        predict_start_date='20250101',
        algorithm='knn',
        mode='predict',
        days=15
    )
    
    assert 'error' not in result
    assert '01' in result  # 至少有一条线路的结果
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_knn_model.py

# 运行特定测试函数
pytest tests/test_knn_model.py::test_knn_prediction

# 查看覆盖率
pytest --cov=. tests/
```

---

## 文档编写

### README 更新

如果您的更改影响到使用方式，请更新 README.md：

- 添加新功能说明
- 更新配置示例
- 补充API文档

### 代码注释

为复杂逻辑添加注释：

```python
def calculate_weighted_prediction(knn_pred, offset_pred, weights):
    """
    计算加权预测结果
    
    使用线性加权组合KNN预测和偏移预测：
    final_pred = w1 * knn_pred + w2 * offset_pred
    
    权重会自动归一化，确保总和为1.0
    """
    # 归一化权重
    total_weight = weights['knn'] + weights['offset']
    w1 = weights['knn'] / total_weight
    w2 = weights['offset'] / total_weight
    
    # 加权求和
    return w1 * knn_pred + w2 * offset_pred
```

### API 文档

如果添加新的API端点，更新API文档：

```python
@app.post("/predict/{flow_type}/daily/{metric_type}")
def predict_daily_flow(
    flow_type: str,
    metric_type: str,
    req: PredictDailyRequest
) -> PredictionResponse:
    """
    执行日客流预测
    
    ## 参数说明
    - **flow_type**: 客流类型（xianwangxianlu/chezhan）
    - **metric_type**: 指标类型（F_PKLCOUNT/F_ENTRANCE等）
    - **req**: 预测请求体
    
    ## 返回值
    返回预测结果，包含：
    - predictions: 预测数据
    - plot_url: 预测图表URL
    - metadata: 元数据信息
    
    ## 示例
    ```json
    {
        "algorithm": "knn",
        "model_version_date": "20250101",
        "predict_start_date": "20250115",
        "days": 15
    }
    ```
    """
    pass
```

---

## Pull Request 清单

提交 PR 前，请确认：

- [ ] 代码符合项目风格规范
- [ ] 已添加必要的单元测试
- [ ] 所有测试通过
- [ ] 已更新相关文档
- [ ] Commit message 符合规范
- [ ] 代码已通过 linter 检查
- [ ] 没有遗留的 debug 代码或注释
- [ ] PR 描述清晰，说明了更改内容和原因

---

## 代码审查

### 审查者关注点

- 代码质量和可读性
- 测试覆盖率
- 性能影响
- 安全性问题
- 文档完整性

### 作者响应

- 及时回复审查意见
- 认真对待建议
- 必要时进行修改
- 保持友好和专业

---

## 问题反馈

如果遇到问题：

1. 检查 [常见问题](README.md#常见问题)
2. 查看现有 [Issues](https://github.com/your-repo/STFS_V1/issues)
3. 创建新 Issue，提供详细信息：
   - 问题描述
   - 重现步骤
   - 环境信息
   - 错误日志

---

## 社区准则

- 尊重他人
- 保持友好和专业
- 接受建设性批评
- 关注项目目标

---

感谢您的贡献！🎉

