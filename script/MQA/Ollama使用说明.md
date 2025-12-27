# Ollama 使用说明

## 1. 安装 Ollama

### Linux/macOS
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
从 [Ollama官网](https://ollama.com/download) 下载安装程序。

## 2. 下载模型

Ollama 支持多种开源大语言模型，推荐用于 NL2SQL 的模型：

```bash
# 推荐模型（按推荐顺序）
ollama pull qwen2.5:7b          # 阿里通义千问，中文理解能力强
ollama pull llama2:7b           # Meta Llama 2，通用性强
ollama pull chatglm3:6b         # 清华ChatGLM，中文优化
ollama pull mistral:7b          # Mistral，性能优秀
ollama pull codellama:7b        # Code Llama，代码生成能力强
```

查看已下载的模型：
```bash
ollama list
```

## 3. 启动 Ollama 服务

Ollama 安装后会自动启动服务，默认运行在 `http://localhost:11434`

验证服务是否运行：
```bash
curl http://localhost:11434/api/tags
```

## 4. 配置系统

### 4.1 环境变量配置

编辑 `backend/.env` 文件：

```env
# 启用LLM引擎
LLM_ENABLED=True

# 使用Ollama
LLM_PROVIDER=ollama

# Ollama API地址（如果Ollama运行在其他机器，修改为对应地址）
LLM_API_BASE=http://localhost:11434

# 模型名称（根据你下载的模型修改）
LLM_MODEL=qwen2.5:7b

# 其他参数
LLM_MAX_TOKENS=2000
LLM_TEMPERATURE=0.1
```

### 4.2 安装Python依赖

```bash
cd backend
pip install ollama
```

或安装所有依赖：
```bash
pip install -r requirements.txt
```

## 5. 使用方式

### 5.1 自动模式（推荐）

系统默认使用混合引擎：
- 简单查询：使用规则引擎（快速）
- 复杂查询：自动切换到LLM引擎

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "查询1号线昨天的客流量"
  }'
```

### 5.2 强制使用LLM

对于复杂查询，可以强制使用LLM：

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "对比1号线和2号线本周的平均客流量，并分析趋势",
    "options": {
      "use_llm": true
    }
  }'
```

## 6. 模型选择建议

### 6.1 中文查询场景

推荐使用：
- **qwen2.5:7b** - 中文理解能力最强，推荐首选
- **chatglm3:6b** - 专为中文优化
- **llama2:7b** - 通用性强，中文支持良好

### 6.2 SQL生成场景

推荐使用：
- **codellama:7b** - 代码生成能力强
- **qwen2.5:7b** - 综合能力强
- **mistral:7b** - 逻辑推理能力强

### 6.3 性能考虑

- **7B模型**：平衡性能和效果，推荐
- **13B模型**：效果更好，但需要更多内存（至少16GB）
- **3B模型**：速度快，但效果可能较差

## 7. 性能优化

### 7.1 模型量化

Ollama 默认使用量化模型，如果内存充足，可以使用完整模型：

```bash
ollama pull qwen2.5:7b-q8_0  # 8-bit量化
ollama pull qwen2.5:7b-f16   # 完整精度（需要更多内存）
```

### 7.2 GPU加速

如果系统有NVIDIA GPU，Ollama会自动使用GPU加速。确保安装了NVIDIA驱动和CUDA。

检查GPU使用情况：
```bash
nvidia-smi
```

### 7.3 调整参数

在 `.env` 中调整参数以平衡速度和效果：

```env
# 减少token数可以加快响应速度
LLM_MAX_TOKENS=1000

# 降低温度可以提高确定性（但可能降低创造性）
LLM_TEMPERATURE=0.0
```

## 8. 故障排查

### 8.1 Ollama服务未启动

```bash
# 检查服务状态
curl http://localhost:11434/api/tags

# 如果失败，启动Ollama服务
ollama serve
```

### 8.2 模型未找到

```bash
# 查看已下载的模型
ollama list

# 如果模型不存在，下载模型
ollama pull qwen2.5:7b
```

### 8.3 内存不足

如果遇到内存不足错误：
- 使用更小的模型（如3B）
- 使用量化模型
- 增加系统内存

### 8.4 响应速度慢

- 使用GPU加速
- 使用更小的模型
- 减少MAX_TOKENS参数
- 使用规则引擎处理简单查询

## 9. 测试LLM功能

### 9.1 测试Ollama连接

```python
import ollama

client = ollama.Client(host='http://localhost:11434')
response = client.generate(
    model='qwen2.5:7b',
    prompt='你好，请用一句话介绍自己。'
)
print(response)
```

### 9.2 测试NL2SQL

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "查询1号线昨天的客流量",
    "options": {
      "use_llm": true
    }
  }'
```

## 10. 最佳实践

1. **混合使用**：简单查询用规则引擎，复杂查询用LLM
2. **模型选择**：根据实际需求选择合适的模型大小
3. **参数调优**：根据查询类型调整temperature和max_tokens
4. **缓存结果**：启用查询缓存以提高性能
5. **监控性能**：关注响应时间和资源使用情况

## 11. 常见问题

**Q: 如何知道应该使用哪个模型？**
A: 建议先尝试 qwen2.5:7b，如果效果不理想，可以尝试其他模型。

**Q: LLM响应很慢怎么办？**
A: 确保使用GPU加速，或使用更小的模型，或减少max_tokens。

**Q: 生成的SQL不正确怎么办？**
A: 可以在prompt中提供更详细的表结构信息，或使用规则引擎作为fallback。

**Q: 可以同时使用多个模型吗？**
A: 可以，但需要修改代码支持模型选择。当前实现使用单一模型。

