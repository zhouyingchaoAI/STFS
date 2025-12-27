# 项目启动指南

## 🚀 快速启动

### 方法1：使用启动脚本（推荐）

```bash
cd /STFS_V1/script/MQA
./启动项目.sh
```

脚本会自动：
- ✅ 检查环境
- ✅ 安装缺失的依赖
- ✅ 检查端口占用
- ✅ 启动服务

### 方法2：手动启动

#### 启动后端

```bash
cd /STFS_V1/script/MQA/backend
python3 -m app.main
```

后端将在 **http://localhost:4577** 启动

#### 启动前端

```bash
cd /STFS_V1/script/MQA/frontend
npm run dev
```

前端将在 **http://localhost:3000** 启动

## 📋 启动前检查清单

### 1. 环境要求

- ✅ Python 3.8+ 已安装
- ✅ Node.js 16+ 已安装
- ✅ npm 已安装

### 2. 依赖安装

**后端依赖：**
```bash
cd /STFS_V1/script/MQA/backend
pip3 install -r requirements.txt
```

**前端依赖：**
```bash
cd /STFS_V1/script/MQA/frontend
npm install
```

### 3. 配置检查

- 数据库连接配置（`backend/app/config.py`）
- Ollama配置（如果需要LLM功能）
- 前端API代理配置（`frontend/vite.config.js`）

## 🔍 常见启动问题

### 问题1：后端无法启动

**错误：** `ModuleNotFoundError`

**解决：**
```bash
cd /STFS_V1/script/MQA/backend
pip3 install -r requirements.txt
```

### 问题2：前端无法启动

**错误：** `command not found: npm`

**解决：**
```bash
# 安装Node.js和npm
./install_nodejs.sh
```

### 问题3：端口被占用

**错误：** `Address already in use`

**解决：**
```bash
# 查找占用进程
lsof -i :4577  # 后端端口
lsof -i :3000  # 前端端口

# 终止进程
kill -9 <PID>
```

### 问题4：数据库连接失败

**错误：** `pymssql.OperationalError`

**解决：**
1. 检查数据库服务是否运行
2. 验证 `backend/app/config.py` 中的数据库配置
3. 测试连接：`python3 get_data_struct.py`

### 问题5：前端显示空白

**解决：**
1. 打开浏览器开发者工具（F12）
2. 查看Console标签的错误信息
3. 检查Network标签的资源加载情况
4. 确认后端服务正在运行

## 🎯 验证启动成功

### 后端验证

```bash
# 健康检查
curl http://localhost:4577/health

# 应该返回: {"status":"healthy"}
```

### 前端验证

1. 打开浏览器
2. 访问 http://localhost:3000
3. 应该看到系统界面

## 📝 完整启动流程

```bash
# 1. 进入项目目录
cd /STFS_V1/script/MQA

# 2. 使用启动脚本（推荐）
./启动项目.sh

# 或者手动启动：

# 终端1：启动后端
cd backend
python3 -m app.main

# 终端2：启动前端
cd frontend
npm run dev

# 3. 打开浏览器访问
# http://localhost:3000
```

## 🔧 调试模式

### 后端调试

```bash
cd /STFS_V1/script/MQA/backend
# 设置DEBUG模式
export DEBUG=True
python3 -m app.main
```

### 前端调试

```bash
cd /STFS_V1/script/MQA/frontend
# 查看详细日志
npm run dev -- --debug
```

## 📚 相关文档

- 系统详细设计文档
- 项目结构说明
- Ollama使用说明

## 🆘 仍然无法启动？

1. **查看错误日志**
   - 后端：查看终端输出
   - 前端：查看浏览器控制台（F12）

2. **检查文件完整性**
   ```bash
   ls -la backend/app/
   ls -la frontend/src/
   ```

3. **重新安装依赖**
   ```bash
   # 后端
   cd backend && pip3 install -r requirements.txt --force-reinstall
   
   # 前端
   cd frontend && rm -rf node_modules && npm install
   ```

4. **检查系统资源**
   ```bash
   free -h  # 检查内存
   df -h    # 检查磁盘空间
   ```

