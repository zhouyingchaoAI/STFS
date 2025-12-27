#!/bin/bash
# 项目启动脚本

set -e

PROJECT_DIR="/STFS_V1/script/MQA"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

echo "=========================================="
echo "地铁客流智能问数系统 - 启动检查"
echo "=========================================="
echo ""

# 检查Python
echo "1. 检查Python环境..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✅ $PYTHON_VERSION"
else
    echo "   ❌ Python3 未安装"
    exit 1
fi

# 检查后端依赖
echo ""
echo "2. 检查后端依赖..."
cd "$BACKEND_DIR"
if python3 -c "import fastapi" 2>/dev/null; then
    echo "   ✅ 后端依赖已安装"
else
    echo "   ⚠️  后端依赖未安装，正在安装..."
    pip3 install -r requirements.txt --quiet
    echo "   ✅ 后端依赖安装完成"
fi

# 检查后端代码
echo ""
echo "3. 检查后端代码..."
if python3 -c "from app.main import app" 2>/dev/null; then
    echo "   ✅ 后端代码正常"
else
    echo "   ❌ 后端代码有错误"
    python3 -c "from app.main import app" 2>&1 | head -10
    exit 1
fi

# 检查Node.js和npm
echo ""
echo "4. 检查前端环境..."
if command -v node &> /dev/null && command -v npm &> /dev/null; then
    echo "   ✅ Node.js: $(node --version)"
    echo "   ✅ npm: $(npm --version)"
else
    echo "   ❌ Node.js 或 npm 未安装"
    echo "   请运行: ./install_nodejs.sh"
    exit 1
fi

# 检查前端依赖
echo ""
echo "5. 检查前端依赖..."
cd "$FRONTEND_DIR"
if [ -d "node_modules" ]; then
    echo "   ✅ 前端依赖已安装"
else
    echo "   ⚠️  前端依赖未安装，正在安装..."
    npm install --registry=https://registry.npmmirror.com --silent
    echo "   ✅ 前端依赖安装完成"
fi

# 检查端口
echo ""
echo "6. 检查端口占用..."
BACKEND_PORT=4577
FRONTEND_PORT=3000

if lsof -i :$BACKEND_PORT > /dev/null 2>&1; then
    echo "   ⚠️  后端端口 $BACKEND_PORT 被占用"
    read -p "   是否终止占用进程？(y/N): " kill_backend
    if [ "$kill_backend" = "y" ] || [ "$kill_backend" = "Y" ]; then
        lsof -ti :$BACKEND_PORT | xargs kill -9 2>/dev/null
        echo "   ✅ 已终止进程"
    fi
else
    echo "   ✅ 后端端口 $BACKEND_PORT 可用"
fi

if lsof -i :$FRONTEND_PORT > /dev/null 2>&1; then
    echo "   ⚠️  前端端口 $FRONTEND_PORT 被占用"
    read -p "   是否终止占用进程？(y/N): " kill_frontend
    if [ "$kill_frontend" = "y" ] || [ "$kill_frontend" = "Y" ]; then
        lsof -ti :$FRONTEND_PORT | xargs kill -9 2>/dev/null
        echo "   ✅ 已终止进程"
    fi
else
    echo "   ✅ 前端端口 $FRONTEND_PORT 可用"
fi

echo ""
echo "=========================================="
echo "启动选项"
echo "=========================================="
echo ""
echo "1. 仅启动后端"
echo "2. 仅启动前端"
echo "3. 同时启动后端和前端（推荐）"
echo ""
read -p "请选择 (1/2/3) [默认: 3]: " choice
choice=${choice:-3}

case $choice in
    1)
        echo ""
        echo "启动后端服务..."
        cd "$BACKEND_DIR"
        python3 -m app.main
        ;;
    2)
        echo ""
        echo "启动前端服务..."
        cd "$FRONTEND_DIR"
        npm run dev
        ;;
    3)
        echo ""
        echo "同时启动后端和前端..."
        echo ""
        echo "后端将在 http://localhost:$BACKEND_PORT 启动"
        echo "前端将在 http://localhost:$FRONTEND_PORT 启动"
        echo ""
        echo "按 Ctrl+C 停止所有服务"
        echo ""
        
        # 启动后端（后台）
        cd "$BACKEND_DIR"
        python3 -m app.main > /tmp/backend.log 2>&1 &
        BACKEND_PID=$!
        echo "✅ 后端已启动 (PID: $BACKEND_PID)"
        
        # 等待后端启动
        sleep 2
        
        # 启动前端（前台）
        cd "$FRONTEND_DIR"
        npm run dev
        
        # 清理
        kill $BACKEND_PID 2>/dev/null
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

