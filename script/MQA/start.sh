#!/bin/bash
# 启动脚本

cd "$(dirname "$0")/backend"

echo "=========================================="
echo "地铁客流智能问数系统 - 启动服务"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

echo "✅ Python版本: $(python3 --version)"

# 检查虚拟环境
if [ -d "venv" ]; then
    echo "✅ 发现虚拟环境，正在激活..."
    source venv/bin/activate
else
    echo "⚠️  未发现虚拟环境，使用系统Python"
fi

# 检查依赖
echo ""
echo "检查依赖..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  依赖未安装，正在安装..."
    pip install -r requirements.txt
fi

# 启动服务
echo ""
echo "启动服务..."
echo "服务地址: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

python3 -m app.main

