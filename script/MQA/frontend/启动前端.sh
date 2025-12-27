#!/bin/bash
# 前端启动脚本（带检查）

cd "$(dirname "$0")"

echo "=========================================="
echo "地铁客流智能问数系统 - 前端启动"
echo "=========================================="
echo ""

# 检查依赖
if [ ! -d "node_modules" ]; then
    echo "⚠️  依赖未安装，正在安装..."
    npm install --registry=https://registry.npmmirror.com
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败"
        exit 1
    fi
    echo "✅ 依赖安装完成"
    echo ""
fi

# 检查端口
if lsof -i :3000 > /dev/null 2>&1; then
    echo "⚠️  端口 3000 已被占用"
    echo "占用进程："
    lsof -i :3000
    echo ""
    read -p "是否终止占用进程？(y/N): " kill_it
    if [ "$kill_it" = "y" ] || [ "$kill_it" = "Y" ]; then
        lsof -ti :3000 | xargs kill -9 2>/dev/null
        sleep 1
        echo "✅ 已终止进程"
    else
        echo "请手动终止进程或修改 vite.config.js 中的端口"
        exit 1
    fi
    echo ""
fi

# 启动服务
echo "正在启动开发服务器..."
echo ""
echo "=========================================="
echo "服务启动后，请在浏览器中访问："
echo "  http://localhost:3000"
echo "  http://127.0.0.1:3000"
echo "=========================================="
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

npm run dev

