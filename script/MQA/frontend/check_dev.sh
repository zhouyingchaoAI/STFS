#!/bin/bash
# 前端启动诊断脚本

echo "=========================================="
echo "前端启动诊断"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# 1. 检查依赖
echo "1. 检查依赖..."
if [ -d "node_modules" ]; then
    echo "   ✅ node_modules 存在"
else
    echo "   ❌ node_modules 不存在"
    echo "   请运行: npm install"
    exit 1
fi

# 2. 检查关键文件
echo ""
echo "2. 检查关键文件..."
files=("src/main.js" "src/App.vue" "src/router/index.js" "src/views/QueryPage.vue" "index.html")
all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file 不存在"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    echo ""
    echo "❌ 缺少关键文件，请检查项目结构"
    exit 1
fi

# 3. 检查端口
echo ""
echo "3. 检查端口 3000..."
if lsof -i :3000 > /dev/null 2>&1; then
    echo "   ⚠️  端口 3000 被占用"
    echo "   占用进程:"
    lsof -i :3000
    echo ""
    read -p "是否终止占用进程？(y/N): " kill_process
    if [ "$kill_process" = "y" ] || [ "$kill_process" = "Y" ]; then
        lsof -ti :3000 | xargs kill -9
        echo "   ✅ 已终止进程"
    else
        echo "   请手动终止进程或修改 vite.config.js 中的端口"
        exit 1
    fi
else
    echo "   ✅ 端口 3000 可用"
fi

# 4. 检查Node.js和npm
echo ""
echo "4. 检查环境..."
if command -v node &> /dev/null; then
    echo "   ✅ Node.js: $(node --version)"
else
    echo "   ❌ Node.js 未安装"
    exit 1
fi

if command -v npm &> /dev/null; then
    echo "   ✅ npm: $(npm --version)"
else
    echo "   ❌ npm 未安装"
    exit 1
fi

# 5. 测试启动
echo ""
echo "5. 测试服务启动..."
echo "   正在启动开发服务器..."
echo ""

# 启动服务（后台）
npm run dev > /tmp/vite_dev.log 2>&1 &
VITE_PID=$!

# 等待服务启动
sleep 5

# 检查服务是否响应
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "   ✅ 服务启动成功！"
    echo ""
    echo "=========================================="
    echo "服务信息"
    echo "=========================================="
    echo "本地访问: http://localhost:3000"
    echo "网络访问: http://$(hostname -I | awk '{print $1}'):3000"
    echo ""
    echo "进程ID: $VITE_PID"
    echo "日志文件: /tmp/vite_dev.log"
    echo ""
    echo "查看日志: tail -f /tmp/vite_dev.log"
    echo "停止服务: kill $VITE_PID"
    echo ""
    echo "按 Ctrl+C 停止服务"
    echo "=========================================="
    
    # 等待用户中断
    wait $VITE_PID
else
    echo "   ❌ 服务启动失败"
    echo ""
    echo "错误日志:"
    cat /tmp/vite_dev.log
    echo ""
    kill $VITE_PID 2>/dev/null
    exit 1
fi

