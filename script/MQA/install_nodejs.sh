#!/bin/bash
# Node.js 和 npm 快速安装脚本

echo "=========================================="
echo "Node.js 和 npm 安装脚本"
echo "=========================================="
echo ""

# 检测系统类型
if [ -f /etc/debian_version ]; then
    echo "检测到 Debian/Ubuntu 系统"
    SYSTEM_TYPE="debian"
elif [ -f /etc/redhat-release ]; then
    echo "检测到 CentOS/RHEL 系统"
    SYSTEM_TYPE="redhat"
else
    echo "无法识别系统类型，尝试通用安装..."
    SYSTEM_TYPE="unknown"
fi

# 检查是否已安装
if command -v node &> /dev/null; then
    echo "Node.js 已安装: $(node --version)"
    read -p "是否重新安装？(y/N): " reinstall
    if [ "$reinstall" != "y" ] && [ "$reinstall" != "Y" ]; then
        echo "跳过安装"
        exit 0
    fi
fi

# 安装方法选择
echo ""
echo "请选择安装方法："
echo "1. 使用系统包管理器（简单，但可能版本较旧）"
echo "2. 使用 NodeSource 仓库（推荐，版本较新）"
echo "3. 使用 NVM（最灵活，可管理多个版本）"
read -p "请输入选项 (1/2/3) [默认: 2]: " choice
choice=${choice:-2}

case $choice in
    1)
        echo "使用系统包管理器安装..."
        if [ "$SYSTEM_TYPE" == "debian" ]; then
            sudo apt update
            sudo apt install -y nodejs npm
        elif [ "$SYSTEM_TYPE" == "redhat" ]; then
            sudo yum install -y nodejs npm
        else
            echo "请手动安装 Node.js"
            exit 1
        fi
        ;;
    2)
        echo "使用 NodeSource 仓库安装 Node.js 18.x LTS..."
        if [ "$SYSTEM_TYPE" == "debian" ]; then
            curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
            sudo apt-get install -y nodejs
        elif [ "$SYSTEM_TYPE" == "redhat" ]; then
            curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
            sudo yum install -y nodejs
        else
            echo "请手动安装 Node.js"
            exit 1
        fi
        ;;
    3)
        echo "安装 NVM (Node Version Manager)..."
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        nvm install --lts
        nvm use --lts
        echo ""
        echo "⚠️  请运行以下命令使 NVM 生效："
        echo "source ~/.bashrc"
        echo "或重新打开终端"
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

# 验证安装
echo ""
echo "验证安装..."
if command -v node &> /dev/null; then
    echo "✅ Node.js: $(node --version)"
else
    echo "❌ Node.js 安装失败"
    exit 1
fi

if command -v npm &> /dev/null; then
    echo "✅ npm: $(npm --version)"
else
    echo "❌ npm 安装失败"
    exit 1
fi

# 配置 npm 镜像（可选）
echo ""
read -p "是否配置 npm 使用国内镜像（加速下载）？(Y/n): " use_mirror
use_mirror=${use_mirror:-Y}

if [ "$use_mirror" == "Y" ] || [ "$use_mirror" == "y" ]; then
    npm config set registry https://registry.npmmirror.com
    echo "✅ 已配置 npm 使用国内镜像"
    echo "镜像地址: $(npm config get registry)"
fi

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. cd /STFS_V1/script/MQA/frontend"
echo "2. npm install"
echo "3. npm run dev"
echo ""

