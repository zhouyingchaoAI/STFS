# 使用官方 Python 3.10 slim 版本
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 更新 apt-get 并安装中文字体和 supervisor
RUN apt-get update && \
    apt-get install -y fonts-noto-cjk supervisor && \
    rm -rf /var/lib/apt/lists/*

# 拷贝 requirements.txt 并安装 Python 包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝应用代码和 supervisord 配置文件
COPY . .
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 暴露 Streamlit 和 FastAPI 常用端口
EXPOSE 8501 8000

# 启动 supervisord 管理两个服务
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
