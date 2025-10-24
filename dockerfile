FROM keliu-model-cpu:v7

# # 设置工作目录
# WORKDIR /STFS_V1

# # 暴露 Streamlit 和 FastAPI 常用端口
# EXPOSE 8501 8000

# # 启动 supervisord 管理两个服务
# CMD ["/usr/bin/supervisord", "-c", "/STFS_V1/supervisord.conf"]

# 设置工作目录
WORKDIR /STFS_TASK

# 启动 supervisord 管理两个服务
CMD ["/usr/local/bin/python", "/STFS_TASK/task.py"]