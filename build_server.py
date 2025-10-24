from setuptools import setup
from Cython.Build import cythonize
import os
import shutil
import sys

# 需要编译的模块
modules = [
    "streamlit_app.py",
    "streamlit_daily.py",
    "streamlit_hourly.py",
    "predict_daily.py",
    "predict_hourly.py",
    "hourknn_model.py",
    "enknn_model.py",
    "config_utils.py",
    "db_utils.py",
    "plot_utils.py",
    "font_utils.py",
    "weather_enum.py",
]

# 新建 STFS_V1 文件夹，新建失败需要提示
target_dir = os.path.join(os.path.dirname(__file__), "STFS_V1")
try:
    os.makedirs(target_dir, exist_ok=True)
except Exception as e:
    print(f"STFS_V1 文件夹创建失败: {e}")
    sys.exit(1)

# 新建 STFS_V1/logs 文件夹，新建失败需要提示
logs_dir = os.path.join(target_dir, "logs")
try:
    os.makedirs(logs_dir, exist_ok=True)
except Exception as e:
    print(f"STFS_V1/logs 文件夹创建失败: {e}")
    sys.exit(1)

# 编译输出到 build 目录
build_dir = os.path.join(os.path.dirname(__file__), "build")
setup(
    name="STFS_V1",
    ext_modules=cythonize(
        modules,
        compiler_directives={"language_level": "3"},  # 指定 Python3
        build_dir=build_dir,
    ),
    zip_safe=False,
)


# 需要复制的文件
copy_files = ["main.py", "server.py", "supervisord.conf", "db_config.yaml", "stationid_stationname_to_lineid.yaml"]
for fname in copy_files:
    src = os.path.join(os.path.dirname(__file__), fname)
    dst = os.path.join(target_dir, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        print(f"日志丢失: {fname} 不存在，无法复制到 STFS_V1 文件夹。")
