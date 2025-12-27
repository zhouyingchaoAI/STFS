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
    print(f"✓ 创建目录: {target_dir}")
except Exception as e:
    print(f"✗ STFS_V1 文件夹创建失败: {e}")
    sys.exit(1)

# 新建 STFS_V1/logs 文件夹，新建失败需要提示
logs_dir = os.path.join(target_dir, "logs")
try:
    os.makedirs(logs_dir, exist_ok=True)
    print(f"✓ 创建目录: {logs_dir}")
except Exception as e:
    print(f"✗ STFS_V1/logs 文件夹创建失败: {e}")
    sys.exit(1)

# 检查 Cython 是否可用
try:
    from Cython.Build import cythonize
except ImportError:
    print("✗ 错误: 未安装 Cython，请运行: pip install Cython")
    sys.exit(1)

# 检查模块文件是否存在
missing_modules = []
existing_modules = []
for module in modules:
    if not os.path.exists(module):
        missing_modules.append(module)
    else:
        existing_modules.append(module)

if missing_modules:
    print(f"⚠ 警告: 以下模块文件不存在: {missing_modules}")
    print(f"继续编译存在的模块: {len(existing_modules)} 个")

if not existing_modules:
    print("✗ 错误: 没有找到可编译的模块")
    sys.exit(1)

# 编译输出到 build 目录
build_dir = os.path.join(os.path.dirname(__file__), "build")
print(f"\n开始编译模块，输出目录: {build_dir}")

# 运行 setup
if __name__ == "__main__":
    try:
        # 使用 sys.argv 来传递 build_ext 命令
        original_argv = sys.argv[:]
        sys.argv = [__file__, "build_ext", "--inplace"]
        
        setup(
            name="STFS_V1",
            ext_modules=cythonize(
                existing_modules,
                compiler_directives={"language_level": "3"},  # 指定 Python3
                build_dir=build_dir,
            ),
            zip_safe=False,
        )
        
        # 恢复原始 argv
        sys.argv = original_argv
        
        print("✓ 编译完成")
        
    except Exception as e:
        print(f"✗ 编译过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# 需要复制的文件
copy_files = ["main.py", "server.py", "supervisord.conf", "db_config.yaml", "stationid_stationname_to_lineid.yaml"]
print("\n开始复制文件...")
for fname in copy_files:
    src = os.path.join(os.path.dirname(__file__), fname)
    dst = os.path.join(target_dir, fname)
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst)
            print(f"✓ 复制: {fname} -> {dst}")
        except Exception as e:
            print(f"✗ 复制失败: {fname} - {e}")
    else:
        print(f"⚠ 警告: {fname} 不存在，无法复制到 STFS_V1 文件夹。")

print("\n✓ 构建完成！")

