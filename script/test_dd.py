"""
测试脚本：检查运行环境是否正常
"""
import sys

print("=== 环境检查 ===")

# 检查Python版本
print(f"Python版本: {sys.version}")

# 检查必要的包
required_packages = ['pandas', 'pymssql', 'matplotlib', 'numpy']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package} 已安装")
    except ImportError:
        print(f"✗ {package} 未安装")
        missing_packages.append(package)

if missing_packages:
    print(f"\n缺少以下包，请安装: pip install {' '.join(missing_packages)}")
else:
    print("\n所有必要的包都已安装")

# 检查数据库连接
print("\n=== 数据库连接测试 ===")
try:
    import pymssql
    DB_CONFIG = {
        "server": "10.1.6.230",
        "user": "sa",
        "password": "YourStrong!Passw0rd",
        "database": "StationFlowPredict",
        "port": 1433
    }
    print(f"尝试连接数据库: {DB_CONFIG['server']}:{DB_CONFIG['port']}")
    conn = pymssql.connect(**DB_CONFIG)
    print("✓ 数据库连接成功")
    conn.close()
except Exception as e:
    print(f"✗ 数据库连接失败: {e}")

print("\n=== 检查完成 ===")

