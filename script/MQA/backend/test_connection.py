#!/usr/bin/env python
"""
连接测试脚本 - 测试数据库和Ollama连接
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.core.query_executor.db_manager import db_manager
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_database_connection():
    """测试数据库连接"""
    print("="*60)
    print("测试数据库连接")
    print("="*60)
    
    databases = ["master", "CxFlowPredict"]
    results = []
    
    for db_name in databases:
        print(f"\n测试数据库: {db_name}")
        try:
            with db_manager.get_connection(db_name) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT @@VERSION")
                version = cursor.fetchone()
                print(f"✅ {db_name} 连接成功")
                print(f"   版本: {version[0][:50]}...")
                cursor.close()
                results.append(True)
        except Exception as e:
            print(f"❌ {db_name} 连接失败: {e}")
            results.append(False)
    
    return all(results)


def test_ollama_connection():
    """测试Ollama连接"""
    print("\n" + "="*60)
    print("测试Ollama连接")
    print("="*60)
    
    if not settings.LLM_ENABLED:
        print("⚠️  LLM未启用，跳过Ollama测试")
        return True
    
    try:
        import ollama
        
        print(f"Ollama地址: {settings.LLM_API_BASE}")
        print(f"模型: {settings.LLM_MODEL}")
        
        client = ollama.Client(host=settings.LLM_API_BASE)
        
        # 测试连接
        try:
            models = client.list()
            print(f"✅ Ollama连接成功")
            print(f"   可用模型: {[m['name'] for m in models.get('models', [])]}")
            
            # 检查指定模型是否存在
            model_names = [m['name'] for m in models.get('models', [])]
            if settings.LLM_MODEL in model_names:
                print(f"✅ 模型 {settings.LLM_MODEL} 已下载")
            else:
                print(f"⚠️  模型 {settings.LLM_MODEL} 未找到")
                print(f"   请运行: ollama pull {settings.LLM_MODEL}")
            
            return True
        except Exception as e:
            print(f"❌ Ollama连接失败: {e}")
            return False
            
    except ImportError:
        print("❌ ollama包未安装")
        print("   请运行: pip install ollama")
        return False
    except Exception as e:
        print(f"❌ Ollama测试异常: {e}")
        return False


def test_nl2sql_engine():
    """测试NL2SQL引擎"""
    print("\n" + "="*60)
    print("测试NL2SQL引擎")
    print("="*60)
    
    try:
        from app.core.nl2sql.hybrid_engine import HybridNL2SQLEngine
        
        engine = HybridNL2SQLEngine()
        print("✅ NL2SQL引擎初始化成功")
        
        # 测试简单查询
        test_question = "查询1号线昨天的客流量"
        print(f"\n测试查询: {test_question}")
        
        result = engine.convert(test_question)
        if result and result.get("sql"):
            print(f"✅ SQL生成成功")
            print(f"   意图: {result.get('intent')}")
            print(f"   SQL: {result.get('sql')[:100]}...")
            return True
        else:
            print(f"❌ SQL生成失败")
            return False
            
    except Exception as e:
        print(f"❌ NL2SQL引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("地铁客流智能问数系统 - 连接测试")
    print("="*60)
    
    results = []
    
    # 测试数据库连接
    db_ok = test_database_connection()
    results.append(("数据库连接", db_ok))
    
    # 测试Ollama连接
    ollama_ok = test_ollama_connection()
    results.append(("Ollama连接", ollama_ok))
    
    # 测试NL2SQL引擎
    nl2sql_ok = test_nl2sql_engine()
    results.append(("NL2SQL引擎", nl2sql_ok))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有连接测试通过！")
        print("可以启动服务: python -m app.main")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        print("请检查配置和连接")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)

