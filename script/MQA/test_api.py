#!/usr/bin/env python
"""
API测试脚本
"""
import requests
import json
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"


def print_response(title: str, response: requests.Response):
    """打印响应结果"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"状态码: {response.status_code}")
    try:
        data = response.json()
        print(f"响应内容:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except:
        print(f"响应内容: {response.text}")
    print(f"{'='*60}\n")


def test_health():
    """测试健康检查"""
    print("测试健康检查接口...")
    response = requests.get(f"{BASE_URL}/health")
    print_response("健康检查", response)
    return response.status_code == 200


def test_root():
    """测试根路径"""
    print("测试根路径...")
    response = requests.get(f"{BASE_URL}/")
    print_response("根路径", response)
    return response.status_code == 200


def test_natural_language_query():
    """测试自然语言查询"""
    print("测试自然语言查询接口...")
    
    test_cases = [
        {
            "name": "简单查询 - 线路客流",
            "question": "查询1号线昨天的客流量"
        },
        {
            "name": "车站查询",
            "question": "五一广场站今天的进站量"
        },
        {
            "name": "强制使用LLM",
            "question": "对比1号线和2号线本周的平均客流量",
            "options": {"use_llm": True}
        }
    ]
    
    results = []
    for test_case in test_cases:
        print(f"\n测试用例: {test_case['name']}")
        payload = {
            "question": test_case["question"]
        }
        if "options" in test_case:
            payload["options"] = test_case["options"]
        
        try:
            response = requests.post(
                f"{BASE_URL}{API_PREFIX}/query",
                json=payload,
                timeout=60
            )
            print_response(f"自然语言查询 - {test_case['name']}", response)
            results.append(response.status_code == 200)
        except requests.exceptions.Timeout:
            print(f"❌ 请求超时: {test_case['name']}")
            results.append(False)
        except Exception as e:
            print(f"❌ 错误: {e}")
            results.append(False)
    
    return all(results)


def test_sql_query():
    """测试SQL直接查询"""
    print("测试SQL直接查询接口...")
    
    sql = """
    SELECT TOP 5 
        f_date as 日期,
        f_linename as 线路名,
        f_klcount as 客流量
    FROM dbo.LineDailyFlowHistory
    ORDER BY f_date DESC
    """
    
    payload = {
        "sql": sql.strip(),
        "database": "master"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}{API_PREFIX}/sql",
            json=payload,
            timeout=30
        )
        print_response("SQL直接查询", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ SQL查询错误: {e}")
        return False


def test_metadata():
    """测试元数据接口"""
    print("测试元数据接口...")
    
    endpoints = [
        ("/metadata/tables", "表列表"),
        ("/metadata/stations", "车站列表"),
        ("/metadata/lines", "线路列表")
    ]
    
    results = []
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{API_PREFIX}{endpoint}")
            print_response(f"元数据 - {name}", response)
            results.append(response.status_code == 200)
        except Exception as e:
            print(f"❌ {name} 错误: {e}")
            results.append(False)
    
    return all(results)


def main():
    """主测试函数"""
    print("="*60)
    print("地铁客流智能问数系统 - API测试")
    print("="*60)
    print(f"测试服务器: {BASE_URL}")
    print()
    
    # 检查服务是否运行
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ 服务未正常运行，状态码: {response.status_code}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务器 {BASE_URL}")
        print("请确保服务已启动: python -m app.main")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 连接错误: {e}")
        sys.exit(1)
    
    print("✅ 服务连接成功\n")
    
    # 运行测试
    tests = [
        ("健康检查", test_health),
        ("根路径", test_root),
        ("自然语言查询", test_natural_language_query),
        ("SQL直接查询", test_sql_query),
        ("元数据接口", test_metadata),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            if result:
                print(f"✅ {name} 测试通过")
            else:
                print(f"❌ {name} 测试失败")
        except Exception as e:
            print(f"❌ {name} 测试异常: {e}")
            results.append((name, False))
    
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
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)

