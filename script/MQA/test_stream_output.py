#!/usr/bin/env python3
"""
测试流式输出功能
"""
import requests
import json
import time

def test_stream_query():
    """测试流式查询"""
    url = "http://localhost:4577/api/v1/query/stream"
    data = {
        "question": "查询1号线昨天的客流量",
        "options": {
            "use_llm": True
        }
    }
    
    print("开始测试流式查询...")
    print(f"请求: {json.dumps(data, ensure_ascii=False, indent=2)}")
    print("\n" + "="*60)
    print("流式响应:")
    print("="*60 + "\n")
    
    try:
        response = requests.post(url, json=data, stream=True, timeout=120)
        response.raise_for_status()
        
        thinking_count = 0
        sql_count = 0
        step_count = 0
        result_count = 0
        error_count = 0
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    try:
                        data = json.loads(line_str[6:])
                        msg_type = data.get('type', 'unknown')
                        
                        if msg_type == 'thinking':
                            thinking_count += 1
                            content = data.get('content', '')
                            print(f"[思考 {thinking_count}] {repr(content[:100])}")
                            
                        elif msg_type == 'sql_generated':
                            sql_count += 1
                            print(f"[SQL生成] SQL: {data.get('sql', '')[:100]}...")
                            
                        elif msg_type == 'step':
                            step_count += 1
                            step = data.get('step', '')
                            status = data.get('status', '')
                            message = data.get('message', '')
                            print(f"[步骤 {step_count}] {step} - {status}: {message}")
                            
                        elif msg_type == 'result_preview':
                            result_count += 1
                            preview_count = data.get('preview_count', 0)
                            total_rows = data.get('total_rows', 0)
                            print(f"[预览 {result_count}] {preview_count}/{total_rows} 行")
                            
                        elif msg_type == 'result_formatted':
                            result_count += 1
                            row_count = data.get('row_count', 0)
                            print(f"[格式化] {row_count} 行数据")
                            
                        elif msg_type == 'error':
                            error_count += 1
                            message = data.get('message', '')
                            print(f"[错误 {error_count}] {message}")
                            
                        elif msg_type == 'complete':
                            print(f"[完成] 查询完成")
                            
                    except json.JSONDecodeError as e:
                        print(f"[解析错误] {e}: {line_str[:100]}")
        
        print("\n" + "="*60)
        print(f"统计: 思考={thinking_count}, SQL={sql_count}, 步骤={step_count}, 结果={result_count}, 错误={error_count}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_stream_query()
