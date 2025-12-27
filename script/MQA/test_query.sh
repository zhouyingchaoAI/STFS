#!/bin/bash
# 查询功能测试脚本

echo "=========================================="
echo "地铁客流智能问数系统 - 查询功能测试"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试函数
test_query() {
    local question="$1"
    local description="$2"
    
    echo -e "${YELLOW}测试: $description${NC}"
    echo "查询: $question"
    
    response=$(curl -s -X POST http://localhost:4577/api/v1/query \
        -H "Content-Type: application/json" \
        -d "{\"question\":\"$question\",\"options\":{}}")
    
    code=$(echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('code', 'ERROR'))" 2>/dev/null)
    message=$(echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('message', 'ERROR'))" 2>/dev/null)
    row_count=$(echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('data', {}).get('row_count', 0))" 2>/dev/null)
    
    if [ "$code" = "200" ]; then
        echo -e "${GREEN}✓ 成功${NC} - 状态码: $code, 消息: $message, 行数: $row_count"
    else
        echo -e "${RED}✗ 失败${NC} - 状态码: $code, 消息: $message"
        echo "响应: $response" | head -3
    fi
    echo ""
}

# 1. 检查后端服务
echo "1. 检查后端服务..."
health=$(curl -s http://localhost:4577/health)
if echo "$health" | grep -q "healthy"; then
    echo -e "${GREEN}✓ 后端服务正常${NC}"
else
    echo -e "${RED}✗ 后端服务异常${NC}"
    echo "请先启动后端服务: cd backend && python3 -m app.main"
    exit 1
fi
echo ""

# 2. 测试各种查询
echo "2. 测试查询功能..."
echo ""

test_query "查询1号线昨天的客流量" "简单线路查询"
test_query "查询1号线最近的客流量" "最近数据查询"
test_query "查询所有线路昨天的客流量" "所有线路查询"
test_query "查询今天的客流量" "今天数据查询"
test_query "查询1号线2024年12月1日的客流量" "指定日期查询"

echo "=========================================="
echo "测试完成"
echo "=========================================="
