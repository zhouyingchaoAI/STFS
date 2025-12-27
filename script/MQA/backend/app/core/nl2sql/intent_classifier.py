"""
意图分类器
"""
from typing import Optional
import re

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class IntentClassifier:
    """意图分类器"""
    
    # 意图关键词映射
    INTENT_KEYWORDS = {
        "line_flow_query": [
            "线路", "线", "客流", "客流量", "客运量",
            "进站", "出站", "换乘", "乘降"
        ],
        "station_flow_query": [
            "车站", "站", "站点", "进站量", "出站量"
        ],
        "prediction_query": [
            "预测", "预计", "未来", "明天", "后天", "下周"
        ],
        "comparison_query": [
            "对比", "比较", "相比", "vs", "与", "和"
        ],
        "statistics_query": [
            "统计", "排名", "最高", "最低", "平均", "总和", "最大", "最小"
        ],
        "trend_query": [
            "趋势", "变化", "增长", "下降", "走势", "曲线"
        ]
    }
    
    def classify(self, question: str) -> str:
        """
        分类查询意图
        
        Args:
            question: 自然语言问题
            
        Returns:
            意图类型
        """
        question_lower = question.lower()
        
        # 检查预测意图（优先级最高）
        if any(keyword in question for keyword in self.INTENT_KEYWORDS["prediction_query"]):
            return "prediction_query"
        
        # 检查对比意图
        if any(keyword in question for keyword in self.INTENT_KEYWORDS["comparison_query"]):
            return "comparison_query"
        
        # 检查统计意图
        if any(keyword in question for keyword in self.INTENT_KEYWORDS["statistics_query"]):
            return "statistics_query"
        
        # 检查趋势意图
        if any(keyword in question for keyword in self.INTENT_KEYWORDS["trend_query"]):
            return "trend_query"
        
        # 检查车站意图
        if any(keyword in question for keyword in self.INTENT_KEYWORDS["station_flow_query"]):
            return "station_flow_query"
        
        # 默认线路意图
        if any(keyword in question for keyword in self.INTENT_KEYWORDS["line_flow_query"]):
            return "line_flow_query"
        
        # 默认返回线路查询
        return "line_flow_query"

