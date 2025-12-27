"""
基于规则的NL2SQL引擎
"""
from typing import Dict, Optional, List
import re
import jieba

from app.core.nl2sql.intent_classifier import IntentClassifier
from app.core.nl2sql.entity_extractor import EntityExtractor
from app.core.nl2sql.sql_generator import SQLGenerator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class RuleBasedNL2SQLEngine:
    """基于规则的NL2SQL引擎"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.sql_generator = SQLGenerator()
    
    def convert(self, question: str) -> Optional[Dict]:
        """
        将自然语言问题转换为SQL
        
        Args:
            question: 自然语言问题
            
        Returns:
            包含SQL、意图、实体等信息的字典
        """
        try:
            # 1. 意图分类
            intent = self.intent_classifier.classify(question)
            logger.debug(f"Classified intent: {intent}")
            
            # 2. 实体抽取
            entities = self.entity_extractor.extract(question)
            logger.debug(f"Extracted entities: {entities}")
            
            # 3. SQL生成
            sql = self.sql_generator.generate(intent, entities)
            logger.debug(f"Generated SQL: {sql}")
            
            if not sql:
                return None
            
            return {
                "sql": sql,
                "intent": intent,
                "entities": entities,
                "thinking_process": self._generate_thinking_process(question, intent, entities)
            }
            
        except Exception as e:
            logger.error(f"NL2SQL conversion error: {e}", exc_info=True)
            return None
    
    def _generate_thinking_process(self, question: str, intent: str, entities: Dict) -> str:
        """生成思考过程描述"""
        thinking = []
        thinking.append(f"分析问题: {question}")
        
        # 意图分析
        intent_map = {
            "line_flow_query": "线路客流查询",
            "station_flow_query": "车站客流查询",
            "prediction_query": "预测查询",
            "comparison_query": "对比查询",
            "statistics_query": "统计查询",
            "trend_query": "趋势查询"
        }
        thinking.append(f"识别意图: {intent_map.get(intent, intent)}")
        
        # 实体分析
        if entities.get("line"):
            thinking.append(f"提取线路: {entities['line']}")
        if entities.get("station"):
            thinking.append(f"提取车站: {entities['station']}")
        if entities.get("date_int"):
            thinking.append(f"提取日期: {entities['date_int']}")
        if entities.get("metric"):
            thinking.append(f"提取指标: {entities['metric']}")
        
        thinking.append("使用规则引擎匹配查询模板")
        thinking.append("生成对应的SQL查询语句")
        
        return "\n".join(thinking)

