"""
混合NL2SQL引擎（规则引擎 + LLM引擎）
"""
from typing import Dict, Optional, List, Any

from app.config import settings
from app.core.nl2sql.rule_based_engine import RuleBasedNL2SQLEngine
from app.core.nl2sql.llm_based_engine import LLMBasedNL2SQLEngine
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridNL2SQLEngine:
    """混合NL2SQL引擎
    
    策略：
    1. 优先使用规则引擎（快速、可控）
    2. 如果规则引擎失败且LLM启用，则使用LLM引擎
    3. 对于复杂查询，可以直接使用LLM引擎
    """
    
    def __init__(self):
        self.rule_engine = RuleBasedNL2SQLEngine()
        self.llm_engine = None
        
        if settings.LLM_ENABLED:
            try:
                self.llm_engine = LLMBasedNL2SQLEngine()
                logger.info("LLM engine initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM engine: {e}")
    
    def convert(self, question: str, use_llm: bool = False, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict]:
        """
        将自然语言问题转换为SQL
        
        Args:
            question: 自然语言问题
            use_llm: 是否强制使用LLM引擎
            conversation_history: 对话历史（用于多轮对话修正）
            
        Returns:
            包含SQL、意图、实体等信息的字典
        """
        # 如果强制使用LLM或规则引擎不可用
        if use_llm and self.llm_engine:
            logger.info("Using LLM engine (forced)")
            result = self.llm_engine.convert(question, conversation_history=conversation_history)
            if result:
                result["engine_type"] = "llm"
                result["thinking_process"] = "使用大语言模型进行意图理解和SQL生成"
            return result
        
        # 优先尝试规则引擎
        result = self.rule_engine.convert(question)
        if result:
            result["engine_type"] = "rule"
            result["thinking_process"] = "使用规则引擎进行意图识别和SQL生成"
        
        # 如果规则引擎失败且LLM可用，则尝试LLM
        if not result and self.llm_engine:
            logger.info("Rule engine failed, trying LLM engine")
            result = self.llm_engine.convert(question, conversation_history=conversation_history)
            if result:
                result["engine_type"] = "llm"
                result["thinking_process"] = "规则引擎无法处理，使用大语言模型进行转换"
        
        # 如果规则引擎成功，但检测到复杂查询，可以尝试LLM优化
        if result and self._is_complex_query(question) and self.llm_engine:
            logger.info("Complex query detected, trying LLM for better results")
            llm_result = self.llm_engine.convert(question, conversation_history=conversation_history)
            if llm_result and llm_result.get("sql"):
                llm_result["engine_type"] = "llm"
                llm_result["thinking_process"] = "检测到复杂查询，使用大语言模型优化SQL生成"
                return llm_result
        
        return result
    
    def _is_complex_query(self, question: str) -> bool:
        """判断是否为复杂查询"""
        complex_keywords = [
            "对比", "比较", "排名", "统计", "趋势", "分析",
            "多个", "所有", "平均", "总和", "最大", "最小"
        ]
        return any(kw in question for kw in complex_keywords)

