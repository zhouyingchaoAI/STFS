"""
查询执行器
"""
from typing import List, Dict, Any
import re

from app.core.query_executor.db_manager import db_manager
from app.core.query_executor.sql_validator import SQLValidator
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryExecutor:
    """查询执行器"""
    
    def __init__(self):
        self.validator = SQLValidator()
    
    def execute(self, sql: str, database: str = "master", max_rows: int = None) -> Dict[str, Any]:
        """
        执行SQL查询
        
        Args:
            sql: SQL查询语句
            database: 数据库名称
            max_rows: 最大返回行数
            
        Returns:
            查询结果字典
        """
        # 1. SQL验证
        if not self.validator.validate(sql):
            raise ValueError("SQL validation failed: dangerous operations detected")
        
        # 2. 设置最大行数
        if max_rows is None:
            max_rows = settings.MAX_QUERY_ROWS
        
        # 3. 执行查询
        try:
            results = db_manager.execute_query(sql, database, max_rows)
            
            return {
                "data": results,
                "row_count": len(results),
                "sql": sql
            }
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

