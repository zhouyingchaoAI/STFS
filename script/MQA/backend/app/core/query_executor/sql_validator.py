"""
SQL验证器
"""
import re
from typing import List

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SQLValidator:
    """SQL验证器"""
    
    # 危险SQL关键字（禁止执行）
    DANGEROUS_KEYWORDS = [
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
        "CREATE", "TRUNCATE", "EXEC", "EXECUTE", "SP_",
        "XP_", "GRANT", "REVOKE", "SHUTDOWN"
    ]
    
    # 允许的SQL关键字（白名单）
    ALLOWED_KEYWORDS = [
        "SELECT", "FROM", "WHERE", "ORDER", "BY", "GROUP",
        "HAVING", "JOIN", "INNER", "LEFT", "RIGHT", "ON",
        "AS", "AND", "OR", "NOT", "IN", "LIKE", "BETWEEN",
        "IS", "NULL", "COUNT", "SUM", "AVG", "MAX", "MIN",
        "DISTINCT", "TOP", "LIMIT", "OFFSET", "CASE", "WHEN",
        "THEN", "ELSE", "END", "UNION", "ALL"
    ]
    
    def validate(self, sql: str) -> bool:
        """
        验证SQL语句的安全性
        
        Args:
            sql: SQL语句
            
        Returns:
            是否通过验证
        """
        sql_upper = sql.upper().strip()
        
        # 1. 检查危险关键字
        for keyword in self.DANGEROUS_KEYWORDS:
            # 使用单词边界匹配，避免误判
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, sql_upper):
                logger.warning(f"Dangerous keyword detected: {keyword}")
                return False
        
        # 2. 必须是SELECT语句
        if not sql_upper.startswith("SELECT"):
            logger.warning("Only SELECT statements are allowed")
            return False
        
        # 3. 检查注释注入（-- 和 /* */）
        if "--" in sql or "/*" in sql:
            # 允许注释，但需要进一步检查
            pass
        
        # 4. 检查分号后的额外语句
        if ";" in sql:
            parts = sql.split(";")
            if len(parts) > 2 or (len(parts) == 2 and parts[1].strip()):
                logger.warning("Multiple statements detected")
                return False
        
        return True
    
    def sanitize(self, sql: str) -> str:
        """
        清理SQL语句（移除注释等）
        
        Args:
            sql: SQL语句
            
        Returns:
            清理后的SQL语句
        """
        # 移除单行注释
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        
        # 移除多行注释
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # 移除多余空白
        sql = ' '.join(sql.split())
        
        return sql.strip()

