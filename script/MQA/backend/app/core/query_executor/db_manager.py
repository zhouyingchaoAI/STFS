"""
数据库连接管理（参考 get_data_struct.py）
统一处理数据库连接和编码问题
"""
import pymssql
from typing import Optional, Dict, List, Any
from contextlib import contextmanager

from app.config import settings
from app.utils.logger import setup_logger
from app.utils.encoding import fix_query_results

logger = setup_logger(__name__)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.connections = {}
        self._init_connections()
    
    def _init_connections(self):
        """初始化数据库连接配置"""
        self.db_configs = {
            "master": {
                "server": settings.DB_MASTER_HOST,
                "port": settings.DB_MASTER_PORT,
                "user": settings.DB_MASTER_USER,
                "password": settings.DB_MASTER_PASSWORD,
                "database": settings.DB_MASTER_DATABASE
            },
            "CxFlowPredict": {
                "server": settings.DB_PREDICT_HOST,
                "port": settings.DB_PREDICT_PORT,
                "user": settings.DB_PREDICT_USER,
                "password": settings.DB_PREDICT_PASSWORD,
                "database": settings.DB_PREDICT_DATABASE
            }
        }
    
    @contextmanager
    def get_connection(self, database: str = "master"):
        """
        获取数据库连接（上下文管理器）
        
        Args:
            database: 数据库名称
            
        Yields:
            数据库连接对象
        """
        if database not in self.db_configs:
            raise ValueError(f"Unknown database: {database}")
        
        config = self.db_configs[database]
        conn = None
        
        try:
            # 参考 get_data_struct.py：不指定 charset，让 pymssql 使用默认设置
            # 编码问题在读取数据时通过 smart_encode_fix 处理
            conn = pymssql.connect(
                server=config["server"],
                port=config["port"],
                user=config["user"],
                password=config["password"],
                database=config["database"],
                timeout=settings.QUERY_TIMEOUT
            )
            logger.debug(f"Database connection established: {database}")
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
                logger.debug(f"Database connection closed: {database}")
    
    def execute_query(self, sql: str, database: str = "master", max_rows: int = None) -> list:
        """
        执行查询并返回结果（自动修复编码问题）
        
        Args:
            sql: SQL查询语句
            database: 数据库名称
            max_rows: 最大返回行数
            
        Returns:
            查询结果列表（已修复编码）
        """
        with self.get_connection(database) as conn:
            cursor = conn.cursor(as_dict=True)
            try:
                cursor.execute(sql)
                
                if max_rows:
                    results = cursor.fetchmany(max_rows)
                else:
                    results = cursor.fetchall()
                
                # 修复查询结果中的编码问题（参考 get_data_struct.py）
                fixed_results = fix_query_results(results)
                
                return fixed_results
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                logger.error(f"SQL: {sql}")
                raise
            finally:
                cursor.close()


# 全局数据库管理器实例
db_manager = DatabaseManager()

