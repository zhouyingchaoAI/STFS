# 数据库连接池模块
"""
该模块提供数据库连接池管理，特点：
- 连接池复用
- 上下文管理器支持
- 自动重连
- 线程安全
"""

import pymssql
import threading
import queue
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass
import os
import yaml

from logger_config import get_db_logger

logger = get_db_logger()


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class DBConfig:
    """数据库配置"""
    server: str
    user: str
    password: str
    database: str = "master"
    port: int = 1433
    charset: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, config_path: str = "db_config.yaml") -> "DBConfig":
        """从 YAML 文件加载配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        db_config = config.get("db", {})
        return cls(
            server=db_config.get("server", "localhost"),
            user=db_config.get("user", "sa"),
            password=db_config.get("password", ""),
            database=db_config.get("database", "master"),
            port=db_config.get("port", 1433),
            charset=db_config.get("charset")
        )


# =============================================================================
# 连接池类
# =============================================================================

class ConnectionPool:
    """
    数据库连接池
    
    特点：
    - 支持连接复用
    - 自动检测失效连接
    - 线程安全
    """
    
    def __init__(
        self,
        config: DBConfig,
        min_connections: int = 2,
        max_connections: int = 10,
        connection_timeout: int = 30,
        idle_timeout: int = 300
    ):
        """
        初始化连接池
        
        参数:
            config: 数据库配置
            min_connections: 最小连接数
            max_connections: 最大连接数
            connection_timeout: 获取连接的超时时间（秒）
            idle_timeout: 连接空闲超时时间（秒）
        """
        self._config = config
        self._min_connections = min_connections
        self._max_connections = max_connections
        self._connection_timeout = connection_timeout
        self._idle_timeout = idle_timeout
        
        self._pool: queue.Queue = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._created_connections = 0
        self._active_connections = 0
        
        # 初始化最小连接数
        self._initialize_pool()
        
        logger.info(f"连接池初始化完成 - 最小连接: {min_connections}, 最大连接: {max_connections}")
    
    def _initialize_pool(self) -> None:
        """初始化连接池"""
        for _ in range(self._min_connections):
            try:
                conn = self._create_connection()
                if conn:
                    self._pool.put((conn, time.time()))
            except Exception as e:
                logger.warning(f"初始化连接失败: {e}")
    
    def _create_connection(self, charset: Optional[str] = None) -> Optional[pymssql.Connection]:
        """
        创建新连接
        
        参数:
            charset: 字符集（None 表示使用默认配置）
            
        返回:
            数据库连接
        """
        with self._lock:
            if self._created_connections >= self._max_connections:
                return None
        
        conn_params = {
            "server": self._config.server,
            "user": self._config.user,
            "password": self._config.password,
            "database": self._config.database,
            "port": self._config.port
        }
        
        # 处理字符集
        effective_charset = charset if charset is not None else self._config.charset
        if effective_charset and effective_charset != "utf8":
            conn_params["charset"] = effective_charset
        
        try:
            conn = pymssql.connect(**conn_params)
            with self._lock:
                self._created_connections += 1
            logger.debug(f"创建新连接，当前总连接数: {self._created_connections}")
            return conn
        except Exception as e:
            logger.error(f"创建数据库连接失败: {e}")
            raise
    
    def _validate_connection(self, conn: pymssql.Connection) -> bool:
        """
        验证连接是否有效
        
        参数:
            conn: 数据库连接
            
        返回:
            是否有效
        """
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception:
            return False
    
    def get_connection(self, charset: Optional[str] = None) -> pymssql.Connection:
        """
        获取连接
        
        参数:
            charset: 字符集
            
        返回:
            数据库连接
            
        注意:
            使用完毕后必须调用 release_connection 归还连接
        """
        start_time = time.time()
        
        while True:
            # 检查超时
            if time.time() - start_time > self._connection_timeout:
                raise TimeoutError(f"获取数据库连接超时 ({self._connection_timeout}s)")
            
            try:
                # 尝试从池中获取
                conn, created_time = self._pool.get_nowait()
                
                # 检查是否过期
                if time.time() - created_time > self._idle_timeout:
                    logger.debug("连接已过期，关闭并创建新连接")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    with self._lock:
                        self._created_connections -= 1
                    continue
                
                # 验证连接有效性
                if not self._validate_connection(conn):
                    logger.debug("连接已失效，关闭并创建新连接")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    with self._lock:
                        self._created_connections -= 1
                    continue
                
                with self._lock:
                    self._active_connections += 1
                
                logger.debug(f"从池中获取连接，活跃连接数: {self._active_connections}")
                return conn
                
            except queue.Empty:
                # 池为空，尝试创建新连接
                conn = self._create_connection(charset)
                if conn:
                    with self._lock:
                        self._active_connections += 1
                    return conn
                
                # 无法创建新连接，等待一会儿再试
                time.sleep(0.1)
    
    def release_connection(self, conn: pymssql.Connection) -> None:
        """
        归还连接到池中
        
        参数:
            conn: 数据库连接
        """
        if conn is None:
            return
        
        with self._lock:
            self._active_connections -= 1
        
        try:
            # 验证连接是否还有效
            if self._validate_connection(conn):
                self._pool.put((conn, time.time()))
                logger.debug(f"连接归还到池，活跃连接数: {self._active_connections}")
            else:
                # 连接已失效，关闭并减少计数
                conn.close()
                with self._lock:
                    self._created_connections -= 1
                logger.debug("连接已失效，已关闭")
        except Exception as e:
            logger.warning(f"归还连接时出错: {e}")
            with self._lock:
                self._created_connections -= 1
    
    def close_all(self) -> None:
        """关闭所有连接"""
        logger.info("关闭所有数据库连接...")
        
        while True:
            try:
                conn, _ = self._pool.get_nowait()
                try:
                    conn.close()
                except Exception:
                    pass
            except queue.Empty:
                break
        
        with self._lock:
            self._created_connections = 0
            self._active_connections = 0
        
        logger.info("所有连接已关闭")
    
    @property
    def stats(self) -> Dict[str, int]:
        """获取连接池统计信息"""
        return {
            "created": self._created_connections,
            "active": self._active_connections,
            "available": self._pool.qsize(),
            "max": self._max_connections
        }


# =============================================================================
# 全局连接池实例
# =============================================================================

_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """获取全局连接池实例"""
    global _pool
    
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                config = DBConfig.from_yaml()
                _pool = ConnectionPool(config)
    
    return _pool


def close_pool() -> None:
    """关闭全局连接池"""
    global _pool
    
    with _pool_lock:
        if _pool is not None:
            _pool.close_all()
            _pool = None


# =============================================================================
# 上下文管理器
# =============================================================================

@contextmanager
def get_db_connection(charset: Optional[str] = None):
    """
    获取数据库连接的上下文管理器
    
    使用方式:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(...)
    
    参数:
        charset: 字符集
        
    返回:
        数据库连接
    """
    pool = get_pool()
    conn = None
    
    try:
        conn = pool.get_connection(charset)
        yield conn
    finally:
        if conn is not None:
            pool.release_connection(conn)


@contextmanager
def get_db_cursor(charset: Optional[str] = None, commit: bool = True):
    """
    获取数据库游标的上下文管理器
    
    使用方式:
        with get_db_cursor() as cursor:
            cursor.execute(...)
    
    参数:
        charset: 字符集
        commit: 是否自动提交（默认为 True）
        
    返回:
        数据库游标
    """
    with get_db_connection(charset) as conn:
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


@contextmanager
def get_db_dict_cursor(charset: Optional[str] = None, commit: bool = True):
    """
    获取字典游标的上下文管理器（返回字典而非元组）
    
    使用方式:
        with get_db_dict_cursor() as cursor:
            cursor.execute(...)
            for row in cursor:
                print(row['column_name'])
    
    参数:
        charset: 字符集
        commit: 是否自动提交
        
    返回:
        字典游标
    """
    with get_db_connection(charset) as conn:
        cursor = conn.cursor(as_dict=True)
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()


# =============================================================================
# 便捷函数
# =============================================================================

def execute_query(sql: str, params: tuple = None, charset: str = 'utf8') -> list:
    """
    执行查询并返回结果
    
    参数:
        sql: SQL 查询语句
        params: 参数
        charset: 字符集
        
    返回:
        查询结果列表
    """
    with get_db_cursor(charset, commit=False) as cursor:
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        return cursor.fetchall()


def execute_non_query(sql: str, params: tuple = None, charset: Optional[str] = None) -> int:
    """
    执行非查询语句（INSERT, UPDATE, DELETE）
    
    参数:
        sql: SQL 语句
        params: 参数
        charset: 字符集
        
    返回:
        受影响的行数
    """
    with get_db_cursor(charset, commit=True) as cursor:
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        return cursor.rowcount


def execute_many(sql: str, params_list: list, charset: Optional[str] = None) -> int:
    """
    批量执行语句
    
    参数:
        sql: SQL 语句
        params_list: 参数列表
        charset: 字符集
        
    返回:
        受影响的总行数
    """
    total_rows = 0
    with get_db_cursor(charset, commit=True) as cursor:
        for params in params_list:
            cursor.execute(sql, params)
            total_rows += cursor.rowcount
    return total_rows


# =============================================================================
# 用于兼容旧代码的函数
# =============================================================================

def get_db_conn(charset: Optional[str] = None) -> pymssql.Connection:
    """
    获取数据库连接（兼容旧代码）
    
    警告: 此函数返回的连接需要手动管理，建议使用 get_db_connection() 上下文管理器
    
    参数:
        charset: 字符集
        
    返回:
        数据库连接
    """
    pool = get_pool()
    return pool.get_connection(charset)


def release_conn(conn: pymssql.Connection) -> None:
    """
    释放数据库连接（兼容旧代码）
    
    参数:
        conn: 数据库连接
    """
    pool = get_pool()
    pool.release_connection(conn)
