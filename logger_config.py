# 统一日志配置模块
"""
该模块提供项目统一的日志配置和管理，特点：
- 统一的日志格式
- 支持控制台和文件输出
- 日志轮转
- 按模块获取 logger
"""

import os
import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional
from datetime import datetime

# =============================================================================
# 常量配置
# =============================================================================

# 日志目录
LOG_DIR = "logs"

# 默认日志格式
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
SIMPLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DETAILED_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d - %(funcName)s): %(message)s"

# 日期格式
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 日志级别映射
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

# 默认配置
DEFAULT_CONFIG = {
    'level': 'INFO',
    'console_output': True,
    'file_output': True,
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
    'format': DEFAULT_FORMAT
}

# =============================================================================
# 全局变量
# =============================================================================

_initialized = False
_loggers = {}


# =============================================================================
# 初始化函数
# =============================================================================

def init_logging(
    level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    log_dir: str = LOG_DIR,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    format_str: str = DEFAULT_FORMAT
) -> None:
    """
    初始化全局日志配置
    
    参数:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        log_dir: 日志文件目录
        log_file: 日志文件名（默认按日期生成）
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的日志文件数量
        format_str: 日志格式字符串
    """
    global _initialized
    
    if _initialized:
        return
    
    # 创建日志目录
    if file_output and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
    
    # 清除现有 handlers
    root_logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(format_str, datefmt=DATE_FORMAT)
    
    # 控制台 handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 文件 handler（带轮转）
    if file_output:
        if log_file is None:
            log_file = f"app_{datetime.now().strftime('%Y%m%d')}.log"
        
        log_path = os.path.join(log_dir, log_file)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    _initialized = True
    root_logger.info(f"日志系统初始化完成 - 级别: {level}, 控制台: {console_output}, 文件: {file_output}")


def get_logger(name: str = None, level: str = None) -> logging.Logger:
    """
    获取指定名称的 logger
    
    参数:
        name: logger 名称（通常为模块名，如 __name__）
        level: 可选的日志级别覆盖
        
    返回:
        Logger 实例
    """
    global _initialized
    
    # 确保日志系统已初始化
    if not _initialized:
        init_logging()
    
    # 获取或创建 logger
    logger = logging.getLogger(name)
    
    # 如果指定了级别，设置该 logger 的级别
    if level:
        logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
    
    # 缓存 logger
    if name:
        _loggers[name] = logger
    
    return logger


# =============================================================================
# 模块专用 Logger 工厂
# =============================================================================

class ModuleLogger:
    """
    模块专用 Logger 包装类
    
    提供更方便的日志记录方法，支持自动添加上下文信息
    """
    
    def __init__(self, name: str, level: str = None):
        """
        初始化模块 Logger
        
        参数:
            name: 模块名称
            level: 日志级别
        """
        self._logger = get_logger(name, level)
        self._name = name
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """记录 DEBUG 级别日志"""
        self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """记录 INFO 级别日志"""
        self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """记录 WARNING 级别日志"""
        self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, exc_info: bool = False, **kwargs) -> None:
        """记录 ERROR 级别日志"""
        self._logger.error(msg, *args, exc_info=exc_info, **kwargs)
    
    def critical(self, msg: str, *args, exc_info: bool = True, **kwargs) -> None:
        """记录 CRITICAL 级别日志"""
        self._logger.critical(msg, *args, exc_info=exc_info, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
        """记录异常信息（自动包含堆栈追踪）"""
        self._logger.exception(msg, *args, **kwargs)
    
    def log_function_call(self, func_name: str, **params) -> None:
        """记录函数调用信息"""
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        self._logger.debug(f"调用 {func_name}({param_str})")
    
    def log_function_result(self, func_name: str, result: any = None, duration: float = None) -> None:
        """记录函数执行结果"""
        msg = f"{func_name} 执行完成"
        if duration is not None:
            msg += f" (耗时: {duration:.3f}s)"
        if result is not None:
            msg += f" - 结果: {result}"
        self._logger.debug(msg)
    
    def log_data_stats(self, name: str, data, **extra) -> None:
        """记录数据统计信息"""
        import pandas as pd
        
        if isinstance(data, pd.DataFrame):
            msg = f"{name}: {len(data)} 行, {len(data.columns)} 列"
            if not data.empty:
                msg += f", 列: {list(data.columns)[:5]}..."
        elif isinstance(data, (list, tuple)):
            msg = f"{name}: {len(data)} 项"
        elif isinstance(data, dict):
            msg = f"{name}: {len(data)} 键"
        else:
            msg = f"{name}: {type(data).__name__}"
        
        for k, v in extra.items():
            msg += f", {k}={v}"
        
        self._logger.info(msg)


# =============================================================================
# 预定义的模块 Logger
# =============================================================================

def get_db_logger() -> ModuleLogger:
    """获取数据库模块 Logger"""
    return ModuleLogger("db_utils")


def get_predict_logger() -> ModuleLogger:
    """获取预测模块 Logger"""
    return ModuleLogger("predict")


def get_model_logger() -> ModuleLogger:
    """获取模型模块 Logger"""
    return ModuleLogger("model")


def get_api_logger() -> ModuleLogger:
    """获取 API 模块 Logger"""
    return ModuleLogger("api")


def get_task_logger() -> ModuleLogger:
    """获取任务模块 Logger"""
    return ModuleLogger("task")


# =============================================================================
# 上下文管理器
# =============================================================================

class LogContext:
    """
    日志上下文管理器
    
    用于在特定代码块中添加上下文信息
    """
    
    def __init__(self, logger: ModuleLogger, context_name: str, **context_data):
        """
        初始化日志上下文
        
        参数:
            logger: Logger 实例
            context_name: 上下文名称
            context_data: 上下文数据
        """
        self._logger = logger
        self._context_name = context_name
        self._context_data = context_data
        self._start_time = None
    
    def __enter__(self):
        """进入上下文"""
        import time
        self._start_time = time.time()
        
        data_str = ", ".join(f"{k}={v}" for k, v in self._context_data.items())
        if data_str:
            self._logger.info(f"[开始] {self._context_name} - {data_str}")
        else:
            self._logger.info(f"[开始] {self._context_name}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        import time
        duration = time.time() - self._start_time
        
        if exc_type is not None:
            self._logger.error(
                f"[失败] {self._context_name} - 耗时: {duration:.3f}s, 错误: {exc_val}",
                exc_info=True
            )
        else:
            self._logger.info(f"[完成] {self._context_name} - 耗时: {duration:.3f}s")
        
        return False  # 不抑制异常


# =============================================================================
# 装饰器
# =============================================================================

def log_execution(logger: ModuleLogger = None, log_args: bool = True, log_result: bool = False):
    """
    函数执行日志装饰器
    
    参数:
        logger: Logger 实例（默认使用函数模块的 logger）
        log_args: 是否记录函数参数
        log_result: 是否记录返回结果
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = ModuleLogger(func.__module__)
            
            # 记录函数调用
            if log_args:
                args_repr = [repr(a) for a in args[:3]]  # 最多显示3个位置参数
                kwargs_repr = [f"{k}={v!r}" for k, v in list(kwargs.items())[:3]]
                signature = ", ".join(args_repr + kwargs_repr)
                if len(args) > 3 or len(kwargs) > 3:
                    signature += ", ..."
                logger.debug(f"调用 {func.__name__}({signature})")
            else:
                logger.debug(f"调用 {func.__name__}()")
            
            # 执行函数
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if log_result:
                    result_repr = repr(result)[:100]  # 限制结果长度
                    logger.debug(f"{func.__name__} 完成 ({duration:.3f}s) -> {result_repr}")
                else:
                    logger.debug(f"{func.__name__} 完成 ({duration:.3f}s)")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} 失败 ({duration:.3f}s): {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


# =============================================================================
# 初始化（模块加载时）
# =============================================================================

# 默认初始化日志系统
init_logging()
