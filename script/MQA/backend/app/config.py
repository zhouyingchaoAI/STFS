"""
系统配置文件
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用配置
    APP_NAME: str = "地铁客流智能问数系统"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API配置
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["*"]
    
    # 数据库配置 - master
    DB_MASTER_HOST: str = "10.1.6.230"
    DB_MASTER_PORT: int = 1433
    DB_MASTER_USER: str = "sa"
    DB_MASTER_PASSWORD: str = "YourStrong!Passw0rd"
    DB_MASTER_DATABASE: str = "master"
    
    # 数据库配置 - CxFlowPredict
    DB_PREDICT_HOST: str = "10.1.6.230"
    DB_PREDICT_PORT: int = 1433
    DB_PREDICT_USER: str = "sa"
    DB_PREDICT_PASSWORD: str = "YourStrong!Passw0rd"
    DB_PREDICT_DATABASE: str = "CxFlowPredict"
    
    # 连接池配置
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    
    # Redis配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_CACHE_TTL: int = 3600  # 缓存过期时间（秒）
    
    # LLM配置 (可选)
    LLM_ENABLED: bool = True
    LLM_PROVIDER: str = "ollama"  # ollama, openai, anthropic, baidu, alibaba
    LLM_API_KEY: Optional[str] = None
    LLM_API_BASE: Optional[str] = "http://10.1.6.230:11434"  # Ollama API地址
    LLM_MODEL: str = "deepseek-r1:70b"  # Ollama模型名称
    LLM_MAX_TOKENS: int = 4096  # 最大生成token数（num_predict）
    LLM_CONTEXT_LENGTH: int = 65536  # 上下文窗口大小（num_ctx），64K（适配约25K字符的prompt）
    LLM_TEMPERATURE: float = 0.1
    
    # 查询配置
    MAX_QUERY_ROWS: int = 50000  # 最大返回行数
    QUERY_TIMEOUT: int = 300  # 查询超时时间（秒）
    ENABLE_QUERY_CACHE: bool = True
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # 安全配置
    API_KEY: Optional[str] = None
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 全局配置实例
settings = Settings()

