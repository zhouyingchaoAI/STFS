"""
查询相关的数据模型
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., description="自然语言查询问题")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="查询选项")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default=None, description="对话历史（用于多轮对话修正）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "查询1号线昨天的客流量",
                "options": {
                    "use_cache": True,
                    "format": "json",
                    "max_rows": 1000
                },
                "conversation_history": None
            }
        }


class SQLQueryRequest(BaseModel):
    """SQL查询请求模型"""
    sql: str = Field(..., description="SQL查询语句")
    database: Optional[str] = Field(default="master", description="数据库名称")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sql": "SELECT * FROM LineDailyFlowHistory WHERE f_date = 20240101",
                "database": "master"
            }
        }


class QueryResponse(BaseModel):
    """查询响应模型"""
    code: int = Field(200, description="响应码")
    message: str = Field("success", description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "sql": "SELECT f_date, f_linename, f_klcount FROM ...",
                    "result": [
                        {
                            "日期": "20240101",
                            "线路名": "1号线",
                            "客流量": 123456.78
                        }
                    ],
                    "chart_config": {
                        "type": "line",
                        "data": {}
                    },
                    "execution_time": 0.123
                },
                "metadata": {
                    "intent": "line_flow_query",
                    "entities": {
                        "line": "1号线",
                        "date": "2024-01-01"
                    }
                }
            }
        }


class QueryHistoryItem(BaseModel):
    """查询历史项"""
    id: str
    question: str
    sql: Optional[str] = None
    intent: Optional[str] = None
    result_count: Optional[int] = None
    execution_time: Optional[float] = None
    status: str
    created_at: datetime

