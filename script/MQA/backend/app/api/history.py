"""
查询历史API
"""
from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime

from app.models.query import QueryHistoryItem
from app.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.get("/history")
async def get_query_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    获取查询历史
    """
    # TODO: 实现查询历史存储和检索
    # 这里暂时返回空列表
    return {
        "code": 200,
        "message": "success",
        "data": {
            "items": [],
            "total": 0,
            "page": page,
            "page_size": page_size
        }
    }

