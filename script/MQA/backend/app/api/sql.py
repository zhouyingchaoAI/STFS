"""
SQL直接查询API
"""
from fastapi import APIRouter, HTTPException
from app.models.query import SQLQueryRequest, QueryResponse
from app.core.query_executor.query_executor import QueryExecutor
from app.core.result_processor.formatter import ResultFormatter
from app.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

query_executor = QueryExecutor()
result_formatter = ResultFormatter()


@router.post("/sql", response_model=QueryResponse)
async def sql_query(request: SQLQueryRequest):
    """
    SQL直接查询接口
    """
    try:
        database = request.database or "master"
        
        query_result = query_executor.execute(
            sql=request.sql,
            database=database
        )
        
        formatted_result = result_formatter.format(query_result)
        
        return QueryResponse(
            code=200,
            message="success",
            data={
                "sql": request.sql,
                "result": formatted_result,
                "row_count": len(formatted_result)
            }
        )
    except Exception as e:
        logger.error(f"SQL query error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": 500,
                "message": f"SQL查询失败: {str(e)}",
                "data": None
            }
        )

