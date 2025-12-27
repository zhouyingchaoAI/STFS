"""
FastAPI应用主入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.config import settings
from app.api import query, sql, history, metadata
from app.utils.logger import setup_logger

# 初始化日志
logger = setup_logger()

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="地铁客流智能问数系统API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 注册路由
app.include_router(query.router, prefix=settings.API_V1_PREFIX, tags=["查询"])
app.include_router(sql.router, prefix=settings.API_V1_PREFIX, tags=["SQL"])
app.include_router(history.router, prefix=settings.API_V1_PREFIX, tags=["历史"])
app.include_router(metadata.router, prefix=settings.API_V1_PREFIX, tags=["元数据"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": "服务器内部错误",
            "data": None
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=4577,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

