# Subway Forecast RESTful Service (FastAPI) - Extended Version with Flow Metric Types
# Compatible with Python 3.8 and Pydantic v1
# 支持多种客流类型：线网线路客流、断面客流、车站客流等
# 支持多种客流指标类型：客运量、进站量、出站量、换乘量、乘降量

import os
import uuid
import logging
import traceback
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator

# === 项目模块 ===
from config_utils import load_yaml_config, save_yaml_config
from predict_daily import predict_and_plot_timeseries_flow_daily
from predict_hourly import predict_and_plot_timeseries_flow
# TODO: 导入其他客流类型的预测模块
# from predict_section import predict_and_plot_section_flow_daily, predict_and_plot_section_flow
# from predict_station import predict_and_plot_station_flow_daily, predict_and_plot_station_flow

# ----------------------------------------------------------------------------
# 常量与初始化
# ----------------------------------------------------------------------------

# 客流指标类型配置
FLOW_METRIC_TYPES = [
    ("F_PKLCOUNT", "客运量"),
    ("F_ENTRANCE", "进站量"),
    ("F_EXIT", "出站量"),
    ("F_TRANSFER", "换乘量"),
    ("F_BOARD_ALIGHT", "乘降量")
]

# 转换为字典格式便于查询
FLOW_METRIC_DICT = {code: name for code, name in FLOW_METRIC_TYPES}

# 客流类型配置
FLOW_TYPES = {
    "xianwangxianlu": {
        "name": "线网线路客流",
        "description": "整个线网的线路客流量预测",
        "daily_algos": ["knn", "prophet", "transformer", "xgboost", "lstm", "lightgbm"],
        "hourly_algos": ["knn", "lstm", "prophet", "xgboost"],
        "config_prefix": "xianwangxianlu",
        "supported_metrics": ["F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER", "F_BOARD_ALIGHT"]
    },
    "duanmian": {
        "name": "断面客流",
        "description": "地铁线路断面客流量预测",
        "daily_algos": ["knn", "prophet", "transformer", "xgboost", "lstm", "lightgbm"],
        "hourly_algos": ["knn", "lstm", "prophet", "xgboost"],
        "config_prefix": "duanmian",
        "supported_metrics": ["F_PKLCOUNT", "F_BOARD_ALIGHT"]
    },
    "chezhan": {
        "name": "车站客流",
        "description": "单个车站客流量预测",
        "daily_algos": ["knn", "prophet", "transformer", "xgboost", "lstm", "lightgbm"],
        "hourly_algos": ["knn", "lstm", "prophet", "xgboost"],
        "config_prefix": "chezhan",
        "supported_metrics": ["F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER", "F_BOARD_ALIGHT"]
    },
    "huhuan": {
        "name": "换乘客流",
        "description": "换乘站客流量预测",
        "daily_algos": ["knn", "prophet", "transformer", "xgboost", "lstm"],
        "hourly_algos": ["knn", "lstm", "prophet", "xgboost"],
        "config_prefix": "huhuan",
        "supported_metrics": ["F_TRANSFER", "F_ENTRANCE", "F_EXIT"]
    },
    "quyuxing": {
        "name": "区域客流",
        "description": "区域性客流量预测",
        "daily_algos": ["knn", "prophet", "xgboost", "lstm"],
        "hourly_algos": ["knn", "lstm", "prophet"],
        "config_prefix": "quyuxing",
        "supported_metrics": ["F_PKLCOUNT", "F_ENTRANCE", "F_EXIT"]
    }
}

MODELS_ROOT = "models"
PLOTS_ROOT = "plots"

# 创建必要目录
for directory in [MODELS_ROOT, PLOTS_ROOT]:
    os.makedirs(directory, exist_ok=True)

# 为每种客流类型和指标类型创建模型目录
for flow_type in FLOW_TYPES.keys():
    flow_config = FLOW_TYPES[flow_type]
    for metric_code in flow_config["supported_metrics"]:
        for granularity in ["daily", "hourly"]:
            flow_model_dir = os.path.join(MODELS_ROOT, flow_type, granularity, metric_code)
            os.makedirs(flow_model_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("subway_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("subway_api")

# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def validate_flow_type(flow_type: str) -> str:
    """验证客流类型"""
    if flow_type not in FLOW_TYPES:
        valid_types = list(FLOW_TYPES.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid flow_type '{flow_type}', must be one of {valid_types}"
        )
    return flow_type

def validate_flow_metric_type(flow_metric_type: str) -> str:
    """验证客流指标类型"""
    if flow_metric_type not in FLOW_METRIC_DICT:
        valid_metrics = list(FLOW_METRIC_DICT.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid flow_metric_type '{flow_metric_type}', must be one of {valid_metrics}"
        )
    return flow_metric_type

def validate_metric_for_flow_type(flow_metric_type: str, flow_type: str) -> str:
    """验证指标类型是否支持指定的客流类型"""
    flow_config = FLOW_TYPES.get(flow_type)
    if not flow_config:
        raise HTTPException(status_code=400, detail=f"Invalid flow_type: {flow_type}")
    
    supported_metrics = flow_config.get("supported_metrics", [])
    if flow_metric_type not in supported_metrics:
        raise HTTPException(
            status_code=400,
            detail=f"Metric '{flow_metric_type}' not supported for {flow_type}. "
                   f"Supported metrics: {supported_metrics}"
        )
    return flow_metric_type

def validate_granularity(granularity: str) -> str:
    """验证时间粒度"""
    if granularity not in ("daily", "hourly"):
        raise HTTPException(
            status_code=400, 
            detail="granularity must be 'daily' or 'hourly'"
        )
    return granularity

def validate_algorithm_for_flow_type(algorithm: str, flow_type: str, granularity: str) -> str:
    """验证算法是否支持指定的客流类型和时间粒度"""
    flow_config = FLOW_TYPES.get(flow_type)
    if not flow_config:
        raise HTTPException(status_code=400, detail=f"Invalid flow_type: {flow_type}")
    
    supported_algos = (flow_config["daily_algos"] if granularity == "daily" 
                      else flow_config["hourly_algos"])
    
    if algorithm not in supported_algos:
        raise HTTPException(
            status_code=400,
            detail=f"Algorithm '{algorithm}' not supported for {flow_type}/{granularity}. "
                   f"Supported algorithms: {supported_algos}"
        )
    return algorithm

def safe_json_encode(data: Any) -> Any:
    """安全的JSON编码，处理numpy类型和其他不可序列化对象"""
    if isinstance(data, dict):
        return {k: safe_json_encode(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [safe_json_encode(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    elif hasattr(data, '__dict__'):
        try:
            return {k: safe_json_encode(v) for k, v in data.__dict__.items() 
                   if not k.startswith('_')}
        except:
            return str(data)
    else:
        try:
            jsonable_encoder(data)
            return data
        except (TypeError, ValueError):
            return str(data)

def validate_date_format(date_str: str) -> str:
    """验证日期格式 YYYYMMDD"""
    if not date_str.isdigit() or len(date_str) != 8:
        raise HTTPException(
            status_code=422, 
            detail=f"Invalid date format '{date_str}', expected YYYYMMDD (e.g., 20250115)"
        )
    
    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError as e:
        raise HTTPException(
            status_code=422, 
            detail=f"Invalid date '{date_str}': {str(e)}"
        )
    
    return date_str

def get_model_versions(model_dir: str) -> List[str]:
    """获取模型版本列表"""
    if not os.path.exists(model_dir):
        return []
    
    try:
        dirs = [d for d in os.listdir(model_dir) 
                if os.path.isdir(os.path.join(model_dir, d))]
        versions = [d for d in dirs if len(d) == 8 and d.isdigit()]
        return sorted(versions, reverse=True)
    except Exception as e:
        logger.error(f"Error reading model versions from {model_dir}: {e}")
        return []

def generate_plot_path(flow_type: str, granularity: str, flow_metric_type: str) -> str:
    """生成唯一的图片路径"""
    filename = f"{flow_type}_{granularity}_{flow_metric_type}_{uuid.uuid4().hex}.png"
    return os.path.join(PLOTS_ROOT, filename)

def validate_model_exists(flow_type: str, granularity: str, flow_metric_type: str, version: str, algorithm: str) -> str:
    """验证模型是否存在"""
    model_dir = os.path.join(MODELS_ROOT, flow_type, granularity, flow_metric_type, version, algorithm)
    if not os.path.isdir(model_dir):
        raise HTTPException(
            status_code=404, 
            detail=f"Model not found: {flow_type}/{granularity}/{flow_metric_type}/{version}/{algorithm}"
        )
    return model_dir

def get_config_filename(flow_type: str, granularity: str, flow_metric_type: str) -> str:
    """获取配置文件名"""
    flow_config = FLOW_TYPES.get(flow_type, {})
    config_prefix = flow_config.get("config_prefix", flow_type)
    return f"model_config_{config_prefix}_{granularity}_{flow_metric_type}.yaml"

def get_prediction_function(flow_type: str, granularity: str):
    """根据客流类型和时间粒度获取预测函数"""
    if flow_type == "xianwangxianlu" or flow_type == "chezhan":
        if granularity == "daily":
            return predict_and_plot_timeseries_flow_daily
        else:
            return predict_and_plot_timeseries_flow
    elif flow_type == "duanmian":
        # TODO: 实现断面客流预测函数
        # if granularity == "daily":
        #     return predict_and_plot_section_flow_daily
        # else:
        #     return predict_and_plot_section_flow
        raise HTTPException(status_code=501, detail="断面客流预测功能正在开发中")
    elif flow_type == "huhuan":
        # TODO: 实现换乘客流预测函数
        raise HTTPException(status_code=501, detail="换乘客流预测功能正在开发中")
    elif flow_type == "quyuxing":
        # TODO: 实现区域客流预测函数
        raise HTTPException(status_code=501, detail="区域客流预测功能正在开发中")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported flow_type: {flow_type}")

# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI(
    title="Subway Forecast API - Extended with Metrics",
    version="2.1.0",
    description="地铁客流量预测API服务 - 支持多种客流类型和指标类型",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/plots", StaticFiles(directory=PLOTS_ROOT), name="plots")

# ----------------------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------------------
class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = True
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = False
    error_type: str
    error_details: Optional[str] = None

class FlowTypeInfo(BaseModel):
    """客流类型信息"""
    name: str
    description: str
    daily_algos: List[str]
    hourly_algos: List[str]
    supported_metrics: List[str]

class FlowMetricInfo(BaseModel):
    """客流指标信息"""
    code: str
    name: str

class TrainRequest(BaseModel):
    """训练请求基类"""
    algorithm: str
    train_end_date: str
    retrain: bool = False
    config_overrides: Optional[Dict[str, Any]] = None

    @validator("train_end_date")
    def validate_date(cls, v):
        return validate_date_format(v)

class PredictRequest(BaseModel):
    """预测请求基类"""
    algorithm: str
    model_version_date: str

    @validator("model_version_date")
    def validate_date(cls, v):
        return validate_date_format(v)

class PredictDailyRequest(PredictRequest):
    """每日预测请求"""
    predict_start_date: str
    days: int = Field(15, ge=1, le=30, description="预测天数，1-30天")

    @validator("predict_start_date")
    def validate_date(cls, v):
        return validate_date_format(v)

class PredictHourlyRequest(PredictRequest):
    """小时预测请求"""
    predict_date: str

    @validator("predict_date")
    def validate_date(cls, v):
        return validate_date_format(v)

class ConfigResponse(BaseModel):
    """配置响应"""
    flow_type: str
    granularity: str
    flow_metric_type: str
    config: Dict[str, Any]

class TrainingResponse(BaseResponse):
    """训练响应"""
    flow_type: str
    granularity: str
    flow_metric_type: str
    algorithm: str
    model_version_date: str
    model_dir: str
    plot_url: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseResponse):
    """预测响应"""
    flow_type: str
    granularity: str
    flow_metric_type: str
    algorithm: str
    model_version_date: str
    plot_url: Optional[str] = None
    predictions: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

# ----------------------------------------------------------------------------
# 异常处理
# ----------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.detail,
            error_type="HTTPException",
            error_details=str(exc)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.exception(f"Unhandled error in {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="Internal server error",
            error_type=type(exc).__name__,
            error_details=str(exc) if app.debug else None
        ).dict()
    )

# ----------------------------------------------------------------------------
# Meta endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0"
    }

@app.get("/flow-types")
def get_flow_types():
    """获取支持的客流类型列表"""
    return {
        "flow_types": {
            flow_type: FlowTypeInfo(**config).dict() 
            for flow_type, config in FLOW_TYPES.items()
        },
        "total_types": len(FLOW_TYPES)
    }

@app.get("/flow-metrics")
def get_flow_metrics():
    """获取支持的客流指标类型列表"""
    return {
        "flow_metrics": [
            FlowMetricInfo(code=code, name=name).dict() 
            for code, name in FLOW_METRIC_TYPES
        ],
        "total_metrics": len(FLOW_METRIC_TYPES)
    }

@app.get("/flow-types/{flow_type}/metrics")
def get_metrics_for_flow_type(flow_type: str):
    """获取指定客流类型支持的指标列表"""
    validate_flow_type(flow_type)
    config = FLOW_TYPES[flow_type]
    
    supported_metrics = config.get("supported_metrics", [])
    metric_details = [
        {"code": code, "name": FLOW_METRIC_DICT[code]} 
        for code in supported_metrics if code in FLOW_METRIC_DICT
    ]
    
    return {
        "flow_type": flow_type,
        "name": config["name"],
        "supported_metrics": metric_details,
        "total_metrics": len(metric_details)
    }

@app.get("/algorithms/{flow_type}")
def get_algorithms_for_flow_type(flow_type: str):
    """获取指定客流类型支持的算法列表"""
    validate_flow_type(flow_type)
    config = FLOW_TYPES[flow_type]
    
    return {
        "flow_type": flow_type,
        "name": config["name"],
        "daily_algorithms": config["daily_algos"],
        "hourly_algorithms": config["hourly_algos"],
        "total_daily": len(config["daily_algos"]),
        "total_hourly": len(config["hourly_algos"])
    }

@app.get("/models/{flow_type}/{granularity}/{flow_metric_type}/versions")
def list_model_versions(flow_type: str, granularity: str, flow_metric_type: str):
    """列出模型版本"""
    validate_flow_type(flow_type)
    validate_granularity(granularity)
    validate_flow_metric_type(flow_metric_type)
    validate_metric_for_flow_type(flow_metric_type, flow_type)
    
    base_dir = os.path.join(MODELS_ROOT, flow_type, granularity, flow_metric_type)
    versions = get_model_versions(base_dir)
    
    return {
        "flow_type": flow_type,
        "granularity": granularity,
        "flow_metric_type": flow_metric_type,
        "versions": versions,
        "total_versions": len(versions),
        "latest_version": versions[0] if versions else None
    }

@app.get("/models/{flow_type}/{granularity}/{flow_metric_type}/{version}/algorithms")
def list_algorithms_for_version(
    flow_type: str,
    granularity: str,
    flow_metric_type: str,
    version: str = Path(..., regex=r"^\d{8}$", description="模型版本日期 (YYYYMMDD)")
):
    """列出指定版本的可用算法"""
    validate_flow_type(flow_type)
    validate_granularity(granularity)
    validate_flow_metric_type(flow_metric_type)
    validate_metric_for_flow_type(flow_metric_type, flow_type)
    
    base_dir = os.path.join(MODELS_ROOT, flow_type, granularity, flow_metric_type, version)
    if not os.path.isdir(base_dir):
        raise HTTPException(
            status_code=404, 
            detail=f"Version {version} not found for {flow_type}/{granularity}/{flow_metric_type}"
        )
    
    try:
        available_algos = [d for d in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, d))]
        
        flow_config = FLOW_TYPES[flow_type]
        supported_algos = (flow_config["daily_algos"] if granularity == "daily" 
                          else flow_config["hourly_algos"])
        valid_algos = [algo for algo in available_algos if algo in supported_algos]
        
        return {
            "flow_type": flow_type,
            "granularity": granularity,
            "flow_metric_type": flow_metric_type,
            "version": version,
            "algorithms": valid_algos,
            "total_algorithms": len(valid_algos)
        }
    except Exception as e:
        logger.error(f"Error listing algorithms for {flow_type}/{granularity}/{flow_metric_type}/{version}: {e}")
        raise HTTPException(status_code=500, detail="Error reading model directory")

# ----------------------------------------------------------------------------
# Config endpoints
# ----------------------------------------------------------------------------
@app.get("/config/{flow_type}/{granularity}/{flow_metric_type}", response_model=ConfigResponse)
def get_config(flow_type: str, granularity: str, flow_metric_type: str, include_defaults: bool = True):
    """获取配置"""
    validate_flow_type(flow_type)
    validate_granularity(granularity)
    validate_flow_metric_type(flow_metric_type)
    validate_metric_for_flow_type(flow_metric_type, flow_type)
    
    try:
        config_file = get_config_filename(flow_type, granularity, flow_metric_type)
        default_daily = granularity == "daily"
        config = load_yaml_config(
            config_file, 
            default_daily=default_daily if include_defaults else False
        )
        
        return ConfigResponse(
            flow_type=flow_type, 
            granularity=granularity,
            flow_metric_type=flow_metric_type,
            config=config
        )
    except Exception as e:
        logger.error(f"Error loading config for {flow_type}/{granularity}/{flow_metric_type}: {e}")
        raise HTTPException(status_code=500, detail="Error loading configuration")

@app.put("/config/{flow_type}/{granularity}/{flow_metric_type}", response_model=ConfigResponse)
def update_config(flow_type: str, granularity: str, flow_metric_type: str, payload: Dict[str, Any]):
    """更新配置"""
    validate_flow_type(flow_type)
    validate_granularity(granularity)
    validate_flow_metric_type(flow_metric_type)
    validate_metric_for_flow_type(flow_metric_type, flow_type)
    
    try:
        config_file = get_config_filename(flow_type, granularity, flow_metric_type)
        default_daily = granularity == "daily"
        
        # 加载现有配置
        config = load_yaml_config(config_file, default_daily=default_daily)
        
        # 更新配置
        if payload:
            config.update(payload)
        
        # 保存配置
        save_yaml_config(config, config_file)
        
        logger.info(f"Updated {flow_type}/{granularity}/{flow_metric_type} configuration")
        return ConfigResponse(
            flow_type=flow_type, 
            granularity=granularity,
            flow_metric_type=flow_metric_type,
            config=config
        )
        
    except Exception as e:
        logger.error(f"Error updating config for {flow_type}/{granularity}/{flow_metric_type}: {e}")
        raise HTTPException(status_code=500, detail="Error updating configuration")

# ----------------------------------------------------------------------------
# Training endpoints
# ----------------------------------------------------------------------------
@app.post("/train/{flow_type}/daily/{flow_metric_type}", response_model=TrainingResponse)
def train_daily_model(flow_type: str, flow_metric_type: str, req: TrainRequest):
    """训练每日预测模型"""
    validate_flow_type(flow_type)
    validate_flow_metric_type(flow_metric_type)
    validate_metric_for_flow_type(flow_metric_type, flow_type)
    validate_algorithm_for_flow_type(req.algorithm, flow_type, "daily")
    
    try:
        # 加载配置
        config_file = get_config_filename(flow_type, "daily", flow_metric_type)
        config = load_yaml_config(config_file, default_daily=True)
        
        # 应用配置覆盖
        if req.config_overrides:
            train_params = config.get("train_params", {})
            train_params.update(req.config_overrides)
            config["train_params"] = train_params
            save_yaml_config(config, config_file)
            logger.info(f"Applied config overrides for {flow_type}/daily/{flow_metric_type}: {req.config_overrides}")

        # 创建模型保存目录
        model_save_dir = os.path.join(
            MODELS_ROOT, flow_type, "daily", flow_metric_type, req.train_end_date, req.algorithm
        )
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 生成图片路径
        plot_path = generate_plot_path(flow_type, "daily", flow_metric_type)

        logger.info(f"Starting {flow_type}/daily/{flow_metric_type} training: {req.algorithm} - {req.train_end_date}")
        
        # 获取预测函数并执行训练
        predict_func = get_prediction_function(flow_type, "daily")
        result = predict_func(
            file_path="",
            predict_start_date=req.train_end_date,
            algorithm=req.algorithm,
            retrain=req.retrain,
            save_path=plot_path,
            mode="train",
            days=15,
            config=config,
            model_version=None,
            model_save_dir=model_save_dir,
            flow_type=flow_type,
            metric_type=flow_metric_type
        )
        
        # 处理结果
        if isinstance(result, dict) and "error" in result:
            raise Exception(result["error"])
        
        # 安全编码结果
        safe_result = safe_json_encode(result)
        metrics = None
        
        if isinstance(safe_result, dict):
            metrics = {k: v for k, v in safe_result.items() 
                      if k in ['mae', 'rmse', 'mape', 'r2_score', 'training_time']}

        logger.info(f"{flow_type}/daily/{flow_metric_type} training completed successfully: {req.algorithm}")
        
        return TrainingResponse(
            message=f"{flow_type}/daily/{flow_metric_type} training completed successfully for {req.algorithm}",
            flow_type=flow_type,
            granularity="daily",
            flow_metric_type=flow_metric_type,
            algorithm=req.algorithm,
            model_version_date=req.train_end_date,
            model_dir=model_save_dir,
            plot_url=f"/plots/{os.path.basename(plot_path)}" if os.path.exists(plot_path) else None,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Error during {flow_type}/daily/{flow_metric_type} training: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Training failed: {str(e)}"
        )

@app.post("/train/{flow_type}/hourly/{flow_metric_type}", response_model=TrainingResponse)
def train_hourly_model(flow_type: str, flow_metric_type: str, req: TrainRequest):
    """训练小时预测模型"""
    validate_flow_type(flow_type)
    validate_flow_metric_type(flow_metric_type)
    validate_metric_for_flow_type(flow_metric_type, flow_type)
    validate_algorithm_for_flow_type(req.algorithm, flow_type, "hourly")
    
    try:
        # 加载配置
        config_file = get_config_filename(flow_type, "hourly", flow_metric_type)
        config = load_yaml_config(config_file, default_daily=False)
        
        # 应用配置覆盖
        if req.config_overrides:
            train_params = config.get("train_params", {})
            train_params.update(req.config_overrides)
            config["train_params"] = train_params
            save_yaml_config(config, config_file)
            logger.info(f"Applied config overrides for {flow_type}/hourly/{flow_metric_type}: {req.config_overrides}")

        # 创建模型保存目录
        model_save_dir = os.path.join(
            MODELS_ROOT, flow_type, "hourly", flow_metric_type, req.train_end_date, req.algorithm
        )
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 生成图片路径
        plot_path = generate_plot_path(flow_type, "hourly", flow_metric_type)

        logger.info(f"Starting {flow_type}/hourly/{flow_metric_type} training: {req.algorithm} - {req.train_end_date}")
        
        # 获取预测函数并执行训练
        predict_func = get_prediction_function(flow_type, "hourly")
        result = predict_func(
            file_path="",
            predict_date=req.train_end_date,
            algorithm=req.algorithm,
            retrain=True,  # 小时模型默认重训练
            save_path=plot_path,
            mode="train",
            config=config,
            model_version=None,
            model_save_dir=model_save_dir,
            flow_type=flow_type,
            metric_type=flow_metric_type
        )
        
        # 处理结果
        if isinstance(result, dict) and "error" in result:
            raise Exception(result["error"])
        
        # 安全编码结果
        safe_result = safe_json_encode(result)
        metrics = None
        
        if isinstance(safe_result, dict):
            metrics = {k: v for k, v in safe_result.items() 
                      if k in ['mae', 'rmse', 'mape', 'r2_score', 'training_time']}

        logger.info(f"{flow_type}/hourly/{flow_metric_type} training completed successfully: {req.algorithm}")
        
        return TrainingResponse(
            message=f"{flow_type}/hourly/{flow_metric_type} training completed successfully for {req.algorithm}",
            flow_type=flow_type,
            granularity="hourly",
            flow_metric_type=flow_metric_type,
            algorithm=req.algorithm,
            model_version_date=req.train_end_date,
            model_dir=model_save_dir,
            plot_url=f"/plots/{os.path.basename(plot_path)}" if os.path.exists(plot_path) else None,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Error during {flow_type}/hourly/{flow_metric_type} training: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Training failed: {str(e)}"
        )

# ----------------------------------------------------------------------------
# Prediction endpoints
# ----------------------------------------------------------------------------
@app.post("/predict/{flow_type}/daily/{flow_metric_type}", response_model=PredictionResponse)
def predict_daily_flow(flow_type: str, flow_metric_type: str, req: PredictDailyRequest):
    """每日流量预测"""
    validate_flow_type(flow_type)
    validate_flow_metric_type(flow_metric_type)
    validate_metric_for_flow_type(flow_metric_type, flow_type)
    validate_algorithm_for_flow_type(req.algorithm, flow_type, "daily")
 
    try:
        # 验证模型存在
        model_dir = validate_model_exists(flow_type, "daily", flow_metric_type, req.model_version_date, req.algorithm)
        
        # 加载配置
        config_file = get_config_filename(flow_type, "daily", flow_metric_type)
        config = load_yaml_config(config_file, default_daily=True)
        
        # 生成图片路径
        plot_path = generate_plot_path(flow_type, "daily", flow_metric_type)

        logger.info(f"Starting {flow_type}/daily/{flow_metric_type} prediction: {req.algorithm} - {req.predict_start_date}")
        
        # 获取预测函数并执行预测
        predict_func = get_prediction_function(flow_type, "daily")

        result = predict_func(
            file_path="",
            predict_start_date=req.predict_start_date,
            algorithm=req.algorithm,
            retrain=False,
            save_path=plot_path,
            mode="predict",
            days=req.days,
            config=config,
            model_version=None,
            model_save_dir=model_dir,
            flow_type=flow_type,
            metric_type=flow_metric_type
        )
        
        # 处理结果
        if isinstance(result, dict) and "error" in result:
            raise Exception(result["error"])
        
        # 安全编码结果
        safe_result = safe_json_encode(result)
        predictions = None
        metadata = None
        
        if isinstance(safe_result, dict):
            predictions = safe_result.get("predict_daily_flow")
            metadata = {
                "flow_type": flow_type,
                "flow_metric_type": flow_metric_type,
                "prediction_start": req.predict_start_date,
                "prediction_days": req.days,
                "model_version": req.model_version_date
            }

        logger.info(f"{flow_type}/daily/{flow_metric_type} prediction completed successfully: {req.algorithm}")
        
        return PredictionResponse(
            message=f"{flow_type}/daily/{flow_metric_type} prediction completed for {req.days} days",
            flow_type=flow_type,
            granularity="daily",
            flow_metric_type=flow_metric_type,
            algorithm=req.algorithm,
            model_version_date=req.model_version_date,
            plot_url=f"/plots/{os.path.basename(plot_path)}" if os.path.exists(plot_path) else None,
            predictions=predictions,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error during {flow_type}/daily/{flow_metric_type} prediction: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/{flow_type}/hourly/{flow_metric_type}", response_model=PredictionResponse)
def predict_hourly_flow(flow_type: str, flow_metric_type: str, req: PredictHourlyRequest):
    """小时流量预测"""
    validate_flow_type(flow_type)
    validate_flow_metric_type(flow_metric_type)
    validate_metric_for_flow_type(flow_metric_type, flow_type)
    validate_algorithm_for_flow_type(req.algorithm, flow_type, "hourly")

    try:
        # 验证模型存在
        model_dir = validate_model_exists(flow_type, "hourly", flow_metric_type, req.model_version_date, req.algorithm)
        
        # 加载配置
        config_file = get_config_filename(flow_type, "hourly", flow_metric_type)
        config = load_yaml_config(config_file, default_daily=False)
        
        # 生成图片路径
        plot_path = generate_plot_path(flow_type, "hourly", flow_metric_type)

        logger.info(f"Starting {flow_type}/hourly/{flow_metric_type} prediction: {req.algorithm} - {req.predict_date}")
        
        # 获取预测函数并执行预测
        predict_func = get_prediction_function(flow_type, "hourly")
        result = predict_func(
            file_path="",
            predict_date=req.predict_date,
            algorithm=req.algorithm,
            retrain=False,
            save_path=plot_path,
            mode="predict",
            config=config,
            model_version=None,
            model_save_dir=model_dir,
            flow_type=flow_type,
            metric_type=flow_metric_type
        )
        
        # 处理结果
        if isinstance(result, dict) and "error" in result:
            raise Exception(result["error"])
        
        # 安全编码结果
        safe_result = safe_json_encode(result)
        predictions = None
        metadata = None
        
        if isinstance(safe_result, dict):
            predictions = safe_result.get("predict_hourly_flow")
            metadata = {
                "flow_type": flow_type,
                "flow_metric_type": flow_metric_type,
                "prediction_date": req.predict_date,
                "model_version": req.model_version_date
            }

        logger.info(f"{flow_type}/hourly/{flow_metric_type} prediction completed successfully: {req.algorithm}")
        
        return PredictionResponse(
            message=f"{flow_type}/hourly/{flow_metric_type} prediction completed for {req.predict_date}",
            flow_type=flow_type,
            granularity="hourly",
            flow_metric_type=flow_metric_type,
            algorithm=req.algorithm,
            model_version_date=req.model_version_date,
            plot_url=f"/plots/{os.path.basename(plot_path)}" if os.path.exists(plot_path) else None,
            predictions=predictions,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error during {flow_type}/hourly/{flow_metric_type} prediction: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

# ----------------------------------------------------------------------------
# 启动配置
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=4566,
        reload=True,
        log_level="info"
    )

# ----------------------------------------------------------------------------
# API使用示例 (curl命令) - 更新版本包含指标类型
# ----------------------------------------------------------------------------
"""
# ============================================================================
# API使用示例 - 支持多种客流类型和指标类型
# ============================================================================

# 1. 健康检查
curl -s http://localhost:4566/health

# 2. 获取支持的客流类型
curl -s http://localhost:4566/flow-types

# 3. 获取支持的客流指标类型
curl -s http://localhost:4566/flow-metrics

# 4. 获取特定客流类型支持的指标
curl -s http://localhost:4566/flow-types/xianwangxianlu/metrics
curl -s http://localhost:4566/flow-types/duanmian/metrics
curl -s http://localhost:4566/flow-types/chezhan/metrics

# 5. 获取特定客流类型的算法
curl -s http://localhost:4566/algorithms/xianwangxianlu
curl -s http://localhost:4566/algorithms/duanmian
curl -s http://localhost:4566/algorithms/chezhan

# 6. 获取模型版本列表
curl -s http://localhost:4566/models/xianwangxianlu/daily/F_PKLCOUNT/versions
curl -s http://localhost:4566/models/duanmian/hourly/F_BOARD_ALIGHT/versions

# 7. 获取特定版本的算法列表
curl -s http://localhost:4566/models/xianwangxianlu/daily/F_PKLCOUNT/20250115/algorithms

# 8. 获取配置
curl -s http://localhost:4566/config/xianwangxianlu/daily/F_PKLCOUNT
curl -s http://localhost:4566/config/duanmian/hourly/F_BOARD_ALIGHT

# 9. 更新配置
curl -s -X PUT http://localhost:4566/config/xianwangxianlu/daily/F_PKLCOUNT \
  -H 'Content-Type: application/json' \
  -d '{"train_params": {"lookback_days": 7, "epochs": 100}}'

# 10. 训练模型

# 训练线网线路客流每日客运量模型
curl -s -X POST http://localhost:4566/train/xianwangxianlu/daily/F_PKLCOUNT \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "knn",
    "train_end_date": "20250115",
    "retrain": true,
    "config_overrides": {"n_neighbors": 5}
  }'

# 训练线网线路客流每日进站量模型
curl -s -X POST http://localhost:4566/train/xianwangxianlu/daily/F_ENTRANCE \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "prophet",
    "train_end_date": "20250115",
    "retrain": true,
    "config_overrides": {"seasonality_mode": "multiplicative"}
  }'

# 训练断面客流小时乘降量模型
curl -s -X POST http://localhost:4566/train/duanmian/hourly/F_BOARD_ALIGHT \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "lstm",
    "train_end_date": "20250115",
    "retrain": true,
    "config_overrides": {"lookback_hours": 72}
  }'

# 训练车站客流每日换乘量模型
curl -s -X POST http://localhost:4566/train/chezhan/daily/F_TRANSFER \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "xgboost",
    "train_end_date": "20250115",
    "retrain": false,
    "config_overrides": {"max_depth": 6}
  }'

# 训练换乘客流小时换乘量模型
curl -s -X POST http://localhost:4566/train/huhuan/hourly/F_TRANSFER \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "knn",
    "train_end_date": "20250115",
    "retrain": true,
    "config_overrides": {"n_neighbors": 3}
  }'

# 11. 预测

# 线网线路客流每日客运量预测
curl -s -X POST http://localhost:4566/predict/xianwangxianlu/daily/F_PKLCOUNT \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "knn",
    "model_version_date": "20250115",
    "predict_start_date": "20250120",
    "days": 15
  }'

# 线网线路客流每日进站量预测
curl -s -X POST http://localhost:4566/predict/xianwangxianlu/daily/F_ENTRANCE \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "prophet",
    "model_version_date": "20250115",
    "predict_start_date": "20250120",
    "days": 7
  }'

# 断面客流小时乘降量预测
curl -s -X POST http://localhost:4566/predict/duanmian/hourly/F_BOARD_ALIGHT \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "lstm",
    "model_version_date": "20250115",
    "predict_date": "20250120"
  }'

# 车站客流每日出站量预测
curl -s -X POST http://localhost:4566/predict/chezhan/daily/F_EXIT \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "lightgbm",
    "model_version_date": "20250115",
    "predict_start_date": "20250120",
    "days": 10
  }'

# 换乘客流小时进站量预测
curl -s -X POST http://localhost:4566/predict/huhuan/hourly/F_ENTRANCE \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "lstm",
    "model_version_date": "20250115",
    "predict_date": "20250120"
  }'

# 区域客流每日客运量预测
curl -s -X POST http://localhost:4566/predict/quyuxing/daily/F_PKLCOUNT \
  -H 'Content-Type: application/json' \
  -d '{
    "algorithm": "xgboost",
    "model_version_date": "20250115",
    "predict_start_date": "20250120",
    "days": 10
  }'

# ============================================================================
# 文件下载和文档访问
# ============================================================================

# 下载生成的图片 (从响应中获取plot_url)
curl -s http://localhost:4566/plots/xianwangxianlu_daily_F_PKLCOUNT_<uuid>.png --output prediction_plot.png

# API文档
# 浏览器访问: http://localhost:4566/docs
# ReDoc文档: http://localhost:4566/redoc

# ============================================================================
# 目录结构示例（更新版本包含指标类型）
# ============================================================================
# models/
# ├── xianwangxianlu/                    # 线网线路客流
# │   ├── daily/
# │   │   ├── F_PKLCOUNT/                # 客运量
# │   │   │   └── 20250115/
# │   │   │       ├── knn/
# │   │   │       ├── prophet/
# │   │   │       └── lstm/
# │   │   ├── F_ENTRANCE/               # 进站量
# │   │   │   └── 20250115/
# │   │   │       ├── knn/
# │   │   │       └── prophet/
# │   │   └── F_EXIT/                   # 出站量
# │   │       └── 20250115/
# │   │           └── lstm/
# │   └── hourly/
# │       ├── F_PKLCOUNT/
# │       │   └── 20250115/
# │       │       ├── knn/
# │       │       └── lstm/
# │       └── F_ENTRANCE/
# │           └── 20250115/
# │               └── lstm/
# ├── duanmian/                         # 断面客流
# │   ├── daily/
# │   │   ├── F_PKLCOUNT/
# │   │   └── F_BOARD_ALIGHT/
# │   └── hourly/
# │       ├── F_PKLCOUNT/
# │       └── F_BOARD_ALIGHT/
# ├── chezhan/                          # 车站客流
# │   ├── daily/
# │   │   ├── F_PKLCOUNT/
# │   │   ├── F_ENTRANCE/
# │   │   ├── F_EXIT/
# │   │   └── F_TRANSFER/
# │   └── hourly/
# │       ├── F_PKLCOUNT/
# │       ├── F_ENTRANCE/
# │       ├── F_EXIT/
# │       └── F_TRANSFER/
# ├── huhuan/                           # 换乘客流
# │   ├── daily/
# │   │   ├── F_TRANSFER/
# │   │   ├── F_ENTRANCE/
# │   │   └── F_EXIT/
# │   └── hourly/
# │       ├── F_TRANSFER/
# │       ├── F_ENTRANCE/
# │       └── F_EXIT/
# └── quyuxing/                         # 区域客流
#     ├── daily/
#     │   ├── F_PKLCOUNT/
#     │   ├── F_ENTRANCE/
#     │   └── F_EXIT/
#     └── hourly/
#         ├── F_PKLCOUNT/
#         ├── F_ENTRANCE/
#         └── F_EXIT/

# ============================================================================
# 配置文件命名规则（更新版本包含指标类型）
# ============================================================================
# model_config_xianwangxianlu_daily_F_PKLCOUNT.yaml
# model_config_xianwangxianlu_daily_F_ENTRANCE.yaml
# model_config_xianwangxianlu_daily_F_EXIT.yaml
# model_config_xianwangxianlu_hourly_F_PKLCOUNT.yaml
# model_config_xianwangxianlu_hourly_F_ENTRANCE.yaml
# model_config_duanmian_daily_F_PKLCOUNT.yaml
# model_config_duanmian_daily_F_BOARD_ALIGHT.yaml
# model_config_duanmian_hourly_F_PKLCOUNT.yaml
# model_config_duanmian_hourly_F_BOARD_ALIGHT.yaml
# ... 以此类推

# ============================================================================
# 主要变更总结
# ============================================================================
# 1. 新增 FLOW_METRIC_TYPES 常量定义五种客流指标类型
# 2. 在 FLOW_TYPES 中为每种客流类型添加 supported_metrics 字段
# 3. 所有API路径增加 {flow_metric_type} 参数
# 4. 模型存储路径增加指标类型层级：models/{flow_type}/{granularity}/{flow_metric_type}/{version}/{algorithm}
# 5. 配置文件名包含指标类型：model_config_{flow_type}_{granularity}_{flow_metric_type}.yaml
# 6. 新增客流指标相关的验证函数和端点
# 7. 响应模型增加 flow_metric_type 字段
# 8. 图片路径生成包含指标类型信息
# 9. 更新所有相关的验证、日志和异常处理逻辑

# ============================================================================
# 新增客流指标类型的开发步骤
# ============================================================================
# 1. 在 FLOW_METRIC_TYPES 中添加新的指标类型
# 2. 在相关的客流类型的 supported_metrics 中添加该指标
# 3. 创建对应的配置文件
# 4. 测试新的API端点
"""