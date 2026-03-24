# 小时预测主流程模块：协调小时客流预测的完整流程
"""
该模块负责小时客流预测的完整流程，包括：
- 数据加载和预处理
- 模型训练
- 预测执行
- 结果可视化和存储
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import uuid
import os
import re

from db_utils import (
    read_line_hourly_flow_history,
    read_station_hourly_flow_history,
    upload_xianwangxianlu_hourly_prediction_sample,
    upload_station_hourly_prediction_sample,
    fetch_holiday_features
)
from hourknn_model import KNNHourlyFlowPredictor
from config_utils import get_version_dir, get_current_version
from plot_utils import plot_hourly_predictions
from weather_enum import WeatherType, get_weather_severity
from common_utils import (
    to_int, safe_get, safe_get_int,
    normalize_line_no, validate_required_columns,
    generate_remarks_hash
)
from logger_config import get_predict_logger

logger = get_predict_logger()


# =============================================================================
# 常量定义
# =============================================================================

# 必需的因子列
REQUIRED_HOURLY_COLUMNS = {
    'F_DATE', 'F_HOUR', 'F_YEAR', 'F_WEEK', 'F_HOLIDAYTYPE',
    'F_HOLIDAYDAYS', 'F_HOLIDAYWHICHDAY', 'F_DAYOFWEEK',
    'F_LINENO', 'F_LINENAME', 'F_KLCOUNT', 'F_WEATHER', 'WEATHER_TYPE'
}

# 保留的因子列
HOURLY_FACTOR_COLUMNS = [
    'F_DATE', 'F_HOUR', 'F_YEAR', 'F_WEEK', 'F_HOLIDAYTYPE',
    'F_HOLIDAYDAYS', 'F_HOLIDAYWHICHDAY', 'F_DAYOFWEEK',
    'F_LINENO', 'F_LINENAME', 'F_KLCOUNT', 'F_WEATHER',
    'WEATHER_TYPE', 'WEATHER_SEVERITY', 'F_RUSH_HOUR_TYPE'
]


# =============================================================================
# 早晚高峰计算函数
# =============================================================================

def calculate_rush_hour_type(hour: Any) -> int:
    """
    计算早晚高峰类型因子
    
    返回:
        0: 非高峰
        1: 早高峰 (6-10点)
        2: 晚高峰 (16-20点)
    
    参数:
        hour: 小时值（可以是 int, float, str）
    """
    try:
        # 处理 NaN/None
        if hour is None or (isinstance(hour, float) and pd.isna(hour)):
            return 0
        
        # 处理字符串
        if isinstance(hour, str):
            hour_str = hour.strip()
            if hour_str.isdigit():
                hour_int = int(hour_str)
            else:
                match = re.search(r'\d+', hour_str)
                hour_int = int(match.group()) if match else 0
        elif isinstance(hour, (int, float)):
            if pd.isna(hour):
                return 0
            hour_int = int(float(hour))
        else:
            return 0
    except (ValueError, TypeError, AttributeError):
        return 0
    
    # 验证范围
    if not (0 <= hour_int <= 23):
        return 0
    
    # 早高峰：6-10点
    if 6 <= hour_int <= 10:
        return 1
    # 晚高峰：16-20点
    elif 16 <= hour_int <= 20:
        return 2
    # 非高峰
    return 0


# =============================================================================
# 数据预处理函数
# =============================================================================

def preprocess_hourly_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    预处理小时客流数据
    
    参数:
        df: 原始数据 DataFrame
        
    返回:
        (处理后的 DataFrame, 线路名称映射字典)
    """
    # 验证必需列
    error_msg = validate_required_columns(df, REQUIRED_HOURLY_COLUMNS)
    if error_msg:
        raise ValueError(error_msg)
    
    # 数据清洗
    df = df.dropna(subset=['F_DATE', 'F_HOUR', 'F_KLCOUNT', 'F_LINENO'])
    df['F_DATE'] = df['F_DATE'].astype(str).str.strip()
    df['F_HOUR'] = df['F_HOUR'].astype(str).str.zfill(2)
    df['F_KLCOUNT'] = pd.to_numeric(df['F_KLCOUNT'], errors='coerce').fillna(0)
    df['F_LINENO'] = df['F_LINENO'].astype(str).str.zfill(2)
    
    # 处理天气类型
    df['WEATHER_TYPE'] = df['WEATHER_TYPE'].apply(
        lambda x: WeatherType.get_weather_by_name(str(x)).value
    )
    df['WEATHER_SEVERITY'] = df['WEATHER_TYPE'].apply(get_weather_severity)
    
    # 添加早晚高峰因子
    df['F_RUSH_HOUR_TYPE'] = df['F_HOUR'].apply(calculate_rush_hour_type)
    
    # 只保留需要的列
    available_cols = [col for col in HOURLY_FACTOR_COLUMNS if col in df.columns]
    df = df[available_cols]
    
    # 生成线路名称映射
    line_name_map = {
        row['F_LINENO']: row['F_LINENAME']
        for _, row in df[['F_LINENO', 'F_LINENAME']].drop_duplicates().iterrows()
    }
    
    return df, line_name_map


def get_hourly_line_data(
    df: pd.DataFrame,
    line_no: str,
    predict_date: str,
    need_train: bool
) -> pd.DataFrame:
    """
    获取指定线路的训练/预测数据
    """
    if need_train:
        return df[(df['F_LINENO'] == line_no) & (df['F_DATE'] <= predict_date)].copy()
    else:
        return df[df['F_LINENO'] == line_no].copy()


# =============================================================================
# 预测结果处理函数
# =============================================================================

def build_hourly_prediction_row(
    line_no: str,
    line_name: str,
    pred_date: str,
    hour: int,
    pred_value: int,
    algorithm: str,
    factor_row: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    构建单条小时预测结果记录
    """
    try:
        line_int = int(line_no)
    except (ValueError, TypeError):
        line_int = 0
    
    hour_str = f"{hour:02d}"
    
    return {
        'ID': str(uuid.uuid4()),
        'F_DATE': int(pred_date),
        'F_HOUR': hour_str,
        'F_LINENO': line_int,
        'F_LINENAME': line_name,
        'F_PKLCOUNT': pred_value,
        'F_BEFPKLCOUNT': None,
        'F_PRUTE': None,
        'CREATETIME': int(datetime.now().strftime('%Y%m%d')),
        'CREATOR': 'knn_predict',
        'REMARKS': generate_remarks_hash(f"KNN小时预测 {algorithm}"),
        'PREDICT_DATE': int(pred_date),
        'PREDICT_WEATHER': None,
        'F_WEEK': safe_get_int(factor_row, 'F_WEEK') if factor_row is not None else None,
        'F_HOLIDAYTYPE': safe_get_int(factor_row, 'F_HOLIDAYTYPE') if factor_row is not None else None,
        'F_HOLIDAYDAYS': safe_get_int(factor_row, 'F_HOLIDAYDAYS') if factor_row is not None else None,
        'F_HOLIDAYWHICHDAY': safe_get_int(factor_row, 'F_HOLIDAYWHICHDAY') if factor_row is not None else None,
        'F_DAYOFWEEK': safe_get_int(factor_row, 'F_DAYOFWEEK') if factor_row is not None else None,
        'WEATHER_TYPE': safe_get_int(factor_row, 'WEATHER_TYPE') if factor_row is not None else None,
        'WEATHER_SEVERITY': safe_get_int(factor_row, 'WEATHER_SEVERITY') if factor_row is not None else None,
        'F_RUSH_HOUR_TYPE': safe_get_int(factor_row, 'F_RUSH_HOUR_TYPE', calculate_rush_hour_type(hour)) if factor_row is not None else calculate_rush_hour_type(hour),
        'F_YEAR': safe_get_int(factor_row, 'F_YEAR', int(pred_date[:4])) if factor_row is not None else int(pred_date[:4]),
    }


def get_hourly_factor_row(
    df: pd.DataFrame,
    pred_date: str,
    hour_str: str,
    line_no: str,
    holiday_features: Optional[pd.DataFrame] = None
) -> Optional[pd.Series]:
    """
    获取预测时段的因子数据
    """
    # 先从原始数据查找
    factor_df = df[(df['F_DATE'] == pred_date) & (df['F_HOUR'] == hour_str) & (df['F_LINENO'] == line_no)]
    if not factor_df.empty:
        return factor_df.iloc[0]
    
    # 从节假日特征查找
    if holiday_features is not None and not holiday_features.empty:
        if 'F_HOUR' in holiday_features.columns:
            hour_features = holiday_features[holiday_features['F_HOUR'] == hour_str]
            if not hour_features.empty:
                return hour_features.iloc[0]
    
    return None


def create_hourly_holiday_features(
    holiday_features: pd.DataFrame,
    predict_date: str
) -> pd.DataFrame:
    """
    为24小时生成因子数据
    """
    if holiday_features is not None and not holiday_features.empty:
        # 处理天气类型
        holiday_features['WEATHER_TYPE'] = holiday_features['WEATHER_TYPE'].apply(
            lambda x: WeatherType.get_weather_by_name(str(x)).value
        )
        holiday_features['WEATHER_SEVERITY'] = holiday_features['WEATHER_TYPE'].apply(get_weather_severity)
        
        # 为每个小时生成因子
        hourly_features = []
        for h in range(24):
            hour_row = holiday_features.iloc[0].copy()
            hour_row['F_HOUR'] = f"{h:02d}"
            hour_row['F_RUSH_HOUR_TYPE'] = calculate_rush_hour_type(h)
            hourly_features.append(hour_row)
        return pd.DataFrame(hourly_features)
    else:
        # 创建默认因子
        return pd.DataFrame({
            'F_DATE': [int(predict_date)] * 24,
            'F_HOUR': [f"{h:02d}" for h in range(24)],
            'F_YEAR': [int(predict_date[:4])] * 24,
            'F_WEEK': [None] * 24,
            'F_HOLIDAYTYPE': [None] * 24,
            'F_HOLIDAYDAYS': [None] * 24,
            'F_HOLIDAYWHICHDAY': [None] * 24,
            'F_DAYOFWEEK': [None] * 24,
            'WEATHER_TYPE': [0] * 24,
            'WEATHER_SEVERITY': [0] * 24,
            'F_RUSH_HOUR_TYPE': [calculate_rush_hour_type(h) for h in range(24)],
        })


# =============================================================================
# 主预测函数
# =============================================================================

def predict_and_plot_timeseries_flow(
    file_path: str,
    predict_date: str,
    algorithm: str = 'knn',
    retrain: bool = False,
    save_path: str = "timeseries_predict_hourly.png",
    mode: str = 'all',
    config: Optional[Dict] = None,
    model_version: Optional[str] = None,
    model_save_dir: Optional[str] = None,
    flow_type: Optional[str] = None,
    metric_type: Optional[str] = None,
) -> Dict:
    """
    预测并绘制小时客流
    
    参数:
        file_path: 文件路径（已废弃，保留用于兼容）
        predict_date: 预测日期 (YYYYMMDD)
        algorithm: 预测算法 (默认 'knn')
        retrain: 是否强制重新训练模型
        save_path: 图表保存路径
        mode: 操作模式 ('all', 'train', 'predict')
        config: 配置字典
        model_version: 指定模型版本
        model_save_dir: 模型保存目录
        flow_type: 客流类型 ('xianwangxianlu', 'chezhan')
        metric_type: 指标类型
        
    返回:
        预测结果字典
    """
    logger.info(f"开始小时客流预测 - 类型: {flow_type}, 指标: {metric_type}, 日期: {predict_date}")
    
    # 1. 版本号和模型目录管理
    version = model_version or get_current_version(config_obj=config, config_path="model_config.yaml")
    model_dir = model_save_dir or get_version_dir(version, config_obj=config)
    
    # 2. 初始化预测器
    predictor = KNNHourlyFlowPredictor(model_dir, version, config or {})
    
    # 3. 读取数据
    try:
        if flow_type == 'chezhan':
            df = read_station_hourly_flow_history(metric_type, None)
        else:
            df = read_line_hourly_flow_history(metric_type, None)
        
        if not isinstance(df, pd.DataFrame):
            return {"error": "数据读取失败（非 DataFrame），请检查数据库结构"}
    except Exception as e:
        logger.error(f"数据库读取失败: {e}")
        return {"error": f"数据库读取失败: {e}"}
    
    # 4. 数据预处理
    try:
        df, line_name_map = preprocess_hourly_data(df)
    except ValueError as e:
        return {"error": str(e)}
    
    lines = sorted(df['F_LINENO'].unique().tolist())
    predict_result = {}
    prediction_rows = []
    
    # 5. 打印训练数据统计
    _log_hourly_training_stats(df, lines, line_name_map, predictor, predict_date, version, retrain)
    
    # 6. 针对每条线路进行训练/预测
    for line in lines:
        model_info_path = os.path.join(predictor.model_dir, f"model_info_line_{line}_hourly_v{version}.json")
        need_train = retrain or not os.path.exists(model_info_path)
        
        line_data = get_hourly_line_data(df, line, predict_date, need_train)
        line_name = line_name_map.get(line, line)
        
        # 处理空数据情况
        if line_data.empty:
            predict_result[line] = _create_empty_hourly_result(algorithm, predict_date, "此线路无数据")
            continue
        
        # 执行训练
        if mode in ['all', 'train'] and need_train:
            result = _train_hourly_model(predictor, line_data, line, algorithm, version, predict_date)
            if result.get('error'):
                predict_result[line] = result
                continue
        elif mode == 'train' and not need_train:
            continue
        
        # 执行预测
        if mode in ['all', 'predict']:
            result = _predict_hourly_line(
                predictor, line_data, line, line_name, algorithm,
                predict_date, version, df
            )
            
            if result.get('error'):
                predict_result[line] = result
            else:
                predict_result[line] = result['result']
                prediction_rows.extend(result['rows'])
    
    # 7. 保存结果和绘图
    if mode in ['all', 'predict'] and prediction_rows:
        _save_and_plot_hourly_results(
            prediction_rows, predict_result, line_name_map,
            predict_date, save_path, flow_type, metric_type
        )
    
    logger.info(f"小时客流预测完成 - 共 {len(lines)} 条线路")
    return predict_result


# =============================================================================
# 内部辅助函数
# =============================================================================

def _log_hourly_training_stats(
    df: pd.DataFrame,
    lines: List[str],
    line_name_map: Dict[str, str],
    predictor: KNNHourlyFlowPredictor,
    predict_date: str,
    version: str,
    retrain: bool
) -> None:
    """打印训练数据统计信息"""
    logger.info("训练数据统计：")
    for line in lines:
        model_info_path = os.path.join(predictor.model_dir, f"model_info_line_{line}_hourly_v{version}.json")
        need_train = retrain or not os.path.exists(model_info_path)
        
        if need_train:
            line_data = df[(df['F_LINENO'] == line) & (df['F_DATE'] <= predict_date)]
        else:
            line_data = df[df['F_LINENO'] == line]
        
        line_name = line_name_map.get(line, line)
        if not line_data.empty:
            min_date = line_data['F_DATE'].min()
            max_date = line_data['F_DATE'].max()
            count = len(line_data)
            logger.info(f"线路: {line} ({line_name})，数据日期: {min_date} ~ {max_date}，共 {count} 条")
        else:
            logger.info(f"线路: {line} ({line_name})，无数据")


def _create_empty_hourly_result(algorithm: str, predict_date: str, error: str) -> Dict:
    """创建空结果"""
    return {
        "algorithm": algorithm,
        "predict_hourly_flow": {f"{h:02d}": 0 for h in range(24)},
        "predict_date": predict_date,
        "error": error
    }


def _train_hourly_model(
    predictor: KNNHourlyFlowPredictor,
    line_data: pd.DataFrame,
    line: str,
    algorithm: str,
    version: str,
    predict_date: str
) -> Dict:
    """训练单条线路模型"""
    mse, mae, error = predictor.train(line_data, line, model_version=version)
    
    if error:
        return _create_empty_hourly_result(algorithm, predict_date, error)
    
    predictor.save_model_info(line, algorithm, mse, mae, datetime.now().strftime('%Y%m%d'), model_version=version)
    return {}


def _predict_hourly_line(
    predictor: KNNHourlyFlowPredictor,
    line_data: pd.DataFrame,
    line: str,
    line_name: str,
    algorithm: str,
    predict_date: str,
    version: str,
    df: pd.DataFrame
) -> Dict:
    """执行单条线路预测"""
    # 获取节假日特征并扩展到24小时
    holiday_features = fetch_holiday_features(predict_date, 1)
    hourly_holiday_features = create_hourly_holiday_features(holiday_features, predict_date)
    
    # 执行预测
    predictions, error = predictor.predict(
        line_data, line, predict_date, 24,
        model_version=version, factor_df=hourly_holiday_features
    )
    
    if error:
        return {"error": error, "result": _create_empty_hourly_result(algorithm, predict_date, error)}
    
    # 构建结果（0-5点设为0）
    hourly_flow = {}
    for h in range(24):
        if h <= 5:
            hourly_flow[f"{h:02d}"] = 0
        else:
            hourly_flow[f"{h:02d}"] = int(predictions[h])
    
    result = {
        "algorithm": algorithm,
        "predict_hourly_flow": hourly_flow,
        "predict_date": predict_date,
        "error": None,
        "line_data": line_data
    }
    
    # 构建入库数据
    prediction_rows = []
    for h in range(24):
        hour_str = f"{h:02d}"
        pred_value = 0 if h <= 5 else int(predictions[h])
        
        factor_row = get_hourly_factor_row(df, predict_date, hour_str, line, hourly_holiday_features)
        
        row = build_hourly_prediction_row(
            line_no=line,
            line_name=line_name,
            pred_date=predict_date,
            hour=h,
            pred_value=pred_value,
            algorithm=algorithm,
            factor_row=factor_row
        )
        prediction_rows.append(row)
    
    return {"result": result, "rows": prediction_rows}


def _save_and_plot_hourly_results(
    prediction_rows: List[Dict],
    predict_result: Dict,
    line_name_map: Dict[str, str],
    predict_date: str,
    save_path: str,
    flow_type: str,
    metric_type: str
) -> None:
    """保存预测结果并绘图"""
    # 保存到数据库
    if flow_type == 'xianwangxianlu':
        upload_xianwangxianlu_hourly_prediction_sample(prediction_rows, metric_type)
    else:
        upload_station_hourly_prediction_sample(prediction_rows, metric_type)
    
    # 绘制图表
    plot_hourly_predictions(predict_result, line_name_map, predict_date, save_path, flow_type, metric_type)
