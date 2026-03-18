# 日预测主流程模块：协调日客流预测的完整流程
"""
该模块负责日客流预测的完整流程，包括：
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

from db_utils import (
    read_line_daily_flow_history,
    read_station_daily_flow_history,
    upload_xianwangxianlu_daily_prediction_sample,
    upload_station_daily_prediction_sample,
    fetch_holiday_features
)
from enknn_model import KNNFlowPredictor
from config_utils import get_version_dir, get_current_version, filter_lines_by_config
from plot_utils import plot_daily_predictions
from weather_enum import WeatherType, get_weather_severity
from holiday_predict_utils import (
    predict_line_flow as expert_predict_line_flow,
    predict_station_flow as expert_predict_station_flow,
)
from common_utils import (
    to_int, safe_get, safe_get_int,
    normalize_line_no, validate_required_columns,
    generate_remarks_hash, parse_temperature_value
)
from logger_config import get_predict_logger

logger = get_predict_logger()


# =============================================================================
# 常量定义
# =============================================================================

# 必需的因子列
REQUIRED_FACTOR_COLUMNS = {
    'F_WEEK', 'F_HOLIDAYTYPE', 'F_HOLIDAYDAYS', 'F_HOLIDAYWHICHDAY',
    'F_DAYOFWEEK', 'WEATHER_TYPE', 'F_WEATHER', 'F_YEAR'
}

# 保留的因子列
FACTOR_COLUMNS = [
    'F_DATE', 'F_YEAR', 'F_WEEK', 'F_HOLIDAYTYPE', 'F_HOLIDAYDAYS',
    'F_HOLIDAYWHICHDAY', 'F_DAYOFWEEK', 'F_LINENO', 'F_LINENAME',
    'F_KLCOUNT', 'F_WEATHER', 'WEATHER_TYPE', 'WEATHER_SEVERITY',
    'TEMPERATURE_AVG'
]

DEFAULT_HOLIDAY_FUSION_ALPHA = 0.9
HOLIDAY_FUSION_ALPHA_BY_TYPE = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 0.1,
    5: 0.7,
    6: 0.9,
    7: 0.5,
}
SPRING_FESTIVAL_SEGMENT_ALPHA = {
    "early": 1.0,
    "mid": 0.0,
    "late": 1.0,
}


# =============================================================================
# 数据预处理函数
# =============================================================================

def preprocess_daily_data(
    df: pd.DataFrame,
    use_enhanced_weather: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    预处理日客流数据
    
    参数:
        df: 原始数据 DataFrame
        
    返回:
        (处理后的 DataFrame, 线路名称映射字典)
    """
    # 验证必需列
    error_msg = validate_required_columns(df, REQUIRED_FACTOR_COLUMNS)
    if error_msg:
        raise ValueError(error_msg)
    
    # 数据清洗
    df = df.dropna(subset=['F_DATE', 'F_KLCOUNT', 'F_LINENO'])
    df['F_DATE'] = df['F_DATE'].astype(str).str.strip()
    df['F_KLCOUNT'] = pd.to_numeric(df['F_KLCOUNT'], errors='coerce').fillna(0)
    df['F_LINENO'] = df['F_LINENO'].astype(str).str.zfill(2)
    
    # 处理天气类型
    if use_enhanced_weather:
        df['WEATHER_TYPE'] = df['WEATHER_TYPE'].apply(
            lambda x: WeatherType.get_weather_by_name(str(x)).value
        )
        df['WEATHER_SEVERITY'] = df['WEATHER_TYPE'].apply(get_weather_severity)
    else:
        df['WEATHER_TYPE'] = df['WEATHER_TYPE'].apply(
            lambda x: WeatherType.get_weather_by_name_legacy(str(x)).value
        )

    if 'TEMPERATURE_AVG' not in df.columns:
        raw_temp_col = next((col for col in ['TEMPERATURE_RAW', 'F_QW'] if col in df.columns), None)
        if raw_temp_col is not None:
            df['TEMPERATURE_AVG'] = df[raw_temp_col].apply(parse_temperature_value)
    
    # 只保留需要的列
    available_cols = [col for col in FACTOR_COLUMNS if col in df.columns]
    df = df[available_cols]
    
    # 生成线路名称映射
    line_name_map = {
        row['F_LINENO']: row['F_LINENAME']
        for _, row in df[['F_LINENO', 'F_LINENAME']].drop_duplicates().iterrows()
    }
    
    return df, line_name_map


def get_line_data(
    df: pd.DataFrame,
    line_no: str,
    predict_start_date: str,
    need_train: bool
) -> pd.DataFrame:
    """
    获取指定线路的训练/预测数据
    
    参数:
        df: 完整数据
        line_no: 线路编号
        predict_start_date: 预测起始日期
        need_train: 是否需要训练（影响数据筛选范围）
        
    返回:
        线路数据 DataFrame
    """
    if need_train:
        # 训练模式：只使用预测日期之前的数据
        return df[(df['F_LINENO'] == line_no) & (df['F_DATE'] <= predict_start_date)].copy()
    else:
        # 预测模式：使用所有该线路的数据
        return df[df['F_LINENO'] == line_no].copy()


# =============================================================================
# 预测结果处理函数
# =============================================================================

def build_prediction_row(
    line_no: str,
    line_name: str,
    pred_date: str,
    pred_value: int,
    predict_start_date: str,
    algorithm: str,
    factor_row: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    构建单条预测结果记录
    
    参数:
        line_no: 线路编号
        line_name: 线路名称
        pred_date: 预测日期
        pred_value: 预测值
        predict_start_date: 预测起始日期
        algorithm: 算法名称
        factor_row: 因子数据行（可选）
        
    返回:
        预测结果字典
    """
    try:
        line_int = int(line_no)
    except (ValueError, TypeError):
        line_int = 0
    
    f_lb_val = 1 if line_int == 0 else 0
    
    return {
        'ID': str(uuid.uuid4()),
        'F_DATE': pred_date,
        'F_HOLIDAYTYPE': safe_get_int(factor_row, 'F_HOLIDAYTYPE') if factor_row is not None else None,
        'F_LB': f_lb_val,
        'F_LINENO': line_int,
        'F_LINENAME': line_name,
        'F_PKLCOUNT': int(pred_value),
        'CREATETIME': int(datetime.now().strftime('%Y%m%d')),
        'CREATOR': 'knn_predict',
        'REMARKS': generate_remarks_hash(f"KNN日预测 {algorithm}"),
        'PREDICT_DATE': int(predict_start_date),
        'F_TYPE': 1,
        'F_WEEK': safe_get_int(factor_row, 'F_WEEK') if factor_row is not None else None,
        'F_HOLIDAYDAYS': safe_get_int(factor_row, 'F_HOLIDAYDAYS') if factor_row is not None else None,
        'F_HOLIDAYWHICHDAY': safe_get_int(factor_row, 'F_HOLIDAYWHICHDAY') if factor_row is not None else None,
        'F_DAYOFWEEK': safe_get_int(factor_row, 'F_DAYOFWEEK') if factor_row is not None else None,
    }


def get_factor_row(
    df: pd.DataFrame,
    pred_date: str,
    line_no: str
) -> Optional[pd.Series]:
    """
    获取预测日期的因子数据
    
    参数:
        df: 数据 DataFrame
        pred_date: 预测日期
        line_no: 线路编号
        
    返回:
        因子数据行，如果不存在返回 None
    """
    factor_df = df[(df['F_DATE'] == pred_date) & (df['F_LINENO'] == line_no)]
    if not factor_df.empty:
        return factor_df.iloc[0]
    return None


def get_holiday_fusion_alpha(factor_row: Optional[pd.Series]) -> float:
    """按节日类型返回融合权重，春节按假期第几天单独处理。"""
    holiday_type_id = safe_get_int(factor_row, 'F_HOLIDAYTYPE') if factor_row is not None else 0
    which_day = safe_get_int(factor_row, 'F_HOLIDAYWHICHDAY') if factor_row is not None else 0

    if holiday_type_id == 2:
        if which_day <= 3:
            return SPRING_FESTIVAL_SEGMENT_ALPHA["early"]
        if which_day <= 5:
            return SPRING_FESTIVAL_SEGMENT_ALPHA["mid"]
        return SPRING_FESTIVAL_SEGMENT_ALPHA["late"]

    return HOLIDAY_FUSION_ALPHA_BY_TYPE.get(holiday_type_id, DEFAULT_HOLIDAY_FUSION_ALPHA)


def build_fusion_prediction_rows(
    ml_rows: List[Dict],
    predict_start_date: str,
    days: int,
    metric_type: str,
) -> List[Dict]:
    """基于机器学习结果与专家系统生成融合预测入库数据。"""
    if not ml_rows:
        return []

    expert_result = expert_predict_line_flow(metric_type, predict_start_date, (datetime.strptime(predict_start_date, '%Y%m%d') + timedelta(days=days - 1)).strftime('%Y%m%d'), history_years=2)
    expert_predictions = expert_result.get("predictions", []) if isinstance(expert_result, dict) else []
    expert_by_key = {}
    for pred in expert_predictions:
        expert_by_key[(str(pred.get("线路名称", "")), str(pred.get("预测日期", "")))] = pred

    fusion_rows = []
    for ml_row in ml_rows:
        line_name = str(ml_row.get('F_LINENAME', ''))
        pred_date = str(ml_row.get('F_DATE', ''))
        expert_row = expert_by_key.get((line_name, pred_date))
        if not expert_row:
            continue

        factor_stub = pd.Series({
            'F_HOLIDAYTYPE': ml_row.get('F_HOLIDAYTYPE'),
            'F_HOLIDAYWHICHDAY': ml_row.get('F_HOLIDAYWHICHDAY'),
        })
        fusion_alpha = get_holiday_fusion_alpha(factor_stub)
        ml_value = int(ml_row.get('F_PKLCOUNT', 0))
        expert_value = int(expert_row.get('预测客流', 0))
        fused_value = int(fusion_alpha * ml_value + (1 - fusion_alpha) * expert_value)

        fusion_row = dict(ml_row)
        fusion_row['ID'] = str(uuid.uuid4())
        fusion_row['F_PKLCOUNT'] = fused_value
        fusion_row['CREATOR'] = 'fusion_predict'
        fusion_row['REMARKS'] = generate_remarks_hash(f"融合日预测 alpha={fusion_alpha:.1f}")
        fusion_rows.append(fusion_row)

    return fusion_rows


# =============================================================================
# 主预测函数
# =============================================================================

def predict_and_plot_timeseries_flow_daily(
    file_path: str,
    predict_start_date: str,
    algorithm: str = 'knn',
    retrain: bool = False,
    save_path: str = "timeseries_predict_daily.png",
    mode: str = 'all',
    days: int = 15,
    config: Optional[Dict] = None,
    model_version: Optional[str] = None,
    model_save_dir: Optional[str] = None,
    flow_type: Optional[str] = None,
    metric_type: Optional[str] = None,
) -> Dict:
    """
    预测并绘制日客流
    
    参数:
        file_path: 文件路径（已废弃，保留用于兼容）
        predict_start_date: 预测起始日期 (YYYYMMDD)
        algorithm: 预测算法 (默认 'knn')
        retrain: 是否强制重新训练模型
        save_path: 图表保存路径
        mode: 操作模式 ('all', 'train', 'predict')
        days: 预测天数
        config: 配置字典
        model_version: 指定模型版本
        model_save_dir: 模型保存目录
        flow_type: 客流类型 ('xianwangxianlu', 'chezhan')
        metric_type: 指标类型
        
    返回:
        预测结果字典
    """
    logger.info(f"开始日客流预测 - 类型: {flow_type}, 指标: {metric_type}, 日期: {predict_start_date}")

    use_enhanced_weather = flow_type != 'chezhan'
    effective_config = dict(config or {})
    if not use_enhanced_weather:
        effective_config["factors"] = [
            'F_WEEK', 'F_HOLIDAYTYPE', 'F_HOLIDAYDAYS',
            'F_HOLIDAYWHICHDAY', 'F_DAYOFWEEK', 'WEATHER_TYPE',
            'TEMPERATURE_AVG', 'F_YEAR'
        ]
    
    # 1. 版本号和模型目录管理
    version = model_version or get_current_version(config_obj=effective_config, config_path="model_config_daily.yaml")
    model_dir = model_save_dir or get_version_dir(version, config_obj=config)
    
    # 2. 初始化预测器
    predictor = KNNFlowPredictor(model_dir, version, effective_config)
    
    # 3. 读取数据
    try:
        if flow_type == 'chezhan':
            df = read_station_daily_flow_history(metric_type)
        else:
            df = read_line_daily_flow_history(metric_type)
        
        if not isinstance(df, pd.DataFrame):
            return {"error": "数据读取失败（非 DataFrame），请检查数据库结构"}
    except Exception as e:
        logger.error(f"数据库读取失败: {e}")
        return {"error": f"数据库读取失败: {e}"}
    
    # 4. 数据预处理
    try:
        df, line_name_map = preprocess_daily_data(df, use_enhanced_weather=use_enhanced_weather)
    except ValueError as e:
        return {"error": str(e)}
    
    lines = sorted(df['F_LINENO'].unique().tolist())
    
    # 5. 根据配置过滤线路（仅线网线路预测时生效）
    if flow_type == 'xianwangxianlu':
        original_count = len(lines)
        lines = filter_lines_by_config(lines, config, "xianwangxianlu_daily_predict")
        if len(lines) < original_count:
            logger.info(f"线路过滤已启用：从 {original_count} 条线路过滤为 {len(lines)} 条")
            logger.info(f"启用的线路: {lines}")
    
    predict_result = {}
    prediction_rows = []
    
    # 6. 打印训练数据统计
    _log_training_stats(df, lines, line_name_map, predictor, predict_start_date, version, retrain)
    
    # 7. 针对每条线路进行训练/预测
    for line in lines:
        model_info_path = os.path.join(predictor.model_dir, f"model_info_line_{line}_daily_v{version}.json")
        need_train = retrain or not os.path.exists(model_info_path)
        
        line_data = get_line_data(df, line, predict_start_date, need_train)
        line_name = line_name_map.get(line, line)
        
        # 处理空数据情况
        if line_data.empty:
            predict_result[line] = _create_empty_result(algorithm, predict_start_date, days, "此线路无数据")
            continue
        
        # 执行训练
        if mode in ['all', 'train'] and need_train:
            result = _train_line_model(predictor, line_data, line, algorithm, version, predict_start_date, days)
            if result.get('error'):
                predict_result[line] = result
                continue
        elif mode == 'train' and not need_train:
            continue
        
        # 执行预测
        if mode in ['all', 'predict']:
            result = _predict_line(
                predictor, line_data, line, line_name, algorithm,
                predict_start_date, days, version, df, use_enhanced_weather
            )
            
            if result.get('error'):
                predict_result[line] = result
            else:
                predict_result[line] = result['result']
                prediction_rows.extend(result['rows'])
    
    # 8. 保存结果和绘图
    if mode in ['all', 'predict'] and prediction_rows:
        _save_and_plot_results(
            prediction_rows, predict_result, line_name_map,
            predict_start_date, days, save_path, flow_type, metric_type
        )
    
    logger.info(f"日客流预测完成 - 共 {len(lines)} 条线路")
    return predict_result


# =============================================================================
# 内部辅助函数
# =============================================================================

def _log_training_stats(
    df: pd.DataFrame,
    lines: List[str],
    line_name_map: Dict[str, str],
    predictor: KNNFlowPredictor,
    predict_start_date: str,
    version: str,
    retrain: bool
) -> None:
    """打印训练数据统计信息"""
    logger.info("训练数据统计：")
    for line in lines:
        model_info_path = os.path.join(predictor.model_dir, f"model_info_line_{line}_daily_v{version}.json")
        need_train = retrain or not os.path.exists(model_info_path)
        
        if need_train:
            line_data = df[(df['F_LINENO'] == line) & (df['F_DATE'] <= predict_start_date)]
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


def _create_empty_result(algorithm: str, predict_start_date: str, days: int, error: str) -> Dict:
    """创建空结果"""
    return {
        "algorithm": algorithm,
        "predict_daily_flow": {
            (datetime.strptime(predict_start_date, '%Y%m%d') + timedelta(days=d)).strftime('%Y%m%d'): 0
            for d in range(days)
        },
        "predict_start_date": predict_start_date,
        "error": error
    }


def _train_line_model(
    predictor: KNNFlowPredictor,
    line_data: pd.DataFrame,
    line: str,
    algorithm: str,
    version: str,
    predict_start_date: str,
    days: int
) -> Dict:
    """训练单条线路模型"""
    mse, mae, error = predictor.train(line_data, line, model_version=version)
    
    if error:
        return _create_empty_result(algorithm, predict_start_date, days, error)
    
    predictor.save_model_info(line, algorithm, mse, mae, datetime.now().strftime('%Y%m%d'), model_version=version)
    return {}


def _predict_line(
    predictor: KNNFlowPredictor,
    line_data: pd.DataFrame,
    line: str,
    line_name: str,
    algorithm: str,
    predict_start_date: str,
    days: int,
    version: str,
    df: pd.DataFrame,
    use_enhanced_weather: bool = True,
) -> Dict:
    """执行单条线路预测"""
    # 获取节假日特征
    holiday_features = fetch_holiday_features(predict_start_date, days)
    if use_enhanced_weather:
        holiday_features['WEATHER_TYPE'] = holiday_features['WEATHER_TYPE'].apply(
            lambda x: WeatherType.get_weather_by_name(str(x)).value
        )
        holiday_features['WEATHER_SEVERITY'] = holiday_features['WEATHER_TYPE'].apply(get_weather_severity)
    else:
        holiday_features['WEATHER_TYPE'] = holiday_features['WEATHER_TYPE'].apply(
            lambda x: WeatherType.get_weather_by_name_legacy(str(x)).value
        )
    holiday_features['TEMPERATURE_AVG'] = holiday_features.get('TEMPERATURE_RAW', pd.Series([None] * len(holiday_features))).apply(parse_temperature_value)
    
    # 执行预测
    predictions, error = predictor.predict(
        line_data, line, predict_start_date, days,
        model_version=version, factor_df=holiday_features
    )
    
    if error:
        return {"error": error, "result": _create_empty_result(algorithm, predict_start_date, days, error)}
    
    # 构建结果
    daily_flow = {
        (datetime.strptime(predict_start_date, '%Y%m%d') + timedelta(days=d)).strftime('%Y%m%d'): int(predictions[d])
        for d in range(days)
    }
    
    result = {
        "algorithm": algorithm,
        "predict_daily_flow": daily_flow,
        "predict_start_date": predict_start_date,
        "error": None,
        "line_data": line_data
    }
    
    # 构建入库数据
    prediction_rows = []
    for d in range(days):
        pred_date = (datetime.strptime(predict_start_date, '%Y%m%d') + timedelta(days=d)).strftime('%Y%m%d')
        factor_row = get_factor_row(df, pred_date, line)
        
        row = build_prediction_row(
            line_no=line,
            line_name=line_name,
            pred_date=pred_date,
            pred_value=int(predictions[d]),
            predict_start_date=predict_start_date,
            algorithm=algorithm,
            factor_row=factor_row
        )
        prediction_rows.append(row)
    
    return {"result": result, "rows": prediction_rows}


def _save_and_plot_results(
    prediction_rows: List[Dict],
    predict_result: Dict,
    line_name_map: Dict[str, str],
    predict_start_date: str,
    days: int,
    save_path: str,
    flow_type: str,
    metric_type: str
) -> None:
    """保存预测结果并绘图"""
    # 保存到数据库
    if flow_type == 'xianwangxianlu':
        upload_xianwangxianlu_daily_prediction_sample(prediction_rows, metric_type)
        fusion_rows = build_fusion_prediction_rows(prediction_rows, predict_start_date, days, metric_type)
        if fusion_rows:
            upload_xianwangxianlu_daily_prediction_sample(fusion_rows, metric_type)
            logger.info(f"融合日预测已写库: {len(fusion_rows)} 条, CREATOR=fusion_predict")
    else:
        upload_station_daily_prediction_sample(prediction_rows, metric_type)
    
    # 绘制图表
    plot_daily_predictions(predict_result, line_name_map, predict_start_date, days, save_path, flow_type, metric_type)
