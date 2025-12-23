# 小时预测主流程模块：协调小时客流预测的完整流程（完全参考日客流预测逻辑，增加F_HOUR和早晚高峰因子）
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
import os

from db_utils import read_line_hourly_flow_history, read_station_hourly_flow_history, upload_xianwangxianlu_hourly_prediction_sample, upload_station_hourly_prediction_sample, fetch_holiday_features
from hourknn_model import KNNHourlyFlowPredictor
from config_utils import get_version_dir, get_current_version
from plot_utils import plot_hourly_predictions
from weather_enum import WeatherType

def calculate_rush_hour_type(hour) -> int:
    """
    计算早晚高峰类型因子
    返回: 0=非高峰, 1=早高峰, 2=晚高峰
    
    早高峰：6-10点
    晚高峰：16-20点
    
    参数:
        hour: 小时值，可以是 int, float, str (如 "00", "01", "7", "17" 等)
    """
    # 处理各种输入类型
    try:
        # 处理 NaN/None
        if hour is None or (isinstance(hour, float) and pd.isna(hour)):
            return 0
        
        # 如果是字符串，先尝试转换为整数
        if isinstance(hour, str):
            hour_str = str(hour).strip()
            # 处理 "00", "01" 等格式
            if hour_str.isdigit():
                hour_int = int(hour_str)
            else:
                # 尝试提取数字部分
                import re
                match = re.search(r'\d+', hour_str)
                if match:
                    hour_int = int(match.group())
                else:
                    return 0
        elif isinstance(hour, (int, float)):
            # 处理 NaN
            if pd.isna(hour):
                return 0
            hour_int = int(float(hour))
        else:
            return 0
    except (ValueError, TypeError, AttributeError):
        return 0
    
    # 确保小时在有效范围内
    if not (0 <= hour_int <= 23):
        return 0
    
    # 早高峰：6-10点
    if 6 <= hour_int <= 10:
        return 1  # 早高峰
    # 晚高峰：16-20点
    elif 16 <= hour_int <= 20:
        return 2  # 晚高峰
    # 非高峰：其他时间
    else:
        return 0  # 非高峰

def predict_and_plot_timeseries_flow(
    file_path: str,
    predict_date: str,
    algorithm: str = 'knn',
    retrain: bool = False,
    save_path: str = "timeseries_predict_hourly.png",
    mode: str = 'all',
    config: Dict = None,
    model_version: Optional[str] = None,
    model_save_dir: Optional[str] = None,
    flow_type: Optional[str] = None, 
    metric_type: Optional[str] = None,
) -> Dict:
    """
    预测并绘制小时客流（完全参考日客流预测逻辑，支持模型版本管理和多因子，增加早晚高峰因子）

    参数:
        file_path: 文件路径（已废弃，仅为兼容性保留）
        predict_date: 预测日期 (YYYYMMDD)
        algorithm: 预测算法 (默认 'knn')
        retrain: 是否强制重新训练模型
        save_path: 图表保存路径
        mode: 操作模式 ('all', 'train', 'predict')
        config: 配置字典
        model_version: 指定推理模型版本（可选）
        model_save_dir: 模型保存目录（可选）
        flow_type: 流量类型 ('xianwangxianlu', 'chezhan')
        metric_type: 指标类型 ('F_PKLCOUNT', 'F_ENTRANCE', ...)

    返回:
        预测结果字典
    """
    # 1. 版本号管理（参考日预测）
    version = model_version if model_version is not None else get_current_version(config_obj=config, config_path="model_config.yaml")
    if model_save_dir is not None:
        model_dir = model_save_dir
    else:
        model_dir = get_version_dir(version, config_obj=config)

    # 2. 选择预测器
    if algorithm == 'knn':
        predictor = KNNHourlyFlowPredictor(model_dir, version, config)
    else:
        predictor = KNNHourlyFlowPredictor(model_dir, version, config)

    # 3. 读取数据（参考日预测逻辑）
    try:
        if flow_type == 'xianwangxianlu':
            df = read_line_hourly_flow_history(metric_type, None)
        elif flow_type == 'chezhan':
            df = read_station_hourly_flow_history(metric_type, None)
        else:
            df = read_line_hourly_flow_history(metric_type, None)

        if not isinstance(df, pd.DataFrame):
            return {"error": "数据读取失败（非 DataFrame），请检查数据库结构"}
    except Exception as e:
        return {"error": f"数据库读取失败: {e}"}

    # 4. 检查必要字段（参考日预测，增加小时和高峰因子）
    required_cols = {
        'F_DATE',
        'F_HOUR',
        'F_YEAR',
        'F_WEEK',
        'F_HOLIDAYTYPE',
        'F_HOLIDAYDAYS',
        'F_HOLIDAYWHICHDAY',
        'F_DAYOFWEEK',
        'F_LINENO',
        'F_LINENAME',
        'F_KLCOUNT',
        'F_WEATHER',
        'WEATHER_TYPE',
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        return {"error": f"缺少必要列: {','.join(missing)}"}

    # 5. 数据预处理（参考日预测逻辑）
    df = df.dropna(subset=['F_DATE', 'F_HOUR', 'F_KLCOUNT', 'F_LINENO'])
    df['F_DATE'] = df['F_DATE'].astype(str).str.strip()
    df['F_HOUR'] = df['F_HOUR'].astype(str).str.zfill(2)
    df['F_KLCOUNT'] = pd.to_numeric(df['F_KLCOUNT'], errors='coerce').fillna(0)
    df['F_LINENO'] = df['F_LINENO'].astype(str).str.zfill(2)
    
    # 处理天气类型（参考日预测，使用weather_enum）
    df['WEATHER_TYPE'] = df['WEATHER_TYPE'].apply(lambda x: WeatherType.get_weather_by_name(str(x)).value)
    
    # 添加早晚高峰因子
    df['F_RUSH_HOUR_TYPE'] = df['F_HOUR'].apply(calculate_rush_hour_type)

    # 6. 保留因子列（参考日预测，增加F_HOUR和F_RUSH_HOUR_TYPE）
    factor_cols = [
        'F_DATE',
        'F_HOUR',
        'F_YEAR',
        'F_WEEK',
        'F_HOLIDAYTYPE',
        'F_HOLIDAYDAYS',
        'F_HOLIDAYWHICHDAY',
        'F_DAYOFWEEK',
        'F_LINENO',
        'F_LINENAME',
        'F_KLCOUNT',
        'F_WEATHER',
        'WEATHER_TYPE',
        'F_RUSH_HOUR_TYPE',
    ]
    # 确保所有因子列都存在
    for col in factor_cols:
        if col not in df.columns and col not in ['F_KLCOUNT', 'F_LINENAME']:  # F_KLCOUNT和F_LINENAME是必需的，已在前面检查
            if col == 'F_RUSH_HOUR_TYPE':
                df[col] = df['F_HOUR'].apply(calculate_rush_hour_type)
            else:
                df[col] = 0
    df = df[factor_cols]

    # 7. 线路名映射（参考日预测）
    line_name_map = {
        row['F_LINENO']: row['F_LINENAME']
        for _, row in df[['F_LINENO', 'F_LINENAME']].drop_duplicates().iterrows()
    }
    lines = sorted(df['F_LINENO'].unique().tolist())
    predict_result = {}
    prediction_rows = []

    # 8. 打印每条线路的训练数据统计信息（参考日预测）
    print("训练数据统计：")
    for line in lines:
        model_info_path = os.path.join(predictor.model_dir, f"model_info_line_{line}_hourly_v{version}.json")
        need_train = retrain or not os.path.exists(model_info_path)
        if need_train:
            line_data = df[(df['F_LINENO'] == line) & (df['F_DATE'] <= predict_date)].copy()
        else:
            line_data = df[df['F_LINENO'] == line].copy()
        if not line_data.empty:
            min_date = line_data['F_DATE'].min()
            max_date = line_data['F_DATE'].max()
            count = len(line_data)
            line_name = line_name_map.get(line, line)
            print(f"线路: {line} ({line_name})，数据日期: {min_date} ~ {max_date}，共 {count} 条")
        else:
            line_name = line_name_map.get(line, line)
            print(f"线路: {line} ({line_name})，无数据")

    # 9. 针对每条线路进行预测/训练（参考日预测逻辑）
    for line in lines:
        model_info_path = os.path.join(predictor.model_dir, f"model_info_line_{line}_hourly_v{version}.json")
        need_train = retrain or not os.path.exists(model_info_path)
        if need_train:
            line_data = df[(df['F_LINENO'] == line) & (df['F_DATE'] <= predict_date)].copy()
        else:
            line_data = df[df['F_LINENO'] == line].copy()
            
        line_name = line_name_map.get(line, line)
        if line_data.empty:
            predict_result[line] = {
                "algorithm": algorithm,
                "predict_hourly_flow": {f"{h:02d}": 0 for h in range(24)},
                "predict_date": predict_date,
                "error": "此线路无数据"
            }
            continue

        # 10. 训练（参考日预测逻辑）
        if mode in ['all', 'train']:
            if need_train:
                mse, mae, error = predictor.train(line_data, line, model_version=version)
                if error:
                    predict_result[line] = {
                        "algorithm": algorithm,
                        "predict_hourly_flow": {f"{h:02d}": 0 for h in range(24)},
                        "predict_date": predict_date,
                        "error": error
                    }
                    continue
                predictor.save_model_info(
                    line, algorithm, mse, mae, datetime.now().strftime('%Y%m%d'), model_version=version
                )
            elif mode == 'train':
                continue

        # 11. 预测（参考日预测逻辑）
        if mode in ['all', 'predict']:
            # 获取预测日期的节假日等因子信息（参考日预测）
            holiday_features = fetch_holiday_features(predict_date, 1)  # 只获取一天的因子
            if holiday_features is not None and not holiday_features.empty:
                # 处理天气类型（参考日预测）
                holiday_features['WEATHER_TYPE'] = holiday_features['WEATHER_TYPE'].apply(
                    lambda x: WeatherType.get_weather_by_name(str(x)).value
                )
                
                # 为24小时生成因子数据（每个小时都需要）
                hourly_holiday_features = []
                for h in range(24):
                    hour_row = holiday_features.iloc[0].copy()
                    hour_row['F_HOUR'] = f"{h:02d}"
                    hour_row['F_RUSH_HOUR_TYPE'] = calculate_rush_hour_type(h)
                    hourly_holiday_features.append(hour_row)
                holiday_features = pd.DataFrame(hourly_holiday_features)
            else:
                # 如果没有节假日特征，创建默认的24小时因子数据
                holiday_features = pd.DataFrame({
                    'F_DATE': [int(predict_date)] * 24,
                    'F_HOUR': [f"{h:02d}" for h in range(24)],
                    'F_YEAR': [int(predict_date[:4])] * 24,
                    'F_WEEK': [None] * 24,
                    'F_HOLIDAYTYPE': [None] * 24,
                    'F_HOLIDAYDAYS': [None] * 24,
                    'F_HOLIDAYWHICHDAY': [None] * 24,
                    'F_DAYOFWEEK': [None] * 24,
                    'WEATHER_TYPE': [0] * 24,  # 默认晴天
                    'F_RUSH_HOUR_TYPE': [calculate_rush_hour_type(h) for h in range(24)],
                })

            # 预测24小时的客流
            predictions, error = predictor.predict(
                line_data, line, predict_date, 24, 
                model_version=version if model_version else None, 
                factor_df=holiday_features
            )

            if error:
                predict_result[line] = {
                    "algorithm": algorithm,
                    "predict_hourly_flow": {f"{h:02d}": 0 for h in range(24)},
                    "predict_date": predict_date,
                    "error": error
                }
                continue

            # 确保0-5点的预测值为0
            hourly_flow = {}
            for h in range(24):
                if h <= 5:
                    hourly_flow[f"{h:02d}"] = 0
                else:
                    hourly_flow[f"{h:02d}"] = int(predictions[h])
            
            predict_result[line] = {
                "algorithm": algorithm,
                "predict_hourly_flow": hourly_flow,
                "predict_date": predict_date,
                "error": None,
                "line_data": line_data
            }

            # 12. 组织入库数据（参考日预测逻辑，增加因子信息）
            for h in range(24):
                pred_date = predict_date
                id_val = str(uuid.uuid4())
                createtime_int = int(datetime.now().strftime('%Y%m%d'))
                predict_date_int = int(predict_date)
                remarks_int = abs(hash(f"KNN小时预测 {algorithm}")) % (10 ** 8)
                try:
                    line_int = int(line)
                except Exception:
                    line_int = 0

                # 获取当前小时的因子（参考日预测逻辑）
                factor_row = None
                hour_str = f"{h:02d}"
                factor_row_df = df[(df['F_DATE'] == predict_date) & 
                                 (df['F_HOUR'] == hour_str) & 
                                 (df['F_LINENO'] == line)]
                if not factor_row_df.empty:
                    factor_row = factor_row_df.iloc[0]
                elif holiday_features is not None and not holiday_features.empty:
                    # 使用节假日因子信息
                    hour_features = holiday_features[holiday_features['F_HOUR'] == hour_str]
                    if not hour_features.empty:
                        factor_row = hour_features.iloc[0]

                # 安全获取因子值的辅助函数（参考日预测）
                def safe_get_factor(key, default=None):
                    """安全地从 factor_row 获取值，如果不存在则返回默认值"""
                    if factor_row is None:
                        return default
                    try:
                        if isinstance(factor_row, pd.Series):
                            return factor_row.get(key, default)
                        elif isinstance(factor_row, dict):
                            return factor_row.get(key, default)
                        else:
                            return getattr(factor_row, key, default)
                    except (KeyError, AttributeError):
                        return default

                def safe_int(val):
                    """尝试将val转换为int，如果失败则返回None"""
                    try:
                        if pd.isnull(val):
                            return None
                        if isinstance(val, (int, float)) and not pd.isnull(val):
                            return int(val)
                        if isinstance(val, str):
                            val_strip = val.strip()
                            if val_strip.isdigit():
                                return int(val_strip)
                            try:
                                float_val = float(val_strip)
                                return int(float_val)
                            except Exception:
                                return None
                        return int(val)
                    except Exception:
                        return None

                # 0-5点的预测值设为0
                pred_value = 0 if h <= 5 else int(predictions[h])
                
                prediction_rows.append({
                    'ID': id_val,
                    'F_DATE': int(predict_date),
                    'F_HOUR': hour_str,
                    'F_LINENO': line_int,
                    'F_LINENAME': line_name,
                    'F_PKLCOUNT': pred_value,
                    'F_BEFPKLCOUNT': None,
                    'F_PRUTE': None,
                    'CREATETIME': createtime_int,
                    'CREATOR': 'knn_predict',
                    'REMARKS': remarks_int,
                    'PREDICT_DATE': predict_date_int,
                    'PREDICT_WEATHER': None,
                    # 新增因子（参考日预测，增加早晚高峰因子）
                    'F_WEEK': safe_int(safe_get_factor('F_WEEK')),
                    'F_HOLIDAYTYPE': safe_int(safe_get_factor('F_HOLIDAYTYPE')),
                    'F_HOLIDAYDAYS': safe_int(safe_get_factor('F_HOLIDAYDAYS')),
                    'F_HOLIDAYWHICHDAY': safe_int(safe_get_factor('F_HOLIDAYWHICHDAY')),
                    'F_DAYOFWEEK': safe_int(safe_get_factor('F_DAYOFWEEK')),
                    'WEATHER_TYPE': safe_int(safe_get_factor('WEATHER_TYPE')),
                    'F_RUSH_HOUR_TYPE': safe_int(safe_get_factor('F_RUSH_HOUR_TYPE', calculate_rush_hour_type(h))),
                    'F_YEAR': safe_int(safe_get_factor('F_YEAR', int(predict_date[:4]))),
                })

    # 13. 可视化与（可选）入库（参考日预测）
    if mode in ['all', 'predict'] and prediction_rows:
        if flow_type == 'xianwangxianlu':
            upload_xianwangxianlu_hourly_prediction_sample(prediction_rows, metric_type)
        else:
            upload_station_hourly_prediction_sample(prediction_rows, metric_type)
        plot_hourly_predictions(predict_result, line_name_map, predict_date, save_path, flow_type, metric_type)

    return predict_result
