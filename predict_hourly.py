# 小时预测主流程模块：协调小时客流预测的完整流程（基于日客流逻辑，增加F_HOUR因子）
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
import os

from db_utils import read_line_hourly_flow_history,read_station_hourly_flow_history, upload_xianwangxianlu_hourly_prediction_sample, upload_station_hourly_prediction_sample, fetch_holiday_features
# from lstm_model import LSTMFlowPredictor
# from prophet_model import ProphetFlowPredictor
# from xgboost_model import XGBoostFlowPredictor
# from enknn_model import KNNFlowPredictor
from hourknn_model import KNNHourlyFlowPredictor
# from lightgbm_model import LightGBMFlowPredictor
# from transformer_model import TransformerFlowPredictor
from config_utils import get_version_dir, get_current_version
from plot_utils import plot_hourly_predictions

# 节假日类型映射字典（与日预测保持一致）
HOLIDAYTYPE_DICT = {
    'Dragon Boat Festival': 0, 
    'Labour Day': 1, 
    'Mid-autumn Festival': 2, 
    'National Day': 3, 
    "New year's Day": 4, 
    'Spring Festival': 5, 
    'Tomb-sweeping Day': 6
}

# 日类型映射字典（与日预测保持一致）
DAYTYPE_DICT = {
    'holiday': 0, 
    'weekday': 1, 
    'weekend': 2
}

def safe_int(val):
    """
    尝试将val转换为int，如果失败则返回None
    """
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

def weather_to_label(weather_str):
    """
    输入天气字符串，输出压缩后的类别数字编码。
    类别编码：
        0: 晴天
        1: 阴天
        2: 雾霾/雾
        3: 小雨
        4: 中到大雨
        5: 雪
        6: 其他
    """
    if weather_str is None or str(weather_str).strip() == "":
        weather_str = "晴"
    else:
        weather_str = str(weather_str).strip()

    def compress_for_metro(t):
        if "转" in t:
            parts = t.split("转")
            main = parts[-1]
            return compress_for_metro(main)
        if " / " in t:
            parts = t.split(" / ")
            def weight(w):
                if w in ["晴", "多云"]:
                    return 1
                if w in ["阴"]:
                    return 2
                if w in ["雾"]:
                    return 3
                if w in ["小雨", "阵雨"]:
                    return 4
                if w in ["中雨", "大雨", "暴雨"]:
                    return 5
                if w in ["小雪", "中雪", "大雪", "冻雨", "雨夹雪"]:
                    return 6
                return 0
            w1, w2 = parts
            return compress_for_metro(w1) if weight(w1) >= weight(w2) else compress_for_metro(w2)
        # 单天气归类
        if t in ["晴", "多云"]:
            return "晴天"
        elif t == "阴":
            return "阴天"
        elif t == "雾":
            return "雾霾/雾"
        elif t in ["小雨", "阵雨"]:
            return "小雨"
        elif t in ["中雨", "大雨", "暴雨"]:
            return "中到大雨"
        elif t in ["小雪", "中雪", "大雪", "冻雨", "雨夹雪"]:
            return "雪"
        else:
            return "其他"
    
    label_map = {
        "晴天": 0,
        "阴天": 1,
        "雾霾/雾": 2,
        "小雨": 3,
        "中到大雨": 4,
        "雪": 5,
        "其他": 6
    }
    
    compressed = compress_for_metro(weather_str)
    return label_map.get(compressed, 6)

def map_holidaytype_to_int(val):
    """
    将F_HOLIDAYTYPE字段映射为int，参考HOLIDAYTYPE_DICT
    """
    if pd.isnull(val):
        return -1
    val_str = str(val).strip()
    if val_str in HOLIDAYTYPE_DICT:
        return HOLIDAYTYPE_DICT[val_str]
    if val_str.replace('.', '', 1).isdigit():
        try:
            return int(float(val_str))
        except Exception:
            return -1
    return -1

def map_daytype_to_int(val):
    """
    将F_DAYTYPE字段映射为int，DAYTYPE_DICT
    """
    if pd.isnull(val):
        return -1
    val_str = str(val).strip()
    if val_str in DAYTYPE_DICT:
        return DAYTYPE_DICT[val_str]
    if val_str.replace('.', '', 1).isdigit():
        try:
            return int(float(val_str))
        except Exception:
            return -1
    return -1

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
    预测并绘制小时客流（基于日客流逻辑，支持模型版本管理和多因子）

    参数:
        file_path: 文件路径（已废弃，仅为兼容性保留）
        predict_date: 预测日期 (YYYYMMDD)
        algorithm: 预测算法 ('knn', 'lstm', 'prophet', 'xgboost', 'lightgbm', 'transformer')
        retrain: 是否强制重新训练模型
        save_path: 图表保存路径
        mode: 操作模式 ('all', 'train', 'predict')
        config: 配置字典
        model_version: 指定推理模型版本（如 "20240601"），可选
        model_save_dir: 模型保存目录（可选）

    返回:
        预测结果字典
    """
    # 1. 版本号管理
    version = model_version if model_version is not None else get_current_version(config_obj=config)
    model_dir = model_save_dir if model_save_dir is not None else get_version_dir(version, config_obj=config)

    # 2. 选择预测器
    if algorithm == 'knn':
        predictor = KNNHourlyFlowPredictor(model_dir, version, config)
    # elif algorithm == 'lstm':
    #     predictor = LSTMFlowPredictor(model_dir, version, config)
    # elif algorithm == 'xgboost':
    #     predictor = XGBoostFlowPredictor(model_dir, version, config)
    # elif algorithm == 'prophet':
    #     predictor = ProphetFlowPredictor(model_dir, version, config)
    # elif algorithm == 'lightgbm':
    #     predictor = LightGBMFlowPredictor(model_dir, version, config)
    # elif algorithm == 'transformer':
    #     predictor = TransformerFlowPredictor(model_dir, version, config)
    else:
        predictor = KNNHourlyFlowPredictor(model_dir, version, config)

    # 3. 读取数据
    try:
        if mode in ['all', 'train']:
            if flow_type == "xianwangxianlu":
                df = read_line_hourly_flow_history(metric_type, None)
            elif flow_type == "chezhan":
                df = read_station_hourly_flow_history(metric_type, None)
            else:
                df = read_station_hourly_flow_history(metric_type, None)
        else:
            if flow_type == "xianwangxianlu":
                df = read_line_hourly_flow_history(metric_type, predict_date, 15)
            elif flow_type == "chezhan":
                df = read_station_hourly_flow_history(metric_type, predict_date, 15)
            else:
                df = read_station_hourly_flow_history(metric_type, predict_date, 15)

        if not isinstance(df, pd.DataFrame):
            return {"error": "数据读取失败（非 DataFrame），请检查数据库结构"}
    except Exception as e:
        return {"error": f"数据库读取失败: {e}"}

    # 4. 检查必要字段（增加小时相关因子）
    required_cols = {
        'F_DATE', 'F_HOUR', 'F_KLCOUNT', 'F_LINENO', 'F_LINENAME',
        'F_WEEK', 'F_DATEFEATURES', 'F_HOLIDAYTYPE', 'F_ISHOLIDAY',
        'F_ISNONGLI', 'F_ISYANGLI', 'F_NEXTDAY', 'F_HOLIDAYDAYS',
        'F_HOLIDAYTHDAY', 'IS_FIRST', 'WEATHER_TYPE'
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        return {"error": f"缺少必要列: {','.join(missing)}"}

    # 5. 数据预处理
    df = df.dropna(subset=['F_DATE', 'F_HOUR', 'F_KLCOUNT', 'F_LINENO'])
    df['F_DATE'] = df['F_DATE'].astype(str).str.strip()
    df['F_HOUR'] = df['F_HOUR'].astype(str).str.zfill(2)
    df['F_KLCOUNT'] = pd.to_numeric(df['F_KLCOUNT'], errors='coerce').fillna(0)
    df['F_LINENO'] = df['F_LINENO'].astype(str).str.zfill(2)
    
    # 处理节假日类型和天气类型
    df['F_HOLIDAYTYPE'] = df['F_HOLIDAYTYPE'].apply(map_holidaytype_to_int)
    df['WEATHER_TYPE'] = df['WEATHER_TYPE'].apply(weather_to_label)
    df['F_DATEFEATURES'] = df['F_DATEFEATURES'].apply(map_daytype_to_int)
    

    # 保留因子列（增加F_HOUR）
    factor_cols = [
        'F_DATE', 'F_HOUR', 'F_WEEK', 'F_DATEFEATURES', 'F_HOLIDAYTYPE',
        'F_ISHOLIDAY', 'F_ISNONGLI', 'F_ISYANGLI', 'F_NEXTDAY',
        'F_HOLIDAYDAYS', 'F_HOLIDAYTHDAY', 'IS_FIRST', 'F_LINENO',
        'F_LINENAME', 'F_KLCOUNT', 'WEATHER_TYPE'
    ]
    df = df[factor_cols]

    # 6. 线路名映射
    line_name_map = {
        row['F_LINENO']: row['F_LINENAME']
        for _, row in df[['F_LINENO', 'F_LINENAME']].drop_duplicates().iterrows()
    }
    lines = sorted(df['F_LINENO'].unique().tolist())
    predict_result = {}
    prediction_rows = []

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

    # 7. 针对每条线路进行预测/训练
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

        # 8. 训练
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

        # 9. 预测
        if mode in ['all', 'predict']:
            # 获取预测日期的节假日等因子信息
            holiday_features = fetch_holiday_features(predict_date, 1)  # 只获取一天的因子
            if holiday_features is not None and 'F_HOLIDAYTYPE' in holiday_features.columns:
                holiday_features['F_HOLIDAYTYPE'] = holiday_features['F_HOLIDAYTYPE'].apply(map_holidaytype_to_int)
                holiday_features['WEATHER_TYPE'] = holiday_features['WEATHER_TYPE'].apply(weather_to_label)
                holiday_features['F_DATEFEATURES'] = holiday_features['F_DATEFEATURES'].apply(map_daytype_to_int)
            # 预测24小时的客流
            if algorithm in ['lstm', 'transformer']:  # 对于时序模型
                predictions, error = predictor.predict(
                    line_data, line, predict_date, model_version=version if model_version else None
                )
            else:  # 对于其他模型（knn, xgboost, lightgbm等）
                predictions, error = predictor.predict(
                    line_data, line, predict_date, 24, model_version=version if model_version else None, 
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

            hourly_flow = {f"{h:02d}": int(predictions[h]) for h in range(24)}
            predict_result[line] = {
                "algorithm": algorithm,
                "predict_hourly_flow": hourly_flow,
                "predict_date": predict_date,
                "error": None,
                "line_data": line_data
            }

            # 10. 组织入库数据（增加因子信息）
            for h in range(24):
                id_val = str(uuid.uuid4())
                createtime_int = int(datetime.now().strftime('%Y%m%d'))
                predict_date_int = int(predict_date)
                remarks_int = abs(hash(f"{algorithm}小时预测")) % (10 ** 8)
                try:
                    line_int = int(line)
                except Exception:
                    line_int = 0

                # 获取当前小时的因子（如有历史数据）
                factor_row = None
                hour_str = f"{h:02d}"
                factor_row_df = df[(df['F_DATE'] == predict_date) & 
                                 (df['F_HOUR'] == hour_str) & 
                                 (df['F_LINENO'] == line)]
                if not factor_row_df.empty:
                    factor_row = factor_row_df.iloc[0]
                elif holiday_features is not None and not holiday_features.empty:
                    # 使用节假日因子信息，但需要添加小时信息
                    factor_row = holiday_features.iloc[0].copy()
                    factor_row['F_HOUR'] = hour_str

                prediction_rows.append({
                    'ID': id_val,
                    'F_DATE': int(predict_date),
                    'F_HOUR': hour_str,
                    'F_LINENO': line_int,
                    'F_LINENAME': line_name,
                    'F_PKLCOUNT': int(predictions[h]),
                    'F_BEFPKLCOUNT': None,
                    'F_PRUTE': None,
                    'CREATETIME': createtime_int,
                    'CREATOR': f'{algorithm}_predict',
                    'REMARKS': remarks_int,
                    'PREDICT_DATE': predict_date_int,
                    'PREDICT_WEATHER': None,
                    # 新增因子（与日预测保持一致的结构）
                    'F_WEEK': safe_int(factor_row['F_WEEK']) if factor_row is not None else None,
                    'F_DATEFEATURES': safe_int(factor_row['F_DATEFEATURES']) if factor_row is not None else None,
                    'F_HOLIDAYTYPE': safe_int(factor_row['F_HOLIDAYTYPE']) if factor_row is not None else None,
                    'F_ISHOLIDAY': safe_int(factor_row['F_ISHOLIDAY']) if factor_row is not None else None,
                    'F_ISNONGLI': safe_int(factor_row['F_ISNONGLI']) if factor_row is not None else None,
                    'F_ISYANGLI': safe_int(factor_row['F_ISYANGLI']) if factor_row is not None else None,
                    'F_NEXTDAY': safe_int(factor_row['F_NEXTDAY']) if factor_row is not None else None,
                    'F_HOLIDAYDAYS': safe_int(factor_row['F_HOLIDAYDAYS']) if factor_row is not None else None,
                    'F_HOLIDAYTHDAY': safe_int(factor_row['F_HOLIDAYTHDAY']) if factor_row is not None else None,
                    'IS_FIRST': safe_int(factor_row['IS_FIRST']) if factor_row is not None else None,
                    'WEATHER_TYPE': safe_int(factor_row['WEATHER_TYPE']) if factor_row is not None else None,
                })

    # 11. 可视化与（可选）入库
    if mode in ['all', 'predict'] and prediction_rows:
        # 如需入库，取消注释下一行
        if flow_type == 'xianwangxianlu':
            upload_xianwangxianlu_hourly_prediction_sample(prediction_rows, metric_type)
        else:
            print("@@@@@@@@@@@@@")
            upload_station_hourly_prediction_sample(prediction_rows, metric_type)
        plot_hourly_predictions(predict_result, line_name_map, predict_date, save_path, flow_type, metric_type)

    return predict_result