# 日预测主流程模块：协调日客流预测的完整流程（支持模型版本管理）
from pickle import DICT
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
from db_utils import read_line_daily_flow_history, read_station_daily_flow_history, upload_prediction_sample, fetch_holiday_features
from enknn_model import KNNFlowPredictor
# from lightgbm_model import LightGBMFlowPredictor
# # from knn_model import KNNFlowPredictor
# from prophet_model import ProphetFlowPredictor
# from transformer_model import TransformerFlowPredictor
# from xgboost_model import XGBoostFlowPredictor
# from lstm_daily_model import LSTMFlowPredictor
from config_utils import get_version_dir, get_current_version
from plot_utils import plot_daily_predictions
import os

HOLIDAYTYPE_DICT =  {'Dragon Boat Festival': 0, 'Labour Day': 1, 'Mid-autumn Festival': 2, 'National Day': 3, "New year's Day": 4, 'Spring Festival': 5, 'Tomb-sweeping Day': 6}
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
        # 如果已经是int或float且不是nan，直接转
        if isinstance(val, (int, float)) and not pd.isnull(val):
            return int(val)
        # 字符串数字
        if isinstance(val, str):
            val_strip = val.strip()
            if val_strip.isdigit():
                return int(val_strip)
            # 允许float字符串
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
    # 如果weather_str为None或空字符串，默认为晴天
    if weather_str is None or str(weather_str).strip() == "":
        weather_str = "晴"
    else:
        weather_str = str(weather_str).strip()

    def compress_for_metro(t):
        if "转" in t:
            parts = t.split("转")
            # 取转后天气为主
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
    return label_map.get(compressed, 6)  # 默认返回“其他”类别编码6

def map_holidaytype_to_int(val):
    """
    将F_HOLIDAYTYPE字段映射为int，参考HOLIDAYTYPE_DICT
    """
    if pd.isnull(val):
        return -1
    val_str = str(val).strip()
    # 优先查字典
    if val_str in HOLIDAYTYPE_DICT:
        return HOLIDAYTYPE_DICT[val_str]
    # 如果是数字，直接转
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

def predict_and_plot_timeseries_flow_daily(
    file_path: str,
    predict_start_date: str,
    algorithm: str = 'knn',
    retrain: bool = False,
    save_path: str = "timeseries_predict_daily.png",
    mode: str = 'all',
    days: int = 15,
    config: Dict = None,
    model_version: Optional[str] = None,
    model_save_dir: Optional[str] = None,
    flow_type: Optional[str] = None, 
    metric_type: Optional[str] = None,
) -> Dict:
    """
    预测并绘制日客流（支持模型版本管理），预测F_KLCOUNT客流，并加上节假日、星期、年份、周数等因子

    参数:
        file_path: 文件路径（已废弃，仅为兼容性保留）
        predict_start_date: 预测起始日期 (YYYYMMDD)
        algorithm: 预测算法 (默认 'knn')
        retrain: 是否强制重新训练模型
        save_path: 图表保存路径
        mode: 操作模式 ('all', 'train', 'predict')
        days: 预测天数
        config: 配置字典
        model_version: 指定推理模型版本（可选）
        model_save_dir: 模型保存目录（可选）

    返回:
        预测结果字典
    """
    # 1. 版本号管理
    version = model_version if model_version is not None else get_current_version(config_obj=config, config_path="model_config_daily.yaml")
    # 优先使用外部传入的模型保存目录
    if model_save_dir is not None:
        model_dir = model_save_dir
    else:
        model_dir = get_version_dir(version, config_obj=config)

    # 2. 选择预测器
    if algorithm == 'knn':
        predictor = KNNFlowPredictor(model_dir, version, config)
    # elif algorithm == 'Prophet':
    #     predictor = ProphetFlowPredictor(model_dir, version, config)
    # elif algorithm == 'transformer':
    #     predictor = TransformerFlowPredictor(model_dir, version, config)
    # elif algorithm == 'xgboost':
    #     predictor = XGBoostFlowPredictor(model_dir, version, config)
    # elif algorithm == 'lstm':
    #     predictor = LSTMFlowPredictor(model_dir, version, config)
    # elif algorithm == 'lightgbm':
    #     predictor = LightGBMFlowPredictor(model_dir, version, config)
    else:
        predictor = KNNFlowPredictor(model_dir, version, config)

    try:
        if flow_type == 'xianwangxianlu':
            df = read_line_daily_flow_history(metric_type)
        elif flow_type == 'chezhan':
            df = read_station_daily_flow_history(metric_type)
        else:
            df = read_line_daily_flow_history()

        if not isinstance(df, pd.DataFrame):
            return {"error": "数据读取失败（非 DataFrame），请检查数据库结构"}
    except Exception as e:
        return {"error": f"数据库读取失败: {e}"}

    # 需要的因子列（根据db_utils.py和上下文，全部大写）
    required_cols = {
        'F_WEEK',
        'F_DATEFEATURES',
        'F_HOLIDAYTYPE',
        'F_ISHOLIDAY',
        'F_ISNONGLI',
        'F_ISYANGLI',
        'F_NEXTDAY',
        'F_HOLIDAYDAYS',
        'F_HOLIDAYTHDAY',
        'IS_FIRST',
        'F_DATE',
        'F_LINENO',
        'F_LINENAME',
        'F_KLCOUNT',
        'WEATHER_TYPE'
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        return {"error": f"缺少必要列: {','.join(missing)}"}

    # 数据预处理
    df = df.dropna(subset=['F_DATE', 'F_KLCOUNT', 'F_LINENO'])
    df['F_DATE'] = df['F_DATE'].astype(str).str.strip()
    df['F_KLCOUNT'] = pd.to_numeric(df['F_KLCOUNT'], errors='coerce').fillna(0)
    df['F_LINENO'] = df['F_LINENO'].astype(str).str.zfill(2)

    # F_HOLIDAYTYPE字段统一转为int，参考HOLIDAYTYPE_DICT
    df['F_HOLIDAYTYPE'] = df['F_HOLIDAYTYPE'].apply(map_holidaytype_to_int)
    df['WEATHER_TYPE'] = df['WEATHER_TYPE'].apply(weather_to_label)
    df['F_DATEFEATURES'] = df['F_DATEFEATURES'].apply(map_daytype_to_int)

    # 保留因子列
    factor_cols = [
        'F_DATE',
        'F_WEEK',
        'F_DATEFEATURES',
        'F_HOLIDAYTYPE',
        'F_ISHOLIDAY',
        'F_ISNONGLI',
        'F_ISYANGLI',
        'F_NEXTDAY',
        'F_HOLIDAYDAYS',
        'F_HOLIDAYTHDAY',
        'IS_FIRST',
        'F_LINENO',
        'F_LINENAME',
        'F_KLCOUNT',
        'WEATHER_TYPE'
    ]
    df = df[factor_cols]

    # 不再需要字符串映射
    # line_name_map和lines等后续逻辑不变
    line_name_map = {row['F_LINENO']: row['F_LINENAME'] for _, row in df[['F_LINENO', 'F_LINENAME']].drop_duplicates().iterrows()}
    lines = sorted(df['F_LINENO'].unique().tolist())
    predict_result = {}
    prediction_rows = []

    # 打印每条线路的训练数据统计信息
    print("训练数据统计：")
    for line in lines:
        model_info_path = os.path.join(predictor.model_dir, f"model_info_line_{line}_daily_v{version}.json")
        need_train = retrain or not os.path.exists(model_info_path)
        if need_train:
            line_data = df[(df['F_LINENO'] == line) & (df['F_DATE'] <= predict_start_date)].copy()
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

    for line in lines:
        model_info_path = os.path.join(predictor.model_dir, f"model_info_line_{line}_daily_v{version}.json")
        need_train = retrain or not os.path.exists(model_info_path)
        if need_train:
            line_data = df[(df['F_LINENO'] == line) & (df['F_DATE'] <= predict_start_date)].copy()
        else:
            line_data = df[df['F_LINENO'] == line].copy()
        line_name = line_name_map.get(line, line)
        if line_data.empty:
            predict_result[line] = {
                "algorithm": algorithm,
                "predict_daily_flow": {(datetime.strptime(predict_start_date, '%Y%m%d') + timedelta(days=d)).strftime('%Y%m%d'): 0 for d in range(days)},
                "predict_start_date": predict_start_date,
                "error": "此线路无数据"
            }
            continue

        # 2. 训练/推理模型文件路径应考虑版本号
        model_info_path = os.path.join(predictor.model_dir, f"model_info_line_{line}_daily_v{version}.json")
        need_train = retrain or not os.path.exists(model_info_path)

        if mode in ['all', 'train']:
            if need_train:
                n_neighbors = config.get("train_params", {}).get("n_neighbors", 5) if config else 5
                # 训练时传入所有因子
                mse, mae, error = predictor.train(line_data, line, model_version=version)
                if error:
                    predict_result[line] = {
                        "algorithm": algorithm,
                        "predict_daily_flow": {(datetime.strptime(predict_start_date, '%Y%m%d') + timedelta(days=d)).strftime('%Y%m%d'): 0 for d in range(days)},
                        "predict_start_date": predict_start_date,
                        "error": error
                    }
                    continue
                predictor.save_model_info(line, algorithm, mse, mae, datetime.now().strftime('%Y%m%d'), model_version=version)
            elif mode == 'train':
                continue

        if mode in ['all', 'predict']:
            # 3. 推理时优先加载指定版本模型
            # 预测时也传入所有因子
            holiday_features = fetch_holiday_features(predict_start_date, days)
            # 对holiday_features的F_HOLIDAYTYPE也做映射
            if holiday_features is not None and 'F_HOLIDAYTYPE' in holiday_features.columns:
                holiday_features['F_HOLIDAYTYPE'] = holiday_features['F_HOLIDAYTYPE'].apply(map_holidaytype_to_int)
                holiday_features['WEATHER_TYPE'] = holiday_features['WEATHER_TYPE'].apply(weather_to_label)
                holiday_features['F_DATEFEATURES'] = holiday_features['F_DATEFEATURES'].apply(map_daytype_to_int)
            # print(holiday_features)
            predictions, error = predictor.predict(line_data, line, predict_start_date, days, model_version=version if model_version else None, factor_df=holiday_features)
            if error:
                predict_result[line] = {
                    "algorithm": algorithm,
                    "predict_daily_flow": {(datetime.strptime(predict_start_date, '%Y%m%d') + timedelta(days=d)).strftime('%Y%m%d'): 0 for d in range(days)},
                    "predict_start_date": predict_start_date,
                    "error": error
                }
                continue

            daily_flow = {(datetime.strptime(predict_start_date, '%Y%m%d') + timedelta(days=d)).strftime('%Y%m%d'): int(predictions[d]) for d in range(days)}
            predict_result[line] = {
                "algorithm": algorithm,
                "predict_daily_flow": daily_flow,
                "predict_start_date": predict_start_date,
                "error": None,
                "line_data": line_data
            }

            # 预测结果写入数据库时也带上因子
            for d in range(days):
                pred_date = (datetime.strptime(predict_start_date, '%Y%m%d') + timedelta(days=d)).strftime('%Y%m%d')
                id_val = str(uuid.uuid4())
                createtime_int = int(datetime.now().strftime('%Y%m%d'))
                predict_date_int = int(predict_start_date)
                f_type_int = 1
                remarks_int = abs(hash(f"KNN日预测 {algorithm}")) % (10 ** 8)
                try:
                    line_int = int(line)
                except Exception:
                    line_int = 0
                f_lb_val = 1 if line_int == 0 else 0

                # 获取预测日期的因子（如有）
                factor_row = None
                if pred_date in df['F_DATE'].values:
                    factor_row_df = df[(df['F_DATE'] == pred_date) & (df['F_LINENO'] == line)]
                    if not factor_row_df.empty:
                        factor_row = factor_row_df.iloc[0]
                    else:
                        factor_row = None

                # 处理F_HOLIDAYTYPE为int，参考HOLIDAYTYPE_DICT
                f_holidaytype_val = None
                if factor_row is not None:
                    f_holidaytype_val = map_holidaytype_to_int(factor_row['F_HOLIDAYTYPE'])
                else:
                    f_holidaytype_val = None

                prediction_rows.append({
                    'ID': id_val,
                    'F_DATE': safe_int(pred_date),
                    'F_HOLIDAYTYPE': f_holidaytype_val,
                    'F_LB': f_lb_val,
                    'F_LINENO': line_int,
                    'F_LINENAME': line_name,
                    'F_PKLCOUNT': int(predictions[d]),
                    'F_BEFPKLCOUNT': None,
                    'F_PRUTE': None,
                    'F_OTHER': None,
                    'F_WH': None,
                    'F_ZDHD': None,
                    'F_YQZH': None,
                    'F_YXYSMS': None,
                    'CREATETIME': createtime_int,
                    'CREATOR': 'knn_predict',
                    'MODIFYTIME': None,
                    'MODIFIER': None,
                    'REMARKS': remarks_int,
                    'PREDICT_DATE': predict_date_int,
                    'PREDICT_WEATHER': None,
                    'F_TYPE': f_type_int,
                    # 新增因子（全部大写，和db_utils.py一致）
                    'F_WEEK': safe_int(factor_row['F_WEEK']) if factor_row is not None else None,
                    'F_DATEFEATURES': safe_int(factor_row['F_DATEFEATURES']) if factor_row is not None else None,
                    'F_ISHOLIDAY': safe_int(factor_row['F_ISHOLIDAY']) if factor_row is not None else None,
                    'F_ISNONGLI': safe_int(factor_row['F_ISNONGLI']) if factor_row is not None else None,
                    'F_ISYANGLI': safe_int(factor_row['F_ISYANGLI']) if factor_row is not None else None,
                    'F_NEXTDAY': safe_int(factor_row['F_NEXTDAY']) if factor_row is not None else None,
                    'F_HOLIDAYDAYS': safe_int(factor_row['F_HOLIDAYDAYS']) if factor_row is not None else None,
                    'F_HOLIDAYTHDAY': safe_int(factor_row['F_HOLIDAYTHDAY']) if factor_row is not None else None,
                    'IS_FIRST': safe_int(factor_row['IS_FIRST']) if factor_row is not None else None,
                })

    if mode in ['all', 'predict'] and prediction_rows:
        # upload_prediction_sample(prediction_rows)
        plot_daily_predictions(predict_result, line_name_map, predict_start_date, days, save_path, flow_type, metric_type)

    return predict_result

