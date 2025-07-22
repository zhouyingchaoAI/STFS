# 小时预测主流程模块：协调小时客流预测的完整流程
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import os

from db_utils import read_line_hourly_flow_history, insert_hourly_prediction_to_db
from lstm_model import LSTMFlowPredictor
from prophet_model import ProphetFlowPredictor
from config_utils import get_version_dir, get_current_version
from plot_utils import plot_hourly_predictions

def predict_and_plot_timeseries_flow(
    file_path: str,
    predict_date: str,
    algorithm: str = 'lstm',
    retrain: bool = False,
    save_path: str = "timeseries_predict_hourly.png",
    mode: str = 'all',
    config: Dict = None,
    model_version: Optional[str] = None,
    model_save_dir: Optional[str] = None
) -> Dict:
    """
    预测并绘制小时客流（支持模型版本管理）

    参数:
        file_path: 文件路径（已废弃，仅为兼容性保留）
        predict_date: 预测日期 (YYYYMMDD)
        algorithm: 预测算法 ('lstm' 或 'prophet')
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
    if algorithm == 'lstm':
        predictor = LSTMFlowPredictor(model_dir, version, config)
    else:
        predictor = ProphetFlowPredictor(model_dir, version, config)

    # 3. 读取数据
    try:
        df = read_line_hourly_flow_history()
        if not isinstance(df, pd.DataFrame):
            return {"error": "数据读取失败（非 DataFrame），请检查数据库结构"}
    except Exception as e:
        return {"error": f"数据库读取失败: {e}"}

    # 4. 检查必要字段
    required_cols = {'F_DATE', 'F_HOUR', 'F_KLCOUNT', 'F_LINENO', 'F_LINENAME'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        return {"error": f"缺少必要列: {','.join(missing)}"}

    # 5. 数据预处理
    df = df.dropna(subset=['F_DATE', 'F_HOUR', 'F_KLCOUNT', 'F_LINENO'])
    df['F_DATE'] = df['F_DATE'].astype(str).str.strip()
    df['F_HOUR'] = df['F_HOUR'].astype(str).str.zfill(2)
    df['F_KLCOUNT'] = pd.to_numeric(df['F_KLCOUNT'], errors='coerce').fillna(0)
    df['F_LINENO'] = df['F_LINENO'].astype(str).str.zfill(2)

    # 6. 线路名映射
    line_name_map = {
        row['F_LINENO']: row['F_LINENAME']
        for _, row in df[['F_LINENO', 'F_LINENAME']].drop_duplicates().iterrows()
    }
    lines = sorted(df['F_LINENO'].unique().tolist())
    predict_result = {}
    prediction_rows = []

    # 7. 针对每条线路进行预测/训练
    for line in lines:
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

        # 训练/推理模型文件路径应考虑版本号
        model_info_path = os.path.join(predictor.model_dir, f"model_info_line_{line}_hourly_v{version}.json")
        need_train = retrain or not os.path.exists(model_info_path)

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
            if algorithm == 'lstm':
                predictions, error = predictor.predict(
                    line_data, line, predict_date, model_version=version if model_version else None
                )
            else:
                predictions, error = predictor.predict(
                    line, predict_date, model_version=version if model_version else None
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

            # 10. 组织入库数据
            for h in range(24):
                id_val = str(uuid.uuid4())
                createtime_int = int(datetime.now().strftime('%Y%m%d'))
                predict_date_int = int(predict_date)
                remarks_int = abs(hash(f"{algorithm}小时预测")) % (10 ** 8)
                try:
                    line_int = int(line)
                except Exception:
                    line_int = 0
                prediction_rows.append({
                    'ID': id_val,
                    'F_DATE': int(predict_date),
                    'F_HOUR': f"{h:02d}",
                    'F_LINENO': line_int,
                    'F_LINENAME': line_name,
                    'F_PKLCOUNT': int(predictions[h]),
                    'F_BEFPKLCOUNT': None,
                    'F_PRUTE': None,
                    'CREATETIME': createtime_int,
                    'CREATOR': f'{algorithm}_predict',
                    'REMARKS': remarks_int,
                    'PREDICT_DATE': predict_date_int,
                    'PREDICT_WEATHER': None
                })

    # 11. 可视化与（可选）入库
    if mode in ['all', 'predict'] and prediction_rows:
        # 如需入库，取消注释下一行
        insert_hourly_prediction_to_db(prediction_rows)
        plot_hourly_predictions(predict_result, line_name_map, predict_date, save_path)

    return predict_result