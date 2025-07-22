# 日预测主流程模块：协调日客流预测的完整流程（支持模型版本管理）
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
from db_utils import read_line_daily_flow_history, upload_prediction_sample
from knn_model import KNNFlowPredictor
from config_utils import get_version_dir, get_current_version
from plot_utils import plot_daily_predictions
import os

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
    model_save_dir: Optional[str] = None
) -> Dict:
    """
    预测并绘制日客流（支持模型版本管理）

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
    predictor = KNNFlowPredictor(model_dir, version, config)

    try:
        df = read_line_daily_flow_history()
        if not isinstance(df, pd.DataFrame):
            return {"error": "数据读取失败（非 DataFrame），请检查数据库结构"}
    except Exception as e:
        return {"error": f"数据库读取失败: {e}"}

    required_cols = {'F_DATE', 'F_KLCOUNT', 'F_LINENO', 'F_LINENAME'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        return {"error": f"缺少必要列: {','.join(missing)}"}

    df = df.dropna(subset=['F_DATE', 'F_KLCOUNT', 'F_LINENO'])
    df['F_DATE'] = df['F_DATE'].astype(str).str.strip()
    df['F_KLCOUNT'] = pd.to_numeric(df['F_KLCOUNT'], errors='coerce').fillna(0)
    df['F_LINENO'] = df['F_LINENO'].astype(str).str.zfill(2)

    line_name_map = {row['F_LINENO']: row['F_LINENAME'] for _, row in df[['F_LINENO', 'F_LINENAME']].drop_duplicates().iterrows()}
    lines = sorted(df['F_LINENO'].unique().tolist())
    predict_result = {}
    prediction_rows = []

    for line in lines:
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
                n_neighbors = config.get("train_params", {}).get("n_neighbors", 5)
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
            predictions, error = predictor.predict(line_data, line, predict_start_date, days, model_version=version if model_version else None)
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
                prediction_rows.append({
                    'ID': id_val,
                    'F_DATE': int(pred_date),
                    'F_HOLIDAYTYPE': None,
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
                    'F_TYPE': f_type_int
                })

    if mode in ['all', 'predict'] and prediction_rows:
        upload_prediction_sample(prediction_rows)
        plot_daily_predictions(predict_result, line_name_map, predict_start_date, days, save_path)
        print(predict_result)

    return predict_result