# KNN 模型模块：实现 KNN 模型用于日客流预测
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

class KNNFlowPredictor:
    def __init__(self, model_dir: str, version: str, config: Dict):
        """
        KNN 预测器初始化
        
        参数:
            model_dir: 模型存储目录
            version: 模型版本
            config: 配置字典
        """
        self.model_dir = model_dir
        self.version = version
        self.config = config
        self.models = {}
        self.scalers = {}
        self.model_info = {}
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备 KNN 模型输入数据
        
        参数:
            data: 日客流数据 DataFrame
            
        返回:
            (X, y) 输入和目标数组
        """
        df = data.sort_values('F_DATE').copy()
        df['weekday'] = df['F_DATE'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').weekday())
        X = df[['weekday']].values
        y = df['F_KLCOUNT'].values
        return X, y

    def train(self, line_data: pd.DataFrame, line_no: str, model_version: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        训练 KNN 模型
        
        参数:
            line_data: 线路数据 DataFrame
            line_no: 线路编号
            model_version: 模型版本（可选）
            
        返回:
            (mse, mae, error) 元组
        """
        try:
            n_neighbors = self.config.get("train_params", {}).get("n_neighbors", 5)
            X, y = self.prepare_data(line_data)
            if len(X) < n_neighbors:
                return None, None, "数据量不足"

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            # 支持模型版本号
            version = model_version if model_version is not None else self.version
            model_path = os.path.join(self.model_dir, f"knn_line_{line_no}_daily_v{version}.pkl")
            scaler_path = os.path.join(self.model_dir, f"knn_scaler_line_{line_no}_daily_v{version}.pkl")
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            self.models[line_no] = model
            self.scalers[line_no] = scaler
            return mse, mae, None
        except Exception as e:
            return None, None, str(e)

    def predict(self, line_data: pd.DataFrame, line_no: str, predict_start_date: str, days: int = 15, model_version: Optional[str] = None) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        预测指定线路和日期范围的日客流
        
        参数:
            line_data: 线路数据 DataFrame
            line_no: 线路编号
            predict_start_date: 预测起始日期 (YYYYMMDD)
            days: 预测天数
            model_version: 指定推理模型版本（可选）
            
        返回:
            (预测结果, 错误信息) 元组
        """
        try:
            version = model_version if model_version is not None else self.version
            model_path = os.path.join(self.model_dir, f"knn_line_{line_no}_daily_v{version}.pkl")
            scaler_path = os.path.join(self.model_dir, f"knn_scaler_line_{line_no}_daily_v{version}.pkl")
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return None, "模型文件未找到"

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            predict_dt = datetime.strptime(predict_start_date, '%Y%m%d')
            pred_dates = [predict_dt + timedelta(days=d) for d in range(days)]
            pred_weekdays = np.array([[dt.weekday()] for dt in pred_dates])
            pred_weekdays_scaled = scaler.transform(pred_weekdays)
            predictions = model.predict(pred_weekdays_scaled)
            predictions = np.maximum(predictions, 0)
            return predictions.tolist(), None
        except Exception as e:
            return None, str(e)

    def save_model_info(self, line_no: str, algorithm: str, mse: Optional[float], mae: Optional[float], train_date: str, model_version: Optional[str] = None) -> None:
        """
        保存模型元数据
        
        参数:
            line_no: 线路编号
            algorithm: 算法名称
            mse: 均方误差
            mae: 平均绝对误差
            train_date: 训练日期
            model_version: 模型版本（可选）
        """
        import json
        version = model_version if model_version is not None else self.version
        info = {
            'algorithm': algorithm,
            'mse': float(mse) if mse is not None else None,
            'mae': float(mae) if mae is not None else None,
            'train_date': train_date,
            'line_no': line_no,
            'version': version
        }
        info_path = os.path.join(self.model_dir, f"model_info_line_{line_no}_daily_v{version}.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        self.model_info[line_no] = info