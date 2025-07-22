# Prophet 模型模块：实现 Prophet 模型用于小时客流预测
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

class ProphetFlowPredictor:
    def __init__(self, model_dir: str, version: str, config: Dict):
        """
        Prophet 预测器初始化
        
        参数:
            model_dir: 模型存储目录
            version: 模型版本
            config: 配置字典
        """
        self.model_dir = model_dir
        self.version = version
        self.config = config
        self.models = {}
        self.model_info = {}
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备 Prophet 模型输入数据
        
        参数:
            data: 小时客流数据 DataFrame
            
        返回:
            Prophet 格式的 DataFrame
        """
        prophet_data = []
        for _, row in data.iterrows():
            date_str = row['F_DATE']
            hour = int(row['F_HOUR'])
            flow = row['F_KLCOUNT']
            dt = datetime.strptime(str(date_str), '%Y%m%d') + timedelta(hours=hour)
            prophet_data.append({'ds': dt, 'y': flow})
        df_prophet = pd.DataFrame(prophet_data).sort_values('ds')
        return df_prophet

    def train(self, line_data: pd.DataFrame, line_no: str, model_version: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        训练 Prophet 模型
        
        参数:
            line_data: 线路数据 DataFrame
            line_no: 线路编号
            model_version: 指定模型版本（可选）
            
        返回:
            (mse, mae, error) 元组
        """
        try:
            df_prophet = self.prepare_data(line_data)
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.8
            )
            model.fit(df_prophet)
            # 只训练，不做未来预测
            if len(df_prophet) >= 24:
                # 取最后24小时的真实值和模型拟合值（in-sample）
                actual = df_prophet.tail(24)['y'].values
                fitted = model.predict(df_prophet.tail(24))['yhat'].values
                mse = mean_squared_error(actual, fitted)
                mae = mean_absolute_error(actual, fitted)
            else:
                mse = mae = None

            # 模型版本管理
            version = model_version if model_version is not None else self.version
            model_path = os.path.join(self.model_dir, f"prophet_line_{line_no}_v{version}.pkl")
            joblib.dump(model, model_path)
            self.models[line_no] = model
            return mse, mae, None
        except Exception as e:
            return None, None, str(e)

    def predict(self, line_no: str, predict_date: str, model_version: Optional[str] = None) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        预测指定线路和日期的小时客流
        
        参数:
            line_no: 线路编号
            predict_date: 预测日期 (YYYYMMDD)
            model_version: 指定推理模型版本（可选）
            
        返回:
            (预测结果, 错误信息) 元组
        """
        try:
            # 模型版本管理
            version = model_version if model_version is not None else self.version
            model_path = os.path.join(self.model_dir, f"prophet_line_{line_no}_v{version}.pkl")
            if not os.path.exists(model_path):
                return None, f"模型文件未找到: {model_path}"
            model = joblib.load(model_path)
            predict_dt = datetime.strptime(str(predict_date), '%Y%m%d')
            # 生成24小时的预测时间点
            future_dates = [predict_dt + timedelta(hours=h) for h in range(24)]
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = model.predict(future_df)
            predictions = forecast['yhat'].values
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
            model_version: 指定模型版本（可选）
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
        info_path = os.path.join(self.model_dir, f"model_info_line_{line_no}_v{version}.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        self.model_info[line_no] = info