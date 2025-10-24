# Prophet 模型模块：实现 Prophet 模型用于日客流预测（支持因子配置，各因子影响权重可配置）
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

# 默认因子配置，参考 knn_model.py
DEFAULT_PROPHET_FACTORS = [
    'weekday', 'f_HolidayType'
]

# 节假日相关因子列表
HOLIDAY_FACTORS = ['f_HolidayType', 'f_HolidayDays', 'f_HolidayWhichDay']

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
        # 支持因子配置
        self.factors = self.config.get("factors", DEFAULT_PROPHET_FACTORS)

        # 各因子权重配置（优先使用 config['factor_weights']，否则默认1.0，节假日因子可单独配置）
        self.factor_weights = {}
        config_factor_weights = self.config.get("factor_weights", {})
        self.holiday_factor_weight = self.config.get("holiday_factor_weight", 3.0)
        for f in self.factors:
            if f in config_factor_weights:
                self.factor_weights[f] = config_factor_weights[f]
            elif f in HOLIDAY_FACTORS:
                self.factor_weights[f] = self.holiday_factor_weight
            else:
                self.factor_weights[f] = 1.0

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备 Prophet 模型输入数据（日客流，支持因子）

        参数:
            data: 日客流数据 DataFrame

        返回:
            Prophet 格式的 DataFrame，含因子
        """
        df = data.sort_values('F_DATE').copy()
        # 自动添加 weekday 因子（如果配置中有）
        if 'weekday' in self.factors:
            df['weekday'] = df['F_DATE'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').weekday())
        # 兼容性处理，若有缺失则填充为0，并应用因子权重
        for col in self.factors:
            if col == 'weekday':
                continue  # 已处理
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            weight = self.factor_weights.get(col, 1.0)
            df[col] = df[col] * weight
        # Prophet 需要 ds, y
        df['ds'] = df['F_DATE'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
        df['y'] = pd.to_numeric(df['F_KLCOUNT'], errors='coerce').fillna(0)
        # 对y做log1p变换
        df['y_trans'] = np.log1p(df['y'])
        # 只保留 Prophet 需要的列和因子
        cols = ['ds', 'y_trans'] + [f for f in self.factors if f != 'weekday']
        if 'weekday' in self.factors:
            cols.append('weekday')
        df_prophet = df[cols]
        return df_prophet

    def train(self, line_data: pd.DataFrame, line_no: str, model_version: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        训练 Prophet 模型

        参数:
            line_data: 线路数据 DataFrame
            line_no: 线路编号
            model_version: 模型版本（可选）

        返回:
            (mse, mae, error) 元组
        """
        try:
            df_prophet = self.prepare_data(line_data)
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.8
            )
            # 添加回归因子，支持各因子prior_scale权重配置
            for col in self.factors:
                prior_scale = self.factor_weights.get(col, 1.0)
                model.add_regressor(col, prior_scale=prior_scale)
            # 用log1p变换后的y训练
            model.fit(df_prophet.rename(columns={'y_trans': 'y'}))
            if len(df_prophet) >= 15:
                actual = df_prophet.tail(15)['y_trans'].values
                fitted = model.predict(df_prophet.tail(15).rename(columns={'y_trans': 'y'}))['yhat'].values
                actual_inv = np.expm1(actual)
                fitted_inv = np.expm1(fitted)
                mse = mean_squared_error(actual_inv, fitted_inv)
                mae = mean_absolute_error(actual_inv, fitted_inv)
            else:
                mse = mae = None
            version = model_version if model_version is not None else self.version
            model_path = os.path.join(self.model_dir, f"prophet_line_{line_no}_daily_v{version}.pkl")
            joblib.dump(model, model_path)
            self.models[line_no] = model
            return mse, mae, None
        except Exception as e:
            return None, None, str(e)

    
    def predict(self, line_data: pd.DataFrame, line_no: str, predict_start_date: str, days: int = 15, model_version: Optional[str] = None) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        预测指定线路和日期范围的日客流（支持因子）

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
            model_path = os.path.join(self.model_dir, f"prophet_line_{line_no}_daily_v{version}.pkl")
            if not os.path.exists(model_path):
                # 返回全0预测和错误信息，避免None导致外部出错
                return [0.0] * days, f"模型文件未找到: {model_path}"
            model = joblib.load(model_path)
            predict_dt = datetime.strptime(predict_start_date, '%Y%m%d')
            pred_dates = [predict_dt + timedelta(days=d) for d in range(days)]
            future_df = pd.DataFrame({'ds': pred_dates})
            # 预测时构造所有配置的因子
            last_row = line_data.sort_values('F_DATE').iloc[-1] if not line_data.empty else None
            for col in self.factors:
                weight = self.factor_weights.get(col, 1.0)
                if col == 'weekday':
                    future_df['weekday'] = [dt.weekday() * weight for dt in pred_dates]
                else:
                    if last_row is not None and col in last_row:
                        val = last_row[col]
                    else:
                        val = 0
                    val = pd.to_numeric(val, errors='coerce').fillna(0)
                    val = val * weight
                    future_df[col] = [val] * days
                    future_df[col] = pd.to_numeric(future_df[col], errors='coerce').fillna(0)
            forecast = model.predict(future_df)
            predictions = np.expm1(forecast['yhat'].values)
            predictions = np.maximum(predictions, 0)
            # 如果结果为None或长度不对，返回全0和错误
            if predictions is None or len(predictions) != days:
                return [0.0] * days, "Prophet推理结果无效"
            return predictions.tolist(), None
        except Exception as e:
            # 返回全0预测和错误信息，避免None导致外部出错
            return [0.0] * days, str(e)

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
            'version': version,
            'factors': self.factors
        }
        info_path = os.path.join(self.model_dir, f"model_info_line_{line_no}_daily_v{version}.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        self.model_info[line_no] = info