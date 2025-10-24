# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
import json
from sklearn.model_selection import GridSearchCV, cross_val_score  # 新增: 用于网格搜索和交叉验证

def get_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger()

DEFAULT_FACTORS = [
    'F_WEEK', 'F_DATEFEATURES', 'F_HOLIDAYTYPE', 'F_ISHOLIDAY',
    'F_ISNONGLI', 'F_ISYANGLI', 'F_NEXTDAY', 'F_HOLIDAYDAYS',
    'F_HOLIDAYTHDAY', 'IS_FIRST', 'WEATHER_TYPE'
]

class XGBoostFlowPredictor:
    """
    XGBoost流量预测器（融合历史同期+偏差与滚动预测）
    1. 训练阶段：训练XGBoost回归模型，特征包括原有因子和简单的lag/rolling特征。
    2. 预测阶段：优先采用"去年同期+偏移"法，缺失时用XGBoost模型预测，最终结果为二者加权融合。
    3. 支持滚动预测：每预测一天，将预测结果补入历史，下一天预测时可用到最新的lag/rolling特征。
    """
    def __init__(self, model_dir: str, version: str, config: Dict):
        self.model_dir = model_dir
        self.version = version
        self.config = config or {}
        self.model_info = {}

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            logger.info(f"创建模型目录: {self.model_dir}")

        self.factors = self.config.get("factors", DEFAULT_FACTORS)
        self.offset_weight = float(self.config.get("offset_weight", 1.0))
        self.model_weight = float(self.config.get("model_weight", 0.1))
        self.min_train_samples = int(self.config.get("min_train_samples", 10))
        self.enable_rolling_predict = bool(self.config.get("enable_rolling_predict", True))
        logger.info(f"初始化XGBoost预测器 v{self.version} factors={len(self.factors)} offset_weight={self.offset_weight} model_weight={self.model_weight} enable_rolling_predict={self.enable_rolling_predict}")

    def _ensure_numeric_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """确保指定列为数值类型，修复数据类型问题，并处理异常值"""
        df = data.copy()
        for col in columns:
            if col not in df.columns:
                df[col] = 0.0
                logger.warning(f"因子 {col} 不存在，填充为0")
            else:
                # 强制转换为数值类型，无法转换的设为0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                # 异常值处理 - clip到历史1-99百分位，避免极端值影响
                if col == 'F_KLCOUNT':
                    lower, upper = df[col].quantile([0.01, 0.99])
                    df[col] = df[col].clip(lower, upper)
        return df

    def _get_hist_map(self, hist: pd.DataFrame) -> Dict[str, float]:
        """优化历史数据映射，确保数据类型一致性"""
        hist_map = {}
        for idx, row in hist.iterrows():
            date_str = pd.to_datetime(row['F_DATE']).strftime('%Y%m%d')
            flow_val = pd.to_numeric(row['F_KLCOUNT'], errors='coerce')
            if pd.isna(flow_val):
                flow_val = 0.0
            hist_map[date_str] = float(flow_val)
        return hist_map

    def _compute_derived_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """计算衍生特征，修复数据类型问题，扩展更多lag/rolling以捕捉波动"""
        df = df.sort_values('F_DATE').copy()
        df['F_DATE'] = pd.to_datetime(df['F_DATE'], format='%Y%m%d')
        
        # 确保基础因子存在且为数值类型
        for col in self.factors:
            if col not in df.columns:
                df[col] = 0.0
        df = self._ensure_numeric_data(df, self.factors)
        
        # 计算衍生特征，确保数据类型
        df['F_KLCOUNT'] = pd.to_numeric(df['F_KLCOUNT'], errors='coerce').fillna(0.0)
        # 扩展lag特征 (短期1天、2天，中期14天)
        df['lag_1'] = df['F_KLCOUNT'].shift(1).astype(np.float64)
        df['lag_2'] = df['F_KLCOUNT'].shift(2).astype(np.float64)
        df['lag_7'] = df['F_KLCOUNT'].shift(7).astype(np.float64)
        df['lag_14'] = df['F_KLCOUNT'].shift(14).astype(np.float64)
        df['lag_365'] = df['F_KLCOUNT'].shift(365).astype(np.float64)
        # 扩展rolling特征 (短期3天均值，标准差捕捉波动)
        df['roll_mean_3'] = df['F_KLCOUNT'].rolling(window=3, min_periods=1).mean().shift(1).astype(np.float64)
        df['roll_mean_7'] = df['F_KLCOUNT'].rolling(window=7, min_periods=1).mean().shift(1).astype(np.float64)
        df['roll_mean_30'] = df['F_KLCOUNT'].rolling(window=30, min_periods=1).mean().shift(1).astype(np.float64)
        df['roll_std_7'] = df['F_KLCOUNT'].rolling(window=7, min_periods=1).std().shift(1).astype(np.float64)
        
        extended_factors = self.factors + ['lag_1', 'lag_2', 'lag_7', 'lag_14', 'lag_365', 'roll_mean_3', 'roll_mean_7', 'roll_mean_30', 'roll_std_7']
        df = self._ensure_numeric_data(df, extended_factors)
        
        return df, extended_factors

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """数据准备，强化数据类型检查"""
        if data is None or data.empty:
            raise ValueError("输入数据为空")
        
        required_cols = ['F_DATE', 'F_KLCOUNT']
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")
        
        df = data.sort_values('F_DATE').copy()
        df['F_DATE'] = pd.to_datetime(df['F_DATE'], format='%Y%m%d')
        
        # 添加星期几特征
        if 'F_WEEK' in self.factors:
            df['F_WEEK'] = df['F_DATE'].dt.weekday.astype(np.float64)
        
        df, extended_factors = self._compute_derived_features(df)
        
        # 构建特征矩阵，强制转换数据类型
        X = df[extended_factors].values.astype(np.float64)
        y = df['F_KLCOUNT'].values.astype(np.float64)
        
        # 数据有效性检查
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("特征矩阵包含无效值，进行填充处理")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.any(y < 0):
            logger.warning("目标变量包含负值，将负值设为0")
            y = np.maximum(y, 0.0)
        
        return X, y, extended_factors

    def train(self, line_data: pd.DataFrame, line_no: str, model_version: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        logger.info(f"开始训练线路 {line_no}")
        try:
            X, y, extended_factors = self.prepare_data(line_data)
            if len(X) < self.min_train_samples:
                error_msg = f"样本数过少: {len(X)}"
                logger.error(error_msg)
                return None, None, error_msg

            import xgboost as xgb
            from sklearn.model_selection import GridSearchCV

            # 网格搜索最佳参数，使用交叉验证
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                enable_categorical=False
            )
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_mse = -grid_search.best_score_

            # 计算最终评估指标
            y_pred = best_model.predict(X)
            mse = float(np.mean((y - y_pred) ** 2))
            mae = float(np.mean(np.abs(y - y_pred)))
            
            # 计算MAPE
            mask = y > 0
            mape = float(np.mean(np.abs((y[mask] - y_pred[mask]) / (y[mask] + 1e-6))) * 100) if mask.sum() > 0 else float('inf')

            version = model_version or self.version
            model_path = os.path.join(self.model_dir, f"xgb_line_{line_no}_daily_v{version}.pkl")
            joblib.dump(best_model, model_path)

            info = {
                'algorithm': 'xgboost',
                'train_date': datetime.now().strftime('%Y%m%d%H%M%S'),
                'line_no': line_no,
                'version': version,
                'metrics': {'mse': mse, 'mae': mae, 'mape': mape},
                'factors': self.factors,
                'extended_factors': extended_factors,
                'config': self.config,
                'best_params': best_params
            }
            info_path = os.path.join(self.model_dir, f"model_info_line_{line_no}_daily_v{version}.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            self.model_info[line_no] = info

            logger.info(f"训练完成 - 最佳参数={best_params}, MSE={mse:.2f}, MAE={mae:.2f}")
            return mse, mae, None

        except Exception as e:
            error_msg = f"训练过程出错: {e}"
            logger.error(error_msg, exc_info=True)
            return None, None, error_msg

    def predict(self, line_data: pd.DataFrame, line_no: str, predict_start_date: str,
                days: int = 15, model_version: Optional[str] = None,
                factor_df: Optional[pd.DataFrame] = None,
                rolling: bool = True) -> Tuple[Optional[List[float]], Optional[str]]:
        if rolling and not self.enable_rolling_predict:
            logger.warning("当前配置已禁用滚动预测，将自动切换为普通批量预测。")
            rolling = False

        logger.info(f"开始预测线路 {line_no} 起始 {predict_start_date} 天数 {days} 滚动预测: {rolling}")
        try:
            version = model_version or self.version
            model_path = os.path.join(self.model_dir, f"xgb_line_{line_no}_daily_v{version}.pkl")
            info_path = os.path.join(self.model_dir, f"model_info_line_{line_no}_daily_v{version}.json")

            if not os.path.exists(model_path) or not os.path.exists(info_path):
                return None, f"模型或信息文件不存在: {model_path} / {info_path}"

            model = joblib.load(model_path)
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)

            extended_factors = info.get('extended_factors', self.factors)
            predict_dt = datetime.strptime(predict_start_date, '%Y%m%d')
            pred_dates = [predict_dt + timedelta(days=i) for i in range(days)]

            # 滚动预测逻辑
            if rolling:
                if line_data is None or line_data.empty:
                    return None, "无历史数据，无法滚动预测"
                
                hist = line_data.sort_values('F_DATE').copy()
                hist['F_DATE'] = pd.to_datetime(hist['F_DATE'], format='%Y%m%d')
                hist = self._ensure_numeric_data(hist, ['F_KLCOUNT'] + self.factors)
                
                predictions_final = []
                for i, d in enumerate(pred_dates):
                    pred_df = pd.DataFrame(index=[0])
                    for col in self.factors:
                        if col == 'F_WEEK':
                            pred_df[col] = [float(d.weekday())]
                        elif col in hist.columns:
                            last_val = pd.to_numeric(hist.iloc[-1][col], errors='coerce')
                            if pd.isna(last_val):
                                last_val = 0.0
                            pred_df[col] = [float(last_val)]
                        else:
                            pred_df[col] = [0.0]
                    
                    hist_map = self._get_hist_map(hist)
                    for lag, lag_col in [(1, 'lag_1'), (2, 'lag_2'), (7, 'lag_7'), (14, 'lag_14'), (365, 'lag_365')]:
                        target = d - timedelta(days=lag)
                        val = hist_map.get(target.strftime('%Y%m%d'), np.nan)
                        if np.isnan(val):
                            recent_vals = [v for k, v in hist_map.items() if k <= (d - timedelta(days=1)).strftime('%Y%m%d')]
                            val = np.nanmean(recent_vals) if recent_vals else 0.0
                        pred_df[lag_col] = [float(val)]
                    
                    if len(hist) > 0:
                        last_kl = hist['F_KLCOUNT'].copy()
                        last_window_3 = float(last_kl.ewm(alpha=0.3).mean().iloc[-1]) if len(last_kl) >= 3 else float(last_kl.mean())
                        last_window_7 = float(last_kl.ewm(alpha=0.3).mean().iloc[-1]) if len(last_kl) >= 7 else float(last_kl.mean())
                        last_window_30 = float(last_kl.ewm(alpha=0.3).mean().iloc[-1]) if len(last_kl) >= 30 else float(last_kl.mean())
                        last_std_7 = float(last_kl.rolling(7, min_periods=1).std().iloc[-1]) if len(last_kl) >= 7 else 0.0
                    else:
                        last_window_3 = last_window_7 = last_window_30 = last_std_7 = 0.0
                    
                    pred_df['roll_mean_3'] = [last_window_3]
                    pred_df['roll_mean_7'] = [last_window_7]
                    pred_df['roll_mean_30'] = [last_window_30]
                    pred_df['roll_std_7'] = [last_std_7]
                    
                    pred_df = self._ensure_numeric_data(pred_df, extended_factors)

                    last_year_date = (d - timedelta(days=365)).strftime('%Y%m%d')
                    base = hist_map.get(last_year_date, np.nan)
                    offsets_this = []
                    offsets_last = []
                    for j in range(1, 8):
                        t_this = (d - timedelta(days=j)).strftime('%Y%m%d')
                        t_last = (d - timedelta(days=j + 365)).strftime('%Y%m%d')
                        offsets_this.append(hist_map.get(t_this, np.nan))
                        offsets_last.append(hist_map.get(t_last, np.nan))
                    valid_this = [v for v in offsets_this if not np.isnan(v)]
                    valid_last = [v for v in offsets_last if not np.isnan(v)]
                    if len(valid_this) > 0 and len(valid_last) > 0:
                        mean_this = np.mean(valid_this)
                        mean_last = np.mean(valid_last)
                        abs_offset_per_day = float(mean_this - mean_last)
                        ratio = float(mean_this / mean_last) if mean_last > 0 else 1.0
                    else:
                        abs_offset_per_day = 0.0
                        ratio = 1.0
                    
                    aux_add = (base + abs_offset_per_day) if not np.isnan(base) else np.nan
                    aux_mul = (base * ratio) if not np.isnan(base) else np.nan
                    
                    X_pred = pred_df[extended_factors].values.astype(np.float64)
                    pred_model = float(model.predict(X_pred)[0])
                    
                    if not np.isnan(aux_add):
                        aux = (aux_add + aux_mul) / 2 if not np.isnan(aux_mul) else aux_add
                        aux = float(aux)
                        
                        local_offset_weight = self.offset_weight * 1.2 if len(valid_this) >= 7 else self.offset_weight * 0.8
                        local_model_weight = self.model_weight * (2 - local_offset_weight / self.offset_weight)
                        total_weight = local_offset_weight + local_model_weight
                        
                        if total_weight > 0:
                            offset_ratio = local_offset_weight / total_weight
                            model_ratio = local_model_weight / total_weight
                            pred = max(offset_ratio * aux + model_ratio * pred_model, 0.0)
                        else:
                            pred = max((aux + pred_model) / 2, 0.0)
                    else:
                        pred = max(pred_model, 0.0)
                    
                    predictions_final.append(float(pred))
                    
                    new_row = {col: float(pred_df.iloc[0][col]) for col in pred_df.columns}
                    new_row['F_DATE'] = d
                    new_row['F_KLCOUNT'] = float(pred)
                    hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
                
                zero_cnt = (np.array(predictions_final) == 0).sum()
                if zero_cnt > 0:
                    logger.warning(f"滚动预测中存在0值 {zero_cnt}/{len(predictions_final)}")
                
                return predictions_final, None

            # 非滚动预测逻辑
            if factor_df is not None:
                if factor_df.shape[0] != days:
                    return None, f"factor_df 行数({factor_df.shape[0]})与预测天数({days})不一致"
                
                pred_df = factor_df.copy()
                for col in extended_factors:
                    if col not in pred_df.columns:
                        pred_df[col] = 0.0
                pred_df = self._ensure_numeric_data(pred_df, extended_factors)
            else:
                if line_data is None or line_data.empty:
                    return None, "无历史数据且未提供 factor_df，无法预测"
                
                hist = line_data.sort_values('F_DATE').copy()
                hist['F_DATE'] = pd.to_datetime(hist['F_DATE'], format='%Y%m%d')
                hist = self._ensure_numeric_data(hist, ['F_KLCOUNT'] + self.factors)
                
                pred_df = pd.DataFrame(index=range(days))
                for col in self.factors:
                    if col == 'F_WEEK':
                        pred_df[col] = [float(d.weekday()) for d in pred_dates]
                    elif col in hist.columns:
                        last_val = pd.to_numeric(hist.iloc[-1][col], errors='coerce')
                        if pd.isna(last_val):
                            last_val = 0.0
                        pred_df[col] = [float(last_val)] * days
                    else:
                        pred_df[col] = [0.0] * days
                
                hist_map = self._get_hist_map(hist)
                for lag, lag_col in [(1, 'lag_1'), (2, 'lag_2'), (7, 'lag_7'), (14, 'lag_14'), (365, 'lag_365')]:
                    values = []
                    for d in pred_dates:
                        target = d - timedelta(days=lag)
                        val = hist_map.get(target.strftime('%Y%m%d'), np.nan)
                        if np.isnan(val):
                            recent_vals = [v for k, v in hist_map.items() if k <= (d - timedelta(days=1)).strftime('%Y%m%d')]
                            val = np.nanmean(recent_vals) if recent_vals else 0.0
                        values.append(float(val))
                    pred_df[lag_col] = values
                
                if len(hist) > 0:
                    last_kl = hist['F_KLCOUNT'].copy()
                    last_window_3 = float(last_kl.ewm(alpha=0.3).mean().iloc[-1]) if len(last_kl) >= 3 else float(last_kl.mean())
                    last_window_7 = float(last_kl.ewm(alpha=0.3).mean().iloc[-1]) if len(last_kl) >= 7 else float(last_kl.mean())
                    last_window_30 = float(last_kl.ewm(alpha=0.3).mean().iloc[-1]) if len(last_kl) >= 30 else float(last_kl.mean())
                    last_std_7 = float(last_kl.rolling(7, min_periods=1).std().iloc[-1]) if len(last_kl) >= 7 else 0.0
                else:
                    last_window_3 = last_window_7 = last_window_30 = last_std_7 = 0.0
                
                pred_df['roll_mean_3'] = [last_window_3] * days
                pred_df['roll_mean_7'] = [last_window_7] * days
                pred_df['roll_mean_30'] = [last_window_30] * days
                pred_df['roll_std_7'] = [last_std_7] * days
                
                pred_df = self._ensure_numeric_data(pred_df, extended_factors)

            if line_data is not None and not line_data.empty:
                hist = line_data.sort_values('F_DATE').copy()
                hist['F_DATE'] = pd.to_datetime(hist['F_DATE'], format='%Y%m%d')
                hist = self._ensure_numeric_data(hist, ['F_KLCOUNT'])
                hist_map = self._get_hist_map(hist)
                
                last_year_base = []
                for d in pred_dates:
                    last_year_date = (d - timedelta(days=365)).strftime('%Y%m%d')
                    base = hist_map.get(last_year_date, np.nan)
                    last_year_base.append(base)
                
                offset_this = []
                offset_last = []
                for d in pred_dates:
                    offsets_this_d = []
                    offsets_last_d = []
                    for j in range(1, 8):
                        t_this = (d - timedelta(days=j)).strftime('%Y%m%d')
                        t_last = (d - timedelta(days=j + 365)).strftime('%Y%m%d')
                        offsets_this_d.append(hist_map.get(t_this, np.nan))
                        offsets_last_d.append(hist_map.get(t_last, np.nan))
                    mean_this_d = np.nanmean(offsets_this_d)
                    mean_last_d = np.nanmean(offsets_last_d)
                    offset_this.append(mean_this_d)
                    offset_last.append(mean_last_d)
                
                abs_offset_per_day = np.nanmean(np.array(offset_this) - np.array(offset_last))
                ratio = np.nanmean(np.array(offset_this) / np.array(offset_last)) if np.nanmean(offset_last) > 0 else 1.0
                
                aux_add = np.array([(b + abs_offset_per_day) if (not np.isnan(b)) else np.nan for b in last_year_base])
                aux_mul = np.array([(b * ratio) if (not np.isnan(b)) else np.nan for b in last_year_base])
                nan_rate = np.isnan(aux_add).mean()
                
                X_pred = pred_df[extended_factors].values.astype(np.float64)
                preds_model = model.predict(X_pred).astype(np.float64)
                
                if nan_rate < 0.5:
                    aux = np.where(np.isnan(aux_add), aux_mul, (aux_add + aux_mul) / 2)
                    fallback_mask = np.isnan(aux)
                    if fallback_mask.any():
                        aux[fallback_mask] = preds_model[fallback_mask]
                    
                    local_offset_weight = self.offset_weight * (1 - nan_rate)
                    local_model_weight = self.model_weight * (1 + nan_rate)
                    total_weight = local_offset_weight + local_model_weight
                    
                    if total_weight > 0:
                        offset_ratio = local_offset_weight / total_weight
                        model_ratio = local_model_weight / total_weight
                        predictions_final = np.maximum(
                            offset_ratio * aux + model_ratio * preds_model, 0.0
                        )
                    else:
                        predictions_final = np.maximum((aux + preds_model) / 2, 0.0)
                    
                    logger.info(f"采用去年同期+偏移法融合模型，offset_weight={local_offset_weight:.2f} model_weight={local_model_weight:.2f}，预测范围 [{predictions_final.min():.1f}, {predictions_final.max():.1f}]")
                else:
                    predictions_final = np.maximum(preds_model, 0.0)
                    logger.info("同期数据缺失较多，直接用模型预测")
            else:
                X_pred = pred_df[extended_factors].values.astype(np.float64)
                predictions_final = np.maximum(model.predict(X_pred), 0.0)
                logger.info("无历史数据，直接用模型预测")

            if len(predictions_final) != days:
                return None, f"预测长度不匹配: {len(predictions_final)} != {days}"

            zero_cnt = (np.array(predictions_final) == 0).sum()
            if zero_cnt > 0:
                logger.warning(f"预测中存在0值 {zero_cnt}/{len(predictions_final)}")

            return [float(x) for x in predictions_final], None

        except Exception as e:
            error_msg = f"预测过程出错: {e}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

    def save_model_info(self, line_no: str, algorithm: str, mse: Optional[float],
                        mae: Optional[float], train_date: str,
                        model_version: Optional[str] = None) -> None:
        version = model_version or self.version
        info = {
            'algorithm': algorithm,
            'mse': float(mse) if mse is not None else None,
            'mae': float(mae) if mae is not None else None,
            'train_date': train_date,
            'line_no': line_no,
            'version': version,
            'factors': self.factors,
            'extended_factors': None,
            'config': self.config,
        }
        prior = self.model_info.get(line_no)
        if isinstance(prior, dict):
            if 'extended_factors' in prior:
                info['extended_factors'] = prior.get('extended_factors')
            if 'metrics' in prior:
                info['metrics'] = prior['metrics']
            if 'best_params' in prior:
                info['best_params'] = prior['best_params']
        
        info_path = os.path.join(self.model_dir, f"model_info_line_{line_no}_daily_v{version}.json")
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            self.model_info[line_no] = info
            logger.info(f"模型信息保存完成: {info_path}")
        except Exception as e:
            logger.error(f"保存模型信息失败: {e}")

    def diagnose_zero_predictions(self, line_data: pd.DataFrame, line_no: str, factor_df: Optional[pd.DataFrame] = None) -> Dict:
        logger.info(f"诊断线路 {line_no}")
        diagnosis = {'data_issues': [], 'model_issues': [], 'recommendations': []}
        try:
            if line_data is None or line_data.empty:
                diagnosis['data_issues'].append("训练数据为空")
                return diagnosis
            if 'F_KLCOUNT' not in line_data.columns:
                diagnosis['data_issues'].append("缺少 F_KLCOUNT 列")
                return diagnosis
            
            line_data_clean = self._ensure_numeric_data(line_data, ['F_KLCOUNT'])
            stats = line_data_clean['F_KLCOUNT'].describe()
            zero_ratio = (line_data_clean['F_KLCOUNT'] == 0).mean()
            
            if zero_ratio > 0.3:
                diagnosis['data_issues'].append(f"目标变量大量为0: {zero_ratio:.1%}")
            
            if stats.get('std', 0) == 0:
                diagnosis['data_issues'].append("目标变量方差为0，无变化")

            info_path = os.path.join(self.model_dir, f"model_info_line_{line_no}_daily_v{self.version}.json")
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                ext_factors = info.get('extended_factors', self.factors)
            else:
                ext_factors = self.factors
            
            df = line_data.copy()
            df['F_DATE'] = pd.to_datetime(df['F_DATE'], format='%Y%m%d')
            
            try:
                df2, ext_factors2 = self._compute_derived_features(df)
            except Exception:
                df2 = df
            
            for f in ext_factors:
                if f not in df2.columns:
                    diagnosis['data_issues'].append(f"训练缺少因子: {f}")
                else:
                    col = pd.to_numeric(df2[f], errors='coerce')
                    if col.isna().mean() > 0.2:
                        diagnosis['data_issues'].append(f"因子 {f} 缺失率 >20%")
                    if col.std() == 0:
                        diagnosis['data_issues'].append(f"因子 {f} 方差为0（无判别能力）")
            
            model_path = os.path.join(self.model_dir, f"xgb_line_{line_no}_daily_v{self.version}.pkl")
            if not os.path.exists(model_path):
                diagnosis['model_issues'].append("模型文件不存在")
            
            diagnosis['recommendations'].extend([
                "优先保证去年同期数据完整性，缺失时再用模型兜底",
                "清洗异常0值，或将极端假日单独建模",
                "对 WEATHER_TYPE 等分类变量考虑 one-hot 或频率编码",
                "增加最近30~90天的历史以提升同期法和模型的可靠性",
                "如同期法失效，可考虑更复杂的时序模型",
                "检查数据类型一致性，确保所有数值字段为float类型"
            ])
        except Exception as e:
            diagnosis['data_issues'].append(f"诊断失败: {e}")
        
        logger.info(f"诊断完成: {diagnosis}")
        return diagnosis

# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 创建示例数据
    dummy = pd.DataFrame({
        'F_DATE': [(datetime(2023,1,1) + timedelta(days=i)).strftime('%Y%m%d') for i in range(400)],
        'F_KLCOUNT': np.random.randint(50000,150000,size=400),
        'F_WEEK': np.random.randint(0, 7, size=400),
        'F_HOLIDAYTYPE': np.random.randint(0, 3, size=400),
        'WEATHER_TYPE': np.random.randint(1, 5, size=400),
    })
    
    # 初始化预测器
    predictor = XGBoostFlowPredictor(
        model_dir="./models_xgb", 
        version="1", 
        config={
            "enable_rolling_predict": True,
            "offset_weight": 2.0,
            "model_weight": 1.0,
        }
    )
    
    # 训练模型
    mse, mae, err = predictor.train(dummy, line_no="testline")
    if err:
        print("训练错误:", err)
    else:
        print(f"训练完成 - MSE: {mse:.2f}, MAE: {mae:.2f}")
        
        # 普通批量预测
        preds, err = predictor.predict(
            dummy, 
            line_no="testline", 
            predict_start_date="20240101", 
            days=7,
            rolling=False
        )
        if err:
            print("预测错误:", err)
        else:
            print("批量预测结果:", [f"{p:.1f}" for p in preds])
        
        # 滚动预测示例
        preds_roll, err_roll = predictor.predict(
            dummy, 
            line_no="testline", 
            predict_start_date="20240101", 
            days=7, 
            rolling=True
        )
        if err_roll:
            print("滚动预测错误:", err_roll)
        else:
            print("滚动预测结果:", [f"{p:.1f}" for p in preds_roll])
        
        # 诊断功能示例
        diagnosis = predictor.diagnose_zero_predictions(dummy, line_no="testline")
        print("诊断结果:", diagnosis)