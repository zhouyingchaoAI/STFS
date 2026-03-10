# KNN 日预测模型模块
"""
该模块实现 KNN 日客流预测算法，特点：
- 支持多因子特征
- KNN + 去年同期偏移量混合预测
- 线路独立权重配置
- 模型版本管理
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

from common_utils import (
    sanitize_filename, calculate_metrics,
    get_last_year_date, ensure_numeric_columns
)
from logger_config import get_model_logger

logger = get_model_logger()


# =============================================================================
# 常量定义
# =============================================================================

DEFAULT_KNN_FACTORS = [
    'F_WEEK', 'F_HOLIDAYTYPE', 'F_HOLIDAYDAYS', 'F_HOLIDAYWHICHDAY',
    'F_DAYOFWEEK', 'WEATHER_TYPE', 'F_YEAR'
]

# 日期类型字典
DAYTYPE_DICT = {
    'holiday': 0,
    'weekday': 1,
    'weekend': 2
}


# =============================================================================
# 工具函数
# =============================================================================

def sanitize_line_no(line_no: str) -> str:
    """清理线路名，将不适合作为文件名的字符替换为安全字符"""
    return sanitize_filename(line_no) if line_no else 'unknown'


# =============================================================================
# KNN 预测器类
# =============================================================================

class KNNFlowPredictor:
    """
    KNN 日客流预测器
    
    特点：
    - 基于工作日特征的 K 近邻预测
    - 支持 KNN + 去年同期偏移量混合预测
    - 支持线路独立权重配置
    - 模型版本管理
    """
    
    def __init__(self, model_dir: str, version: str, config: Dict):
        """
        初始化 KNN 预测器
        
        参数:
            model_dir: 模型存储目录
            version: 模型版本
            config: 配置字典
        """
        self.model_dir = model_dir
        self.version = version
        self.config = config or {}
        self.models: Dict[str, KNeighborsRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_info: Dict[str, Dict] = {}
        
        # 创建模型目录
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 加载因子配置
        self.factors = self.config.get("factors", DEFAULT_KNN_FACTORS)
        
        # 初始化权重配置
        self._init_weights()
        
        logger.info(f"初始化 KNN 预测器 - 版本: {self.version}, 因子数量: {len(self.factors)}")
    
    def _init_weights(self) -> None:
        """初始化算法权重配置"""
        # 全局权重
        self.global_algorithm_weights = self.config.get("algorithm_weights", {
            "knn": 0.8,
            "last_year_offset": 0.2
        })
        
        # 归一化全局权重
        self._normalize_weights(self.global_algorithm_weights, "全局")
        
        # 线路独立权重
        config_line_weights = self.config.get("line_algorithm_weights", {})
        self.line_algorithm_weights: Dict[str, Dict[str, float]] = {}
        
        for line_no, weights in config_line_weights.items():
            self.line_algorithm_weights[line_no] = weights.copy()
            self._normalize_weights(self.line_algorithm_weights[line_no], f"线路{line_no}")
        
        logger.info(f"全局算法权重: {self.global_algorithm_weights}")
        logger.info(f"已配置 {len(self.line_algorithm_weights)} 条线路的独立权重")
    
    def _normalize_weights(self, weights: Dict[str, float], name: str) -> None:
        """归一化权重配置"""
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6 and total > 0:
            logger.warning(f"{name}权重总和为 {total}，将自动归一化")
            for key in weights:
                weights[key] /= total
    
    def get_line_algorithm_weights(self, line_no: str) -> Dict[str, float]:
        """获取线路算法权重（优先使用线路独立权重）"""
        weights = self.line_algorithm_weights.get(line_no)
        if weights is not None:
            self._normalize_weights(weights, f"线路{line_no}")
            return weights
        return self.global_algorithm_weights.copy()
    
    def set_line_algorithm_weights(self, line_no: str, weights: Dict[str, float]) -> None:
        """设置线路算法权重"""
        if not isinstance(weights, dict):
            raise ValueError("权重配置必须是字典类型")
        self._normalize_weights(weights, f"线路{line_no}")
        self.line_algorithm_weights[line_no] = weights.copy()
        logger.info(f"线路 {line_no} 算法权重已设置为: {weights}")
    
    def update_algorithm_weights(
        self,
        new_weights: Dict[str, float],
        line_no: Optional[str] = None
    ) -> None:
        """更新算法权重"""
        if line_no:
            self.set_line_algorithm_weights(line_no, new_weights)
        else:
            self._normalize_weights(new_weights, "全局")
            self.global_algorithm_weights = new_weights.copy()
            logger.info(f"全局算法权重已更新: {self.global_algorithm_weights}")
    
    # =========================================================================
    # 数据准备
    # =========================================================================
    
    def _ensure_numeric_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """确保指定列为数值类型"""
        df = data.copy()
        
        for col in columns:
            if col.lower() == 'weekday':
                continue
            
            # 查找实际列名（大小写不敏感）
            actual_col = next((c for c in df.columns if c.upper() == col.upper()), None)
            
            if actual_col:
                df[col] = pd.to_numeric(df[actual_col], errors='coerce').fillna(0.0)
                if actual_col != col:
                    df = df.drop(actual_col, axis=1)
            else:
                df[col] = 0.0
                logger.warning(f"因子 {col} 不存在，填充为 0")
        
        return df
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        参数:
            data: 原始数据
            
        返回:
            (特征矩阵 X, 目标变量 y)
        """
        logger.info(f"准备数据 - 原始形状: {data.shape}")
        
        # 验证必需列
        required_cols = ['F_DATE', 'F_KLCOUNT']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")
        
        df = data.sort_values('F_DATE').copy()
        
        # 记录日期范围
        if not df.empty:
            logger.info(f"训练数据日期范围: {df['F_DATE'].iloc[0]} ~ {df['F_DATE'].iloc[-1]}")
        
        # 添加 weekday 因子
        if 'weekday' in [f.lower() for f in self.factors]:
            df['weekday'] = df['F_DATE'].apply(
                lambda x: datetime.strptime(str(x), '%Y%m%d').weekday()
            )
        
        # 确保数值类型
        df = self._ensure_numeric_data(df, self.factors)
        
        X = df[self.factors].values.astype(np.float64)
        y = df['F_KLCOUNT'].values.astype(np.float64)
        
        # 验证数据
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("特征矩阵包含无效值")
        if np.any(y < 0):
            raise ValueError("目标变量包含负值")
        
        # 对数变换
        y_transformed = np.log1p(y)
        
        logger.info(f"数据准备完成 - X: {X.shape}, y: {y_transformed.shape}")
        return X, y_transformed
    
    # =========================================================================
    # 模型训练
    # =========================================================================
    
    def train(
        self,
        line_data: pd.DataFrame,
        line_no: str,
        model_version: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        训练 KNN 模型
        
        参数:
            line_data: 线路数据
            line_no: 线路编号
            model_version: 模型版本
            
        返回:
            (MSE, MAE, 错误信息)
        """
        logger.info(f"开始训练线路 {line_no} 的 KNN 模型")
        
        try:
            train_params = self.config.get("train_params", {})
            n_neighbors_list = train_params.get("n_neighbors_list", [3, 5, 7, 9])
            
            X, y = self.prepare_data(line_data)
            
            if len(X) < min(n_neighbors_list):
                error_msg = f"数据量不足 - 样本数: {len(X)}, 最小 K 值: {min(n_neighbors_list)}"
                logger.error(error_msg)
                return None, None, error_msg
            
            # 网格搜索最佳 K 值
            best_mse = float('inf')
            best_model = None
            best_scaler = None
            best_k = n_neighbors_list[0]
            
            for k in n_neighbors_list:
                if len(X) < k:
                    continue
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model = KNeighborsRegressor(
                    n_neighbors=k,
                    weights='distance',
                    metric='euclidean'
                )
                model.fit(X_scaled, y)
                
                y_pred = model.predict(X_scaled)
                mse = mean_squared_error(y, y_pred)
                
                logger.debug(f"K={k}, 训练 MSE={mse:.6f}")
                
                if mse < best_mse:
                    best_mse = mse
                    best_model = model
                    best_scaler = scaler
                    best_k = k
            
            if best_model is None:
                return None, None, "无法找到合适的模型"
            
            # 计算最终指标
            X_scaled = best_scaler.transform(X)
            y_pred_transformed = best_model.predict(X_scaled)
            y_true_original = np.expm1(y)
            y_pred_original = np.maximum(np.expm1(y_pred_transformed), 0)
            
            mse = mean_squared_error(y_true_original, y_pred_original)
            mae = mean_absolute_error(y_true_original, y_pred_original)
            
            logger.info(f"最佳模型: K={best_k}, MSE={mse:.2f}, MAE={mae:.2f}")
            
            # 保存模型
            version = model_version or self.version
            safe_line_no = sanitize_line_no(line_no)
            
            model_path = os.path.join(self.model_dir, f"knn_line_{safe_line_no}_daily_v{version}.pkl")
            scaler_path = os.path.join(self.model_dir, f"knn_scaler_line_{safe_line_no}_daily_v{version}.pkl")
            
            joblib.dump(best_model, model_path)
            joblib.dump(best_scaler, scaler_path)
            
            self.models[line_no] = best_model
            self.scalers[line_no] = best_scaler
            
            logger.info(f"模型保存完成: {model_path}")
            return mse, mae, None
            
        except Exception as e:
            error_msg = f"训练过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, None, error_msg
    
    # =========================================================================
    # 预测
    # =========================================================================
    
    def predict(
        self,
        line_data: pd.DataFrame,
        line_no: str,
        predict_start_date: str,
        days: int = 15,
        model_version: Optional[str] = None,
        factor_df: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        执行预测（混合 KNN 和去年同期偏移）
        
        参数:
            line_data: 线路历史数据
            line_no: 线路编号
            predict_start_date: 预测起始日期
            days: 预测天数
            model_version: 模型版本
            factor_df: 预测因子数据
            
        返回:
            (预测结果列表, 错误信息)
        """
        logger.info(f"开始预测线路 {line_no} - 起始日期: {predict_start_date}, 天数: {days}")
        
        try:
            # 1. KNN 预测
            knn_predictions, knn_error = self._predict_knn(
                line_data, line_no, predict_start_date, days, model_version, factor_df
            )
            
            if knn_predictions is None:
                logger.warning(f"KNN 预测失败: {knn_error}")
                knn_predictions = [0.0] * days
            
            # 2. 去年同期偏移预测
            offset_predictions, offset_info = self._calculate_last_year_offset_prediction(
                line_data, predict_start_date, days
            )
            
            # 3. 融合预测结果
            weights = self.get_line_algorithm_weights(line_no)
            knn_weight = weights.get("knn", 0.8)
            offset_weight = weights.get("last_year_offset", 0.2)
            
            final_predictions = []
            for i in range(days):
                knn_pred = knn_predictions[i] if i < len(knn_predictions) else 0.0
                offset_pred = offset_predictions[i] if i < len(offset_predictions) else 0.0
                final_pred = knn_weight * knn_pred + offset_weight * offset_pred
                final_predictions.append(max(final_pred, 0.0))
            
            logger.info(f"预测融合完成 - KNN 权重: {knn_weight}, 偏移权重: {offset_weight}")
            logger.info(f"最终预测范围: [{min(final_predictions):.2f}, {max(final_predictions):.2f}]")
            
            if len(final_predictions) != days:
                return None, f"预测结果长度错误: {len(final_predictions)} != {days}"
            
            zero_count = sum(1 for p in final_predictions if p == 0)
            if zero_count > 0:
                logger.warning(f"有 {zero_count}/{len(final_predictions)} 个预测结果为 0")
            
            return final_predictions, None
            
        except Exception as e:
            error_msg = f"预测过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
    
    def _predict_knn(
        self,
        line_data: pd.DataFrame,
        line_no: str,
        predict_start_date: str,
        days: int = 15,
        model_version: Optional[str] = None,
        factor_df: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[List[float]], Optional[str]]:
        """KNN 预测"""
        try:
            version = model_version or self.version
            safe_line_no = sanitize_line_no(line_no)
            
            model_path = os.path.join(self.model_dir, f"knn_line_{safe_line_no}_daily_v{version}.pkl")
            scaler_path = os.path.join(self.model_dir, f"knn_scaler_line_{safe_line_no}_daily_v{version}.pkl")
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return None, f"模型文件未找到: {model_path}"
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            logger.debug(f"加载 KNN 模型: K={getattr(model, 'n_neighbors', '未知')}")
            
            # 准备预测数据
            predict_dt = datetime.strptime(predict_start_date, '%Y%m%d')
            
            if factor_df is not None:
                if factor_df.shape[0] != days:
                    return None, f"factor_df 行数 ({factor_df.shape[0]}) 与预测天数 ({days}) 不一致"
                
                pred_df = factor_df.copy()
                pred_df.columns = [c.upper() if c.lower().startswith('f_') else c for c in pred_df.columns]
                
                for col in self.factors:
                    if col not in pred_df.columns:
                        pred_df[col] = 0.0
                
                pred_df = self._ensure_numeric_data(pred_df, self.factors)
            else:
                logger.warning("未提供因子数据，使用历史数据填充")
                
                if line_data.empty:
                    return None, "历史数据为空，无法生成默认因子"
                
                last_row = line_data.sort_values('F_DATE').iloc[-1]
                pred_dates = [predict_dt + timedelta(days=d) for d in range(days)]
                
                pred_df = pd.DataFrame(index=range(days))
                for col in self.factors:
                    if col.lower() == 'weekday':
                        pred_df[col] = [dt.weekday() for dt in pred_dates]
                    elif col in last_row.index:
                        val = float(last_row[col]) if pd.notna(last_row[col]) else 0.0
                        pred_df[col] = [val] * days
                    else:
                        pred_df[col] = [0.0] * days
                
                pred_df = self._ensure_numeric_data(pred_df, self.factors)
            
            X_pred = pred_df[self.factors].values.astype(np.float64)
            
            if np.any(np.isnan(X_pred)) or np.any(np.isinf(X_pred)):
                return None, "预测特征矩阵包含无效值"
            
            X_pred_scaled = scaler.transform(X_pred)
            predictions_transformed = model.predict(X_pred_scaled)
            predictions = np.maximum(np.expm1(predictions_transformed), 0)
            
            logger.debug(f"KNN 预测完成 - 范围: [{predictions.min():.2f}, {predictions.max():.2f}]")
            return predictions.tolist(), None
            
        except Exception as e:
            error_msg = f"KNN 预测过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
    
    def _calculate_last_year_offset_prediction(
        self,
        line_data: pd.DataFrame,
        predict_start_date: str,
        days: int
    ) -> Tuple[List[float], Dict]:
        """
        计算去年同期 + 整体偏移的预测结果
        
        节假日预测时偏移量上浮 20%
        """
        logger.debug("计算去年同期 + 偏移预测")
        
        try:
            line_data_sorted = line_data.sort_values('F_DATE')
            pred_start_dt = datetime.strptime(predict_start_date, "%Y%m%d")
            pred_end_dt = pred_start_dt + timedelta(days=days - 1)
            
            # 去年同期日期
            last_year_start_dt = get_last_year_date(pred_start_dt)
            
            # 偏移区间：预测日期前 days 天
            offset_this_year_start = pred_start_dt - timedelta(days=days)
            offset_this_year_end = pred_start_dt - timedelta(days=1)
            offset_last_year_start = get_last_year_date(offset_this_year_start)
            offset_last_year_end = get_last_year_date(offset_this_year_end)
            
            # 获取去年基准数据
            last_year_base_dates = [last_year_start_dt + timedelta(days=i) for i in range(days)]
            last_year_base_strs = [dt.strftime('%Y%m%d') for dt in last_year_base_dates]
            
            last_year_base_flows = []
            for d in last_year_base_strs:
                match = line_data_sorted[line_data_sorted['F_DATE'] == d]
                if not match.empty:
                    last_year_base_flows.append(float(match['F_KLCOUNT'].values[0]))
                else:
                    last_year_base_flows.append(np.nan)
            
            # 获取偏移区间数据
            offset_this_year_flows = self._get_period_flows(
                line_data_sorted, offset_this_year_start, days
            )
            offset_last_year_flows = self._get_period_flows(
                line_data_sorted, offset_last_year_start, days
            )
            
            # 计算偏移量
            valid_offset_this = [v for v in offset_this_year_flows if not np.isnan(v)]
            valid_offset_last = [v for v in offset_last_year_flows if not np.isnan(v)]
            
            if len(valid_offset_this) >= int(days * 0.5) and len(valid_offset_last) >= int(days * 0.5):
                sum_offset_this = np.nansum(offset_this_year_flows)
                sum_offset_last = np.nansum(offset_last_year_flows)
                overall_offset = sum_offset_this - sum_offset_last
                overall_offset_per_day = overall_offset / days
                logger.debug(f"整体偏移量: {overall_offset:.2f}, 每日偏移: {overall_offset_per_day:.2f}")
            else:
                overall_offset_per_day = 0.0
                logger.warning(f"偏移区间数据不足，设置偏移量为 0")
            
            # 生成预测结果
            predictions = []
            valid_predictions = 0
            
            for i in range(days):
                base_flow = last_year_base_flows[i] if i < len(last_year_base_flows) else np.nan
                
                # 判断是否节假日
                is_holiday = self._is_holiday(line_data_sorted, pred_start_dt, i)
                
                # 计算偏移量（节假日上浮 20%）
                offset = overall_offset_per_day
                if is_holiday and offset != 0:
                    offset = offset + abs(offset) * 0.2
                
                if not np.isnan(base_flow):
                    prediction = max(base_flow + offset, 0.0)
                    predictions.append(prediction)
                    valid_predictions += 1
                else:
                    if valid_offset_this:
                        fallback_base = np.nanmean(offset_this_year_flows)
                        predictions.append(max(fallback_base, 0.0))
                        logger.warning(f"第 {i + 1} 天去年同期数据缺失，使用偏移区间平均值")
                    else:
                        predictions.append(0.0)
                        logger.warning(f"第 {i + 1} 天数据完全缺失，设置为 0")
            
            info = {
                'valid_predictions': valid_predictions,
                'total_predictions': days,
                'overall_offset_per_day': overall_offset_per_day,
                'last_year_base_available': len([v for v in last_year_base_flows if not np.isnan(v)]),
                'prediction_range': [min(predictions), max(predictions)] if predictions else [0, 0]
            }
            
            logger.debug(f"去年同期 + 偏移预测完成 - 有效预测: {valid_predictions}/{days}")
            return predictions, info
            
        except Exception as e:
            error_msg = f"计算去年同期 + 偏移预测时出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [0.0] * days, {'error': error_msg}
    
    def _get_period_flows(
        self,
        data: pd.DataFrame,
        start_dt: datetime,
        days: int
    ) -> List[float]:
        """获取指定时期的客流数据"""
        flows = []
        for i in range(days):
            d = (start_dt + timedelta(days=i)).strftime('%Y%m%d')
            match = data[data['F_DATE'] == d]
            if not match.empty:
                flows.append(float(match['F_KLCOUNT'].values[0]))
            else:
                flows.append(np.nan)
        return flows
    
    def _is_holiday(
        self,
        data: pd.DataFrame,
        pred_start_dt: datetime,
        day_offset: int
    ) -> bool:
        """判断预测日期是否为节假日"""
        pred_date = (pred_start_dt + timedelta(days=day_offset)).strftime("%Y%m%d")
        pred_day_rows = data[data['F_DATE'] == pred_date]
        
        if pred_day_rows.empty:
            return False
        
        row = pred_day_rows.iloc[0]
        
        # 如果 F_HOLIDAYTYPE 不为空，则为节假日
        if pd.notna(row.get('F_HOLIDAYTYPE')) and str(row.get('F_HOLIDAYTYPE')).strip():
            return True
        
        return False
    
    # =========================================================================
    # 模型管理
    # =========================================================================
    
    def save_model_info(
        self,
        line_no: str,
        algorithm: str,
        mse: Optional[float],
        mae: Optional[float],
        train_date: str,
        model_version: Optional[str] = None
    ) -> None:
        """保存模型信息"""
        version = model_version or self.version
        weights = self.get_line_algorithm_weights(line_no)
        
        info = {
            'algorithm': algorithm,
            'mse': float(mse) if mse is not None else None,
            'mae': float(mae) if mae is not None else None,
            'train_date': train_date,
            'line_no': line_no,
            'version': version,
            'factors': self.factors,
            'config': self.config,
            'algorithm_weights': weights
        }
        
        safe_line_no = sanitize_line_no(line_no)
        info_path = os.path.join(self.model_dir, f"model_info_line_{safe_line_no}_daily_v{version}.json")
        
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            self.model_info[line_no] = info
            logger.info(f"模型信息保存完成: {info_path}")
        except Exception as e:
            logger.error(f"保存模型信息失败: {str(e)}")
    
    def get_prediction_details(
        self,
        line_data: pd.DataFrame,
        line_no: str,
        predict_start_date: str,
        days: int = 15,
        model_version: Optional[str] = None,
        factor_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """获取详细的预测结果"""
        logger.debug(f"获取线路 {line_no} 的详细预测结果")
        
        try:
            knn_predictions, knn_error = self._predict_knn(
                line_data, line_no, predict_start_date, days, model_version, factor_df
            )
            offset_predictions, offset_info = self._calculate_last_year_offset_prediction(
                line_data, predict_start_date, days
            )
            final_predictions, final_error = self.predict(
                line_data, line_no, predict_start_date, days, model_version, factor_df
            )
            
            predict_dt = datetime.strptime(predict_start_date, '%Y%m%d')
            pred_dates = [(predict_dt + timedelta(days=d)).strftime('%Y%m%d') for d in range(days)]
            weights = self.get_line_algorithm_weights(line_no)
            
            return {
                'line_no': line_no,
                'predict_dates': pred_dates,
                'knn_predictions': knn_predictions or [0.0] * days,
                'knn_error': knn_error,
                'offset_predictions': offset_predictions,
                'offset_info': offset_info,
                'final_predictions': final_predictions or [0.0] * days,
                'final_error': final_error,
                'algorithm_weights': weights,
                'prediction_summary': {
                    'knn_range': [min(knn_predictions or [0]), max(knn_predictions or [0])],
                    'offset_range': [min(offset_predictions), max(offset_predictions)],
                    'final_range': [min(final_predictions or [0]), max(final_predictions or [0])]
                }
            }
        except Exception as e:
            logger.error(f"获取详细预测结果时出错: {str(e)}", exc_info=True)
            return {'error': str(e)}
    
    def diagnose_zero_predictions(
        self,
        line_data: pd.DataFrame,
        line_no: str,
        factor_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """诊断预测为 0 的问题"""
        logger.info(f"诊断线路 {line_no} 预测为 0 的问题")
        
        diagnosis = {
            'data_issues': [],
            'model_issues': [],
            'algorithm_issues': [],
            'recommendations': []
        }
        
        try:
            if line_data.empty:
                diagnosis['data_issues'].append("训练数据为空")
                return diagnosis
            
            # 检查目标变量
            target_stats = line_data['F_KLCOUNT'].describe()
            zero_ratio = (line_data['F_KLCOUNT'] == 0).mean()
            
            if zero_ratio > 0.5:
                diagnosis['data_issues'].append(f"训练数据中 {zero_ratio:.1%} 的目标变量为 0")
            
            if target_stats.get('std', 0) == 0:
                diagnosis['data_issues'].append("目标变量方差为 0，无变化")
            
            # 检查模型文件
            version = self.version
            safe_line_no = sanitize_line_no(line_no)
            model_path = os.path.join(self.model_dir, f"knn_line_{safe_line_no}_daily_v{version}.pkl")
            
            if not os.path.exists(model_path):
                diagnosis['model_issues'].append("KNN 模型文件不存在")
            
            # 生成建议
            if diagnosis['data_issues']:
                diagnosis['recommendations'].extend([
                    "检查数据质量，确保目标变量和特征变量有足够变化",
                    "检查历史数据的完整性和连续性"
                ])
            
            if diagnosis['model_issues']:
                diagnosis['recommendations'].append("重新训练 KNN 模型")
            
            weights = self.get_line_algorithm_weights(line_no)
            diagnosis['recommendations'].append(
                f"当前算法权重: KNN={weights.get('knn', 0.8)}, 偏移={weights.get('last_year_offset', 0.2)}"
            )
            
        except Exception as e:
            diagnosis['data_issues'].append(f"诊断过程出错: {str(e)}")
        
        logger.info(f"诊断完成: {diagnosis}")
        return diagnosis
