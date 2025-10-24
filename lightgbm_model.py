# LightGBM 模型模块：增加"去年同期+整体偏移"算法逻辑
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

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

DEFAULT_LGBM_FACTORS = [
    'F_WEEK', 'F_DATEFEATURES', 'F_HOLIDAYTYPE', 'F_ISHOLIDAY',
    'F_ISNONGLI', 'F_ISYANGLI', 'F_NEXTDAY', 'F_HOLIDAYDAYS',
    'F_HOLIDAYTHDAY', 'IS_FIRST', 'WEATHER_TYPE'
]

# 预设每条线路的权重配置（基于不同线路的特点优化）
DEFAULT_LINE_WEIGHTS = {
    # 主干线路：客流稳定，LightGBM权重较高
    # '0': {"lgbm": 0.4, "last_year_offset": 0.6},  # 线网，客流相对稳定
    # '1': {"lgbm": 0.5, "last_year_offset": 0.5},  # 1号线主干线，平衡配置
    # '2': {"lgbm": 0.4, "last_year_offset": 0.6},  # 2号线主干线，偏向历史模式
    # '3': {"lgbm": 0.3, "last_year_offset": 0.7},  # 3号线，季节性较强
    # '4': {"lgbm": 0.6, "last_year_offset": 0.4},  # 4号线，现代化程度高，模式较新
    # '5': {"lgbm": 0.4, "last_year_offset": 0.6},  # 5号线，中等配置
    # '6': {"lgbm": 0.5, "last_year_offset": 0.5},  # 6号线，平衡配置
    
    # # 支线：客流波动大，更依赖去年同期模式
    # '31': {"lgbm": 0.2, "last_year_offset": 0.8},  
    # '60': {"lgbm": 0.3, "last_year_offset": 0.7},  
    # '83': {"lgbm": 0.3, "last_year_offset": 0.7},  
    
}

class LightGBMFlowPredictor:
    def __init__(self, model_dir: str, version: str, config: Dict):
        self.model_dir = model_dir
        self.version = version
        self.config = config
        self.models = {}
        self.scalers = {}
        self.model_info = {}

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            logger.info(f"创建模型目录: {self.model_dir}")

        self.factors = self.config.get("factors", DEFAULT_LGBM_FACTORS)
        
        # 支持全局和每线路权重配置
        self.global_algorithm_weights = self.config.get("algorithm_weights", {
            "lgbm": 0.2,
            "last_year_offset": 0.8
        })
        
        # 初始化每条线路的权重配置
        config_line_weights = self.config.get("line_algorithm_weights", {})
        self.line_algorithm_weights: Dict[str, Dict[str, float]] = {}
        
        # 为每条线路设置权重配置
        for line_no in ['0', '1', '2', '3', '4', '5', '6', '31', '60', '83']:
            if line_no in config_line_weights:
                self.line_algorithm_weights[line_no] = config_line_weights[line_no].copy()
            elif line_no in DEFAULT_LINE_WEIGHTS:
                self.line_algorithm_weights[line_no] = DEFAULT_LINE_WEIGHTS[line_no].copy()
        
        # 归一化全局权重
        total_weight = sum(self.global_algorithm_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"全局算法权重总和为{total_weight}，将自动归一化")
            for key in self.global_algorithm_weights:
                self.global_algorithm_weights[key] /= total_weight

        # 归一化所有线路权重
        for line_no, weights in self.line_algorithm_weights.items():
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-6 and total > 0:
                logger.warning(f"线路{line_no}权重总和为{total}，将自动归一化")
                self.line_algorithm_weights[line_no] = {k: v/total for k, v in weights.items()}

        # LightGBM 训练参数
        self.lgbm_params = self.config.get("lgbm_params", {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
            'early_stopping_rounds': 10,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        })

        logger.info(f"初始化LightGBM预测器 - 版本: {self.version}, 因子数量: {len(self.factors)}")
        logger.info(f"全局算法权重配置: {self.global_algorithm_weights}")
        logger.info(f"已配置{len(self.line_algorithm_weights)}条线路的独立权重")
        
        # 打印每条线路的权重配置（只显示已配置的线路）
        configured_lines = list(self.line_algorithm_weights.keys())
        if configured_lines:
            logger.info(f"已配置权重的线路: {sorted(configured_lines)}")
            for line_no in sorted(configured_lines):
                weights = self.get_line_algorithm_weights(line_no)
                logger.info(f"线路{line_no}权重: LightGBM={weights['lgbm']:.2f}, 偏移={weights['last_year_offset']:.2f}")
        
        # 显示默认全局权重
        logger.info(f"未单独配置线路将使用全局权重: LightGBM={self.global_algorithm_weights['lgbm']:.2f}, 偏移={self.global_algorithm_weights['last_year_offset']:.2f}")

    def get_line_algorithm_weights(self, line_no: str) -> Dict[str, float]:
        weights = self.line_algorithm_weights.get(line_no)
        if weights is not None:
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-6 and total > 0:
                weights = {k: v/total for k, v in weights.items()}
                self.line_algorithm_weights[line_no] = weights
            return weights
        else:
            return self.global_algorithm_weights

    def set_line_algorithm_weights(self, line_no: str, weights: Dict[str, float]):
        if not isinstance(weights, dict):
            raise ValueError("权重配置必须是字典类型")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6 and total > 0:
            logger.warning(f"线路{line_no}权重总和为{total}，将自动归一化")
            weights = {k: v/total for k, v in weights.items()}
        self.line_algorithm_weights[line_no] = weights
        logger.info(f"线路{line_no}算法权重已设置为: {weights}")

    def get_all_line_weights(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for line_no in ['0', '1', '2', '3', '4', '5', '6', '31', '60', '83']:
            result[line_no] = self.get_line_algorithm_weights(line_no)
        return result

    def set_batch_line_weights(self, line_weights: Dict[str, Dict[str, float]]):
        for line_no, weights in line_weights.items():
            try:
                self.set_line_algorithm_weights(line_no, weights)
            except Exception as e:
                logger.error(f"设置线路{line_no}权重失败: {str(e)}")

    def _ensure_numeric_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = data.copy()
        for col in columns:
            if col == 'weekday':
                continue
            if col not in df.columns:
                df[col] = 0.0
                logger.warning(f"因子 {col} 不存在，填充为0")
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df

    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建额外的时间特征来提升LightGBM性能"""
        df = data.copy()
        
        # 确保F_DATE列存在且为字符串
        if 'F_DATE' in df.columns:
            df['F_DATE'] = df['F_DATE'].astype(str)
            
            # 解析日期并创建时间特征
            df['date_dt'] = pd.to_datetime(df['F_DATE'], format='%Y%m%d')
            df['month'] = df['date_dt'].dt.month
            df['day_of_month'] = df['date_dt'].dt.day
            df['quarter'] = df['date_dt'].dt.quarter
            df['day_of_year'] = df['date_dt'].dt.dayofyear
            df['week_of_year'] = df['date_dt'].dt.isocalendar().week
            
            # 周期性特征（用sin/cos编码）
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
            df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
            df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
            
            # 滞后特征
            df = df.sort_values('F_DATE')
            for lag in [1, 2, 3, 7, 14]:
                df[f'flow_lag_{lag}'] = df['F_KLCOUNT'].shift(lag)
            
            # 移动平均特征
            for window in [3, 7, 14]:
                df[f'flow_ma_{window}'] = df['F_KLCOUNT'].rolling(window=window).mean()
                df[f'flow_std_{window}'] = df['F_KLCOUNT'].rolling(window=window).std()
            
            # 删除临时列
            df = df.drop(['date_dt'], axis=1)
            
        return df

    def _calculate_last_year_offset_prediction(self, line_data: pd.DataFrame, 
                                             predict_start_date: str, 
                                             days: int) -> Tuple[List[float], Dict]:
        """计算去年同期+整体偏移的预测结果"""
        logger.info("开始计算去年同期+整体偏移预测")
        try:
            line_data_sorted = line_data.sort_values('F_DATE')
            pred_start_dt = datetime.strptime(predict_start_date, "%Y%m%d")
            pred_end_dt = pred_start_dt + timedelta(days=days-1)
            last_year_start_dt = pred_start_dt - timedelta(days=365)
            last_year_end_dt = pred_end_dt - timedelta(days=365)
            offset_this_year_start = pred_start_dt - timedelta(days=days)
            offset_this_year_end = pred_start_dt - timedelta(days=1)
            offset_last_year_start = last_year_start_dt - timedelta(days=days)
            offset_last_year_end = last_year_start_dt - timedelta(days=1)
            
            last_year_base_dates = [last_year_start_dt + timedelta(days=i) for i in range(days)]
            last_year_base_strs = [dt.strftime('%Y%m%d') for dt in last_year_base_dates]
            last_year_base_flows = []
            for d in last_year_base_strs:
                match = line_data_sorted[line_data_sorted['F_DATE'] == d]
                if not match.empty:
                    last_year_base_flows.append(float(match['F_KLCOUNT'].values[0]))
                else:
                    last_year_base_flows.append(np.nan)
                    
            offset_this_year_dates = [offset_this_year_start + timedelta(days=i) for i in range(days)]
            offset_this_year_strs = [dt.strftime('%Y%m%d') for dt in offset_this_year_dates]
            offset_this_year_flows = []
            for d in offset_this_year_strs:
                match = line_data_sorted[line_data_sorted['F_DATE'] == d]
                if not match.empty:
                    offset_this_year_flows.append(float(match['F_KLCOUNT'].values[0]))
                else:
                    offset_this_year_flows.append(np.nan)
                    
            offset_last_year_dates = [offset_last_year_start + timedelta(days=i) for i in range(days)]
            offset_last_year_strs = [dt.strftime('%Y%m%d') for dt in offset_last_year_dates]
            offset_last_year_flows = []
            for d in offset_last_year_strs:
                match = line_data_sorted[line_data_sorted['F_DATE'] == d]
                if not match.empty:
                    offset_last_year_flows.append(float(match['F_KLCOUNT'].values[0]))
                else:
                    offset_last_year_flows.append(np.nan)
                    
            valid_offset_this = [v for v in offset_this_year_flows if not np.isnan(v)]
            valid_offset_last = [v for v in offset_last_year_flows if not np.isnan(v)]
            
            if len(valid_offset_this) >= int(days * 0.5) and len(valid_offset_last) >= int(days * 0.5):
                sum_offset_this = np.nansum(offset_this_year_flows)
                sum_offset_last = np.nansum(offset_last_year_flows)
                overall_offset = sum_offset_this - sum_offset_last
                overall_offset_per_day = overall_offset / days
                logger.info(f"计算得到整体偏移量: {overall_offset:.2f}, 平均每日偏移: {overall_offset_per_day:.2f}")
            else:
                overall_offset_per_day = 0.0
                logger.warning(f"偏移区间数据不足，设置偏移量为0")
                
            predictions = []
            valid_predictions = 0
            for i in range(days):
                base_flow = last_year_base_flows[i] if i < len(last_year_base_flows) else np.nan
                if not np.isnan(base_flow):
                    prediction = max(base_flow + overall_offset_per_day, 0.0)
                    predictions.append(prediction)
                    valid_predictions += 1
                else:
                    if valid_offset_this:
                        fallback_base = np.nanmean(offset_this_year_flows)
                        prediction = max(fallback_base, 0.0)
                        predictions.append(prediction)
                    else:
                        predictions.append(0.0)
                        
            info = {
                'valid_predictions': valid_predictions,
                'total_predictions': days,
                'overall_offset_per_day': overall_offset_per_day,
                'last_year_base_available': len([v for v in last_year_base_flows if not np.isnan(v)]),
                'offset_this_year_available': len(valid_offset_this),
                'offset_last_year_available': len(valid_offset_last),
                'prediction_range': [min(predictions), max(predictions)] if predictions else [0, 0]
            }
            
            logger.info(f"去年同期+偏移预测完成 - 有效预测: {valid_predictions}/{days}")
            return predictions, info
            
        except Exception as e:
            error_msg = f"计算去年同期+偏移预测时出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [0.0] * days, {'error': error_msg}

    def prepare_data(self, data: pd.DataFrame, add_time_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"开始准备数据 - 原始数据形状: {data.shape}")
        required_cols = ['F_DATE', 'F_KLCOUNT']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")
            
        df = data.sort_values('F_DATE').copy()
        
        if 'weekday' in self.factors:
            df['weekday'] = df['F_DATE'].apply(
                lambda x: datetime.strptime(str(x), '%Y%m%d').weekday()
            )
        
        # 添加时间特征来提升LightGBM性能
        if add_time_features:
            df = self._create_time_features(df)
            
            # 更新因子列表以包含新的时间特征
            time_features = [
                'month', 'day_of_month', 'quarter', 'day_of_year', 'week_of_year',
                'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos',
                'flow_lag_1', 'flow_lag_2', 'flow_lag_3', 'flow_lag_7', 'flow_lag_14',
                'flow_ma_3', 'flow_ma_7', 'flow_ma_14', 'flow_std_3', 'flow_std_7', 'flow_std_14'
            ]
            
            # 扩展因子列表
            extended_factors = self.factors.copy()
            for feature in time_features:
                if feature in df.columns and feature not in extended_factors:
                    extended_factors.append(feature)
        else:
            extended_factors = self.factors
            
        df = self._ensure_numeric_data(df, extended_factors)
        
        # 删除包含NaN的行（由于滞后特征和移动平均造成）
        valid_indices = df[extended_factors + ['F_KLCOUNT']].dropna().index
        df = df.loc[valid_indices]
        
        X = df[extended_factors].values.astype(np.float64)
        y = df['F_KLCOUNT'].values.astype(np.float64)
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.error("特征矩阵包含无效值")
            raise ValueError("特征矩阵包含无效值")
            
        if np.any(y < 0):
            logger.error("目标变量包含负值")
            raise ValueError("目标变量包含负值")
            
        # 对目标变量进行对数变换以提高稳定性
        y_transformed = np.log1p(y)
        
        logger.info(f"数据准备完成 - X形状: {X.shape}, y形状: {y_transformed.shape}")
        logger.info(f"使用因子数量: {len(extended_factors)}")
        
        return X, y_transformed

    def train(self, line_data: pd.DataFrame, line_no: str, 
              model_version: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        logger.info(f"开始训练线路 {line_no} 的LightGBM模型")
        try:
            X, y = self.prepare_data(line_data)
            
            if len(X) < 20:  # LightGBM需要足够的样本
                error_msg = f"数据量不足 - 样本数: {len(X)}, 最少需要20个样本"
                logger.error(error_msg)
                return None, None, error_msg
                
            # 划分训练集和验证集
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # 创建LightGBM数据集
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # 训练模型
            params = self.lgbm_params.copy()
            early_stopping_rounds = params.pop('early_stopping_rounds', 10)
            n_estimators = params.pop('n_estimators', 100)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
            )
            
            # 预测和评估
            y_pred_transformed = model.predict(X, num_iteration=model.best_iteration)
            y_true_original = np.expm1(y)
            y_pred_original = np.maximum(np.expm1(y_pred_transformed), 0)
            
            mse = mean_squared_error(y_true_original, y_pred_original)
            mae = mean_absolute_error(y_true_original, y_pred_original)
            
            logger.info(f"LightGBM模型训练完成 - MSE: {mse:.2f}, MAE: {mae:.2f}")
            logger.info(f"最佳迭代次数: {model.best_iteration}, 特征重要性排序前5:")
            
            # 输出特征重要性
            importance = model.feature_importance(importance_type='gain')
            if hasattr(self, 'extended_factors'):
                feature_names = self.extended_factors
            else:
                feature_names = self.factors
                
            if len(importance) == len(feature_names):
                feature_importance = list(zip(feature_names, importance))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                for i, (feature, imp) in enumerate(feature_importance[:5]):
                    logger.info(f"  {i+1}. {feature}: {imp:.2f}")
            
            # 保存模型
            version = model_version or self.version
            model_path = os.path.join(self.model_dir, f"lgbm_line_{line_no}_daily_v{version}.pkl")
            joblib.dump(model, model_path)
            self.models[line_no] = model
            
            logger.info(f"模型保存完成: {model_path}")
            return mse, mae, None
            
        except Exception as e:
            error_msg = f"训练过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, None, error_msg

    def predict(self, line_data: pd.DataFrame, line_no: str, predict_start_date: str,
                days: int = 15, model_version: Optional[str] = None,
                factor_df: Optional[pd.DataFrame] = None) -> Tuple[Optional[List[float]], Optional[str]]:
        logger.info(f"开始预测线路 {line_no} - 起始日期: {predict_start_date}, 天数: {days}")
        try:
            # 1. LightGBM预测
            lgbm_predictions, lgbm_error = self._predict_lgbm(
                line_data, line_no, predict_start_date, days, model_version, factor_df
            )
            if lgbm_predictions is None:
                logger.warning(f"LightGBM预测失败: {lgbm_error}")
                lgbm_predictions = [0.0] * days
                
            # 2. 去年同期+偏移预测
            offset_predictions, offset_info = self._calculate_last_year_offset_prediction(
                line_data, predict_start_date, days
            )
            
            # 3. 融合预测结果
            final_predictions = []
            weights = self.get_line_algorithm_weights(line_no)
            lgbm_weight = weights.get("lgbm", 0.3)
            offset_weight = weights.get("last_year_offset", 0.7)
            
            for i in range(days):
                lgbm_pred = lgbm_predictions[i] if i < len(lgbm_predictions) else 0.0
                offset_pred = offset_predictions[i] if i < len(offset_predictions) else 0.0
                final_pred = lgbm_weight * lgbm_pred + offset_weight * offset_pred
                final_predictions.append(max(final_pred, 0.0))
                
            logger.info(f"预测融合完成 - LightGBM权重: {lgbm_weight}, 偏移权重: {offset_weight}")
            logger.info(f"最终预测范围: [{min(final_predictions):.2f}, {max(final_predictions):.2f}]")
            
            return final_predictions, None
            
        except Exception as e:
            error_msg = f"预测过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

    def _predict_lgbm(self, line_data: pd.DataFrame, line_no: str, predict_start_date: str,
                      days: int = 15, model_version: Optional[str] = None,
                      factor_df: Optional[pd.DataFrame] = None) -> Tuple[Optional[List[float]], Optional[str]]:
        """LightGBM预测的核心逻辑"""
        try:
            version = model_version or self.version
            model_path = os.path.join(self.model_dir, f"lgbm_line_{line_no}_daily_v{version}.pkl")
            
            if not os.path.exists(model_path):
                return None, f"模型文件未找到: {model_path}"
                
            model = joblib.load(model_path)
            logger.info(f"成功加载LightGBM模型，特征数量: {model.num_feature()}")
            
            predict_dt = datetime.strptime(predict_start_date, '%Y%m%d')
            pred_dates = [predict_dt + timedelta(days=d) for d in range(days)]
            
            if factor_df is not None:
                logger.info("使用用户提供的因子数据")
                if factor_df.shape[0] != days:
                    return None, f"factor_df 行数({factor_df.shape[0]})与预测天数({days})不一致"
                    
                pred_df = factor_df.copy()
                # 添加日期列以便生成时间特征
                pred_df['F_DATE'] = [dt.strftime('%Y%m%d') for dt in pred_dates]
                pred_df['F_KLCOUNT'] = 0  # 临时填充，用于生成特征
            else:
                logger.warning("未提供因子数据，基于历史数据生成预测特征")
                if line_data.empty:
                    return None, "历史数据为空，无法生成默认因子"
                    
                # 创建预测数据框
                pred_df = pd.DataFrame()
                pred_df['F_DATE'] = [dt.strftime('%Y%m%d') for dt in pred_dates]
                pred_df['F_KLCOUNT'] = 0  # 临时填充
                
                # 基于历史数据填充因子
                line_data_sorted = line_data.sort_values('F_DATE')
                last_row = line_data_sorted.iloc[-1]
                
                for col in self.factors:
                    if col == 'weekday':
                        pred_df[col] = [dt.weekday() for dt in pred_dates]
                    elif col in last_row.index:
                        val = float(last_row[col]) if pd.notna(last_row[col]) else 0.0
                        pred_df[col] = [val] * days
                    else:
                        pred_df[col] = [0.0] * days
                        
            # 组合历史数据和预测数据以便生成时间特征
            combined_data = pd.concat([line_data, pred_df], ignore_index=True)
            combined_data = self._create_time_features(combined_data)
            
            # 提取预测部分的特征
            pred_data = combined_data.tail(days).copy()
            
            # 确定要使用的特征列
            time_features = [
                'month', 'day_of_month', 'quarter', 'day_of_year', 'week_of_year',
                'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos',
                'flow_lag_1', 'flow_lag_2', 'flow_lag_3', 'flow_lag_7', 'flow_lag_14',
                'flow_ma_3', 'flow_ma_7', 'flow_ma_14', 'flow_std_3', 'flow_std_7', 'flow_std_14'
            ]
            
            extended_factors = self.factors.copy()
            for feature in time_features:
                if feature in pred_data.columns and feature not in extended_factors:
                    extended_factors.append(feature)
            
            # 确保所有特征都存在
            for col in extended_factors:
                if col not in pred_data.columns:
                    pred_data[col] = 0.0
                    
            pred_data = self._ensure_numeric_data(pred_data, extended_factors)
            
            # 处理可能的NaN值（由于滞后特征）
            pred_data = pred_data.fillna(method='ffill').fillna(0)
            
            X_pred = pred_data[extended_factors].values.astype(np.float64)
            
            if np.any(np.isnan(X_pred)) or np.any(np.isinf(X_pred)):
                return None, "预测特征矩阵包含无效值"
                
            # 确保特征数量匹配
            if X_pred.shape[1] != model.num_feature():
                logger.warning(f"特征数量不匹配：预测数据{X_pred.shape[1]}，模型需要{model.num_feature()}")
                # 调整特征数量
                if X_pred.shape[1] > model.num_feature():
                    X_pred = X_pred[:, :model.num_feature()]
                else:
                    # 补充特征
                    missing_features = model.num_feature() - X_pred.shape[1]
                    X_pred = np.hstack([X_pred, np.zeros((X_pred.shape[0], missing_features))])
            
            logger.info(f"LightGBM预测特征矩阵形状: {X_pred.shape}")
            
            # 进行预测
            predictions_transformed = model.predict(X_pred, num_iteration=model.best_iteration)
            predictions = np.maximum(np.expm1(predictions_transformed), 0)
            
            logger.info(f"LightGBM预测完成 - 结果范围: [{predictions.min():.2f}, {predictions.max():.2f}]")
            return predictions.tolist(), None
            
        except Exception as e:
            error_msg = f"LightGBM预测过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

    def get_prediction_details(self, line_data: pd.DataFrame, line_no: str, 
                              predict_start_date: str, days: int = 15,
                              model_version: Optional[str] = None,
                              factor_df: Optional[pd.DataFrame] = None) -> Dict:
        """获取详细的预测结果，包含各个算法的单独预测结果"""
        logger.info(f"获取线路 {line_no} 的详细预测结果")
        try:
            lgbm_predictions, lgbm_error = self._predict_lgbm(
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
                'lgbm_predictions': lgbm_predictions or [0.0] * days,
                'lgbm_error': lgbm_error,
                'offset_predictions': offset_predictions,
                'offset_info': offset_info,
                'final_predictions': final_predictions or [0.0] * days,
                'final_error': final_error,
                'algorithm_weights': weights,
                'prediction_summary': {
                    'lgbm_range': [min(lgbm_predictions or [0]), max(lgbm_predictions or [0])],
                    'offset_range': [min(offset_predictions), max(offset_predictions)],
                    'final_range': [min(final_predictions or [0]), max(final_predictions or [0])]
                }
            }
            
        except Exception as e:
            logger.error(f"获取详细预测结果时出错: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def save_model_info(self, line_no: str, algorithm: str, mse: Optional[float], 
                       mae: Optional[float], train_date: str, 
                       model_version: Optional[str] = None) -> None:
        import json
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
            'lgbm_params': self.lgbm_params,
            'config': self.config,
            'algorithm_weights': weights
        }
        
        info_path = os.path.join(self.model_dir, f"model_info_line_{line_no}_daily_v{version}.json")
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            self.model_info[line_no] = info
            logger.info(f"模型信息保存完成: {info_path}")
        except Exception as e:
            logger.error(f"保存模型信息失败: {str(e)}")

    def diagnose_zero_predictions(self, line_data: pd.DataFrame, line_no: str, 
                                factor_df: Optional[pd.DataFrame] = None) -> Dict:
        """诊断预测为0的问题"""
        logger.info(f"开始诊断线路 {line_no} 预测为0的问题")
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
                diagnosis['data_issues'].append(f"训练数据中{zero_ratio:.1%}的目标变量为0")
            if target_stats.get('std', 0) == 0:
                diagnosis['data_issues'].append("目标变量方差为0，无变化")
                
            # 检查特征变量
            try:
                X, y = self.prepare_data(line_data)
                for i, factor in enumerate(self.factors):
                    if i < X.shape[1]:
                        factor_std = X[:, i].std()
                        if factor_std == 0:
                            diagnosis['data_issues'].append(f"因子 {factor} 方差为0，无区分度")
            except Exception as e:
                diagnosis['data_issues'].append(f"数据准备失败: {str(e)}")
                
            # 检查模型文件
            version = self.version
            model_path = os.path.join(self.model_dir, f"lgbm_line_{line_no}_daily_v{version}.pkl")
            if not os.path.exists(model_path):
                diagnosis['model_issues'].append("LightGBM模型文件不存在")
            else:
                try:
                    model = joblib.load(model_path)
                    logger.info(f"模型加载成功，特征数量: {model.num_feature()}")
                except Exception as e:
                    diagnosis['model_issues'].append(f"模型加载失败: {str(e)}")
                    
            # 检查去年同期算法
            try:
                predict_start_date = datetime.now().strftime('%Y%m%d')
                _, offset_info = self._calculate_last_year_offset_prediction(line_data, predict_start_date, 7)
                if 'error' in offset_info:
                    diagnosis['algorithm_issues'].append(f"去年同期+偏移算法出错: {offset_info['error']}")
                else:
                    if offset_info.get('last_year_base_available', 0) == 0:
                        diagnosis['algorithm_issues'].append("去年同期数据完全缺失")
                    if offset_info.get('offset_this_year_available', 0) == 0:
                        diagnosis['algorithm_issues'].append("今年偏移区间数据缺失")
                    if offset_info.get('offset_last_year_available', 0) == 0:
                        diagnosis['algorithm_issues'].append("去年偏移区间数据缺失")
            except Exception as e:
                diagnosis['algorithm_issues'].append(f"检查去年同期数据时出错: {str(e)}")
                
            # 生成建议
            if diagnosis['data_issues']:
                diagnosis['recommendations'].extend([
                    "检查数据质量，确保目标变量和特征变量有足够变化",
                    "检查历史数据的完整性和连续性",
                    "考虑数据清洗和异常值处理"
                ])
                
            if diagnosis['model_issues']:
                diagnosis['recommendations'].append("重新训练LightGBM模型")
                
            if diagnosis['algorithm_issues']:
                diagnosis['recommendations'].extend([
                    "检查历史数据覆盖范围，确保包含足够的去年同期数据",
                    "考虑调整算法权重配置"
                ])
                
            weights = self.get_line_algorithm_weights(line_no)
            diagnosis['recommendations'].extend([
                "确保预测因子与训练因子分布一致",
                "尝试调整LightGBM参数和权重配置",
                "考虑特征工程或增加更多时间特征",
                f"当前算法权重: LightGBM={weights.get('lgbm', 0.3)}, 去年同期+偏移={weights.get('last_year_offset', 0.7)}"
            ])
            
        except Exception as e:
            diagnosis['data_issues'].append(f"诊断过程出错: {str(e)}")
            
        logger.info(f"诊断完成: {diagnosis}")
        return diagnosis

    def update_algorithm_weights(self, new_weights: Dict[str, float], line_no: Optional[str] = None) -> None:
        """更新算法权重配置"""
        if not isinstance(new_weights, dict):
            raise ValueError("权重配置必须是字典类型")
            
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 1e-6 and total_weight > 0:
            logger.warning(f"权重总和为{total_weight}，将自动归一化")
            new_weights = {k: v/total_weight for k, v in new_weights.items()}
            
        if line_no is not None:
            old_weights = self.line_algorithm_weights.get(line_no, self.global_algorithm_weights).copy()
            self.line_algorithm_weights[line_no] = new_weights
            logger.info(f"线路{line_no}算法权重已更新: {old_weights} -> {new_weights}")
        else:
            old_weights = self.global_algorithm_weights.copy()
            self.global_algorithm_weights.update(new_weights)
            logger.info(f"全局算法权重已更新: {old_weights} -> {self.global_algorithm_weights}")

    def get_algorithm_performance(self, line_data: pd.DataFrame, line_no: str,
                                 test_start_date: str, test_days: int = 7,
                                 model_version: Optional[str] = None) -> Dict:
        """评估不同算法在历史数据上的性能表现"""
        logger.info(f"开始评估线路 {line_no} 的算法性能")
        try:
            test_start_dt = datetime.strptime(test_start_date, '%Y%m%d')
            test_dates = [(test_start_dt + timedelta(days=d)).strftime('%Y%m%d') 
                         for d in range(test_days)]
            
            line_data_sorted = line_data.sort_values('F_DATE')
            
            # 获取测试期真实值
            true_values = []
            available_test_dates = []
            for date_str in test_dates:
                match = line_data_sorted[line_data_sorted['F_DATE'] == date_str]
                if not match.empty:
                    true_values.append(float(match['F_KLCOUNT'].values[0]))
                    available_test_dates.append(date_str)
                    
            if not true_values:
                return {'error': f'测试期间 {test_start_date} 后 {test_days} 天内无可用数据'}
                
            # 准备训练数据（测试日期之前的数据）
            train_data = line_data_sorted[line_data_sorted['F_DATE'] < test_start_date]
            if train_data.empty:
                return {'error': '训练数据为空'}
                
            actual_test_days = len(available_test_dates)
            
            # LightGBM预测
            lgbm_predictions, lgbm_error = self._predict_lgbm(
                train_data, line_no, test_start_date, actual_test_days, model_version
            )
            
            # 去年同期+偏移预测
            offset_predictions, offset_info = self._calculate_last_year_offset_prediction(
                train_data, test_start_date, actual_test_days
            )
            
            # 融合预测
            weights = self.get_line_algorithm_weights(line_no)
            lgbm_weight = weights.get("lgbm", 0.3)
            offset_weight = weights.get("last_year_offset", 0.7)
            
            if lgbm_predictions and len(lgbm_predictions) >= actual_test_days:
                final_predictions = []
                for i in range(actual_test_days):
                    lgbm_pred = lgbm_predictions[i] if lgbm_predictions else 0.0
                    offset_pred = offset_predictions[i] if i < len(offset_predictions) else 0.0
                    final_pred = lgbm_weight * lgbm_pred + offset_weight * offset_pred
                    final_predictions.append(max(final_pred, 0.0))
            else:
                final_predictions = [0.0] * actual_test_days
                
            def calculate_metrics(predictions, true_vals):
                if not predictions or len(predictions) != len(true_vals):
                    return {'mse': float('inf'), 'mae': float('inf'), 'mape': float('inf')}
                    
                predictions = np.array(predictions[:len(true_vals)])
                true_vals = np.array(true_vals)
                
                mse = mean_squared_error(true_vals, predictions)
                mae = mean_absolute_error(true_vals, predictions)
                
                # 计算MAPE
                non_zero_mask = true_vals != 0
                if non_zero_mask.any():
                    mape = np.mean(np.abs((true_vals[non_zero_mask] - predictions[non_zero_mask]) 
                                        / true_vals[non_zero_mask])) * 100
                else:
                    mape = float('inf')
                    
                return {'mse': float(mse), 'mae': float(mae), 'mape': float(mape)}
            
            # 计算各算法性能指标
            lgbm_metrics = calculate_metrics(lgbm_predictions[:actual_test_days] if lgbm_predictions else [], true_values)
            offset_metrics = calculate_metrics(offset_predictions[:actual_test_days], true_values)
            final_metrics = calculate_metrics(final_predictions, true_values)
            
            performance = {
                'test_period': {
                    'start_date': test_start_date,
                    'requested_days': test_days,
                    'actual_days': actual_test_days,
                    'available_dates': available_test_dates
                },
                'true_values': true_values,
                'lgbm_performance': {
                    'predictions': lgbm_predictions[:actual_test_days] if lgbm_predictions else [0.0] * actual_test_days,
                    'error': lgbm_error,
                    'metrics': lgbm_metrics
                },
                'offset_performance': {
                    'predictions': offset_predictions[:actual_test_days],
                    'info': offset_info,
                    'metrics': offset_metrics
                },
                'final_performance': {
                    'predictions': final_predictions,
                    'weights': weights,
                    'metrics': final_metrics
                },
                'best_algorithm': min(['lgbm', 'offset', 'final'], 
                                    key=lambda x: locals()[f'{x}_metrics']['mse'])
            }
            
            logger.info(f"性能评估完成 - 最佳算法: {performance['best_algorithm']}")
            logger.info(f"LightGBM MSE: {lgbm_metrics['mse']:.2f}, 偏移 MSE: {offset_metrics['mse']:.2f}, 融合 MSE: {final_metrics['mse']:.2f}")
            
            return performance
            
        except Exception as e:
            error_msg = f"性能评估过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {'error': error_msg}

    def auto_optimize_weights(self, line_data: pd.DataFrame, line_no: str,
                             test_start_date: str, test_days: int = 7,
                             model_version: Optional[str] = None,
                             weight_candidates: Optional[List[Tuple[float, float]]] = None) -> Dict:
        """自动优化算法权重配置"""
        logger.info(f"开始为线路 {line_no} 自动优化权重配置")
        
        if weight_candidates is None:
            weight_candidates = [
                (1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4),
                (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.0, 1.0)
            ]
            
        try:
            # 记录原始权重
            original_weights = self.line_algorithm_weights.get(line_no, self.global_algorithm_weights).copy()
            best_weights = None
            best_mse = float('inf')
            optimization_results = []
            
            for lgbm_weight, offset_weight in weight_candidates:
                # 设置权重
                self.update_algorithm_weights({
                    "lgbm": lgbm_weight,
                    "last_year_offset": offset_weight
                }, line_no=line_no)
                
                # 评估性能
                performance = self.get_algorithm_performance(
                    line_data, line_no, test_start_date, test_days, model_version
                )
                
                if 'error' not in performance:
                    mse = performance['final_performance']['metrics']['mse']
                    mae = performance['final_performance']['metrics']['mae']
                    mape = performance['final_performance']['metrics']['mape']
                    
                    result = {
                        'lgbm_weight': lgbm_weight,
                        'offset_weight': offset_weight,
                        'mse': mse,
                        'mae': mae,
                        'mape': mape
                    }
                    optimization_results.append(result)
                    
                    logger.info(f"线路{line_no}权重({lgbm_weight:.1f}, {offset_weight:.1f}) - MSE: {mse:.2f}, MAE: {mae:.2f}")
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_weights = (lgbm_weight, offset_weight)
                else:
                    logger.warning(f"线路{line_no}权重({lgbm_weight:.1f}, {offset_weight:.1f})评估失败")
                    
            # 设置最佳权重
            if best_weights:
                self.update_algorithm_weights({
                    "lgbm": best_weights[0],
                    "last_year_offset": best_weights[1]
                }, line_no=line_no)
                logger.info(f"找到线路{line_no}最佳权重: LightGBM={best_weights[0]:.1f}, 偏移={best_weights[1]:.1f}, MSE={best_mse:.2f}")
            else:
                # 恢复原始权重
                self.line_algorithm_weights[line_no] = original_weights
                logger.warning(f"未找到有效权重配置，线路{line_no}恢复原始设置")
                
            return {
                'line_no': line_no,
                'original_weights': original_weights,
                'best_weights': {
                    "lgbm": best_weights[0] if best_weights else None,
                    "last_year_offset": best_weights[1] if best_weights else None
                } if best_weights else None,
                'best_mse': best_mse if best_weights else None,
                'optimization_results': optimization_results,
                'total_candidates': len(weight_candidates),
                'successful_evaluations': len(optimization_results)
            }
            
        except Exception as e:
            # 恢复原始权重
            self.line_algorithm_weights[line_no] = original_weights
            error_msg = f"权重优化过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {'error': error_msg}

    def update_lgbm_params(self, new_params: Dict) -> None:
        """更新LightGBM参数配置"""
        if not isinstance(new_params, dict):
            raise ValueError("参数配置必须是字典类型")
            
        old_params = self.lgbm_params.copy()
        self.lgbm_params.update(new_params)
        logger.info(f"LightGBM参数已更新: {new_params}")
        
    def get_feature_importance(self, line_no: str, model_version: Optional[str] = None) -> Optional[Dict]:
        """获取特征重要性"""
        try:
            version = model_version or self.version
            model_path = os.path.join(self.model_dir, f"lgbm_line_{line_no}_daily_v{version}.pkl")
            
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return None
                
            model = joblib.load(model_path)
            importance_gain = model.feature_importance(importance_type='gain')
            importance_split = model.feature_importance(importance_type='split')
            
            # 构建特征名称列表
            time_features = [
                'month', 'day_of_month', 'quarter', 'day_of_year', 'week_of_year',
                'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos',
                'flow_lag_1', 'flow_lag_2', 'flow_lag_3', 'flow_lag_7', 'flow_lag_14',
                'flow_ma_3', 'flow_ma_7', 'flow_ma_14', 'flow_std_3', 'flow_std_7', 'flow_std_14'
            ]
            
            extended_factors = self.factors + time_features
            feature_names = extended_factors[:len(importance_gain)]
            
            feature_importance = {
                'feature_names': feature_names,
                'importance_gain': importance_gain.tolist(),
                'importance_split': importance_split.tolist(),
                'top_features_by_gain': sorted(
                    zip(feature_names, importance_gain), 
                    key=lambda x: x[1], reverse=True
                )[:10]
            }
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"获取特征重要性失败: {str(e)}")
            return None