from sklearn.neighbors import KNeighborsRegressor
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

DEFAULT_KNN_FACTORS = [
    'F_WEEK', 'F_HOLIDAYTYPE', 'F_HOLIDAYDAYS', 'F_HOLIDAYWHICHDAY',
    'F_DAYOFWEEK', 'WEATHER_TYPE', 'F_YEAR'
]
def sanitize_line_no(line_no: str) -> str:
    """清理线路名，将不适合作为文件名的字符替换为安全字符"""
    if not isinstance(line_no, str):
        line_no = str(line_no)
    sanitized = line_no.replace('/', '-').replace('\\', '-').replace(':', '-')
    sanitized = sanitized.replace('*', '-').replace('?', '-').replace('"', '-')
    sanitized = sanitized.replace('<', '-').replace('>', '-').replace('|', '-')
    sanitized = sanitized.strip('. ')
    if not sanitized:
        sanitized = 'unknown'
    return sanitized


# # 预设每条线路的权重配置
# DEFAULT_LINE_WEIGHTS = {
#     # '0': {"knn": 0.6, "last_year_offset": 0.4},
#     # '1': {"knn": 0.7, "last_year_offset": 0.3},
#     # '2': {"knn": 0.5, "last_year_offset": 0.5},
#     # '3': {"knn": 0.6, "last_year_offset": 0.4},
#     # '4': {"knn": 0.8, "last_year_offset": 0.2},
#     # '5': {"knn": 0.4, "last_year_offset": 0.6},
#     # '6': {"knn": 0.6, "last_year_offset": 0.4},
#     # '31': {"knn": 0.1, "last_year_offset": 0.9},
#     # '60': {"knn": 0.7, "last_year_offset": 0.3},
#     # '83': {"knn": 0.6, "last_year_offset": 0.4},
# }

class KNNFlowPredictor:
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

        self.factors = self.config.get("factors", DEFAULT_KNN_FACTORS)
        
        # 支持全局和每线路权重配置
        self.global_algorithm_weights = self.config.get("algorithm_weights", {
            "knn": 0.8,
            "last_year_offset": 0.2
        })
        
        # 初始化每条线路的权重配置，优先使用配置文件中的设置，否则使用默认值
        config_line_weights = self.config.get("line_algorithm_weights", {})
        self.line_algorithm_weights: Dict[str, Dict[str, float]] = {}
        
        # # 为每条线路设置权重配置
        # for line_no in ['0', '1', '2', '3', '4', '5', '6', '31', '60', '83']:
        #     if line_no in config_line_weights:
        #         # 使用配置文件中的权重
        #         self.line_algorithm_weights[line_no] = config_line_weights[line_no].copy()
        #     elif line_no in DEFAULT_LINE_WEIGHTS:
        #         # 使用默认权重配置
        #         self.line_algorithm_weights[line_no] = DEFAULT_LINE_WEIGHTS[line_no].copy()
        #     # 如果都没有，则会使用全局权重（通过get_line_algorithm_weights方法处理）
        
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

        logger.info(f"初始化KNN预测器 - 版本: {self.version}, 因子数量: {len(self.factors)}")
        logger.info(f"全局算法权重配置: {self.global_algorithm_weights}")
        logger.info(f"已配置{len(self.line_algorithm_weights)}条线路的独立权重")
        
        # 打印每条线路的权重配置
        for line_no in ['0', '1', '2', '3', '4', '5', '6', '31', '60', '83']:
            weights = self.get_line_algorithm_weights(line_no)
            logger.info(f"线路{line_no}权重: KNN={weights['knn']:.2f}, 偏移={weights['last_year_offset']:.2f}")

    def get_line_algorithm_weights(self, line_no: str) -> Dict[str, float]:
        # 优先返回每线路权重，否则返回全局权重
        weights = self.line_algorithm_weights.get(line_no)
        if weights is not None:
            # 确保归一化
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-6 and total > 0:
                weights = {k: v/total for k, v in weights.items()}
                self.line_algorithm_weights[line_no] = weights
            return weights
        else:
            return self.global_algorithm_weights

    def set_line_algorithm_weights(self, line_no: str, weights: Dict[str, float]):
        # 设置单线路权重并归一化
        if not isinstance(weights, dict):
            raise ValueError("权重配置必须是字典类型")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6 and total > 0:
            logger.warning(f"线路{line_no}权重总和为{total}，将自动归一化")
            weights = {k: v/total for k, v in weights.items()}
        self.line_algorithm_weights[line_no] = weights
        logger.info(f"线路{line_no}算法权重已设置为: {weights}")

    # def get_all_line_weights(self) -> Dict[str, Dict[str, float]]:
    #     """获取所有线路的权重配置"""
    #     result = {}
    #     for line_no in ['0', '1', '2', '3', '4', '5', '6', '31', '60', '83']:
    #         result[line_no] = self.get_line_algorithm_weights(line_no)
    #     return result

    def set_batch_line_weights(self, line_weights: Dict[str, Dict[str, float]]):
        """批量设置多条线路的权重配置"""
        for line_no, weights in line_weights.items():
            try:
                self.set_line_algorithm_weights(line_no, weights)
            except Exception as e:
                logger.error(f"设置线路{line_no}权重失败: {str(e)}")

    def _ensure_numeric_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """确保指定列为数值类型"""
        df = data.copy()
        for col in columns:
            if col.lower() == 'weekday':
                continue
            # 标准化列名到大写
            actual_col = [c for c in df.columns if c.upper() == col.upper()]
            if actual_col:
                actual_col = actual_col[0]
                df[col] = pd.to_numeric(df[actual_col], errors='coerce').fillna(0.0)
                if actual_col != col:
                    df = df.drop(actual_col, axis=1)
            else:
                df[col] = 0.0
                logger.warning(f"因子 {col} 不存在，填充为0")
        return df

    def _calculate_last_year_offset_prediction(self, line_data: pd.DataFrame, 
                                             predict_start_date: str, 
                                             days: int) -> Tuple[List[float], Dict]:
        """
        计算去年同期+整体偏移的预测结果
        节假日预测时，偏移量上浮20%
        处理闰年：使用date.replace(year=year-1)
        """
        # 引入DAYTYPE_DICT
        DAYTYPE_DICT = {
            'holiday': 0, 
            'weekday': 1, 
            'weekend': 2
        }
        logger.info("开始计算去年同期+整体偏移预测")
        try:
            line_data_sorted = line_data.sort_values('F_DATE')
            pred_start_dt = datetime.strptime(predict_start_date, "%Y%m%d")
            pred_end_dt = pred_start_dt + timedelta(days=days-1)
            # 去年同期：处理闰年
            def get_last_year_date(dt: datetime) -> datetime:
                try:
                    return dt.replace(year=dt.year - 1)
                except ValueError:  # 闰年2月29日
                    return dt.replace(year=dt.year - 1, month=2, day=28)
            last_year_start_dt = get_last_year_date(pred_start_dt)
            last_year_end_dt = get_last_year_date(pred_end_dt)
            offset_this_year_start = pred_start_dt - timedelta(days=days)
            offset_this_year_end = pred_start_dt - timedelta(days=1)
            offset_last_year_start = get_last_year_date(offset_this_year_start)
            offset_last_year_end = get_last_year_date(offset_this_year_end)
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
                logger.warning(f"偏移区间数据不足，设置偏移量为0。今年有效数据: {len(valid_offset_this)}, 去年有效数据: {len(valid_offset_last)}")
            predictions = []
            valid_predictions = 0
            for i in range(days):
                base_flow = last_year_base_flows[i] if i < len(last_year_base_flows) else np.nan
                # 获取预测日的日期类型（基于F_HOLIDAYTYPE和F_DAYOFWEEK计算）
                pred_date = (pred_start_dt + timedelta(days=i)).strftime("%Y%m%d")
                pred_day_rows = line_data_sorted[line_data_sorted['F_DATE'] == pred_date]
                if not pred_day_rows.empty:
                    row = pred_day_rows.iloc[0]
                    # 如果F_HOLIDAYTYPE不为空，则为节假日(0)
                    if pd.notna(row.get('F_HOLIDAYTYPE')) and str(row.get('F_HOLIDAYTYPE')).strip() != '':
                        pred_day_features = 0  # 节假日
                    else:
                        # 如果F_DAYOFWEEK是6或7（周六或周日），则为周末(2)
                        day_of_week = row.get('F_DAYOFWEEK')
                        if pd.notna(day_of_week):
                            try:
                                dow = int(day_of_week)
                                if dow in [6, 7]:
                                    pred_day_features = 2  # 周末
                                else:
                                    pred_day_features = 1  # 平日
                            except (ValueError, TypeError):
                                pred_day_features = 1  # 默认平日
                        else:
                            pred_day_features = 1  # 默认平日
                else:
                    pred_day_features = None
                # 判断是否节假日
                is_holiday = (pred_day_features == DAYTYPE_DICT['holiday']) if pred_day_features is not None else False
                # 偏移量调整
                offset = overall_offset_per_day
                if is_holiday and offset != 0:
                    offset = offset + abs(offset) * 0.2  # 节假日偏移上浮20%（取绝对值）
                if not np.isnan(base_flow):
                    prediction = max(base_flow + offset, 0.0)
                    predictions.append(prediction)
                    valid_predictions += 1
                else:
                    if valid_offset_this:
                        fallback_base = np.nanmean(offset_this_year_flows)
                        prediction = max(fallback_base, 0.0)
                        predictions.append(prediction)
                        logger.warning(f"第{i+1}天去年同期数据缺失，使用偏移区间平均值: {prediction:.2f}")
                    else:
                        predictions.append(0.0)
                        logger.warning(f"第{i+1}天数据完全缺失，设置为0")
            info = {
                'valid_predictions': valid_predictions,
                'total_predictions': days,
                'overall_offset_per_day': overall_offset_per_day,
                'last_year_base_available': len([v for v in last_year_base_flows if not np.isnan(v)]),
                'offset_this_year_available': len(valid_offset_this),
                'offset_last_year_available': len(valid_offset_last),
                'prediction_range': [min(predictions), max(predictions)] if predictions else [0, 0]
            }
            logger.info(f"去年同期+偏移预测完成 - 有效预测: {valid_predictions}/{days}, 预测范围: [{info['prediction_range'][0]:.2f}, {info['prediction_range'][1]:.2f}]")
            return predictions, info
        except Exception as e:
            error_msg = f"计算去年同期+偏移预测时出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [0.0] * days, {'error': error_msg}

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"开始准备数据 - 原始数据形状: {data.shape}")
        required_cols = ['F_DATE', 'F_KLCOUNT']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")
        df = data.sort_values('F_DATE').copy()
        # 日志：训练数据起止日期
        if not df.empty:
            min_date = str(df['F_DATE'].iloc[0])
            max_date = str(df['F_DATE'].iloc[-1])
            logger.info(f"训练数据日期范围: {min_date} ~ {max_date}")
        else:
            logger.warning("训练数据为空，无法显示日期范围")
        if 'weekday' in [f.lower() for f in self.factors]:
            df['weekday'] = df['F_DATE'].apply(
                lambda x: datetime.strptime(str(x), '%Y%m%d').weekday()
            )
        df = self._ensure_numeric_data(df, self.factors)
        X = df[self.factors].values.astype(np.float64)
        y = df['F_KLCOUNT'].values.astype(np.float64)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.error("特征矩阵包含无效值")
            raise ValueError("特征矩阵包含无效值")
        if np.any(y < 0):
            logger.error("目标变量包含负值")
            raise ValueError("目标变量包含负值")
        y_transformed = np.log1p(y)
        logger.info(f"数据准备完成 - X形状: {X.shape}, y形状: {y_transformed.shape}")
        return X, y_transformed

    def train(self, line_data: pd.DataFrame, line_no: str, 
              model_version: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        logger.info(f"开始训练线路 {line_no} 的KNN模型")
        try:
            train_params = self.config.get("train_params", {})
            n_neighbors_list = train_params.get("n_neighbors_list", [3, 5, 7, 9])
            X, y = self.prepare_data(line_data)
            if len(X) < min(n_neighbors_list):
                error_msg = f"数据量不足 - 样本数: {len(X)}, 最小K值: {min(n_neighbors_list)}"
                logger.error(error_msg)
                return None, None, error_msg
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
                logger.info(f"K={k}, 训练MSE={mse:.6f}")
                if mse < best_mse:
                    best_mse = mse
                    best_model = model
                    best_scaler = scaler
                    best_k = k
            if best_model is None:
                return None, None, "无法找到合适的模型"
            X_scaled = best_scaler.transform(X)
            y_pred_transformed = best_model.predict(X_scaled)
            y_true_original = np.expm1(y)
            y_pred_original = np.maximum(np.expm1(y_pred_transformed), 0)
            mse = mean_squared_error(y_true_original, y_pred_original)
            mae = mean_absolute_error(y_true_original, y_pred_original)
            logger.info(f"最佳模型: K={best_k}, MSE={mse:.2f}, MAE={mae:.2f}")
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

    def predict(self, line_data: pd.DataFrame, line_no: str, predict_start_date: str,
                days: int = 15, model_version: Optional[str] = None,
                factor_df: Optional[pd.DataFrame] = None) -> Tuple[Optional[List[float]], Optional[str]]:
        logger.info(f"开始预测线路 {line_no} - 起始日期: {predict_start_date}, 天数: {days}")

        # if factor_df is not None:
        #     logger.info(f"factor_df 内容: {factor_df.to_string(index=False)}")
        try:
            # 1. KNN预测
            knn_predictions, knn_error = self._predict_knn(
                line_data, line_no, predict_start_date, days, model_version, factor_df
            )
            if knn_predictions is None:
                logger.warning(f"KNN预测失败: {knn_error}")
                knn_predictions = [0.0] * days
            # 2. 去年同期+偏移预测
            offset_predictions, offset_info = self._calculate_last_year_offset_prediction(
                line_data, predict_start_date, days
            )
            # 3. 融合预测结果
            final_predictions = []
            weights = self.get_line_algorithm_weights(line_no)
            knn_weight = weights.get("knn", 0.8)
            offset_weight = weights.get("last_year_offset", 0.2)
            for i in range(days):
                knn_pred = knn_predictions[i] if i < len(knn_predictions) else 0.0
                offset_pred = offset_predictions[i] if i < len(offset_predictions) else 0.0
                final_pred = knn_weight * knn_pred + offset_weight * offset_pred
                final_predictions.append(max(final_pred, 0.0))
            logger.info(f"预测融合完成 - KNN权重: {knn_weight}, 偏移权重: {offset_weight}")
            logger.info(f"最终预测范围: [{min(final_predictions):.2f}, {max(final_predictions):.2f}]")
            if len(final_predictions) != days:
                return None, f"预测结果长度错误: {len(final_predictions)} != {days}"
            zero_count = sum(1 for p in final_predictions if p == 0)
            if zero_count > 0:
                logger.warning(f"有 {zero_count}/{len(final_predictions)} 个预测结果为0")
            return final_predictions, None
        except Exception as e:
            error_msg = f"预测过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

    def _predict_knn(self, line_data: pd.DataFrame, line_no: str, predict_start_date: str,
                     days: int = 15, model_version: Optional[str] = None,
                     factor_df: Optional[pd.DataFrame] = None) -> Tuple[Optional[List[float]], Optional[str]]:
        """KNN预测的原始逻辑"""
        try:
            version = model_version or self.version
            safe_line_no = sanitize_line_no(line_no)
            model_path = os.path.join(self.model_dir, f"knn_line_{safe_line_no}_daily_v{version}.pkl")
            scaler_path = os.path.join(self.model_dir, f"knn_scaler_line_{safe_line_no}_daily_v{version}.pkl")
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return None, f"模型文件未找到: {model_path}"
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logger.info(f"成功加载KNN模型: K={getattr(model, 'n_neighbors', '未知')}")
            predict_dt = datetime.strptime(predict_start_date, '%Y%m%d')
            pred_dates = [predict_dt + timedelta(days=d) for d in range(days)]
            if factor_df is not None:
                logger.info("使用用户提供的因子数据")
                if factor_df.shape[0] != days:
                    return None, f"factor_df 行数({factor_df.shape[0]})与预测天数({days})不一致"
                pred_df = factor_df.copy()
                # 标准化列名到大写
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
            logger.info(f"KNN预测特征矩阵形状: {X_pred.shape}")
            if np.any(np.isnan(X_pred)) or np.any(np.isinf(X_pred)):
                return None, "预测特征矩阵包含无效值"
            X_pred_scaled = scaler.transform(X_pred)
            predictions_transformed = model.predict(X_pred_scaled)
            predictions = np.maximum(np.expm1(predictions_transformed), 0)
            logger.info(f"KNN预测完成 - 结果范围: [{predictions.min():.2f}, {predictions.max():.2f}]")
            return predictions.tolist(), None
        except Exception as e:
            error_msg = f"KNN预测过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

    def get_prediction_details(self, line_data: pd.DataFrame, line_no: str, 
                              predict_start_date: str, days: int = 15,
                              model_version: Optional[str] = None,
                              factor_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        获取详细的预测结果，包含各个算法的单独预测结果
        """
        logger.info(f"获取线路 {line_no} 的详细预测结果")
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
            target_stats = line_data['F_KLCOUNT'].describe()
            zero_ratio = (line_data['F_KLCOUNT'] == 0).mean()
            if zero_ratio > 0.5:
                diagnosis['data_issues'].append(f"训练数据中{zero_ratio:.1%}的目标变量为0")
            if target_stats.get('std', 0) == 0:
                diagnosis['data_issues'].append("目标变量方差为0，无变化")
            try:
                X, y = self.prepare_data(line_data)
                for i, factor in enumerate(self.factors):
                    factor_std = X[:, i].std()
                    if factor_std == 0:
                        diagnosis['data_issues'].append(f"因子 {factor} 方差为0，无区分度")
            except Exception as e:
                diagnosis['data_issues'].append(f"数据准备失败: {str(e)}")
            version = self.version
            safe_line_no = sanitize_line_no(line_no)
            model_path = os.path.join(self.model_dir, f"knn_line_{safe_line_no}_daily_v{version}.pkl")
            if not os.path.exists(model_path):
                diagnosis['model_issues'].append("KNN模型文件不存在")
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
            if diagnosis['data_issues']:
                diagnosis['recommendations'].append("检查数据质量，确保目标变量和特征变量有足够变化")
                diagnosis['recommendations'].append("检查历史数据的完整性和连续性")
            if diagnosis['model_issues']:
                diagnosis['recommendations'].append("重新训练KNN模型")
            if diagnosis['algorithm_issues']:
                diagnosis['recommendations'].append("检查历史数据覆盖范围，确保包含足够的去年同期数据")
                diagnosis['recommendations'].append("考虑调整算法权重配置，降低去年同期+偏移算法的权重")
            weights = self.get_line_algorithm_weights(line_no)
            diagnosis['recommendations'].extend([
                "确保预测因子与训练因子分布一致",
                "尝试不同的K值参数和权重配置",
                "考虑特征工程或数据预处理",
                f"当前算法权重配置: KNN={weights.get('knn', 0.8)}, 去年同期+偏移={weights.get('last_year_offset', 0.2)}"
            ])
        except Exception as e:
            diagnosis['data_issues'].append(f"诊断过程出错: {str(e)}")
        logger.info(f"诊断完成: {diagnosis}")
        return diagnosis

    def update_algorithm_weights(self, new_weights: Dict[str, float], line_no: Optional[str] = None) -> None:
        """
        更新算法权重配置
        参数:
            new_weights: 新的权重配置字典，如 {"knn": 0.7, "last_year_offset": 0.3}
            line_no: 若指定则为该线路单独设置权重，否则为全局设置
        """
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
            self.global_algorithm_weights = new_weights
            logger.info(f"全局算法权重已更新: {old_weights} -> {self.global_algorithm_weights}")

    def get_algorithm_performance(self, line_data: pd.DataFrame, line_no: str,
                                 test_start_date: str, test_days: int = 7,
                                 model_version: Optional[str] = None) -> Dict:
        """
        评估不同算法在历史数据上的性能表现
        """
        logger.info(f"开始评估线路 {line_no} 的算法性能")
        try:
            test_start_dt = datetime.strptime(test_start_date, '%Y%m%d')
            test_dates = [(test_start_dt + timedelta(days=d)).strftime('%Y%m%d') 
                         for d in range(test_days)]
            line_data_sorted = line_data.sort_values('F_DATE')
            true_values = []
            available_test_dates = []
            for date_str in test_dates:
                match = line_data_sorted[line_data_sorted['F_DATE'] == date_str]
                if not match.empty:
                    true_values.append(float(match['F_KLCOUNT'].values[0]))
                    available_test_dates.append(date_str)
            if not true_values:
                return {'error': f'测试期间 {test_start_date} 后 {test_days} 天内无可用数据'}
            train_data = line_data_sorted[line_data_sorted['F_DATE'] < test_start_date]
            if train_data.empty:
                return {'error': '训练数据为空'}
            actual_test_days = len(available_test_dates)
            knn_predictions, knn_error = self._predict_knn(
                train_data, line_no, test_start_date, actual_test_days, model_version
            )
            offset_predictions, offset_info = self._calculate_last_year_offset_prediction(
                train_data, test_start_date, actual_test_days
            )
            weights = self.get_line_algorithm_weights(line_no)
            knn_weight = weights.get("knn", 0.8)
            offset_weight = weights.get("last_year_offset", 0.2)
            if knn_predictions and len(knn_predictions) >= actual_test_days:
                final_predictions = []
                for i in range(actual_test_days):
                    knn_pred = knn_predictions[i] if knn_predictions else 0.0
                    offset_pred = offset_predictions[i] if i < len(offset_predictions) else 0.0
                    final_pred = knn_weight * knn_pred + offset_weight * offset_pred
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
                non_zero_mask = true_vals != 0
                if non_zero_mask.any():
                    mape = np.mean(np.abs((true_vals[non_zero_mask] - predictions[non_zero_mask]) 
                                        / true_vals[non_zero_mask])) * 100
                else:
                    mape = float('inf')
                return {'mse': float(mse), 'mae': float(mae), 'mape': float(mape)}
            knn_metrics = calculate_metrics(knn_predictions[:actual_test_days] if knn_predictions else [], true_values)
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
                'knn_performance': {
                    'predictions': knn_predictions[:actual_test_days] if knn_predictions else [0.0] * actual_test_days,
                    'error': knn_error,
                    'metrics': knn_metrics
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
                'best_algorithm': min(['knn', 'offset', 'final'], 
                                    key=lambda x: locals()[f'{x}_metrics']['mse'])
            }
            logger.info(f"性能评估完成 - 最佳算法: {performance['best_algorithm']}")
            logger.info(f"KNN MSE: {knn_metrics['mse']:.2f}, 偏移 MSE: {offset_metrics['mse']:.2f}, 融合 MSE: {final_metrics['mse']:.2f}")
            return performance
        except Exception as e:
            error_msg = f"性能评估过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {'error': error_msg}

    def auto_optimize_weights(self, line_data: pd.DataFrame, line_no: str,
                             test_start_date: str, test_days: int = 7,
                             model_version: Optional[str] = None,
                             weight_candidates: Optional[List[Tuple[float, float]]] = None) -> Dict:
        """
        自动优化算法权重配置
        """
        logger.info(f"开始为线路 {line_no} 自动优化权重配置")
        if weight_candidates is None:
            weight_candidates = [
                (1.0, 0.0),
                (0.9, 0.1),
                (0.8, 0.2),
                (0.7, 0.3),
                (0.6, 0.4),
                (0.5, 0.5),
                (0.4, 0.6),
                (0.3, 0.7),
                (0.2, 0.8),
                (0.1, 0.9),
                (0.0, 1.0),
            ]
        try:
            # 记录原始线路权重
            original_weights = self.line_algorithm_weights.get(line_no, self.global_algorithm_weights).copy()
            best_weights = None
            best_mse = float('inf')
            optimization_results = []
            for knn_weight, offset_weight in weight_candidates:
                # 单线路权重设置
                self.update_algorithm_weights({
                    "knn": knn_weight,
                    "last_year_offset": offset_weight
                }, line_no=line_no)
                performance = self.get_algorithm_performance(
                    line_data, line_no, test_start_date, test_days, model_version
                )
                if 'error' not in performance:
                    mse = performance['final_performance']['metrics']['mse']
                    mae = performance['final_performance']['metrics']['mae']
                    mape = performance['final_performance']['metrics']['mape']
                    result = {
                        'knn_weight': knn_weight,
                        'offset_weight': offset_weight,
                        'mse': mse,
                        'mae': mae,
                        'mape': mape
                    }
                    optimization_results.append(result)
                    logger.info(f"线路{line_no}权重({knn_weight:.1f}, {offset_weight:.1f}) - MSE: {mse:.2f}, MAE: {mae:.2f}")
                    if mse < best_mse:
                        best_mse = mse
                        best_weights = (knn_weight, offset_weight)
                else:
                    logger.warning(f"线路{line_no}权重({knn_weight:.1f}, {offset_weight:.1f})评估失败: {performance.get('error', '未知错误')}")
            # 设置最佳权重
            if best_weights:
                self.update_algorithm_weights({
                    "knn": best_weights[0],
                    "last_year_offset": best_weights[1]
                }, line_no=line_no)
                logger.info(f"找到线路{line_no}最佳权重配置: KNN={best_weights[0]:.1f}, 偏移={best_weights[1]:.1f}, MSE={best_mse:.2f}")
            else:
                # 恢复原始权重
                self.line_algorithm_weights[line_no] = original_weights
                logger.warning(f"未找到有效的权重配置，线路{line_no}恢复原始设置")
            return {
                'line_no': line_no,
                'original_weights': original_weights,
                'best_weights': {
                    "knn": best_weights[0] if best_weights else None,
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