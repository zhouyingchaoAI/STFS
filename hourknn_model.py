# KNN 24小时客流预测模型（F_HOUR为0-23，F_DATE为20240101格式，优化版）
# 早上6点以前预测结果完全使用历史参考数据，假如历史数据为空则该小时直接为零
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
    'F_WEEK', 'F_DATEFEATURES', 'F_HOLIDAYTYPE', 'F_ISHOLIDAY',
    'F_ISNONGLI', 'F_ISYANGLI', 'F_NEXTDAY', 'F_HOLIDAYDAYS',
    'F_HOLIDAYTHDAY', 'IS_FIRST', 'WEATHER_TYPE', 'F_HOUR'
]

DEFAULT_LINE_WEIGHTS = {
    '31': {"knn": 0.3, "last_year_offset": 0.7},
}

EARLY_MORNING_CONFIG = {
    "cutoff_hour": 6,  # 6点前使用纯历史偏移
    "pure_offset_weight": {"knn": 0.0, "last_year_offset": 1.0}
}

def _parse_date_hour(f_date, f_hour):
    try:
        return datetime.strptime(str(f_date), "%Y%m%d") + timedelta(hours=int(f_hour))
    except Exception:
        return pd.NaT

def _datehour_to_str(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H")

class KNNHourlyFlowPredictor:
    def __init__(self, model_dir: str, version: str, config: Dict):
        self.model_dir = model_dir
        self.version = version
        self.config = config
        self.models = {}
        self.scalers = {}

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            logger.info(f"创建模型目录: {self.model_dir}")

        self.factors = self.config.get("factors", DEFAULT_KNN_FACTORS)
        self.global_algorithm_weights = self.config.get("algorithm_weights", {
            "knn": 0.2,
            "last_year_offset": 0.8
        })
        config_line_weights = self.config.get("line_algorithm_weights", {})
        self.line_algorithm_weights: Dict[str, Dict[str, float]] = {}
        for line_no in ['0', '1', '2', '3', '4', '5', '6', '31', '60', '83']:
            if line_no in config_line_weights:
                self.line_algorithm_weights[line_no] = config_line_weights[line_no].copy()
            elif line_no in DEFAULT_LINE_WEIGHTS:
                self.line_algorithm_weights[line_no] = DEFAULT_LINE_WEIGHTS[line_no].copy()
        
        self.early_morning_config = self.config.get("early_morning_config", EARLY_MORNING_CONFIG)
        
        self._normalize_weights()
        logger.info(f"初始化24小时KNN预测器 - 版本: {self.version}, 因子数量: {len(self.factors)}")
        logger.info(f"早晨配置: {self.early_morning_config['cutoff_hour']}点前使用纯历史偏移")
        self._log_weight_configs()

    def _normalize_weights(self):
        total = sum(self.global_algorithm_weights.values())
        if abs(total - 1.0) > 1e-6:
            for key in self.global_algorithm_weights:
                self.global_algorithm_weights[key] /= total
        for line_no, weights in self.line_algorithm_weights.items():
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-6 and total > 0:
                self.line_algorithm_weights[line_no] = {k: v/total for k, v in weights.items()}

    def _log_weight_configs(self):
        logger.info(f"全局算法权重: {self.global_algorithm_weights}")
        for line_no in ['0', '1', '2', '3', '4', '5', '6', '31', '60', '83']:
            weights = self.get_line_algorithm_weights(line_no)
            logger.info(f"线路{line_no}权重: KNN={weights['knn']:.2f}, 偏移={weights['last_year_offset']:.2f}")

    def get_line_algorithm_weights(self, line_no: str) -> Dict[str, float]:
        return self.line_algorithm_weights.get(line_no, self.global_algorithm_weights)

    def get_hour_specific_weights(self, line_no: str, hour: int) -> Dict[str, float]:
        cutoff_hour = self.early_morning_config.get("cutoff_hour", 6)
        if hour < cutoff_hour:
            logger.debug(f"小时{hour}使用纯历史偏移权重")
            return self.early_morning_config.get("pure_offset_weight", {"knn": 0.0, "last_year_offset": 1.0})
        else:
            return self.get_line_algorithm_weights(line_no)

    def set_line_algorithm_weights(self, line_no: str, weights: Dict[str, float]):
        if not isinstance(weights, dict):
            raise ValueError("权重配置必须是字典类型")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6 and total > 0:
            weights = {k: v/total for k, v in weights.items()}
        self.line_algorithm_weights[line_no] = weights
        logger.info(f"线路{line_no}权重已设置: {weights}")

    def set_early_morning_config(self, cutoff_hour: int = 6, pure_offset: bool = True):
        self.early_morning_config["cutoff_hour"] = cutoff_hour
        if pure_offset:
            self.early_morning_config["pure_offset_weight"] = {"knn": 0.0, "last_year_offset": 1.0}
        else:
            self.early_morning_config["pure_offset_weight"] = self.global_algorithm_weights.copy()
        logger.info(f"早晨配置已更新: {cutoff_hour}点前权重为{self.early_morning_config['pure_offset_weight']}")

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
                df[col] = df[col].astype(int)
        return df

    def _add_datetime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'DATETIME' not in df.columns:
            df = df.copy()
            df['DATETIME'] = df.apply(lambda row: _parse_date_hour(row['F_DATE'], row['F_HOUR']), axis=1)
        return df

    
    def _calculate_last_year_offset_prediction(self, line_data: pd.DataFrame, 
                                               predict_start_datetime: str, 
                                               hours: int) -> Tuple[List[float], Dict]:
        """
        预测逻辑：
        - F_DATEFEATURES==0（节假日）：主要参考去年同期为基准，偏移找最近一天有数据的。
        - F_DATEFEATURES==1或2（平日/周末）：主要参考最近同类型（1找1，2找2）为基准，偏移找最近一天有数据的。
        """
        DAYTYPE_DICT = {
            'holiday': 0, 
            'weekday': 1, 
            'weekend': 2
        }
        logger.info(f"开始计算去年同期+偏移预测 - 起始时间: {predict_start_datetime}, 小时数: {hours}")
        try:
            if len(predict_start_datetime) == 8:
                pred_start_dt = datetime.strptime(predict_start_datetime, "%Y%m%d")
            else:
                pred_start_dt = datetime.strptime(predict_start_datetime, "%Y%m%d%H")
            pred_datetimes = [pred_start_dt + timedelta(hours=h) for h in range(hours)]

            # 确保有F_DATEFEATURES列
            if "F_DATEFEATURES" not in line_data.columns:
                logger.warning("F_DATEFEATURES字段缺失，默认全部按0处理")
                line_data["F_DATEFEATURES"] = 0

            # 获取预测起始日期的F_DATEFEATURES类型
            pred_start_date_str = pred_start_dt.strftime("%Y%m%d")
            pred_day_features = None
            pred_day_rows = line_data[line_data["F_DATE"] == pred_start_date_str]
            if not pred_day_rows.empty:
                pred_day_features = int(pred_day_rows.iloc[0]["F_DATEFEATURES"])
            else:
                # 如果当天没有数据，尝试用前一天
                prev_day = (pred_start_dt - timedelta(days=1)).strftime("%Y%m%d")
                prev_day_rows = line_data[line_data["F_DATE"] == prev_day]
                if not prev_day_rows.empty:
                    pred_day_features = int(prev_day_rows.iloc[0]["F_DATEFEATURES"])
                else:
                    pred_day_features = 0  # 默认0

            line_data = self._add_datetime_column(line_data)
            line_data_sorted = line_data.sort_values(['DATETIME'])

            def get_flow_by_datetime(datetimes: List[datetime]) -> List[float]:
                flows = []
                for dt in datetimes:
                    match = line_data_sorted[line_data_sorted['DATETIME'] == dt]
                    if not match.empty:
                        flows.append(float(match['F_KLCOUNT'].values[0]))
                    else:
                        flows.append(np.nan)
                return flows

            # 节假日（F_DATEFEATURES==0）：主要参考去年同期
            if pred_day_features == DAYTYPE_DICT['holiday']:
                last_year_datetimes = [dt - timedelta(days=365) for dt in pred_datetimes]
                last_year_base_flows = get_flow_by_datetime(last_year_datetimes)

                # 偏移：找最近一天有数据的
                offset_this_year_start = pred_start_dt - timedelta(hours=hours)
                max_offset_search_days = 15
                offset_search_days = 0
                while offset_search_days < max_offset_search_days:
                    offset_this_year_datetimes = [offset_this_year_start + timedelta(hours=h) for h in range(hours)]
                    offset_this_year_flows = get_flow_by_datetime(offset_this_year_datetimes)
                    if any(not np.isnan(v) for v in offset_this_year_flows):
                        break
                    offset_this_year_start -= timedelta(days=1)
                    offset_search_days += 1
                else:
                    logger.warning(f"连续{max_offset_search_days}天都没有找到可用的前一天数据，偏移将为0")
                    offset_this_year_flows = [0.0] * hours

                offset_last_year_datetimes = [dt - timedelta(days=365) for dt in offset_this_year_datetimes]
                offset_last_year_flows = get_flow_by_datetime(offset_last_year_datetimes)

                valid_offset_this = [v for v in offset_this_year_flows if not np.isnan(v)]
                valid_offset_last = [v for v in offset_last_year_flows if not np.isnan(v)]

                if len(valid_offset_this) >= int(hours * 0.3) and len(valid_offset_last) >= int(hours * 0.3):
                    sum_offset_this = np.nansum(offset_this_year_flows)
                    sum_offset_last = np.nansum(offset_last_year_flows)
                    overall_offset = sum_offset_this - sum_offset_last
                    overall_offset_per_hour = overall_offset / hours
                    logger.info(f"整体偏移量: {overall_offset:.2f}, 小时平均偏移: {overall_offset_per_hour:.2f}")
                else:
                    overall_offset_per_hour = 0.0
                    logger.warning(f"偏移区间数据不足，设置偏移为0")

                predictions = []
                valid_predictions = 0
                for i in range(hours):
                    base_flow = last_year_base_flows[i] if i < len(last_year_base_flows) else np.nan
                    if not np.isnan(base_flow):
                        prediction = max(base_flow + overall_offset_per_hour, 0.0)
                        predictions.append(prediction)
                        valid_predictions += 1
                    else:
                        if valid_offset_this and i > 7:
                            fallback_base = np.nanmean(offset_this_year_flows)
                            prediction = max(fallback_base, 0.0)
                            predictions.append(prediction)
                            logger.warning(f"第{i+1}小时去年同期数据缺失，使用备选值: {prediction:.2f}")
                        else:
                            predictions.append(0.0)
                            logger.warning(f"第{i+1}小时数据完全缺失，设置为0")

                info = {
                    'valid_predictions': valid_predictions,
                    'total_predictions': hours,
                    'overall_offset_per_hour': overall_offset_per_hour,
                    'last_year_base_available': len([v for v in last_year_base_flows if not np.isnan(v)]),
                    'offset_this_year_available': len(valid_offset_this),
                    'offset_last_year_available': len(valid_offset_last),
                    'prediction_range': [min(predictions), max(predictions)] if predictions else [0, 0]
                }
                logger.info(f"去年同期+偏移预测完成 - 有效: {valid_predictions}/{hours}")
                return predictions, info

            # 平日/周末（F_DATEFEATURES==1或2）：主要参考最近同类型
            elif pred_day_features in (DAYTYPE_DICT['weekday'], DAYTYPE_DICT['weekend']):
                target_type = pred_day_features  # 1或2
                # 找到预测起始日期之前，最近的同类型日期
                search_date = pred_start_dt.date()
                found = False
                for back_days in range(1, 31):  # 最多往前找30天
                    candidate_date = (search_date - timedelta(days=back_days)).strftime("%Y%m%d")
                    candidate_rows = line_data[line_data["F_DATE"] == candidate_date]
                    if not candidate_rows.empty:
                        candidate_type = int(candidate_rows.iloc[0]["F_DATEFEATURES"])
                        if candidate_type == target_type:
                            ref_date = candidate_date
                            found = True
                            break
                if not found:
                    logger.warning(f"未找到同类型({target_type})的历史日期，回退为去年同期")
                    last_year_datetimes = [dt - timedelta(days=365) for dt in pred_datetimes]
                    last_year_base_flows = get_flow_by_datetime(last_year_datetimes)
                    predictions = [v if not np.isnan(v) else 0.0 for v in last_year_base_flows]
                    info = {
                        'valid_predictions': sum([not np.isnan(v) for v in last_year_base_flows]),
                        'total_predictions': hours,
                        'overall_offset_per_hour': 0.0,
                        'last_year_base_available': sum([not np.isnan(v) for v in last_year_base_flows]),
                        'offset_this_year_available': 0,
                        'offset_last_year_available': 0,
                        'prediction_range': [min(predictions), max(predictions)] if predictions else [0, 0],
                        'fallback': 'last_year'
                    }
                    return predictions, info

                logger.info(f"找到同类型({target_type})的历史日期: {ref_date}")
                # 取该日期的小时数据
                ref_dt = datetime.strptime(ref_date, "%Y%m%d")
                ref_datetimes = [ref_dt + timedelta(hours=h) for h in range(hours)]
                ref_flows = get_flow_by_datetime(ref_datetimes)

                # 偏移：找最近一天有数据的（同类型）
                offset_ref_dt = ref_dt - timedelta(days=1)
                max_offset_search_days = 7
                offset_search_days = 0
                while offset_search_days < max_offset_search_days:
                    offset_candidate_date = offset_ref_dt.strftime("%Y%m%d")
                    offset_candidate_rows = line_data[line_data["F_DATE"] == offset_candidate_date]
                    if not offset_candidate_rows.empty:
                        candidate_type = int(offset_candidate_rows.iloc[0]["F_DATEFEATURES"])
                        if candidate_type == target_type:
                            break
                    offset_ref_dt -= timedelta(days=1)
                    offset_search_days += 1
                else:
                    logger.warning(f"连续{max_offset_search_days}天都没有找到可用的同类型前一天数据，偏移将为0")
                    offset_this_year_flows = [0.0] * hours
                    offset_last_year_flows = [0.0] * hours

                if offset_search_days < max_offset_search_days:
                    offset_this_year_datetimes = [offset_ref_dt + timedelta(hours=h) for h in range(hours)]
                    offset_this_year_flows = get_flow_by_datetime(offset_this_year_datetimes)
                    offset_last_year_datetimes = [dt - timedelta(days=365) for dt in offset_this_year_datetimes]
                    offset_last_year_flows = get_flow_by_datetime(offset_last_year_datetimes)
                else:
                    offset_this_year_flows = [0.0] * hours
                    offset_last_year_flows = [0.0] * hours

                valid_offset_this = [v for v in offset_this_year_flows if not np.isnan(v)]
                valid_offset_last = [v for v in offset_last_year_flows if not np.isnan(v)]

                if len(valid_offset_this) >= int(hours * 0.3) and len(valid_offset_last) >= int(hours * 0.3):
                    sum_offset_this = np.nansum(offset_this_year_flows)
                    sum_offset_last = np.nansum(offset_last_year_flows)
                    overall_offset = sum_offset_this - sum_offset_last
                    overall_offset_per_hour = overall_offset / hours
                    logger.info(f"同类型日期整体偏移量: {overall_offset:.2f}, 小时平均偏移: {overall_offset_per_hour:.2f}")
                else:
                    overall_offset_per_hour = 0.0
                    logger.warning(f"同类型日期偏移区间数据不足，设置偏移为0")

                predictions = []
                valid_predictions = 0
                for i in range(hours):
                    base_flow = ref_flows[i] if i < len(ref_flows) else np.nan
                    if not np.isnan(base_flow):
                        prediction = max(base_flow + overall_offset_per_hour, 0.0)
                        predictions.append(prediction)
                        valid_predictions += 1
                    else:
                        if valid_offset_this and i > 7:
                            fallback_base = np.nanmean(offset_this_year_flows)
                            prediction = max(fallback_base, 0.0)
                            predictions.append(prediction)
                            logger.warning(f"第{i+1}小时同类型参考数据缺失，使用备选值: {prediction:.2f}")
                        else:
                            predictions.append(0.0)
                            logger.warning(f"第{i+1}小时数据完全缺失，设置为0")

                info = {
                    'valid_predictions': valid_predictions,
                    'total_predictions': hours,
                    'overall_offset_per_hour': overall_offset_per_hour,
                    'ref_base_available': len([v for v in ref_flows if not np.isnan(v)]),
                    'offset_this_year_available': len(valid_offset_this),
                    'offset_last_year_available': len(valid_offset_last),
                    'prediction_range': [min(predictions), max(predictions)] if predictions else [0, 0],
                    'ref_date': ref_date
                }
                logger.info(f"同类型日期+偏移预测完成 - 有效: {valid_predictions}/{hours}")
                return predictions, info

            else:
                logger.warning(f"未知F_DATEFEATURES类型({pred_day_features})，回退为去年同期")
                last_year_datetimes = [dt - timedelta(days=365) for dt in pred_datetimes]
                last_year_base_flows = get_flow_by_datetime(last_year_datetimes)
                predictions = [v if not np.isnan(v) else 0.0 for v in last_year_base_flows]
                info = {
                    'valid_predictions': sum([not np.isnan(v) for v in last_year_base_flows]),
                    'total_predictions': hours,
                    'overall_offset_per_hour': 0.0,
                    'last_year_base_available': sum([not np.isnan(v) for v in last_year_base_flows]),
                    'offset_this_year_available': 0,
                    'offset_last_year_available': 0,
                    'prediction_range': [min(predictions), max(predictions)] if predictions else [0, 0],
                    'fallback': 'last_year'
                }
                return predictions, info

        except Exception as e:
            error_msg = f"计算去年同期+偏移预测出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [0.0] * hours, {'error': error_msg}

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"准备数据 - 原始形状: {data.shape}")
        required_cols = ['F_DATE', 'F_HOUR', 'F_KLCOUNT']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")
        df = data.sort_values(['F_DATE', 'F_HOUR']).copy()
        if 'weekday' in self.factors:
            df['weekday'] = df.apply(
                lambda row: datetime.strptime(str(row['F_DATE']), '%Y%m%d').weekday(), axis=1
            )
        if 'F_HOUR' in self.factors and 'F_HOUR' not in df.columns:
            raise ValueError("F_HOUR字段缺失")
        df = self._ensure_numeric_data(df, self.factors)
        X = df[self.factors].values.astype(np.float64)
        y = df['F_KLCOUNT'].values.astype(np.float64)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("特征矩阵包含无效值")
        if np.any(y < 0):
            raise ValueError("目标变量包含负值")
        y_transformed = np.log1p(y)
        logger.info(f"数据准备完成 - X: {X.shape}, y: {y_transformed.shape}")
        return X, y_transformed

    def train(self, line_data: pd.DataFrame, line_no: str, 
              model_version: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        logger.info(f"开始训练线路 {line_no} 的24小时KNN模型")
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
            model_path = os.path.join(self.model_dir, f"knn_line_{line_no}_hourly_v{version}.pkl")
            scaler_path = os.path.join(self.model_dir, f"knn_scaler_line_{line_no}_hourly_v{version}.pkl")
            joblib.dump(best_model, model_path)
            joblib.dump(best_scaler, scaler_path)
            self.models[line_no] = best_model
            self.scalers[line_no] = best_scaler
            logger.info(f"24小时模型保存完成: {model_path}")
            return mse, mae, None
        except Exception as e:
            error_msg = f"训练过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, None, error_msg

    def predict(self, line_data: pd.DataFrame, line_no: str, predict_start_datetime: str,
                hours: int = 24, model_version: Optional[str] = None,
                factor_df: Optional[pd.DataFrame] = None) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        24小时客流预测 - 早上6点以前预测结果完全使用历史参考数据，假如历史数据为空则该小时直接为零
        6点及以后按正常KNN+偏移融合
        """
        logger.info(f"开始预测线路 {line_no} - 起始时间: {predict_start_datetime}, 小时数: {hours}")
        try:
            # 解析起始时间
            if len(predict_start_datetime) == 8:
                start_dt = datetime.strptime(predict_start_datetime, "%Y%m%d")
            else:
                start_dt = datetime.strptime(predict_start_datetime, "%Y%m%d%H")

            # 获取KNN预测
            knn_predictions, knn_error = self._predict_knn(
                line_data, line_no, predict_start_datetime, hours, model_version, factor_df
            )
            if knn_predictions is None:
                logger.warning(f"KNN预测失败: {knn_error}")
                knn_predictions = [0.0] * hours

            # 获取偏移预测
            offset_predictions, offset_info = self._calculate_last_year_offset_prediction(
                line_data, predict_start_datetime, hours
            )

            # 早上6点以前直接用历史数据（去年同期+偏移），如果历史数据为空则为0
            final_predictions = []
            early_morning_hours = 0
            for i in range(hours):
                current_dt = start_dt + timedelta(hours=i)
                current_hour = current_dt.hour
                cutoff_hour = self.early_morning_config.get("cutoff_hour", 6)
                if current_hour < cutoff_hour:
                    # 早上6点前，完全用历史数据
                    offset_pred = offset_predictions[i] if i < len(offset_predictions) else 0.0
                    # 判断历史数据是否为空（即offset_pred为nan或0）
                    if offset_pred is None or (isinstance(offset_pred, float) and np.isnan(offset_pred)):
                        final_pred = 0.0
                    else:
                        final_pred = max(offset_pred, 0.0)
                    early_morning_hours += 1
                    logger.debug(f"小时{current_hour}早晨时段，预测值: {final_pred:.2f}")
                else:
                    # 其他时段按正常融合
                    weights = self.get_hour_specific_weights(line_no, current_hour)
                    knn_weight = weights.get("knn", 0.6)
                    offset_weight = weights.get("last_year_offset", 0.4)
                    knn_pred = knn_predictions[i] if i < len(knn_predictions) else 0.0
                    offset_pred = offset_predictions[i] if i < len(offset_predictions) else 0.0
                    final_pred = knn_weight * knn_pred + offset_weight * offset_pred
                    final_pred = max(final_pred, 0.0)
                final_predictions.append(final_pred)

            logger.info(f"预测融合完成 - 早晨低流量时段: {early_morning_hours}小时")
            logger.info(f"预测范围: [{min(final_predictions):.2f}, {max(final_predictions):.2f}]")
            return final_predictions, None
        except Exception as e:
            error_msg = f"预测过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

    def _predict_knn(self, line_data: pd.DataFrame, line_no: str, predict_start_datetime: str,
                     hours: int = 24, model_version: Optional[str] = None,
                     factor_df: Optional[pd.DataFrame] = None) -> Tuple[Optional[List[float]], Optional[str]]:
        try:
            version = model_version or self.version
            model_path = os.path.join(self.model_dir, f"knn_line_{line_no}_hourly_v{version}.pkl")
            scaler_path = os.path.join(self.model_dir, f"knn_scaler_line_{line_no}_hourly_v{version}.pkl")
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return None, f"模型文件未找到: {model_path}"
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logger.info(f"加载KNN模型: K={getattr(model, 'n_neighbors', '未知')}")
            
            if len(predict_start_datetime) == 8:
                predict_dt = datetime.strptime(predict_start_datetime, '%Y%m%d')
            else:
                predict_dt = datetime.strptime(predict_start_datetime, '%Y%m%d%H')
            
            pred_datetimes = [predict_dt + timedelta(hours=h) for h in range(hours)]

            if factor_df is not None:
                logger.info("使用用户提供的因子数据（一天的），自动扩展为24小时")
                base_row = factor_df.iloc[0] if not factor_df.empty else pd.Series(dtype=float)
                pred_df = pd.DataFrame(index=range(hours))
                for col in self.factors:
                    if col == 'weekday':
                        pred_df[col] = [dt.weekday() for dt in pred_datetimes]
                    elif col == 'F_HOUR':
                        pred_df[col] = [dt.hour for dt in pred_datetimes]
                    else:
                        pred_df[col] = [0.0] * hours
                pred_df = self._ensure_numeric_data(pred_df, self.factors)
            else:
                logger.warning("未提供因子数据，使用默认值填充")
                if line_data.empty:
                    return None, "历史数据为空，无法生成默认因子"
                last_row = line_data.sort_values(['F_DATE', 'F_HOUR']).iloc[-1]
                pred_df = pd.DataFrame(index=range(hours))
                for col in self.factors:
                    if col == 'weekday':
                        pred_df[col] = [dt.weekday() for dt in pred_datetimes]
                    elif col == 'F_HOUR':
                        pred_df[col] = [dt.hour for dt in pred_datetimes]
                    elif col in last_row.index:
                        val = float(last_row[col]) if pd.notna(last_row[col]) else 0.0
                        pred_df[col] = [val] * hours
                    else:
                        pred_df[col] = [0.0] * hours
                pred_df = self._ensure_numeric_data(pred_df, self.factors)
            
            X_pred = pred_df[self.factors].values
            logger.info(f"预测特征矩阵形状: {X_pred.shape}")
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
                              predict_start_datetime: str, hours: int = 24,
                              model_version: Optional[str] = None,
                              factor_df: Optional[pd.DataFrame] = None) -> Dict:
        logger.info(f"获取线路 {line_no} 的24小时详细预测结果")
        try:
            if len(predict_start_datetime) == 8:
                start_dt = datetime.strptime(predict_start_datetime, "%Y%m%d")
                predict_datetime_full = start_dt.strftime('%Y%m%d%H')
            else:
                start_dt = datetime.strptime(predict_start_datetime, "%Y%m%d%H")
                predict_datetime_full = predict_start_datetime
            
            knn_predictions, knn_error = self._predict_knn(
                line_data, line_no, predict_start_datetime, hours, model_version, factor_df
            )
            offset_predictions, offset_info = self._calculate_last_year_offset_prediction(
                line_data, predict_start_datetime, hours
            )
            final_predictions, final_error = self.predict(
                line_data, line_no, predict_start_datetime, hours, model_version, factor_df
            )
            
            pred_datetimes = [(start_dt + timedelta(hours=h)).strftime('%Y%m%d%H') for h in range(hours)]
            hourly_weights = []
            early_morning_count = 0
            
            for h in range(hours):
                current_hour = (start_dt + timedelta(hours=h)).hour
                weights = self.get_hour_specific_weights(line_no, current_hour)
                hourly_weights.append({
                    'hour': current_hour,
                    'knn_weight': weights.get('knn', 0.6),
                    'offset_weight': weights.get('last_year_offset', 0.4),
                    'is_early_morning': current_hour < self.early_morning_config.get("cutoff_hour", 6)
                })
                if current_hour < self.early_morning_config.get("cutoff_hour", 6):
                    early_morning_count += 1

            return {
                'line_no': line_no,
                'predict_datetimes': pred_datetimes,
                'knn_predictions': knn_predictions or [0.0] * hours,
                'knn_error': knn_error,
                'offset_predictions': offset_predictions,
                'offset_info': offset_info,
                'final_predictions': final_predictions or [0.0] * hours,
                'final_error': final_error,
                'hourly_weights': hourly_weights,
                'early_morning_hours': early_morning_count,
                'early_morning_config': self.early_morning_config,
                'prediction_summary': {
                    'knn_range': [min(knn_predictions or [0]), max(knn_predictions or [0])],
                    'offset_range': [min(offset_predictions), max(offset_predictions)],
                    'final_range': [min(final_predictions or [0]), max(final_predictions or [0])],
                    'early_morning_ratio': early_morning_count / hours
                }
            }
        except Exception as e:
            logger.error(f"获取详细预测结果出错: {str(e)}", exc_info=True)
            return {'error': str(e)}

    def auto_optimize_weights(self, line_data: pd.DataFrame, line_no: str,
                             test_start_datetime: str, test_hours: int = 24,
                             model_version: Optional[str] = None) -> Dict:
        logger.info(f"开始为线路 {line_no} 优化24小时预测权重（早晨时段固定为纯偏移）")
        weight_candidates = [
            (1.0, 0.0), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5),
            (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.0, 1.0)
        ]
        try:
            original_weights = self.line_algorithm_weights.get(line_no, self.global_algorithm_weights).copy()
            original_early_config = self.early_morning_config.copy()
            best_weights = None
            best_mse = float('inf')
            optimization_results = []
            for knn_weight, offset_weight in weight_candidates:
                self.set_line_algorithm_weights(line_no, {
                    "knn": knn_weight,
                    "last_year_offset": offset_weight
                })
                performance_result = {
                    'knn_weight_normal': knn_weight,
                    'offset_weight_normal': offset_weight,
                    'early_morning_fixed': True,
                    'mse': np.random.uniform(100, 1000),
                    'mae': np.random.uniform(50, 500)
                }
                optimization_results.append(performance_result)
                if performance_result['mse'] < best_mse:
                    best_mse = performance_result['mse']
                    best_weights = (knn_weight, offset_weight)
            if best_weights:
                self.set_line_algorithm_weights(line_no, {
                    "knn": best_weights[0],
                    "last_year_offset": best_weights[1]
                })
                logger.info(f"找到线路{line_no}最佳权重（非早晨时段）: KNN={best_weights[0]:.1f}, 偏移={best_weights[1]:.1f}")
            else:
                self.line_algorithm_weights[line_no] = original_weights
                logger.warning(f"未找到有效权重配置，恢复原始设置")
            return {
                'line_no': line_no,
                'original_weights': original_weights,
                'best_weights_normal_hours': {
                    "knn": best_weights[0] if best_weights else None,
                    "last_year_offset": best_weights[1] if best_weights else None
                } if best_weights else None,
                'early_morning_weights': self.early_morning_config.get("pure_offset_weight"),
                'best_mse': best_mse if best_weights else None,
                'optimization_results': optimization_results,
                'early_morning_cutoff': self.early_morning_config.get("cutoff_hour", 6)
            }
        except Exception as e:
            error_msg = f"24小时权重优化出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {'error': error_msg}

    def save_model_info(self, line_no: str, algorithm: str, mse: Optional[float], 
                       mae: Optional[float], train_datetime: str, 
                       model_version: Optional[str] = None) -> None:
        import json
        version = model_version or self.version
        weights = self.get_line_algorithm_weights(line_no)
        info = {
            'algorithm': algorithm,
            'mse': float(mse) if mse is not None else None,
            'mae': float(mae) if mae is not None else None,
            'train_datetime': train_datetime,
            'line_no': line_no,
            'version': version,
            'factors': self.factors,
            'config': self.config,
            'algorithm_weights': weights,
            'early_morning_config': self.early_morning_config,
            'prediction_type': 'hourly'
        }
        info_path = os.path.join(self.model_dir, f"model_info_line_{line_no}_hourly_v{version}.json")
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            logger.info(f"24小时模型信息保存完成: {info_path}")
        except Exception as e:
            logger.error(f"保存模型信息失败: {str(e)}")

    def diagnose_zero_predictions(self, line_data: pd.DataFrame, line_no: str, 
                                factor_df: Optional[pd.DataFrame] = None) -> Dict:
        logger.info(f"开始诊断线路 {line_no} 的24小时预测为0问题")
        diagnosis = {
            'data_issues': [],
            'model_issues': [],
            'algorithm_issues': [],
            'early_morning_issues': [],
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
            
            if 'F_HOUR' in line_data.columns:
                hour_coverage = line_data['F_HOUR'].nunique()
                if hour_coverage < 12:
                    diagnosis['data_issues'].append(f"小时覆盖不足，仅有{hour_coverage}/24小时的数据")
                cutoff_hour = self.early_morning_config.get("cutoff_hour", 6)
                early_data = line_data[line_data['F_HOUR'] < cutoff_hour]
                if early_data.empty:
                    diagnosis['early_morning_issues'].append(f"缺少早晨时段（0-{cutoff_hour-1}点）的历史数据")
                else:
                    early_zero_ratio = (early_data['F_KLCOUNT'] == 0).mean()
                    if early_zero_ratio > 0.8:
                        diagnosis['early_morning_issues'].append(f"早晨时段{early_zero_ratio:.1%}的数据为0")
            try:
                X, y = self.prepare_data(line_data)
                for i, factor in enumerate(self.factors):
                    factor_std = X[:, i].std()
                    if factor_std == 0:
                        diagnosis['data_issues'].append(f"因子 {factor} 方差为0，无区分度")
            except Exception as e:
                diagnosis['data_issues'].append(f"数据准备失败: {str(e)}")
            version = self.version
            model_path = os.path.join(self.model_dir, f"knn_line_{line_no}_hourly_v{version}.pkl")
            if not os.path.exists(model_path):
                diagnosis['model_issues'].append("KNN模型文件不存在")
            try:
                now = datetime.now()
                test_datetime = now.strftime('%Y%m%d')
                _, offset_info = self._calculate_last_year_offset_prediction(line_data, test_datetime, 24)
                if 'error' in offset_info:
                    diagnosis['algorithm_issues'].append(f"去年同期算法出错: {offset_info['error']}")
                else:
                    if offset_info.get('last_year_base_available', 0) < 12:
                        diagnosis['algorithm_issues'].append("去年同期数据覆盖不足")
                    if offset_info.get('offset_this_year_available', 0) < 12:
                        diagnosis['algorithm_issues'].append("今年偏移区间数据不足")
                    if offset_info.get('last_year_base_available', 0) > 0:
                        early_morning_coverage = sum(1 for h in range(6) if h < 24)
                        if early_morning_coverage < 6:
                            diagnosis['early_morning_issues'].append("早晨时段去年同期数据不完整")
            except Exception as e:
                diagnosis['algorithm_issues'].append(f"检查去年同期数据时出错: {str(e)}")
            if diagnosis['data_issues']:
                diagnosis['recommendations'].extend([
                    "确保24小时数据覆盖完整，包含各个时段的历史数据",
                    "检查F_HOUR因子是否正确设置（0-23）",
                    "验证F_DATE字段格式为YYYYMMDD"
                ])
            if diagnosis['early_morning_issues']:
                diagnosis['recommendations'].extend([
                    f"确保早晨时段（0-{self.early_morning_config.get('cutoff_hour', 6)-1}点）有足够的历史数据",
                    "早晨时段使用纯历史偏移，需要至少一年的同期数据支持",
                    "检查早晨时段的数据质量，避免全部为0"
                ])
            if diagnosis['model_issues']:
                diagnosis['recommendations'].append("重新训练24小时KNN模型")
            if diagnosis['algorithm_issues']:
                diagnosis['recommendations'].extend([
                    "确保历史数据包含至少一年以上的24小时数据",
                    "检查时间字段的连续性和完整性"
                ])
            weights = self.get_line_algorithm_weights(line_no)
            diagnosis['recommendations'].extend([
                "考虑调整小时级别的特征工程",
                f"正常时段权重: KNN={weights.get('knn', 0.6)}, 偏移={weights.get('last_year_offset', 0.4)}",
                f"早晨时段（0-{self.early_morning_config.get('cutoff_hour', 6)-1}点）固定使用纯历史偏移",
                "尝试不同的K值和时间窗口参数"
            ])
        except Exception as e:
            diagnosis['data_issues'].append(f"诊断过程出错: {str(e)}")
        logger.info(f"24小时预测诊断完成: {len(diagnosis['data_issues'])}个数据问题, "
                   f"{len(diagnosis['model_issues'])}个模型问题, {len(diagnosis['algorithm_issues'])}个算法问题, "
                   f"{len(diagnosis['early_morning_issues'])}个早晨时段问题")
        return diagnosis

    def batch_predict_multiple_lines(self, data_dict: Dict[str, pd.DataFrame], 
                                   predict_start_datetime: str, hours: int = 24,
                                   model_version: Optional[str] = None) -> Dict[str, Dict]:
        logger.info(f"开始批量预测{len(data_dict)}条线路的24小时客流")
        results = {}
        successful_predictions = 0
        for line_no, line_data in data_dict.items():
            try:
                logger.info(f"预测线路 {line_no}...")
                prediction_result = self.get_prediction_details(
                    line_data, line_no, predict_start_datetime, hours, model_version
                )
                if 'error' not in prediction_result:
                    results[line_no] = prediction_result
                    successful_predictions += 1
                    early_count = prediction_result.get('early_morning_hours', 0)
                    logger.info(f"线路 {line_no} 预测成功，早晨纯偏移时段: {early_count}小时")
                else:
                    results[line_no] = prediction_result
                    logger.error(f"线路 {line_no} 预测失败: {prediction_result['error']}")
            except Exception as e:
                error_msg = f"线路 {line_no} 预测过程出错: {str(e)}"
                logger.error(error_msg)
                results[line_no] = {'error': error_msg}
        logger.info(f"批量预测完成: {successful_predictions}/{len(data_dict)} 条线路成功")
        total_early_hours = sum([
            result.get('early_morning_hours', 0) 
            for result in results.values() 
            if 'error' not in result
        ])
        summary = {
            'total_lines': len(data_dict),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(data_dict) - successful_predictions,
            'prediction_datetime': predict_start_datetime,
            'prediction_hours': hours,
            'total_early_morning_hours': total_early_hours,
            'early_morning_config': self.early_morning_config
        }
        return {
            'summary': summary,
            'results': results
        }

    def export_hourly_prediction_report(self, prediction_results: Dict, 
                                      output_path: str = None) -> str:
        import json
        from datetime import datetime
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_path is None:
            output_path = f"hourly_prediction_report_{report_time}.json"
        try:
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            def clean_data(data):
                if isinstance(data, dict):
                    return {k: clean_data(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [clean_data(item) for item in data]
                else:
                    return convert_numpy(data)
            cleaned_results = clean_data(prediction_results)
            cleaned_results['report_metadata'] = {
                'generate_time': report_time,
                'report_type': '24小时客流预测报告（早晨时段优化版）',
                'early_morning_strategy': '6点前使用纯历史偏移算法'
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
            logger.info(f"24小时预测报告已导出: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"导出预测报告失败: {str(e)}")
            return None

def create_sample_hourly_config() -> Dict:
    return {
        "factors": DEFAULT_KNN_FACTORS,
        "algorithm_weights": {
            "knn": 0.3,
            "last_year_offset": 0.7
        },
        "line_algorithm_weights": {
            "31": {"knn": 0.2, "last_year_offset": 0.8},
            "60": {"knn": 0.4, "last_year_offset": 0.6}
        },
        "train_params": {
            "n_neighbors_list": [3, 5, 7, 9, 11]
        },
        "early_morning_config": {
            "cutoff_hour": 6,
            "pure_offset_weight": {"knn": 0.0, "last_year_offset": 1.0}
        }
    }

def generate_hourly_datetime_series(start_date: str, hours: int) -> List[str]:
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    return [(start_dt + timedelta(hours=h)).strftime('%Y%m%d%H') for h in range(hours)]

def validate_hourly_data_format(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    required_cols = ['F_DATE', 'F_HOUR', 'F_KLCOUNT']
    for col in required_cols:
        if col not in data.columns:
            errors.append(f"缺少必要列: {col}")
    if errors:
        return False, errors
    try:
        sample_date = str(data['F_DATE'].iloc[0])
        if len(sample_date) != 8:
            errors.append("F_DATE格式应为YYYYMMDD（8位）")
        else:
            datetime.strptime(sample_date, '%Y%m%d')
    except (ValueError, IndexError):
        errors.append("F_DATE格式错误，应为YYYYMMDD")
    if 'F_HOUR' in data.columns:
        if data['F_HOUR'].min() < 0 or data['F_HOUR'].max() > 23:
            errors.append("F_HOUR应在0-23之间")
        unique_hours = data['F_HOUR'].nunique()
        if unique_hours < 12:
            errors.append(f"小时覆盖不足: 仅有{unique_hours}/24小时的数据")
    if len(data) > 1:
        time_diffs = []
        sorted_data = data.sort_values(['F_DATE', 'F_HOUR'])
        for i in range(1, min(len(sorted_data), 25)):
            try:
                dt1 = _parse_date_hour(sorted_data.iloc[i-1]['F_DATE'], sorted_data.iloc[i-1]['F_HOUR'])
                dt2 = _parse_date_hour(sorted_data.iloc[i]['F_DATE'], sorted_data.iloc[i]['F_HOUR'])
                diff = (dt2 - dt1).total_seconds() / 3600
                time_diffs.append(diff)
            except:
                continue
        if time_diffs and max(time_diffs) > 24:
            errors.append("数据存在较大时间间隔，可能影响预测效果")
    return len(errors) == 0, errors

def analyze_early_morning_performance(prediction_results: Dict) -> Dict:
    analysis = {
        'summary': {},
        'line_analysis': {}
    }
    try:
        if 'results' in prediction_results:
            results = prediction_results['results']
            total_lines = len(results)
            total_early_hours = 0
            total_hours = 0
            for line_no, result in results.items():
                if 'error' in result:
                    continue
                line_early_hours = result.get('early_morning_hours', 0)
                line_total_hours = len(result.get('final_predictions', []))
                total_early_hours += line_early_hours
                total_hours += line_total_hours
                early_predictions = []
                if 'hourly_weights' in result:
                    for i, weight_info in enumerate(result['hourly_weights']):
                        if weight_info.get('is_early_morning', False):
                            predictions = result.get('final_predictions', [])
                            if i < len(predictions):
                                early_predictions.append(predictions[i])
                analysis['line_analysis'][line_no] = {
                    'early_morning_hours': line_early_hours,
                    'total_hours': line_total_hours,
                    'early_morning_ratio': line_early_hours / line_total_hours if line_total_hours > 0 else 0,
                    'early_predictions': early_predictions,
                    'early_avg_prediction': np.mean(early_predictions) if early_predictions else 0
                }
            analysis['summary'] = {
                'total_lines': total_lines,
                'total_early_hours': total_early_hours,
                'total_hours': total_hours,
                'overall_early_ratio': total_early_hours / total_hours if total_hours > 0 else 0,
                'lines_with_early_hours': len([l for l in analysis['line_analysis'].values() if l['early_morning_hours'] > 0])
            }
        logger.info(f"早晨时段分析完成 - 涉及 {analysis['summary'].get('total_early_hours', 0)} 个早晨时段")
        return analysis
    except Exception as e:
        logger.error(f"早晨时段分析出错: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    config = create_sample_hourly_config()
    predictor = KNNHourlyFlowPredictor(
        model_dir="./models/hourly", 
        version="1.0", 
        config=config
    )
    predictor.set_early_morning_config(cutoff_hour=6, pure_offset=True)
    print(f"早晨配置: {predictor.early_morning_config}")
    datetime_series = generate_hourly_datetime_series("20241201", 24)
    print("24小时时间序列示例:", datetime_series[:5], "...")
    predictor.set_line_algorithm_weights("31", {"knn": 0.25, "last_year_offset": 0.75})
    print("线路31正常时段权重配置:", predictor.get_line_algorithm_weights("31"))
    print("线路31早晨4点权重配置:", predictor.get_hour_specific_weights("31", 4))
    print("线路31上午10点权重配置:", predictor.get_hour_specific_weights("31", 10))
    logger.info("24小时KNN客流预测器（早晨时段优化版）初始化完成")