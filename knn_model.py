# KNN 模型模块：修复数据类型问题
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
    'F_WEEK', 'F_HOLIDAYTYPE', 'F_HOLIDAYDAYS',
    'WEATHER_TYPE'
]

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
        logger.info(f"初始化KNN预测器 - 版本: {self.version}, 因子数量: {len(self.factors)}")

    def _ensure_numeric_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """确保指定列为数值类型"""
        df = data.copy()
        for col in columns:
            if col == 'weekday':
                continue
            if col not in df.columns:
                df[col] = 0.0
                logger.warning(f"因子 {col} 不存在，填充为0")
            else:
                # 强制转换为数值类型，无法转换的设为0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"开始准备数据 - 原始数据形状: {data.shape}")

        # 检查必要列
        required_cols = ['F_DATE', 'F_KLCOUNT']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要列: {missing_cols}")

        df = data.sort_values('F_DATE').copy()
        
        # 添加weekday因子
        if 'weekday' in self.factors:
            df['weekday'] = df['F_DATE'].apply(
                lambda x: datetime.strptime(str(x), '%Y%m%d').weekday()
            )

        # 确保所有因子为数值类型
        df = self._ensure_numeric_data(df, self.factors)

        # 构建特征矩阵
        X = df[self.factors].values.astype(np.float64)  # 强制转换为float64
        y = df['F_KLCOUNT'].values.astype(np.float64)

        # 检查数据有效性
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.error("特征矩阵包含无效值")
            raise ValueError("特征矩阵包含无效值")

        if np.any(y < 0):
            logger.error("目标变量包含负值")
            raise ValueError("目标变量包含负值")

        # 对数变换
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

            # 网格搜索最佳K值
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

            # 计算最终评估指标
            X_scaled = best_scaler.transform(X)
            y_pred_transformed = best_model.predict(X_scaled)
            
            y_true_original = np.expm1(y)
            y_pred_original = np.maximum(np.expm1(y_pred_transformed), 0)
            
            mse = mean_squared_error(y_true_original, y_pred_original)
            mae = mean_absolute_error(y_true_original, y_pred_original)
            
            logger.info(f"最佳模型: K={best_k}, MSE={mse:.2f}, MAE={mae:.2f}")

            # 保存模型
            version = model_version or self.version
            model_path = os.path.join(self.model_dir, f"knn_line_{line_no}_daily_v{version}.pkl")
            scaler_path = os.path.join(self.model_dir, f"knn_scaler_line_{line_no}_daily_v{version}.pkl")
            
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
        
        try:
            # 加载模型
            version = model_version or self.version
            model_path = os.path.join(self.model_dir, f"knn_line_{line_no}_daily_v{version}.pkl")
            scaler_path = os.path.join(self.model_dir, f"knn_scaler_line_{line_no}_daily_v{version}.pkl")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return None, f"模型文件未找到: {model_path}"

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logger.info(f"成功加载模型: K={getattr(model, 'n_neighbors', '未知')}")

            # 生成预测日期
            predict_dt = datetime.strptime(predict_start_date, '%Y%m%d')
            pred_dates = [predict_dt + timedelta(days=d) for d in range(days)]

            # 准备预测特征
            if factor_df is not None:
                logger.info("使用用户提供的因子数据")
                if factor_df.shape[0] != days:
                    return None, f"factor_df 行数({factor_df.shape[0]})与预测天数({days})不一致"
                
                pred_df = factor_df.copy()
                # 确保所有因子列存在且为数值类型
                for col in self.factors:
                    if col not in pred_df.columns:
                        pred_df[col] = 0.0
                factor_row = factor_df.to_dict()
                
                pred_df = self._ensure_numeric_data(pred_df, self.factors)
                
            else:
                logger.warning("未提供因子数据，使用历史数据填充")
                if line_data.empty:
                    return None, "历史数据为空，无法生成默认因子"
                
                last_row = line_data.sort_values('F_DATE').iloc[-1]
                pred_df = pd.DataFrame(index=range(days))
                
                for col in self.factors:
                    if col == 'weekday':
                        pred_df[col] = [dt.weekday() for dt in pred_dates]
                    elif col in last_row.index:
                        val = float(last_row[col]) if pd.notna(last_row[col]) else 0.0
                        pred_df[col] = [val] * days
                    else:
                        pred_df[col] = [0.0] * days
                
                pred_df = self._ensure_numeric_data(pred_df, self.factors)

            # 构建预测特征矩阵
            X_pred = pred_df[self.factors].values.astype(np.float64)  # 强制转换为float64
            logger.info(f"预测特征矩阵形状: {X_pred.shape}")

            # 数据有效性检查（现在应该不会出错了）
            if np.any(np.isnan(X_pred)) or np.any(np.isinf(X_pred)):
                return None, "预测特征矩阵包含无效值"

            # 标准化和预测
            X_pred_scaled = scaler.transform(X_pred)
            predictions_transformed = model.predict(X_pred_scaled)
            
            # 逆变换
            predictions = np.maximum(np.expm1(predictions_transformed), 0)
            
            logger.info(f"预测完成 - 结果范围: [{predictions.min():.2f}, {predictions.max():.2f}]")
            
            # 检查预测结果
            if len(predictions) != days:
                return None, f"预测结果长度错误: {len(predictions)} != {days}"
            
            zero_count = (predictions == 0).sum()
            if zero_count > 0:
                logger.warning(f"有 {zero_count}/{len(predictions)} 个预测结果为0")

            return predictions.tolist(), None

        except Exception as e:
            error_msg = f"预测过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

    def save_model_info(self, line_no: str, algorithm: str, mse: Optional[float], 
                       mae: Optional[float], train_date: str, 
                       model_version: Optional[str] = None) -> None:
        import json
        version = model_version or self.version
        info = {
            'algorithm': algorithm,
            'mse': float(mse) if mse is not None else None,
            'mae': float(mae) if mae is not None else None,
            'train_date': train_date,
            'line_no': line_no,
            'version': version,
            'factors': self.factors,
            'config': self.config
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
            X, y = self.prepare_data(line_data)
            for i, factor in enumerate(self.factors):
                factor_std = X[:, i].std()
                if factor_std == 0:
                    diagnosis['data_issues'].append(f"因子 {factor} 方差为0，无区分度")
            
            # 检查模型文件
            version = self.version
            model_path = os.path.join(self.model_dir, f"knn_line_{line_no}_daily_v{version}.pkl")
            if not os.path.exists(model_path):
                diagnosis['model_issues'].append("模型文件不存在")
            
            # 给出建议
            if diagnosis['data_issues']:
                diagnosis['recommendations'].append("检查数据质量，确保目标变量和特征变量有足够变化")
            if diagnosis['model_issues']:
                diagnosis['recommendations'].append("重新训练模型")
            
            diagnosis['recommendations'].extend([
                "确保预测因子与训练因子分布一致",
                "尝试不同的K值参数",
                "考虑特征工程或数据预处理"
            ])
            
        except Exception as e:
            diagnosis['data_issues'].append(f"诊断过程出错: {str(e)}")
        
        logger.info(f"诊断完成: {diagnosis}")
        return diagnosis