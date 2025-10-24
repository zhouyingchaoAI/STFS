# Transformer 模型模块：实现 Transformer 用于日客流预测
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src):
        # src: (batch, seq_len, input_dim)
        x = self.input_linear(src)
        x = self.transformer_encoder(x)
        out = self.output_linear(x)
        return out.squeeze(-1)  # (batch, seq_len)

class TransformerFlowPredictor:
    def __init__(self, model_dir: str, version: str, config: Dict):
        """
        Transformer 预测器初始化

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

    def prepare_data(self, data: pd.DataFrame, lookback_days: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备 Transformer 输入数据

        参数:
            data: 日客流数据 DataFrame
            lookback_days: 序列长度

        返回:
            (X, y) 输入和目标数组
        """
        df = data.sort_values('F_DATE').copy()
        df['weekday'] = df['F_DATE'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').weekday())
        features = ['F_KLCOUNT', 'weekday']
        X, y = [], []
        arr = df[features].values
        for i in range(len(arr) - lookback_days):
            X.append(arr[i:i+lookback_days])
            y.append(arr[i+lookback_days][0])  # 预测下一个F_KLCOUNT
        X = np.array(X)  # (num_samples, lookback_days, num_features)
        y = np.array(y)  # (num_samples,)
        return X, y

    def train(self, line_data: pd.DataFrame, line_no: str, model_version: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        训练 Transformer 模型

        参数:
            line_data: 线路数据 DataFrame
            line_no: 线路编号
            model_version: 模型版本（可选）

        返回:
            (mse, mae, error) 元组
        """
        try:
            train_params = self.config.get("train_params", {})
            lookback_days = train_params.get("lookback_days", 7)
            d_model = train_params.get("hidden_size", 64)
            nhead = train_params.get("nhead", 4)
            num_layers = train_params.get("num_layers", 2)
            dropout = train_params.get("dropout", 0.1)
            batch_size = train_params.get("batch_size", 32)
            epochs = train_params.get("epochs", 100)
            patience = train_params.get("patience", 10)
            learning_rate = train_params.get("learning_rate", 0.001)

            X, y = self.prepare_data(line_data, lookback_days)
            if len(X) < 10:
                return None, None, "数据量不足"

            # 归一化
            scaler = MinMaxScaler()
            X_shape = X.shape
            X_reshape = X.reshape(-1, X_shape[-1])
            X_scaled = scaler.fit_transform(X_reshape).reshape(X_shape)
            y = y.astype(np.float32)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TimeSeriesTransformer(input_dim=X.shape[2], d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # Early stopping
            best_loss = float('inf')
            patience_counter = 0
            for epoch in range(epochs):
                model.train()
                permutation = np.random.permutation(len(X_scaled))
                X_shuffled = X_scaled[permutation]
                y_shuffled = y[permutation]
                batch_losses = []
                for i in range(0, len(X_shuffled), batch_size):
                    xb = torch.tensor(X_shuffled[i:i+batch_size], dtype=torch.float32, device=device)
                    yb = torch.tensor(y_shuffled[i:i+batch_size], dtype=torch.float32, device=device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out[:, -1], yb)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                avg_loss = np.mean(batch_losses)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break

            # 恢复最佳模型
            model.load_state_dict(best_model_state)

            # 训练集评估
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
                y_pred = model(X_tensor)[:, -1].cpu().numpy()
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            # 保存模型和scaler
            version = model_version if model_version is not None else self.version
            model_path = os.path.join(self.model_dir, f"transformer_line_{line_no}_daily_v{version}.pt")
            scaler_path = os.path.join(self.model_dir, f"transformer_scaler_line_{line_no}_daily_v{version}.pkl")
            torch.save(model.state_dict(), model_path)
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
            train_params = self.config.get("train_params", {})
            lookback_days = train_params.get("lookback_days", 7)
            d_model = train_params.get("hidden_size", 64)
            nhead = train_params.get("nhead", 4)
            num_layers = train_params.get("num_layers", 2)
            dropout = train_params.get("dropout", 0.1)

            version = model_version if model_version is not None else self.version
            model_path = os.path.join(self.model_dir, f"transformer_line_{line_no}_daily_v{version}.pt")
            scaler_path = os.path.join(self.model_dir, f"transformer_scaler_line_{line_no}_daily_v{version}.pkl")
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return None, "模型文件未找到"

            scaler = joblib.load(scaler_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TimeSeriesTransformer(input_dim=2, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # 构造预测输入序列
            df = line_data.sort_values('F_DATE').copy()
            df['weekday'] = df['F_DATE'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').weekday())
            features = ['F_KLCOUNT', 'weekday']
            arr = df[features].values
            # 取最近lookback_days天作为初始输入
            if len(arr) < lookback_days:
                return None, "历史数据不足"
            input_seq = arr[-lookback_days:].copy()
            preds = []
            last_date = datetime.strptime(predict_start_date, '%Y%m%d')
            for i in range(days):
                # 归一化
                input_scaled = scaler.transform(input_seq)
                input_tensor = torch.tensor(input_scaled[np.newaxis, :, :], dtype=torch.float32, device=device)
                with torch.no_grad():
                    pred_val = model(input_tensor)[:, -1].cpu().numpy()[0]
                pred_val = max(pred_val, 0)
                preds.append(pred_val)
                # 构造下一个输入
                next_weekday = (last_date + timedelta(days=i)).weekday()
                next_input = np.array([pred_val, next_weekday])
                input_seq = np.vstack([input_seq[1:], next_input])
            return preds, None
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