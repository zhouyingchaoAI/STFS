# LSTM 模型模块：实现 LSTM 模型、数据集处理和预测逻辑
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict


class FlowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class LSTMFlowPredictor:
    def __init__(self, model_dir: str, version: str, config: Dict):
        self.model_dir = model_dir
        self.version = version
        self.config = config
        self.models = {}
        self.scalers = {}
        self.model_info = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def prepare_data(self, data: pd.DataFrame, lookback_days: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if lookback_days is None:
            lookback_days = self.config.get("train_params", {}).get("lookback_days", 7)
        dates = sorted(data['F_DATE'].unique())
        full_data = []
        for date in dates:
            for hour in range(24):
                hour_str = str(hour).zfill(2)
                flow = data[(data['F_DATE'] == date) & (data['F_HOUR'] == hour_str)]['F_KLCOUNT']
                flow_val = flow.values[0] if not flow.empty else 0
                full_data.append([date, hour, flow_val])
        df_full = pd.DataFrame(full_data, columns=['date', 'hour', 'flow'])
        X, y = [], []
        lookback_hours = lookback_days * 24
        for i in range(lookback_hours, len(df_full)):
            sequence = [[df_full.iloc[j]['flow'], df_full.iloc[j]['hour']] for j in range(i - lookback_hours, i)]
            X.append(sequence)
            y.append(df_full.iloc[i]['flow'])
        return np.array(X), np.array(y)

    def train(self, line_data: pd.DataFrame, line_no: str, model_version: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        训练 LSTM 模型

        参数:
            line_data: 线路数据 DataFrame
            line_no: 线路编号
            model_version: 模型版本（可选）

        返回:
            (mse, mae, error) 元组
        """
        try:
            params = self.config.get("train_params", {})
            lookback_days = params.get("lookback_days", 7)
            batch_size = params.get("batch_size", 32)
            epochs = params.get("epochs", 100)
            patience = params.get("patience", 10)
            learning_rate = params.get("learning_rate", 0.001)

            X, y = self.prepare_data(line_data, lookback_days)
            if len(X) < 50:
                return None, None, "数据量不足"

            scaler = MinMaxScaler()
            X_scaled = X.copy()
            X_scaled[:, :, 0] = scaler.fit_transform(X[:, :, 0])
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

            train_size = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

            train_dataset = FlowDataset(X_train, y_train)
            test_dataset = FlowDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model = LSTMModel(
                input_size=2,
                hidden_size=params.get("hidden_size", 50),
                num_layers=params.get("num_layers", 2),
                dropout=params.get("dropout", 0.2)
            ).to(self.device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            for epoch in range(epochs):
                model.train()
                train_losses = []
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        val_losses.append(loss.item())

                avg_val_loss = np.mean(val_losses)
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            model.load_state_dict(best_model_state)
            model.eval()
            y_pred, y_true = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    y_pred.extend(outputs.squeeze().cpu().numpy())
                    y_true.extend(batch_y.cpu().numpy())

            y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

            # 支持模型版本管理
            version = model_version if model_version is not None else self.version
            model_path = os.path.join(self.model_dir, f"lstm_line_{line_no}_{version}.pth") if version else os.path.join(self.model_dir, f"lstm_line_{line_no}.pth")
            scaler_path = os.path.join(self.model_dir, f"scaler_line_{line_no}_{version}.pkl") if version else os.path.join(self.model_dir, f"scaler_line_{line_no}.pkl")
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': 2,
                'hidden_size': params.get("hidden_size", 50),
                'num_layers': params.get("num_layers", 2),
                'dropout': params.get("dropout", 0.2),
                'lookback_hours': lookback_days * 24
            }, model_path)
            joblib.dump(scaler, scaler_path)

            self.models[line_no] = model
            self.scalers[line_no] = scaler
            return mse, mae, None
        except Exception as e:
            return None, None, str(e)

    def predict(self, line_data: pd.DataFrame, line_no: str, predict_date: str, model_version: Optional[str] = None) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        预测指定线路和日期的小时客流，支持模型版本管理

        参数:
            line_data: 线路数据 DataFrame
            line_no: 线路编号
            predict_date: 预测日期 (YYYYMMDD)
            model_version: 模型版本（可选）

        返回:
            (预测结果, 错误信息) 元组
        """
        try:
            # 支持模型版本管理
            version = model_version if model_version is not None else self.version
            model_path = os.path.join(self.model_dir, f"lstm_line_{line_no}_{version}.pth") if version else os.path.join(self.model_dir, f"lstm_line_{line_no}.pth")
            scaler_path = os.path.join(self.model_dir, f"scaler_line_{line_no}_{version}.pkl") if version else os.path.join(self.model_dir, f"scaler_line_{line_no}.pkl")
            # 兼容老版本文件名
            if not os.path.exists(model_path):
                model_path = os.path.join(self.model_dir, f"lstm_line_{line_no}.pth")
            if not os.path.exists(scaler_path):
                scaler_path = os.path.join(self.model_dir, f"scaler_line_{line_no}.pkl")
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return None, "模型文件未找到"

            checkpoint = torch.load(model_path, map_location=self.device)
            model = LSTMModel(
                input_size=2,
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                dropout=checkpoint['dropout']
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            scaler = joblib.load(scaler_path)

            lookback_hours = checkpoint.get('lookback_hours', 168)
            predict_dt = datetime.strptime(predict_date, '%Y%m%d')
            start_date = predict_dt - timedelta(hours=lookback_hours)
            recent_data = []
            for i in range(lookback_hours):
                current_dt = start_date + timedelta(hours=i)
                date_str = current_dt.strftime('%Y%m%d')
                hour = current_dt.hour
                hour_str = str(hour).zfill(2)
                flow = line_data[(line_data['F_DATE'] == date_str) & (line_data['F_HOUR'] == hour_str)]['F_KLCOUNT']
                flow_val = flow.values[0] if not flow.empty else 0
                recent_data.append([flow_val, hour])

            if len(recent_data) < lookback_hours:
                padding_length = lookback_hours - len(recent_data)
                padding_data = [[0, (i % 24)] for i in range(padding_length)]
                recent_data = padding_data + recent_data

            predictions = []
            input_sequence = np.array(recent_data).copy()
            with torch.no_grad():
                for hour in range(24):
                    X_input = input_sequence.reshape(1, lookback_hours, 2)
                    X_input_scaled = X_input.copy()
                    X_input_scaled[:, :, 0] = scaler.transform(X_input[:, :, 0].reshape(-1, 1)).flatten()
                    X_input_tensor = torch.FloatTensor(X_input_scaled).to(self.device)
                    pred = model(X_input_tensor)
                    pred_val = scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1))[0, 0]
                    pred_val = max(0, pred_val)
                    predictions.append(pred_val)
                    new_point = np.array([[pred_val, hour]])
                    input_sequence = np.vstack([input_sequence[1:], new_point])
            return predictions, None
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
        info_path = os.path.join(self.model_dir, f"model_info_line_{line_no}_{version}.json") if version else os.path.join(self.model_dir, f"model_info_line_{line_no}.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        self.model_info[line_no] = info