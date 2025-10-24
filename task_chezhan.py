import requests
import json
import time
from datetime import datetime, timedelta
import yaml
import os

# =========================
# 配置区（从 YAML 加载）
# =========================

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 4566,
    "predict_schedule_times": ["08:00"],
    "train_schedule_times": ["07:15"],
    "train_algorithm": "knn",
    "predict_daily_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER"
    ],
    "predict_hourly_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER"
    ],
    "train_daily_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER"
    ],
    "train_hourly_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER"
    ]
}

def ensure_config_file(config_path="task_chezhan_config.yaml"):
    """
    如果 task_config.yaml 不存在，则自动生成一个默认配置文件。
    """
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True)
        print(f"已自动生成默认配置文件: {config_path}")

def load_config(config_path="task_config.yaml"):
    ensure_config_file(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f)
    # 合并默认配置和用户配置
    config = DEFAULT_CONFIG.copy()
    if user_config:
        config.update(user_config)
    return config

config = load_config()

HOST = config["host"]
PORT = config["port"]
PREDICT_SCHEDULE_TIMES = config["predict_schedule_times"]
TRAIN_SCHEDULE_TIMES = config["train_schedule_times"]
TRAIN_ALGORITHM = config["train_algorithm"]

PREDICT_HEADERS = {"Content-Type": "application/json"}
PREDICT_MODEL_VERSION_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")  # 设置为当天系统时间

# 日客流预测配置
PREDICT_DAILY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/predict/chezhan/daily/{{}}"
PREDICT_DAILY_METRIC_TYPES = config["predict_daily_metric_types"]

# 小时客流预测配置
PREDICT_HOURLY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/predict/chezhan/hourly/{{}}"
PREDICT_HOURLY_METRIC_TYPES = config["predict_hourly_metric_types"]

# 训练相关配置
TRAIN_HEADERS = {"Content-Type": "application/json"}
TRAIN_END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

# 日客流训练配置
TRAIN_DAILY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/train/chezhan/daily/{{}}"
TRAIN_DAILY_METRIC_TYPES = config["train_daily_metric_types"]

# 小时客流训练配置
TRAIN_HOURLY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/train/chezhan/hourly/{{}}"
TRAIN_HOURLY_METRIC_TYPES = config["train_hourly_metric_types"]

# =========================
# 工具函数
# =========================

def get_today_str():
    return datetime.now().strftime("%Y%m%d")

def build_predict_daily_payload():
    return {
        "algorithm": "knn",
        "model_version_date": PREDICT_MODEL_VERSION_DATE,
        "predict_start_date": get_today_str(),
        "days": 15
    }

def build_predict_hourly_payload():
    return {
        "algorithm": "knn",
        "model_version_date": PREDICT_MODEL_VERSION_DATE,
        "predict_date": get_today_str()
        # 小时预测接口不需要days参数
    }

def build_train_daily_payload():
    return {
        "algorithm": TRAIN_ALGORITHM,
        "train_end_date": TRAIN_END_DATE
    }

def build_train_hourly_payload():
    return {
        "algorithm": TRAIN_ALGORITHM,
        "train_end_date": TRAIN_END_DATE
    }

def get_next_run_time(now, schedule_times):
    today = now.date()
    times_today = [datetime.strptime(f"{today} {t}", "%Y-%m-%d %H:%M") for t in schedule_times]
    future_times = [t for t in times_today if t > now]
    if future_times:
        return min(future_times)
    # 如果今天的时间点都过了，取明天的第一个时间点
    tomorrow = today + timedelta(days=1)
    return datetime.strptime(f"{tomorrow} {schedule_times[0]}", "%Y-%m-%d %H:%M")

# =========================
# 预测任务
# =========================

def run_predict_daily_task():
    for metric in PREDICT_DAILY_METRIC_TYPES:
        api_url = PREDICT_DAILY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_predict_daily_payload()
            response = requests.post(api_url, headers=PREDICT_HEADERS, data=json.dumps(payload))
            print(f"[{datetime.now()}] Predict Daily Metric: {metric} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"[{datetime.now()}] Predict Daily Metric: {metric} | Error: {e}")

def run_predict_hourly_task():
    for metric in PREDICT_HOURLY_METRIC_TYPES:
        api_url = PREDICT_HOURLY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_predict_hourly_payload()
            response = requests.post(api_url, headers=PREDICT_HEADERS, data=json.dumps(payload))
            print(f"[{datetime.now()}] Predict Hourly Metric: {metric} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"[{datetime.now()}] Predict Hourly Metric: {metric} | Error: {e}")

def predict_scheduler():
    print(f"[{datetime.now()}] 预测任务首次启动，立即执行。")
    run_predict_daily_task()
    run_predict_hourly_task()
    while True:
        now = datetime.now()
        next_run = get_next_run_time(now, PREDICT_SCHEDULE_TIMES)
        sleep_seconds = (next_run - now).total_seconds()
        print(f"[{now}] Next predict run scheduled at {next_run} (in {int(sleep_seconds)} seconds)")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        run_predict_daily_task()
        run_predict_hourly_task()

# =========================
# 训练任务
# =========================

def run_train_daily_task():
    for metric in TRAIN_DAILY_METRIC_TYPES:
        api_url = TRAIN_DAILY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_train_daily_payload()
            response = requests.post(api_url, headers=TRAIN_HEADERS, data=json.dumps(payload))
            print(f"[{datetime.now()}] Train Daily Metric: {metric} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"[{datetime.now()}] Train Daily Metric: {metric} | Error: {e}")

def run_train_hourly_task():
    for metric in TRAIN_HOURLY_METRIC_TYPES:
        api_url = TRAIN_HOURLY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_train_hourly_payload()
            response = requests.post(api_url, headers=TRAIN_HEADERS, data=json.dumps(payload))
            print(f"[{datetime.now()}] Train Hourly Metric: {metric} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"[{datetime.now()}] Train Hourly Metric: {metric} | Error: {e}")

def train_scheduler():
    print(f"[{datetime.now()}] 训练任务首次启动，立即执行。")
    run_train_daily_task()
    # run_train_hourly_task()
    while True:
        now = datetime.now()
        next_run = get_next_run_time(now, TRAIN_SCHEDULE_TIMES)
        sleep_seconds = (next_run - now).total_seconds()
        print(f"[{now}] Next train run scheduled at {next_run} (in {int(sleep_seconds)} seconds)")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        run_train_daily_task()
        run_train_hourly_task()

# =========================
# 主入口
# =========================

if __name__ == "__main__":
    import threading

    print("定时任务启动，按 Ctrl+C 退出。")
    # 训练和预测分别跑在不同线程
    train_thread = threading.Thread(target=train_scheduler, daemon=True)
    predict_thread = threading.Thread(target=predict_scheduler, daemon=True)

    # train_thread.start()
    predict_thread.start()

    # 主线程保持运行
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("任务已手动终止。")