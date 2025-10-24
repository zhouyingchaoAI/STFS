import requests
import json
import time
from datetime import datetime, timedelta
import yaml
import os
import logging

# =========================
# 日志配置
# =========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 动态日志文件管理
def get_log_handler():
    """获取当天的日志文件handler"""
    log_file = os.path.join(LOG_DIR, f"task_{datetime.now().strftime('%Y%m%d')}.log")
    return logging.FileHandler(log_file, encoding="utf-8")

# 初始化日志配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(threadName)s %(message)s")

# 添加控制台handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 添加文件handler
file_handler = get_log_handler()
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 用于记录当前日志文件日期
_current_log_date = datetime.now().date()

def update_log_file_if_needed():
    """检查日期变化，必要时更新日志文件"""
    global _current_log_date, file_handler
    current_date = datetime.now().date()
    if current_date != _current_log_date:
        # 移除旧的文件handler
        logger.removeHandler(file_handler)
        file_handler.close()
        # 添加新的文件handler
        file_handler = get_log_handler()
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        _current_log_date = current_date
        logger.info(f"日志文件已更新至新日期: {current_date}")

# =========================
# 配置区（从 YAML 加载）
# =========================

DEFAULT_CONFIG = {
    "host": "10.1.6.230",
    "port": 8900,
    "predict_schedule_times": ["05:15"],
    "train_schedule_times": ["05:00"],
    "train_algorithm": "knn",
    "predict_daily_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER", "F_BOARD_ALIGHT"
    ],
    "predict_hourly_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER", "F_BOARD_ALIGHT"
    ],
    "train_daily_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER", "F_BOARD_ALIGHT"
    ],
    "train_hourly_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER", "F_BOARD_ALIGHT"
    ],
    # 新增线网线路相关配置
    "xianwangxianlu_predict_daily_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER", "F_BOARD_ALIGHT"
    ],
    "xianwangxianlu_predict_hourly_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER", "F_BOARD_ALIGHT"
    ],
    "xianwangxianlu_train_daily_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER", "F_BOARD_ALIGHT"
    ],
    "xianwangxianlu_train_hourly_metric_types": [
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER", "F_BOARD_ALIGHT"
    ]
}

def ensure_config_file(config_path="task_all_config.yaml"):
    """
    如果 task_config.yaml 不存在，则自动生成一个默认配置文件。
    """
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True)
        logger.info(f"已自动生成默认配置文件: {config_path}")

def load_config(config_path="task_all_config.yaml"):
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

# =========================
# 车站相关API配置
# =========================
PREDICT_DAILY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/predict/chezhan/daily/{{}}"
PREDICT_DAILY_METRIC_TYPES = config["predict_daily_metric_types"]

PREDICT_HOURLY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/predict/chezhan/hourly/{{}}"
PREDICT_HOURLY_METRIC_TYPES = config["predict_hourly_metric_types"]

TRAIN_HEADERS = {"Content-Type": "application/json"}

TRAIN_DAILY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/train/chezhan/daily/{{}}"
TRAIN_DAILY_METRIC_TYPES = config["train_daily_metric_types"]

TRAIN_HOURLY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/train/chezhan/hourly/{{}}"
TRAIN_HOURLY_METRIC_TYPES = config["train_hourly_metric_types"]

# =========================
# 线网线路相关API配置
# =========================
XWX_DAILY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/predict/xianwangxianlu/daily/{{}}"
XWX_DAILY_METRIC_TYPES = config.get("xianwangxianlu_predict_daily_metric_types", ["F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER"])

XWX_HOURLY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/predict/xianwangxianlu/hourly/{{}}"
XWX_HOURLY_METRIC_TYPES = config.get("xianwangxianlu_predict_hourly_metric_types", ["F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER"])

XWX_TRAIN_DAILY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/train/xianwangxianlu/daily/{{}}"
XWX_TRAIN_DAILY_METRIC_TYPES = config.get("xianwangxianlu_train_daily_metric_types", ["F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER"])

XWX_TRAIN_HOURLY_API_URL_TEMPLATE = f"http://{HOST}:{PORT}/train/xianwangxianlu/hourly/{{}}"
XWX_TRAIN_HOURLY_METRIC_TYPES = config.get("xianwangxianlu_train_hourly_metric_types", ["F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER"])

# =========================
# 工具函数（动态获取日期）
# =========================

def get_yesterday_str():
    """获取昨天的日期字符串"""
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

def get_today_str():
    """获取今天的日期字符串"""
    return datetime.now().strftime("%Y%m%d")

def get_tomorrow_str():
    """获取明天的日期字符串"""
    return (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")

def build_predict_daily_payload():
    """构建日预测请求体 - 每次调用时动态获取日期"""
    model_version_date = get_yesterday_str()  # 模型版本日期为昨天
    predict_start_date = get_today_str()  # 预测开始日期为今天
    logger.debug(f"构建日预测payload: model_version_date={model_version_date}, predict_start_date={predict_start_date}")
    return {
        "algorithm": "knn",
        "model_version_date": model_version_date,
        "predict_start_date": predict_start_date,
        "days": 15
    }

def build_predict_hourly_payload(predict_date=None):
    """构建小时预测请求体 - 每次调用时动态获取日期"""
    model_version_date = get_yesterday_str()  # 模型版本日期为昨天
    if predict_date is None:
        predict_date = get_today_str()
    logger.debug(f"构建小时预测payload: model_version_date={model_version_date}, predict_date={predict_date}")
    return {
        "algorithm": "knn",
        "model_version_date": model_version_date,
        "predict_date": predict_date
    }

def build_train_daily_payload():
    """构建日训练请求体 - 每次调用时动态获取日期"""
    train_end_date = get_yesterday_str()  # 训练结束日期为昨天
    logger.debug(f"构建日训练payload: train_end_date={train_end_date}")
    return {
        "algorithm": TRAIN_ALGORITHM,
        "train_end_date": train_end_date
    }

def build_train_hourly_payload():
    """构建小时训练请求体 - 每次调用时动态获取日期"""
    train_end_date = get_yesterday_str()  # 训练结束日期为昨天
    logger.debug(f"构建小时训练payload: train_end_date={train_end_date}")
    return {
        "algorithm": TRAIN_ALGORITHM,
        "train_end_date": train_end_date
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
    start_time = time.time()
    logger.info(f"开始执行日预测任务 - 当前日期: {get_today_str()}, 模型版本日期: {get_yesterday_str()}")
    
    # 车站
    for metric in PREDICT_DAILY_METRIC_TYPES:
        api_url = PREDICT_DAILY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_predict_daily_payload()
            response = requests.post(api_url, headers=PREDICT_HEADERS, data=json.dumps(payload))
            logger.info(f"Predict Daily Metric (CheZhan): {metric} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logger.error(f"Predict Daily Metric (CheZhan): {metric} | Error: {e}", exc_info=True)
    
    # 线网线路
    for metric in XWX_DAILY_METRIC_TYPES:
        api_url = XWX_DAILY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_predict_daily_payload()
            response = requests.post(api_url, headers=PREDICT_HEADERS, data=json.dumps(payload))
            logger.info(f"Predict Daily Metric (XianWangXianLu): {metric} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logger.error(f"Predict Daily Metric (XianWangXianLu): {metric} | Error: {e}", exc_info=True)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Predict Daily Task completed. Time used: {duration:.2f} seconds.")

def run_predict_hourly_task_for_date(target_date_str, date_description):
    """
    为指定日期运行小时预测任务
    
    Args:
        target_date_str: 目标日期字符串，格式为YYYYMMDD
        date_description: 日期描述，用于日志输出
    """
    start_time = time.time()
    logger.info(f"开始执行{date_description}的小时预测任务 (日期: {target_date_str}, 模型版本: {get_yesterday_str()})")
    
    # 车站
    for metric in PREDICT_HOURLY_METRIC_TYPES:
        api_url = PREDICT_HOURLY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_predict_hourly_payload(target_date_str)
            response = requests.post(api_url, headers=PREDICT_HEADERS, data=json.dumps(payload))
            logger.info(f"Predict Hourly Metric (CheZhan) {date_description}: {metric} | Date: {target_date_str} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logger.error(f"Predict Hourly Metric (CheZhan) {date_description}: {metric} | Date: {target_date_str} | Error: {e}", exc_info=True)
    
    # 线网线路
    for metric in XWX_HOURLY_METRIC_TYPES:
        api_url = XWX_HOURLY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_predict_hourly_payload(target_date_str)
            response = requests.post(api_url, headers=PREDICT_HEADERS, data=json.dumps(payload))
            logger.info(f"Predict Hourly Metric (XianWangXianLu) {date_description}: {metric} | Date: {target_date_str} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logger.error(f"Predict Hourly Metric (XianWangXianLu) {date_description}: {metric} | Date: {target_date_str} | Error: {e}", exc_info=True)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Predict Hourly Task for {date_description} (Date: {target_date_str}) completed. Time used: {duration:.2f} seconds.")

def run_predict_hourly_task():
    """运行小时预测任务 - 预测当天和第二天"""
    overall_start_time = time.time()
    
    # 预测当天
    today_str = get_today_str()
    run_predict_hourly_task_for_date(today_str, "当天")
    
    # 预测第二天
    tomorrow_str = get_tomorrow_str()
    run_predict_hourly_task_for_date(tomorrow_str, "第二天")
    
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    logger.info(f"All Predict Hourly Tasks completed (today + tomorrow). Total time used: {overall_duration:.2f} seconds.")

def predict_scheduler():
    while True:
        now = datetime.now()
        next_run = get_next_run_time(now, PREDICT_SCHEDULE_TIMES)
        sleep_seconds = (next_run - now).total_seconds()
        logger.info(f"Next predict run scheduled at {next_run} (in {int(sleep_seconds)} seconds)")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        start_time = time.time()
        run_predict_daily_task()
        run_predict_hourly_task()
        end_time = time.time()
        logger.info(f"预测任务本次执行完成，总耗时: {end_time - start_time:.2f} 秒。")

# =========================
# 训练任务
# =========================

def run_train_daily_task():
    start_time = time.time()
    logger.info(f"开始执行日训练任务 - 训练结束日期: {get_yesterday_str()}")
    
    # 车站
    for metric in TRAIN_DAILY_METRIC_TYPES:
        api_url = TRAIN_DAILY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_train_daily_payload()
            response = requests.post(api_url, headers=TRAIN_HEADERS, data=json.dumps(payload))
            logger.info(f"Train Daily Metric (CheZhan): {metric} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logger.error(f"Train Daily Metric (CheZhan): {metric} | Error: {e}", exc_info=True)
    
    # 线网线路
    for metric in XWX_TRAIN_DAILY_METRIC_TYPES:
        api_url = XWX_TRAIN_DAILY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_train_daily_payload()
            response = requests.post(api_url, headers=TRAIN_HEADERS, data=json.dumps(payload))
            logger.info(f"Train Daily Metric (XianWangXianLu): {metric} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logger.error(f"Train Daily Metric (XianWangXianLu): {metric} | Error: {e}", exc_info=True)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Train Daily Task completed. Time used: {duration:.2f} seconds.")

def run_train_hourly_task():
    start_time = time.time()
    logger.info(f"开始执行小时训练任务 - 训练结束日期: {get_yesterday_str()}")
    
    # 车站
    for metric in TRAIN_HOURLY_METRIC_TYPES:
        api_url = TRAIN_HOURLY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_train_hourly_payload()
            response = requests.post(api_url, headers=TRAIN_HEADERS, data=json.dumps(payload))
            logger.info(f"Train Hourly Metric (CheZhan): {metric} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logger.error(f"Train Hourly Metric (CheZhan): {metric} | Error: {e}", exc_info=True)
    
    # 线网线路
    for metric in XWX_TRAIN_HOURLY_METRIC_TYPES:
        api_url = XWX_TRAIN_HOURLY_API_URL_TEMPLATE.format(metric)
        try:
            payload = build_train_hourly_payload()
            response = requests.post(api_url, headers=TRAIN_HEADERS, data=json.dumps(payload))
            logger.info(f"Train Hourly Metric (XianWangXianLu): {metric} | Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logger.error(f"Train Hourly Metric (XianWangXianLu): {metric} | Error: {e}", exc_info=True)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Train Hourly Task completed. Time used: {duration:.2f} seconds.")

def train_scheduler():
    while True:
        now = datetime.now()
        next_run = get_next_run_time(now, TRAIN_SCHEDULE_TIMES)
        sleep_seconds = (next_run - now).total_seconds()
        logger.info(f"Next train run scheduled at {next_run} (in {int(sleep_seconds)} seconds)")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        start_time = time.time()
        run_train_daily_task()
        run_train_hourly_task()
        end_time = time.time()
        logger.info(f"训练任务本次执行完成，总耗时: {end_time - start_time:.2f} 秒。")

# =========================
# 主入口
# =========================

if __name__ == "__main__":
    logger.info("定时任务启动，按 Ctrl+C 退出。")
    import threading

    train_scheduler_thread = threading.Thread(target=train_scheduler, daemon=True, name="TrainSchedulerThread")
    train_scheduler_thread.start()
    predict_thread = threading.Thread(target=predict_scheduler, daemon=True, name="PredictSchedulerThread")
    predict_thread.start()
    
    logger.info("训练任务首次启动，立即执行。")
    start_time = time.time()
    run_train_daily_task()
    run_train_hourly_task()
    end_time = time.time()
    logger.info(f"训练任务首次执行完成，总耗时: {end_time - start_time:.2f} 秒。")

    logger.info("预测任务首次启动，立即执行。")
    start_time = time.time()
    run_predict_daily_task()
    run_predict_hourly_task()
    end_time = time.time()
    logger.info(f"预测任务首次执行完成，总耗时: {end_time - start_time:.2f} 秒。")

    # 主线程保持运行
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("任务已手动终止。")