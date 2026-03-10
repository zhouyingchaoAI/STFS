# 配置管理模块：负责加载和保存 YAML 配置文件
import os
import yaml
from datetime import datetime
from typing import Dict, Optional

DEFAULT_HOURLY_CONFIG_PATH = "model_config.yaml"  # 默认小时预测配置文件路径
DEFAULT_DAILY_CONFIG_PATH = "model_config_daily.yaml"  # 默认日预测配置文件路径

def load_yaml_config(config_path: str = DEFAULT_HOURLY_CONFIG_PATH, default_daily: bool = False) -> Dict:
    """
    加载 YAML 配置文件，若不存在则创建默认配置
    
    参数:
        config_path: 配置文件路径
        default_daily: 是否创建日预测默认配置
        
    返回:
        配置字典
    """
    if not os.path.exists(config_path):
        config = {
            "model_root_dir": "models/daily" if default_daily else "models/hour",
            "current_version": None,
            "default_algorithm": "knn" if default_daily else "lstm",
            "train_params": {
                "n_neighbors": 5 if default_daily else None,
                "lookback_days": None if default_daily else 7,
                "hidden_size": None if default_daily else 50,
                "num_layers": None if default_daily else 2,
                "dropout": None if default_daily else 0.2,
                "batch_size": None if default_daily else 32,
                "epochs": None if default_daily else 100,
                "patience": None if default_daily else 10,
                "learning_rate": None if default_daily else 0.001
            }
        }
        save_yaml_config(config, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml_config(config: Dict, config_path: str = DEFAULT_HOURLY_CONFIG_PATH) -> None:
    """
    保存配置到 YAML 文件
    
    参数:
        config: 配置字典
        config_path: 保存路径
    """
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

def get_today_version() -> str:
    """
    获取当天版本号（日期字符串）
    
    返回:
        YYYYMMDD 格式的版本号
    """
    return datetime.now().strftime("%Y%m%d")

def get_version_dir(version: Optional[str] = None, config_obj: Optional[Dict] = None) -> str:
    """
    获取模型存储目录
    
    参数:
        version: 模型版本（默认使用当天日期）
        config_obj: 配置字典
        
    返回:
        版本目录路径
    """
    if config_obj is None:
        config_obj = load_yaml_config()
    root = config_obj.get("model_root_dir", "models/hour")
    if version is None:
        version = get_today_version()
    return os.path.join(root, version)

def get_current_version(config_obj: Optional[Dict] = None, config_path: str = DEFAULT_HOURLY_CONFIG_PATH) -> str:
    """
    获取当前版本号，若未指定则设为当天
    
    参数:
        config_obj: 配置字典
        config_path: 配置文件路径
        
    返回:
        版本号字符串
    """
    if config_obj is None:
        config_obj = load_yaml_config(config_path)
    version = config_obj.get("current_version")
    if version is None:
        version = get_today_version()
        config_obj["current_version"] = version
        save_yaml_config(config_obj, config_path)
    return version


def filter_lines_by_config(
    lines: list,
    config_obj: Optional[Dict] = None,
    config_key: str = "xianwangxianlu_daily_predict"
) -> list:
    """
    根据配置过滤线路列表
    
    参数:
        lines: 原始线路列表
        config_obj: 配置字典
        config_key: 配置项键名（如 xianwangxianlu_daily_predict）
        
    返回:
        过滤后的线路列表
    """
    if config_obj is None:
        config_obj = load_yaml_config(DEFAULT_DAILY_CONFIG_PATH)
    
    # 获取线路配置
    line_config = config_obj.get(config_key, {})
    
    # 是否启用过滤
    enable_filter = line_config.get("enable_filter", False)
    if not enable_filter:
        return lines  # 不过滤，返回全部线路
    
    # 获取启用和禁用的线路列表
    enabled_lines = line_config.get("enabled_lines", [])
    disabled_lines = line_config.get("disabled_lines", [])
    
    # 如果 enabled_lines 为空，则启用所有线路
    if not enabled_lines:
        filtered_lines = lines
    else:
        # 只保留启用的线路
        filtered_lines = [line for line in lines if line in enabled_lines]
    
    # 排除禁用的线路
    if disabled_lines:
        filtered_lines = [line for line in filtered_lines if line not in disabled_lines]
    
    return filtered_lines