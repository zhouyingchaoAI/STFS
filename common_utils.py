# 通用工具模块：提取项目中重复使用的辅助函数
"""
该模块包含项目中多处重复使用的通用工具函数，包括：
- 类型转换函数
- 日期处理函数
- 数据验证函数
- 文件名处理函数
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Optional, Union, List


# =============================================================================
# 类型转换函数
# =============================================================================

def to_int(val: Any) -> Optional[int]:
    """
    将任意值安全转换为整数
    
    参数:
        val: 待转换的值（可以是 int, float, str, None 等）
        
    返回:
        转换后的整数，如果转换失败返回 None
    """
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if pd.isna(val) or np.isinf(val):
            return None
        return int(val)
    if isinstance(val, str):
        val_strip = val.strip()
        if not val_strip:
            return None
        # 尝试直接转换整数字符串
        if val_strip.lstrip('-').isdigit():
            return int(val_strip)
        # 尝试转换浮点数字符串
        try:
            float_val = float(val_strip)
            if pd.isna(float_val) or np.isinf(float_val):
                return None
            return int(float_val)
        except ValueError:
            return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def to_float(val: Any) -> Optional[float]:
    """
    将任意值安全转换为浮点数
    
    参数:
        val: 待转换的值
        
    返回:
        转换后的浮点数，如果转换失败返回 None
    """
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if pd.isna(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return float(val)
    if isinstance(val, str):
        val_strip = val.strip()
        if not val_strip:
            return None
        try:
            result = float(val_strip)
            if pd.isna(result) or np.isinf(result):
                return None
            return result
        except ValueError:
            return None
    try:
        result = float(val)
        if pd.isna(result) or np.isinf(result):
            return None
        return result
    except (ValueError, TypeError):
        return None


def to_str(val: Any) -> Optional[str]:
    """
    将任意值安全转换为字符串
    
    参数:
        val: 待转换的值
        
    返回:
        转换后的字符串，如果是 None 则返回 None
    """
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if pd.isna(val):
        return None
    try:
        return str(val)
    except (ValueError, TypeError):
        return None


def to_bool(val: Any, default: bool = False) -> bool:
    """
    将任意值安全转换为布尔值
    
    参数:
        val: 待转换的值
        default: 默认值
        
    返回:
        转换后的布尔值
    """
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        val_lower = val.strip().lower()
        if val_lower in ('true', '1', 'yes', 'on', 't', 'y'):
            return True
        if val_lower in ('false', '0', 'no', 'off', 'f', 'n', ''):
            return False
    return default


def parse_temperature_value(val: Any) -> Optional[float]:
    """
    从气温字符串中提取数值并返回平均气温。

    支持示例:
        - "22℃/29℃" -> 25.5
        - "-2℃~5℃" -> 1.5
        - "28℃" -> 28.0

    参数:
        val: 原始气温值

    返回:
        平均气温，解析失败返回 None
    """
    if val is None:
        return None
    if isinstance(val, (int, float, np.integer, np.floating)):
        if pd.isna(val) or np.isinf(val):
            return None
        return float(val)

    text = str(val).strip()
    if not text or text.lower() == "nan":
        return None

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None

    values = [float(num) for num in numbers]
    return float(sum(values) / len(values))


# =============================================================================
# 日期处理函数
# =============================================================================

def parse_date(date_str: str) -> Optional[datetime]:
    """
    解析多种格式的日期字符串
    
    支持格式:
        - YYYYMMDD (如 20250115)
        - YYYY-MM-DD (如 2025-01-15)
        - YYYY/MM/DD (如 2025/01/15)
        
    参数:
        date_str: 日期字符串
        
    返回:
        datetime 对象，如果解析失败返回 None
    """
    if date_str is None:
        return None
    if not isinstance(date_str, str):
        date_str = str(date_str)
    
    date_str = date_str.strip()
    
    # 尝试 YYYYMMDD 格式
    if len(date_str) == 8 and date_str.isdigit():
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            pass
    
    # 尝试 YYYY-MM-DD 格式
    if '-' in date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            pass
    
    # 尝试 YYYY/MM/DD 格式
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            try:
                return datetime(
                    year=int(parts[0]),
                    month=int(parts[1]),
                    day=int(parts[2])
                )
            except (ValueError, TypeError):
                pass
    
    return None


def format_date_int(date_str: str) -> Optional[int]:
    """
    将日期字符串转换为整数格式 YYYYMMDD
    
    参数:
        date_str: 日期字符串（支持多种格式）
        
    返回:
        整数格式的日期，如 20250115
    """
    dt = parse_date(date_str)
    if dt is None:
        return None
    return int(dt.strftime("%Y%m%d"))


def format_date_str(date_str: str, output_format: str = "%Y%m%d") -> Optional[str]:
    """
    将日期字符串转换为指定格式
    
    参数:
        date_str: 日期字符串（支持多种格式）
        output_format: 输出格式
        
    返回:
        格式化后的日期字符串
    """
    dt = parse_date(date_str)
    if dt is None:
        return None
    return dt.strftime(output_format)


def get_last_year_date(dt: datetime) -> datetime:
    """
    获取去年同期日期（处理闰年情况）
    
    参数:
        dt: 日期对象
        
    返回:
        去年同期的日期对象
    """
    try:
        return dt.replace(year=dt.year - 1)
    except ValueError:
        # 闰年 2 月 29 日的情况
        return dt.replace(year=dt.year - 1, month=2, day=28)


def get_date_range(start_date: str, days: int) -> List[str]:
    """
    获取从起始日期开始的日期范围
    
    参数:
        start_date: 起始日期（YYYYMMDD 格式）
        days: 天数
        
    返回:
        日期字符串列表
    """
    dt = parse_date(start_date)
    if dt is None:
        return []
    
    return [(dt + timedelta(days=d)).strftime('%Y%m%d') for d in range(days)]


def to_datetime_int(val: Any) -> Optional[int]:
    """
    将日期时间值转换为整数格式（YYYYMMDD）
    
    参数:
        val: 日期时间值（可以是 datetime, str, int 等）
        
    返回:
        整数格式的日期
    """
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, datetime):
        return int(val.strftime('%Y%m%d'))
    if isinstance(val, str):
        # 尝试解析 YYYY-MM-DD HH:MM:SS 格式
        try:
            dt = datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
            return int(dt.strftime('%Y%m%d'))
        except ValueError:
            pass
        # 尝试其他日期格式
        return format_date_int(val)
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# =============================================================================
# 文件名处理函数
# =============================================================================

def sanitize_filename(name: str) -> str:
    """
    清理文件名，将不适合作为文件名的字符替换为安全字符
    
    参数:
        name: 原始名称
        
    返回:
        清理后的安全文件名
    """
    if not isinstance(name, str):
        name = str(name)
    
    # 替换不安全字符
    unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    sanitized = name
    for char in unsafe_chars:
        sanitized = sanitized.replace(char, '-')
    
    # 移除开头和结尾的点和空格
    sanitized = sanitized.strip('. ')
    
    # 如果结果为空，返回默认值
    if not sanitized:
        sanitized = 'unknown'
    
    return sanitized


# =============================================================================
# 数据验证函数
# =============================================================================

def is_valid_date_format(date_str: str, format_str: str = "%Y%m%d") -> bool:
    """
    验证日期字符串是否符合指定格式
    
    参数:
        date_str: 日期字符串
        format_str: 期望的日期格式
        
    返回:
        是否有效
    """
    if not date_str:
        return False
    try:
        datetime.strptime(date_str, format_str)
        return True
    except ValueError:
        return False


def validate_required_columns(df: pd.DataFrame, required_cols: set) -> Optional[str]:
    """
    验证 DataFrame 是否包含所有必需的列
    
    参数:
        df: 待验证的 DataFrame
        required_cols: 必需的列名集合
        
    返回:
        如果验证通过返回 None，否则返回缺少的列名字符串
    """
    if df is None or df.empty:
        return "DataFrame 为空"
    
    missing = required_cols - set(df.columns)
    if missing:
        return f"缺少必要列: {', '.join(missing)}"
    
    return None


# =============================================================================
# 哈希和ID生成函数
# =============================================================================

def generate_remarks_hash(text: str) -> int:
    """
    生成备注哈希值（用于数据库 REMARKS 字段）
    
    参数:
        text: 文本
        
    返回:
        8 位整数哈希值
    """
    return abs(hash(str(text))) % (10 ** 8)


def generate_type_hash(val: Any) -> Optional[int]:
    """
    生成类型哈希值（用于数据库 F_TYPE 等字段）
    
    参数:
        val: 值
        
    返回:
        整数哈希值或原值
    """
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return abs(hash(str(val))) % (10 ** 8)


# =============================================================================
# 安全值获取函数
# =============================================================================

def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """
    安全地从对象中获取值
    
    支持 dict, pd.Series, 以及具有属性的对象
    
    参数:
        obj: 对象
        key: 键名
        default: 默认值
        
    返回:
        获取到的值或默认值
    """
    if obj is None:
        return default
    
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        if isinstance(obj, pd.Series):
            return obj.get(key, default)
        if hasattr(obj, key):
            return getattr(obj, key, default)
        if hasattr(obj, '__getitem__'):
            return obj[key]
    except (KeyError, AttributeError, TypeError, IndexError):
        pass
    
    return default


def safe_get_int(obj: Any, key: str, default: Optional[int] = None) -> Optional[int]:
    """
    安全地从对象中获取整数值
    
    参数:
        obj: 对象
        key: 键名
        default: 默认值
        
    返回:
        整数值或默认值
    """
    val = safe_get(obj, key)
    if val is None:
        return default
    result = to_int(val)
    return result if result is not None else default


def safe_get_float(obj: Any, key: str, default: Optional[float] = None) -> Optional[float]:
    """
    安全地从对象中获取浮点数值
    
    参数:
        obj: 对象
        key: 键名
        default: 默认值
        
    返回:
        浮点数值或默认值
    """
    val = safe_get(obj, key)
    if val is None:
        return default
    result = to_float(val)
    return result if result is not None else default


def safe_get_str(obj: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    """
    安全地从对象中获取字符串值
    
    参数:
        obj: 对象
        key: 键名
        default: 默认值
        
    返回:
        字符串值或默认值
    """
    val = safe_get(obj, key)
    if val is None:
        return default
    result = to_str(val)
    return result if result is not None else default


# =============================================================================
# 数据处理函数
# =============================================================================

def ensure_numeric_columns(df: pd.DataFrame, columns: List[str], fill_value: float = 0.0) -> pd.DataFrame:
    """
    确保 DataFrame 的指定列为数值类型
    
    参数:
        df: DataFrame
        columns: 需要转换的列名列表
        fill_value: 无效值的填充值
        
    返回:
        处理后的 DataFrame
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_value)
        else:
            df[col] = fill_value
    
    return df


def normalize_line_no(line_no: Any, width: int = 2) -> str:
    """
    标准化线路编号（补零）
    
    参数:
        line_no: 线路编号
        width: 目标宽度
        
    返回:
        标准化后的线路编号字符串
    """
    if line_no is None:
        return '00'
    return str(line_no).strip().zfill(width)


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> dict:
    """
    计算预测评估指标
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        
    返回:
        包含 MSE, MAE, MAPE 的字典
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 计算 MAPE（避免除零）
    non_zero_mask = y_true != 0
    if non_zero_mask.any():
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'mape': float(mape)
    }
