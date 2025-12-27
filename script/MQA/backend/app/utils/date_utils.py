"""
日期处理工具函数
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple
import re


def parse_date_expression(text: str, base_date: Optional[datetime] = None) -> Optional[datetime]:
    """
    解析日期表达式，如"昨天"、"今天"、"本周"等
    
    Args:
        text: 日期表达式文本
        base_date: 基准日期，默认为今天
        
    Returns:
        解析后的日期，如果无法解析返回None
    """
    if base_date is None:
        base_date = datetime.now()
    
    text = text.strip()
    
    # 今天
    if text in ["今天", "今日", "now", "today"]:
        return base_date
    
    # 昨天
    if text in ["昨天", "昨日", "yesterday"]:
        return base_date - timedelta(days=1)
    
    # 明天
    if text in ["明天", "明日", "tomorrow"]:
        return base_date + timedelta(days=1)
    
    # 前天
    if text in ["前天", "前日"]:
        return base_date - timedelta(days=2)
    
    # 后天
    if text in ["后天", "后日"]:
        return base_date + timedelta(days=2)
    
    # 本周
    if text in ["本周", "这周", "this week"]:
        days_since_monday = base_date.weekday()
        return base_date - timedelta(days=days_since_monday)
    
    # 上周
    if text in ["上周", "上星期"]:
        days_since_monday = base_date.weekday()
        last_monday = base_date - timedelta(days=days_since_monday + 7)
        return last_monday
    
    # 下周
    if text in ["下周", "下星期"]:
        days_since_monday = base_date.weekday()
        next_monday = base_date - timedelta(days=days_since_monday - 7)
        return next_monday
    
    # 本月
    if text in ["本月", "这个月", "this month"]:
        return base_date.replace(day=1)
    
    # 上月
    if text in ["上月", "上个月", "last month"]:
        if base_date.month == 1:
            return base_date.replace(year=base_date.year - 1, month=12, day=1)
        return base_date.replace(month=base_date.month - 1, day=1)
    
    # 今年
    if text in ["今年", "this year"]:
        return base_date.replace(month=1, day=1)
    
    # 去年
    if text in ["去年", "last year"]:
        return base_date.replace(year=base_date.year - 1, month=1, day=1)
    
    # N天前
    match = re.search(r'(\d+)天前', text)
    if match:
        days = int(match.group(1))
        return base_date - timedelta(days=days)
    
    # N天后
    match = re.search(r'(\d+)天后', text)
    if match:
        days = int(match.group(1))
        return base_date + timedelta(days=days)
    
    # 最近N天
    match = re.search(r'最近(\d+)天', text)
    if match:
        days = int(match.group(1))
        return base_date - timedelta(days=days - 1)
    
    # 标准日期格式 YYYY-MM-DD
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        pass
    
    # 日期格式 YYYYMMDD
    try:
        return datetime.strptime(text, "%Y%m%d")
    except ValueError:
        pass
    
    return None


def date_to_int(date: datetime) -> int:
    """
    将日期转换为整数格式 YYYYMMDD
    
    Args:
        date: 日期对象
        
    Returns:
        整数日期，如 20240101
    """
    return int(date.strftime("%Y%m%d"))


def int_to_date(date_int: int) -> datetime:
    """
    将整数日期转换为日期对象
    
    Args:
        date_int: 整数日期，如 20240101
        
    Returns:
        日期对象
    """
    date_str = str(date_int)
    return datetime.strptime(date_str, "%Y%m%d")


def get_date_range(start_date: datetime, end_date: datetime) -> list:
    """
    获取日期范围内的所有日期
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        日期列表
    """
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def parse_date_range(text: str, base_date: Optional[datetime] = None) -> Optional[Tuple[datetime, datetime]]:
    """
    解析日期范围表达式
    
    Args:
        text: 日期范围表达式，如"最近7天"、"本周"等
        base_date: 基准日期
        
    Returns:
        (开始日期, 结束日期) 元组，如果无法解析返回None
    """
    if base_date is None:
        base_date = datetime.now()
    
    text = text.strip()
    
    # 最近N天
    match = re.search(r'最近(\d+)天', text)
    if match:
        days = int(match.group(1))
        end_date = base_date
        start_date = base_date - timedelta(days=days - 1)
        return (start_date, end_date)
    
    # 本周
    if text in ["本周", "这周"]:
        days_since_monday = base_date.weekday()
        start_date = base_date - timedelta(days=days_since_monday)
        end_date = start_date + timedelta(days=6)
        return (start_date, end_date)
    
    # 上周
    if text in ["上周", "上星期"]:
        days_since_monday = base_date.weekday()
        start_date = base_date - timedelta(days=days_since_monday + 7)
        end_date = start_date + timedelta(days=6)
        return (start_date, end_date)
    
    # 本月
    if text in ["本月", "这个月"]:
        start_date = base_date.replace(day=1)
        if base_date.month == 12:
            end_date = base_date.replace(year=base_date.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end_date = base_date.replace(month=base_date.month + 1, day=1) - timedelta(days=1)
        return (start_date, end_date)
    
    return None

