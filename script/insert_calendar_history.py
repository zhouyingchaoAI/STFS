#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
假期特征表数据生成脚本
生成2025年1月1日到2025年5月31日的LSTM_COMMON_HOLIDAYFEATURE表数据
"""

import pymssql
import datetime
from datetime import date, timedelta
import calendar

# 数据库连接配置
conn = pymssql.connect(
    server='192.168.10.76',
    user='sa',
    password='Chency@123',
    database='master',
    port=1433
)
cursor = conn.cursor()

# 中国法定节假日定义（2022-2025年）
HOLIDAYS = {
    # ===== 2022年 =====
    # 元旦：1月1日-3日放假3天
    '2022-01-01': {'type': "New year's Day", 'days': 3, 'day_num': 1},
    '2022-01-02': {'type': "New year's Day", 'days': 3, 'day_num': 2},
    '2022-01-03': {'type': "New year's Day", 'days': 3, 'day_num': 3},
    
    # 春节：1月31日-2月6日放假7天
    '2022-01-31': {'type': 'Spring Festival', 'days': 7, 'day_num': 1},
    '2022-02-01': {'type': 'Spring Festival', 'days': 7, 'day_num': 2},
    '2022-02-02': {'type': 'Spring Festival', 'days': 7, 'day_num': 3},
    '2022-02-03': {'type': 'Spring Festival', 'days': 7, 'day_num': 4},
    '2022-02-04': {'type': 'Spring Festival', 'days': 7, 'day_num': 5},
    '2022-02-05': {'type': 'Spring Festival', 'days': 7, 'day_num': 6},
    '2022-02-06': {'type': 'Spring Festival', 'days': 7, 'day_num': 7},
    
    # 清明节：4月3日-5日放假3天
    '2022-04-03': {'type': 'Tomb-sweeping Day', 'days': 3, 'day_num': 1},
    '2022-04-04': {'type': 'Tomb-sweeping Day', 'days': 3, 'day_num': 2},
    '2022-04-05': {'type': 'Tomb-sweeping Day', 'days': 3, 'day_num': 3},
    
    # 劳动节：4月30日-5月4日放假5天
    '2022-04-30': {'type': 'Labour Day', 'days': 5, 'day_num': 1},
    '2022-05-01': {'type': 'Labour Day', 'days': 5, 'day_num': 2},
    '2022-05-02': {'type': 'Labour Day', 'days': 5, 'day_num': 3},
    '2022-05-03': {'type': 'Labour Day', 'days': 5, 'day_num': 4},
    '2022-05-04': {'type': 'Labour Day', 'days': 5, 'day_num': 5},
    
    # 端午节：6月3日-5日放假3天
    '2022-06-03': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 1},
    '2022-06-04': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 2},
    '2022-06-05': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 3},
    
    # 中秋节：9月10日-12日放假3天
    '2022-09-10': {'type': 'Mid-autumn Festival', 'days': 3, 'day_num': 1},
    '2022-09-11': {'type': 'Mid-autumn Festival', 'days': 3, 'day_num': 2},
    '2022-09-12': {'type': 'Mid-autumn Festival', 'days': 3, 'day_num': 3},
    
    # 国庆节：10月1日-7日放假7天
    '2022-10-01': {'type': 'National Day', 'days': 7, 'day_num': 1},
    '2022-10-02': {'type': 'National Day', 'days': 7, 'day_num': 2},
    '2022-10-03': {'type': 'National Day', 'days': 7, 'day_num': 3},
    '2022-10-04': {'type': 'National Day', 'days': 7, 'day_num': 4},
    '2022-10-05': {'type': 'National Day', 'days': 7, 'day_num': 5},
    '2022-10-06': {'type': 'National Day', 'days': 7, 'day_num': 6},
    '2022-10-07': {'type': 'National Day', 'days': 7, 'day_num': 7},
    
    # ===== 2023年 =====
    # 元旦：12月31日-1月2日放假3天
    '2022-12-31': {'type': "New year's Day", 'days': 3, 'day_num': 1},
    '2023-01-01': {'type': "New year's Day", 'days': 3, 'day_num': 2},
    '2023-01-02': {'type': "New year's Day", 'days': 3, 'day_num': 3},
    
    # 春节：1月21日-27日放假7天
    '2023-01-21': {'type': 'Spring Festival', 'days': 7, 'day_num': 1},
    '2023-01-22': {'type': 'Spring Festival', 'days': 7, 'day_num': 2},
    '2023-01-23': {'type': 'Spring Festival', 'days': 7, 'day_num': 3},
    '2023-01-24': {'type': 'Spring Festival', 'days': 7, 'day_num': 4},
    '2023-01-25': {'type': 'Spring Festival', 'days': 7, 'day_num': 5},
    '2023-01-26': {'type': 'Spring Festival', 'days': 7, 'day_num': 6},
    '2023-01-27': {'type': 'Spring Festival', 'days': 7, 'day_num': 7},
    
    # 清明节：4月5日放假1天
    '2023-04-05': {'type': 'Tomb-sweeping Day', 'days': 1, 'day_num': 1},
    
    # 劳动节：4月29日-5月3日放假5天
    '2023-04-29': {'type': 'Labour Day', 'days': 5, 'day_num': 1},
    '2023-04-30': {'type': 'Labour Day', 'days': 5, 'day_num': 2},
    '2023-05-01': {'type': 'Labour Day', 'days': 5, 'day_num': 3},
    '2023-05-02': {'type': 'Labour Day', 'days': 5, 'day_num': 4},
    '2023-05-03': {'type': 'Labour Day', 'days': 5, 'day_num': 5},
    
    # 端午节：6月22日-24日放假3天
    '2023-06-22': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 1},
    '2023-06-23': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 2},
    '2023-06-24': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 3},
    
    # 中秋国庆：9月29日-10月6日放假8天
    '2023-09-29': {'type': 'Mid-autumn Festival', 'days': 8, 'day_num': 1},
    '2023-09-30': {'type': 'Mid-autumn Festival', 'days': 8, 'day_num': 2},
    '2023-10-01': {'type': 'National Day', 'days': 8, 'day_num': 3},
    '2023-10-02': {'type': 'National Day', 'days': 8, 'day_num': 4},
    '2023-10-03': {'type': 'National Day', 'days': 8, 'day_num': 5},
    '2023-10-04': {'type': 'National Day', 'days': 8, 'day_num': 6},
    '2023-10-05': {'type': 'National Day', 'days': 8, 'day_num': 7},
    '2023-10-06': {'type': 'National Day', 'days': 8, 'day_num': 8},
    
    # ===== 2024年 =====
    # 元旦：1月1日放假1天
    '2024-01-01': {'type': "New year's Day", 'days': 1, 'day_num': 1},
    
    # 春节：2月10日-17日放假8天
    '2024-02-10': {'type': 'Spring Festival', 'days': 8, 'day_num': 1},
    '2024-02-11': {'type': 'Spring Festival', 'days': 8, 'day_num': 2},
    '2024-02-12': {'type': 'Spring Festival', 'days': 8, 'day_num': 3},
    '2024-02-13': {'type': 'Spring Festival', 'days': 8, 'day_num': 4},
    '2024-02-14': {'type': 'Spring Festival', 'days': 8, 'day_num': 5},
    '2024-02-15': {'type': 'Spring Festival', 'days': 8, 'day_num': 6},
    '2024-02-16': {'type': 'Spring Festival', 'days': 8, 'day_num': 7},
    '2024-02-17': {'type': 'Spring Festival', 'days': 8, 'day_num': 8},
    
    # 清明节：4月4日-6日放假3天
    '2024-04-04': {'type': 'Tomb-sweeping Day', 'days': 3, 'day_num': 1},
    '2024-04-05': {'type': 'Tomb-sweeping Day', 'days': 3, 'day_num': 2},
    '2024-04-06': {'type': 'Tomb-sweeping Day', 'days': 3, 'day_num': 3},
    
    # 劳动节：5月1日-5日放假5天
    '2024-05-01': {'type': 'Labour Day', 'days': 5, 'day_num': 1},
    '2024-05-02': {'type': 'Labour Day', 'days': 5, 'day_num': 2},
    '2024-05-03': {'type': 'Labour Day', 'days': 5, 'day_num': 3},
    '2024-05-04': {'type': 'Labour Day', 'days': 5, 'day_num': 4},
    '2024-05-05': {'type': 'Labour Day', 'days': 5, 'day_num': 5},
    
    # 端午节：6月8日-10日放假3天
    '2024-06-08': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 1},
    '2024-06-09': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 2},
    '2024-06-10': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 3},
    
    # 中秋节：9月15日-17日放假3天
    '2024-09-15': {'type': 'Mid-autumn Festival', 'days': 3, 'day_num': 1},
    '2024-09-16': {'type': 'Mid-autumn Festival', 'days': 3, 'day_num': 2},
    '2024-09-17': {'type': 'Mid-autumn Festival', 'days': 3, 'day_num': 3},
    
    # 国庆节：10月1日-7日放假7天
    '2024-10-01': {'type': 'National Day', 'days': 7, 'day_num': 1},
    '2024-10-02': {'type': 'National Day', 'days': 7, 'day_num': 2},
    '2024-10-03': {'type': 'National Day', 'days': 7, 'day_num': 3},
    '2024-10-04': {'type': 'National Day', 'days': 7, 'day_num': 4},
    '2024-10-05': {'type': 'National Day', 'days': 7, 'day_num': 5},
    '2024-10-06': {'type': 'National Day', 'days': 7, 'day_num': 6},
    '2024-10-07': {'type': 'National Day', 'days': 7, 'day_num': 7},
    
    # ===== 2025年（根据国务院办公厅2024年11月12日通知） =====
    # 元旦：1月1日放假1天
    '2025-01-01': {'type': "New year's Day", 'days': 1, 'day_num': 1},
    
    # 春节：1月28日（除夕）-2月4日放假8天（新增除夕）
    '2025-01-28': {'type': 'Spring Festival', 'days': 8, 'day_num': 1},
    '2025-01-29': {'type': 'Spring Festival', 'days': 8, 'day_num': 2},
    '2025-01-30': {'type': 'Spring Festival', 'days': 8, 'day_num': 3},
    '2025-01-31': {'type': 'Spring Festival', 'days': 8, 'day_num': 4},
    '2025-02-01': {'type': 'Spring Festival', 'days': 8, 'day_num': 5},
    '2025-02-02': {'type': 'Spring Festival', 'days': 8, 'day_num': 6},
    '2025-02-03': {'type': 'Spring Festival', 'days': 8, 'day_num': 7},
    '2025-02-04': {'type': 'Spring Festival', 'days': 8, 'day_num': 8},
    
    # 清明节：4月4日-6日放假3天
    '2025-04-04': {'type': 'Tomb-sweeping Day', 'days': 3, 'day_num': 1},
    '2025-04-05': {'type': 'Tomb-sweeping Day', 'days': 3, 'day_num': 2},
    '2025-04-06': {'type': 'Tomb-sweeping Day', 'days': 3, 'day_num': 3},
    
    # 劳动节：5月1日-5日放假5天（新增5月2日）
    '2025-05-01': {'type': 'Labour Day', 'days': 5, 'day_num': 1},
    '2025-05-02': {'type': 'Labour Day', 'days': 5, 'day_num': 2},
    '2025-05-03': {'type': 'Labour Day', 'days': 5, 'day_num': 3},
    '2025-05-04': {'type': 'Labour Day', 'days': 5, 'day_num': 4},
    '2025-05-05': {'type': 'Labour Day', 'days': 5, 'day_num': 5},
    
    # 端午节：5月31日-6月2日放假3天
    '2025-05-31': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 1},
    '2025-06-01': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 2},
    '2025-06-02': {'type': 'Dragon Boat Festival', 'days': 3, 'day_num': 3},
    
    # 国庆节中秋节：10月1日-8日放假8天（中秋国庆连休）
    '2025-10-01': {'type': 'National Day', 'days': 8, 'day_num': 1},
    '2025-10-02': {'type': 'National Day', 'days': 8, 'day_num': 2},
    '2025-10-03': {'type': 'National Day', 'days': 8, 'day_num': 3},
    '2025-10-04': {'type': 'National Day', 'days': 8, 'day_num': 4},
    '2025-10-05': {'type': 'National Day', 'days': 8, 'day_num': 5},
    '2025-10-06': {'type': 'Mid-autumn Festival', 'days': 8, 'day_num': 6},
    '2025-10-07': {'type': 'Mid-autumn Festival', 'days': 8, 'day_num': 7},
    '2025-10-08': {'type': 'Mid-autumn Festival', 'days': 8, 'day_num': 8},
    
    # ===== 2026年（仅1月1日） =====
    # 元旦：1月1日放假1天
    '2026-01-01': {'type': "New year's Day", 'days': 1, 'day_num': 1},
}

# 调休工作日（周末调班）2022-2025年
WORKDAYS = {
    # 2022年调休工作日
    '2022-01-29': True,  # 春节前调休
    '2022-01-30': True,  # 春节前调休
    '2022-04-02': True,  # 清明节前调休
    '2022-04-24': True,  # 五一前调休
    '2022-05-07': True,  # 五一后调休
    '2022-10-08': True,  # 国庆后调休
    '2022-10-09': True,  # 国庆后调休
    
    # 2023年调休工作日
    '2023-01-28': True,  # 春节前调休
    '2023-01-29': True,  # 春节前调休
    '2023-04-23': True,  # 五一前调休
    '2023-05-06': True,  # 五一后调休
    '2023-06-25': True,  # 端午后调休
    '2023-10-07': True,  # 中秋国庆后调休
    '2023-10-08': True,  # 中秋国庆后调休
    
    # 2024年调休工作日
    '2024-02-04': True,  # 春节前调休
    '2024-02-18': True,  # 春节后调休
    '2024-04-07': True,  # 清明后调休
    '2024-04-28': True,  # 五一前调休
    '2024-05-11': True,  # 五一后调休
    '2024-09-14': True,  # 中秋前调休
    '2024-09-29': True,  # 国庆前调休
    '2024-10-12': True,  # 国庆后调休
    
    # 2025年调休工作日（根据国务院办公厅2024年11月12日通知）
    '2025-01-26': True,  # 春节前调休（周日上班）
    '2025-02-08': True,  # 春节后调休（周六上班）
    '2025-04-27': True,  # 五一前调休（周日上班）
    '2025-05-30': True,  # 端午前调休（周五上班，实际不是调休）
    '2025-09-28': True,  # 国庆前调休（周日上班）
    '2025-10-11': True,  # 国庆后调休（周六上班）
}

def get_week_number(date_obj):
    """获取日期是星期几（1=周一，7=周日）"""
    return date_obj.isoweekday()

def is_weekend(date_obj):
    """判断是否为周末"""
    return date_obj.weekday() >= 5  # 5=周六, 6=周日

def is_holiday(date_str):
    """判断是否为法定节假日"""
    return date_str in HOLIDAYS

def is_workday_adjustment(date_str):
    """判断是否为调休工作日"""
    return date_str in WORKDAYS

def get_holiday_days_for_date(current_date):
    """获取当前日期的假期天数标记
    
    规则：
    1. 假期本身标记为假期总天数
    2. 假期前一天也标记为假期总天数
    """
    date_str = current_date.strftime('%Y-%m-%d')
    next_date_str = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 如果当前日期是假期
    if date_str in HOLIDAYS:
        return HOLIDAYS[date_str]['days']
    
    # 如果第二天是假期的第一天，当前日期标记为假期天数
    if next_date_str in HOLIDAYS and HOLIDAYS[next_date_str]['day_num'] == 1:
        return HOLIDAYS[next_date_str]['days']
    
    return 0

def calculate_holiday_progress(current_date):
    """计算假期进度（小数形式）
    
    规则：
    - 假期中的每一天按照进度计算
    - 例如3天假期：第1天=0.3333333333，第2天=0.6666666667，第3天=1.0000000000
    - 非假期日期返回0
    """
    date_str = current_date.strftime('%Y-%m-%d')
    
    if date_str in HOLIDAYS:
        holiday_info = HOLIDAYS[date_str]
        day_num = holiday_info['day_num']
        total_days = holiday_info['days']
        
        # 计算进度：当前天数/总天数，保留10位小数
        progress = round(day_num / total_days, 10)
        return progress
    
    return 0

def generate_holiday_features(start_date, end_date):
    """生成假期特征数据"""
    
    # 先删除可能存在的重复数据
    print("删除可能存在的重复数据...")
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    delete_sql = f"""
    DELETE FROM LSTM_COMMON_HOLIDAYFEATURE 
    WHERE F_DATE >= {start_date_str} AND F_DATE < {end_date_str}
    """
    cursor.execute(delete_sql)
    conn.commit()
    
    print(f"开始生成 {start_date} 到 {end_date} 的假期特征数据...")
    print(f"总共需要生成 {(end_date - start_date).days + 1} 天的数据...")
    
    current_date = start_date
    batch_data = []
    processed_count = 0
    
    while current_date < end_date:  # 改为小于，不包含结束日期
        date_str = current_date.strftime('%Y-%m-%d')
        date_int = int(current_date.strftime('%Y%m%d'))
        
        # 基础日期信息
        week_num = get_week_number(current_date)
        is_wkend = is_weekend(current_date)
        is_hol = is_holiday(date_str)
        is_work_adj = is_workday_adjustment(date_str)
        
        # 确定日期特征
        if is_work_adj:
            # 调休工作日
            f_datefeatures = 'weekday'
        elif is_hol:
            # 法定节假日
            f_datefeatures = 'holiday'
        elif is_wkend:
            # 普通周末
            f_datefeatures = 'weekend'
        else:
            # 普通工作日
            f_datefeatures = 'weekday'
        
        # 假期类型和天数
        if is_hol:
            holiday_info = HOLIDAYS[date_str]
            f_holidaytype = holiday_info['type']
            f_holidaydays = get_holiday_days_for_date(current_date)  # 使用新的计算方法
            f_holidaythday = calculate_holiday_progress(current_date)  # 计算假期进度
        else:
            f_holidaytype = None
            f_holidaydays = get_holiday_days_for_date(current_date)  # 非假期日期也可能有标记
            f_holidaythday = 0
        
        # 各种标志位
        f_isholiday = 1 if is_hol else 0
        f_isnongli = 0  # 暂不考虑农历节日
        f_isyangli = 1 if is_hol and date_str.endswith(('-01-01', '-05-01')) else 0
        f_nextday = 1 if (current_date + timedelta(days=1)).strftime('%Y-%m-%d') in HOLIDAYS else 0
        f_isspacial = 0  # 特殊假期标志（忽略，设为0）
        
        # 夏季标志（6-8月）
        is_summer = 1 if current_date.month in [6, 7, 8] else 0
        
        # 是否为节假日后第一个工作日
        is_first = 0
        if f_datefeatures == 'weekday' and not is_hol:
            # 检查前一天是否为假期
            prev_date = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
            if prev_date in HOLIDAYS or (current_date - timedelta(days=1)).weekday() == 6:
                is_first = 1
        
        # 构建数据行
        row_data = (
            f"1748707201{current_date.strftime('%Y%m%d')}",  # ID
            date_int,                    # F_DATE
            week_num,                   # F_WEEK
            f_datefeatures,             # F_DATEFEATURES
            f_holidaytype,              # F_HOLIDAYTYPE
            f_isholiday,               # F_ISHOLIDAY
            f_isnongli,                # F_ISNONGLI
            f_isyangli,                # F_ISYANGLI
            f_nextday,                 # F_NEXTDAY
            f_holidaydays,             # F_HOLIDAYDAYS
            f_holidaythday,            # F_HOLIDAYTHDAY
            f_isspacial,               # F_ISSPACIAL
            datetime.datetime.now(),    # CREATETIME
            'SYSTEM',                   # CREATOR
            datetime.datetime.now(),    # MODIFYTIME
            'SYSTEM',                   # MODIFIER
            '',                         # REMARKS
            is_summer,                  # IS_SUMMER
            is_first                    # IS_FIRST（移除YEAR列）
        )
        
        batch_data.append(row_data)
        processed_count += 1
        
        # 每处理100天显示进度
        if processed_count % 100 == 0:
            print(f"已处理 {processed_count} 天数据...")
        
        # 每100条记录批量插入一次
        if len(batch_data) >= 100:
            insert_batch_data(batch_data)
            batch_data = []
        
        current_date += timedelta(days=1)
    
    # 插入剩余数据
    if batch_data:
        insert_batch_data(batch_data)
    
    print(f"数据生成完成！总共处理了 {processed_count} 天的数据")

def insert_batch_data(batch_data):
    """批量插入数据"""
    insert_sql = """
    INSERT INTO LSTM_COMMON_HOLIDAYFEATURE (
        ID, F_DATE, F_WEEK, F_DATEFEATURES, F_HOLIDAYTYPE,
        F_ISHOLIDAY, F_ISNONGLI, F_ISYANGLI, F_NEXTDAY, F_HOLIDAYDAYS,
        F_HOLIDAYTHDAY, F_ISSPACIAL, CREATETIME, CREATOR, MODIFYTIME,
        MODIFIER, REMARKS, IS_SUMMER, IS_FIRST
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """
    
    try:
        cursor.executemany(insert_sql, batch_data)
        conn.commit()
        print(f"成功插入 {len(batch_data)} 条记录")
    except Exception as e:
        print(f"插入数据时出错: {e}")
        conn.rollback()

def create_table_if_not_exists():
    """如果表不存在则创建表"""
    # 先删除表（如果存在）
    drop_table_sql = """
    IF EXISTS (SELECT * FROM sysobjects WHERE name='LSTM_COMMON_HOLIDAYFEATURE' AND xtype='U')
    DROP TABLE LSTM_COMMON_HOLIDAYFEATURE
    """
    
    create_table_sql = """
    CREATE TABLE LSTM_COMMON_HOLIDAYFEATURE (
        ID VARCHAR(50) PRIMARY KEY,
        F_DATE INT,
        F_WEEK INT,
        F_DATEFEATURES VARCHAR(100),
        F_HOLIDAYTYPE VARCHAR(50),
        F_ISHOLIDAY INT,
        F_ISNONGLI INT,
        F_ISYANGLI INT,
        F_NEXTDAY INT,
        F_HOLIDAYDAYS INT,
        F_HOLIDAYTHDAY DECIMAL(21,10),
        F_ISSPACIAL INT,
        CREATETIME DATETIME,
        CREATOR VARCHAR(100),
        MODIFYTIME DATETIME,
        MODIFIER VARCHAR(100),
        REMARKS VARCHAR(1000),
        IS_SUMMER INT,
        IS_FIRST INT
    )
    """
    
    try:
        cursor.execute(drop_table_sql)
        conn.commit()
        print("删除旧表完成")
        
        cursor.execute(create_table_sql)
        conn.commit()
        print("表创建完成")
    except Exception as e:
        print(f"创建表时出错: {e}")
        conn.rollback()

def main():
    """主函数"""
    try:
        # 检查并创建表
        create_table_if_not_exists()
        
        # 定义日期范围
        start_date = date(2022, 1, 1)
        end_date = date(2026, 1, 1)
        
        # 生成假期特征数据
        generate_holiday_features(start_date, end_date)
        
        # 查询并显示生成的数据统计
        cursor.execute("""
            SELECT 
                COUNT(*) as total_count,
                SUM(F_ISHOLIDAY) as holiday_count,
                SUM(CASE WHEN F_DATEFEATURES = 'weekend' THEN 1 ELSE 0 END) as weekend_count,
                SUM(CASE WHEN F_DATEFEATURES = 'weekday' THEN 1 ELSE 0 END) as weekday_count
            FROM LSTM_COMMON_HOLIDAYFEATURE 
            WHERE F_DATE >= 20220101 AND F_DATE < 20260101
        """)
        
        result = cursor.fetchone()
        if result:
            total, holidays, weekends, weekdays = result
            print(f"\n数据统计:")
            print(f"总记录数: {total}")
            print(f"节假日数: {holidays}")
            print(f"周末数: {weekends}")
            print(f"工作日数: {weekdays}")
        
        print("\n脚本执行完成！")
        
    except Exception as e:
        print(f"执行过程中出错: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()