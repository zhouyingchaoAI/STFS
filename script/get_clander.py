#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节假日数据生成器 - 优化版
生成指定时间范围内的节假日安排数据
"""

import pandas as pd
from datetime import datetime, timedelta
import calendar
from typing import Dict, List, Tuple

class HolidayGenerator:
    """节假日数据生成器"""
    
    def __init__(self):
        """初始化节假日类型"""
        self.holiday_types = {
            1: "元旦",
            2: "春节", 
            3: "清明节",
            4: "五一劳动节",
            5: "端午节",
            6: "中秋节",
            7: "国庆节",
            8: "周末",
            9: "节假日调休补班",
            11: "暑假",
            12: "寒假"
        }
        
        # 优先级定义：数值越大优先级越高
        self.priority_order = {
            8: 1,   # 周末 - 最低优先级
            9: 2,   # 调休补班
            11: 3,  # 暑假
            12: 3,  # 寒假  
            1: 4,   # 元旦 - 法定节假日
            2: 4,   # 春节 - 法定节假日
            3: 4,   # 清明节 - 法定节假日
            4: 4,   # 劳动节 - 法定节假日
            5: 4,   # 端午节 - 法定节假日
            6: 4,   # 中秋节 - 法定节假日
            7: 4,   # 国庆节 - 法定节假日
        }
        
        # 定义各年度节假日安排（基于历年国务院通知和学校安排）
        self.holiday_arrangements = {
            2017: {
                "元旦": [("2016-12-31", "2017-01-02")],
                "春节": [("2017-01-21", "2017-01-29")],
                "清明节": [("2017-04-03", "2017-04-04")],
                "劳动节": [("2017-05-01", "2017-05-01")],
                "端午节": [("2017-05-28", "2017-05-30")],
                "国庆节": [("2017-10-01", "2017-10-08")],
                "调休补班": [("2017-01-22","2017-01-22"),("2017-02-04","2017-02-04"),("2017-04-01","2017-04-01"),("2017-05-27","2017-05-27"),("2017-09-30","2017-09-30")],
                "暑假": [("2017-07-03", "2017-09-03")]
            },
            2018: {
                "元旦": [("2017-12-30", "2018-01-01")],
                "春节": [("2018-02-11", "2018-02-21")],
                "清明节": [("2018-04-05", "2018-04-07")],
                "劳动节": [("2018-04-29", "2018-05-01")],
                "端午节": [("2018-06-16", "2018-06-18")],
                "中秋节": [("2018-09-22", "2018-09-24")],
                "国庆节": [("2018-10-01", "2018-10-07")],
                "调休补班": [("2018-02-11","2018-02-11"),("2018-02-24","2018-02-24"),("2018-04-08","2018-04-08"),("2018-04-28","2018-04-28"),("2018-09-29","2018-09-29"),("2018-09-30","2018-09-30")],  
                "暑假": [("2018-07-02", "2018-09-02")]
            },
            2019: {
                "元旦": [("2018-12-30", "2019-01-01")],
                "春节": [("2019-02-04", "2019-02-10")],
                "清明节": [("2019-04-05", "2019-04-07")],
                "劳动节": [("2019-05-01", "2019-05-04")],
                "端午节": [("2019-06-07", "2019-06-09")],
                "中秋节": [("2019-09-13", "2019-09-15")],
                "国庆节": [("2019-10-01", "2019-10-07")],
                "调休补班": [("2019-02-02","2019-02-02"),("2019-02-03","2019-02-03"),("2019-04-28","2019-04-28"),("2019-05-05","2019-05-05"),("2019-09-29","2019-09-29"),("2019-10-12","2019-10-12")],  
                "暑假": [("2019-07-01", "2019-09-01")]
            },
            2020: {
                "元旦": [("2019-12-31", "2020-01-01")],
                "春节": [("2020-01-24", "2020-02-02")],  # 因疫情延长
                "清明节": [("2020-04-04", "2020-04-06")],
                "劳动节": [("2020-05-01", "2020-05-05")],
                "端午节": [("2020-06-25", "2020-06-27")],
                "国庆节": [("2020-10-01", "2020-10-08")],
                "调休补班": [("2020-01-19","2020-01-19"),("2020-04-26","2020-04-26"),("2020-05-09","2020-05-09"),("2020-06-28","2020-06-28"),("2020-09-27","2020-09-27"),("2020-10-10","2020-10-10")],  
                "暑假": [("2020-06-29", "2020-08-30")],
            },
            2021: {
                "元旦": [("2021-01-01", "2021-01-03")],
                "春节": [("2021-02-11", "2021-02-17")],
                "清明节": [("2021-04-03", "2021-04-05")],
                "劳动节": [("2021-05-01", "2021-05-05")],
                "端午节": [("2021-06-12", "2021-06-14")],
                "中秋节": [("2021-09-19", "2021-09-21")],
                "国庆节": [("2021-10-01", "2021-10-07")],
                "调休补班": [("2021-02-07","2021-02-07"),("2021-02-20","2021-02-20"),("2021-04-25","2021-04-25"),("2021-05-08","2021-05-08"),("2021-09-18","2021-09-18"),("2021-09-26","2021-09-26"),("2021-10-09","2021-10-09")],  
                "暑假": [("2021-07-05", "2021-09-05")]
            },
            2022: {
                "元旦": [("2022-01-01", "2022-01-03")],
                "春节": [("2022-01-31", "2022-02-06")],
                "清明节": [("2022-04-03", "2022-04-05")],
                "劳动节": [("2022-04-30", "2022-05-04")],
                "端午节": [("2022-06-03", "2022-06-05")],
                "中秋节": [("2022-09-10", "2022-09-12")],
                "国庆节": [("2022-10-01", "2022-10-07")],
                "调休补班": [("2022-01-29","2022-01-29"),("2022-01-30","2022-01-30"),("2022-04-02","2022-04-02"),("2022-04-24","2022-04-24"),("2022-05-07","2022-05-07"),("2022-10-08","2022-10-08"),("2022-10-09","2022-10-09")],  
                "暑假": [("2022-07-04", "2022-09-04")]
            },
            2023: {
                "元旦": [("2022-12-31", "2023-01-02")],
                "春节": [("2023-01-21", "2023-01-27")],
                "清明节": [("2023-04-05", "2023-04-05")],
                "劳动节": [("2023-04-29", "2023-05-03")],
                "端午节": [("2023-06-22", "2023-06-24")],
                "国庆节": [("2023-09-29", "2023-10-06")],
                "调休补班": [("2023-01-28","2023-01-28"),("2023-01-29","2023-01-29"),("2023-04-23","2023-04-23"),("2023-05-06","2023-05-06"),("2023-06-25","2023-06-25"),("2023-10-07","2023-10-07"),("2023-10-08","2023-10-08")],  
                "暑假": [("2023-07-03", "2023-09-03")]
            },
            2024: {
                "元旦": [("2024-01-01", "2024-01-01")],
                "春节": [("2024-02-10", "2024-02-17")],
                "清明节": [("2024-04-04", "2024-04-06")],
                "劳动节": [("2024-05-01", "2024-05-05")],
                "端午节": [("2024-06-08", "2024-06-10")],
                "中秋节": [("2024-09-15", "2024-09-17")],
                "国庆节": [("2024-10-01", "2024-10-07")],
                "调休补班": [("2024-02-04","2024-02-04"),("2024-02-18","2024-02-18"),("2024-04-07","2024-04-07"),("2024-04-28","2024-04-28"),("2024-05-11","2024-05-11"),("2024-09-14","2024-09-14"),("2024-09-29","2024-09-29"),("2024-10-12","2024-10-12")],  
                "暑假": [("2024-07-01", "2024-09-01")]
            },
            # 2025年预估（实际以国务院通知为准）
            2025: {
                "元旦": [("2025-01-01", "2025-01-01")],
                "春节": [("2025-01-28", "2025-02-03")],  # 预估
                "清明节": [("2025-04-04", "2025-04-06")],  # 预估
                "劳动节": [("2025-05-01", "2025-05-05")],  # 预估
                "端午节": [("2025-05-31", "2025-06-02")],  # 预估
                "国庆节": [("2025-10-01", "2025-10-08")],  # 预估
                "调休补班": [("2025-01-26","2025-01-26"),("2025-02-08","2025-02-08"),("2025-04-27","2025-04-27"),("2025-09-28","2025-09-28"),("2025-10-11","2025-10-11")],  # 预估
                "暑假": [("2025-07-01", "2025-08-31")]   # 预估
            }
        }
    
    def get_week_of_year(self, date: datetime) -> int:
        """获取日期在年中的周数"""
        return date.isocalendar()[1]
    
    def get_day_of_week(self, date: datetime) -> int:
        """获取星期几（1=周一，7=周日）"""
        return date.isoweekday()
    
    def is_weekend(self, date: datetime) -> bool:
        """判断是否为周末"""
        return date.weekday() >= 5  # 5=周六, 6=周日
    
    def parse_date_range(self, start_str: str, end_str: str) -> List[datetime]:
        """解析日期范围，返回日期列表"""
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates
    
    def get_all_holiday_matches(self, date: datetime) -> List[Tuple[int, int, int]]:
        """
        获取指定日期的所有可能节假日匹配
        返回: List[(节假日类型, 节假日总天数, 节假日第几天)]
        """
        matches = []
        date_str = date.strftime("%Y-%m-%d")
        
        # 检查周末（最低优先级）
        if self.is_weekend(date):
            matches.append((8, 0, 0))
        
        # 检查多个年份的节假日配置（处理跨年情况）
        years_to_check = [date.year - 1, date.year, date.year + 1]
        
        for year in years_to_check:
            if year not in self.holiday_arrangements:
                continue
                
            year_holidays = self.holiday_arrangements[year]
            
            # 检查调休补班
            if "调休补班" in year_holidays:
                for start_str, end_str in year_holidays["调休补班"]:
                    makeup_dates = self.parse_date_range(start_str, end_str)
                    if date in makeup_dates:
                        matches.append((9, 0, 0))
            
            # 检查寒假和暑假
            for vacation_name, vacation_type in [("暑假", 11), ("寒假", 12)]:
                if vacation_name in year_holidays:
                    for start_str, end_str in year_holidays[vacation_name]:
                        vacation_dates = self.parse_date_range(start_str, end_str)
                        if date in vacation_dates:
                            matches.append((vacation_type, 0, 0))
            
            # 检查法定节假日
            holiday_mapping = {
                "元旦": 1, "春节": 2, "清明节": 3, "劳动节": 4,
                "端午节": 5, "中秋节": 6, "国庆节": 7
            }
            
            for holiday_name, holiday_type in holiday_mapping.items():
                if holiday_name not in year_holidays:
                    continue
                    
                for start_str, end_str in year_holidays[holiday_name]:
                    holiday_dates = self.parse_date_range(start_str, end_str)
                    total_days = len(holiday_dates)
                    
                    # 检查是否在节假日期间
                    if date in holiday_dates:
                        which_day = holiday_dates.index(date) + 1
                        matches.append((holiday_type, total_days, which_day))
                    
                    # 检查节前一天
                    if len(holiday_dates) > 0:
                        prev_day = holiday_dates[0] - timedelta(days=1)
                        if date == prev_day:
                            matches.append((holiday_type, 0, 0))  # 节前一天，f_HolidayDays=0
                    
                    # 检查节后一天
                    if len(holiday_dates) > 0:
                        next_day = holiday_dates[-1] + timedelta(days=1)
                        if date == next_day:
                            matches.append((holiday_type, 0, total_days + 1))  # 节后一天，f_HolidayDays=0
                    
                    # 对于国庆这样的长假，可能需要检查节后第二天
                    if holiday_type == 7 and total_days >= 7:  # 国庆等长假
                        if len(holiday_dates) > 0:
                            next_next_day = holiday_dates[-1] + timedelta(days=2)
                            if date == next_next_day:
                                matches.append((holiday_type, 0, total_days + 2))  # 节后第二天，f_HolidayDays=0
        
        return matches
    
    def get_holiday_info(self, date: datetime) -> Tuple[int, int, int]:
        """
        获取指定日期的节假日信息（按优先级处理）
        返回: (节假日类型, 节假日总天数, 节假日第几天)
        """
        matches = self.get_all_holiday_matches(date)
        
        if not matches:
            return 0, 0, 0
        
        # 按优先级排序，选择优先级最高的
        matches.sort(key=lambda x: self.priority_order.get(x[0], 0), reverse=True)
        
        return matches[0]
    
    def generate_holiday_data(self, start_date: str = "20170101", end_date: str = "20251231") -> pd.DataFrame:
        """
        生成指定范围内的节假日数据
        
        Args:
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
        
        Returns:
            包含节假日信息的DataFrame
        """
        # 解析日期
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        data = []
        current = start_dt
        
        print(f"开始生成数据：{start_date} - {end_date}")
        total_days = (end_dt - start_dt).days + 1
        processed = 0
        
        while current <= end_dt:
            # 获取基本信息
            f_date = current.strftime("%Y%m%d")
            f_year = current.year
            f_DayOfWeek = self.get_day_of_week(current)
            f_week = self.get_week_of_year(current)
            
            # 获取节假日信息
            f_HolidayType, f_HolidayDays, f_HolidayWhichDay = self.get_holiday_info(current)
            
            # COVID19标记（2020年设为1，其他年份为0）
            COVID19 = 1 if current.year == 2020 else 0
            
            # 天气编码（示例固定值，可根据需要修改）
            f_weather = 15
            
            data.append([
                f_date, f_year, f_DayOfWeek, f_week, f_HolidayType,
                f_HolidayDays, f_HolidayWhichDay, COVID19, f_weather
            ])
            
            current += timedelta(days=1)
            processed += 1
            
            if processed % 365 == 0:
                print(f"已处理: {processed}/{total_days} 天 ({processed/total_days*100:.1f}%)")
        
        # 创建DataFrame
        columns = [
            'f_date', 'f_year', 'f_DayOfWeek', 'f_week', 'f_HolidayType',
            'f_HolidayDays', 'f_HolidayWhichDay', 'COVID19', 'f_weather'
        ]
        
        df = pd.DataFrame(data, columns=columns)

        print(f"数据生成完成，共 {len(df)} 条记录")
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = "holiday_data.csv"):
        """保存数据到CSV文件"""
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"数据已保存到: {filename}")
    
    def print_sample(self, df: pd.DataFrame, n: int = 10):
        """打印样本数据"""
        print(f"\n前{n}条数据样本:")
        print(df.head(n).to_string(index=False))
        
        print(f"\n后{n}条数据样本:")
        print(df.tail(n).to_string(index=False))


def main():
    """主函数"""
    generator = HolidayGenerator()
    
    # 先测试跨年元旦期间的数据
    print("=" * 50)
    print("测试2024年跨年元旦期间数据")
    print("=" * 50)
    
    test_df = generator.generate_holiday_data("20231229", "20240103")
    print("\n2024年跨年元旦期间数据:")
    print(test_df.to_string(index=False))
    
    # 测试2024年国庆期间的数据
    print("\n" + "=" * 50)
    print("测试2024年国庆期间数据")
    print("=" * 50)
    
    test_df2 = generator.generate_holiday_data("20240928", "20241010")
    print("\n2024年国庆期间数据:")
    print(test_df2.to_string(index=False))
    
    print("\n" + "=" * 50)
    print("生成完整数据集")
    print("=" * 50)
    
    # 生成指定范围的数据
    start_date = "20171211"
    end_date = "20251231"
    
    # 生成数据
    df = generator.generate_holiday_data(start_date, end_date)
    
    # 显示样本
    generator.print_sample(df)
    
    # 保存到文件
    filename = f"holiday_data_{start_date}_{end_date}.csv"
    generator.save_to_csv(df, filename)
    
    # 统计信息
    print("\n" + "=" * 50)
    print("数据统计:")
    print("=" * 50)
    print(f"总天数: {len(df)}")
    print(f"年份范围: {df['f_year'].min()} - {df['f_year'].max()}")
    print(f"节假日类型统计:")
    holiday_stats = df[df['f_HolidayType'] > 0]['f_HolidayType'].value_counts().sort_index()
    for holiday_type, count in holiday_stats.items():
        if holiday_type in generator.holiday_types:
            print(f"  {holiday_type}: {generator.holiday_types[holiday_type]} - {count}天")


if __name__ == "__main__":
    main()