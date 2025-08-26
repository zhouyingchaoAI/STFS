import pandas as pd
import pymssql
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_weather_duplicates(start_date: str, end_date: str):
    """
    分析天气数据表中是否存在同一天多条信息的情况

    参数:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
    """
    try:
        conn = pymssql.connect(
            server='192.168.10.76',
            user='sa',
            password='Chency@123',
            database='master',
            port='1433'
        )
        # 检查WeatherHistory表同一天多条
        weather_query = f"""
        SELECT 
            F_DATE,
            COUNT(*) as cnt
        FROM 
            dbo.WeatherHistory
        WHERE 
            F_DATE >= '{start_date}'
            AND F_DATE <= '{end_date}'
        GROUP BY 
            F_DATE
        HAVING COUNT(*) > 1
        ORDER BY 
            F_DATE
        """
        weather_df = pd.read_sql(weather_query, conn)
        if weather_df.empty:
            print("没有发现同一天有多条天气信息的情况。")
        else:
            print("发现以下日期有多条天气信息：")
            print(weather_df)

        # 检查LSTM_COMMON_HOLIDAYFEATURE表同一天多条
        holiday_query = f"""
        SELECT 
            F_DATE,
            COUNT(*) as cnt
        FROM 
            dbo.LSTM_COMMON_HOLIDAYFEATURE
        WHERE 
            F_DATE >= '{start_date}'
            AND F_DATE <= '{end_date}'
        GROUP BY 
            F_DATE
        HAVING COUNT(*) > 1
        ORDER BY 
            F_DATE
        """
        holiday_df = pd.read_sql(holiday_query, conn)
        if holiday_df.empty:
            print("没有发现同一天有多条节日安排的情况。")
        else:
            print("发现以下日期有多条节日安排：")
            print(holiday_df)

        conn.close()
    except Exception as e:
        logger.error(f"分析数据表时出错: {e}")

# 示例用法
if __name__ == "__main__":
    # 这里可以根据需要修改日期范围
    analyze_weather_duplicates("20230101", "20260531")