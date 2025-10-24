
import pandas as pd
import pymssql
from datetime import datetime, timedelta

def query_line_daily_counts_all() -> pd.DataFrame:
    """
    查询LineDailyFlowHistory表，统计每个线路在每一天有多少条数据，返回全量数据。
    返回:
        DataFrame，包含F_DATE, F_LINENO, F_LINENAME, record_count
    """
    try:
        conn = pymssql.connect(
            server='192.168.10.76',
            user='sa',
            password='Chency@123',
            database='master',
            port='1433'
        )
        query = f"""
        SELECT 
            F_DATE,
            F_LINENO,
            F_LINENAME,
            COUNT(*) AS record_count
        FROM 
            dbo.LineDailyFlowHistory
        GROUP BY 
            F_DATE, F_LINENO, F_LINENAME
        ORDER BY 
            F_LINENO, F_DATE
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

def query_line_hourly_flow_history() -> pd.DataFrame:
    """
    查询LineHourlyFlowHistory表，返回2023年1月1日及以后的所有小时客流数据。
    返回:
        DataFrame，包含H.ID, H.F_DATE, H.F_HOUR, H.F_KLCOUNT, H.F_LINENO, H.F_LINENAME, H.CREATETIME, H.CREATOR
    """
    try:
        conn = pymssql.connect(
            server='192.168.10.76',
            user='sa',
            password='Chency@123',
            database='master',
            port='1433'
        )
        query = f"""
        SELECT 
            H.ID,
            H.F_DATE,
            H.F_HOUR,
            H.F_KLCOUNT,
            H.F_LINENO,
            H.F_LINENAME,
            H.CREATETIME,
            H.CREATOR
        FROM 
            dbo.LineHourlyFlowHistory AS H
        WHERE 
            H.F_DATE >= 20230101
        ORDER BY 
            H.F_DATE, H.F_LINENO, H.F_HOUR
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

def stat_line_daily_flow(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    统计每个线路每天的总客流量（日客流），返回DataFrame。
    返回:
        DataFrame，包含F_DATE, F_LINENO, F_LINENAME, daily_klcount
    """
    # 确保F_DATE为字符串
    df_hourly['F_DATE'] = df_hourly['F_DATE'].astype(str)
    grouped = df_hourly.groupby(['F_DATE', 'F_LINENO', 'F_LINENAME'], as_index=False)['F_KLCOUNT'].sum()
    grouped = grouped.rename(columns={'F_KLCOUNT': 'daily_klcount'})
    return grouped

def find_continuous_periods(dates):
    """
    输入已排序的日期字符串列表，返回连续区间的 (start, end, count) 列表
    """
    if not dates:
        return []
    periods = []
    start = dates[0]
    prev = dates[0]
    count = 1
    for d in dates[1:]:
        prev_dt = datetime.strptime(prev, "%Y%m%d")
        curr_dt = datetime.strptime(d, "%Y%m%d")
        if (curr_dt - prev_dt).days == 1:
            count += 1
        else:
            periods.append((start, prev, count))
            start = d
            count = 1
        prev = d
    periods.append((start, prev, count))
    return periods

def stat_line_continuous_periods(df: pd.DataFrame):
    """
    统计每个线路的连续时间段及天数（同一线路合并，不区分线路名）
    返回：dict，key为F_LINENO，value为连续区间列表
    """
    results = {}
    # 合并同一线路（F_LINENO），不区分F_LINENAME
    for lineno, group in df.groupby('F_LINENO'):
        # 取所有该线路的所有日期
        dates = sorted(group['F_DATE'].astype(str).unique())
        periods = find_continuous_periods(dates)
        # 取该线路出现最多的线路名作为代表
        linename = group['F_LINENAME'].mode().iloc[0] if not group['F_LINENAME'].empty else ""
        results[lineno] = {
            "linename": linename,
            "periods": periods
        }
    return results

def print_line_periods(periods_dict):
    """
    打印每个线路的连续时间段统计（同一线路合并）
    """
    for lineno in sorted(periods_dict.keys()):
        linename = periods_dict[lineno]["linename"]
        periods = periods_dict[lineno]["periods"]
        print(f"线路{lineno}（{linename}）:")
        for start, end, days in periods:
            print(f"  {start}～{end}  共{days}天")
        print("-" * 30)

def print_daily_flow_stats(df_daily: pd.DataFrame):
    """
    打印每个线路每天的总客流量（日客流）统计
    """
    for lineno, group in df_daily.groupby('F_LINENO'):
        linename = group['F_LINENAME'].mode().iloc[0] if not group['F_LINENAME'].empty else ""
        print(f"线路{lineno}（{linename}）日客流：")
        # for _, row in group.iterrows():
        #     print(f"  {row['F_DATE']}: {row['daily_klcount']}")
        print("-" * 30)

def print_hourly_flow_stats(df_hourly: pd.DataFrame):
    """
    打印每个线路小时客流数据的统计（只统计天数和总小时数，不逐天打印）
    """
    # 确保F_DATE为字符串
    df_hourly['F_DATE'] = df_hourly['F_DATE'].astype(str)
    for lineno, group in df_hourly.groupby('F_LINENO'):
        linename = group['F_LINENAME'].mode().iloc[0] if not group['F_LINENAME'].empty else ""
        unique_days = group['F_DATE'].nunique()
        total_hours = len(group)
        print(f"线路{lineno}（{linename}）小时客流：")
        print(f"  覆盖天数: {unique_days}天，总小时数: {total_hours}")
        print("-" * 30)

def main():
    # 原有功能：统计LineDailyFlowHistory表的连续时间段
    df = query_line_daily_counts_all()
    periods_dict = stat_line_continuous_periods(df)
    print("所有线路的连续时间段（同一线路合并）:")
    print_line_periods(periods_dict)

    # 新增功能：统计LineHourlyFlowHistory表的日客流
    print("\n==== 日客流统计（基于LineHourlyFlowHistory） ====")
    df_hourly = query_line_hourly_flow_history()
    df_daily = stat_line_daily_flow(df_hourly)
    print_daily_flow_stats(df_daily)

    # 新增功能：小时客流统计（不逐天打印，只统计天数和总小时数）
    print("\n==== 小时客流统计（基于LineHourlyFlowHistory） ====")
    print_hourly_flow_stats(df_hourly)

if __name__ == "__main__":
    main()