import pandas as pd
import pymssql


def read_station_flow_history(station_name):
    """
    读取 StationFlowPredict.dbo.STATION_FLOW_HISTORY 表中指定“STATION_NAME”的所有数据，
    并按 SQUAD_DATE 合并（相同日期的各项数值求和）

    参数:
        station_name: 车站名（字符串）
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
        SELECT [ID],
              [STATION_NAME],
              [ENTRY_NUM],
              [EXIT_NUM],
              [CHANGE_NUM],
              [PASSENGER_NUM],
              [FLOW_NUM],
              [SQUAD_DATE]
        FROM [StationFlowPredict].[dbo].[STATION_FLOW_HISTORY]
        WHERE [STATION_NAME] = N'{station_name}'
        """
        df = pd.read_sql(query, conn)
        conn.close()
        # 检查同一日期是否有多行
        duplicated_dates = df['SQUAD_DATE'].duplicated(keep=False)
        if duplicated_dates.any():
            print("同一个日期有多行，正在合并相同日期的数据。")
        # 按 SQUAD_DATE 合并（相同日期的各项数值求和）
        grouped = df.groupby('SQUAD_DATE', as_index=False).agg({
            'ENTRY_NUM': 'sum',
            'EXIT_NUM': 'sum',
            'CHANGE_NUM': 'sum',
            'PASSENGER_NUM': 'sum',
            'FLOW_NUM': 'sum'
        })
        # 如果需要保留 STATION_NAME 字段，可以加上如下代码（假设所有记录的 STATION_NAME 都相同）
        grouped['STATION_NAME'] = station_name
        # 按列顺序排列
        grouped = grouped[['SQUAD_DATE', 'STATION_NAME', 'ENTRY_NUM', 'EXIT_NUM', 'CHANGE_NUM', 'PASSENGER_NUM', 'FLOW_NUM']]
        # pd.set_option('display.max_columns', None)  # 不要缩略显示列
        print(grouped)
        return grouped
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

QUERY_START_DATE = 20230101



def read_station_daily_flow_history(station_name: str) -> pd.DataFrame:
    """
    从StationFlowPredict数据库读取指定站点的客流历史数据，并增加天气因子（通过SQUAD_DATE关联）
    对于相同日期F_DATE并且相同F_LINENAME的数据进行合并（数值字段求和，其余字段取第一个）

    参数:
        station_name: 车站名（字符串）

    返回:
        站点客流数据的 DataFrame
    """
    try:
        conn = pymssql.connect(
            server='192.168.10.76',
            user='sa',
            password='Chency@123',
            database='StationFlowPredict',  # 修改数据库
            port='1433'
        )

        query = f"""
        SELECT 
            S.ID,
            S.SQUAD_DATE AS F_DATE,
            S.STATION_ID AS F_LINENO,
            S.STATION_NAME AS F_LINENAME,
            S.FLOW_NUM AS F_KLCOUNT,
            C.F_WEEK,
            C.F_DATEFEATURES,
            C.F_HOLIDAYTYPE,
            C.F_ISHOLIDAY,
            C.F_ISNONGLI,
            C.F_ISYANGLI,
            C.F_NEXTDAY,
            C.F_HOLIDAYDAYS,
            C.F_HOLIDAYTHDAY,
            C.IS_FIRST,
            W.F_TQQK AS WEATHER_TYPE
        FROM 
            [master].[dbo].[STATION_FLOW_HISTORY] AS S
        LEFT JOIN 
            master.dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
            ON S.SQUAD_DATE = C.F_DATE
        LEFT JOIN
            master.dbo.WeatherHistory AS W
            ON S.SQUAD_DATE = W.F_DATE
        WHERE 
            S.SQUAD_DATE >= '{QUERY_START_DATE}'
            AND S.STATION_NAME = N'{station_name}'
        ORDER BY 
            S.SQUAD_DATE, S.STATION_ID
        """

        df = pd.read_sql(query, conn)
        conn.close()

        # 合并相同F_DATE和F_LINENAME的数据（数值字段求和，其余字段取第一个）
        sum_fields = ['F_KLCOUNT']
        first_fields = [
            'F_LINENO', 'F_LINENAME', 'F_WEEK', 'F_DATEFEATURES', 'F_HOLIDAYTYPE',
            'F_ISHOLIDAY', 'F_ISNONGLI', 'F_ISYANGLI', 'F_NEXTDAY', 'F_HOLIDAYDAYS',
            'F_HOLIDAYTHDAY', 'IS_FIRST', 'WEATHER_TYPE'
        ]
        grouped = df.groupby(['F_DATE', 'F_LINENAME'], as_index=False).agg(
            {**{f: 'sum' for f in sum_fields},
             **{f: 'first' for f in first_fields}}
        )
        return grouped

    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")


if __name__ == "__main__":
    # df = read_station_flow_history("碧沙湖")  #五一广场、碧沙湖、橘子洲
    df = read_station_daily_flow_history("五一广场 ")
    print(df.head())
