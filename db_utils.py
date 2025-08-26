# 数据库操作模块：处理数据库连接、数据读取和预测结果存储
import pymssql
import pandas as pd
from typing import List, Dict
from datetime import datetime
import uuid

# 全局配置：所有查询数据的起始日期
QUERY_START_DATE = 20230101

def read_station_daily_flow_history(metric_type: str) -> pd.DataFrame:
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
        metric_field_mapping = {
            "F_PKLCOUNT": "S.PASSENGER_NUM",
            "F_ENTRANCE": "S.ENTRY_NUM", 
            "F_EXIT": "S.EXIT_NUM",
            "F_TRANSFER": "S.CHANGE_NUM",
            "F_BOARD_ALIGHT": "S.FLOW_NUM"
        }

        # 检查metric_type是否有效
        if metric_type not in metric_field_mapping:
            raise ValueError(f"无效的metric_type: {metric_type}。有效值: {list(metric_field_mapping.keys())}")
        
        selected_field = metric_field_mapping[metric_type]

        query = f"""
        SELECT 
            S.ID,
            S.SQUAD_DATE AS F_DATE,
            S.STATION_ID AS F_LINENO,
            S.STATION_NAME AS F_LINENAME,
            {selected_field} AS F_KLCOUNT,
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
            AND S.STATION_NAME IN (N'五一广场', N'碧沙湖', N'橘子洲')
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

def read_line_hourly_flow_history(query_start_date: str = None, days: int = None) -> pd.DataFrame:
    """
    从数据库读取小时客流历史数据，并增加节假日和天气因子（通过F_DATE关联）

    参数:
        query_start_date (str, optional): 查询的起始日期（格式为YYYYMMDD或YYYY-MM-DD），为None时不限制
        days (int, optional): 查询天数，为None时不限制

    返回:
        小时客流数据的 DataFrame
    """
    try:
        conn = pymssql.connect(
            server='192.168.10.76',
            user='sa',
            password='Chency@123',
            database='master',
            port='1433'
        )

        def format_date_int(date_str):
            if date_str is None:
                return None
            if '-' in date_str:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            elif '/' in date_str:
                parts = date_str.split('/')
                if len(parts) == 3:
                    dt = datetime(year=int(parts[0]), month=int(parts[1]), day=int(parts[2]))
                else:
                    return None
            elif len(date_str) == 8 and date_str.isdigit():
                dt = datetime.strptime(date_str, "%Y%m%d")
            else:
                return None
            return int(dt.strftime("%Y%m%d"))

        def format_date_int_last_year(date_str):
            if date_str is None:
                return None
            if '-' in date_str:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            elif '/' in date_str:
                parts = date_str.split('/')
                if len(parts) == 3:
                    dt = datetime(year=int(parts[0]), month=int(parts[1]), day=int(parts[2]))
                else:
                    return None
            elif len(date_str) == 8 and date_str.isdigit():
                dt = datetime.strptime(date_str, "%Y%m%d")
            else:
                return None
            last_year_dt = dt.replace(year=dt.year - 1)
            return int(last_year_dt.strftime("%Y%m%d"))

        # 构建SQL查询条件
        where_clauses = ["H.CREATOR = 'chency'"]

        if query_start_date is None:
            # 查询所有
            pass
        else:
            q_start = format_date_int(query_start_date)
            q_last_year = format_date_int_last_year(query_start_date)
            date_conditions = []

            if days is not None and days > 0:
                # 查询query_start_date前后days天
                # 例如days=2, 则查[date-2, date+2]共5天
                from pandas import Timedelta
                dt = datetime.strptime(str(q_start), "%Y%m%d")
                dt_start = (dt - pd.Timedelta(days=days)).strftime("%Y%m%d")
                dt_end = (dt + pd.Timedelta(days=days)).strftime("%Y%m%d")
                # F_DATE between dt_start and dt_end (inclusive)
                date_conditions.append(f"(H.F_DATE >= {dt_start} AND H.F_DATE <= {dt_end})")
            else:
                # 只查当天
                date_conditions.append(f"(H.F_DATE = {q_start})")

            # 查去年同一天
            if q_last_year is not None:
                date_conditions.append(f"(H.F_DATE = {q_last_year})")

            where_clauses.append(f"({' OR '.join(date_conditions)})")

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
        SELECT 
            H.ID,
            H.F_DATE,
            H.F_HOUR,
            H.F_KLCOUNT,
            H.F_LINENO,
            H.F_LINENAME,
            H.CREATETIME,
            H.CREATOR,
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
            dbo.LineHourlyFlowHistory AS H
        LEFT JOIN 
            dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
            ON H.F_DATE = C.F_DATE
        LEFT JOIN
            dbo.WeatherHistory AS W
            ON H.F_DATE = W.F_DATE
        {where_sql}
        ORDER BY 
            H.F_DATE, H.F_LINENO, H.F_HOUR
        """
        df = pd.read_sql(sql, conn)
        print(f"小时历史数据库读取完成，共有 {len(df)} 条数据")
        conn.close()
        return df
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")


def fetch_holiday_features(predict_start_date: str = None, days: int = None) -> pd.DataFrame:
    """
    从数据库读取节假日特征数据（仅dbo.LSTM_COMMON_HOLIDAYFEATURE表）

    参数:
        predict_start_date (str, optional): 预测开始日期，格式为 'YYYYMMDD' 或 'YYYY-MM-DD'
        days (int, optional): 从预测开始日期往后获取的天数

    返回:
        节假日特征数据的 DataFrame
    """
    try:
        conn = pymssql.connect(
            server='192.168.10.76',
            user='sa',
            password='Chency@123',
            database='master',
            port='1433'
        )

        base_query = """
        
        SELECT 
            C.F_DATE,
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
            dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
        LEFT JOIN
            dbo.WeatherHistory AS W
            ON C.F_DATE = W.F_DATE
        """

        where_conditions = []

        # 增加全局起始日期限制
        where_conditions.append(f"C.F_DATE >= {QUERY_START_DATE}")

        # 处理predict_start_date格式为'YYYYMMDD'或'YYYY-MM-DD'，但F_DATE为int型(yyyymmdd)
        def format_date_int(date_str):
            if date_str is None:
                return None
            if '-' in date_str:
                # '2025-07-26' -> 20250726
                return int(date_str.replace('-', ''))
            if len(date_str) == 8 and date_str.isdigit():
                return int(date_str)
            # 允许'2025/07/26'
            if '/' in date_str:
                parts = date_str.split('/')
                if len(parts) == 3:
                    return int(parts[0] + parts[1].zfill(2) + parts[2].zfill(2))
            return None

        start_date_int = format_date_int(predict_start_date)

        if start_date_int is not None:
            if days is not None:
                # 计算结束日期
                from datetime import datetime, timedelta
                start_dt = datetime.strptime(str(start_date_int), "%Y%m%d")
                end_dt = start_dt + timedelta(days=days)
                end_date_int = int(end_dt.strftime("%Y%m%d"))
                where_conditions.append(f"C.F_DATE >= {start_date_int}")
                where_conditions.append(f"C.F_DATE < {end_date_int}")
            else:
                where_conditions.append(f"C.F_DATE >= {start_date_int}")
        elif days is not None:
            # 只给了天数，从今天开始
            from datetime import datetime, timedelta
            today = datetime.now()
            start_date_int = int(today.strftime("%Y%m%d"))
            end_date_int = int((today + timedelta(days=days)).strftime("%Y%m%d"))
            where_conditions.append(f"C.F_DATE >= {start_date_int}")
            where_conditions.append(f"C.F_DATE < {end_date_int}")

        if where_conditions:
            query = base_query + " WHERE " + " AND ".join(where_conditions)
        else:
            query = base_query

        query += " ORDER BY C.F_DATE"

        try:
            df = pd.read_sql(query, conn)
            # print("节假日特征数据：")
            # print(df)
        except Exception as e:
            # 处理数据库连接异常
            raise RuntimeError(f"数据库读取失败: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return df

    except Exception as e:
        # 处理数据库连接失败等异常
        raise RuntimeError(f"数据库连接失败: {e}")


def read_line_daily_flow_history(metric_type: str) -> pd.DataFrame:
    """
    从数据库读取日客流历史数据，并增加天气因子（通过F_DATE关联）

    返回:
        日客流数据的 DataFrame
    """
    try:
        conn = pymssql.connect(
            server='192.168.10.76',
            user='sa',
            password='Chency@123',
            database='master',
            port='1433'
        )

        metric_field_mapping = {
            "F_PKLCOUNT": "L.F_KLCOUNT",
            "F_ENTRANCE": "L.ENTRY_NUM", 
            "F_EXIT": "L.EXIT_NUM",
            "F_TRANSFER": "L.CHANGE_NUM",
            "F_BOARD_ALIGHT": "L.FLOW_NUM"
        }

        # 检查metric_type是否有效
        if metric_type not in metric_field_mapping:
            raise ValueError(f"无效的metric_type: {metric_type}。有效值: {list(metric_field_mapping.keys())}")
        
        selected_field = metric_field_mapping[metric_type]


        # 增加起始日期限制
        query = f"""
        SELECT 
            L.ID,
            L.F_DATE,
            L.F_LB,
            L.F_LINENO,
            L.F_LINENAME,
            {selected_field} AS F_KLCOUNT,
            L.CREATETIME,
            L.CREATOR,
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
            dbo.LineDailyFlowHistory AS L
        LEFT JOIN 
            dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
            ON L.F_DATE = C.F_DATE
        LEFT JOIN
            dbo.WeatherHistory AS W
            ON L.F_DATE = W.F_DATE
        WHERE 
            L.CREATOR = 'chency' AND L.F_DATE >= {QUERY_START_DATE}
        ORDER BY 
            L.F_DATE, L.F_LINENO
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

# def read_line_daily_flow_history(metric_type: str) -> pd.DataFrame:
#     """
#     从数据库读取日客流历史数据，并增加天气因子（通过F_DATE关联）

#     返回:
#         日客流数据的 DataFrame
#     """
#     try:
#         conn = pymssql.connect(
#             server='192.168.10.76',
#             user='sa',
#             password='Chency@123',
#             database='master',
#             port='1433'
#         )
#         # 增加起始日期限制
#         query = f"""
#         SELECT 
#             L.ID,
#             L.F_DATE,
#             L.F_LB,
#             L.F_LINENO,
#             L.F_LINENAME,
#             L.F_KLCOUNT,
#             L.CREATETIME,
#             L.CREATOR,
#             C.F_WEEK,
#             C.F_DATEFEATURES,
#             C.F_HOLIDAYTYPE,
#             C.F_ISHOLIDAY,
#             C.F_ISNONGLI,
#             C.F_ISYANGLI,
#             C.F_NEXTDAY,
#             C.F_HOLIDAYDAYS,
#             C.F_HOLIDAYTHDAY,
#             C.IS_FIRST,
#             W.F_TQQK AS WEATHER_TYPE
#         FROM 
#             dbo.LineDailyFlowHistory AS L
#         LEFT JOIN 
#             dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
#             ON L.F_DATE = C.F_DATE
#         LEFT JOIN
#             dbo.WeatherHistory AS W
#             ON L.F_DATE = W.F_DATE
#         WHERE 
#             L.CREATOR = 'chency' AND L.F_DATE >= {QUERY_START_DATE}
#         ORDER BY 
#             L.F_DATE, L.F_LINENO
#         """
#         df = pd.read_sql(query, conn)
#         conn.close()
#         return df
#     except Exception as e:
#         raise RuntimeError(f"数据库读取失败: {e}")


def insert_hourly_prediction_to_db(prediction_rows: List[Dict]) -> None:
    """
    将小时预测结果插入数据库（更新模式，参考F_DATE、F_LINENO、F_HOUR）

    参数:
        prediction_rows: 预测结果字典列表
    """
    if not prediction_rows:
        return
    try:
        conn = pymssql.connect(
            server='192.168.10.76',
            user='sa',
            password='Chency@123',
            database='master',
            port='1433'
        )
        cursor = conn.cursor()
        for row in prediction_rows:
            # 先尝试更新（增加F_HOUR条件）
            update_sql = """
            UPDATE LineHourlyFlowPrediction SET
                F_LINENAME=%s,
                F_PKLCOUNT=%s,
                F_BEFPKLCOUNT=%s,
                F_PRUTE=%s,
                CREATETIME=%s,
                CREATOR=%s,
                REMARKS=%s,
                PREDICT_DATE=%s,
                PREDICT_WEATHER=%s
            WHERE F_DATE=%s AND F_LINENO=%s AND F_HOUR=%s
            """
            update_params = (
                row.get('F_LINENAME'),
                row.get('F_PKLCOUNT'),
                row.get('F_BEFPKLCOUNT'),
                row.get('F_PRUTE'),
                row.get('CREATETIME'),
                row.get('CREATOR'),
                row.get('REMARKS'),
                row.get('PREDICT_DATE'),
                row.get('PREDICT_WEATHER'),
                row.get('F_DATE'),
                row.get('F_LINENO'),
                row.get('F_HOUR')
            )
            cursor.execute(update_sql, update_params)
            if cursor.rowcount == 0:
                # 没有更新到，插入
                insert_sql = """
                INSERT INTO LineHourlyFlowPrediction (
                    ID, F_DATE, F_HOUR, F_LINENO, F_LINENAME, F_PKLCOUNT, F_BEFPKLCOUNT, F_PRUTE, 
                    CREATETIME, CREATOR, REMARKS, PREDICT_DATE, PREDICT_WEATHER
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                cursor.execute(insert_sql, (
                    row.get('ID', str(uuid.uuid4())),
                    row.get('F_DATE'),
                    row.get('F_HOUR'),
                    row.get('F_LINENO'),
                    row.get('F_LINENAME'),
                    row.get('F_PKLCOUNT'),
                    row.get('F_BEFPKLCOUNT'),
                    row.get('F_PRUTE'),
                    row.get('CREATETIME'),
                    row.get('CREATOR'),
                    row.get('REMARKS'),
                    row.get('PREDICT_DATE'),
                    row.get('PREDICT_WEATHER')
                ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"插入小时预测结果失败: {e}")

def upload_prediction_sample(prediction_rows: List[Dict]) -> None:
    """
    将日预测结果插入数据库（更新模式，参考F_DATE和F_LINENO）

    参数:
        prediction_rows: 预测结果字典列表
    """
    if not prediction_rows:
        return
    try:
        conn = pymssql.connect(
            server='192.168.10.76',
            user='sa',
            password='Chency@123',
            database='master',
            port='1433'
        )
        cursor = conn.cursor()
        for row in prediction_rows:
            id_val = row.get('ID', str(uuid.uuid4()))
            def to_int(val):
                if val is None:
                    return None
                try:
                    return int(val)
                except:
                    return None
            def to_float(val):
                if val is None:
                    return None
                try:
                    return float(val)
                except:
                    return None
            def to_createtime_int(val):
                if val is None:
                    return None
                if isinstance(val, int):
                    return val
                try:
                    dt = datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                    return int(dt.strftime('%Y%m%d'))
                except:
                    try:
                        return int(val)
                    except:
                        return None
            def to_remarks_int(val):
                if val is None:
                    return None
                try:
                    return int(val)
                except:
                    return abs(hash(str(val))) % (10 ** 8)
            def to_ftype_int(val):
                if val is None:
                    return None
                try:
                    return int(val)
                except:
                    return abs(hash(str(val))) % (10 ** 8)
            # 先尝试更新
            update_sql = """
            UPDATE LineDailyFlowPrediction SET
                F_HOLIDAYTYPE=%s,
                F_LB=%s,
                F_LINENAME=%s,
                F_PKLCOUNT=%s,
                F_BEFPKLCOUNT=%s,
                F_PRUTE=%s,
                F_OTHER=%s,
                F_WH=%s,
                F_ZDHD=%s,
                F_YQZH=%s,
                F_YXYSMS=%s,
                CREATETIME=%s,
                CREATOR=%s,
                MODIFYTIME=%s,
                MODIFIER=%s,
                REMARKS=%s,
                PREDICT_DATE=%s,
                PREDICT_WEATHER=%s,
                F_TYPE=%s
            WHERE F_DATE=%s AND F_LINENO=%s
            """
            update_params = (
                to_int(row.get('F_HOLIDAYTYPE')),
                to_int(row.get('F_LB')),
                row.get('F_LINENAME'),
                to_float(row.get('F_PKLCOUNT')),
                to_float(row.get('F_BEFPKLCOUNT')),
                to_float(row.get('F_PRUTE')),
                to_int(row.get('F_OTHER')),
                to_int(row.get('F_WH')),
                to_int(row.get('F_ZDHD')),
                to_int(row.get('F_YQZH')),
                to_int(row.get('F_YXYSMS')),
                to_createtime_int(row.get('CREATETIME')),
                row.get('CREATOR'),
                to_createtime_int(row.get('MODIFYTIME')),
                to_int(row.get('MODIFIER')),
                to_remarks_int(row.get('REMARKS')),
                to_int(row.get('PREDICT_DATE')),
                to_int(row.get('PREDICT_WEATHER')),
                to_ftype_int(row.get('F_TYPE')),
                to_int(row.get('F_DATE')),
                to_int(row.get('F_LINENO'))
            )
            cursor.execute(update_sql, update_params)
            if cursor.rowcount == 0:
                # 没有更新到，插入
                insert_sql = """
                INSERT INTO LineDailyFlowPrediction (
                    ID, F_DATE, F_HOLIDAYTYPE, F_LB, F_LINENO, F_LINENAME, F_PKLCOUNT, F_BEFPKLCOUNT, 
                    F_PRUTE, F_OTHER, F_WH, F_ZDHD, F_YQZH, F_YXYSMS, CREATETIME, CREATOR, 
                    MODIFYTIME, MODIFIER, REMARKS, PREDICT_DATE, PREDICT_WEATHER, F_TYPE
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                insert_params = (
                    id_val,
                    to_int(row.get('F_DATE')),
                    to_int(row.get('F_HOLIDAYTYPE')),
                    to_int(row.get('F_LB')),
                    to_int(row.get('F_LINENO')),
                    row.get('F_LINENAME'),
                    to_float(row.get('F_PKLCOUNT')),
                    to_float(row.get('F_BEFPKLCOUNT')),
                    to_float(row.get('F_PRUTE')),
                    to_int(row.get('F_OTHER')),
                    to_int(row.get('F_WH')),
                    to_int(row.get('F_ZDHD')),
                    to_int(row.get('F_YQZH')),
                    to_int(row.get('F_YXYSMS')),
                    to_createtime_int(row.get('CREATETIME')),
                    row.get('CREATOR'),
                    to_createtime_int(row.get('MODIFYTIME')),
                    to_int(row.get('MODIFIER')),
                    to_remarks_int(row.get('REMARKS')),
                    to_int(row.get('PREDICT_DATE')),
                    to_int(row.get('PREDICT_WEATHER')),
                    to_ftype_int(row.get('F_TYPE'))
                )
                cursor.execute(insert_sql, insert_params)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"插入日预测结果失败: {e}")
