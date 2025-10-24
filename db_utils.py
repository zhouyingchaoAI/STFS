# 数据库操作模块：处理数据库连接、数据读取和预测结果存储
from operator import truediv
import pymssql
import pandas as pd
from typing import List, Dict
from datetime import datetime
import uuid
import os
import yaml

# 配置文件路径
CONFIG_FILE = "db_config.yaml"

# 默认配置
DEFAULT_CONFIG = {
    "db": {
        "server": "192.168.10.76",
        "user": "sa",
        "password": "Chency@123",
        "database": "master",
        "port": 1433
    },
    "QUERY_START_DATE": 20230101,
    "STATION_FILTER_NAMES": ["五一广场", "碧沙湖", "橘子洲"]
}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        # 如果配置文件不存在，写入默认配置
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True)
        return DEFAULT_CONFIG
    else:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # 检查缺失项，补全
        changed = False
        for k, v in DEFAULT_CONFIG.items():
            if k not in config:
                config[k] = v
                changed = True
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if subk not in config[k]:
                        config[k][subk] = subv
                        changed = True
        if changed:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True)
        return config

_config = load_config()
_db_conf = _config["db"]
QUERY_START_DATE = _config.get("QUERY_START_DATE", 20230101)
STATION_FILTER_NAMES = _config.get("STATION_FILTER_NAMES", ["五一广场", "碧沙湖", "橘子洲"])

def get_db_conn():
    return pymssql.connect(
        server=_db_conf["server"],
        user=_db_conf["user"],
        password=_db_conf["password"],
        database=_db_conf["database"],
        port=_db_conf["port"]
    )

def load_stationid_stationname_to_lineid(yaml_path="stationid_stationname_to_lineid.yaml"):
    """
    读取 (station_id, station_name) -> [line_id, ...] 的映射表
    返回: dict，key为(station_id, station_name)元组，value为line_id列表
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        mapping_strkey = yaml.safe_load(f)
    # 还原key为元组
    mapping = {}
    for k, v in mapping_strkey.items():
        if "|" in k:
            station_id, station_name = k.split("|", 1)
            mapping[(station_id, station_name)] = v
    return mapping

def get_lineids_by_station(station, mapping=None, yaml_path="stationid_stationname_to_lineid.yaml"):
    """
    输入 station_id 或 station_name，返回对应的线路ID列表
    station: 可以是 station_id（字符串），也可以是 station_name（字符串）
    mapping: 可选，已加载的映射表（dict），否则自动加载
    返回: set(line_id)
    """
    if mapping is None:
        mapping = load_stationid_stationname_to_lineid(yaml_path)
    result = set()
    # 先尝试作为station_id查找
    for (station_id, station_name), line_ids in mapping.items():
        if station == station_id or station == station_name:
            result.update(line_ids)
    return list(result)

def _get_station_filter_sql(alias="S"):
    """
    根据配置文件中的STATION_FILTER_NAMES生成SQL过滤条件
    alias: 表别名
    返回: SQL字符串，如 "AND S.STATION_NAME IN (N'五一广场', N'碧沙湖', N'橘子洲')" 或 ""
    """
    if STATION_FILTER_NAMES and len(STATION_FILTER_NAMES) > 0:
        names = ", ".join([f"N'{name}'" for name in STATION_FILTER_NAMES])
        return f"AND {alias}.STATION_NAME IN ({names})"
    else:
        return ""

def _get_station_filter_where(alias="S"):
    """
    用于WHERE子句的过滤（不带AND），如 "S.STATION_NAME IN (...)" 或 ""
    """
    if STATION_FILTER_NAMES and len(STATION_FILTER_NAMES) > 0:
        names = ", ".join([f"N'{name}'" for name in STATION_FILTER_NAMES])
        return f"{alias}.STATION_NAME IN ({names})"
    else:
        return ""

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
        conn = get_db_conn()
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

        # 站点过滤条件，根据配置文件决定
        station_filter = _get_station_filter_sql(alias="S")

        query = f"""
        SELECT 
            S.ID,
            REPLACE(S.SQUAD_DATE, '-', '') AS F_DATE,
            S.STATION_ID AS F_LINENO,
            S.STATION_NAME AS F_LINENAME,
            {selected_field} AS F_KLCOUNT,
            C.F_DATEFEATURES,
            C.F_ISHOLIDAY,
            C.F_ISNONGLI,
            C.F_ISYANGLI,
            C.F_NEXTDAY,
            C.F_HOLIDAYTHDAY,
            C.IS_FIRST,
            CC.F_YEAR,
            CC.F_DAYOFWEEK,
            CC.F_WEEK,
            CC.F_HOLIDAYTYPE,
            CC.F_HOLIDAYDAYS,
            CC.F_HOLIDAYWHICHDAY,
            CC.COVID19,
            CC.F_WEATHER,
            W.F_TQQK AS WEATHER_TYPE
        FROM 
            [StationFlowPredict].[dbo].[STATION_FLOW_HISTORY] AS S
        LEFT JOIN 
            master.dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
            ON REPLACE(S.SQUAD_DATE, '-', '') = C.F_DATE
        LEFT JOIN 
            master.dbo.CalendarHistory AS CC
            ON REPLACE(S.SQUAD_DATE, '-', '') = CC.F_DATE
        LEFT JOIN
            master.dbo.WeatherHistory AS W
            ON REPLACE(S.SQUAD_DATE, '-', '') = W.F_DATE
        WHERE 
            REPLACE(S.SQUAD_DATE, '-', '') >= '{QUERY_START_DATE}'
            {station_filter}
        ORDER BY 
            REPLACE(S.SQUAD_DATE, '-', ''), S.STATION_ID
        """


        df = pd.read_sql(query, conn)
        conn.close()

        # 合并相同F_DATE和F_LINENAME的数据（数值字段求和，其余字段取第一个）
        sum_fields = ['F_KLCOUNT']
        first_fields = [
            'F_LINENO', 'F_LINENAME', 'F_WEEK', 'F_DATEFEATURES', 'F_HOLIDAYTYPE',
            'F_ISHOLIDAY', 'F_ISNONGLI', 'F_ISYANGLI', 'F_NEXTDAY', 'F_HOLIDAYDAYS',
            'F_HOLIDAYTHDAY', 'IS_FIRST', 'WEATHER_TYPE', 'F_YEAR', 'F_DAYOFWEEK', 'F_HOLIDAYWHICHDAY', 'COVID19', 'F_WEATHER'
        ]
        grouped = df.groupby(['F_DATE', 'F_LINENAME'], as_index=False).agg(
            {**{f: 'sum' for f in sum_fields},
             **{f: 'first' for f in first_fields}}
        )
        return grouped

    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

def read_line_hourly_flow_history(metric_type: str, query_start_date: str = None, days: int = None) -> pd.DataFrame:
    """
    从数据库读取小时客流历史数据，并增加节假日和天气因子（通过F_DATE关联）

    参数:
        query_start_date (str, optional): 查询的起始日期（格式为YYYYMMDD或YYYY-MM-DD），为None时不限制
        days (int, optional): 查询天数，为None时不限制

    返回:
        小时客流数据的 DataFrame
    """
    try:
        conn = get_db_conn()

        metric_field_mapping = {
            "F_PKLCOUNT": "H.F_KLCOUNT",
            "F_ENTRANCE": "H.ENTRY_NUM", 
            "F_EXIT": "H.EXIT_NUM",
            "F_TRANSFER": "H.CHANGE_NUM",
            "F_BOARD_ALIGHT": "H.FLOW_NUM"
        }

        # 检查metric_type是否有效
        if metric_type not in metric_field_mapping:
            raise ValueError(f"无效的metric_type: {metric_type}。有效值: {list(metric_field_mapping.keys())}")
        
        selected_field = metric_field_mapping[metric_type]

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
            {selected_field} AS F_KLCOUNT,
            H.F_LINENO,
            H.F_LINENAME,
            H.CREATETIME,
            H.CREATOR,
            C.F_DATEFEATURES,
            C.F_ISHOLIDAY,
            C.F_ISNONGLI,
            C.F_ISYANGLI,
            C.F_NEXTDAY,
            C.F_HOLIDAYTHDAY,
            C.IS_FIRST,
            CC.F_YEAR,
            CC.F_DAYOFWEEK,
            CC.F_WEEK,
            CC.F_HOLIDAYTYPE,
            CC.F_HOLIDAYDAYS,
            CC.F_HOLIDAYWHICHDAY,
            CC.COVID19,
            CC.F_WEATHER,
            W.F_TQQK AS WEATHER_TYPE
        FROM 
            dbo.LineHourlyFlowHistory AS H
        LEFT JOIN 
            dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
            ON H.F_DATE = C.F_DATE
        LEFT JOIN 
            dbo.CalendarHistory AS CC
            ON H.F_DATE = CC.F_DATE
        LEFT JOIN
            dbo.WeatherHistory AS W
            ON H.F_DATE = W.F_DATE
        {where_sql}
        ORDER BY 
            H.F_DATE, H.F_LINENO, H.F_HOUR
        """
        # print(sql)
        df = pd.read_sql(sql, conn)
        print(f"小时历史数据库读取完成，共有 {len(df)} 条数据")
        conn.close()
        return df
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

def read_station_hourly_flow_history(metric_type: str, query_start_date: str = None, days: int = None) -> pd.DataFrame:
    """
    从StationFlowPredict数据库读取指定站点的小时客流历史数据，并增加天气因子（通过SQUAD_DATE关联）
    对于相同日期F_DATE、F_HOUR并且相同F_LINENAME的数据进行合并（数值字段求和，其余字段取第一个）

    参数:
        metric_type (str): 指标类型（如"F_PKLCOUNT", "F_ENTRANCE", ...）
        query_start_date (str, optional): 查询的起始日期（格式为YYYYMMDD或YYYY-MM-DD），为None时不限制
        days (int, optional): 查询天数，为None时不限制

    返回:
        站点小时客流数据的 DataFrame
    """
    try:
        conn = get_db_conn()
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

        # 处理日期格式
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

        # 构建SQL查询条件
        where_conditions = []
        # 仅查询部分车站（如需全量请去掉此行）
        station_filter = _get_station_filter_where(alias="S")
        if station_filter:
            where_conditions.append(station_filter)
        # 增加全局起始日期限制
        where_conditions.append(f"S.SQUAD_DATE >= '{QUERY_START_DATE}'")

        # 处理query_start_date和days
        q_start = None
        if query_start_date is not None:
            q_start = format_date_int(query_start_date)
            if days is not None and days > 0:
                # 查询query_start_date前后days天
                dt = datetime.strptime(str(q_start), "%Y%m%d")
                dt_start = (dt - pd.Timedelta(days=days)).strftime("%Y%m%d")
                dt_end = (dt + pd.Timedelta(days=days)).strftime("%Y%m%d")
                where_conditions.append(f"S.SQUAD_DATE >= '{dt_start}' AND S.SQUAD_DATE <= '{dt_end}'")
            else:
                where_conditions.append(f"S.SQUAD_DATE = '{q_start}'")

        where_sql = ""
        if where_conditions:
            where_sql = "WHERE " + " AND ".join(where_conditions)

        query = f"""
        SELECT 
            S.ID,
            S.SQUAD_DATE AS F_DATE,
            S.TIME_SECTION_ID AS F_HOUR,
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
            [master].[dbo].[STATION_HOUR_HISTORY] AS S
        LEFT JOIN 
            master.dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
            ON S.SQUAD_DATE = C.F_DATE
        LEFT JOIN
            master.dbo.WeatherHistory AS W
            ON S.SQUAD_DATE = W.F_DATE
        {where_sql}
        ORDER BY 
            S.SQUAD_DATE, S.TIME_SECTION_ID, S.STATION_ID
        """

        df = pd.read_sql(query, conn)
        # 如果指定了days且没有查到数据，尝试查找历史最大日期前推days天的数据
        if days is not None and days > 0 and (df is None or len(df) == 0):
            # 查找历史最大日期
            max_date_query = "SELECT MAX(SQUAD_DATE) AS MAX_DATE FROM [master].[dbo].[STATION_HOUR_HISTORY]"
            max_date_df = pd.read_sql(max_date_query, conn)
            max_date_val = max_date_df['MAX_DATE'].iloc[0]
            if max_date_val is not None:
                # max_date_val 可能是int或str
                if isinstance(max_date_val, int):
                    max_date_str = str(max_date_val)
                else:
                    max_date_str = str(max_date_val)
                # 重新构建where条件
                dt_max = datetime.strptime(max_date_str, "%Y%m%d")
                dt_start = (dt_max - pd.Timedelta(days=days)).strftime("%Y%m%d")
                dt_end = max_date_str
                # 重新构建where_sql
                where_conditions2 = []
                if station_filter:
                    where_conditions2.append(station_filter)
                where_conditions2.append(f"S.SQUAD_DATE >= '{dt_start}' AND S.SQUAD_DATE <= '{dt_end}'")
                where_sql2 = "WHERE " + " AND ".join(where_conditions2)
                query2 = f"""
                SELECT 
                    S.ID,
                    REPLACE(S.SQUAD_DATE, '-', '') AS F_DATE,
                    S.TIME_SECTION_ID AS F_HOUR,
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
                    [StationFlowPredict].[dbo].[STATION_HOUR_HISTORY] AS S
                LEFT JOIN 
                    master.dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
                    ON REPLACE(S.SQUAD_DATE, '-', '') = C.F_DATE
                LEFT JOIN
                    master.dbo.WeatherHistory AS W
                    ON REPLACE(S.SQUAD_DATE, '-', '') = W.F_DATE
                {where_sql2}
                ORDER BY 
                    REPLACE(S.SQUAD_DATE, '-', ''), S.TIME_SECTION_ID, S.STATION_ID
                """
                df = pd.read_sql(query2, conn)
        conn.close()
        if 'F_HOUR' in df.columns:
            df['F_HOUR'] = pd.to_numeric(df['F_HOUR'], errors='coerce').astype('Int64')

        # 合并相同F_DATE、F_HOUR和F_LINENAME的数据（数值字段求和，其余字段取第一个）
        sum_fields = ['F_KLCOUNT']
        first_fields = [
            'F_LINENO', 'F_LINENAME', 'F_WEEK', 'F_DATEFEATURES', 'F_HOLIDAYTYPE',
            'F_ISHOLIDAY', 'F_ISNONGLI', 'F_ISYANGLI', 'F_NEXTDAY', 'F_HOLIDAYDAYS',
            'F_HOLIDAYTHDAY', 'IS_FIRST', 'WEATHER_TYPE'
        ]
        grouped = df.groupby(['F_DATE', 'F_HOUR', 'F_LINENAME'], as_index=False).agg(
            {**{f: 'sum' for f in sum_fields},
             **{f: 'first' for f in first_fields}}
        )
        return grouped

    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

        
def read_station_hourly_flow_history_old(metric_type: str, query_start_date: str = None, days: int = None) -> pd.DataFrame:
    """
    从StationFlowPredict数据库读取指定站点的小时客流历史数据，并增加天气因子（通过SQUAD_DATE关联）
    对于相同日期F_DATE、F_HOUR并且相同F_LINENAME的数据进行合并（数值字段求和，其余字段取第一个）

    参数:
        metric_type (str): 指标类型（如"F_PKLCOUNT", "F_ENTRANCE", ...）
        query_start_date (str, optional): 查询的起始日期（格式为YYYYMMDD或YYYY-MM-DD），为None时不限制
        days (int, optional): 查询天数，为None时不限制

    返回:
        站点小时客流数据的 DataFrame
    """
    try:
        conn = get_db_conn()
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

        # 处理日期格式
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

        # 构建SQL查询条件
        where_conditions = []
        # 仅查询部分车站（如需全量请去掉此行）
        station_filter = _get_station_filter_where(alias="S")
        if station_filter:
            where_conditions.append(station_filter)
        # 增加全局起始日期限制
        where_conditions.append(f"S.SQUAD_DATE >= '{QUERY_START_DATE}'")

        # 处理query_start_date和days
        if query_start_date is not None:
            q_start = format_date_int(query_start_date)
            if days is not None and days > 0:
                # 查询query_start_date前后days天
                dt = datetime.strptime(str(q_start), "%Y%m%d")
                dt_start = (dt - pd.Timedelta(days=days)).strftime("%Y%m%d")
                dt_end = (dt + pd.Timedelta(days=days)).strftime("%Y%m%d")
                where_conditions.append(f"S.SQUAD_DATE >= '{dt_start}' AND S.SQUAD_DATE <= '{dt_end}'")
            else:
                where_conditions.append(f"S.SQUAD_DATE = '{q_start}'")

        where_sql = ""
        if where_conditions:
            where_sql = "WHERE " + " AND ".join(where_conditions)

        query = f"""
        SELECT 
            S.ID,
            S.SQUAD_DATE AS F_DATE,
            S.TIME_SECTION_ID AS F_HOUR,
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
            [master].[dbo].[STATION_HOUR_HISTORY] AS S
        LEFT JOIN 
            master.dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
            ON S.SQUAD_DATE = C.F_DATE
        LEFT JOIN
            master.dbo.WeatherHistory AS W
            ON S.SQUAD_DATE = W.F_DATE
        {where_sql}
        ORDER BY 
            S.SQUAD_DATE, S.TIME_SECTION_ID, S.STATION_ID
        """

        df = pd.read_sql(query, conn)
        conn.close()
        if 'F_HOUR' in df.columns:
            df['F_HOUR'] = pd.to_numeric(df['F_HOUR'], errors='coerce').astype('Int64')

        # 合并相同F_DATE、F_HOUR和F_LINENAME的数据（数值字段求和，其余字段取第一个）
        sum_fields = ['F_KLCOUNT']
        first_fields = [
            'F_LINENO', 'F_LINENAME', 'F_WEEK', 'F_DATEFEATURES', 'F_HOLIDAYTYPE',
            'F_ISHOLIDAY', 'F_ISNONGLI', 'F_ISYANGLI', 'F_NEXTDAY', 'F_HOLIDAYDAYS',
            'F_HOLIDAYTHDAY', 'IS_FIRST', 'WEATHER_TYPE'
        ]
        grouped = df.groupby(['F_DATE', 'F_HOUR', 'F_LINENAME'], as_index=False).agg(
            {**{f: 'sum' for f in sum_fields},
             **{f: 'first' for f in first_fields}}
        )
        return grouped

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
        conn = get_db_conn()

        base_query = """
        
        SELECT 
            C.F_DATE,
            C.F_DATEFEATURES,
            C.F_ISHOLIDAY,
            C.F_ISNONGLI,
            C.F_ISYANGLI,
            C.F_NEXTDAY,
            C.F_HOLIDAYTHDAY,
            C.IS_FIRST,
            W.F_TQQK AS WEATHER_TYPE,
            CC.F_YEAR,
            CC.F_DAYOFWEEK,
            CC.F_WEEK, 
            CC.F_HOLIDAYTYPE,
            CC.F_HOLIDAYDAYS,
            CC.F_HOLIDAYWHICHDAY,
            CC.COVID19,
            CC.F_WEATHER 
        FROM 
            dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
        LEFT JOIN
            dbo.WeatherHistory AS W
            ON C.F_DATE = W.F_DATE
        LEFT JOIN
            dbo.CalendarHistory AS CC
            ON C.F_DATE = CC.F_DATE
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
        conn = get_db_conn()

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
            C.F_DATEFEATURES,
            C.F_ISHOLIDAY,
            C.F_ISNONGLI,
            C.F_ISYANGLI,
            C.F_NEXTDAY,
            C.F_HOLIDAYTHDAY,
            C.IS_FIRST,
            CC.F_YEAR,
            CC.F_DAYOFWEEK,
            CC.F_WEEK,
            CC.F_HOLIDAYTYPE,
            CC.F_HOLIDAYDAYS,
            CC.F_HOLIDAYWHICHDAY,
            CC.COVID19,
            CC.F_WEATHER,
            W.F_TQQK AS WEATHER_TYPE
        FROM 
            dbo.LineDailyFlowHistory AS L
        LEFT JOIN 
            dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
            ON L.F_DATE = C.F_DATE
        LEFT JOIN 
            dbo.CalendarHistory AS CC
            ON L.F_DATE = CC.F_DATE
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

def upload_xianwangxianlu_hourly_prediction_sample(prediction_rows: List[Dict], metric_type :str) -> None:
    """
    将小时预测结果插入数据库（更新模式，参考F_DATE、F_LINENO、F_HOUR）

    参数:
        prediction_rows: 预测结果字典列表
    """

    def to_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except:
            return None

    if not prediction_rows:
        return
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        for row in prediction_rows:

            # 根据metric_type动态选择要更新/插入的字段
            metric_field_map = {
                "F_PKLCOUNT": "F_PKLCOUNT",
                "F_ENTRANCE": "ENTRY_NUM",
                "F_EXIT": "EXIT_NUM",
                "F_TRANSFER": "CHANGE_NUM",
                "F_BOARD_ALIGHT": "FLOW_NUM"
            }
            # 默认字段
            metric_field = metric_field_map.get(metric_type, "F_PKLCOUNT")
            metric_value = to_float(row.get(metric_field, row.get('F_PKLCOUNT')))

            # 先尝试更新（根据map动态更新字段）
            update_sql = f"""
            UPDATE [CxFlowPredict].[dbo].[LineHourlyFlowPrediction] SET
                F_LINENAME=%s,
                {metric_field}=%s,
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
                metric_value,
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
                # 没有更新到，插入（根据map动态插入字段）
                insert_field_list = [
                    "ID", "F_DATE", "F_HOUR", "F_LINENO", "F_LINENAME",
                    metric_field, "F_BEFPKLCOUNT", "F_PRUTE", "CREATETIME",
                    "CREATOR", "REMARKS", "PREDICT_DATE", "PREDICT_WEATHER"
                ]
                insert_placeholder = ", ".join(["%s"] * len(insert_field_list))
                insert_sql = f"""
                INSERT INTO [CxFlowPredict].[dbo].[LineHourlyFlowPrediction] (
                    {', '.join(insert_field_list)}
                ) VALUES (
                    {insert_placeholder}
                )
                """
                insert_params = [
                    row.get('ID', str(uuid.uuid4())),
                    row.get('F_DATE'),
                    row.get('F_HOUR'),
                    row.get('F_LINENO'),
                    row.get('F_LINENAME'),
                    metric_value,
                    row.get('F_BEFPKLCOUNT'),
                    row.get('F_PRUTE'),
                    row.get('CREATETIME'),
                    row.get('CREATOR'),
                    row.get('REMARKS'),
                    row.get('PREDICT_DATE'),
                    row.get('PREDICT_WEATHER')
                ]
                cursor.execute(insert_sql, insert_params)

            
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"插入小时预测结果失败: {e}")

def upload_xianwangxianlu_daily_prediction_sample(prediction_rows: List[Dict], metric_type: str) -> None:
    """
    将日预测结果插入数据库（更新模式，参考F_DATE和F_LINENO）

    参数:
        prediction_rows: 预测结果字典列表
    """
    if not prediction_rows:
        return
    try:
        conn = get_db_conn()
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
            # 根据metric_type动态选择要更新/插入的字段
            metric_field_map = {
                "F_PKLCOUNT": "F_PKLCOUNT",
                "F_ENTRANCE": "ENTRY_NUM",
                "F_EXIT": "EXIT_NUM",
                "F_TRANSFER": "CHANGE_NUM",
                "F_BOARD_ALIGHT": "FLOW_NUM"
            }
            # 默认字段
            metric_field = metric_field_map.get(metric_type, "F_PKLCOUNT")
            metric_value = to_float(row.get(metric_field, row.get('F_PKLCOUNT')))

            # 构建UPDATE语句
            update_sql = f"""
            UPDATE [CxFlowPredict].[dbo].LineDailyFlowPrediction SET
                F_HOLIDAYTYPE=%s,
                F_LB=%s,
                F_LINENAME=%s,
                {metric_field}=%s,
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
                metric_value,
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
                insert_sql = f"""
                INSERT INTO [CxFlowPredict].[dbo].LineDailyFlowPrediction (
                    ID, F_DATE, F_HOLIDAYTYPE, F_LB, F_LINENO, F_LINENAME, {metric_field}, F_BEFPKLCOUNT, 
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
                    metric_value,
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



def upload_station_daily_prediction_sample(prediction_rows: List[Dict], metric_type: str) -> None:
    """
    将日预测结果插入数据库（更新模式，参考F_DATE和F_LINENO）

    参数:
        prediction_rows: 预测结果字典列表
    """

    if not prediction_rows:
        return
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        for row in prediction_rows:
            # id_val = row.get('ID', str(uuid.uuid4()))
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

            def to_str(val):
                if val is None:
                    return None
                try:
                    return str(val)
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
            # 根据metric_type动态选择要更新/插入的字段
            metric_field_map = {
                "F_PKLCOUNT": "PASSENGER_NUM",
                "F_ENTRANCE": "ENTRY_NUM",
                "F_EXIT": "EXIT_NUM",
                "F_TRANSFER": "CHANGE_NUM",
                "F_BOARD_ALIGHT": "FLOW_NUM"
            }
            # 默认字段
            metric_field = metric_field_map.get(metric_type, "F_PKLCOUNT")
            metric_value = to_float(row.get(metric_field, row.get('F_PKLCOUNT')))

            # 构建UPDATE语句
            update_sql = f"""
            UPDATE [CxFlowPredict].[dbo].STATION_FLOW_PREDICT SET
                LINE_ID=%s,
                STATION_ID=%s,
                STATION_NAME=%s,
                SQUAD_DATE=%s,
                PREDICT_DATE=%s,
                {metric_field}=%s
            WHERE LINE_ID=%s AND STATION_ID=%s AND SQUAD_DATE=%s
            """

            lineids = get_lineids_by_station(row.get('F_LINENAME'))
            for lid in lineids:
                id_val = str(uuid.uuid4())
                update_params = (
                    lid,
                    row.get('F_LINENO'),
                    row.get('F_LINENAME'),
                    to_str(row.get('F_DATE')),
                    to_str(row.get('PREDICT_DATE')),
                    to_int(metric_value),
                    lid,
                    row.get('F_LINENO'),
                    to_str(row.get('F_DATE')),
                )

                cursor.execute(update_sql, update_params)

                if cursor.rowcount == 0:
                    # 没有更新到，插入
                    insert_sql = f"""
                    INSERT INTO [CxFlowPredict].[dbo].STATION_FLOW_PREDICT (
                        ID, LINE_ID, STATION_ID, STATION_NAME, SQUAD_DATE, PREDICT_DATE, 
                        {metric_field}
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    insert_params = (
                        id_val,
                        lid,
                        row.get('F_LINENO'),
                        row.get('F_LINENAME'),
                        to_str(row.get('F_DATE')),
                        to_str(row.get('PREDICT_DATE')),
                        metric_value
                    )
                    cursor.execute(insert_sql, insert_params)
            
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"插入日预测结果失败: {e}")


def upload_station_hourly_prediction_sample(prediction_rows: List[Dict], metric_type: str) -> None:
    """
    将小时预测结果插入数据库（更新模式，参考F_DATE、F_LINENO、F_HOUR）

    参数:
        prediction_rows: 预测结果字典列表
    """

    if not prediction_rows:
        return

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

    def to_str(val):
        if val is None:
            return None
        try:
            return str(val)
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

    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        for row in prediction_rows:
            id_val = row.get('ID', str(uuid.uuid4()))
            # 根据metric_type动态选择要更新/插入的字段
            metric_field_map = {
                "F_PKLCOUNT": "PASSENGER_NUM",
                "F_ENTRANCE": "ENTRY_NUM",
                "F_EXIT": "EXIT_NUM",
                "F_TRANSFER": "CHANGE_NUM",
                "F_BOARD_ALIGHT": "FLOW_NUM"
            }
            # 默认字段
            metric_field = metric_field_map.get(metric_type, "PASSENGER_NUM")
            metric_value = to_float(row.get(metric_field, row.get('F_PKLCOUNT')))

            lineids = get_lineids_by_station(row.get('F_LINENAME'))
            for lid in lineids:
                id_val = str(uuid.uuid4())

                # 构建UPDATE语句
                update_sql = f"""
                UPDATE [CxFlowPredict].[dbo].STATION_HOUR_PREDICT SET
                    LINE_ID=%s,
                    STATION_ID=%s,
                    STATION_NAME=%s,
                    SQUAD_DATE=%s,
                    PREDICT_DATE=%s,
                    TIME_SECTION_ID=%s,
                    {metric_field}=%s
                WHERE LINE_ID=%s AND STATION_ID=%s AND SQUAD_DATE=%s AND TIME_SECTION_ID=%s
                """
                update_params = (
                    lid,
                    row.get('F_LINENO'),
                    row.get('F_LINENAME'),
                    to_str(row.get('F_DATE')),
                    to_str(row.get('PREDICT_DATE')),
                    to_str(row.get('F_HOUR')),
                    to_int(metric_value),
                    lid,
                    row.get('F_LINENO'),
                    to_str(row.get('F_DATE')),
                    to_str(f"{int(row.get('F_HOUR')):02d}") if row.get('F_HOUR') is not None else None,
                )

                cursor.execute(update_sql, update_params)

                if cursor.rowcount == 0:
                    
                    # 没有更新到，插入
                    insert_sql = f"""
                    INSERT INTO [CxFlowPredict].[dbo].STATION_HOUR_PREDICT (
                        ID, LINE_ID, STATION_ID, STATION_NAME, SQUAD_DATE, PREDICT_DATE, TIME_SECTION_ID, 
                        {metric_field}
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    insert_params = (
                        id_val,
                        lid,
                        row.get('F_LINENO'),
                        row.get('F_LINENAME'),
                        to_str(row.get('F_DATE')),
                        to_str(row.get('PREDICT_DATE')),
                        to_str(f"{int(row.get('F_HOUR')):02d}") if row.get('F_HOUR') is not None else None,
                        metric_value
                    )
                
                    cursor.execute(insert_sql, insert_params)

        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"插入小时预测结果失败: {e}")

