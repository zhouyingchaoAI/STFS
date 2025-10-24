import pandas as pd
import yaml
import os

# 读取配置文件
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

CONFIG = load_config()

# 数据库配置（优先使用配置文件，否则使用默认值）
DB_CONFIG = CONFIG.get('db', {
    "server": "10.1.6.230",
    "user": "sa",
    "password": "YourStrong!Passw0rd",
    "database": "master",
    "port": 1433
})

# 车站过滤配置
STATION_FILTER_NAMES = CONFIG.get('STATION_FILTER_NAMES', [])
QUERY_START_DATE = CONFIG.get('QUERY_START_DATE', '20230101')

def fix_chinese_encoding(text):
    """修复中文字符编码问题"""
    if not isinstance(text, str):
        return text
    
    try:
        # 检查是否包含乱码字符（检查常见的乱码模式）
        has_garbled_chars = False
        
        # 检查是否包含乱码字符
        for char in text:
            if ord(char) > 127:  # 非ASCII字符
                # 检查是否是常见的乱码字符
                if char in 'éÙ×ÓÖÞäåæçèéêëìíîïðñòóôõöøùúûüýþÿ':
                    has_garbled_chars = True
                    break
        
        if has_garbled_chars:
            # 尝试多种编码修复方法
            try:
                # 方法1: 从latin-1解码再重新编码为utf-8
                fixed = text.encode('latin-1').decode('utf-8')
                print(f"编码修复: '{text}' -> '{fixed}'")
                return fixed
            except (UnicodeDecodeError, UnicodeEncodeError):
                try:
                    # 方法2: 从cp1252解码再重新编码为utf-8
                    fixed = text.encode('cp1252').decode('utf-8')
                    print(f"编码修复: '{text}' -> '{fixed}'")
                    return fixed
                except (UnicodeDecodeError, UnicodeEncodeError):
                    try:
                        # 方法3: 从iso-8859-1解码再重新编码为utf-8
                        fixed = text.encode('iso-8859-1').decode('utf-8')
                        print(f"编码修复: '{text}' -> '{fixed}'")
                        return fixed
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        print(f"编码修复失败: '{text}'")
                        return text
        return text
    except Exception as e:
        print(f"编码修复异常: {e}")
        return text

def get_db_conn():
    import pymssql
    return pymssql.connect(
        server=DB_CONFIG["server"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        port=DB_CONFIG["port"]
    )

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

def read_station_daily_flow_history(metric_type: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从StationFlowPredict数据库读取指定站点的客流历史数据，并增加天气因子（通过SQUAD_DATE关联）
    对于相同日期F_DATE并且相同F_LINENAME的数据进行合并（数值字段求和，其余字段取第一个）

    参数:
        metric_type: 指标类型（F_PKLCOUNT, F_ENTRANCE, F_EXIT, F_TRANSFER, F_BOARD_ALIGHT）
        start_date: 开始日期（yyyymmdd格式）
        end_date: 结束日期（yyyymmdd格式）

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
        
        if station_filter:
            print(f"应用车站过滤：{STATION_FILTER_NAMES}")

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
            REPLACE(S.SQUAD_DATE, '-', '') >= '{start_date}'
            AND REPLACE(S.SQUAD_DATE, '-', '') <= '{end_date}'
            {station_filter}
        ORDER BY 
            REPLACE(S.SQUAD_DATE, '-', ''), S.STATION_ID
        """

        if station_filter:
            print(f"完整SQL:\n{query}")
        else:
            print(f"车站查询SQL (前200字符): {query[:200]}...")
        
        df = pd.read_sql(query, conn)

        conn.close()
        print(f"车站数据查询完成：{len(df)}行")

        if df.empty:
            print("⚠️ 车站查询结果为空")
            return df
        
        # 合并相同F_DATE和F_LINENAME的数据（数值字段求和，其余字段取第一个）
        print(f"开始合并车站数据，原始{len(df)}行...")
        sum_fields = ['F_KLCOUNT']
        first_fields = [
            'F_LINENO', 'F_LINENAME', 'F_WEEK', 'F_DATEFEATURES', 'F_HOLIDAYTYPE',
            'F_ISHOLIDAY', 'F_ISNONGLI', 'F_ISYANGLI', 'F_NEXTDAY', 'F_HOLIDAYDAYS',
            'F_HOLIDAYTHDAY', 'IS_FIRST', 'WEATHER_TYPE', 'F_YEAR', 'F_DAYOFWEEK', 'F_HOLIDAYWHICHDAY', 'COVID19', 'F_WEATHER'
        ]
        
        # 检查必需字段是否存在
        missing_fields = []
        for field in sum_fields + first_fields + ['F_DATE', 'F_LINENAME']:
            if field not in df.columns:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ 缺少字段: {missing_fields}")
            print(f"实际列: {df.columns.tolist()}")
            # 只用存在的字段
            sum_fields = [f for f in sum_fields if f in df.columns]
            first_fields = [f for f in first_fields if f in df.columns]
        
        grouped = df.groupby(['F_DATE', 'F_LINENAME'], as_index=False).agg(
            {**{f: 'sum' for f in sum_fields},
             **{f: 'first' for f in first_fields}}
        )
        print(f"✓ 合并完成：{len(grouped)}行")
        
        # 智能编码修复
        for col in grouped.columns:
            if grouped[col].dtype == 'object':  # 字符串列
                def smart_encode_fix(x):
                    if not isinstance(x, str):
                        return x
                    
                    # 检查是否包含乱码字符（latin-1编码的中文字符）
                    if any(ord(c) > 127 and ord(c) < 256 for c in x):
                        try:
                            # 尝试从latin-1重新编码
                            return x.encode('latin-1').decode('gbk')
                        except:
                            try:
                                # 尝试从latin-1重新编码为utf-8
                                return x.encode('latin-1').decode('utf-8')
                            except:
                                return x
                    return x
                
                grouped[col] = grouped[col].astype(str).apply(smart_encode_fix)
        
        return grouped

    except Exception as e:
        print(f"❌ 车站数据读取异常：{e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"数据库读取失败: {e}")

def read_line_daily_flow_history(metric_type: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    查询指定期间的日客流历史数据，并增加天气因子（通过F_DATE关联）

    参数:
        metric_type (str): 指标类型，如 "F_PKLCOUNT"
        start_date (str): 查询开始日期（格式：yyyymmdd）
        end_date (str): 查询结束日期（格式：yyyymmdd）

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

        query = f"""
        SELECT 
            L.ID,
            L.F_DATE,
            L.F_LB,
            L.F_LINENO,
            L.F_LINENAME,
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
            L.CREATOR = 'chency' 
            AND L.F_DATE >= '{start_date}'
            AND L.F_DATE <= '{end_date}'
        ORDER BY 
            L.F_DATE, L.F_LINENO
        """

        df = pd.read_sql(query, conn)
        conn.close()
        
        # 智能编码修复
        for col in df.columns:
            if df[col].dtype == 'object':  # 字符串列
                def smart_encode_fix(x):
                    if not isinstance(x, str):
                        return x
                    
                    # 检查是否包含乱码字符（latin-1编码的中文字符）
                    if any(ord(c) > 127 and ord(c) < 256 for c in x):
                        try:
                            # 尝试从latin-1重新编码
                            return x.encode('latin-1').decode('gbk')
                        except:
                            try:
                                # 尝试从latin-1重新编码为utf-8
                                return x.encode('latin-1').decode('utf-8')
                            except:
                                return x
                    return x
                
                df[col] = df[col].astype(str).apply(smart_encode_fix)
        
        return df
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="查询指定日期范围内的客流历史数据")
    parser.add_argument("--metric_type", type=str, required=True, choices=[
        "F_PKLCOUNT", "F_ENTRANCE", "F_EXIT", "F_TRANSFER", "F_BOARD_ALIGHT"
    ], help="指标类型")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期（yyyymmdd）")
    parser.add_argument("--end_date", type=str, required=True, help="结束日期（yyyymmdd）")
    parser.add_argument("--out_csv", type=str, default=None, help="输出CSV文件路径（可选）")
    args = parser.parse_args()

    df = read_line_daily_flow_history(args.metric_type, args.start_date, args.end_date)
    print(df)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False, encoding="utf_8_sig")
        print(f"数据已导出到 {args.out_csv}")

if __name__ == "__main__":
    main()