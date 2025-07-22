# 数据库操作模块：处理数据库连接、数据读取和预测结果存储
import pymssql
import pandas as pd
from typing import List, Dict
from datetime import datetime
import uuid

def read_line_hourly_flow_history() -> pd.DataFrame:
    """
    从数据库读取小时客流历史数据

    返回:
        小时客流数据的 DataFrame
    """
    try:
        conn = pymssql.connect(
            server='10.1.6.230',
            user='sa',
            password='YourStrong!Passw0rd',
            database='master',
            port='1433'
        )
        sql = "SELECT F_DATE, F_HOUR, F_KLCOUNT, F_LINENO, F_LINENAME FROM LineHourlyFlowHistory"
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

def read_line_daily_flow_history() -> pd.DataFrame:
    """
    从数据库读取日客流历史数据

    返回:
        日客流数据的 DataFrame
    """
    try:
        conn = pymssql.connect(
            server='10.1.6.230',
            user='sa',
            password='YourStrong!Passw0rd',
            database='master',
            port='1433'
        )
        sql = "SELECT F_DATE, F_KLCOUNT, F_LINENO, F_LINENAME FROM LineDailyFlowHistory"
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

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
            server='10.1.6.230',
            user='sa',
            password='YourStrong!Passw0rd',
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
            server='10.1.6.230',
            user='sa',
            password='YourStrong!Passw0rd',
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