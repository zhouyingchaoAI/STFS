import pandas as pd
import pymssql
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据库连接配置，来自 db_config.yaml
DB_CONFIG = {
    "server": "10.1.6.230",
    "user": "sa",
    "password": "YourStrong!Passw0rd",
    "database": "master",
    "port": 1433
}

def clear_table(table_name: str):
    """
    清空指定表的所有内容
    """
    try:
        conn = pymssql.connect(
            server=DB_CONFIG["server"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            port=DB_CONFIG["port"]
        )
        cursor = conn.cursor()
        # 执行清空表操作
        cursor.execute(f"TRUNCATE TABLE master.dbo.{table_name}")
        conn.commit()
        logger.info(f"表 {table_name} 已清空。")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"清空表内容时出错: {e}")

def query_station_flow_predict(limit: int = 20):
    """
    查询STATION_FLOW_PREDICT表的内容，默认显示前limit行，按SQUAD_DATE升序排序

    参数:
        limit: 显示的最大行数
    """
    try:
        conn = pymssql.connect(
            server=DB_CONFIG["server"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            port=DB_CONFIG["port"]
        )
        
        # 字段列表，参考 STATION_FLOW_PREDICT 表结构
        columns = [
            "ID", "LINE_ID", "STATION_ID", "STATION_NAME", "SQUAD_DATE", "PREDICT_DATE",
            "CHANGE_NUM", "ENTRY_NUM", "EXIT_NUM", "PASSENGER_NUM", "FLOW_NUM"
        ]
        col_str = ", ".join(f"[{col}]" for col in columns)
        # 按SQUAD_DATE升序排序
        query = f"SELECT TOP ({limit}) {col_str} FROM master.dbo.STATION_FLOW_PREDICT ORDER BY [SQUAD_DATE] ASC"
        df = pd.read_sql(query, conn)
        print(f"表 STATION_FLOW_PREDICT 的前 {limit} 行内容（按SQUAD_DATE升序）：")
        conn.close()
    except Exception as e:
        logger.error(f"查询STATION_FLOW_PREDICT表内容时出错: {e}")

if __name__ == "__main__":
    table = "STATION_FLOW_PREDICT"
    # 先清空表
    # clear_table(table)
    # 再查看表内容
    query_station_flow_predict(limit=20)