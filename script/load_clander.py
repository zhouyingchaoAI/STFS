
import pandas as pd
import pymssql

DB_CONFIG = {
    "server": "10.1.6.230",
    "user": "sa",
    "password": "YourStrong!Passw0rd",
    "database": "master",
    "port": 1433
}

def import_calendar_csv_to_db(csv_path, table_name="[master].[dbo].[CalendarHistory]"):
    # 读取CSV文件
    df = pd.read_csv(csv_path, encoding='utf-8')
    # 连接数据库
    conn = pymssql.connect(
        server=DB_CONFIG["server"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        port=DB_CONFIG["port"]
    )
    cursor = conn.cursor()
    # 先清空表内容
    try:
        cursor.execute(f"TRUNCATE TABLE {table_name}")
        print(f"已清空表 {table_name} 的所有内容。")
    except Exception as e:
        print(f"清空表 {table_name} 失败: {e}")
        conn.rollback()
        cursor.close()
        conn.close()
        return
    # 获取字段名
    columns = df.columns.tolist()
    col_str = ", ".join(f"[{col}]" for col in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholders})"
    # 插入数据
    for row in df.itertuples(index=False, name=None):
        cursor.execute(insert_sql, row)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"成功导入 {len(df)} 条记录到 {table_name}")

if __name__ == "__main__":
    import_calendar_csv_to_db("holiday_data_20171211_20261231.csv")
    # import_calendar_csv_to_db("BasicInformation.csv")
