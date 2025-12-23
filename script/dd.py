import pandas as pd
import pymssql
from datetime import datetime

# 数据库连接参数
DB_CONFIG = {
    "server": "10.1.6.230",
    "user": "sa",
    "password": "YourStrong!Passw0rd",
    "database": "master",
    "port": 1433
}

# 用pymssql连接数据库
conn = pymssql.connect(
    server=DB_CONFIG["server"],
    user=DB_CONFIG["user"],
    password=DB_CONFIG["password"],
    database=DB_CONFIG["database"],
    port=DB_CONFIG["port"]
)

# 查询2025年1月数据
select_sql = """
SELECT 
    [ID], [F_DATE], [F_WEEK], [F_DATEFEATURES], [F_HOLIDAYTYPE], [F_ISHOLIDAY], 
    [F_ISNONGLI], [F_ISYANGLI], [F_NEXTDAY], [F_HOLIDAYDAYS], [F_HOLIDAYTHDAY], 
    [F_ISSPACIAL], [CREATETIME], [CREATOR], [MODIFYTIME], [MODIFIER], [REMARKS], [IS_SUMMER], [IS_FIRST]
FROM [master].[dbo].[LSTM_COMMON_HOLIDAYFEATURE]
WHERE F_DATE >= 20250101 AND F_DATE <= 20250131
"""

df = pd.read_sql(select_sql, conn)

if df.empty:
    print("未找到2025年1月数据")
else:
    # 将日期字段变更为2026年同期
    def shift_to_2026(date_val):
        """
        传入20250101格式（int或str），返回20260101格式（int）
        """
        if pd.isnull(date_val):
            return date_val
        if isinstance(date_val, int):
            date_str = str(date_val)
        elif isinstance(date_val, str):
            date_str = date_val
        else:
            return date_val
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except Exception:
            return date_val
        try:
            # 尝试直接替换年份到2026
            shifted = date_obj.replace(year=2026)
        except ValueError:
            # 特殊场景处理，如闰年的2月29日
            if date_obj.month == 2 and date_obj.day == 29:
                shifted = date_obj.replace(year=2026, day=28)
            else:
                return date_val
        return int(shifted.strftime("%Y%m%d"))

    def update_datetime_to_2026(dt_val):
        if pd.isnull(dt_val):
            return dt_val
        if isinstance(dt_val, str):
            try:
                dt = pd.to_datetime(dt_val)
                dt = dt.replace(year=2026)
                return dt
            except Exception:
                return dt_val
        elif isinstance(dt_val, (datetime, pd.Timestamp)):
            try:
                return dt_val.replace(year=2026)
            except Exception:
                # 处理闰年2月29号等
                if hasattr(dt_val, "month") and dt_val.month == 2 and dt_val.day == 29:
                    return dt_val.replace(year=2026, day=28)
                return dt_val
        return dt_val

    # F_DATE 字段转为2026年同期
    df['F_DATE'] = df['F_DATE'].apply(shift_to_2026).astype('Int64')
    
    # 根据新的日期生成ID，格式：1748707201 + YYYYMMDD
    def generate_id(date_val):
        """根据日期生成ID"""
        if pd.isnull(date_val):
            return None
        if isinstance(date_val, int):
            date_str = str(date_val)
        elif isinstance(date_val, str):
            date_str = date_val
        else:
            return None
        # ID格式：1748707201 + YYYYMMDD
        return f"1748707201{date_str}"
    
    # 生成新的ID
    df['ID'] = df['F_DATE'].apply(generate_id)

    # CREATETIME与MODIFYTIME转到2026年同期
    df['CREATETIME'] = df['CREATETIME'].apply(update_datetime_to_2026)
    df['MODIFYTIME'] = df['MODIFYTIME'].apply(update_datetime_to_2026)

    # 准备插入数据
    insert_cols = [
        "ID", "F_DATE", "F_WEEK", "F_DATEFEATURES", "F_HOLIDAYTYPE", "F_ISHOLIDAY",
        "F_ISNONGLI", "F_ISYANGLI", "F_NEXTDAY", "F_HOLIDAYDAYS", "F_HOLIDAYTHDAY",
        "F_ISSPACIAL", "CREATETIME", "CREATOR", "MODIFYTIME", "MODIFIER", "REMARKS", "IS_SUMMER", "IS_FIRST"
    ]
    col_str = ", ".join(f"[{col}]" for col in insert_cols)
    placeholders = ", ".join(["%s"] * len(insert_cols))
    insert_sql = f"INSERT INTO [master].[dbo].[LSTM_COMMON_HOLIDAYFEATURE] ({col_str}) VALUES ({placeholders})"

    cursor = conn.cursor()
    count = 0
    
    # 将DataFrame转换为列表，并处理数据类型
    def convert_value(val):
        """将pandas值转换为Python原生类型"""
        # 处理NaN/None
        if pd.isnull(val):
            return None
        # 日期时间类型直接返回
        if isinstance(val, (pd.Timestamp, datetime)):
            return val
        # 尝试转换为Python原生类型
        try:
            # 如果是pandas标量类型，使用item()方法
            if hasattr(val, 'item'):
                item_val = val.item()
                if pd.isnull(item_val):
                    return None
                # 如果是整数，转换为int
                if isinstance(item_val, (int, float)) and not isinstance(item_val, bool):
                    return int(item_val) if item_val.is_integer() else item_val
                return item_val
            # 直接是Python类型
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return int(val) if isinstance(val, float) and val.is_integer() else val
            return val
        except (ValueError, TypeError, AttributeError):
            return None
    
    # 将DataFrame转换为列表
    for idx, row in df.iterrows():
        row_values = [convert_value(row[col]) for col in insert_cols]
        
        try:
            cursor.execute(insert_sql, tuple(row_values))
            count += 1
            if count % 10 == 0:
                print(f"已插入 {count} 条数据...")
        except Exception as e:
            print(f"插入第 {count + 1} 条数据时出错: {e}")
            print(f"问题数据 (前10个字段): {row_values[:10]}")
            print(f"数据类型: {[type(v).__name__ for v in row_values[:10]]}")
            import traceback
            traceback.print_exc()
            conn.rollback()
            raise
    
    conn.commit()
    print(f"成功插入 {count} 条2026年1月数据。")
    cursor.close()

conn.close()
