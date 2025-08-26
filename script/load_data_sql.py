import pymssql
import csv
import os
import sys
import time
import pandas as pd

# CSV文件路径及对应表名
csv_table_list = [
    # (r'线路客流预测结果样例_线路小时客流_20250609-0709.csv', 'LineHourlyFlowPrediction'),
    # (r'线路客流预测结果样例_线网线路日客流_20250609-0709.csv', 'LineDailyFlowPrediction'),
    # (r'线网线路历史日客流数据样例.csv', 'LineDailyFlowHistory'),
    # (r'线路历史小时客流数据样例.csv', 'LineHourlyFlowHistory'),
    # (r'20250714未来天气表样例.csv', 'WeatherFuture'),
    # (r'20250714历史天气表样例.csv', 'WeatherHistory'),
    (r'基础数据表样例-日历.xlsx', 'CalendarHistory'),
]

def infer_sql_type(values, col_name=None):
    """
    简单推断字段类型：优先级 char > int > float
    如果是ID字段，强制为VARCHAR(50)
    """
    # 兼容 "ID"、'ID'、ID 三种写法
    if col_name is not None and col_name.replace('"', '').replace("'", '').upper() == 'ID':
        # 允许足够长的字符串
        return 'VARCHAR(50)'
    
    # 过滤空值
    non_empty_values = [str(v).strip() for v in values if pd.notna(v) and str(v).strip() != '']
    if not non_empty_values:
        return 'VARCHAR(32)'
    
    is_int = True
    is_float = True
    max_len = 0
    
    for v in non_empty_values:
        max_len = max(max_len, len(str(v)))
        
        # 检查是否为整数
        try:
            int(v)
        except:
            is_int = False
        
        # 检查是否为浮点数
        try:
            float(v)
        except:
            is_float = False
    
    # 按照优先级 char > int > float 进行判断
    # 如果不是纯数字，则返回字符类型（最高优先级）
    if not is_int and not is_float:
        # 判断是否全为ASCII
        if all(all(ord(c) < 128 for c in str(v)) for v in non_empty_values):
            return f'VARCHAR({max(32, max_len)})'
        else:
            return f'NVARCHAR({max(32, max_len)})'
    
    # 如果是整数，返回INT（第二优先级）
    if is_int:
        return 'INT'
    if is_float:
        return 'FLOAT'

    
    # 兜底返回VARCHAR
    return 'VARCHAR(32)'

def get_columns_and_types_from_csv(csv_file):
    with open(csv_file, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        # 跳过BOM和多余的问号
        if header[0].startswith('\ufeff'):
            header[0] = header[0].replace('\ufeff', '')
        if header[0].startswith('???'):
            header[0] = header[0].replace('???', '')
        # 去掉所有字段名的引号
        header = [col.replace('"', '').replace("'", '') for col in header]
        # 采样前100行推断类型
        sample_rows = []
        for i, row in enumerate(reader):
            sample_rows.append(row)
            if i >= 99:
                break
        # 转置
        columns_data = list(zip(*sample_rows)) if sample_rows else [[] for _ in header]
        types = []
        for i, col in enumerate(header):
            col_values = columns_data[i] if i < len(columns_data) else []
            # 强制ID为字符串类型
            sql_type = infer_sql_type(col_values, col_name=col)
            types.append((col, sql_type))
        return header, types

def get_columns_and_types_from_excel(excel_file):
    try:
        # 尝试不同的编码方式读取Excel文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_excel(excel_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # 如果是其他错误，尝试不指定编码
                try:
                    df = pd.read_excel(excel_file)
                    break
                except:
                    continue
        
        if df is None:
            raise Exception("无法读取Excel文件，尝试了多种编码方式")
        
        header = df.columns.tolist()
        # 去掉所有字段名的引号
        header = [col.replace('"', '').replace("'", '') for col in header]
        
        types = []
        for col in header:
            col_values = df[col].tolist()
            sql_type = infer_sql_type(col_values, col_name=col)
            types.append((col, sql_type))
        
        return header, types, df
    except Exception as e:
        raise Exception(f"读取Excel文件失败: {e}")

def get_columns_and_types(file_path):
    """根据文件扩展名选择读取方法"""
    if file_path.lower().endswith('.xlsx') or file_path.lower().endswith('.xls'):
        return get_columns_and_types_from_excel(file_path)
    else:
        header, types = get_columns_and_types_from_csv(file_path)
        return header, types, None

def drop_table_if_exists(cursor, table_name):
    # 先判断表是否存在，存在则删除
    sql = f"IF OBJECT_ID(N'{table_name}', N'U') IS NOT NULL DROP TABLE [{table_name}];"
    cursor.execute(sql)

def create_table(cursor, columns_types, table_name):
    col_defs = []
    for col, typ in columns_types:
        if col.upper() == 'ID':
            # 强制ID为字符串类型
            typ = 'VARCHAR(50)'
            col_defs.append(f'[{col}] {typ} PRIMARY KEY')
        else:
            col_defs.append(f'[{col}] {typ}')
    col_defs_str = ', '.join(col_defs)
    sql = f'''
    CREATE TABLE [{table_name}] (
        {col_defs_str}
    )
    '''
    cursor.execute(sql)

def insert_data(cursor, header, rows, table_name, show_progress=False):
    placeholders = ','.join(['%s'] * len(header))
    col_names = ','.join([f'[{col}]' for col in header])
    sql = f'INSERT INTO [{table_name}] ({col_names}) VALUES ({placeholders})'
    total = len(rows)
    for idx, row in enumerate(rows, 1):
        cursor.execute(sql, row)
        if show_progress and (idx % 100 == 0 or idx == total):
            percent = int(idx / total * 100)
            sys.stdout.write(f"\r    已插入 {idx}/{total} 行 ({percent}%)")
            sys.stdout.flush()
    if show_progress:
        sys.stdout.write("\n")

def import_csv_to_sql(csv_file, table_name, cursor, conn):
    try:
        # 自动推断字段和类型
        header, columns_types, df = get_columns_and_types(csv_file)

        # 发现存在表则先删除
        drop_table_if_exists(cursor, table_name)

        # 建表，ID为主键
        create_table(cursor, columns_types, table_name)
        conn.commit()

        # 读取数据并插入
        if df is not None:
            # Excel文件，使用DataFrame
            rows = []
            for _, row in df.iterrows():
                row_data = []
                for col in header:
                    value = row[col]
                    if pd.isna(value):
                        row_data.append(None)
                    else:
                        row_data.append(str(value))
                rows.append(row_data)
        else:
            # CSV文件，使用原有逻辑
            with open(csv_file, encoding='utf-8-sig', newline='') as f:
                reader = csv.reader(f)
                csv_header = next(reader)
                # 跳过BOM和多余的问号
                if csv_header[0].startswith('\ufeff'):
                    csv_header[0] = csv_header[0].replace('\ufeff', '')
                if csv_header[0].startswith('???'):
                    csv_header[0] = csv_header[0].replace('???', '')
                # 去掉所有字段名的引号
                csv_header = [col.replace('"', '').replace("'", '') for col in csv_header]
                rows = []
                for row in reader:
                    # 只取前len(header)列，防止多余
                    rows.append(row[:len(header)])
        
        print(f"    正在插入 {len(rows)} 行数据 ...")
        insert_data(cursor, header, rows, table_name, show_progress=True)
        conn.commit()
        print(f"    数据已成功导入表 {table_name}。")
    except Exception as e:
        print(f"    导入 {csv_file} 到表 {table_name} 失败: {e}")

def main():
    try:
        # 使用 insert_calendar_history.py 中的数据库连接配置
        conn = pymssql.connect(
            server='192.168.10.76',
            user='sa',
            password='Chency@123',
            database='master',
            port=1433
        )
        cursor = conn.cursor()
        total_tables = len(csv_table_list)
        print(f"共需导入 {total_tables} 张表。")
        for idx, (csv_file, table_name) in enumerate(csv_table_list, 1):
            print(f"[{idx}/{total_tables}] 正在导入 {csv_file} 到表 {table_name} ...")
            import_csv_to_sql(csv_file, table_name, cursor, conn)
        conn.close()
        print("全部表已导入完成。")
    except Exception as e:
        print("连接失败或导入出错:", e)

if __name__ == "__main__":
    main()
