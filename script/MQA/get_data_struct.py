import pymssql
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_CONFIGS = [
    {
        "server": "10.1.6.230",
        "user": "sa",
        "password": "YourStrong!Passw0rd",
        "database": "master",
        "port": 1433,
        "label": "master",
    },
    {
        "server": "10.1.6.230",
        "user": "sa",
        "password": "YourStrong!Passw0rd",
        "database": "CxFlowPredict",
        "port": 1433,
        "label": "CxFlowPredict",
    }
]


def get_db_connection(db_config):
    """
    获取数据库连接（参考 db_utils.py）
    不指定 charset，让 pymssql 使用默认设置，编码问题在读取数据时通过 smart_encode_fix 处理
    """
    return pymssql.connect(
        server=db_config["server"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["database"],
        port=db_config["port"]
    )

def smart_encode_fix(x):
    """
    智能编码修复函数（参考 db_utils.py）
    处理乱码字符（latin-1编码的中文字符），尝试修复为正确的中文
    """
    if x is None:
        return ''
    
    if isinstance(x, bytes):
        # 如果是 bytes，先尝试解码
        try:
            x = x.decode('utf-8')
        except:
            try:
                x = x.decode('gbk')
            except:
                return x.decode('utf-8', errors='replace')
    
    if not isinstance(x, str):
        return str(x) if x is not None else ''
    
    # 检查是否包含乱码字符（latin-1编码的中文字符）
    # 乱码特征：字符的 ord 值在 127-255 之间，但不是有效的中文字符
    if any(ord(c) > 127 and ord(c) < 256 for c in x):
        try:
            # 尝试从latin-1重新编码为gbk（SQL Server中文数据通常是GBK）
            fixed = x.encode('latin-1').decode('gbk')
            # 验证修复结果是否包含中文
            if has_chinese(fixed):
                return fixed
        except:
            try:
                # 尝试从latin-1重新编码为utf-8
                fixed = x.encode('latin-1').decode('utf-8')
                if has_chinese(fixed):
                    return fixed
            except:
                pass
    
    return x

def try_decode(val):
    """
    Helper to decode bytes or byte-string style string to unicode.
    参考 db_utils.py 的编码处理方式，使用 smart_encode_fix 处理编码问题
    """
    if val is None:
        return ''
    
    if isinstance(val, bytes):
        # 尝试多种编码
        for enc in ['utf-8', 'gbk', 'gb2312', 'cp936']:
            try:
                decoded = val.decode(enc)
                # 使用 smart_encode_fix 进一步处理
                return smart_encode_fix(decoded)
            except (UnicodeDecodeError, LookupError):
                continue
        # 如果都失败，使用 errors='replace'
        try:
            return val.decode('utf-8', errors='replace')
        except:
            return val.decode('gbk', errors='replace')
    
    if isinstance(val, str):
        # 使用 smart_encode_fix 处理字符串
        return smart_encode_fix(val)
    
    return str(val) if val is not None else ''

def has_chinese(s):
    """Detects whether a string contains Chinese characters."""
    if not isinstance(s, str):
        return False
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)

def format_value_for_sql(val):
    """
    对于含中文的字符串输出 N'xxx' 形式，否则，数字直接输出，None 为 '', 其他字符串正常输出
    """
    if val is None:
        return ""
    val = try_decode(val)
    if isinstance(val, str):
        # 再做一次去除 \r\n\t，这些对于文本导出无意义
        val = val.replace('\r', '').replace('\n', '').replace('\t', ' ')
        if has_chinese(val):
            # 把单引号转义，避免生成的 N'xx' 里出错
            val_escaped = val.replace("'", "''")
            return f"N'{val_escaped}'"
        else:
            return val
    return str(val)

def fetch_table_structures_for_db(db_config):
    """
    获取指定数据库所有数据表结构的详细信息（包括字段备注）
    """
    conn = get_db_connection(db_config)
    # 设置连接的字符集处理
    cursor = conn.cursor(as_dict=True)

    cursor.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
    """)
    tables = cursor.fetchall()
    all_structures = {}

    for table in tables:
        schema = table['TABLE_SCHEMA']
        table_name = table['TABLE_NAME']
        full_table_name = f"{schema}.{table_name}"

        # 使用参数化查询，但需要正确处理
        try:
            cursor.execute("""
            SELECT
                c.COLUMN_NAME,
                c.DATA_TYPE,
                c.CHARACTER_MAXIMUM_LENGTH,
                c.IS_NULLABLE,
                c.COLUMN_DEFAULT,
                ep.value AS COLUMN_COMMENT
            FROM INFORMATION_SCHEMA.COLUMNS c
            LEFT JOIN sys.columns sc
                ON sc.object_id = OBJECT_ID(%s) AND sc.name = c.COLUMN_NAME
            LEFT JOIN sys.extended_properties ep
                ON ep.major_id = sc.object_id 
                AND ep.minor_id = sc.column_id 
                AND ep.class = 1 
                AND ep.name = 'MS_Description'
            WHERE c.TABLE_SCHEMA = %s AND c.TABLE_NAME = %s
            ORDER BY c.ORDINAL_POSITION
        """, (full_table_name, schema, table_name))
        except Exception as e:
            logger.warning(f"参数化查询失败，尝试使用字符串拼接: {e}")
            # 如果参数化查询失败，使用字符串拼接（注意SQL注入风险，但这里表名是系统表，相对安全）
            cursor.execute(f"""
                SELECT
                    c.COLUMN_NAME,
                    c.DATA_TYPE,
                    c.CHARACTER_MAXIMUM_LENGTH,
                    c.IS_NULLABLE,
                    c.COLUMN_DEFAULT,
                    ep.value AS COLUMN_COMMENT
                FROM INFORMATION_SCHEMA.COLUMNS c
                LEFT JOIN sys.columns sc
                    ON sc.object_id = OBJECT_ID('{full_table_name}') AND sc.name = c.COLUMN_NAME
                LEFT JOIN sys.extended_properties ep
                    ON ep.major_id = sc.object_id 
                    AND ep.minor_id = sc.column_id 
                    AND ep.class = 1 
                    AND ep.name = 'MS_Description'
                WHERE c.TABLE_SCHEMA = '{schema}' AND c.TABLE_NAME = '{table_name}'
                ORDER BY c.ORDINAL_POSITION
            """)
        columns = cursor.fetchall()
        for col in columns:
            # 对所有字段值进行解码处理
            for key in col:
                if col[key] is not None:
                    col[key] = try_decode(col[key])
        all_structures[full_table_name] = columns

    cursor.close()
    conn.close()
    return all_structures

def fetch_one_example_row(db_config, full_table_name, column_list):
    """
    获取单个表的一行示例数据。如果没有数据，返回空字符串。
    返回格式的值：对于含中文的字符串(如车站名)，用 N'五一广场' 这样形式
    """
    conn = None
    example_data = []
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        # 使用参数化查询避免SQL注入，但这里表名和列名是已知的，所以直接拼接
        sql = f"SELECT TOP 1 {', '.join([f'[{col}]' for col in column_list])} FROM {full_table_name}"
        cursor.execute(sql)
        row = cursor.fetchone()
        if row:
            # MSSQL-python driver可能返回 tuple
            # 行内处理每个元素，含中文的字符串直接转 N'xxx'
            example_data = []
            for x in row:
                # 先解码，再格式化
                decoded_val = try_decode(x)
                example_data.append(format_value_for_sql(decoded_val))
        cursor.close()
    except Exception as e:
        logger.warning(f"获取表 {full_table_name} 示例数据失败: {e}")
        example_data = []
    finally:
        if conn:
            conn.close()
    return example_data

def save_all_db_table_structures_to_txt(output_path="table_structures.txt"):
    """
    保存所有数据库的所有表结构详细信息到txt文档（含备注信息），并额外输出每个表示例数据一行
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for db_conf in DB_CONFIGS:
            db_name = db_conf["database"]
            logger.info(f"正在导出数据库: {db_name}")
            structures = fetch_table_structures_for_db(db_conf)
            f.write(f"\n# 数据库: {db_name}\n")
            for table, columns in structures.items():
                f.write(f"\n表：{table}\n")
                f.write("字段名\t类型\t长度\t是否可空\t默认值\t备注\n")
                for col in columns:
                    # 所有字段值已经在 fetch_table_structures_for_db 中解码过了
                    comment = col.get('COLUMN_COMMENT', '') or ''
                    colname = col.get('COLUMN_NAME', '')
                    dtype = col.get('DATA_TYPE', '')
                    clen = str(col.get('CHARACTER_MAXIMUM_LENGTH', '')) if col.get('CHARACTER_MAXIMUM_LENGTH') else ''
                    isnull = col.get('IS_NULLABLE', '')
                    default = str(col.get('COLUMN_DEFAULT', '')) if col.get('COLUMN_DEFAULT') else ''
                    f.write(
                        f"{colname}\t"
                        f"{dtype}\t"
                        f"{clen}\t"
                        f"{isnull}\t"
                        f"{default}\t"
                        f"{comment}\n"
                    )
                # 输出一行示例数据
                if columns:
                    field_names = [col.get('COLUMN_NAME', '') for col in columns]
                    example_row = fetch_one_example_row(db_conf, table, field_names)
                    if example_row:
                        # 确保所有值都是字符串，并且正确处理中文
                        example_strs = [str(val) if val is not None else '' for val in example_row]
                        f.write("示例数据\t" + "\t".join(example_strs) + "\n")
                    else:
                        f.write("示例数据\t(无数据)\n")
    logger.info(f"所有数据库表结构详细信息已输出到: {output_path}")

if __name__ == "__main__":
    print("正在输出所有数据库的表结构详细信息到txt文档...")
    save_all_db_table_structures_to_txt()
