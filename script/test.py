
import pandas as pd
import pymssql

def get_all_user_tables(conn):
    """
    获取所有用户表（排除系统表）
    """
    # 只选 is_ms_shipped=0 的用户表，且排除系统schema如 sys, INFORMATION_SCHEMA, db_owner, db_accessadmin, db_backupoperator, db_datareader, db_datawriter, db_ddladmin, db_denydatareader, db_denydatawriter, db_securityadmin, guest, dbo
    query = """
    SELECT 
        s.name AS TABLE_SCHEMA, 
        t.name AS TABLE_NAME
    FROM sys.tables t
    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE t.is_ms_shipped = 0
      AND s.name NOT IN (
        'sys', 'INFORMATION_SCHEMA', 'db_owner', 'db_accessadmin', 'db_backupoperator', 
        'db_datareader', 'db_datawriter', 'db_ddladmin', 'db_denydatareader', 
        'db_denydatawriter', 'db_securityadmin', 'guest'
      )
    ORDER BY s.name, t.name
    """
    return pd.read_sql(query, conn)

def get_table_columns_with_comments(conn, schema, table):
    """
    获取指定表的所有字段、类型、是否可空、默认值、注释
    """
    query = f"""
    SELECT 
        c.COLUMN_NAME AS 字段名,
        c.DATA_TYPE + 
            CASE 
                WHEN c.CHARACTER_MAXIMUM_LENGTH IS NOT NULL AND c.DATA_TYPE IN ('char','varchar','nvarchar','nchar') 
                THEN '(' + 
                    CASE WHEN c.CHARACTER_MAXIMUM_LENGTH = -1 THEN 'MAX' ELSE CAST(c.CHARACTER_MAXIMUM_LENGTH AS VARCHAR) END 
                    + ')' 
                ELSE '' 
            END AS 数据类型,
        CASE WHEN c.IS_NULLABLE = 'YES' THEN '是' ELSE '否' END AS 允许空值,
        ISNULL(dc.definition, '') AS 默认值,
        ISNULL(ep.value, '') AS 字段说明
    FROM INFORMATION_SCHEMA.COLUMNS c
    LEFT JOIN sys.columns sc
        ON sc.object_id = OBJECT_ID('{schema}.{table}') AND sc.name = c.COLUMN_NAME
    LEFT JOIN sys.extended_properties ep
        ON ep.major_id = sc.object_id AND ep.minor_id = sc.column_id AND ep.name = 'MS_Description'
    LEFT JOIN sys.default_constraints dc
        ON dc.parent_object_id = sc.object_id AND dc.parent_column_id = sc.column_id
    WHERE c.TABLE_SCHEMA = '{schema}' AND c.TABLE_NAME = '{table}'
    ORDER BY c.ORDINAL_POSITION
    """
    return pd.read_sql(query, conn)

def get_table_comment(conn, schema, table):
    """
    获取表的注释（说明）
    """
    query = f"""
    SELECT ISNULL(ep.value, '') AS 表说明
    FROM sys.tables t
    LEFT JOIN sys.extended_properties ep
        ON ep.major_id = t.object_id AND ep.minor_id = 0 AND ep.name = 'MS_Description'
    WHERE t.name = '{table}' AND SCHEMA_NAME(t.schema_id) = '{schema}'
    """
    df = pd.read_sql(query, conn)
    if not df.empty:
        return df.iloc[0, 0]
    return ""

def export_db_schema_to_excel(server, user, password, database, port, excel_path):
    """
    导出数据库所有用户表结构到Excel，每个表一个sheet，包含字段注释
    """
    conn = pymssql.connect(
        server=server,
        user=user,
        password=password,
        database=database,
        port=port
    )
    tables = get_all_user_tables(conn)
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    for idx, row in tables.iterrows():
        schema = row['TABLE_SCHEMA']
        table = row['TABLE_NAME']
        df = get_table_columns_with_comments(conn, schema, table)
        table_comment = get_table_comment(conn, schema, table)
        # 在DataFrame顶部插入表名和表注释
        meta = pd.DataFrame({
            '字段名': [f'表名: {schema}.{table}'],
            '数据类型': [f'表说明: {table_comment}'],
            '允许空值': [''],
            '默认值': [''],
            '字段说明': ['']
        })
        df_out = pd.concat([meta, df], ignore_index=True)
        # Excel sheet名不能超过31字符
        sheet_name = f"{schema}.{table}"
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        df_out.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.close()
    conn.close()
    print(f"所有用户表结构已导出到 {excel_path}")

def main():
    # 配置数据库连接信息
    server = '192.168.10.76'
    user = 'sa'
    password = 'Chency@123'
    database = 'master'
    port = 1433
    excel_path = 'master_db_schema.xlsx'
    export_db_schema_to_excel(server, user, password, database, port, excel_path)

if __name__ == "__main__":
    main()