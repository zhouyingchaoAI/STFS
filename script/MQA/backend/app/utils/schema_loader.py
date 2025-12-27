"""
表结构信息加载器
从 table_structures.txt 文件读取最新的表结构信息
"""
import os
import re
from typing import Dict, List
from pathlib import Path

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_table_structures(file_path: str = None) -> str:
    """
    从 table_structures.txt 文件加载表结构信息
    
    Args:
        file_path: 表结构文件路径，如果为 None，则从项目根目录查找
        
    Returns:
        格式化的表结构字符串
    """
    if file_path is None:
        # 从项目根目录查找 table_structures.txt
        current_dir = Path(__file__).parent.parent.parent.parent
        file_path = current_dir / "table_structures.txt"
        
        # 如果不存在，尝试从 backend 目录的父目录查找
        if not file_path.exists():
            file_path = current_dir.parent / "table_structures.txt"
    
    if not os.path.exists(file_path):
        logger.warning(f"Table structures file not found: {file_path}, using default schema")
        return _get_default_schema()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析并格式化表结构
        schema_info = _parse_table_structures(content)
        logger.info(f"Loaded table structures from {file_path}")
        return schema_info
    except Exception as e:
        logger.error(f"Failed to load table structures: {e}")
        return _get_default_schema()


def _parse_table_structures(content: str) -> str:
    """
    解析 table_structures.txt 内容并格式化为 LLM 可用的格式
    包含表结构、字段信息和示例数据
    
    Args:
        content: 文件内容
        
    Returns:
        格式化的表结构字符串
    """
    lines = content.split('\n')
    current_db = None
    current_table = None
    tables_info = []
    current_table_info = []
    field_names = []  # 存储字段名，用于解析示例数据
    example_data = None  # 存储示例数据
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 检测数据库
        if line.startswith('# 数据库:'):
            if current_table and current_table_info:
                # 添加示例数据
                if example_data:
                    current_table_info.append(f"\n示例数据： {example_data}")
                tables_info.append('\n'.join(current_table_info))
            current_db = line.replace('# 数据库:', '').strip()
            current_table = None
            current_table_info = []
            field_names = []
            example_data = None
            i += 1
            continue
        
        # 检测表
        if line.startswith('表：'):
            if current_table and current_table_info:
                # 添加示例数据
                if example_data:
                    current_table_info.append(f"\n示例数据： {example_data}")
                tables_info.append('\n'.join(current_table_info))
            current_table = line.replace('表：', '').strip()
            # 提取表名（去掉 dbo. 前缀）
            table_display = current_table.split('.')[-1] if '.' in current_table else current_table
            current_table_info = [f"\n表：{current_table} ({table_display})"]
            field_names = []
            example_data = None
            i += 1
            continue
        
        # 检测字段表头
        if line.startswith('字段名'):
            current_table_info.append("\n字段：")
            # 清空字段名列表，准备重新收集
            field_names = []
            i += 1
            continue
        
        # 检测示例数据
        if line.startswith('示例数据'):
            parts = line.split('\t')
            if len(parts) > 1:
                # 解析示例数据
                # 第一列是"示例数据"标签，后面的列对应字段的值
                example_values = [p.strip() for p in parts[1:] if p.strip()]
                if example_values and field_names:
                    # 构建示例数据描述（只显示关键字段）
                    example_parts = []
                    
                    # 关键字段列表（用于示例数据）
                    key_field_patterns = ['date', 'DATE', 'lineno', 'LINENO', 'linename', 'LINENAME', 
                                         'STATION', 'klcount', 'PKLCOUNT', 'PASSENGER', 'hour', 'HOUR',
                                         'TQQK', 'HOLIDAYTYPE', 'WEATHER']
                    
                    # 匹配关键字段并显示示例值
                    for idx, field_name in enumerate(field_names):
                        if idx >= len(example_values):
                            break
                        # 检查是否是关键字段
                        if any(pattern.lower() in field_name.lower() for pattern in key_field_patterns):
                            value = example_values[idx]
                            if value:
                                # 清理示例值（去掉 N'...' 格式）
                                if value.startswith("N'") and value.endswith("'"):
                                    value = value[2:-1]
                                # 限制值长度
                                if len(str(value)) > 25:
                                    value = str(value)[:25] + "..."
                                example_parts.append(f"{field_name}={value}")
                                # 最多显示8个关键字段
                                if len(example_parts) >= 8:
                                    break
                    
                    # 如果关键字段没有匹配到，显示前5个字段
                    if not example_parts and len(example_values) > 0:
                        for idx in range(min(5, len(example_values), len(field_names))):
                            field_name = field_names[idx]
                            value = example_values[idx]
                            if value:
                                if value.startswith("N'") and value.endswith("'"):
                                    value = value[2:-1]
                                if len(str(value)) > 25:
                                    value = str(value)[:25] + "..."
                                example_parts.append(f"{field_name}={value}")
                    
                    if example_parts:
                        example_data = ", ".join(example_parts)
            i += 1
            continue
        
        # 处理字段信息行（不是表头，不是示例数据，包含制表符）
        if current_table and line and '\t' in line and not line.startswith('示例数据') and not line.startswith('字段名'):
            parts = line.split('\t')
            if len(parts) >= 2:
                field_name = parts[0].strip()
                field_type = parts[1].strip() if len(parts) > 1 else ''
                field_length = parts[2].strip() if len(parts) > 2 else ''
                field_nullable = parts[3].strip() if len(parts) > 3 else ''
                field_default = parts[4].strip() if len(parts) > 4 else ''
                field_comment = parts[5].strip() if len(parts) > 5 else ''
                
                # 保存字段名到列表（用于后续示例数据匹配）
                if field_name and field_name not in field_names:
                    field_names.append(field_name)
                
                # 格式化字段信息
                field_desc = f"  - {field_name} ({field_type}"
                if field_length:
                    field_desc += f", 长度: {field_length}"
                field_desc += ")"
                if field_nullable == 'YES':
                    field_desc += " [可空]"
                if field_comment:
                    field_desc += f" - {field_comment}"
                
                current_table_info.append(field_desc)
        
        i += 1
    
    # 添加最后一个表
    if current_table and current_table_info:
        if example_data:
            current_table_info.append(f"\n示例数据： {example_data}")
        tables_info.append('\n'.join(current_table_info))
    
    # 组合所有表信息，按数据库分组
    schema_text = "\n\n".join(tables_info)
    
    # 添加数据库说明和格式说明
    schema_text = f"""数据库表结构信息（包含字段类型、备注和示例数据）：

{schema_text}

重要数据格式说明：

1. 日期字段格式（非常重要！）：
   - master数据库的历史表：日期字段使用整数格式 YYYYMMDD
     * CalendarHistory.f_date: int, 示例: 20171211
     * LineDailyFlowHistory.f_date: int, 示例: 20220101
     * LineHourlyFlowHistory.f_date: int, 示例: 20220101
     * STATION_FLOW_HISTORY.SQUAD_DATE: int, 示例: 20220101
     * STATION_HOUR_HISTORY.SQUAD_DATE: int, 示例: 20220101
     * WeatherHistory.F_DATE: int, 示例: 20230920
     * 查询时使用: f_date = 20220101 或 SQUAD_DATE = 20220101
   
   - CxFlowPredict数据库的预测表：日期字段使用date类型（YYYY-MM-DD格式）
     * STATION_FLOW_PREDICT.SQUAD_DATE: date, 示例: 2025-11-20
     * STATION_HOUR_PREDICT.SQUAD_DATE: date, 示例: 2025-11-03
     * 查询时使用: SQUAD_DATE = '2025-11-20' 或 SQUAD_DATE >= '2025-11-01' AND SQUAD_DATE <= '2025-11-30'
   
   - CxFlowPredict数据库的线路预测表：日期字段使用整数格式 YYYYMMDD
     * LineDailyFlowPrediction.F_DATE: int, 示例: 20250428
     * LineHourlyFlowPrediction.F_DATE: int, 示例: 20251009
     * 查询时使用: F_DATE = 20250428（整数格式，不是字符串 '2025-04-28'）
     * **重要**：LineDailyFlowPrediction和LineHourlyFlowPrediction表的客流量字段是 F_PKLCOUNT（不是 F_KLCCOUNT 或 f_klcount）
     * 正确示例：SELECT F_DATE AS 日期, SUM(F_PKLCOUNT) AS 客流量 FROM CxFlowPredict.dbo.LineDailyFlowPrediction WHERE F_LINENO = 1 AND F_DATE = 20251205 GROUP BY F_DATE
     * 错误示例：SELECT F_DATE AS 日期, SUM(F_KLCCOUNT) AS 客流量 FROM CxFlowPredict.dbo.LineDailyFlowPrediction WHERE F_LINENO = 1 AND F_DATE = '2025-12-05' ❌

2. 线路号格式：
   - 历史表：f_lineno (int), 示例: 1（表示1号线）
   - 预测表：F_LINENO (int), 示例: 5（表示5号线）
   - 车站表：LINE_ID (char), 示例: '03' 或 '04'（字符串格式，可能带前导空格）

3. 线路名格式：
   - 历史表：f_linename (varchar), 示例: '1号线'（使用 N'1号线' 或直接字符串）
   - 预测表：F_LINENAME (nvarchar), 示例: '5号线'（使用 N'5号线' 或直接字符串）
   - 车站名：STATION_NAME (varchar/nvarchar), 示例: '星沙', '长沙火车南站'（使用 N'xxx' 格式）

4. 小时格式：
   - LineHourlyFlowHistory.f_hour: int, 示例: 0（0-23）
   - LineHourlyFlowPrediction.F_HOUR: int, 示例: 6（0-23）
   - STATION_HOUR_HISTORY.TIME_SECTION_ID: char, 示例: '00'（字符串格式，可能带空格）
   - STATION_HOUR_PREDICT.TIME_SECTION_ID: char, 示例: '07'（字符串格式）

注意事项：
- 只生成SELECT查询语句
- 使用中文别名，如 "as 日期", "as 线路名"
- 表名必须包含dbo前缀，如 "FROM dbo.LineDailyFlowHistory"
- 历史表在master数据库，预测表在CxFlowPredict数据库
- 如果问题中提到"昨天"、"今天"等相对时间，请根据当前日期计算具体日期
- 如果问题中提到线路名称（如"1号线"），使用 f_lineno = 1 或 F_LINENO = 1
- 如果问题中提到车站名称，需要查询STATION_FLOW_HISTORY或STATION_HOUR_HISTORY表
- 车站名匹配规则：station_line_mapping.yaml中的车站名通常不带"站"字（如"五一广场"），
  用户可能输入"五一广场站"，在SQL中使用时应使用yaml中的原始名称（去掉"站"字）
- 车站名格式：必须使用 N'xxx' 格式，例如：STATION_NAME = N'五一广场'
- **车站表日期字段（非常重要！）**：
  * 车站表（STATION_FLOW_HISTORY, STATION_HOUR_HISTORY, STATION_FLOW_PREDICT, STATION_HOUR_PREDICT）的日期字段是 SQUAD_DATE，不是 f_date 或 F_DATE！
  * 正确示例：SELECT SUM(ENTRY_NUM) as 进站量 FROM master.dbo.STATION_FLOW_HISTORY WHERE SQUAD_DATE = 20251204 AND STATION_NAME = N'五一广场' GROUP BY SQUAD_DATE, STATION_NAME
  * 错误示例：SELECT SUM(ENTRY_NUM) as 进站量 FROM master.dbo.STATION_FLOW_HISTORY WHERE f_date = 20251204 ❌
- 如果问题中提到预测，需要查询CxFlowPredict数据库中的预测表
- 数据库前缀规则：
  * master数据库的表：FROM dbo.LineDailyFlowHistory（默认数据库，可以不指定）
  * CxFlowPredict数据库的表：FROM CxFlowPredict.dbo.STATION_FLOW_PREDICT（必须指定数据库前缀）
- 特别注意：STATION_FLOW_PREDICT和STATION_HOUR_PREDICT表的SQUAD_DATE是date类型，使用'YYYY-MM-DD'格式，不是整数！
- **日期字段类型和函数使用（非常重要！）**：
  * master数据库的历史表中，f_date字段是int类型（YYYYMMDD格式，如20220101），不是datetime类型
  * 不能使用DATEADD、DATEDIFF等日期函数直接操作int类型的f_date字段
  * 如果需要计算日期范围，应该：
    - 先计算目标日期，转换为YYYYMMDD整数格式
    - 然后使用整数比较：f_date >= 20231201 AND f_date <= 20231207
    - 示例（错误）：f_date BETWEEN DATEADD(year, -1, '2025-12-03') AND '2025-12-03' ❌
    - 示例（正确）：f_date >= 20241203 AND f_date <= 20251203 ✅
    - 或者：f_date BETWEEN 20241203 AND 20251203 ✅
  * 如果必须使用日期函数，需要先转换：CONVERT(datetime, CAST(f_date AS varchar(8)), 112)，但这样效率低，不推荐
- **车站数据合并规则（非常重要！）**：
  * 同一时间同一车站可能有多条数据（不同线路或不同记录），必须使用 GROUP BY 和 SUM 合并
  * 车站日客流（STATION_FLOW_HISTORY, STATION_FLOW_PREDICT）：
    - 按 SQUAD_DATE, STATION_NAME 分组
    - 数值字段（ENTRY_NUM, EXIT_NUM, CHANGE_NUM, PASSENGER_NUM, FLOW_NUM）使用 SUM 求和
    - 其他字段（LINE_ID, STATION_ID, STATION_NAME）使用 MIN 取第一个值
    - 示例：SELECT SQUAD_DATE as 日期, MIN(STATION_NAME) as 车站名, SUM(PASSENGER_NUM) as 客运量 FROM dbo.STATION_FLOW_HISTORY WHERE STATION_NAME = N'五一广场' GROUP BY SQUAD_DATE, STATION_NAME
  * 车站小时客流（STATION_HOUR_HISTORY, STATION_HOUR_PREDICT）：
    - 按 SQUAD_DATE, TIME_SECTION_ID, STATION_NAME 分组
    - 数值字段使用 SUM 求和，其他字段使用 MIN 取第一个值
    - 示例：SELECT SQUAD_DATE as 日期, MIN(TIME_SECTION_ID) as 小时, MIN(STATION_NAME) as 车站名, SUM(PASSENGER_NUM) as 客运量 FROM dbo.STATION_HOUR_HISTORY WHERE STATION_NAME = N'五一广场' GROUP BY SQUAD_DATE, TIME_SECTION_ID, STATION_NAME
- **天气表字段说明（非常重要！参考table_structures.txt）**：
  * WeatherHistory表（天气历史表，master.dbo.WeatherHistory）：
    - 日期字段：F_DATE (int类型，YYYYMMDD格式，如 20230920)，不是date类型，也不是DATE字段名
    - 天气类型字段：F_TQQK (nvarchar，如 N'小雨 /  小雨')
    - 气温字段：F_QW (nvarchar，如 N'22℃/29℃')
    - 风力字段：F_FLFX (nvarchar，如 N'北风 1-3级')
    - 详细天气字段：MAPPED_WEATHER (nvarchar)
    - 注意：WeatherHistory表没有DATE字段，也没有RECORD_DATE或WEATHER_CONDITION字段！
    - 正确示例：SELECT F_DATE as 日期, F_TQQK as 天气状况 FROM master.dbo.WeatherHistory WHERE F_DATE = 20241204
    - 错误示例：SELECT DATE as 日期, WEATHER_CONDITION as 天气 FROM WeatherData WHERE DATE = '2024-12-03' ❌
    - 错误示例：SELECT RECORD_DATE as 日期, WEATHER_CONDITION as 天气状况 FROM master.dbo.WeatherData WHERE RECORD_DATE = 20241204 ❌
- **线路预测表字段说明（非常重要！参考table_structures.txt）**：
  * LineDailyFlowPrediction和LineHourlyFlowPrediction表（CxFlowPredict数据库）：
    - 日期字段：F_DATE (int类型，YYYYMMDD格式，如 20250428)，不是date类型，不能使用字符串格式 'YYYY-MM-DD'
    - 客流量字段：F_PKLCOUNT (float类型)，不是 F_KLCCOUNT 或 f_klcount
    - 注意：预测表的字段名与历史表不同，历史表使用 f_klcount，预测表使用 F_PKLCOUNT
    - 正确示例：SELECT F_DATE AS 日期, SUM(F_PKLCOUNT) AS 客流量 FROM CxFlowPredict.dbo.LineDailyFlowPrediction WHERE F_LINENO = 1 AND F_DATE = 20251205 GROUP BY F_DATE
    - 错误示例：SELECT F_DATE AS 日期, SUM(F_KLCCOUNT) AS 客流量 FROM CxFlowPredict.dbo.LineDailyFlowPrediction WHERE F_LINENO = 1 AND F_DATE = '2025-12-05' ❌
- **时间排序（重要！）**：
  - 所有查询结果都应该按时间字段排序（升序，从早到晚），方便查看时间趋势
  - 车站表使用：ORDER BY SQUAD_DATE ASC
  - 线路预测表使用：ORDER BY F_DATE ASC
  - 线路历史表使用：ORDER BY f_date ASC
  - 如果SQL中没有ORDER BY子句，系统会自动添加，但建议在生成SQL时就包含ORDER BY
- **预测全部指标说明（重要！）**：
  - "预测全部指标"或"所有预测指标"指的是预测表中所有数值类型的指标字段（numeric、float、int等数值类型）
  - 需要根据查询的表结构，自动识别所有数值类型的字段作为预测指标
  - 常见的客流指标包括（但不限于）：
    1. 客流量（客运量）：PASSENGER_NUM（车站表）或 f_klcount/F_PKLCOUNT（线路表）
    2. 进站量：ENTRY_NUM（车站表）或 entry_num（线路表）
    3. 出站量：EXIT_NUM（车站表）或 exit_num（线路表）
    4. 换乘量：CHANGE_NUM（车站表）或 change_num（线路表）
    5. 乘降量：FLOW_NUM（车站表）或 flow_num（线路表）
    6. 其他数值字段：如 F_BEFPKLCOUNT、F_PRUTE、PREDICT_NUM 等（根据表结构确定）
  - 当用户询问"预测全部指标"时，需要：
    1. 查看表结构（参考上面的表结构信息）
    2. 识别所有数值类型的字段（numeric、float、int等，排除ID、日期等非指标字段）
    3. 在SELECT中包含所有这些数值字段
  - 示例（车站预测 - STATION_FLOW_PREDICT表）：
    SELECT SQUAD_DATE AS 日期, STATION_NAME AS 车站名, PASSENGER_NUM AS 客运量, ENTRY_NUM AS 进站量, EXIT_NUM AS 出站量, CHANGE_NUM AS 换乘量, FLOW_NUM AS 乘降量 FROM CxFlowPredict.dbo.STATION_FLOW_PREDICT WHERE STATION_NAME = N'五一广场' ORDER BY SQUAD_DATE ASC
  - 示例（车站小时预测 - STATION_HOUR_PREDICT表）：
    SELECT SQUAD_DATE AS 日期, TIME_SECTION_ID AS 小时, STATION_NAME AS 车站名, PASSENGER_NUM AS 客运量, ENTRY_NUM AS 进站量, EXIT_NUM AS 出站量, CHANGE_NUM AS 换乘量, FLOW_NUM AS 乘降量, PREDICT_NUM AS 预测数量 FROM CxFlowPredict.dbo.STATION_HOUR_PREDICT WHERE STATION_NAME = N'五一广场' ORDER BY SQUAD_DATE ASC, TIME_SECTION_ID ASC
  - 示例（线路预测 - LineDailyFlowPrediction表）：
    SELECT F_DATE AS 日期, F_LINENAME AS 线路名, F_PKLCOUNT AS 客运量, F_BEFPKLCOUNT AS 前一日客运量, F_PRUTE AS 预测率, entry_num AS 进站量, exit_num AS 出站量, change_num AS 换乘量, flow_num AS 乘降量 FROM CxFlowPredict.dbo.LineDailyFlowPrediction WHERE F_LINENO = 1 ORDER BY F_DATE ASC
  - 示例（线路小时预测 - LineHourlyFlowPrediction表）：
    SELECT F_DATE AS 日期, F_HOUR AS 小时, F_LINENAME AS 线路名, F_PKLCOUNT AS 客流量, F_BEFPKLCOUNT AS 前一日客流量, F_PRUTE AS 预测率, entry_num AS 进站量, exit_num AS 出站量, change_num AS 换乘量, flow_num AS 乘降量 FROM CxFlowPredict.dbo.LineHourlyFlowPrediction WHERE F_LINENO = 1 ORDER BY F_DATE ASC, F_HOUR ASC
- 不要使用JOIN，除非必要
- SQL语句必须以SELECT开头"""
    
    return schema_text


def _get_default_schema() -> str:
    """
    返回默认的表结构信息（如果文件不存在时使用）
    """
    return """
数据库表结构（master数据库）：

1. CalendarHistory (日历历史表)
   - f_date (int): 日期 (格式: YYYYMMDD)
   - f_year (int): 年
   - f_DayOfWeek (int): 每周第几天
   - f_week (int): 每年第几周
   - f_HolidayType (int): 节假日类型
   - f_HolidayDays (int): 该节假日天数
   - f_HolidayWhichDay (int): 节假日第几天
   - COVID19 (int): 是否疫情期间
   - f_weather (int): 天气类型

2. LineDailyFlowHistory (线路日客流历史表)
   - f_date (int): 日期 (格式: YYYYMMDD)
   - f_lb (int): 线网或者线路
   - f_lineno (int): 线路号
   - f_linename (varchar): 线路名
   - f_klcount (numeric): 客运量
   - entry_num (numeric): 进站量
   - exit_num (numeric): 出站量
   - change_num (numeric): 换乘量
   - flow_num (numeric): 乘降量

数据库表结构（CxFlowPredict数据库）：

3. LineDailyFlowPrediction (线路日客流预测表)
   - F_DATE (int): 日期
   - F_LINENO (int): 线路号
   - F_LINENAME (nvarchar): 线路名
   - F_PKLCOUNT (float): 预测客运量
   - entry_num (numeric): 进站量
   - exit_num (numeric): 出站量
   - change_num (numeric): 换乘量
   - flow_num (numeric): 乘降量
   - PREDICT_DATE (int): 预测日期
   - PREDICT_WEATHER (varchar): 预测天气
"""
