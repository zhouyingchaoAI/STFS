"""
基于大语言模型的NL2SQL引擎（支持Ollama）
"""
from typing import Dict, Optional, List, Any
import json
import re
from datetime import datetime, timedelta

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama package not installed, LLM engine will not work")


class LLMBasedNL2SQLEngine:
    """基于大语言模型的NL2SQL引擎"""
    
    def __init__(self):
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package is required for LLM engine")
        
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        self.api_base = settings.LLM_API_BASE or "http://localhost:11434"
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.context_length = settings.LLM_CONTEXT_LENGTH  # 上下文窗口大小（num_ctx）
        self.temperature = settings.LLM_TEMPERATURE
        
        # 初始化Ollama客户端
        if self.provider == "ollama":
            # Ollama使用本地API，不需要API key
            self.client = ollama.Client(host=self.api_base)
        
        # 数据库表结构信息（用于构建prompt）
        self.schema_info = self._load_schema_info()
        
        # 线路列表和车站列表（用于构建prompt）
        self.line_list, self.station_list = self._load_line_station_lists()
    
    def _load_schema_info(self) -> str:
        """
        加载数据库表结构信息
        优先从 table_structures.txt 文件读取，如果文件不存在则使用默认信息
        """
        try:
            from app.utils.schema_loader import load_table_structures
            schema = load_table_structures()
            if schema and schema.strip():
                return schema
        except Exception as e:
            logger.warning(f"Failed to load table structures from file: {e}, using default schema")
        
        # 如果加载失败，使用默认表结构
        schema = """
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

3. LineHourlyFlowHistory (线路小时客流历史表)
   - f_date (int): 日期
   - f_hour (int): 小时 (0-23)
   - f_lineno (int): 线路号
   - f_linename (varchar): 线路名
   - f_klcount (numeric): 客运量
   - entry_num (numeric): 进站量
   - exit_num (numeric): 出站量
   - change_num (numeric): 换乘量
   - flow_num (numeric): 乘降量

4. LSTM_COMMON_HOLIDAYFEATURE (LSTM通用节假日特征表)
   - F_DATE (int): 日期
   - F_WEEK (int): 本年第几周
   - F_DATEFEATURES (varchar): 节假日工作日周末类别
   - F_HOLIDAYTYPE (varchar): 节假日类型
   - F_ISHOLIDAY (int): 是否节假日
   - F_ISNONGLI (int): 是否农历
   - F_ISYANGLI (int): 是否阳历
   - F_NEXTDAY (int): 是否节假日后一天
   - F_HOLIDAYDAYS (int): 节假日天数
   - F_HOLIDAYTHDAY (decimal): 节假日第几天

5. STATION_FLOW_HISTORY (车站客流历史表)
   - SQUAD_DATE (int): 日期 (格式: YYYYMMDD)
   - LINE_ID (char): 线路号
   - STATION_ID (char): 车站号
   - STATION_NAME (varchar): 车站名
   - ENTRY_NUM (numeric): 进站量
   - EXIT_NUM (numeric): 出站量
   - CHANGE_NUM (numeric): 换乘量
   - PASSENGER_NUM (numeric): 客运量
   - FLOW_NUM (numeric): 乘降量
   - 注意：车站表使用 SQUAD_DATE 作为日期字段，不是 f_date 或 F_DATE！

6. STATION_HOUR_HISTORY (车站小时客流历史表)
   - SQUAD_DATE (int): 日期 (格式: YYYYMMDD)
   - TIME_SECTION_ID (char): 小时
   - LINE_ID (char): 线路号
   - STATION_ID (char): 车站号
   - STATION_NAME (varchar): 车站名
   - ENTRY_NUM (numeric): 进站量
   - EXIT_NUM (numeric): 出站量
   - CHANGE_NUM (numeric): 换乘量
   - PASSENGER_NUM (numeric): 客运量
   - FLOW_NUM (numeric): 乘降量
   - 注意：车站表使用 SQUAD_DATE 作为日期字段，不是 f_date 或 F_DATE！

7. WeatherFuture (未来天气表)
   - F_DATE (int): 日期
   - F_TQQK (nvarchar): 天气情况
   - F_QW (nvarchar): 气温
   - F_FLFX (nvarchar): 风力风向
   - PREDICT_DATE (int): 预测日期
   - MAPPED_WEATHER (nvarchar): 映射天气

8. WeatherHistory (天气历史表)
   - F_DATE (int): 日期
   - F_TQQK (nvarchar): 天气类型
   - F_QW (nvarchar): 气温
   - F_FLFX (nvarchar): 风力等级
   - MAPPED_WEATHER (nvarchar): 详细天气

数据库表结构（CxFlowPredict数据库）：

9. LineDailyFlowPrediction (线路日客流预测表)
   - F_DATE (int): 日期
   - F_HOLIDAYTYPE (varchar): 节假日类别
   - F_LB (int): 线网或者线路
   - F_LINENO (int): 线路号
   - F_LINENAME (nvarchar): 线路名
   - F_PKLCOUNT (float): 预测客运量
   - entry_num (numeric): 进站量
   - exit_num (numeric): 出站量
   - change_num (numeric): 换乘量
   - flow_num (numeric): 乘降量
   - PREDICT_DATE (int): 预测日期
   - PREDICT_WEATHER (varchar): 预测天气
   - F_TYPE (varchar): 类别

10. LineHourlyFlowPrediction (线路小时客流预测表)
    - F_DATE (int): 日期
    - F_HOUR (int): 小时
    - F_LINENO (int): 线路号
    - F_LINENAME (nvarchar): 线路名
    - F_PKLCOUNT (float): 预测客流量
    - entry_num (numeric): 进站量
    - exit_num (numeric): 出站量
    - change_num (numeric): 换乘量
    - flow_num (numeric): 乘降量
    - PREDICT_DATE (int): 预测日期
    - PREDICT_WEATHER (varchar): 预测天气

11. STATION_FLOW_PREDICT (车站客流预测表)
    - SQUAD_DATE (date): 日期 (格式: 'YYYY-MM-DD')
    - PREDICT_DATE (date): 预测日期
    - LINE_ID (char): 线路号
    - STATION_ID (char): 车站号
    - STATION_NAME (varchar): 车站名
    - ENTRY_NUM (numeric): 进站量
    - EXIT_NUM (numeric): 出站量
    - CHANGE_NUM (numeric): 换乘量
    - PASSENGER_NUM (numeric): 客运量
    - FLOW_NUM (numeric): 乘降量
    - 注意：车站表使用 SQUAD_DATE 作为日期字段，不是 f_date 或 F_DATE！

12. STATION_HOUR_PREDICT (车站小时客流预测表)
    - SQUAD_DATE (date): 日期 (格式: 'YYYY-MM-DD')
    - PREDICT_DATE (date): 预测日期
    - TIME_SECTION_ID (char): 小时
    - LINE_ID (char): 线路号
    - STATION_ID (char): 车站号
    - STATION_NAME (varchar): 车站名
    - ENTRY_NUM (numeric): 进站量
    - EXIT_NUM (numeric): 出站量
    - CHANGE_NUM (numeric): 换乘量
    - PASSENGER_NUM (numeric): 客运量
    - FLOW_NUM (numeric): 乘降量
    - 注意：车站表使用 SQUAD_DATE 作为日期字段，不是 f_date 或 F_DATE！

注意事项：
- 日期字段使用整数格式 YYYYMMDD，如 20240101 表示 2024年1月1日
- 只生成SELECT查询语句
- 使用中文别名，如 "as 日期", "as 线路名"
- 表名必须包含dbo前缀，如 "dbo.LineDailyFlowHistory"
- 历史表在master数据库，预测表在CxFlowPredict数据库
- 不要使用JOIN，除非必要
- SQL语句必须以SELECT开头
"""
        return schema
    
    def _load_line_station_lists(self) -> tuple:
        """
        加载线路列表和车站列表
        返回: (线路列表, 车站列表)
        """
        lines = ["1号线", "2号线", "3号线", "4号线", "5号线", "6号线"]
        stations = []
        
        try:
            import yaml
            from pathlib import Path
            
            # 尝试多个可能的路径
            possible_paths = [
                Path(__file__).parent.parent.parent.parent.parent / "station_line_mapping.yaml",
                Path(__file__).parent.parent.parent.parent / "station_line_mapping.yaml",
                Path("station_line_mapping.yaml"),
            ]
            
            for mapping_file in possible_paths:
                if mapping_file.exists():
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        station_mapping = yaml.safe_load(f) or {}
                        stations = list(station_mapping.keys())
                        # 去重并排序
                        stations = sorted(list(set(stations)))
                        break
        except Exception as e:
            logger.warning(f"Failed to load station mapping: {e}, using empty list")
        
        return lines, stations
    
    def convert(self, question: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict]:
        """
        使用LLM将自然语言问题转换为SQL
        
        Args:
            question: 自然语言问题
            conversation_history: 对话历史，用于多轮对话修正
            
        Returns:
            包含SQL、意图、实体等信息的字典
        """
        try:
            # 构建prompt（包含对话历史上下文）
            prompt = self._build_prompt(question, conversation_history)
            
            logger.debug(f"LLM Prompt: {prompt[:200]}...")
            
            # 调用LLM
            if self.provider == "ollama":
                response = self._call_ollama(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
            # 解析响应
            sql_result = self._parse_response(response, question)
            
            return sql_result
            
        except Exception as e:
            logger.error(f"LLM NL2SQL conversion error: {e}", exc_info=True)
            return None
    
    def _build_prompt(self, question: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        构建LLM prompt，支持多轮对话上下文
        
        Args:
            question: 当前问题
            conversation_history: 对话历史，包含之前的错误信息
        """
        # 获取当前时间
        now = datetime.now()
        current_time = now.strftime("%Y年%m月%d日 %H时%M分%S秒")
        current_date_int = int(now.strftime("%Y%m%d"))
        current_hour = now.hour
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")  # 标准日期时间格式
        current_weekday = now.strftime("%A")  # 星期几（英文）
        current_weekday_cn = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'][now.weekday()]  # 星期几（中文）
        
        # 计算本周的日期范围（周一到周日）
        days_since_monday = now.weekday()  # 0=Monday, 6=Sunday
        week_start = now - timedelta(days=days_since_monday)
        week_end = week_start + timedelta(days=6)
        week_start_int = int(week_start.strftime("%Y%m%d"))
        week_end_int = int(week_end.strftime("%Y%m%d"))
        
        # 计算上周的日期范围
        last_week_start = week_start - timedelta(days=7)
        last_week_end = last_week_start + timedelta(days=6)
        last_week_start_int = int(last_week_start.strftime("%Y%m%d"))
        last_week_end_int = int(last_week_end.strftime("%Y%m%d"))
        
        # 计算本月的日期范围
        month_start = datetime(now.year, now.month, 1)
        if now.month == 12:
            month_end = datetime(now.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(now.year, now.month + 1, 1) - timedelta(days=1)
        month_start_int = int(month_start.strftime("%Y%m%d"))
        month_end_int = int(month_end.strftime("%Y%m%d"))
        
        # 计算昨天
        yesterday_int = int((now - timedelta(days=1)).strftime("%Y%m%d"))
        
        # 构建对话历史上下文
        history_context = ""
        if conversation_history:
            history_context = "\n\n## 对话历史（之前的错误信息）：\n"
            for i, hist in enumerate(conversation_history[-3:], 1):  # 只取最近3条历史
                if hist.get("error"):
                    history_context += f"\n第{i}轮对话：\n"
                    history_context += f"- 问题：{hist.get('question', '')}\n"
                    history_context += f"- 错误：{hist.get('error', '')}\n"
                    if hist.get("failed_sql"):
                        history_context += f"- 失败的SQL：{hist.get('failed_sql', '')}\n"
                    history_context += "\n"
        
        prompt = f"""你是一个专业的SQL生成助手，负责将自然语言问题转换为SQL Server查询语句。

**当前日期时间信息（非常重要！必须使用此信息计算相对时间）：**
- 当前时间：{current_time}
- 当前日期时间（标准格式）：{current_datetime}
- **当前日期（整数格式，用于SQL查询）：{current_date_int}** ← 这是今天的日期，必须使用！
- 今天日期：{current_date_int}（YYYYMMDD格式）
- 当前小时：{current_hour}
- 当前星期：{current_weekday_cn}（{current_weekday}）
**相对时间计算规则（必须严格遵守！）：**
- **今天**：使用日期 {current_date_int}
- **昨天**：使用日期 {yesterday_int}
- **本周**：本周的日期范围是 {week_start_int} 到 {week_end_int}（周一到周日）
  * 如果查询 LineDailyFlowPrediction 表，使用：F_DATE BETWEEN {week_start_int} AND {week_end_int}
  * 如果查询 STATION_FLOW_PREDICT 表，需要转换为日期格式：SQUAD_DATE BETWEEN '{week_start.strftime("%Y-%m-%d")}' AND '{week_end.strftime("%Y-%m-%d")}'
- **上周**：上周的日期范围是 {last_week_start_int} 到 {last_week_end_int}（周一到周日）
- **本月**：本月的日期范围是 {month_start_int} 到 {month_end_int}（1号到最后一天）
- **绝对禁止使用示例数据中的旧日期（如2023年、2024年的日期）！必须使用上面提供的当前日期进行计算！**

{self.schema_info}
{history_context}
请根据以下自然语言问题生成对应的SQL查询语句：

问题：{question}

要求：
1. **思考过程（重要！）**：在生成SQL之前，请先展示你的思考过程，包括：
   - 理解用户问题的意图
   - 识别需要查询的表和字段
   - 分析日期、线路、车站等条件
   - 确定需要使用的聚合函数（如SUM、GROUP BY等）
   - 思考过程可以写在SQL之前，用自然语言描述
2. **GROUP BY 规则（非常重要！）**：
   - 当使用 GROUP BY 时，SELECT 中的所有字段必须满足以下条件之一：
     a) 使用聚合函数（SUM、COUNT、AVG、MAX、MIN等）
     b) 出现在 GROUP BY 子句中
   - 错误示例：SELECT F_DATE, F_LINENAME, SUM(F_PKLCOUNT), F_BEFPKLCOUNT FROM ... GROUP BY F_DATE, F_LINENAME ❌
     （F_BEFPKLCOUNT 既没有聚合函数，也不在 GROUP BY 中）
   - 正确做法1（数值字段使用聚合）：SELECT F_DATE, F_LINENAME, SUM(F_PKLCOUNT), MIN(F_BEFPKLCOUNT) FROM ... GROUP BY F_DATE, F_LINENAME ✅
   - 正确做法2（添加到 GROUP BY）：SELECT F_DATE, F_LINENAME, SUM(F_PKLCOUNT), F_BEFPKLCOUNT FROM ... GROUP BY F_DATE, F_LINENAME, F_BEFPKLCOUNT ✅
   - 注意：如果同一分组中某字段的值都相同，可以使用 MIN() 或 MAX() 聚合；如果值可能不同，应该添加到 GROUP BY 中
3. 使用中文别名（如 "as 日期", "as 线路名"）
3. 日期条件格式（非常重要！）：
   - master数据库的历史表：使用整数格式 YYYYMMDD（如 f_date = 20240101）
     * 注意：f_date是int类型，不是datetime类型，不能使用DATEADD、DATEDIFF等日期函数
     * 错误示例：f_date BETWEEN DATEADD(year, -1, '2025-12-03') AND '2025-12-03' ❌
     * 正确做法：先计算目标日期为整数，然后使用整数比较
     * 正确示例：f_date >= 20241203 AND f_date <= 20251203 ✅
     * 或者：f_date BETWEEN 20241203 AND 20251203 ✅
   - CxFlowPredict数据库的STATION_FLOW_PREDICT和STATION_HOUR_PREDICT表：使用date格式 'YYYY-MM-DD'（如 SQUAD_DATE = '2025-11-20'）
   - CxFlowPredict数据库的LineDailyFlowPrediction和LineHourlyFlowPrediction表：使用整数格式 YYYYMMDD（如 F_DATE = 20250428）
     * **重要**：LineDailyFlowPrediction和LineHourlyFlowPrediction表的客流量字段是 F_PKLCOUNT（不是 F_KLCCOUNT 或 f_klcount）
     * **重要**：F_DATE 是 int 类型，必须使用整数格式 YYYYMMDD，不能使用字符串格式 'YYYY-MM-DD'
     * 正确示例：SELECT F_DATE AS 日期, SUM(F_PKLCOUNT) AS 客流量 FROM CxFlowPredict.dbo.LineDailyFlowPrediction WHERE F_LINENO = 1 AND F_DATE = 20251205 GROUP BY F_DATE
     * 错误示例：SELECT F_DATE AS 日期, SUM(F_KLCCOUNT) AS 客流量 FROM CxFlowPredict.dbo.LineDailyFlowPrediction WHERE F_LINENO = 1 AND F_DATE = '2025-12-05' ❌
4. **相对时间计算（非常重要！必须严格遵守！）**：
   - **今天**：使用日期 {current_date_int}
   - **昨天**：使用日期 {yesterday_int}
   - **本周**：本周的日期范围是 {week_start_int} 到 {week_end_int}（周一到周日）
     * 如果查询 LineDailyFlowPrediction 表，使用：F_DATE BETWEEN {week_start_int} AND {week_end_int}
     * 如果查询 STATION_FLOW_PREDICT 表，使用：SQUAD_DATE BETWEEN '{week_start.strftime("%Y-%m-%d")}' AND '{week_end.strftime("%Y-%m-%d")}'
   - **上周**：上周的日期范围是 {last_week_start_int} 到 {last_week_end_int}（周一到周日）
   - **本月**：本月的日期范围是 {month_start_int} 到 {month_end_int}（1号到最后一天）
   - **绝对禁止使用示例数据中的旧日期（如2023年、2024年的日期）！必须使用上面提供的当前日期进行计算！**
   - 所有相对时间都必须转换为整数格式 YYYYMMDD（历史表和线路预测表）或 'YYYY-MM-DD' 格式（车站预测表），不要使用 DATEADD 等日期函数
5. **线路名称映射（重要！）**：
   - 可用线路列表：{', '.join(self.line_list)}
   - 线路名称到线路号的映射：
     * "1号线"、"一号线"、"1线" → 使用 f_lineno = 1 或 F_LINENO = 1
     * "2号线"、"二号线"、"2线" → 使用 f_lineno = 2 或 F_LINENO = 2
     * "3号线"、"三号线"、"3线" → 使用 f_lineno = 3 或 F_LINENO = 3
     * "4号线"、"四号线"、"4线" → 使用 f_lineno = 4 或 F_LINENO = 4
     * "5号线"、"五号线"、"5线" → 使用 f_lineno = 5 或 F_LINENO = 5
     * "6号线"、"六号线"、"6线" → 使用 f_lineno = 6 或 F_LINENO = 6
6. **车站名称映射（非常重要！）**：
   - 可用车站列表（共{len(self.station_list)}个车站）：{', '.join(self.station_list[:50])}{'...' if len(self.station_list) > 50 else ''}
   - **完整车站列表**（按字母顺序）：{', '.join(sorted(self.station_list))}
   - 如果问题中提到车站名称，需要查询STATION_FLOW_HISTORY、STATION_HOUR_HISTORY、STATION_FLOW_PREDICT或STATION_HOUR_PREDICT表
   - **车站名格式规则（必须严格遵守！）**：
     * station_line_mapping.yaml中的车站名通常不带"站"字（如"五一广场"、"碧沙湖"），如果用户说"五一广场站"或"碧沙湖地铁站"，应匹配为"五一广场"或"碧沙湖"（去掉"站"、"地铁站"等后缀）
     * **重要**：用户可能说"碧沙湖地铁站"，但数据库中存储的是"碧沙湖"（不带"站"字），必须使用"碧沙湖"
     * **重要**：用户可能说"碧波站"，但正确的车站名是"碧沙湖"，不是"碧波"！请仔细检查车站列表，确保使用正确的车站名称
     * 对于中文字段值（如车站名），必须使用 N'xxx' 格式，例如：STATION_NAME = N'碧沙湖'（不是 N'碧波'）
     * 车站名必须完全匹配，区分大小写，例如："五一广场" ≠ "五一廣場"，"碧沙湖" ≠ "碧波"
     * **匹配步骤**：1) 去掉用户输入中的"站"、"地铁站"等后缀；2) 在完整车站列表中查找最匹配的车站名；3) 使用找到的准确车站名生成SQL
   - **重要**：车站表（STATION_FLOW_HISTORY, STATION_HOUR_HISTORY, STATION_FLOW_PREDICT, STATION_HOUR_PREDICT）的日期字段是 SQUAD_DATE，不是 f_date 或 F_DATE！
     * 正确示例：SELECT SUM(ENTRY_NUM) as 进站量 FROM master.dbo.STATION_FLOW_HISTORY WHERE SQUAD_DATE = 20251204 AND STATION_NAME = N'五一广场' GROUP BY SQUAD_DATE, STATION_NAME
     * 错误示例：SELECT SUM(ENTRY_NUM) as 进站量 FROM master.dbo.STATION_FLOW_HISTORY WHERE f_date = 20251204 ❌
7. 如果问题中提到预测，需要查询CxFlowPredict数据库中的预测表
8. 表名必须包含数据库前缀和dbo前缀：
   - master数据库的表：FROM master.dbo.LineDailyFlowHistory 或 FROM dbo.LineDailyFlowHistory（默认数据库）
   - CxFlowPredict数据库的表：FROM CxFlowPredict.dbo.STATION_FLOW_PREDICT（必须指定数据库）
9. **输出格式**：
   - 先输出思考过程（用自然语言描述你的分析过程）
   - 然后输出SQL语句
   - 不要使用markdown代码块标记
   - SQL必须以SELECT开头
10. **时间排序（重要！）**：
    - 所有查询结果都应该按时间字段排序（升序，从早到晚）
    - 车站表使用：ORDER BY SQUAD_DATE ASC
    - 线路预测表使用：ORDER BY F_DATE ASC
    - 线路历史表使用：ORDER BY f_date ASC
    - 如果SQL中没有ORDER BY子句，系统会自动添加，但建议在生成SQL时就包含ORDER BY
11. 示例输出格式：
    思考过程：
    用户想查询五一广场站的客流量。这是一个车站客流查询，需要查询STATION_FLOW_PREDICT表。
    需要筛选STATION_NAME = N'五一广场'，日期范围是2025-12-02到2025-12-08。
    由于同一车站同一日期可能有多条记录，需要使用GROUP BY和SUM聚合。
    结果按时间升序排序，方便查看时间趋势。
    
    SELECT SQUAD_DATE AS 日期, SUM(PASSENGER_NUM) AS 客流量 FROM CxFlowPredict.dbo.STATION_FLOW_PREDICT WHERE STATION_NAME = N'五一广场' AND SQUAD_DATE BETWEEN '2025-12-02' AND '2025-12-08' GROUP BY SQUAD_DATE ORDER BY SQUAD_DATE ASC
11. 参考示例数据中的格式和值，确保字段类型匹配
12. 对于中文字段值（如线路名、车站名），必须使用 N'xxx' 格式，例如：STATION_NAME = N'五一广场'
13. 特别注意：查询STATION_FLOW_PREDICT或STATION_HOUR_PREDICT时，SQUAD_DATE必须使用 'YYYY-MM-DD' 格式，不能使用整数格式
14. 特别注意：CxFlowPredict数据库的表必须使用完整路径：CxFlowPredict.dbo.表名
15. 字段名注意：
   - 车站表的客运量字段是 PASSENGER_NUM（不是 passenger_number）
   - 线路表的客流量字段：
     * master数据库历史表：f_klcount（LineDailyFlowHistory, LineHourlyFlowHistory）
     * CxFlowPredict数据库预测表：F_PKLCOUNT（LineDailyFlowPrediction, LineHourlyFlowPrediction）
     * **重要**：预测表必须使用 F_PKLCOUNT，不能使用 F_KLCCOUNT 或 f_klcount
   - 所有字段名必须与表结构中的字段名完全一致（区分大小写），请参考 table_structures.txt 中的准确字段名
16. 天气表字段（非常重要！参考table_structures.txt）：
   - WeatherHistory表（天气历史表，master.dbo.WeatherHistory）：
     * 日期字段：F_DATE (int类型，YYYYMMDD格式，如 20230920)，不是date类型，也不是DATE字段名
     * 天气类型字段：F_TQQK (nvarchar，如 N'小雨 /  小雨')
     * 气温字段：F_QW (nvarchar，如 N'22℃/29℃')
     * 风力字段：F_FLFX (nvarchar，如 N'北风 1-3级')
     * 详细天气字段：MAPPED_WEATHER (nvarchar)
     * 注意：WeatherHistory表没有DATE字段，也没有RECORD_DATE或WEATHER_CONDITION字段！
     * 正确示例：SELECT F_DATE as 日期, F_TQQK as 天气状况 FROM master.dbo.WeatherHistory WHERE F_DATE = 20241204
     * 错误示例：SELECT DATE as 日期, WEATHER_CONDITION as 天气 FROM WeatherData WHERE DATE = '2024-12-03' ❌
     * 错误示例：SELECT RECORD_DATE as 日期, WEATHER_CONDITION as 天气状况 FROM master.dbo.WeatherData WHERE RECORD_DATE = 20241204 ❌
17. **预测全部指标说明（重要！）**：
   - "预测全部指标"或"所有预测指标"指的是预测表中所有数值类型的指标字段（numeric、float、int等数值类型）
   - 需要根据查询的表结构，自动识别所有数值类型的字段作为预测指标
   - 常见的客流指标包括（但不限于）：
     1. 客流量（客运量）：PASSENGER_NUM（车站表）或 f_klcount/F_PKLCOUNT（线路表）
     2. 进站量：ENTRY_NUM（车站表）或 entry_num（线路表）
     3. 出站量：EXIT_NUM（车站表）或 exit_num（线路表）
     4. 换乘量：CHANGE_NUM（车站表）或 change_num（线路表）
     5. 乘降量：FLOW_NUM（车站表）或 flow_num（线路表）
     6. 其他数值字段：如 F_BEFPKLCOUNT（前一日客运量）、F_PRUTE（预测率）、PREDICT_NUM（预测数量）等（根据表结构确定）
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

SQL语句："""
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """调用Ollama API（非流式）"""
        try:
            # 使用generate方法，返回完整响应
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "num_ctx": self.context_length,  # 上下文窗口大小
                }
            )
            
            # Ollama的generate方法可能返回流式响应或完整响应
            # 根据实际返回类型处理
            if isinstance(response, dict):
                # 如果是字典，直接获取response字段
                full_response = response.get('response', '')
            elif hasattr(response, '__iter__'):
                # 如果是生成器，拼接所有chunk
                full_response = ""
                for chunk in response:
                    if isinstance(chunk, dict) and 'response' in chunk:
                        full_response += chunk['response']
                    elif isinstance(chunk, str):
                        full_response += chunk
            else:
                full_response = str(response)
            
            logger.debug(f"Ollama response: {full_response[:200]}...")
            return full_response
            
        except Exception as e:
            logger.error(f"Ollama API call error: {e}")
            raise
    
    def _call_ollama_stream(self, prompt: str):
        """
        调用Ollama API（流式）
        返回生成器，每个chunk格式: {"response": "text", "done": False}
        """
        import time
        chunk_count = 0
        start_time = time.time()
        
        try:
            logger.info(f"开始调用Ollama流式API，model={self.model}, prompt_length={len(prompt)}")
            # 使用stream=True获取流式响应
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "num_ctx": self.context_length,  # 上下文窗口大小
                }
            )
            
            logger.info("Ollama流式响应生成器已创建，开始迭代chunks...")
            
            # 返回生成器，Ollama返回的chunk格式通常是 {"response": "text", "done": False}
            for chunk in response:
                chunk_count += 1
                elapsed = time.time() - start_time
                
                if isinstance(chunk, dict):
                    # Ollama返回格式: {"response": "text", "done": False, ...}
                    # 记录前20个chunk的详细信息
                    if chunk_count <= 20:
                        logger.debug(f"[Ollama Chunk {chunk_count}] (elapsed={elapsed:.3f}s): {chunk}")
                    elif chunk_count % 50 == 0:
                        logger.info(f"[Ollama Chunk {chunk_count}] (elapsed={elapsed:.3f}s): 已处理 {chunk_count} 个chunk")
                    
                    yield chunk
                elif isinstance(chunk, str):
                    # 如果是字符串，包装成字典
                    if chunk_count <= 20:
                        logger.debug(f"[Ollama Chunk {chunk_count}] (elapsed={elapsed:.3f}s): 字符串类型，内容={repr(chunk[:50])}")
                    yield {"response": chunk, "done": False}
                else:
                    # 其他类型转换为字符串
                    if chunk_count <= 20:
                        logger.debug(f"[Ollama Chunk {chunk_count}] (elapsed={elapsed:.3f}s): 其他类型，转换为字符串: {type(chunk)}")
                    yield {"response": str(chunk), "done": False}
            
            total_time = time.time() - start_time
            logger.info(f"Ollama流式响应完成: 共处理 {chunk_count} 个chunk，总耗时 {total_time:.3f}秒")
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Ollama stream API call error (elapsed={elapsed:.3f}s): {e}", exc_info=True)
            raise
    
    def _post_process_sql(self, sql: str) -> str:
        """
        后处理SQL，确保格式正确
        特别注意：STATION_FLOW_PREDICT 和 STATION_HOUR_PREDICT 表的 SQUAD_DATE 是 date 类型，需要使用 'YYYY-MM-DD' 格式
        """
        # 移除markdown代码块标记
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        sql = sql.strip()
        
        # 确保表名包含数据库前缀和dbo前缀
        # 匹配 FROM 后面的表名
        patterns = [
            # master数据库表（默认数据库，可以不指定）
            (r'FROM\s+CalendarHistory\b', 'FROM dbo.CalendarHistory'),
            (r'FROM\s+LineDailyFlowHistory\b', 'FROM dbo.LineDailyFlowHistory'),
            (r'FROM\s+LineHourlyFlowHistory\b', 'FROM dbo.LineHourlyFlowHistory'),
            (r'FROM\s+LSTM_COMMON_HOLIDAYFEATURE\b', 'FROM dbo.LSTM_COMMON_HOLIDAYFEATURE'),
            (r'FROM\s+STATION_FLOW_HISTORY\b', 'FROM dbo.STATION_FLOW_HISTORY'),
            (r'FROM\s+STATION_HOUR_HISTORY\b', 'FROM dbo.STATION_HOUR_HISTORY'),
            (r'FROM\s+WeatherFuture\b', 'FROM dbo.WeatherFuture'),
            (r'FROM\s+WeatherHistory\b', 'FROM dbo.WeatherHistory'),
            # WeatherData 是LLM臆造的表名，这里自动纠正为实际存在的 WeatherHistory
            (r'FROM\s+master\.dbo\.WeatherData\b', 'FROM master.dbo.WeatherHistory'),
            (r'FROM\s+dbo\.WeatherData\b', 'FROM dbo.WeatherHistory'),
            # CxFlowPredict数据库表（必须指定数据库前缀）
            (r'FROM\s+LineDailyFlowPrediction\b', 'FROM CxFlowPredict.dbo.LineDailyFlowPrediction'),
            (r'FROM\s+LineHourlyFlowPrediction\b', 'FROM CxFlowPredict.dbo.LineHourlyFlowPrediction'),
            (r'FROM\s+STATION_FLOW_PREDICT\b', 'FROM CxFlowPredict.dbo.STATION_FLOW_PREDICT'),
            (r'FROM\s+STATION_HOUR_PREDICT\b', 'FROM CxFlowPredict.dbo.STATION_HOUR_PREDICT'),
            # 如果已经有dbo前缀但没有数据库前缀，为CxFlowPredict表添加数据库前缀
            (r'FROM\s+dbo\.LineDailyFlowPrediction\b', 'FROM CxFlowPredict.dbo.LineDailyFlowPrediction'),
            (r'FROM\s+dbo\.LineHourlyFlowPrediction\b', 'FROM CxFlowPredict.dbo.LineHourlyFlowPrediction'),
            (r'FROM\s+dbo\.STATION_FLOW_PREDICT\b', 'FROM CxFlowPredict.dbo.STATION_FLOW_PREDICT'),
            (r'FROM\s+dbo\.STATION_HOUR_PREDICT\b', 'FROM CxFlowPredict.dbo.STATION_HOUR_PREDICT'),
        ]
        
        for pattern, replacement in patterns:
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)

        # WeatherData->WeatherHistory 字段名自动修正：
        # 参考 table_structures.txt：WeatherHistory表的准确字段是 F_DATE (int), F_TQQK (nvarchar), F_QW, F_FLFX, MAPPED_WEATHER
        # 常见错误字段名修正：
        # - RECORD_DATE -> F_DATE
        # - DATE -> F_DATE (如果LLM错误地使用了DATE字段名)
        # - WEATHER_CONDITION -> F_TQQK
        # - WEATHER -> F_TQQK (如果LLM简写)
        # 仅在涉及 WeatherData/WeatherHistory 且使用这些伪字段时进行替换
        if re.search(r'WeatherData|WeatherHistory', sql, re.IGNORECASE):
            # 修正日期字段名（多种可能的错误写法）
            sql = re.sub(r'\bRECORD_DATE\b', 'F_DATE', sql, flags=re.IGNORECASE)
            # 修正：WHERE DATE = 或 SELECT DATE as 或 , DATE, 等（但要避免误替换其他表的DATE字段）
            # 只在WeatherHistory表上下文中替换单独的DATE字段名
            sql = re.sub(r'\bDATE\s+(as\s+日期|as\s+日期字段|=\s*\d{8}|=\s*\d{4}-\d{2}-\d{2})', r'F_DATE \1', sql, flags=re.IGNORECASE)
            sql = re.sub(r'(SELECT|,)\s+DATE\s+(as|FROM|WHERE|,|$)', r'\1 F_DATE \2', sql, flags=re.IGNORECASE)
            sql = re.sub(r'WHERE\s+DATE\s*([>=<]=?)', r'WHERE F_DATE \1', sql, flags=re.IGNORECASE)
            
            # 修正天气字段名
            sql = re.sub(r'\bWEATHER_CONDITION\b', 'F_TQQK', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bWEATHER\b(?=\s+as\s+天气|\s+as\s+天气状况|\s+as\s+天气类型)', 'F_TQQK', sql, flags=re.IGNORECASE)
            
            # 确保日期格式正确：如果WHERE子句中有 F_DATE = '2024-12-03' 这样的date格式，转换为整数格式
            def fix_weather_date_format(match):
                date_str = match.group(1)
                # 如果是 'YYYY-MM-DD' 格式，转换为 YYYYMMDD 整数
                if '-' in date_str:
                    date_str = date_str.replace("'", "").replace('"', '')
                    parts = date_str.split('-')
                    if len(parts) == 3:
                        return f"F_DATE = {parts[0]}{parts[1]}{parts[2]}"
                return match.group(0)  # 如果已经是整数格式，保持不变
            
            sql = re.sub(r'F_DATE\s*=\s*(["\']?\d{4}-\d{2}-\d{2}["\']?)', fix_weather_date_format, sql, flags=re.IGNORECASE)
        
        # 修正车站表的日期字段名：车站表（历史表和预测表）都使用 SQUAD_DATE，不是 f_date
        # 参考 table_structures.txt：
        # - STATION_FLOW_HISTORY: SQUAD_DATE (int, YYYYMMDD)
        # - STATION_HOUR_HISTORY: SQUAD_DATE (int, YYYYMMDD)
        # - STATION_FLOW_PREDICT: SQUAD_DATE (date, 'YYYY-MM-DD')
        # - STATION_HOUR_PREDICT: SQUAD_DATE (date, 'YYYY-MM-DD')
        if re.search(r'STATION_FLOW|STATION_HOUR', sql, re.IGNORECASE):
            # 将车站表中的 f_date 或 F_DATE 修正为 SQUAD_DATE
            # 注意：必须在日期格式修正之前进行，因为修正后字段名是 SQUAD_DATE
            # WHERE子句中的 f_date = 或 f_date >= 等
            sql = re.sub(r'\bf_date\s*([>=<]=?)', r'SQUAD_DATE \1', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bF_DATE\s*([>=<]=?)(?=.*STATION)', r'SQUAD_DATE \1', sql, flags=re.IGNORECASE)
            # 更精确的替换：在SELECT、WHERE、GROUP BY、ORDER BY中的 f_date/F_DATE
            sql = re.sub(r'(SELECT|WHERE|GROUP BY|ORDER BY|,)\s+f_date\b', r'\1 SQUAD_DATE', sql, flags=re.IGNORECASE)
            sql = re.sub(r'(SELECT|WHERE|GROUP BY|ORDER BY|,)\s+F_DATE\b(?=.*STATION)', r'\1 SQUAD_DATE', sql, flags=re.IGNORECASE)
            # 如果SQL中包含STATION表，将独立的 f_date 或 F_DATE 替换为 SQUAD_DATE
            sql = re.sub(r'\bf_date\b(?=.*STATION)', 'SQUAD_DATE', sql, flags=re.IGNORECASE | re.DOTALL)
            sql = re.sub(r'\bF_DATE\b(?=.*STATION)', 'SQUAD_DATE', sql, flags=re.IGNORECASE | re.DOTALL)
        
        # 修正日期格式：如果查询的是 STATION_FLOW_PREDICT 或 STATION_HOUR_PREDICT 表
        # 需要将 SQUAD_DATE 的整数格式转换为 'YYYY-MM-DD' 格式
        if re.search(r'STATION_FLOW_PREDICT|STATION_HOUR_PREDICT', sql, re.IGNORECASE):
            # 匹配 SQUAD_DATE = 20220101 或 SQUAD_DATE=20220101 这样的格式
            def fix_station_date(match):
                date_int = int(match.group(1))
                # 转换为 'YYYY-MM-DD' 格式
                date_str = f"{date_int // 10000:04d}-{(date_int // 100) % 100:02d}-{date_int % 100:02d}"
                return f"SQUAD_DATE = '{date_str}'"
            
            # 匹配 SQUAD_DATE = 20220101 或 SQUAD_DATE=20220101
            sql = re.sub(r'SQUAD_DATE\s*=\s*(\d{8})', fix_station_date, sql, flags=re.IGNORECASE)
            
            # 匹配日期范围：SQUAD_DATE >= 20220101 AND SQUAD_DATE <= 20220131
            def fix_station_date_range(match):
                start_int = int(match.group(1))
                end_int = int(match.group(2))
                start_str = f"{start_int // 10000:04d}-{(start_int // 100) % 100:02d}-{start_int % 100:02d}"
                end_str = f"{end_int // 10000:04d}-{(end_int // 100) % 100:02d}-{end_int % 100:02d}"
                return f"SQUAD_DATE >= '{start_str}' AND SQUAD_DATE <= '{end_str}'"
            
            sql = re.sub(r'SQUAD_DATE\s*>=\s*(\d{8})\s+AND\s+SQUAD_DATE\s*<=\s*(\d{8})', fix_station_date_range, sql, flags=re.IGNORECASE)
            sql = re.sub(r'SQUAD_DATE\s*BETWEEN\s*(\d{8})\s+AND\s*(\d{8})', fix_station_date_range, sql, flags=re.IGNORECASE)
        
        # 修正车站名格式：确保使用 N'xxx' 格式
        # 匹配 STATION_NAME = '五一广场' 或 STATION_NAME='五一广场'，转换为 N'五一广场'
        def fix_station_name(match):
            quote_char = match.group(1)  # 引号类型 ' 或 "
            station_value = match.group(2)  # 车站名（不含引号）
            # 如果已经是 N'xxx' 格式，保持不变
            full_match = match.group(0)
            if 'N' + quote_char in full_match or 'N"' in full_match:
                return full_match
            # 否则添加 N 前缀
            return f"STATION_NAME = N{quote_char}{station_value}{quote_char}"
        
        # 匹配 STATION_NAME = 'xxx' 或 STATION_NAME='xxx' 或 STATION_NAME = "xxx"
        # 使用非贪婪匹配，避免匹配到其他内容
        sql = re.sub(r'STATION_NAME\s*=\s*(["\'])([^"\']+?)\1', fix_station_name, sql, flags=re.IGNORECASE)
        
        # 修正LineDailyFlowPrediction和LineHourlyFlowPrediction表的字段名和日期格式
        # 参考 table_structures.txt：
        # - LineDailyFlowPrediction: F_PKLCOUNT (float), F_DATE (int, YYYYMMDD)
        # - LineHourlyFlowPrediction: F_PKLCOUNT (float), F_DATE (int, YYYYMMDD)
        if re.search(r'LineDailyFlowPrediction|LineHourlyFlowPrediction', sql, re.IGNORECASE):
            # 修正字段名：F_KLCCOUNT -> F_PKLCOUNT（常见错误）
            sql = re.sub(r'\bF_KLCCOUNT\b', 'F_PKLCOUNT', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bf_klcount\b(?=.*LineDailyFlowPrediction|.*LineHourlyFlowPrediction)', 'F_PKLCOUNT', sql, flags=re.IGNORECASE | re.DOTALL)
            
            # 修正日期格式：F_DATE = 'YYYY-MM-DD' -> F_DATE = YYYYMMDD（整数格式）
            def fix_prediction_date_format(match):
                date_str = match.group(1)
                # 如果是 'YYYY-MM-DD' 格式，转换为 YYYYMMDD 整数
                if '-' in date_str:
                    date_str = date_str.replace("'", "").replace('"', '')
                    parts = date_str.split('-')
                    if len(parts) == 3:
                        return f"F_DATE = {parts[0]}{parts[1]}{parts[2]}"
                return match.group(0)  # 如果已经是整数格式，保持不变
            
            # 匹配 F_DATE = '2025-12-05' 或 F_DATE='2025-12-05' 或 F_DATE = "2025-12-05"
            sql = re.sub(r'F_DATE\s*=\s*(["\']?\d{4}-\d{2}-\d{2}["\']?)', fix_prediction_date_format, sql, flags=re.IGNORECASE)
            
            # 匹配日期范围：F_DATE >= '2025-12-01' AND F_DATE <= '2025-12-05'
            def fix_prediction_date_range(match):
                start_str = match.group(1).replace("'", "").replace('"', '')
                end_str = match.group(2).replace("'", "").replace('"', '')
                start_parts = start_str.split('-')
                end_parts = end_str.split('-')
                if len(start_parts) == 3 and len(end_parts) == 3:
                    start_int = f"{start_parts[0]}{start_parts[1]}{start_parts[2]}"
                    end_int = f"{end_parts[0]}{end_parts[1]}{end_parts[2]}"
                    return f"F_DATE >= {start_int} AND F_DATE <= {end_int}"
                return match.group(0)
            
            sql = re.sub(r'F_DATE\s*>=\s*(["\']?\d{4}-\d{2}-\d{2}["\']?)\s+AND\s+F_DATE\s*<=\s*(["\']?\d{4}-\d{2}-\d{2}["\']?)', fix_prediction_date_range, sql, flags=re.IGNORECASE)
            sql = re.sub(r'F_DATE\s*BETWEEN\s*(["\']?\d{4}-\d{2}-\d{2}["\']?)\s+AND\s+(["\']?\d{4}-\d{2}-\d{2}["\']?)', fix_prediction_date_range, sql, flags=re.IGNORECASE)
            
            logger.info(f"已修正LineDailyFlowPrediction/LineHourlyFlowPrediction表的字段名和日期格式")
        
        # 使用表结构验证器进行全面的验证和修正
        try:
            from app.utils.table_structure_validator import get_validator
            validator = get_validator()
            sql, fixes = validator.validate_and_fix_sql(sql)
            
            if fixes:
                logger.info(f"SQL验证修正完成，共修正 {len(fixes)} 处：")
                for fix in fixes:
                    logger.info(f"  - {fix}")
        except Exception as e:
            logger.warning(f"表结构验证器执行失败: {e}，继续使用原有修正逻辑")
        
        # 修正日期函数错误：对于int类型的f_date字段，不能使用DATEADD等日期函数
        # 检测并记录警告
        if re.search(r'DATEADD|DATEDIFF|DATEPART', sql, re.IGNORECASE) and re.search(r'\bf_date\b', sql, re.IGNORECASE):
            logger.warning("检测到SQL中使用了日期函数操作int类型的f_date字段，这会导致错误。SQL: " + sql[:200])
            # 这里可以添加自动修复逻辑，但比较复杂，暂时只记录警告
            # 更好的方法是在提示词中明确说明不要使用日期函数
        
        return sql.strip()
    
    def _parse_response(self, response: str, question: str) -> Dict:
        """解析LLM响应，提取SQL语句和思考过程"""
        # 清理响应文本
        original_response = response.strip()
        response = original_response
        
        # 先提取思考过程（在提取SQL之前）
        thinking_process = self._extract_thinking_process(original_response, question, "")
        
        # 移除markdown代码块标记（如果有）
        response = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # 移除<think>标签及其内容（如果存在）
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = response.strip()
        
        # 提取SQL语句 - 查找SELECT开头的语句
        sql = ""
        lines = response.split('\n')
        sql_lines = []
        found_select = False
        
        for line in lines:
            line_stripped = line.strip()
            # 跳过注释和空行
            if not line_stripped or line_stripped.startswith('--'):
                continue
            
            # 如果找到SELECT，开始收集SQL
            if line_stripped.upper().startswith('SELECT'):
                found_select = True
                sql_lines.append(line_stripped)
            elif found_select:
                # 继续收集SQL直到遇到分号或非SQL内容
                if line_stripped.endswith(';'):
                    sql_lines.append(line_stripped.rstrip(';'))
                    break
                elif any(keyword in line_stripped.upper() for keyword in ['FROM', 'WHERE', 'ORDER', 'GROUP', 'HAVING', 'JOIN', 'UNION']):
                    sql_lines.append(line_stripped)
                elif line_stripped and not line_stripped.startswith('```'):
                    # 可能是SQL的继续部分
                    sql_lines.append(line_stripped)
                else:
                    break
        
        sql = ' '.join(sql_lines).strip()
        
        # 如果还是没有找到SQL，尝试从整个响应中提取
        if not sql or not sql.upper().startswith('SELECT'):
            # 尝试匹配SELECT语句
            select_match = re.search(r'(SELECT\s+.*?)(?=\n\n|\n[A-Z]|$)', response, re.DOTALL | re.IGNORECASE)
            if select_match:
                sql = select_match.group(1).strip()
                # 清理SQL
                sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
                sql = re.sub(r'```\s*', '', sql)
                sql = sql.strip()
                if sql.endswith(';'):
                    sql = sql[:-1]
        
        # 后处理SQL，确保格式正确
        if sql:
            sql = self._post_process_sql(sql)
        
        # 如果还是没有SQL，记录详细信息并返回None
        if not sql or not sql.upper().startswith('SELECT'):
            logger.warning("无法从LLM响应中提取SQL")
            resp_preview = original_response[:500] if len(original_response) > 500 else original_response
            clean_preview = response[:500] if len(response) > 500 else response
            logger.warning(f"响应长度: {len(original_response)}, 响应前500字符: {resp_preview}")
            logger.warning(f"清理后的响应前500字符: {clean_preview}")
            # 返回None，让调用方决定如何处理
            return None
        
        # 简单的意图识别（基于问题关键词）
        intent = self._classify_intent(question)
        
        # 简单的实体抽取（基于问题关键词）
        entities = self._extract_entities(question)
        
        # 如果思考过程为空，使用默认的
        if not thinking_process:
            thinking_process = self._generate_default_thinking(question, sql)
        
        return {
            "sql": sql,
            "intent": intent,
            "entities": entities,
            "raw_response": original_response,
            "thinking_process": thinking_process
        }
    
    def _extract_thinking_process(self, response: str, question: str, sql: str) -> str:
        """提取LLM的思考过程"""
        thinking_parts = []
        
        # 1. 尝试提取<think>标签中的内容
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking_parts.append(think_match.group(1).strip())
        
        # 2. 尝试提取其他格式的思考过程（在SQL之前的内容）
        if not thinking_parts:
            lines = response.split('\n')
            thinking_lines = []
            found_sql = False
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                line_lower = line_stripped.lower()
                
                # 如果遇到SELECT，停止收集思考过程
                if line_stripped.upper().startswith('SELECT'):
                    found_sql = True
                    break
                
                # 跳过markdown代码块标记
                if line_stripped.startswith('```'):
                    continue
                
                # 收集思考内容（在SQL之前的所有内容）
                if line_stripped and not found_sql:
                    # 检查是否是思考相关的内容
                    if any(kw in line_lower for kw in ["思考", "分析", "理解", "考虑", "think", "analyze", "问题", "需要", "首先", "然后"]):
                        thinking_lines.append(line_stripped)
                    elif thinking_lines:  # 如果已经开始收集，继续收集直到遇到SQL
                        thinking_lines.append(line_stripped)
            
            if thinking_lines:
                thinking_text = '\n'.join(thinking_lines).strip()
                # 清理一些明显的SQL片段
                thinking_text = re.sub(r'SELECT\s+.*', '', thinking_text, flags=re.DOTALL | re.IGNORECASE)
                if thinking_text:
                    thinking_parts.append(thinking_text)
        
        # 3. 如果没有提取到，生成默认的思考过程
        if not thinking_parts:
            thinking_parts.append(self._generate_default_thinking(question, sql if sql else ""))
        
        return '\n\n'.join(thinking_parts)
    
    def _generate_default_thinking(self, question: str, sql: str) -> str:
        """生成默认的思考过程描述"""
        thinking = []
        thinking.append(f"📝 分析问题: {question}")
        thinking.append("")
        thinking.append("🔍 理解步骤:")
        thinking.append("1. 识别查询意图（线路查询/车站查询/预测查询等）")
        thinking.append("2. 提取关键实体（线路名、车站名、日期、指标等）")
        thinking.append("3. 确定查询的表和字段")
        thinking.append("4. 构建WHERE条件")
        thinking.append("")
        thinking.append("💡 SQL生成:")
        thinking.append(f"根据分析结果，生成SQL查询语句")
        thinking.append("")
        thinking.append(f"✅ 生成的SQL:")
        thinking.append(sql[:200] + ("..." if len(sql) > 200 else ""))
        return '\n'.join(thinking)
    
    def _classify_intent(self, question: str) -> str:
        """简单的意图分类（作为fallback）"""
        question_lower = question.lower()
        
        if any(kw in question for kw in ["预测", "预计", "未来"]):
            return "prediction_query"
        elif any(kw in question for kw in ["对比", "比较", "相比"]):
            return "comparison_query"
        elif any(kw in question for kw in ["统计", "排名", "最高", "最低"]):
            return "statistics_query"
        elif any(kw in question for kw in ["趋势", "变化", "走势"]):
            return "trend_query"
        elif any(kw in question for kw in ["车站", "站"]):
            return "station_flow_query"
        else:
            return "line_flow_query"
    
    def _extract_entities(self, question: str) -> Dict:
        """简单的实体抽取（作为fallback）"""
        entities = {}
        
        # 抽取线路
        import re
        line_match = re.search(r'(\d+)号线', question)
        if line_match:
            entities["line"] = line_match.group(0)
            entities["line_id"] = line_match.group(1)
        
        # 抽取日期关键词
        if "昨天" in question:
            from datetime import datetime, timedelta
            yesterday = datetime.now() - timedelta(days=1)
            entities["date"] = yesterday
            entities["date_int"] = int(yesterday.strftime("%Y%m%d"))
        elif "今天" in question:
            from datetime import datetime
            today = datetime.now()
            entities["date"] = today
            entities["date_int"] = int(today.strftime("%Y%m%d"))
        
        return entities

