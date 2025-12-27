"""
SQL生成器
注意：不同表的日期字段格式不同
- 历史表：int类型，YYYYMMDD格式（如 20220101）
- 车站预测表（STATION_FLOW_PREDICT, STATION_HOUR_PREDICT）：date类型，'YYYY-MM-DD'格式（如 '2025-11-20'）
- 线路预测表（LineDailyFlowPrediction, LineHourlyFlowPrediction）：int类型，YYYYMMDD格式（如 20250428）
"""
from typing import Dict, Optional
from datetime import datetime

from app.utils.date_utils import date_to_int, int_to_date
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class SQLGenerator:
    """SQL生成器"""
    
    def __init__(self):
        # 指标字段映射
        self.metric_mapping = {
            "客流量": "f_klcount",
            "进站量": "entry_num",
            "出站量": "exit_num",
            "换乘量": "change_num",
            "乘降量": "flow_num"
        }
        
        # 指标中文名映射
        self.metric_cn_mapping = {
            "客流量": "客流量",
            "进站量": "进站量",
            "出站量": "出站量",
            "换乘量": "换乘量",
            "乘降量": "乘降量"
        }
    
    def generate(self, intent: str, entities: Dict) -> Optional[str]:
        """
        根据意图和实体生成SQL
        
        Args:
            intent: 查询意图
            entities: 实体字典
            
        Returns:
            SQL查询语句
        """
        try:
            if intent == "line_flow_query":
                return self._generate_line_flow_sql(entities)
            elif intent == "station_flow_query":
                return self._generate_station_flow_sql(entities)
            elif intent == "prediction_query":
                return self._generate_prediction_sql(entities)
            elif intent == "comparison_query":
                return self._generate_comparison_sql(entities)
            elif intent == "statistics_query":
                return self._generate_statistics_sql(entities)
            elif intent == "trend_query":
                return self._generate_trend_sql(entities)
            else:
                # 默认线路查询
                return self._generate_line_flow_sql(entities)
        except Exception as e:
            logger.error(f"SQL generation error: {e}", exc_info=True)
            return None
    
    def _generate_line_flow_sql(self, entities: Dict) -> str:
        """生成线路客流查询SQL"""
        metric = entities.get("metric", "客流量")
        metric_field = self.metric_mapping.get(metric, "f_klcount")
        metric_cn = self.metric_cn_mapping.get(metric, "客流量")
        
        # 判断是日客流还是小时客流
        if entities.get("time_hour") is not None:
            # 小时客流
            table = "dbo.LineHourlyFlowHistory"
            hour_condition = f"AND f_hour = {entities['time_hour']}"
            hour_select = "f_hour as 小时,"
            order_by = "ORDER BY f_date DESC, f_hour DESC"
        else:
            # 日客流
            table = "dbo.LineDailyFlowHistory"
            hour_condition = ""
            hour_select = ""
            order_by = "ORDER BY f_date DESC"
        
        # 构建WHERE条件
        conditions = []
        
        # 线路条件
        line_name = entities.get("line") or ""
        # 检查是否查询所有线路
        if line_name and ("所有" in line_name or "全部" in line_name or "all" in line_name.lower()):
            # 不添加线路条件，查询所有线路
            pass
        elif entities.get("line_id"):
            conditions.append(f"f_lineno = {entities['line_id']}")
        elif entities.get("line") and line_name:
            # 如果只有线路名，尝试从线路名提取
            if "1" in line_name or "一" in line_name:
                conditions.append("f_lineno = 1")
            elif "2" in line_name or "二" in line_name:
                conditions.append("f_lineno = 2")
            elif "3" in line_name or "三" in line_name:
                conditions.append("f_lineno = 3")
            elif "4" in line_name or "四" in line_name:
                conditions.append("f_lineno = 4")
            elif "5" in line_name or "五" in line_name:
                conditions.append("f_lineno = 5")
            elif "6" in line_name or "六" in line_name:
                conditions.append("f_lineno = 6")
            # 可以继续扩展其他线路
        
        # 日期条件
        if entities.get("date_int"):
            conditions.append(f"f_date = {entities['date_int']}")
        elif entities.get("date_range"):
            # 日期范围查询
            date_range = entities["date_range"]
            if isinstance(date_range, dict) and "start" in date_range and "end" in date_range:
                start_int = date_to_int(date_range["start"])
                end_int = date_to_int(date_range["end"])
                conditions.append(f"f_date >= {start_int} AND f_date <= {end_int}")
        else:
            # 检查是否有"最近"、"今天"等关键词
            # 如果没有明确日期，默认查询最近7天的数据
            from datetime import datetime, timedelta
            today = datetime.now()
            yesterday = today - timedelta(days=1)
            # 默认查询昨天的数据（更可能有多数据）
            date_int = date_to_int(yesterday)
            conditions.append(f"f_date = {date_int}")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"""
        SELECT 
            f_date as 日期,
            {hour_select}
            f_linename as 线路名,
            f_klcount as 客流量,
            entry_num as 进站量,
            exit_num as 出站量,
            change_num as 换乘量,
            flow_num as 乘降量
        FROM {table}
        WHERE {where_clause}
        {hour_condition}
        {order_by}
        """
        
        return sql.strip()
    
    def _generate_station_flow_sql(self, entities: Dict) -> str:
        """
        生成车站客流查询SQL
        注意：STATION_FLOW_HISTORY 和 STATION_HOUR_HISTORY 表的 SQUAD_DATE 是 int 类型，使用 YYYYMMDD 格式
        参考 db_utils.py 的处理模式：同一时间同一车站会有多条数据，需要进行合并（数值字段求和）
        """
        metric = entities.get("metric", "客流量")
        metric_field = self.metric_mapping.get(metric, "PASSENGER_NUM")
        
        # 判断是日客流还是小时客流
        if entities.get("time_hour") is not None:
            table = "dbo.STATION_HOUR_HISTORY"
            # TIME_SECTION_ID 是 char 类型，需要字符串格式，可能带空格
            hour_value = f"{entities['time_hour']:02d}"
            hour_condition = f"AND TIME_SECTION_ID = '{hour_value}'"
            hour_select = "TIME_SECTION_ID as 小时,"
            # 小时客流：按 SQUAD_DATE, TIME_SECTION_ID, STATION_NAME 分组
            group_by_fields = "SQUAD_DATE, TIME_SECTION_ID, STATION_NAME"
        else:
            table = "dbo.STATION_FLOW_HISTORY"
            hour_condition = ""
            hour_select = ""
            # 日客流：按 SQUAD_DATE, STATION_NAME 分组
            group_by_fields = "SQUAD_DATE, STATION_NAME"
        
        conditions = []
        
        # 车站条件（支持按车站名查询，参考 db_utils.py 的处理）
        if entities.get("station_name"):
            # 使用 N'xxx' 格式处理中文车站名
            station_name = str(entities['station_name']).strip()
            conditions.append(f"STATION_NAME = N'{station_name}'")
        elif entities.get("station_id") and entities.get("line_id"):
            # LINE_ID 和 STATION_ID 是 char 类型，可能需要处理空格
            line_id = str(entities['line_id']).strip()
            station_id = str(entities['station_id']).strip()
            conditions.append(f"LINE_ID = '{line_id}'")
            conditions.append(f"STATION_ID = '{station_id}'")
        
        # 日期条件：STATION_FLOW_HISTORY 和 STATION_HOUR_HISTORY 使用 int 类型，YYYYMMDD 格式
        if entities.get("date_int"):
            conditions.append(f"SQUAD_DATE = {entities['date_int']}")
        elif entities.get("date_range"):
            # 日期范围查询
            date_range = entities["date_range"]
            if isinstance(date_range, dict) and "start" in date_range and "end" in date_range:
                start_int = date_to_int(date_range["start"])
                end_int = date_to_int(date_range["end"])
                conditions.append(f"SQUAD_DATE >= {start_int} AND SQUAD_DATE <= {end_int}")
        else:
            from datetime import timedelta
            yesterday = datetime.now() - timedelta(days=1)
            date_int = date_to_int(yesterday)
            conditions.append(f"SQUAD_DATE = {date_int}")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # 参考 db_utils.py：对数值字段使用 SUM，对其他字段使用 MIN 或取第一个值
        # 数值字段需要求和合并
        sum_fields = {
            "ENTRY_NUM": "进站量",
            "EXIT_NUM": "出站量",
            "CHANGE_NUM": "换乘量",
            "PASSENGER_NUM": "客运量",
            "FLOW_NUM": "乘降量"
        }
        
        # 构建 SELECT 字段（数值字段使用 SUM，其他字段使用 MIN 或取第一个）
        select_fields = []
        select_fields.append("SQUAD_DATE as 日期")
        if hour_select:
            select_fields.append(hour_select.rstrip(','))
        
        # 对于分组字段，使用 MIN 或取第一个值
        if entities.get("time_hour"):
            select_fields.append("MIN(TIME_SECTION_ID) as 小时")
        select_fields.append("MIN(LINE_ID) as 线路号")
        select_fields.append("MIN(STATION_ID) as 车站号")
        select_fields.append("MIN(STATION_NAME) as 车站名")
        
        # 数值字段使用 SUM 合并
        for field, alias in sum_fields.items():
            select_fields.append(f"SUM({field}) as {alias}")
        
        select_clause = ",\n            ".join(select_fields)
        
        sql = f"""
        SELECT 
            {select_clause}
        FROM {table}
        WHERE {where_clause}
        {hour_condition}
        GROUP BY {group_by_fields}
        ORDER BY SQUAD_DATE DESC
        """
        
        return sql.strip()
    
    def _generate_prediction_sql(self, entities: Dict) -> str:
        """生成预测数据查询SQL"""
        # 类似历史查询，但使用预测表
        if entities.get("station_id"):
            return self._generate_station_prediction_sql(entities)
        else:
            return self._generate_line_prediction_sql(entities)
    
    def _generate_line_prediction_sql(self, entities: Dict) -> str:
        """生成线路预测SQL"""
        table = "dbo.LineDailyFlowPrediction"
        
        conditions = []
        
        if entities.get("line_id"):
            conditions.append(f"F_LINENO = {entities['line_id']}")
        
        if entities.get("date_int"):
            conditions.append(f"F_DATE = {entities['date_int']}")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"""
        SELECT 
            F_DATE as 日期,
            F_LINENAME as 线路名,
            F_PKLCOUNT as 预测客流量,
            entry_num as 进站量,
            exit_num as 出站量,
            change_num as 换乘量,
            flow_num as 乘降量,
            PREDICT_DATE as 预测日期,
            PREDICT_WEATHER as 预测天气
        FROM {table}
        WHERE {where_clause}
        ORDER BY F_DATE DESC
        """
        
        return sql.strip()
    
    def _generate_station_prediction_sql(self, entities: Dict) -> str:
        """
        生成车站预测SQL
        注意：STATION_FLOW_PREDICT 和 STATION_HOUR_PREDICT 表的 SQUAD_DATE 是 date 类型，需要使用 'YYYY-MM-DD' 格式
        参考 db_utils.py 的处理模式：同一时间同一车站会有多条数据，需要进行合并（数值字段求和）
        """
        # 判断是日预测还是小时预测
        if entities.get("time_hour") is not None:
            table = "CxFlowPredict.dbo.STATION_HOUR_PREDICT"
            hour_condition = f"AND TIME_SECTION_ID = '{entities['time_hour']:02d}'"
            hour_select = "TIME_SECTION_ID as 小时,"
            # 小时客流：按 SQUAD_DATE, TIME_SECTION_ID, STATION_NAME 分组
            group_by_fields = "SQUAD_DATE, TIME_SECTION_ID, STATION_NAME"
        else:
            table = "CxFlowPredict.dbo.STATION_FLOW_PREDICT"
            hour_condition = ""
            hour_select = ""
            # 日客流：按 SQUAD_DATE, STATION_NAME 分组
            group_by_fields = "SQUAD_DATE, STATION_NAME"
        
        conditions = []
        
        # 车站条件（支持按车站名查询）
        if entities.get("station_name"):
            station_name = str(entities['station_name']).strip()
            conditions.append(f"STATION_NAME = N'{station_name}'")
        elif entities.get("station_id") and entities.get("line_id"):
            conditions.append(f"LINE_ID = '{entities['line_id']}'")
            conditions.append(f"STATION_ID = '{entities['station_id']}'")
        
        # 日期条件：STATION_FLOW_PREDICT 和 STATION_HOUR_PREDICT 使用 date 类型，需要 'YYYY-MM-DD' 格式
        if entities.get("date"):
            date_str = entities['date'].strftime('%Y-%m-%d')
            conditions.append(f"SQUAD_DATE = '{date_str}'")
        elif entities.get("date_int"):
            # 将整数日期转换为 'YYYY-MM-DD' 格式
            date_obj = int_to_date(entities['date_int'])
            date_str = date_obj.strftime("%Y-%m-%d")
            conditions.append(f"SQUAD_DATE = '{date_str}'")
        elif entities.get("date_range"):
            # 日期范围查询
            date_range = entities["date_range"]
            if isinstance(date_range, dict) and "start" in date_range and "end" in date_range:
                start_date = date_range["start"]
                end_date = date_range["end"]
                # 确保是 datetime 对象
                if not isinstance(start_date, datetime):
                    if isinstance(start_date, str):
                        start_date = datetime.strptime(start_date, "%Y-%m-%d")
                    else:
                        start_date = int_to_date(date_to_int(start_date))
                if not isinstance(end_date, datetime):
                    if isinstance(end_date, str):
                        end_date = datetime.strptime(end_date, "%Y-%m-%d")
                    else:
                        end_date = int_to_date(date_to_int(end_date))
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                conditions.append(f"SQUAD_DATE >= '{start_str}' AND SQUAD_DATE <= '{end_str}'")
        else:
            # 默认查询昨天的数据
            from datetime import timedelta
            yesterday = datetime.now() - timedelta(days=1)
            date_str = yesterday.strftime("%Y-%m-%d")
            conditions.append(f"SQUAD_DATE = '{date_str}'")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # 参考 db_utils.py：对数值字段使用 SUM，对其他字段使用 MIN 或取第一个值
        # 数值字段需要求和合并
        sum_fields = {
            "ENTRY_NUM": "进站量",
            "EXIT_NUM": "出站量",
            "CHANGE_NUM": "换乘量",
            "PASSENGER_NUM": "客运量",
            "FLOW_NUM": "乘降量"
        }
        
        # 构建 SELECT 字段（数值字段使用 SUM，其他字段使用 MIN 或取第一个）
        select_fields = []
        select_fields.append("SQUAD_DATE as 日期")
        if hour_select:
            select_fields.append(hour_select.rstrip(','))
        
        # 对于分组字段，使用 MIN 或取第一个值
        if entities.get("time_hour"):
            select_fields.append("MIN(TIME_SECTION_ID) as 小时")
        select_fields.append("MIN(LINE_ID) as 线路号")
        select_fields.append("MIN(STATION_ID) as 车站号")
        select_fields.append("MIN(STATION_NAME) as 车站名")
        
        # 数值字段使用 SUM 合并
        for field, alias in sum_fields.items():
            select_fields.append(f"SUM({field}) as {alias}")
        
        # PREDICT_DATE 也使用 MIN（取第一个值）
        select_fields.append("MIN(PREDICT_DATE) as 预测日期")
        
        select_clause = ",\n            ".join(select_fields)
        
        sql = f"""
        SELECT 
            {select_clause}
        FROM {table}
        WHERE {where_clause}
        {hour_condition}
        GROUP BY {group_by_fields}
        ORDER BY SQUAD_DATE DESC
        """
        
        return sql.strip()
    
    def _generate_comparison_sql(self, entities: Dict) -> str:
        """生成对比查询SQL"""
        # 简化实现，可以后续扩展
        return self._generate_line_flow_sql(entities)
    
    def _generate_statistics_sql(self, entities: Dict) -> str:
        """生成统计查询SQL"""
        # 简化实现，可以后续扩展
        return self._generate_line_flow_sql(entities)
    
    def _generate_trend_sql(self, entities: Dict) -> str:
        """生成趋势查询SQL"""
        # 简化实现，可以后续扩展
        return self._generate_line_flow_sql(entities)

