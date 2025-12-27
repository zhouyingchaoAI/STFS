"""
实体抽取器
"""
from typing import Dict, Optional, List
import re
import yaml
from pathlib import Path

from app.utils.date_utils import parse_date_expression, date_to_int
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class EntityExtractor:
    """实体抽取器"""
    
    def __init__(self):
        # 加载车站线路映射
        self.station_mapping = self._load_station_mapping()
        
        # 线路名称映射
        self.line_mapping = {
            "1号线": "1", "一号线": "1", "1线": "1",
            "2号线": "2", "二号线": "2", "2线": "2",
            "3号线": "3", "三号线": "3", "3线": "3",
            "4号线": "4", "四号线": "4", "4线": "4",
            "5号线": "5", "五号线": "5", "5线": "5",
            "6号线": "6", "六号线": "6", "6线": "6",
        }
        
        # 指标关键词映射
        self.metric_keywords = {
            "客流量": ["客流量", "客运量", "客流", "passenger"],
            "进站量": ["进站量", "进站", "entry"],
            "出站量": ["出站量", "出站", "exit"],
            "换乘量": ["换乘量", "换乘", "change", "transfer"],
            "乘降量": ["乘降量", "乘降", "flow"]
        }
    
    def _load_station_mapping(self) -> Dict:
        """加载车站映射"""
        try:
            # 尝试多个可能的路径
            possible_paths = [
                Path(__file__).parent.parent.parent.parent.parent / "station_line_mapping.yaml",
                Path(__file__).parent.parent.parent.parent / "station_line_mapping.yaml",
                Path("station_line_mapping.yaml"),
            ]
            
            for mapping_file in possible_paths:
                if mapping_file.exists():
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load station mapping: {e}")
        return {}
    
    def extract(self, question: str) -> Dict:
        """
        从问题中抽取实体
        
        Args:
            question: 自然语言问题
            
        Returns:
            实体字典
        """

        entities = {
            "line": None,
            "line_id": None,
            "station": None,
            "station_id": None,
            "line_id": None,
            "date": None,
            "date_int": None,
            "date_range": None,
            "metric": None,
            "time_hour": None
        }
        
        # 抽取线路
        line_info = self._extract_line(question)
        if line_info:
            entities["line"] = line_info.get("name")
            entities["line_id"] = line_info.get("id")
        
        # 抽取车站
        station_info = self._extract_station(question)
        if station_info:
            entities["station"] = station_info.get("name")
            entities["station_id"] = station_info.get("station_id")
            entities["line_id"] = station_info.get("line_id")
        
        # 抽取日期
        date_info = self._extract_date(question)
        if date_info:
            entities["date"] = date_info.get("date")
            entities["date_int"] = date_info.get("date_int")
            entities["date_range"] = date_info.get("date_range")
        
        # 抽取时间（小时）
        hour = self._extract_hour(question)
        if hour is not None:
            entities["time_hour"] = hour
        
        # 抽取指标
        metric = self._extract_metric(question)
        if metric:
            entities["metric"] = metric
        
        return entities
    
    def _extract_line(self, question: str) -> Optional[Dict]:
        """抽取线路信息"""
        for line_name, line_id in self.line_mapping.items():
            if line_name in question:
                return {"name": line_name, "id": line_id}
        return None
    
    def _extract_station(self, question: str) -> Optional[Dict]:
        """
        抽取车站信息
        注意：station_line_mapping.yaml 中的车站名通常不带"站"字（如"五一广场"），
        但用户可能输入"五一广场站"，需要去掉"站"字后匹配。
        匹配规则：
        1. 直接匹配：station_name 在问题中（如"五一广场"）
        2. 带"站"匹配：station_name + "站" 在问题中（如"五一广场站"）
        3. 去掉"站"匹配：问题中的"xxx站"去掉"站"后匹配 station_name
        """
        # 先尝试精确匹配（避免部分匹配问题）
        for station_name, station_info in self.station_mapping.items():
            # 1. 直接匹配：station_name 在问题中
            if station_name in question:
                if isinstance(station_info, list) and len(station_info) > 0:
                    info = station_info[0]
                    return {
                        "name": station_name,  # 返回 yaml 中的原始名称
                        "line_id": info.get("LINE_ID", "").strip(),
                        "station_id": info.get("STATION_ID", "").strip()
                    }
            
            # 2. 带"站"匹配：station_name + "站" 在问题中
            station_name_with_station = station_name + "站"
            if station_name_with_station in question:
                if isinstance(station_info, list) and len(station_info) > 0:
                    info = station_info[0]
                    return {
                        "name": station_name,  # 返回 yaml 中的原始名称（不带"站"）
                        "line_id": info.get("LINE_ID", "").strip(),
                        "station_id": info.get("STATION_ID", "").strip()
                    }
        
        # 3. 去掉"站"匹配：问题中的"xxx站"去掉"站"后匹配 station_name
        # 使用正则表达式匹配"xxx站"模式
        import re
        station_pattern = re.compile(r'([^站]+)站')
        matches = station_pattern.findall(question)
        for match in matches:
            # 去掉可能的空格
            potential_station = match.strip()
            if potential_station in self.station_mapping:
                station_info = self.station_mapping[potential_station]
                if isinstance(station_info, list) and len(station_info) > 0:
                    info = station_info[0]
                    return {
                        "name": potential_station,  # 返回 yaml 中的原始名称
                        "line_id": info.get("LINE_ID", "").strip(),
                        "station_id": info.get("STATION_ID", "").strip()
                    }
        
        return None
    
    def _extract_date(self, question: str) -> Optional[Dict]:
        """抽取日期信息"""
        from datetime import datetime
        from app.utils.date_utils import parse_date_range
        
        # 先尝试解析日期范围（如"最近7天"）
        date_range = parse_date_range(question)
        if date_range:
            start_date, end_date = date_range
            return {
                "date": start_date,
                "date_int": date_to_int(start_date),
                "date_range": {
                    "start": start_date,
                    "end": end_date
                }
            }
        
        # 尝试解析日期表达式（如"昨天"、"今天"）
        date_obj = parse_date_expression(question)
        if date_obj:
            return {
                "date": date_obj,
                "date_int": date_to_int(date_obj)
            }
        
        # 尝试匹配日期格式 YYYY-MM-DD 或 YYYYMMDD
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{4}\d{2}\d{2})',
            r'(\d{4})年(\d{1,2})月(\d{1,2})日'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, question)
            if match:
                try:
                    if '年' in pattern:
                        year, month, day = match.groups()
                        date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    else:
                        date_str = match.group(1)
                        if '-' in date_str:
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        else:
                            date_obj = datetime.strptime(date_str, "%Y%m%d")
                    
                    return {
                        "date": date_obj,
                        "date_int": date_to_int(date_obj)
                    }
                except ValueError:
                    continue
        
        return None
    
    def _extract_hour(self, question: str) -> Optional[int]:
        """抽取小时信息"""
        # 匹配 "X点"、"X时"、"X:00" 等格式
        patterns = [
            r'(\d{1,2})点',
            r'(\d{1,2})时',
            r'(\d{1,2}):00',
            r'第(\d{1,2})小时'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                hour = int(match.group(1))
                if 0 <= hour <= 23:
                    return hour
        
        return None
    
    def _extract_metric(self, question: str) -> Optional[str]:
        """抽取指标信息"""
        for metric, keywords in self.metric_keywords.items():
            if any(keyword in question for keyword in keywords):
                return metric
        return "客流量"  # 默认返回客流量

