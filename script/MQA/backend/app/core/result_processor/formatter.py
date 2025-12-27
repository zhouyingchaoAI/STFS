"""
结果格式化器
"""
from typing import List, Dict, Any
from datetime import datetime
from decimal import Decimal

from app.utils.date_utils import int_to_date
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResultFormatter:
    """结果格式化器"""
    
    def format(self, query_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        格式化查询结果
        
        Args:
            query_result: 查询结果字典
            
        Returns:
            格式化后的结果列表
        """
        data = query_result.get("data", [])
        formatted_data = []
        
        for row in data:
            formatted_row = {}
            for key, value in row.items():
                # 格式化日期字段
                if "日期" in key or "date" in key.lower():
                    if isinstance(value, int) and len(str(value)) == 8:
                        try:
                            date_obj = int_to_date(value)
                            formatted_row[key] = date_obj.strftime("%Y-%m-%d")
                        except:
                            formatted_row[key] = value
                    else:
                        formatted_row[key] = value
                # 格式化数值字段（包括Decimal类型）
                elif isinstance(value, Decimal):
                    # Decimal类型转换为float或int
                    if value % 1 == 0:
                        formatted_row[key] = int(value)
                    else:
                        formatted_row[key] = round(float(value), 2)
                elif isinstance(value, (float, int)) and "量" in key:
                    formatted_row[key] = round(float(value), 2) if value else 0
                else:
                    formatted_row[key] = value
            
            formatted_data.append(formatted_row)
        
        return formatted_data

