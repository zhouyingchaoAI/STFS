"""
图表配置生成器
"""
from typing import List, Dict, Any, Optional

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChartGenerator:
    """图表配置生成器"""
    
    def generate(self, data: List[Dict[str, Any]], intent: str = None) -> Optional[Dict[str, Any]]:
        """
        根据数据和意图生成图表配置
        同时生成曲线图（默认显示）和柱状图（可折叠）
        
        Args:
            data: 数据列表
            intent: 查询意图
            
        Returns:
            包含曲线图和柱状图的配置字典
        """
        if not data:
            return None
        
        # 生成曲线图（默认显示）
        line_chart = self._generate_line_chart(data)
        
        # 生成柱状图（可折叠显示）
        bar_chart = self._generate_bar_chart(data)
        
        # 如果两种图表都生成成功，返回组合配置
        if line_chart and bar_chart:
            return {
                "line_chart": line_chart,  # 曲线图（默认显示）
                "bar_chart": bar_chart,    # 柱状图（可折叠）
                "default_chart": "line"     # 默认显示曲线图
            }
        elif line_chart:
            return {
                "line_chart": line_chart,
                "default_chart": "line"
            }
        elif bar_chart:
            return {
                "bar_chart": bar_chart,
                "default_chart": "bar"
            }
        else:
            return None
    
    def _generate_line_chart(self, data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """生成曲线图配置（折线图，平滑曲线）"""
        # 查找日期字段和数值字段
        date_field = None
        value_fields = []
        
        for key in data[0].keys():
            if "日期" in key or "date" in key.lower() or "DATE" in key:
                date_field = key
            elif "量" in key or "数" in key or isinstance(data[0].get(key), (int, float)):
                # 排除非数值字段
                if key not in ["线路号", "车站号", "ID", "id"]:
                    value_fields.append(key)
        
        if not value_fields:
            return None
        
        # 如果没有日期字段，使用索引或第一个分类字段
        if not date_field:
            # 尝试找分类字段
            for key in data[0].keys():
                if "名" in key or "name" in key.lower() or "NAME" in key:
                    date_field = key
                    break
            # 如果还是没有，使用索引
            if not date_field:
                dates = [str(i + 1) for i in range(len(data))]
            else:
                dates = [str(row.get(date_field, "")) for row in data]
        else:
            dates = [str(row.get(date_field, "")) for row in data]
        
        # 提取数据
        series = []
        
        for field in value_fields[:5]:  # 最多显示5个指标
            values = []
            for row in data:
                val = row.get(field, 0)
                # 确保是数值
                if isinstance(val, (int, float)):
                    values.append(float(val))
                else:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        values.append(0)
            
            series.append({
                "name": field,
                "type": "line",
                "data": values,
                "smooth": True,  # 平滑曲线
                "symbol": "circle",
                "symbolSize": 6,
                "lineStyle": {
                    "width": 2
                }
            })
        
        if not series:
            return None
        
        return {
            "type": "line",
            "title": "趋势曲线图",
            "config": {
                "xAxis": {
                    "type": "category",
                    "data": dates,
                    "boundaryGap": False,
                    "axisLabel": {
                        "rotate": 45 if len(dates) > 10 else 0
                    }
                },
                "yAxis": {
                    "type": "value",
                    "name": "数值"
                },
                "series": series,
                "tooltip": {
                    "trigger": "axis",
                    "axisPointer": {
                        "type": "cross"
                    }
                },
                "legend": {
                    "data": [s["name"] for s in series],
                    "top": "10px"
                },
                "grid": {
                    "left": "3%",
                    "right": "4%",
                    "bottom": "15%",
                    "top": "15%",
                    "containLabel": True
                }
            }
        }
    
    def _generate_bar_chart(self, data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """生成柱状图配置"""
        # 查找分类字段和数值字段
        category_field = None
        value_fields = []
        
        for key in data[0].keys():
            if "名" in key or "name" in key.lower() or "NAME" in key:
                category_field = key
            elif "量" in key or "数" in key or isinstance(data[0].get(key), (int, float)):
                # 排除非数值字段
                if key not in ["线路号", "车站号", "ID", "id"]:
                    value_fields.append(key)
        
        # 如果没有分类字段，使用日期字段或索引
        if not category_field:
            for key in data[0].keys():
                if "日期" in key or "date" in key.lower() or "DATE" in key:
                    category_field = key
                    break
            if not category_field:
                categories = [str(i + 1) for i in range(len(data))]
            else:
                categories = [str(row.get(category_field, "")) for row in data]
        else:
            categories = [str(row.get(category_field, "")) for row in data]
        
        if not value_fields:
            return None
        
        # 提取数据
        series = []
        
        for field in value_fields[:5]:  # 最多显示5个指标
            values = []
            for row in data:
                val = row.get(field, 0)
                # 确保是数值
                if isinstance(val, (int, float)):
                    values.append(float(val))
                else:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        values.append(0)
            
            series.append({
                "name": field,
                "type": "bar",
                "data": values
            })
        
        if not series:
            return None
        
        return {
            "type": "bar",
            "title": "柱状图",
            "config": {
                "xAxis": {
                    "type": "category",
                    "data": categories,
                    "axisLabel": {
                        "rotate": 45 if len(categories) > 10 else 0
                    }
                },
                "yAxis": {
                    "type": "value",
                    "name": "数值"
                },
                "series": series,
                "tooltip": {
                    "trigger": "axis",
                    "axisPointer": {
                        "type": "shadow"
                    }
                },
                "legend": {
                    "data": [s["name"] for s in series],
                    "top": "10px"
                },
                "grid": {
                    "left": "3%",
                    "right": "4%",
                    "bottom": "15%",
                    "top": "15%",
                    "containLabel": True
                }
            }
        }

