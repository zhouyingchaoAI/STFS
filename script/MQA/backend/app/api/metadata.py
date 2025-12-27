"""
元数据API
"""
from fastapi import APIRouter
from app.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.get("/metadata/tables")
async def get_tables():
    """
    获取所有表列表（从 table_structures.txt 自动读取）
    """
    try:
        from pathlib import Path
        
        # 尝试从 table_structures.txt 读取表名
        current_dir = Path(__file__).parent.parent.parent.parent
        file_path = current_dir / "table_structures.txt"
        if not file_path.exists():
            file_path = current_dir.parent / "table_structures.txt"
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析表名
            tables = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('表：'):
                    table_name = line.replace('表：', '').strip()
                    if table_name:
                        # 提取表名（去掉 dbo. 前缀，只保留表名）
                        if '.' in table_name:
                            table_name = table_name.split('.')[-1]
                        tables.append(table_name)
            
            if tables:
                # 去重
                tables = list(dict.fromkeys(tables))
                return {
                    "code": 200,
                    "message": "success",
                    "data": {
                        "tables": tables
                    }
                }
    except Exception as e:
        logger.warning(f"Failed to load tables from file: {e}, using default list")
    
    # 默认表列表（如果文件不存在或解析失败）
    return {
        "code": 200,
        "message": "success",
        "data": {
            "tables": [
                "CalendarHistory",
                "LineDailyFlowHistory",
                "LineHourlyFlowHistory",
                "LSTM_COMMON_HOLIDAYFEATURE",
                "STATION_FLOW_HISTORY",
                "STATION_HOUR_HISTORY",
                "WeatherFuture",
                "WeatherHistory",
                "LineDailyFlowPrediction",
                "LineHourlyFlowPrediction",
                "STATION_FLOW_PREDICT",
                "STATION_HOUR_PREDICT",
            ]
        }
    }


@router.get("/metadata/stations")
async def get_stations():
    """
    获取所有车站列表
    """
    import yaml
    from pathlib import Path
    
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
                    stations = yaml.safe_load(f) or {}
                    return {
                        "code": 200,
                        "message": "success",
                        "data": {
                            "stations": list(stations.keys())
                        }
                    }
    except Exception as e:
        logger.error(f"Failed to load stations: {e}")
    
    return {
        "code": 200,
        "message": "success",
        "data": {
            "stations": []
        }
    }


@router.get("/metadata/lines")
async def get_lines():
    """
    获取所有线路列表
    """
    return {
        "code": 200,
        "message": "success",
        "data": {
            "lines": ["1号线", "2号线", "3号线", "4号线", "5号线", "6号线"]
        }
    }

