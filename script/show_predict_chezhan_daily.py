import pandas as pd
import pymssql
import yaml
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "server": "10.1.6.230",
    "user": "sa",
    "password": "YourStrong!Passw0rd",
    "database": "master",
    "port": 1433
}

def save_station_line_mapping_yaml(yaml_path: str = "station_line_mapping.yaml"):
    """
    查询STATION_FLOW_HISTORY获得每个STATION_NAME对应的LINE_ID和STATION_ID
    并保存为yaml格式文件。
    """
    query = """
        SELECT DISTINCT
            STATION_NAME,
            LINE_ID,
            STATION_ID
        FROM [master].[dbo].[STATION_FLOW_HISTORY]
        WHERE STATION_NAME IS NOT NULL
        ORDER BY LINE_ID, STATION_ID
    """
    try:
        conn = pymssql.connect(
            server=DB_CONFIG["server"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            port=DB_CONFIG["port"]
        )
        df = pd.read_sql(query, conn)
        conn.close()

        # 修复STATION_NAME的乱码（智能编码修复，参考db_utils.py用法）
        def smart_encode_fix(x):
            if not isinstance(x, str):
                return x
            # 检查是否包含乱码字符（latin-1编码下的中文）
            if any(ord(c) > 127 and ord(c) < 256 for c in x):
                try:
                    return x.encode('latin-1').decode('gbk')
                except:
                    try:
                        return x.encode('latin-1').decode('utf-8')
                    except:
                        return x
            return x
        if 'STATION_NAME' in df.columns:
            df['STATION_NAME'] = df['STATION_NAME'].astype(str).apply(smart_encode_fix)

        

        # 构建字典: {STATION_NAME: [{LINE_ID: xx, STATION_ID: xx}, ...], ...}
        station_map = {}
        for _idx, row in df.iterrows():
            station_name = str(row['STATION_NAME'])
            line_id = row['LINE_ID']
            station_id = row['STATION_ID']
            entry = {"LINE_ID": line_id, "STATION_ID": station_id}
            station_map.setdefault(station_name, []).append(entry)

        # 输出为YAML文件
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(station_map, f, allow_unicode=True, sort_keys=False)
        logger.info(f"YAML文件已保存：{yaml_path}")

        # 展示示例
        logger.info("通过STATION_NAME，可找到对应的LINE_ID, STATION_ID组合，例如：")
        example_name = next(iter(station_map))
        logger.info(f"{example_name}: {station_map[example_name]}")
    except Exception as e:
        logger.error(f"查询或输出yaml出错: {e}")

def load_line_station_by_station_name(station_name: str, yaml_path: str = "station_line_mapping.yaml") -> Optional[List[Dict[str, Any]]]:
    """
    读取yaml，通过输入STATION_NAME返回[{LINE_ID: xx, STATION_ID: xx}, ...]
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            station_map = yaml.safe_load(f)
        results = station_map.get(station_name)
        if results:
            logger.info(f"{station_name} 对应的LINE_ID, STATION_ID列表: {results}")
            return results
        else:
            logger.warning(f"未找到车站 {station_name} 的对应LINE_ID和STATION_ID。")
            return None
    except FileNotFoundError:
        logger.error(f"YAML文件 {yaml_path} 不存在。请先运行save_station_line_mapping_yaml生成。")
    except Exception as e:
        logger.error(f"读取或查找yaml出错: {e}")
    return None

if __name__ == "__main__":
    # 先生成yaml文件
    save_station_line_mapping_yaml()
    # 示例：根据站名查询
    example_station_name = input("请输入要查询的STATION_NAME（如需示例请直接回车）：").strip()
    if not example_station_name:
        with open('station_line_mapping.yaml', 'r', encoding='utf-8') as f:
            station_map = yaml.safe_load(f)
            example_station_name = next(iter(station_map))
            print(f"示例车站: {example_station_name}")
    load_line_station_by_station_name(example_station_name)
