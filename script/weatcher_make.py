import pandas as pd
import pymssql
from enum import Enum
from datetime import datetime
from typing import Dict, Set, Tuple, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_weather_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定期间的天气数据

    参数:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)

    返回:
        天气数据的 DataFrame
    """
    try:
        conn = pymssql.connect(
            server='10.1.6.230',
            user='sa',
            password='YourStrong!Passw0rd',
            database='master',
            port='1433'
        )

        query = f"""
        SELECT 
            ID,
            F_DATE,
            F_TQQK,
            F_QW,
            F_FLFX,
            CREATETIME,
            CREATOR,
            MODIFYTIME,
            MODIFIER,
            REMARKS,
            MAPPED_WEATHER
        FROM 
            dbo.WeatherHistory
        WHERE 
            F_DATE >= '{start_date}'
            AND F_DATE <= '{end_date}'
        ORDER BY 
            F_DATE
        """

        logger.info(f"执行查询: {start_date} 到 {end_date}")
        df = pd.read_sql(query, conn)
        conn.close()

        logger.info(f"成功获取 {len(df)} 条记录")
        return df

    except Exception as e:
        logger.error(f"数据库读取失败: {e}")
        raise RuntimeError(f"数据库读取失败: {e}")

def extract_weather_types(df: pd.DataFrame) -> Tuple[List[str], Dict[str, Set[str]]]:
    """
    提取所有唯一的F_TQQK（天气现象）类型，并为每种类型编号，同时检查是否有不同的MAPPED_WEATHER对应同一个F_TQQK

    参数:
        df: 天气数据DataFrame

    返回:
        weather_types: 唯一天气类型列表（顺序编号）
        tqqk_to_mapped: F_TQQK到MAPPED_WEATHER的映射
    """
    weather_types = []
    tqqk_to_mapped = {}

    seen = set()
    for idx, row in df.iterrows():
        val = row['F_TQQK']
        mapped = row['MAPPED_WEATHER']

        if pd.isnull(val):
            continue

        val_str = str(val).strip()
        mapped_str = str(mapped).strip() if not pd.isnull(mapped) else None

        if val_str not in seen:
            weather_types.append(val_str)
            seen.add(val_str)

        if val_str not in tqqk_to_mapped:
            tqqk_to_mapped[val_str] = set()
        if mapped_str is not None and mapped_str != 'nan':
            tqqk_to_mapped[val_str].add(mapped_str)

    return weather_types, tqqk_to_mapped

def generate_weather_enum(weather_types: List[str], tqqk_to_mapped: Dict[str, Set[str]]) -> str:
    """
    生成天气类型枚举代码（编号）

    参数:
        weather_types: 唯一天气类型列表
        tqqk_to_mapped: F_TQQK到MAPPED_WEATHER的映射

    返回:
        枚举类代码字符串
    """
    enum_code = "from enum import Enum\n\nclass WeatherType(Enum):\n"
    enum_code += '    """天气类型编号枚举"""\n\n'

    for i, weather in enumerate(weather_types):
        # 生成枚举名称（移除特殊字符，转换为大写）
        enum_name = weather.replace(' ', '_').replace('-', '_').replace('/', '_')
        enum_name = ''.join(c for c in enum_name if c.isalnum() or c == '_')
        enum_name = enum_name.upper()
        if not enum_name or enum_name[0].isdigit():
            enum_name = f"WEATHER_{i+1}"

        mapped_info = ""
        if weather in tqqk_to_mapped and tqqk_to_mapped[weather]:
            mapped_list = list(tqqk_to_mapped[weather])
            if len(mapped_list) == 1:
                mapped_info = f" (映射: {mapped_list[0]})"
            else:
                mapped_info = f" (映射: {', '.join(mapped_list)})"

        enum_code += f'    {enum_name} = {i}  # "{weather}"{mapped_info}\n'

    enum_code += "\n    @classmethod\n"
    enum_code += "    def get_all_types(cls):\n"
    enum_code += "        \"\"\"获取所有天气类型编号\"\"\"\n"
    enum_code += "        return [weather.value for weather in cls]\n"

    enum_code += "\n    @classmethod\n"
    enum_code += "    def get_weather_by_name(cls, name: str):\n"
    enum_code += "        \"\"\"通过原始名称获取天气类型枚举\"\"\"\n"
    enum_code += "        mapping = {\n"
    for i, weather in enumerate(weather_types):
        enum_name = weather.replace(' ', '_').replace('-', '_').replace('/', '_')
        enum_name = ''.join(c for c in enum_name if c.isalnum() or c == '_')
        enum_name = enum_name.upper()
        if not enum_name or enum_name[0].isdigit():
            enum_name = f"WEATHER_{i+1}"
        enum_code += f'            "{weather}": cls.{enum_name},\n'
    enum_code += "        }\n"
    enum_code += "        return mapping.get(name, None)\n"

    return enum_code

def print_statistics(weather_types: List[str], tqqk_to_mapped: Dict[str, Set[str]]):
    """
    打印天气类型及编号统计信息
    """
    print("=" * 60)
    print("天气类型编号统计报告")
    print("=" * 60)

    print(f"\n总共发现 {len(weather_types)} 种不同的天气类型\n")

    print("F_TQQK 字段类型及编号：")
    print("-" * 40)
    for idx, weather in enumerate(weather_types):
        mapped_info = ""
        if weather in tqqk_to_mapped and tqqk_to_mapped[weather]:
            mapped_list = list(tqqk_to_mapped[weather])
            mapped_info = f" -> {', '.join(mapped_list)}"
        print(f"{idx}: {weather}{mapped_info}")

    # 检查映射一致性
    print("\n映射一致性检查：")
    print("-" * 40)
    inconsistent_found = False

    for tqqk, mapped_set in tqqk_to_mapped.items():
        if len(mapped_set) > 1:
            inconsistent_found = True
            print(f"⚠️  '{tqqk}' 对应多个MAPPED_WEATHER: {mapped_set}")

    if not inconsistent_found:
        print("✅ 未发现同一个F_TQQK对应多个MAPPED_WEATHER的情况")

def save_enum_to_file(enum_code: str, filename: str = "weather_enum.py"):
    """
    保存枚举代码到文件
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(enum_code)
        print(f"\n✅ 天气类型枚举已保存到 {filename}")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")

def main():
    """主函数"""
    # 设置查询时间范围
    start_date = '20200101'
    end_date = datetime.now().strftime("%Y%m%d")

    try:
        # 获取数据
        print(f"开始获取天气数据 ({start_date} - {end_date})...")
        df = get_weather_data(start_date, end_date)

        # 提取天气类型
        print("正在分析天气类型...")
        weather_types, tqqk_to_mapped = extract_weather_types(df)

        # 打印统计信息
        print_statistics(weather_types, tqqk_to_mapped)

        # 生成枚举代码
        print("\n正在生成天气类型编号枚举...")
        enum_code = generate_weather_enum(weather_types, tqqk_to_mapped)

        # 打印枚举代码
        print("\n" + "=" * 60)
        print("生成的天气类型编号枚举代码：")
        print("=" * 60)
        print(enum_code)

        # 保存到文件
        save_enum_to_file(enum_code)

        print("\n" + "=" * 60)
        print("统计完成！")
        print("=" * 60)

    except Exception as e:
        logger.error(f"执行失败: {e}")
        print(f"❌ 执行失败: {e}")

if __name__ == "__main__":
    main()