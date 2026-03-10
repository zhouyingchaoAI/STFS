"""
绘制近三年长沙线网、1-6号线、西环线每日客流曲线及年增长率
每条线路一个子图，可视化其每日客流趋势及年度增长情况
"""

import pandas as pd
import pymssql
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 中文字体适配
try:
    from font_utils import get_chinese_font, configure_fonts
except ImportError:
    try:
        from script.font_utils import get_chinese_font, configure_fonts
    except ImportError:
        def get_chinese_font():
            return None
        def configure_fonts():
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

configure_fonts()
my_font = get_chinese_font()
if my_font is not None:
    font_name = my_font.get_name()
    plt.rcParams['font.sans-serif'] = ['SimHei', font_name]
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family=plt.rcParams['font.sans-serif'][0])
plt.ticklabel_format(style='plain', axis='y')

# 连接数据库设置
def get_db_conn_script(charset='utf8'):
    DB_CONFIG = {
        "server": "10.1.6.230",
        "user": "sa",
        "password": "YourStrong!Passw0rd",
        "database": "StationFlowPredict",
        "port": 1433
    }
    conn_params = DB_CONFIG.copy()
    if charset:
        conn_params["charset"] = charset
    return pymssql.connect(**conn_params)

try:
    from db_utils import fix_dataframe_encoding
except ImportError:
    def fix_dataframe_encoding(df):
        return df

# -------- 主要参数 ---------
# 需要保留的线路 LINE_ID（支持多种格式，如 '0', '00', '01' 等）
# 只保留：线网00，线路01-06，以及31、60、西环线
TARGET_LINE_IDS = ['0', '00', '01', '02', '03', '04', '05', '06', '31', '60', '83']

now = datetime.now()
year_range = [now.year-4, now.year-3, now.year-2, now.year-1, now.year]   # 最近三年

# -------- 查询并分析 ---------
print("正在从数据库抓取线网及各线路近三年客流数据...")

try:
    conn = get_db_conn_script(charset='utf8')
    print("✓ 数据库连接成功")

    # 先从 LineDailyFlowHistory 表查询 LINE_ID 和线路名称的映射关系
    # 参考 dd.py 的方式，从数据库动态获取线路名称
    line_name_mapping_query = """
    SELECT DISTINCT 
        CAST(L.F_LINENO AS VARCHAR) AS LINE_ID,
        L.F_LINENAME AS LINE_NAME
    FROM [master].[dbo].[LineDailyFlowHistory] AS L
    WHERE L.CREATOR = 'chency'
      AND L.F_LINENAME IS NOT NULL
      AND L.F_LINENAME != ''
    """
    try:
        df_line_mapping = pd.read_sql(line_name_mapping_query, conn)
        df_line_mapping = fix_dataframe_encoding(df_line_mapping)
        df_line_mapping['LINE_ID'] = df_line_mapping['LINE_ID'].astype(str).str.strip()
        # 创建 LINE_ID 到线路名称的映射字典
        LINE_NAME_MAPPING = {}
        for _, row in df_line_mapping.iterrows():
            line_id = str(row['LINE_ID']).strip()
            line_name = str(row['LINE_NAME']).strip()
            if line_id and line_name:
                LINE_NAME_MAPPING[line_id] = line_name
                # 同时支持去除前导零的格式
                line_id_clean = line_id.lstrip('0') if line_id.lstrip('0') else '0'
                if line_id_clean != line_id:
                    LINE_NAME_MAPPING[line_id_clean] = line_name
        print(f"✓ 从数据库获取到 {len(LINE_NAME_MAPPING)} 个线路名称映射")
    except Exception as e:
        print(f"⚠ 无法从 LineDailyFlowHistory 获取线路名称映射: {e}")
        print("  将使用默认的线路名称生成规则")
        LINE_NAME_MAPPING = {}

    # 先查询数据库中所有可用的 LINE_ID（近三年内有数据的）
    line_ids_query = f"""
    SELECT DISTINCT [LINE_ID]
    FROM [StationFlowPredict].[dbo].[STATION_FLOW_HISTORY]
    WHERE YEAR([SQUAD_DATE]) >= {year_range[0]}
      AND YEAR([SQUAD_DATE]) <= {year_range[-1]}
    ORDER BY [LINE_ID]
    """
    df_line_ids = pd.read_sql(line_ids_query, conn)
    df_line_ids = fix_dataframe_encoding(df_line_ids)
    df_line_ids['LINE_ID'] = df_line_ids['LINE_ID'].astype(str).str.strip()
    all_line_ids = df_line_ids['LINE_ID'].unique().tolist()
    
    # 过滤出需要保留的线路（支持多种格式匹配）
    available_line_ids = []
    for line_id in all_line_ids:
        # 检查是否在目标列表中
        # 1. 直接匹配
        if line_id in TARGET_LINE_IDS:
            available_line_ids.append(line_id)
            continue
        # 2. 去除前导零后匹配（处理 '00' -> '0', '01' -> '1' 等）
        line_id_clean = line_id.lstrip('0') if line_id.lstrip('0') else '0'
        if line_id_clean in TARGET_LINE_IDS:
            available_line_ids.append(line_id)
            continue
        # 3. 添加前导零后匹配（处理 '1' -> '01' 等）
        if line_id.isdigit():
            line_id_padded = line_id.zfill(2)  # '1' -> '01'
            if line_id_padded in TARGET_LINE_IDS:
                available_line_ids.append(line_id)
    
    if len(available_line_ids) == 0:
        print("未找到需要保留的线路数据")
        conn.close()
        exit(1)
    
    print(f"✓ 找到 {len(available_line_ids)} 条需要保留的线路: {', '.join(sorted(available_line_ids))}")
    
    # 验证：确保 available_line_ids 中的 LINE_ID 都在 TARGET_LINE_IDS 的匹配范围内
    # 如果某个 LINE_ID 经过格式转换后仍不在 TARGET_LINE_IDS 中，则移除
    filtered_available_line_ids = []
    for line_id in available_line_ids:
        line_id_clean = line_id.lstrip('0') if line_id.lstrip('0') else '0'
        line_id_padded = line_id.zfill(2) if line_id.isdigit() else line_id
        if (line_id in TARGET_LINE_IDS or 
            line_id_clean in TARGET_LINE_IDS or 
            line_id_padded in TARGET_LINE_IDS):
            filtered_available_line_ids.append(line_id)
        else:
            print(f"⚠ 警告：LINE_ID '{line_id}' 不在目标列表中，将被过滤")
    available_line_ids = filtered_available_line_ids
    
    if len(available_line_ids) == 0:
        print("未找到需要保留的线路数据")
        conn.close()
        exit(1)
    
    print(f"✓ 最终保留 {len(available_line_ids)} 条线路: {', '.join(sorted(available_line_ids))}")

    # 查询三年内所有线路的站点天总客流
    line_ids_list = [f"'{lid}'" for lid in available_line_ids]
    select_sql = f"""
    SELECT 
        [LINE_ID],
        [SQUAD_DATE],
        SUM([PASSENGER_NUM]) AS DAY_LINE_FLOW
    FROM [StationFlowPredict].[dbo].[STATION_FLOW_HISTORY]
    WHERE [LINE_ID] IN ({','.join(line_ids_list)})
      AND YEAR([SQUAD_DATE]) >= {year_range[0]}
      AND YEAR([SQUAD_DATE]) <= {year_range[-1]}
    GROUP BY [LINE_ID], [SQUAD_DATE]
    """
    df_raw = pd.read_sql(select_sql, conn)
    conn.close()
    df_raw = fix_dataframe_encoding(df_raw)

    # 时间格式化
    df_raw['SQUAD_DATE'] = pd.to_datetime(df_raw['SQUAD_DATE'])
    
    # LINE_ID 转换为字符串并去除空格
    df_raw['LINE_ID'] = df_raw['LINE_ID'].astype(str).str.strip()

    # 按照每条线路，保留每日客流数据
    # 使用数据库中的实际 LINE_ID，并根据映射表生成显示名称
    data_daily = {}  # 存储每日数据
    data_year = {}   # 存储年度汇总数据（用于计算增长率）
    line_order = []
    
    # 定义线路显示顺序：线网00，线路01-06，31、60，西环线
    def get_display_order(line_id):
        """获取线路的显示顺序"""
        line_id_clean = line_id.lstrip('0') if line_id.lstrip('0') else '0'
        order_map = {
            '0': 0, '00': 0,      # 线网
            '1': 1, '01': 1,      # 1号线
            '2': 2, '02': 2,      # 2号线
            '3': 3, '03': 3,      # 3号线
            '4': 4, '04': 4,      # 4号线
            '5': 5, '05': 5,      # 5号线
            '6': 6, '06': 6,      # 6号线
            '31': 7,              # 31号线
            '60': 8,              # 60号线
            '83': 9,              # 西环线
        }
        return order_map.get(line_id_clean, 99)
    
    # 按照显示顺序排序
    sorted_line_ids = sorted(available_line_ids, key=lambda x: (get_display_order(x), x))
    
    def get_line_display_name(line_id):
        """根据 LINE_ID 获取显示名称，优先使用数据库中的名称，否则智能生成"""
        # 1. 直接匹配
        if line_id in LINE_NAME_MAPPING:
            return LINE_NAME_MAPPING[line_id]
        # 2. 去除前导零后匹配
        line_id_clean = line_id.lstrip('0') if line_id.lstrip('0') else '0'
        if line_id_clean in LINE_NAME_MAPPING:
            return LINE_NAME_MAPPING[line_id_clean]
        # 3. 智能生成名称
        if line_id_clean == '0' or line_id == '00':
            return '线网'
        elif line_id_clean.isdigit():
            return f'{line_id_clean}号线'
        else:
            return f'线路{line_id}'
    
    # 处理所有线路（包括线网），线网和线路一样独立处理，不汇总
    for line_id in sorted_line_ids:
        # 获取显示名称（从数据库动态获取或智能生成）
        line_name = get_line_display_name(line_id)
        line_order.append(line_name)
        
        dfl = df_raw[df_raw['LINE_ID'] == line_id].copy()
        if len(dfl) == 0:
            # 如果没有数据，创建空数据框
            empty_daily = pd.DataFrame({'SQUAD_DATE': pd.date_range(start=f'{year_range[0]}-01-01', end=f'{year_range[-1]}-12-31', freq='D'), 'DAY_LINE_FLOW': [0]})
            empty_daily = empty_daily[empty_daily['SQUAD_DATE'].dt.year.isin(year_range)]
            data_daily[line_name] = empty_daily
            yearly = pd.DataFrame({'YEAR': year_range, 'DAY_LINE_FLOW': [0]*len(year_range), 'LINE': line_name})
            data_year[line_name] = yearly
            continue
        
        # 按日期排序
        dfl = dfl.sort_values('SQUAD_DATE').copy()
        # 确保日期唯一（如果有重复，取平均值）
        dfl_daily = dfl.groupby('SQUAD_DATE')['DAY_LINE_FLOW'].sum().reset_index()
        data_daily[line_name] = dfl_daily[['SQUAD_DATE', 'DAY_LINE_FLOW']]
        
        # 同时计算年度汇总（用于增长率计算）
        dfl['YEAR'] = dfl['SQUAD_DATE'].dt.year
        yearly = (
            dfl.groupby('YEAR')['DAY_LINE_FLOW']
            .sum()
            .reindex(year_range, fill_value=0)
            .reset_index()
        )
        yearly['LINE'] = line_name
        data_year[line_name] = yearly

    # --------- 绘制每日客流曲线图 ---------
    # 动态计算子图布局
    num_lines = len(line_order)
    if num_lines == 0:
        print("没有可用的线路数据")
        exit(1)
    
    # 计算合适的行列数（尽量接近正方形布局）
    cols = 4
    rows = (num_lines + cols - 1) // cols  # 向上取整
    fig, axes = plt.subplots(rows, cols, figsize=(23, 6*rows))
    if num_lines == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#2e8b57", "#8b3a62", "#d62728", "#9467bd", "#bcbd22"]
    growth_results = {}

    for idx, line_name in enumerate(line_order):
        if line_name not in data_daily:
            continue
        
        daily_data = data_daily[line_name]
        yearly = data_year[line_name]
        ax = axes[idx]
        
        # 绘制每日客流曲线
        ax.plot(
            daily_data['SQUAD_DATE'],
            daily_data['DAY_LINE_FLOW']/1e4, # 转成万人次
            color=colors[idx % len(colors)],
            linewidth=1.5,
            alpha=0.7,
            label=f"{line_name}每日客流"
        )
        
        # 添加年度平均线（可选，用虚线表示）
        for year in year_range:
            year_data = daily_data[daily_data['SQUAD_DATE'].dt.year == year]
            if len(year_data) > 0:
                year_avg = year_data['DAY_LINE_FLOW'].mean() / 1e4
                ax.axhline(y=year_avg, color=colors[idx % len(colors)], linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_title(f"{line_name}", fontsize=15, fontproperties=my_font)
        ax.set_xlabel("日期", fontsize=12, fontproperties=my_font)
        ax.set_ylabel("日客流（万人次）", fontsize=12, fontproperties=my_font)
        ax.grid(axis='y', alpha=0.3)
        ax.grid(axis='x', alpha=0.2)
        
        # 设置x轴日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 计算并存储年度增长率
        # 确保 yearly 按照 year_range 的顺序排列
        yearly_sorted = yearly.set_index('YEAR').reindex(year_range, fill_value=0).reset_index()
        growth_rate = []
        for i in range(1, len(yearly_sorted)):
            prev = yearly_sorted.loc[i-1, 'DAY_LINE_FLOW']
            curr = yearly_sorted.loc[i, 'DAY_LINE_FLOW']
            rate = (curr-prev)/prev if prev>0 else np.nan
            growth_rate.append(rate)
        growth_results[line_name] = growth_rate if growth_rate else [np.nan]*(len(year_range)-1)
        
        # 在右上角显示年度增长率
        growth_text = ""
        for i, rate in enumerate(growth_results[line_name]):
            if not np.isnan(rate):
                year_pair = f"{year_range[i]}→{year_range[i+1]}"
                growth_text += f"{year_pair}: {rate*100:.1f}%\n"
        
        if growth_text:
            ax.text(0.98, 0.98, growth_text.strip(), 
                   transform=ax.transAxes, 
                   fontsize=9,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontproperties=my_font)
        
        if idx == 0:
            ax.legend(loc='upper left', prop=my_font, fontsize=10)
    for i in range(len(line_order), len(axes)):
        axes[i].axis('off')

    plt.suptitle("长沙线网及各线路近三年每日客流曲线与年增长率", fontsize=18, fontproperties=my_font)
    plt.tight_layout(rect=[0,0,1,0.97])

    output_path = "长沙地铁线网分线路每日客流曲线三年增长.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 全线路每日客流曲线图已保存至: {os.path.abspath(output_path)}")
    plt.close()

    # --------- 输出各线路年增长率总结 ---------
    print('\n' + '='*80)
    print("各线路年度总客流与年增长率（单位：千万人次/年，增长率%）")
    print('='*80)
    summary_table = []
    num_years = len(year_range)
    num_growths = num_years - 1  # 增长率数量 = 年份数 - 1
    
    for line_name in line_order:
        if line_name not in data_year:
            # 如果线路没有数据，填充空值
            row = [line_name] + ["-"]*num_years + ["-"]*num_growths
            summary_table.append(row)
            continue
        yearly = data_year[line_name]
        # 确保flows有正确的元素数量
        flows_values = yearly['DAY_LINE_FLOW'].values/1e7
        if len(flows_values) < num_years:
            flows = list(flows_values) + [np.nan]*(num_years - len(flows_values))
        elif len(flows_values) > num_years:
            flows = flows_values[:num_years]
        else:
            flows = flows_values
        
        growths = growth_results.get(line_name, [np.nan]*num_growths)
        # 确保growths有正确的元素数量
        if len(growths) < num_growths:
            growths = list(growths) + [np.nan]*(num_growths - len(growths))
        elif len(growths) > num_growths:
            growths = growths[:num_growths]
        
        # 格式化数据
        flows_str = [f"{x:.2f}" if not np.isnan(x) else "-" for x in flows]
        growths_str = [f"{y*100:.1f}%" if not np.isnan(y) else "-" for y in growths]
        row = [line_name] + flows_str + growths_str
        summary_table.append(row)
    
    # 生成表头
    header = ["线路"] + [str(year) for year in year_range]
    for i in range(num_growths):
        header.append(f"{year_range[i]}→{year_range[i+1]}")
    
    print('\t'.join(header))
    for row in summary_table:
        print('\t'.join(row))
    print('='*80)
    print("✓ 分析完成。")
except pymssql.Error as db_error:
    print(f"数据库连接错误: {db_error}")
    print("请检查数据库配置、权限和网络。")
    import traceback
    traceback.print_exc()
except ImportError as import_error:
    print(f"导入模块错误: {import_error}")
    print("pip install pandas pymssql matplotlib numpy")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()
