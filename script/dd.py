"""
绘制2021-2025年12月31日、1月1-3日各线路客流情况图表
不同线网线路分不同子图，不同年份用不同颜色
"""
import pandas as pd
import pymssql
from datetime import datetime, timedelta
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.dates as mdates
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入字体配置工具
try:
    from font_utils import get_chinese_font, configure_fonts
except ImportError:
    try:
        from script.font_utils import get_chinese_font, configure_fonts
    except ImportError:
        # 如果无法导入，使用默认配置
        def get_chinese_font():
            return None
        def configure_fonts():
            # 设置中文字体列表，按优先级排序
            matplotlib.rcParams['font.sans-serif'] = [
                'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 
                'WenQuanYi Micro Hei', 'STHeiti', 'Arial Unicode MS', 
                'DejaVu Sans'
            ]
            matplotlib.rcParams['axes.unicode_minus'] = False
            # 抑制字体警告
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 配置字体和格式
configure_fonts()
my_font = get_chinese_font()
if my_font is not None:
    font_name = my_font.get_name()
    plt.rcParams['font.sans-serif'] = ['SimHei', font_name]
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family=plt.rcParams['font.sans-serif'][0])

# 数据库配置
DB_CONFIG = {
    "server": "10.1.6.230",
    "user": "sa",
    "password": "YourStrong!Passw0rd",
    "database": "master",
    "port": 1433
}

def get_db_conn(charset='utf8'):
    """
    获取数据库连接
    """
    conn_params = DB_CONFIG.copy()
    if charset:
        conn_params["charset"] = charset
    return pymssql.connect(**conn_params)

def has_chinese(text: str) -> bool:
    """
    检测字符串是否包含中文字符
    """
    if not isinstance(text, str):
        return False
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

def smart_encode_fix(value):
    """
    智能编码修复函数
    修复 pymssql 读取 NVARCHAR 字段时产生的编码乱码问题
    """
    if value is None:
        return ''
    
    if isinstance(value, bytes):
        try:
            value = value.decode('utf-8')
        except UnicodeDecodeError:
            try:
                value = value.decode('gbk')
            except UnicodeDecodeError:
                return value.decode('utf-8', errors='replace')
    
    if not isinstance(value, str):
        return str(value) if value is not None else ''
    
    if any(ord(c) > 127 and ord(c) < 256 for c in value):
        try:
            fixed = value.encode('latin-1').decode('gbk')
            if has_chinese(fixed):
                return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                fixed = value.encode('latin-1').decode('utf-8')
                if has_chinese(fixed):
                    return fixed
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
    
    return value

def fix_dataframe_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复 DataFrame 中所有字符串列的编码问题
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).apply(smart_encode_fix)
    
    return df

# 查询2021-2025年的12月31日、1月1-3日数据
start_date = 20210101
end_date = 20251231

try:
    # 连接数据库
    conn = get_db_conn(charset='utf8')
    
    # 查询线路日客流历史数据
    query = f"""
    SELECT 
        L.F_DATE,
        L.F_LINENO,
        L.F_LINENAME,
        L.F_KLCOUNT
    FROM 
        [master].[dbo].[LineDailyFlowHistory] AS L
    WHERE 
        L.CREATOR = 'chency' 
        AND L.F_DATE >= {start_date}
        AND L.F_DATE <= {end_date}
        AND (
            (L.F_DATE % 10000 = 1231) OR  -- 12月31日
            (L.F_DATE % 10000 >= 101 AND L.F_DATE % 10000 <= 103)  -- 1月1-3日
        )
    ORDER BY 
        L.F_DATE, L.F_LINENO
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        print("未查询到数据")
        sys.exit(0)
    
    # 应用编码修复
    df = fix_dataframe_encoding(df)
    
    # 确保数据类型正确
    df['F_DATE'] = pd.to_datetime(df['F_DATE'].astype(str), format='%Y%m%d', errors='coerce')
    
    # 添加年份和日期标识
    df['YEAR'] = df['F_DATE'].dt.year
    df['DATE_LABEL'] = df['F_DATE'].dt.strftime('%m-%d')
    
    # 筛选特定日期：12月31日和1月1-3日
    target_date_labels = ['12-31', '01-01', '01-02', '01-03']
    
    # 按照跨年规则重新分组：
    # 2021年12月31日 + 2022年1月1-3日 → 看作2022年
    # 2022年12月31日 + 2023年1月1-3日 → 看作2023年
    # 2023年12月31日 + 2024年1月1-3日 → 看作2024年
    # 2024年12月31日 + 2025年1月1-3日 → 看作2025年
    specific_dates_list = []
    
    # 2021年12月31日 + 2022年1月1-3日 → 看作2022年
    df_2021_1231 = df[(df['YEAR'] == 2021) & (df['DATE_LABEL'] == '12-31')].copy()
    df_2022_0101_03 = df[(df['YEAR'] == 2022) & (df['DATE_LABEL'].isin(['01-01', '01-02', '01-03']))].copy()
    if not df_2021_1231.empty or not df_2022_0101_03.empty:
        df_group_2022 = pd.concat([df_2021_1231, df_2022_0101_03], ignore_index=True)
        df_group_2022['YEAR_GROUP'] = 2022
        specific_dates_list.append(df_group_2022)
    
    # 2022年12月31日 + 2023年1月1-3日 → 看作2023年
    df_2022_1231 = df[(df['YEAR'] == 2022) & (df['DATE_LABEL'] == '12-31')].copy()
    df_2023_0101_03 = df[(df['YEAR'] == 2023) & (df['DATE_LABEL'].isin(['01-01', '01-02', '01-03']))].copy()
    if not df_2022_1231.empty or not df_2023_0101_03.empty:
        df_group_2023 = pd.concat([df_2022_1231, df_2023_0101_03], ignore_index=True)
        df_group_2023['YEAR_GROUP'] = 2023
        specific_dates_list.append(df_group_2023)
    
    # 2023年12月31日 + 2024年1月1-3日 → 看作2024年
    df_2023_1231 = df[(df['YEAR'] == 2023) & (df['DATE_LABEL'] == '12-31')].copy()
    df_2024_0101_03 = df[(df['YEAR'] == 2024) & (df['DATE_LABEL'].isin(['01-01', '01-02', '01-03']))].copy()
    if not df_2023_1231.empty or not df_2024_0101_03.empty:
        df_group_2024 = pd.concat([df_2023_1231, df_2024_0101_03], ignore_index=True)
        df_group_2024['YEAR_GROUP'] = 2024
        specific_dates_list.append(df_group_2024)
    
    # 2024年12月31日 + 2025年1月1-3日 → 看作2025年
    df_2024_1231 = df[(df['YEAR'] == 2024) & (df['DATE_LABEL'] == '12-31')].copy()
    df_2025_0101_03 = df[(df['YEAR'] == 2025) & (df['DATE_LABEL'].isin(['01-01', '01-02', '01-03']))].copy()
    if not df_2024_1231.empty or not df_2025_0101_03.empty:
        df_group_2025 = pd.concat([df_2024_1231, df_2025_0101_03], ignore_index=True)
        df_group_2025['YEAR_GROUP'] = 2025
        specific_dates_list.append(df_group_2025)
    
    if specific_dates_list:
        specific_dates_df = pd.concat(specific_dates_list, ignore_index=True)
        
        # 年份组列表
        year_groups = [2022, 2023, 2024, 2025]
        
        # 打印分组日期信息
        print("\n" + "="*80)
        print("日期分组信息（跨年分组）")
        print("="*80)
        
        for year_group in year_groups:
            year_group_data = specific_dates_df[specific_dates_df['YEAR_GROUP'] == year_group]
            if not year_group_data.empty:
                dates_in_group = sorted(year_group_data['F_DATE'].unique())
                date_labels_in_group = sorted(year_group_data['DATE_LABEL'].unique())
                actual_years = sorted(year_group_data['YEAR'].unique())
                print(f"\n{year_group}年组（包含实际年份: {actual_years}）:")
                print(f"  日期数量: {len(dates_in_group)}")
                print(f"  日期列表: {[d.strftime('%Y-%m-%d') for d in dates_in_group]}")
                print(f"  日期标识: {date_labels_in_group}")
                # 按日期标识分组显示
                for date_label in target_date_labels:
                    date_data = year_group_data[year_group_data['DATE_LABEL'] == date_label]
                    if not date_data.empty:
                        dates = sorted(date_data['F_DATE'].unique())
                        print(f"    {date_label}: {[d.strftime('%Y-%m-%d') for d in dates]}")
                    else:
                        print(f"    {date_label}: 无数据")
        
        print("\n" + "="*80)
        print("按线路和年份组汇总日期信息")
        print("="*80)
        
        # 获取所有线路
        lines = sorted(specific_dates_df['F_LINENAME'].unique())
        
        for line_name in lines:
            line_data = specific_dates_df[specific_dates_df['F_LINENAME'] == line_name]
            if not line_data.empty:
                print(f"\n线路: {line_name}")
                for year_group in year_groups:
                    year_group_line_data = line_data[line_data['YEAR_GROUP'] == year_group]
                    if not year_group_line_data.empty:
                        date_labels = sorted(year_group_line_data['DATE_LABEL'].unique())
                        dates = sorted(year_group_line_data['F_DATE'].unique())
                        print(f"  {year_group}年组: {date_labels} - 共{len(dates)}个日期")
        
        print("\n" + "="*80 + "\n")
        
        # 为不同年份组设置颜色
        year_colors_map = {
            2022: '#FF6B6B',  # 红色
            2023: '#4ECDC4',  # 青色
            2024: '#95E1D3',  # 浅青色
            2025: '#FFD93D',  # 黄色
        }
        
        # 计算子图布局
        n_lines = len(lines)
        ncols = min(3, n_lines)
        nrows = int(np.ceil(n_lines / ncols))
        
        # 创建图表
        fig = plt.figure(figsize=(8 * ncols, 6 * nrows))
        
        for idx, line_name in enumerate(lines):
            ax = fig.add_subplot(nrows, ncols, idx + 1)
            
            # 筛选该线路的数据
            line_data = specific_dates_df[
                specific_dates_df['F_LINENAME'] == line_name
            ].copy()
            
            if not line_data.empty:
                # 按年份组和日期汇总该线路的客流量
                line_yearly_dates = line_data.groupby(['YEAR_GROUP', 'DATE_LABEL']).agg(
                    passenger_total=('F_KLCOUNT', 'sum')  # 客流量
                ).reset_index()
                
                # 为每个年份组绘制数据
                for year_group in year_groups:
                    year_group_data = line_yearly_dates[line_yearly_dates['YEAR_GROUP'] == year_group].sort_values('DATE_LABEL')
                    
                    if not year_group_data.empty:
                        x_vals = []
                        y_vals = []
                        for date_label in target_date_labels:
                            x_vals.append(target_date_labels.index(date_label))
                            date_data = year_group_data[year_group_data['DATE_LABEL'] == date_label]
                            if not date_data.empty:
                                y_vals.append(date_data['passenger_total'].iloc[0])
                            else:
                                # 如果该日期没有数据，使用NaN
                                y_vals.append(np.nan)
                        
                        # 只有当至少有一个有效数据点时才绘制
                        if not all(np.isnan(y_vals)):
                            ax.plot(x_vals, y_vals, 
                                   marker='o', linewidth=2.5, markersize=8,
                                   color=year_colors_map.get(year_group, '#CCCCCC'),
                                   label=f'{year_group}年组', alpha=0.8)
                
                ax.set_xlabel('日期', fontsize=11, fontproperties=my_font)
                ax.set_ylabel('客流量（人次）', fontsize=11, fontproperties=my_font)
                ax.set_title(f'{line_name} - 跨年四天客流量（2022-2025年组）', 
                           fontsize=12, fontweight='bold', fontproperties=my_font)
                ax.set_xticks(range(len(target_date_labels)))
                ax.set_xticklabels(target_date_labels, fontsize=10, fontproperties=my_font)
                ax.legend(prop=my_font, fontsize=9, loc='best', ncol=2)
                ax.grid(True, alpha=0.3)
                ax.ticklabel_format(style='plain', axis='y')
            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, fontproperties=my_font)
                ax.set_title(f'{line_name}', fontsize=12, fontweight='bold', fontproperties=my_font)
        
        plt.suptitle('跨年四天各线路客流量情况（2022-2025年组，不同年份组不同颜色）', 
                    fontsize=16, fontweight='bold', y=0.995, fontproperties=my_font)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # 保存图表
        output_file = '2021-2025年12月31日_1月1-3日各线路客流情况.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存至: {output_file}")
        plt.close(fig)
        
        # 保存数据到CSV
        # 准备CSV数据：按线路和年份组汇总
        csv_data_list = []
        for line_name in lines:
            line_data = specific_dates_df[specific_dates_df['F_LINENAME'] == line_name].copy()
            if not line_data.empty:
                line_yearly_dates = line_data.groupby(['YEAR_GROUP', 'DATE_LABEL']).agg(
                    passenger_total=('F_KLCOUNT', 'sum')
                ).reset_index()
                
                for year_group in year_groups:
                    year_group_data = line_yearly_dates[line_yearly_dates['YEAR_GROUP'] == year_group].sort_values('DATE_LABEL')
                    if not year_group_data.empty:
                        for _, row in year_group_data.iterrows():
                            csv_data_list.append({
                                '线路名称': line_name,
                                '年份组': year_group,
                                '日期标识': row['DATE_LABEL'],
                                '客运量': row['passenger_total']
                            })
        
        if csv_data_list:
            csv_df = pd.DataFrame(csv_data_list)
            # 重新排列列顺序
            csv_df = csv_df[['线路名称', '年份组', '日期标识', '客运量']]
            csv_output_file = '跨年四天各线路客流数据.csv'
            csv_df.to_csv(csv_output_file, index=False, encoding='utf-8-sig')
            print(f"✓ 跨年四天客流数据已保存至: {csv_output_file}")
    else:
        print("未查询到符合条件的日期数据")
    
    # ========== 绘制2025年全年节假日客流图 ==========
    print("\n" + "="*80)
    print("开始绘制2025年全年节假日客流图")
    print("="*80)
    
    # 查询2025年全年节假日数据
    query_2025 = f"""
    SELECT 
        L.F_DATE,
        L.F_LINENO,
        L.F_LINENAME,
        L.F_KLCOUNT,
        CC.F_HOLIDAYTYPE
    FROM 
        [master].[dbo].[LineDailyFlowHistory] AS L
    LEFT JOIN 
        [master].[dbo].[CalendarHistory] AS CC
        ON L.F_DATE = CC.F_DATE
    WHERE 
        L.CREATOR = 'chency' 
        AND L.F_DATE >= 20250101
        AND L.F_DATE <= 20251231
        AND CC.F_HOLIDAYTYPE IS NOT NULL
        AND CC.F_HOLIDAYTYPE != 0
    ORDER BY 
        L.F_DATE, L.F_LINENO
    """
    
    conn_2025 = get_db_conn(charset='utf8')
    df_2025 = pd.read_sql(query_2025, conn_2025)
    conn_2025.close()
    
    if not df_2025.empty:
        # 应用编码修复
        df_2025 = fix_dataframe_encoding(df_2025)
        
        # 确保数据类型正确
        df_2025['F_DATE'] = pd.to_datetime(df_2025['F_DATE'].astype(str), format='%Y%m%d', errors='coerce')
        
        # 按日期和线路汇总客运量
        daily_line_2025 = df_2025.groupby(['F_DATE', 'F_LINENAME']).agg(
            passenger_total=('F_KLCOUNT', 'sum')  # 客运量
        ).reset_index().sort_values(['F_DATE', 'F_LINENAME'])
        
        # 获取所有线路
        lines_2025 = sorted(daily_line_2025['F_LINENAME'].unique())
        
        # 为不同线路设置颜色（使用颜色映射）
        colors_2025 = cm.get_cmap('tab20', len(lines_2025))
        line_colors_map = {line: colors_2025(i) for i, line in enumerate(lines_2025)}
        
        # 创建图表
        fig_2025 = plt.figure(figsize=(20, 10))
        ax_2025 = fig_2025.add_subplot(1, 1, 1)
        
        # 为每条线路绘制数据
        for line_name in lines_2025:
            line_data_2025 = daily_line_2025[daily_line_2025['F_LINENAME'] == line_name].sort_values('F_DATE')
            
            if not line_data_2025.empty:
                dates_plot = line_data_2025['F_DATE']
                values_plot = line_data_2025['passenger_total']
                
                ax_2025.plot(dates_plot, values_plot, 
                           marker='o', linewidth=2, markersize=4,
                           color=line_colors_map[line_name],
                           label=line_name, alpha=0.7)
        
        ax_2025.set_xlabel('时间（日期）', fontsize=14, fontproperties=my_font)
        ax_2025.set_ylabel('客运量（人次）', fontsize=14, fontproperties=my_font)
        ax_2025.set_title('2025年全年节假日客流情况（不同线路不同颜色）', 
                       fontsize=16, fontweight='bold', fontproperties=my_font)
        ax_2025.legend(prop=my_font, fontsize=10, loc='best', ncol=3)
        ax_2025.grid(True, alpha=0.3)
        ax_2025.ticklabel_format(style='plain', axis='y')
        
        # 格式化x轴日期
        ax_2025.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_2025.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax_2025.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9, fontproperties=my_font)
        
        plt.tight_layout()
        
        # 保存图表
        output_file_2025 = '2025年全年节假日客流情况.png'
        plt.savefig(output_file_2025, dpi=300, bbox_inches='tight')
        print(f"✓ 2025年节假日客流图表已保存至: {output_file_2025}")
        plt.close(fig_2025)
        
        # 保存数据到CSV
        csv_2025_df = daily_line_2025.copy()
        csv_2025_df['日期'] = csv_2025_df['F_DATE'].dt.strftime('%Y-%m-%d')
        csv_2025_df = csv_2025_df.rename(columns={
            'F_LINENAME': '线路名称',
            'passenger_total': '客运量'
        })
        # 重新排列列顺序
        csv_2025_df = csv_2025_df[['日期', '线路名称', '客运量']]
        csv_2025_output_file = '2025年全年节假日客流数据.csv'
        csv_2025_df.to_csv(csv_2025_output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 2025年节假日客流数据已保存至: {csv_2025_output_file}")
        
        # 打印统计信息
        print(f"\n2025年节假日数据统计:")
        print(f"  日期范围: {daily_line_2025['F_DATE'].min().strftime('%Y-%m-%d')} 至 {daily_line_2025['F_DATE'].max().strftime('%Y-%m-%d')}")
        print(f"  线路数量: {len(lines_2025)}")
        print(f"  日期数量: {daily_line_2025['F_DATE'].nunique()}")
        print(f"  总客运量: {daily_line_2025['passenger_total'].sum():,.0f} 人次")
    else:
        print("未查询到2025年节假日数据")
        
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
