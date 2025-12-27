"""
绘制"侯家塘"地铁站指定日期客流情况
包含日客流趋势、月度统计、年度对比等多个维度的可视化，并输出详细表格
"""
import pandas as pd
import pymssql
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入字体工具和数据库工具
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

# 定义数据库连接函数
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

# 导入编码修复函数
try:
    from db_utils import fix_dataframe_encoding
except ImportError:
    def fix_dataframe_encoding(df):
        return df

# 配置中文字体
configure_fonts()
my_font = get_chinese_font()
if my_font is not None:
    font_name = my_font.get_name()
    plt.rcParams['font.sans-serif'] = ['SimHei', font_name]
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family=plt.rcParams['font.sans-serif'][0])

# 不使用科学计数法
plt.ticklabel_format(style='plain', axis='y')

# 指定的日期列表
SPECIFIC_DATES = [
    '2022-06-17', '2022-08-07', '2022-11-06', '2022-12-05', '2022-12-10',
    '2023-04-29', '2023-07-07', '2023-07-08', '2023-07-09', '2023-07-14',
    '2023-07-16', '2023-07-23', '2023-07-26', '2023-07-30', '2023-08-06',
    '2023-08-23', '2023-09-02', '2023-09-12', '2023-10-27', '2023-10-28',
    '2023-10-29', '2024-04-12', '2024-04-13', '2024-04-14', '2024-05-30',
    '2024-05-31', '2024-06-01', '2024-06-02', '2024-06-14', '2024-06-15',
    '2024-06-16', '2024-11-24', '2024-11-25', '2024-11-26', '2024-11-27',
    '2024-12-14', '2024-12-15', '2025-03-22', '2025-03-23', '2025-05-10',
    '2025-05-21', '2025-05-22', '2025-05-23', '2025-05-24', '2025-05-25',
    '2025-06-21', '2025-06-22', '2025-08-15', '2025-08-16', '2025-08-17',
    '2025-09-07', '2025-10-05', '2025-10-10', '2025-10-18', '2025-10-31',
    '2025-11-08', '2025-11-14', '2025-11-15', '2025-11-16', '2025-11-22',
    '2025-11-29', '2025-12-03', '2025-12-04', '2025-12-05', '2025-12-06',
    '2025-12-07', '2025-12-08', '2025-12-14', '2025-12-20', '2025-12-24'
]

# 转换为日期格式
specific_dates_dt = [datetime.strptime(d, '%Y-%m-%d') for d in SPECIFIC_DATES]

# 侯家塘地铁站的名称
STATION_NAME = "侯家塘"

# 获取查询的日期范围
start_date = min(SPECIFIC_DATES)
end_date = max(SPECIFIC_DATES)

print(f"正在查询{STATION_NAME}地铁站指定日期客流数据...")
print(f"共 {len(SPECIFIC_DATES)} 个指定日期")
print(f"日期范围: {start_date} 至 {end_date}")

# 连接数据库并查询数据
try:
    conn = get_db_conn_script(charset='utf8')
    print("✓ 数据库连接成功")
    
    # 查询侯家塘地铁站小时客流数据
    select_sql = """
    SELECT 
        [ID],
        [LINE_ID],
        [STATION_ID],
        [STATION_NAME],
        [ENTRY_NUM],
        [EXIT_NUM],
        [CHANGE_NUM],
        [PASSENGER_NUM],
        [FLOW_NUM],
        [SQUAD_DATE]
    FROM [StationFlowPredict].[dbo].[STATION_FLOW_HISTORY]
    WHERE [STATION_NAME] = %s
      AND [SQUAD_DATE] >= %s
      AND [SQUAD_DATE] <= %s
    ORDER BY [SQUAD_DATE]
    """
    
    df = pd.read_sql(select_sql, conn, params=(STATION_NAME, start_date, end_date))
    conn.close()
    
    # 应用编码修复
    df = fix_dataframe_encoding(df)
    
    if df.empty:
        print(f"未找到{STATION_NAME}地铁站数据")
    else:
        print(f"成功获取 {len(df)} 条小时数据记录")
        
        # 日期处理
        df["SQUAD_DATE"] = pd.to_datetime(df["SQUAD_DATE"])
        
        # 按天统计
        df_day = df.groupby(df["SQUAD_DATE"].dt.date).agg({
            "ENTRY_NUM": "sum",
            "EXIT_NUM": "sum",
            "CHANGE_NUM": "sum",
            "PASSENGER_NUM": "sum",
            "FLOW_NUM": "sum"
        }).reset_index()
        df_day.columns = ["DATE", "ENTRY_NUM", "EXIT_NUM", "CHANGE_NUM", "PASSENGER_NUM", "FLOW_NUM"]
        
        # 转换DATE为datetime类型
        df_day["DATE"] = pd.to_datetime(df_day["DATE"])
        
        # 过滤只保留指定日期
        df_filtered = df_day[df_day["DATE"].isin(specific_dates_dt)].copy()
        df_filtered = df_filtered.sort_values("DATE").reset_index(drop=True)
        
        print(f"\n✓ 成功筛选出 {len(df_filtered)} 个指定日期的数据")
        
        if df_filtered.empty:
            print("警告: 没有找到任何指定日期的数据")
        else:
            # 添加年月列用于分组统计
            df_filtered["YEAR"] = df_filtered["DATE"].dt.year
            df_filtered["MONTH"] = df_filtered["DATE"].dt.month
            df_filtered["YEAR_MONTH"] = df_filtered["DATE"].dt.to_period("M")
            df_filtered["DATE_STR"] = df_filtered["DATE"].dt.strftime("%Y-%m-%d")
            
            # ============ 输出详细表格到CSV ============
            output_table = df_filtered[["DATE_STR", "PASSENGER_NUM", "ENTRY_NUM", 
                                       "EXIT_NUM", "CHANGE_NUM", "FLOW_NUM"]].copy()
            output_table.columns = ["日期", "总客流", "进站量", "出站量", "换乘量", "流动量"]
            
            csv_path = f"{STATION_NAME}_指定日期客流数据表.csv"
            output_table.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ 数据表格已保存至: {os.path.abspath(csv_path)}")
            
            # 在控制台也打印表格
            print("\n" + "="*80)
            print(f"{STATION_NAME}地铁站指定日期客流数据明细表")
            print("="*80)
            print(output_table.to_string(index=False))
            print("="*80)
            
            # ============ 创建多子图布局 ============
            fig = plt.figure(figsize=(20, 12))
            
            # 1. 指定日期客流趋势图（带日期标注）
            ax1 = plt.subplot(2, 2, 1)
            x_indices = np.arange(len(df_filtered))
            ax1.plot(x_indices, df_filtered["PASSENGER_NUM"], 
                    label="总客流", color="#1f77b4", linewidth=2, marker='o', markersize=4)
            ax1.plot(x_indices, df_filtered["ENTRY_NUM"], 
                    label="进站量", color="#2ca02c", linewidth=1.5, marker='s', markersize=3, alpha=0.7)
            ax1.plot(x_indices, df_filtered["EXIT_NUM"], 
                    label="出站量", color="#ff7f0e", linewidth=1.5, marker='^', markersize=3, alpha=0.7)
            ax1.plot(x_indices, df_filtered["CHANGE_NUM"], 
                    label="换乘量", color="#d62728", linewidth=1.5, marker='d', markersize=3, alpha=0.7)
            
            ax1.set_title(f"{STATION_NAME}地铁站指定日期客流趋势 (共{len(df_filtered)}天)", 
                         fontsize=14, fontproperties=my_font, pad=15)
            ax1.set_xlabel("日期序号", fontsize=12, fontproperties=my_font)
            ax1.set_ylabel("客流量（人次）", fontsize=12, fontproperties=my_font)
            
            # 设置x轴标签（显示部分日期避免拥挤）
            step = max(1, len(df_filtered) // 15)
            ax1.set_xticks(x_indices[::step])
            ax1.set_xticklabels(df_filtered["DATE_STR"].iloc[::step], 
                               rotation=45, ha='right', fontproperties=my_font, fontsize=8)
            
            ax1.legend(prop=my_font, loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.ticklabel_format(style='plain', axis='y')
            
            # 2. 月度统计柱状图
            ax2 = plt.subplot(2, 2, 2)
            df_monthly = df_filtered.groupby("YEAR_MONTH").agg({
                "PASSENGER_NUM": ["sum", "mean", "count"],
                "ENTRY_NUM": "sum",
                "EXIT_NUM": "sum"
            }).reset_index()
            df_monthly.columns = ["YEAR_MONTH", "PASSENGER_SUM", "PASSENGER_MEAN", 
                                 "DAY_COUNT", "ENTRY_SUM", "EXIT_SUM"]
            df_monthly["YEAR_MONTH_STR"] = df_monthly["YEAR_MONTH"].astype(str)
            
            x_pos = np.arange(len(df_monthly))
            width = 0.25
            ax2.bar(x_pos - width, df_monthly["PASSENGER_SUM"]/10000, width, 
                   label=f"总客流（万人次）", color="#1f77b4", alpha=0.8)
            ax2.bar(x_pos, df_monthly["ENTRY_SUM"]/10000, width, 
                   label=f"进站量（万人次）", color="#2ca02c", alpha=0.8)
            ax2.bar(x_pos + width, df_monthly["EXIT_SUM"]/10000, width, 
                   label=f"出站量（万人次）", color="#ff7f0e", alpha=0.8)
            
            # 在柱子上标注天数
            for i, count in enumerate(df_monthly["DAY_COUNT"]):
                ax2.text(i, ax2.get_ylim()[1]*0.95, f'{int(count)}天', 
                        ha='center', va='top', fontsize=8, fontproperties=my_font)
            
            ax2.set_title(f"{STATION_NAME}地铁站指定日期月度统计", 
                         fontsize=14, fontproperties=my_font, pad=15)
            ax2.set_xlabel("年月", fontsize=12, fontproperties=my_font)
            ax2.set_ylabel("客流量（万人次）", fontsize=12, fontproperties=my_font)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(df_monthly["YEAR_MONTH_STR"], 
                               rotation=45, ha='right', fontproperties=my_font)
            ax2.legend(prop=my_font, loc='upper left')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.ticklabel_format(style='plain', axis='y')
            
            # 3. 年度对比
            ax3 = plt.subplot(2, 2, 3)
            df_yearly = df_filtered.groupby("YEAR").agg({
                "PASSENGER_NUM": ["sum", "mean", "count"],
                "ENTRY_NUM": "sum",
                "EXIT_NUM": "sum"
            }).reset_index()
            df_yearly.columns = ["YEAR", "PASSENGER_SUM", "PASSENGER_MEAN", 
                                "DAY_COUNT", "ENTRY_SUM", "EXIT_SUM"]
            
            years = df_yearly["YEAR"].astype(str)
            x_pos = np.arange(len(years))
            width = 0.35
            
            ax3.bar(x_pos - width/2, df_yearly["PASSENGER_SUM"] / 10000, width, 
                   label="总客流（万人次）", color="#1f77b4", alpha=0.8)
            ax3.bar(x_pos + width/2, df_yearly["ENTRY_SUM"] / 10000, width, 
                   label="进站量（万人次）", color="#2ca02c", alpha=0.8)
            
            ax3.set_title(f"{STATION_NAME}地铁站指定日期年度对比", 
                         fontsize=14, fontproperties=my_font, pad=15)
            ax3.set_xlabel("年份", fontsize=12, fontproperties=my_font)
            ax3.set_ylabel("客流量（万人次）", fontsize=12, fontproperties=my_font)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(years, fontproperties=my_font)
            ax3.legend(prop=my_font)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 在柱状图上添加数值和天数
            for i, (year, total, entry, count) in enumerate(zip(years, 
                                                                df_yearly["PASSENGER_SUM"] / 10000,
                                                                df_yearly["ENTRY_SUM"] / 10000,
                                                                df_yearly["DAY_COUNT"])):
                ax3.text(i - width/2, total, f'{total:.1f}\n({int(count)}天)', 
                        ha='center', va='bottom', fontsize=9, fontproperties=my_font)
                ax3.text(i + width/2, entry, f'{entry:.1f}', 
                        ha='center', va='bottom', fontsize=9, fontproperties=my_font)
            
            # 4. 客流分布箱线图
            ax4 = plt.subplot(2, 2, 4)
            
            # 按年份分组的箱线图数据
            box_data = [df_filtered[df_filtered["YEAR"] == year]["PASSENGER_NUM"].values 
                       for year in sorted(df_filtered["YEAR"].unique())]
            box_labels = [str(year) for year in sorted(df_filtered["YEAR"].unique())]
            
            bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True,
                           notch=True, showmeans=True)
            
            # 美化箱线图
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax4.set_title(f"{STATION_NAME}地铁站指定日期客流分布", 
                         fontsize=14, fontproperties=my_font, pad=15)
            ax4.set_xlabel("年份", fontsize=12, fontproperties=my_font)
            ax4.set_ylabel("总客流量（人次）", fontsize=12, fontproperties=my_font)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.ticklabel_format(style='plain', axis='y')
            
            # 添加图例说明
            ax4.text(0.02, 0.98, '○ 均值\n━ 中位数', 
                    transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', fontproperties=my_font,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.suptitle(f"{STATION_NAME}地铁站指定日期客流分析（共{len(df_filtered)}天）", 
                        fontsize=16, fontproperties=my_font, y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            # 保存图片
            output_path = f"{STATION_NAME}_指定日期客流分析.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ 图表已成功保存至: {os.path.abspath(output_path)}")
            
            # ============ 显示统计信息 ============
            print("\n" + "="*80)
            print("统计摘要")
            print("="*80)
            print(f"指定日期总数: {len(SPECIFIC_DATES)} 天")
            print(f"有数据日期数: {len(df_filtered)} 天")
            print(f"数据时间范围: {df_filtered['DATE'].min().strftime('%Y-%m-%d')} 至 {df_filtered['DATE'].max().strftime('%Y-%m-%d')}")
            
            print(f"\n总客流统计:")
            print(f"  日均: {df_filtered['PASSENGER_NUM'].mean():.0f} 人次")
            print(f"  最大值: {df_filtered['PASSENGER_NUM'].max():.0f} 人次 ({df_filtered.loc[df_filtered['PASSENGER_NUM'].idxmax(), 'DATE_STR']})")
            print(f"  最小值: {df_filtered['PASSENGER_NUM'].min():.0f} 人次 ({df_filtered.loc[df_filtered['PASSENGER_NUM'].idxmin(), 'DATE_STR']})")
            print(f"  总计: {df_filtered['PASSENGER_NUM'].sum() / 10000:.2f} 万人次")
            print(f"  标准差: {df_filtered['PASSENGER_NUM'].std():.0f} 人次")
            
            print(f"\n进站量统计:")
            print(f"  日均: {df_filtered['ENTRY_NUM'].mean():.0f} 人次")
            print(f"  总计: {df_filtered['ENTRY_NUM'].sum() / 10000:.2f} 万人次")
            
            print(f"\n出站量统计:")
            print(f"  日均: {df_filtered['EXIT_NUM'].mean():.0f} 人次")
            print(f"  总计: {df_filtered['EXIT_NUM'].sum() / 10000:.2f} 万人次")
            
            print(f"\n换乘量统计:")
            print(f"  日均: {df_filtered['CHANGE_NUM'].mean():.0f} 人次")
            print(f"  总计: {df_filtered['CHANGE_NUM'].sum() / 10000:.2f} 万人次")
            
            # 年度对比
            print(f"\n年度对比:")
            for _, row in df_yearly.iterrows():
                print(f"  {int(row['YEAR'])}年: {int(row['DAY_COUNT'])}天数据, "
                      f"总客流 {row['PASSENGER_SUM']/10000:.2f}万人次, "
                      f"日均 {row['PASSENGER_MEAN']:.0f}人次")
            
            # 月度统计
            print(f"\n月度统计（前5个月）:")
            for _, row in df_monthly.head(5).iterrows():
                print(f"  {row['YEAR_MONTH_STR']}: {int(row['DAY_COUNT'])}天数据, "
                      f"总客流 {row['PASSENGER_SUM']/10000:.2f}万人次, "
                      f"日均 {row['PASSENGER_MEAN']:.0f}人次")
            if len(df_monthly) > 5:
                print(f"  ... (共{len(df_monthly)}个月)")
            
            print("="*80)
            
            # 关闭图形
            plt.close()
            print("\n✓ 所有分析完成！")
            
except pymssql.Error as db_error:
    print(f"数据库连接错误: {db_error}")
    print("请检查:")
    print("1. 数据库服务器是否可访问")
    print("2. 数据库连接配置是否正确")
    print("3. 网络连接是否正常")
    import traceback
    traceback.print_exc()
except ImportError as import_error:
    print(f"导入模块错误: {import_error}")
    print("请确保已安装必要的依赖包:")
    print("  pip install pandas pymssql matplotlib numpy")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()
