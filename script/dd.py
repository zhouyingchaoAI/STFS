"""
绘制近三年"侯家塘"地铁站客流情况
包含日客流趋势、月度统计、年度对比等多个维度的可视化
"""
import pandas as pd
import pymssql
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os

# 添加项目根目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入字体工具和数据库工具
try:
    from font_utils import get_chinese_font, configure_fonts
except ImportError:
    # 如果导入失败，尝试从script目录导入
    try:
        from script.font_utils import get_chinese_font, configure_fonts
    except ImportError:
        # 如果还是失败，定义简化版本
        def get_chinese_font():
            return None
        def configure_fonts():
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

# 定义数据库连接函数（使用正确的服务器地址）
def get_db_conn_script(charset='utf8'):
    """
    脚本专用的数据库连接函数，使用正确的服务器地址
    """
    DB_CONFIG = {
        "server": "10.1.6.230",  # 脚本专用数据库服务器
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
    # 如果导入失败，使用简化版本
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

# 获取当前日期并定位近三年
today = datetime.today()
start_date = (today.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=365*3)).strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")

# 侯家塘地铁站的名称
STATION_NAME = "侯家塘"

print(f"正在查询{STATION_NAME}地铁站近3年客流数据...")
print(f"查询时间范围: {start_date} 至 {end_date}")

# 连接数据库并查询数据
try:
    # 使用脚本专用的数据库连接（直接连接正确的服务器）
    conn = get_db_conn_script(charset='utf8')
    print("✓ 数据库连接成功")
    
    # 查询近3年侯家塘地铁站小时客流数据
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
        print(f"未找到近3年{STATION_NAME}地铁站数据")
    else:
        print(f"成功获取 {len(df)} 条小时数据记录")
        
        # 日期处理
        df["SQUAD_DATE"] = pd.to_datetime(df["SQUAD_DATE"])
        
        # 按天统计
        df_day = df.groupby("SQUAD_DATE").agg({
            "ENTRY_NUM": "sum",
            "EXIT_NUM": "sum",
            "CHANGE_NUM": "sum",
            "PASSENGER_NUM": "sum",
            "FLOW_NUM": "sum"
        }).reset_index()
        df_day = df_day.sort_values("SQUAD_DATE")
        
        # 添加年月列用于分组统计
        df_day["YEAR"] = df_day["SQUAD_DATE"].dt.year
        df_day["MONTH"] = df_day["SQUAD_DATE"].dt.month
        df_day["YEAR_MONTH"] = df_day["SQUAD_DATE"].dt.to_period("M")
        
        # 创建多子图布局
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 日客流趋势图（总客流、进站、出站、换乘）
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(df_day["SQUAD_DATE"], df_day["PASSENGER_NUM"], 
                label="总客流", color="#1f77b4", linewidth=1.5, alpha=0.8)
        ax1.plot(df_day["SQUAD_DATE"], df_day["ENTRY_NUM"], 
                label="进站量", color="#2ca02c", linewidth=1.5, alpha=0.7)
        ax1.plot(df_day["SQUAD_DATE"], df_day["EXIT_NUM"], 
                label="出站量", color="#ff7f0e", linewidth=1.5, alpha=0.7)
        ax1.plot(df_day["SQUAD_DATE"], df_day["CHANGE_NUM"], 
                label="换乘量", color="#d62728", linewidth=1.5, alpha=0.7)
        ax1.set_title(f"{STATION_NAME}地铁站近3年日客流趋势", 
                     fontsize=14, fontproperties=my_font, pad=15)
        ax1.set_xlabel("日期", fontsize=12, fontproperties=my_font)
        ax1.set_ylabel("客流量（人次）", fontsize=12, fontproperties=my_font)
        ax1.legend(prop=my_font, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='y')
        
        # 2. 月度平均客流对比
        ax2 = plt.subplot(2, 2, 2)
        df_monthly = df_day.groupby("YEAR_MONTH").agg({
            "PASSENGER_NUM": "mean",
            "ENTRY_NUM": "mean",
            "EXIT_NUM": "mean"
        }).reset_index()
        df_monthly["YEAR_MONTH_STR"] = df_monthly["YEAR_MONTH"].astype(str)
        
        x_pos = np.arange(len(df_monthly))
        width = 0.25
        ax2.bar(x_pos - width, df_monthly["PASSENGER_NUM"], width, 
               label="总客流", color="#1f77b4", alpha=0.8)
        ax2.bar(x_pos, df_monthly["ENTRY_NUM"], width, 
               label="进站量", color="#2ca02c", alpha=0.8)
        ax2.bar(x_pos + width, df_monthly["EXIT_NUM"], width, 
               label="出站量", color="#ff7f0e", alpha=0.8)
        
        ax2.set_title(f"{STATION_NAME}地铁站近3年月度平均客流", 
                     fontsize=14, fontproperties=my_font, pad=15)
        ax2.set_xlabel("年月", fontsize=12, fontproperties=my_font)
        ax2.set_ylabel("平均客流量（人次/天）", fontsize=12, fontproperties=my_font)
        ax2.set_xticks(x_pos[::3])  # 每3个月显示一个标签
        ax2.set_xticklabels(df_monthly["YEAR_MONTH_STR"][::3], 
                           rotation=45, ha='right', fontproperties=my_font)
        ax2.legend(prop=my_font)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.ticklabel_format(style='plain', axis='y')
        
        # 3. 年度对比（按年份统计）
        ax3 = plt.subplot(2, 2, 3)
        df_yearly = df_day.groupby("YEAR").agg({
            "PASSENGER_NUM": ["sum", "mean", "max", "min"],
            "ENTRY_NUM": "sum",
            "EXIT_NUM": "sum"
        }).reset_index()
        df_yearly.columns = ["YEAR", "PASSENGER_SUM", "PASSENGER_MEAN", 
                            "PASSENGER_MAX", "PASSENGER_MIN", "ENTRY_SUM", "EXIT_SUM"]
        
        years = df_yearly["YEAR"].astype(str)
        x_pos = np.arange(len(years))
        width = 0.35
        
        ax3.bar(x_pos - width/2, df_yearly["PASSENGER_SUM"] / 10000, width, 
               label="总客流（万人次）", color="#1f77b4", alpha=0.8)
        ax3.bar(x_pos + width/2, df_yearly["ENTRY_SUM"] / 10000, width, 
               label="进站量（万人次）", color="#2ca02c", alpha=0.8)
        
        ax3.set_title(f"{STATION_NAME}地铁站近3年年度客流对比", 
                     fontsize=14, fontproperties=my_font, pad=15)
        ax3.set_xlabel("年份", fontsize=12, fontproperties=my_font)
        ax3.set_ylabel("客流量（万人次）", fontsize=12, fontproperties=my_font)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(years, fontproperties=my_font)
        ax3.legend(prop=my_font)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for i, (year, total, entry) in enumerate(zip(years, df_yearly["PASSENGER_SUM"] / 10000, 
                                                     df_yearly["ENTRY_SUM"] / 10000)):
            ax3.text(i - width/2, total, f'{total:.1f}', 
                    ha='center', va='bottom', fontsize=9, fontproperties=my_font)
            ax3.text(i + width/2, entry, f'{entry:.1f}', 
                    ha='center', va='bottom', fontsize=9, fontproperties=my_font)
        
        # 4. 月度分布热力图（按月份和年份）
        ax4 = plt.subplot(2, 2, 4)
        df_monthly_pivot = df_day.groupby(["YEAR", "MONTH"])["PASSENGER_NUM"].mean().reset_index()
        pivot_table = df_monthly_pivot.pivot(index="MONTH", columns="YEAR", values="PASSENGER_NUM")
        
        im = ax4.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax4.set_title(f"{STATION_NAME}地铁站近3年月度平均客流热力图", 
                     fontsize=14, fontproperties=my_font, pad=15)
        ax4.set_xlabel("年份", fontsize=12, fontproperties=my_font)
        ax4.set_ylabel("月份", fontsize=12, fontproperties=my_font)
        ax4.set_xticks(range(len(pivot_table.columns)))
        ax4.set_xticklabels(pivot_table.columns.astype(str), fontproperties=my_font)
        ax4.set_yticks(range(len(pivot_table.index)))
        ax4.set_yticklabels(pivot_table.index.astype(str), fontproperties=my_font)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('平均客流量（人次/天）', fontsize=10, fontproperties=my_font)
        
        # 在热力图上添加数值
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                if not pd.isna(value):
                    ax4.text(j, i, f'{int(value)}', 
                            ha='center', va='center', 
                            color='white' if value > pivot_table.values.mean() else 'black',
                            fontsize=8, fontproperties=my_font)
        
        plt.suptitle(f"{STATION_NAME}地铁站近3年客流情况综合分析", 
                    fontsize=16, fontproperties=my_font, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # 保存图片
        output_path = f"{STATION_NAME}_近3年客流分析.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 图表已成功保存至: {os.path.abspath(output_path)}")
        
        # 显示统计信息
        print("\n=== 统计摘要 ===")
        print(f"数据时间范围: {df_day['SQUAD_DATE'].min().strftime('%Y-%m-%d')} 至 {df_day['SQUAD_DATE'].max().strftime('%Y-%m-%d')}")
        print(f"总天数: {len(df_day)} 天")
        print(f"\n总客流统计:")
        print(f"  日均: {df_day['PASSENGER_NUM'].mean():.0f} 人次")
        print(f"  最大值: {df_day['PASSENGER_NUM'].max():.0f} 人次 ({df_day.loc[df_day['PASSENGER_NUM'].idxmax(), 'SQUAD_DATE'].strftime('%Y-%m-%d')})")
        print(f"  最小值: {df_day['PASSENGER_NUM'].min():.0f} 人次 ({df_day.loc[df_day['PASSENGER_NUM'].idxmin(), 'SQUAD_DATE'].strftime('%Y-%m-%d')})")
        print(f"  总计: {df_day['PASSENGER_NUM'].sum() / 10000:.1f} 万人次")
        
        print(f"\n进站量统计:")
        print(f"  日均: {df_day['ENTRY_NUM'].mean():.0f} 人次")
        print(f"  总计: {df_day['ENTRY_NUM'].sum() / 10000:.1f} 万人次")
        
        print(f"\n出站量统计:")
        print(f"  日均: {df_day['EXIT_NUM'].mean():.0f} 人次")
        print(f"  总计: {df_day['EXIT_NUM'].sum() / 10000:.1f} 万人次")
        
        print(f"\n换乘量统计:")
        print(f"  日均: {df_day['CHANGE_NUM'].mean():.0f} 人次")
        print(f"  总计: {df_day['CHANGE_NUM'].sum() / 10000:.1f} 万人次")
        
        # 年度对比
        print(f"\n年度对比:")
        for _, row in df_yearly.iterrows():
            print(f"  {int(row['YEAR'])}年: 总客流 {row['PASSENGER_SUM']/10000:.1f}万人次, "
                  f"日均 {row['PASSENGER_MEAN']:.0f}人次")
        
        # 关闭图形以释放内存（不显示窗口，避免在无GUI环境中卡住）
        plt.close()
        print("✓ 图表生成完成")
        
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
