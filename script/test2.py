import pandas as pd
import pymssql
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from font_utils import configure_fonts
import os
import math

def plot_ticklabel_format_plain(ax=None):
    """
    设置y轴不使用科学计数法
    """
    if ax is None:
        plt.ticklabel_format(style='plain', axis='y')
    else:
        ax.ticklabel_format(style='plain', axis='y')
    try:
        matplotlib.ticker.ScalarFormatter(useMathText=False)
    except Exception:
        pass

def read_line_daily_flow_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    查询指定期间的日客流历史数据

    参数:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)

    返回:
        日客流数据的 DataFrame
    """
    try:
        conn = pymssql.connect(
            server='192.168.10.76',
            user='sa',
            password='Chency@123',
            database='master',
            port='1433'
        )
        query = f"""
        SELECT 
            L.ID,
            L.F_DATE,
            L.F_LB,
            L.F_LINENO,
            L.F_LINENAME,
            L.F_KLCOUNT,
            L.CREATETIME,
            L.CREATOR,
            C.F_WEEK,
            C.F_DATEFEATURES,
            C.F_HOLIDAYTYPE,
            C.F_ISHOLIDAY,
            C.F_ISNONGLI,
            C.F_ISYANGLI,
            C.F_NEXTDAY,
            C.F_HOLIDAYDAYS,
            C.F_HOLIDAYTHDAY,
            C.IS_FIRST
        FROM 
            dbo.LineDailyFlowHistory AS L
        LEFT JOIN 
            dbo.LSTM_COMMON_HOLIDAYFEATURE AS C
            ON L.F_DATE = C.F_DATE
        WHERE 
            L.CREATOR = 'chency'
            AND L.F_DATE >= '{start_date}'
            AND L.F_DATE <= '{end_date}'
        ORDER BY 
            L.F_DATE, L.F_LINENO
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        raise RuntimeError(f"数据库读取失败: {e}")

def get_key_xticks(days, start_dates, date_lists):
    """
    生成关键时间节点的x轴刻度和标签
    - 只显示第1天、第days/4、第days/2、第3*days/4、最后一天
    - 标签显示实际日期（如3月1日、3月23日、4月15日、5月7日、5月29日）
    """
    idxs = sorted(set([
        0,
        days // 4,
        days // 2,
        (3 * days) // 4,
        days - 1
    ]))
    idxs = [i for i in idxs if i < days]
    key_dates = [date_lists[0][i] for i in idxs]
    key_labels = []
    for d in key_dates:
        dt = datetime.strptime(d, "%Y%m%d")
        key_labels.append(f"{dt.month}月{dt.day}日")
    return idxs, key_labels

def plot_compare_by_start_and_days_multi(
    dfs: list, start_dates: list, days: int, labels: list, save_dir: str = "compare_plots"
):
    """
    对比绘制多个时间段的日客流量，不同线路绘制在不同子图，横坐标为关键时间节点
    图片保存到指定目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 优化中文乱码问题
    configure_fonts()

    # 确保F_DATE为字符串
    for df in dfs:
        df['F_DATE'] = df['F_DATE'].astype(str)

    # 获取所有线路集合
    lines_list = [set(df['F_LINENO'].unique()) for df in dfs]
    common_lines = sorted(set.intersection(*lines_list)) if len(lines_list) > 1 else sorted(lines_list[0])

    # 生成日期序列
    date_lists = [
        [(datetime.strptime(start_date, "%Y%m%d") + timedelta(days=i)).strftime("%Y%m%d") for i in range(days)]
        for start_date in start_dates
    ]

    x = list(range(days))
    key_xticks, key_xticklabels = get_key_xticks(days, start_dates, date_lists)

    # 不再自动检测get_chinese_font，直接用matplotlib全局字体
    font_kwargs = {}

    color_map = plt.get_cmap('tab20')

    if common_lines:
        n_lines = len(common_lines)
        ncols = 2 if n_lines > 1 else 1
        nrows = math.ceil(n_lines / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows), squeeze=False)

        for idx, line in enumerate(common_lines):
            row_idx = idx // ncols
            col_idx = idx % ncols
            ax = axes[row_idx][col_idx]

            line_names = []
            klcounts_all = []
            for i, df in enumerate(dfs):
                df_line = df[(df['F_LINENO'] == line) & (df['F_DATE'].isin(date_lists[i]))].sort_values('F_DATE')
                line_name = df_line['F_LINENAME'].iloc[0] if not df_line.empty else str(line)
                line_names.append(line_name)
                klcount = []
                for d in date_lists[i]:
                    row = df_line[df_line['F_DATE'] == d]
                    klcount.append(int(row['F_KLCOUNT'].iloc[0]) if not row.empty else 0)
                klcounts_all.append(klcount)

            line_name = next((name for name in line_names if name), str(line))

            for i, klcount in enumerate(klcounts_all):
                color = color_map(i)
                linestyle = ['-', '--', '-.', ':'][i % 4]
                ax.plot(
                    x, klcount, marker='o', label=f"{labels[i]}（{start_dates[i]}起）",
                    color=color, linewidth=2, linestyle=linestyle
                )

            plot_ticklabel_format_plain(ax)

            title_dates = " vs ".join(start_dates)
            ax.set_title(f"{line_name} 线路日客流对比\n({title_dates}，共{days}天)", fontsize=14, **font_kwargs)
            ax.set_xlabel("天数", fontsize=11, **font_kwargs)
            ax.set_ylabel("客流量", fontsize=11, **font_kwargs)
            ax.set_xticks(key_xticks)
            ax.set_xticklabels(key_xticklabels, **font_kwargs)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

        for idx in range(n_lines, nrows * ncols):
            row_idx = idx // ncols
            col_idx = idx % ncols
            fig.delaxes(axes[row_idx][col_idx])

        plt.tight_layout()
        fname = f"lines_subplots_{'_vs_'.join(start_dates)}_{days}days.png"
        fname = fname.replace("/", "-").replace("\\", "-")
        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {os.path.join(save_dir, fname)}")
    else:
        idx_first = next((i for i, df in enumerate(dfs) if len(df) > 0), None)
        if idx_first is not None:
            lines = sorted(lines_list[idx_first])
            label = labels[idx_first]
            start_date = start_dates[idx_first]
            date_list = date_lists[idx_first]
            df = dfs[idx_first]
        else:
            print("没有可用的线路数据，无法绘图。")
            return

        n_lines = len(lines)
        ncols = 2 if n_lines > 1 else 1
        nrows = math.ceil(n_lines / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows), squeeze=False)

        for idx, line in enumerate(lines):
            row_idx = idx // ncols
            col_idx = idx % ncols
            ax = axes[row_idx][col_idx]

            df_line = df[(df['F_LINENO'] == line) & (df['F_DATE'].isin(date_list))].sort_values('F_DATE')
            line_name = df_line['F_LINENAME'].iloc[0] if not df_line.empty else str(line)

            klcount = []
            for d in date_list:
                row = df_line[df_line['F_DATE'] == d]
                klcount.append(int(row['F_KLCOUNT'].iloc[0]) if not row.empty else 0)

            color1 = color_map(0)
            ax.plot(x, klcount, marker='o', label=f"{label}（{start_date}起）", color=color1, linewidth=2)

            plot_ticklabel_format_plain(ax)

            ax.set_title(f"{line_name} 线路日客流\n({start_date}，共{days}天)", fontsize=14, **font_kwargs)
            ax.set_xlabel("天数", fontsize=11, **font_kwargs)
            ax.set_ylabel("客流量", fontsize=11, **font_kwargs)
            ax.set_xticks(key_xticks)
            ax.set_xticklabels(key_xticklabels, **font_kwargs)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

        for idx in range(n_lines, nrows * ncols):
            row_idx = idx // ncols
            col_idx = idx % ncols
            fig.delaxes(axes[row_idx][col_idx])

        plt.tight_layout()
        fname = f"lines_single_{start_date}_{days}days.png"
        fname = fname.replace("/", "-").replace("\\", "-")
        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {os.path.join(save_dir, fname)}")

def main():
    # 设置四个开始日期和天数（加上2025年）
    # start_date1 = '20220101'
    start_date2 = '20230101'
    start_date3 = '20240101'
    start_date4 = '20250101'
    days = 90

    # end_date1 = (datetime.strptime(start_date1, "%Y%m%d") + timedelta(days=days-1)).strftime("%Y%m%d")
    end_date2 = (datetime.strptime(start_date2, "%Y%m%d") + timedelta(days=days-1)).strftime("%Y%m%d")
    end_date3 = (datetime.strptime(start_date3, "%Y%m%d") + timedelta(days=days-1)).strftime("%Y%m%d")
    end_date4 = (datetime.strptime(start_date4, "%Y%m%d") + timedelta(days=days-1)).strftime("%Y%m%d")

    # df_2022 = read_line_daily_flow_history(start_date1, end_date1)
    df_2023 = read_line_daily_flow_history(start_date2, end_date2)
    df_2024 = read_line_daily_flow_history(start_date3, end_date3)
    df_2025 = read_line_daily_flow_history(start_date4, end_date4)

    plot_compare_by_start_and_days_multi(
        [df_2023, df_2024, df_2025],
        [start_date2, start_date3, start_date4],
        days,
        labels=[f"{start_date2}起", f"{start_date3}起", f"{start_date4}起"]
    )

if __name__ == "__main__":
    main()