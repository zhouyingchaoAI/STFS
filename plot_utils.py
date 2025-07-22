# 可视化模块：生成小时和日客流预测图表
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from font_utils import get_chinese_font, configure_fonts

def plot_hourly_predictions(result: Dict, line_name_map: Dict, predict_date: str, save_path: str = "timeseries_predict_hourly.png") -> None:
    """
    绘制小时客流预测图

    参数:
        result: 预测结果字典
        line_name_map: 线路编号到名称的映射
        predict_date: 预测日期 (YYYYMMDD)
        save_path: 图表保存路径
    """
    configure_fonts()
    my_font = get_chinese_font()
    # 解决中文乱码问题
    if my_font is not None:
        font_name = my_font.get_name()
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        matplotlib.rc('font', family=font_name)
    else:
        # 若未找到中文字体，使用常见的中文字体尝试
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

    lines = sorted(result.keys())
    n_lines = len(lines)
    ncols = 2 if n_lines > 1 else 1
    nrows = int(np.ceil(n_lines / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, line in enumerate(lines):
        line_name = line_name_map.get(line, line)
        info = result[line]
        if info.get("error"):
            continue

        ax = axes[idx]
        try:
            line_data = info.get("line_data", pd.DataFrame())
            predict_dt = datetime.strptime(predict_date, '%Y%m%d')
            prev_date = (predict_dt - timedelta(days=1)).strftime('%Y%m%d')
            prev_day_data = line_data[line_data['F_DATE'] == prev_date]
            prev_hours = [int(h) for h in prev_day_data['F_HOUR']] if not prev_day_data.empty else []
            prev_flows = prev_day_data['F_KLCOUNT'].tolist() if not prev_day_data.empty else []
        except Exception:
            prev_hours = []
            prev_flows = []

        pred_hours = list(range(24))
        pred_flows = list(info["predict_hourly_flow"].values())

        if prev_hours:
            ax.plot(prev_hours, prev_flows, marker='o', label='前一日客流', color='tab:gray', linewidth=2)
        ax.plot(pred_hours, pred_flows, marker='o', label=f'{info["algorithm"].upper()}预测', color='tab:blue', linewidth=2)
        ax.set_title(f"{line_name} 小时客流预测", fontsize=14, fontproperties=my_font if my_font is not None else None)
        ax.set_xlabel("小时", fontsize=12, fontproperties=my_font if my_font is not None else None)
        ax.set_ylabel("客流量", fontsize=12, fontproperties=my_font if my_font is not None else None)
        ax.set_xticks(range(0, 24, 1))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], rotation=0, fontsize=10, fontproperties=my_font if my_font is not None else None)
        ax.grid(True, alpha=0.3)
        ax.legend(prop=my_font if my_font is not None else None)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"各线路小时客流预测 (预测日: {predict_date})", fontsize=16, fontproperties=my_font if my_font is not None else None)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_daily_predictions(result: Dict, line_name_map: Dict, predict_start_date: str, days: int, save_path: str = "timeseries_predict_daily.png") -> None:
    """
    绘制日客流预测图

    参数:
        result: 预测结果字典
        line_name_map: 线路编号到名称的映射
        predict_start_date: 预测起始日期 (YYYYMMDD)
        days: 预测天数
        save_path: 图表保存路径
    """
    configure_fonts()
    my_font = get_chinese_font()
    # 解决中文乱码问题
    if my_font is not None:
        font_name = my_font.get_name()
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        matplotlib.rc('font', family=font_name)
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

    lines = sorted(result.keys())
    n_lines = len(lines)
    ncols = 2 if n_lines > 1 else 1
    nrows = int(np.ceil(n_lines / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for idx, line in enumerate(lines):
        line_name = line_name_map.get(line, line)
        info = result[line]
        if info.get("error"):
            continue

        ax = axes[idx]
        try:
            line_data = info.get("line_data", pd.DataFrame())
            line_data_sorted = line_data.sort_values('F_DATE')
            line_data_hist = line_data_sorted[line_data_sorted['F_DATE'] < predict_start_date].tail(15)
            hist_dates = [datetime.strptime(str(d), '%Y%m%d') for d in line_data_hist['F_DATE']]
            hist_flows = line_data_hist['F_KLCOUNT'].tolist()
        except Exception:
            hist_dates = []
            hist_flows = []

        daily_flow = info["predict_daily_flow"]
        pred_dates = [datetime.strptime(d, '%Y%m%d') for d in daily_flow.keys()]
        pred_flows = list(daily_flow.values())
        all_dates = hist_dates + pred_dates
        all_labels = [dt.strftime('%Y-%m-%d') for dt in all_dates]

        if hist_dates:
            ax.plot([dt.strftime('%Y-%m-%d') for dt in hist_dates], hist_flows, marker='o', label='历史客流', color='tab:gray', linewidth=2)
        ax.plot([dt.strftime('%Y-%m-%d') for dt in pred_dates], pred_flows, marker='o', label='KNN预测', color='tab:blue', linewidth=2)
        if hist_dates and pred_dates:
            ax.axvline(x=pred_dates[0].strftime('%Y-%m-%d'), color='red', linestyle='--', linewidth=1, label='预测起点')
        ax.set_title(f"{line_name} 历史与KNN日预测", fontsize=14, fontproperties=my_font if my_font is not None else None)
        ax.set_xlabel("日期", fontsize=12, fontproperties=my_font if my_font is not None else None)
        ax.set_ylabel("客流量", fontsize=12, fontproperties=my_font if my_font is not None else None)
        ax.set_xticks(all_labels)
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10, fontproperties=my_font if my_font is not None else None)
        ax.grid(True, alpha=0.3)
        ax.legend(prop=my_font if my_font is not None else None)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"各线路历史与KNN日预测客流量 (预测起始日: {predict_start_date})", fontsize=16, fontproperties=my_font if my_font is not None else None)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()