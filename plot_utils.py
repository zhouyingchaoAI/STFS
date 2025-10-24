
# 可视化模块：生成小时和日客流预测图表

from operator import truediv
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from font_utils import get_chinese_font, configure_fonts

FLOW_OPTIONS = {
    "xianwangxianlu": "线路线网",
    "duanmian": "断面",
    "chezhan": "车站",
    "F_PKLCOUNT": "客运量",
    "F_ENTRANCE": "进站量",
    "F_EXIT": "出站量",
    "F_TRANSFER": "换乘量",
    "F_BOARD_ALIGHT": "乘降量"
}


def plot_hourly_predictions(
    result: Dict,
    line_name_map: Dict,
    predict_date: str,
    save_path: str = "timeseries_predict_hourly.png",
    flow_type: str = None, 
    metric_type: str = None,
    show_cur_hist: bool = True,
    show_pred: bool = True,
    show_pred_error: bool = True,
    show_last_year: bool = False,
    show_last_year_offset: bool = False,
    show_last_year_offset_error: bool = False,
) -> None:
    """
    绘制小时客流预测图，并增加上一年历史同小时的曲线和"去年同期+整体偏移"曲线
    并增加"去年同期+整体偏移"与"历史当小时客流"的准确率曲线

    参数:
        result: 预测结果字典
        line_name_map: 线路编号到名称的映射
        predict_date: 预测日期 (YYYYMMDD)
        save_path: 图表保存路径
        show_cur_hist: 是否显示历史当小时客流曲线
        show_pred: 是否显示预测曲线
        show_pred_error: 是否显示预测准确率曲线
        show_last_year: 是否显示去年同期曲线
        show_last_year_offset: 是否显示去年同期+整体偏移曲线
        show_last_year_offset_error: 是否显示去年同期+整体偏移准确率曲线
    """
    # 配置字体和格式
    configure_fonts()
    my_font = get_chinese_font()
    if my_font is not None:
        font_name = my_font.get_name()
        plt.rcParams['font.sans-serif'] = ['SimHei']  
        plt.rcParams['axes.unicode_minus'] = False  
        matplotlib.rc('font', family=font_name)
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei']  
        plt.rcParams['axes.unicode_minus'] = False  

    # 不使用科学计数法
    plt.ticklabel_format(style='plain', axis='y')
    matplotlib.ticker.ScalarFormatter(useMathText=False)

    # 设置子图布局
    lines = sorted(result.keys())
    n_lines = len(lines)
    ncols = 2 if n_lines > 1 else 1
    nrows = int(np.ceil(n_lines / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    # 颜色映射
    color_map = {
        '历史当小时客流': 'tab:orange',
        '预测': 'tab:blue',
        '相对准确率百分比': 'tab:red',
        '去年同期': 'tab:green',
        '去年同期+偏移': 'tab:purple',
        '去年同期+偏移准确率': 'tab:brown'
    }

    for idx, line in enumerate(lines):
        line_name = line_name_map.get(line, line)
        info = result[line]
        if info.get("error"):
            continue

        ax = axes[idx]
        ax.ticklabel_format(style='plain', axis='y')
        
        try:
            line_data = info.get("line_data", pd.DataFrame())
            predict_dt = datetime.strptime(predict_date, '%Y%m%d')
            hours = list(range(24))
            
            # 1. 历史当小时客流
            curr_day_data = line_data[line_data['F_DATE'] == predict_date]
            hist_hour_flow = {}
            curr_hist_flows = []
            if not curr_day_data.empty:
                hist_hour_flow = {int(h): float(f) for h, f in zip(curr_day_data['F_HOUR'], curr_day_data['F_KLCOUNT'])}
            curr_hist_flows = [hist_hour_flow.get(h, np.nan) for h in hours]
            
            # 2. 预测小时客流
            pred_flows = []
            if "predict_hourly_flow" in info:
                pred_flows = list(info["predict_hourly_flow"].values())
                if len(pred_flows) < 24:
                    pred_flows += [np.nan] * (24 - len(pred_flows))
                pred_flows = pred_flows[:24]
            else:
                pred_flows = [np.nan] * 24

            # 3. 去年同期小时客流
            last_year_date = (predict_dt - timedelta(days=365)).strftime('%Y%m%d')
            last_year_data = line_data[line_data['F_DATE'] == last_year_date]
            last_year_hour_flow = {}
            last_year_flows = []
            if not last_year_data.empty:
                last_year_hour_flow = {int(h): float(f) for h, f in zip(last_year_data['F_HOUR'], last_year_data['F_KLCOUNT'])}
            last_year_flows = [last_year_hour_flow.get(h, np.nan) for h in hours]

            # 4. 计算去年同期+整体偏移
            # 4.1 计算整体偏移量（基于前一天的客流差）
            offset_this_year_date = (predict_dt - timedelta(days=1)).strftime('%Y%m%d')
            offset_last_year_date = (predict_dt - timedelta(days=366)).strftime('%Y%m%d')
            
            this_year_offset_data = line_data[line_data['F_DATE'] == offset_this_year_date]
            last_year_offset_data = line_data[line_data['F_DATE'] == offset_last_year_date]
            
            this_year_offset_sum = this_year_offset_data['F_KLCOUNT'].sum() if not this_year_offset_data.empty else np.nan
            last_year_offset_sum = last_year_offset_data['F_KLCOUNT'].sum() if not last_year_offset_data.empty else np.nan
            
            overall_offset = this_year_offset_sum - last_year_offset_sum if not np.isnan(this_year_offset_sum) and not np.isnan(last_year_offset_sum) else np.nan
            
            # 4.2 将整体偏移平均分配到每小时
            overall_offset_per_hour = overall_offset / 24 if not np.isnan(overall_offset) else np.nan
            
            # 4.3 计算去年同期+整体偏移流量
            last_year_plus_offset_flows = []
            for h in hours:
                base_flow = last_year_hour_flow.get(h, np.nan)
                if not np.isnan(base_flow) and not np.isnan(overall_offset_per_hour):
                    last_year_plus_offset_flows.append(base_flow + overall_offset_per_hour)
                else:
                    last_year_plus_offset_flows.append(np.nan)

            # 5. 计算预测准确率
            pred_accs = []
            for h in hours:
                hist_val = hist_hour_flow.get(h, np.nan)
                pred_val = pred_flows[h] if h < len(pred_flows) else np.nan
                if not np.isnan(hist_val) and not np.isnan(pred_val) and hist_val != 0:
                    acc = (1 - abs(hist_val - pred_val) / abs(hist_val)) * 100
                    acc = max(0, min(acc, 100))
                else:
                    acc = np.nan
                pred_accs.append(acc)

            # 6. 计算去年同期+整体偏移准确率
            last_year_plus_offset_accs = []
            for h in hours:
                hist_val = hist_hour_flow.get(h, np.nan)
                offset_val = last_year_plus_offset_flows[h] if h < len(last_year_plus_offset_flows) else np.nan
                if not np.isnan(hist_val) and not np.isnan(offset_val) and hist_val != 0:
                    acc = (1 - abs(hist_val - offset_val) / abs(hist_val)) * 100
                    acc = max(0, min(acc, 100))
                else:
                    acc = np.nan
                last_year_plus_offset_accs.append(acc)

        except Exception as e:
            # 异常处理，使用默认空值
            hours = list(range(24))
            curr_hist_flows = [np.nan] * 24
            pred_flows = [np.nan] * 24
            pred_accs = [np.nan] * 24
            last_year_flows = [np.nan] * 24
            last_year_plus_offset_flows = [np.nan] * 24
            last_year_plus_offset_accs = [np.nan] * 24

        # x轴标签
        hour_labels = [f"{h:02d}" for h in hours]

        # 绘制主要曲线
        line_handles = []
        line_labels = []

        # 历史当小时客流
        if show_cur_hist and any([not np.isnan(v) for v in curr_hist_flows]):
            line1, = ax.plot(hours, curr_hist_flows, marker='o', label='历史当小时客流', 
                           color=color_map['历史当小时客流'], linewidth=2)
            line_handles.append(line1)
            line_labels.append(f'真实当小时客流（{color_map["历史当小时客流"]}）')

        # 去年同期
        if show_last_year and any([not np.isnan(v) for v in last_year_flows]):
            line2, = ax.plot(hours, last_year_flows, marker='^', label='去年同期', 
                           color=color_map['去年同期'], linewidth=2, linestyle='-.')
            line_handles.append(line2)
            line_labels.append(f'去年同期（{color_map["去年同期"]}）')

        # 去年同期+整体偏移
        if show_last_year_offset and any([not np.isnan(v) for v in last_year_plus_offset_flows]):
            line3, = ax.plot(hours, last_year_plus_offset_flows, marker='D', label='去年同期+偏移', 
                           color=color_map['去年同期+偏移'], linewidth=2, linestyle=':')
            line_handles.append(line3)
            line_labels.append(f'去年同期+偏移（{color_map["去年同期+偏移"]}）')

        # 预测
        if show_pred and any([not np.isnan(v) for v in pred_flows]):
            line4, = ax.plot(hours, pred_flows, marker='s', label='KNN预测', 
                           color=color_map['预测'], linewidth=2)
            line_handles.append(line4)
            line_labels.append(f'预测（{color_map["预测"]}）')

        # 准确率曲线（右侧y轴）
        has_pred_acc = show_pred_error and any([not np.isnan(e) for e in pred_accs])
        has_offset_acc = show_last_year_offset_error and any([not np.isnan(e) for e in last_year_plus_offset_accs])

        # 计算平均准确率（仅统计7点到23点）
        mean_acc_text = None
        if has_pred_acc:
            start_hour = 7
            end_hour = 23
            acc_range = range(start_hour, end_hour + 1)
            valid_accs = [pred_accs[h] for h in acc_range if not np.isnan(pred_accs[h])]
            if len(valid_accs) > 0:
                mean_acc = np.mean(valid_accs)
                mean_acc_text = f"平均准确率(7-23点): {mean_acc:.2f}%"

        lines_right = []
        labels_right = []
        if has_pred_acc or has_offset_acc:
            ax2 = ax.twinx()
            ax2.ticklabel_format(style='plain', axis='y')
            
            if has_pred_acc:
                line5, = ax2.plot(hours, pred_accs, marker='s', linestyle='--', 
                                color=color_map['相对准确率百分比'], label='准确率(%)')
                lines_right.append(line5)
                labels_right.append(f'准确率(%)（{color_map["相对准确率百分比"]}）')
                
                # 显示平均准确率
                if mean_acc_text is not None:
                    ax2.text(0.98, 0.98, mean_acc_text,
                           transform=ax2.transAxes, fontsize=12,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                           fontproperties=my_font if my_font is not None else None,
                           color=color_map['相对准确率百分比'])
            
            if has_offset_acc:
                line6, = ax2.plot(hours, last_year_plus_offset_accs, marker='x', linestyle='--', 
                                color=color_map['去年同期+偏移准确率'], label='去年同期+偏移准确率(%)')
                lines_right.append(line6)
                labels_right.append(f'去年同期+偏移准确率(%)（{color_map["去年同期+偏移准确率"]}）')
            
            # 设置右侧y轴标签
            if has_pred_acc and has_offset_acc:
                ax2.set_ylabel("准确率(%)", fontsize=12, 
                             fontproperties=my_font if my_font is not None else None,
                             color=color_map['相对准确率百分比'])
            elif has_pred_acc:
                ax2.set_ylabel("准确率(%)", fontsize=12,
                             fontproperties=my_font if my_font is not None else None,
                             color=color_map['相对准确率百分比'])
            elif has_offset_acc:
                ax2.set_ylabel("去年同期+偏移准确率(%)", fontsize=12,
                             fontproperties=my_font if my_font is not None else None,
                             color=color_map['去年同期+偏移准确率'])
            
            ax2.tick_params(axis='y', labelcolor=color_map['相对准确率百分比'])
            
            # 合并图例
            handles = line_handles + lines_right
            labels = line_labels + labels_right
            ax2.legend(handles, labels, loc='upper left', 
                      prop=my_font if my_font is not None else None)
        else:
            # 只显示主y轴图例
            ax.legend(line_handles, line_labels, 
                     prop=my_font if my_font is not None else None)

        # 设置图表标题和标签
        ax.set_title(f"{line_name} 小时预测", fontsize=14, 
                    fontproperties=my_font if my_font is not None else None)
        ax.set_xlabel("小时", fontsize=12, 
                     fontproperties=my_font if my_font is not None else None)
        ax.set_ylabel("客流量", fontsize=12, 
                     fontproperties=my_font if my_font is not None else None)
        ax.set_xticks(hours)
        ax.set_xticklabels(hour_labels, rotation=0, fontsize=10, 
                          fontproperties=my_font if my_font is not None else None)
        ax.grid(True, alpha=0.3)

    # 删除多余的子图
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    flow_type_str = FLOW_OPTIONS.get(flow_type, str(flow_type) if flow_type is not None else "未知类型")
    metric_type_str = FLOW_OPTIONS.get(metric_type, str(metric_type) if metric_type is not None else "未知指标")

    plt.suptitle(
        f"{flow_type_str} - {metric_type_str} 小时客流量预测 (预测日: {predict_date})",
        fontsize=16,
        fontproperties=my_font if my_font is not None else None
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_daily_predictions(
    result: Dict,
    line_name_map: Dict,
    predict_start_date: str,
    days: int,
    save_path: str = "timeseries_predict_daily.png",
    flow_type: str = None, 
    metric_type: str = None,
    show_cur_hist: bool = True,
    show_pred: bool = True,
    show_pred_error: bool = True,
    show_last_year: bool = False,
    show_last_year_offset: bool = False,
    show_last_year_offset_error: bool = False,
) -> None:
    """
    绘制日客流预测图，并增加上一年历史同日期的曲线和“去年同期+整体偏移”曲线
    并增加“去年同期+整体偏移”与“历史当天客流”的准确率曲线

    参数:
        result: 预测结果字典
        line_name_map: 线路编号到名称的映射
        predict_start_date: 预测起始日期 (YYYYMMDD)
        days: 预测天数
        save_path: 图表保存路径
        show_cur_hist: 是否显示历史当天客流曲线
        show_pred: 是否显示预测曲线
        show_pred_error: 是否显示预测准确率曲线
        show_last_year: 是否显示去年同期曲线
        show_last_year_offset: 是否显示去年同期+整体偏移曲线
        show_last_year_offset_error: 是否显示去年同期+整体偏移准确率曲线
    """
    configure_fonts()
    my_font = get_chinese_font()
    # 解决中文乱码问题
    if my_font is not None:
        font_name = my_font.get_name()
        plt.rcParams['font.sans-serif'] = ['SimHei']  
        plt.rcParams['axes.unicode_minus'] = False  
        matplotlib.rc('font', family=font_name)
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei']  
        plt.rcParams['axes.unicode_minus'] = False  

    # 不使用科学计数法
    plt.ticklabel_format(style='plain', axis='y')
    matplotlib.ticker.ScalarFormatter(useMathText=False)

    lines = sorted(result.keys())
    n_lines = len(lines)
    ncols = 2 if n_lines > 1 else 1
    nrows = int(np.ceil(n_lines / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    # 颜色和含义
    color_map = {
        '历史当天客流': 'tab:orange',
        '预测': 'tab:blue',
        '相对准确率百分比': 'tab:red',
        '去年同期': 'tab:green',
        '去年同期+偏移': 'tab:purple',
        '去年同期+偏移准确率': 'tab:brown'
    }

    for idx, line in enumerate(lines):
        line_name = line_name_map.get(line, line)
        info = result[line]
        if info.get("error"):
            continue

        ax = axes[idx]
        # 不使用科学计数法
        ax.ticklabel_format(style='plain', axis='y')
        try:
            line_data = info.get("line_data", pd.DataFrame())
            line_data_sorted = line_data.sort_values('F_DATE')
            daily_flow = info["predict_daily_flow"]
            pred_dates_str = list(daily_flow.keys())
            pred_dates = [datetime.strptime(d, '%Y%m%d') for d in pred_dates_str]
            # 历史当天客流
            curr_hist_dates = []
            curr_hist_flows = []
            for d in pred_dates_str:
                match = line_data_sorted[line_data_sorted['F_DATE'] == d]
                if not match.empty:
                    curr_hist_dates.append(datetime.strptime(d, '%Y%m%d'))
                    curr_hist_flows.append(match['F_KLCOUNT'].values[0])
            # 计算准确率（预测 vs 历史当天客流）
            abs_perc_accs = []
            for i, d in enumerate(pred_dates_str):
                try:
                    pred_val = daily_flow[d]
                    if d in [dt.strftime('%Y%m%d') for dt in curr_hist_dates]:
                        idx_hist = [dt.strftime('%Y%m%d') for dt in curr_hist_dates].index(d)
                        hist_val = curr_hist_flows[idx_hist]
                        if hist_val != 0:
                            acc = (1 - abs(hist_val - pred_val) / abs(hist_val)) * 100
                            acc = max(0, min(acc, 100))
                        else:
                            acc = np.nan
                    else:
                        acc = np.nan
                except Exception:
                    acc = np.nan
                abs_perc_accs.append(acc)
            # 新增：上一年同期历史客流
            last_year_dates = []
            last_year_flows = []
            for d in pred_dates_str:
                try:
                    dt_last_year = (datetime.strptime(d, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')
                    match_last = line_data_sorted[line_data_sorted['F_DATE'] == dt_last_year]
                    if not match_last.empty:
                        last_year_dates.append(datetime.strptime(d, '%Y%m%d'))  # x轴对齐当前预测日期
                        last_year_flows.append(match_last['F_KLCOUNT'].values[0])
                except Exception:
                    continue

            # 新增：去年同期+整体偏移曲线
            # 1. 预测区间
            pred_start_dt = datetime.strptime(predict_start_date, "%Y%m%d")
            pred_end_dt = pred_start_dt + timedelta(days=days-1)
            # 2. 去年同期区间
            last_year_start_dt = pred_start_dt - timedelta(days=365)
            last_year_end_dt = pred_end_dt - timedelta(days=365)
            # 3. 偏移区间（去年同期前的days天）
            offset_this_year_start = pred_start_dt - timedelta(days=days)
            offset_this_year_end = pred_start_dt - timedelta(days=1)
            offset_last_year_start = last_year_start_dt - timedelta(days=days)
            offset_last_year_end = last_year_start_dt - timedelta(days=1)
            # 4. 取去年同期区间的客流
            last_year_base_dates = [last_year_start_dt + timedelta(days=i) for i in range(days)]
            last_year_base_strs = [dt.strftime('%Y%m%d') for dt in last_year_base_dates]
            last_year_base_flows = []
            for d in last_year_base_strs:
                match = line_data_sorted[line_data_sorted['F_DATE'] == d]
                if not match.empty:
                    last_year_base_flows.append(match['F_KLCOUNT'].values[0])
                else:
                    last_year_base_flows.append(np.nan)
            # 5. 取偏移区间的客流
            offset_this_year_dates = [offset_this_year_start + timedelta(days=i) for i in range(days)]
            offset_this_year_strs = [dt.strftime('%Y%m%d') for dt in offset_this_year_dates]
            offset_this_year_flows = []
            for d in offset_this_year_strs:
                match = line_data_sorted[line_data_sorted['F_DATE'] == d]
                if not match.empty:
                    offset_this_year_flows.append(match['F_KLCOUNT'].values[0])
                else:
                    offset_this_year_flows.append(np.nan)
            offset_last_year_dates = [offset_last_year_start + timedelta(days=i) for i in range(days)]
            offset_last_year_strs = [dt.strftime('%Y%m%d') for dt in offset_last_year_dates]
            offset_last_year_flows = []
            for d in offset_last_year_strs:
                match = line_data_sorted[line_data_sorted['F_DATE'] == d]
                if not match.empty:
                    offset_last_year_flows.append(match['F_KLCOUNT'].values[0])
                else:
                    offset_last_year_flows.append(np.nan)
            # 6. 计算整体偏移量
            # 只要两段都没有nan才参与整体偏移计算
            valid_offset_this = [v for v in offset_this_year_flows if not np.isnan(v)]
            valid_offset_last = [v for v in offset_last_year_flows if not np.isnan(v)]
            if len(valid_offset_this) == days and len(valid_offset_last) == days:
                sum_offset_this = np.nansum(offset_this_year_flows)
                sum_offset_last = np.nansum(offset_last_year_flows)
                overall_offset = sum_offset_this - sum_offset_last
                # 平均分配到每一天
                overall_offset_per_day = overall_offset / days
            else:
                overall_offset_per_day = np.nan

            # 7. 计算“去年同期+整体偏移”曲线
            last_year_plus_offset_flows = []
            for i in range(days):
                base = last_year_base_flows[i] if i < len(last_year_base_flows) else np.nan
                if not np.isnan(base) and not np.isnan(overall_offset_per_day):
                    last_year_plus_offset_flows.append(base + overall_offset_per_day)
                else:
                    last_year_plus_offset_flows.append(np.nan)
            last_year_plus_offset_dates = [dt for dt in pred_dates]  # x轴对齐预测区间

            # 新增：去年同期+整体偏移 和 历史当天客流 的准确率
            last_year_plus_offset_accs = []
            for i, d in enumerate(pred_dates_str):
                try:
                    # 真实值
                    if d in [dt.strftime('%Y%m%d') for dt in curr_hist_dates]:
                        idx_hist = [dt.strftime('%Y%m%d') for dt in curr_hist_dates].index(d)
                        hist_val = curr_hist_flows[idx_hist]
                        # 去年同期+整体偏移
                        if i < len(last_year_plus_offset_flows):
                            offset_val = last_year_plus_offset_flows[i]
                            if not np.isnan(hist_val) and not np.isnan(offset_val) and hist_val != 0:
                                acc = (1 - abs(hist_val - offset_val) / abs(hist_val)) * 100
                                acc = max(0, min(acc, 100))
                            else:
                                acc = np.nan
                        else:
                            acc = np.nan
                    else:
                        acc = np.nan
                except Exception:
                    acc = np.nan
                last_year_plus_offset_accs.append(acc)

        except Exception:
            pred_dates = []
            pred_dates_str = []
            curr_hist_dates = []
            curr_hist_flows = []
            abs_perc_accs = []
            last_year_dates = []
            last_year_flows = []
            last_year_plus_offset_flows = []
            last_year_plus_offset_dates = []
            last_year_plus_offset_accs = []

        if 'daily_flow' not in locals():
            daily_flow = info["predict_daily_flow"]
            pred_dates_str = list(daily_flow.keys())
            pred_dates = [datetime.strptime(d, '%Y%m%d') for d in pred_dates_str]
        pred_flows = list(daily_flow.values())
        all_dates = pred_dates
        all_labels = [dt.strftime('%Y-%m-%d') for dt in all_dates]

        # 配置化：各曲线是否显示
        line_handles = []
        line_labels = []

        # 历史当天客流
        if show_cur_hist and curr_hist_dates:
            line1, = ax.plot([dt.strftime('%Y-%m-%d') for dt in curr_hist_dates], curr_hist_flows, marker='o', label='历史当天客流', color=color_map['历史当天客流'], linewidth=2)
        else:
            line1 = None
        if line1 is not None:
            line_handles.append(line1)
            line_labels.append(f'真实当天客流（{color_map["历史当天客流"]}）')

        # 去年同期
        if show_last_year and last_year_dates:
            line0, = ax.plot([dt.strftime('%Y-%m-%d') for dt in last_year_dates], last_year_flows, marker='^', label='去年同期', color=color_map['去年同期'], linewidth=2, linestyle='-.')
        else:
            line0 = None
        if line0 is not None:
            line_handles.append(line0)
            line_labels.append(f'去年同期（{color_map["去年同期"]}）')

        # 去年同期+整体偏移
        if show_last_year_offset and last_year_plus_offset_flows and any([not np.isnan(v) for v in last_year_plus_offset_flows]):
            line4, = ax.plot([dt.strftime('%Y-%m-%d') for dt in last_year_plus_offset_dates], last_year_plus_offset_flows, marker='D', label='去年同期+偏移', color=color_map['去年同期+偏移'], linewidth=2, linestyle=':')
        else:
            line4 = None
        if line4 is not None:
            line_handles.append(line4)
            line_labels.append(f'去年同期+偏移（{color_map["去年同期+偏移"]}）')

        # 预测
        if show_pred:
            line2, = ax.plot([dt.strftime('%Y-%m-%d') for dt in pred_dates], pred_flows, marker='o', label='KNN预测', color=color_map['预测'], linewidth=2)
        else:
            line2 = None
        if line2 is not None:
            line_handles.append(line2)
            line_labels.append(f'预测（{color_map["预测"]}）')

        # 准确率曲线
        has_pred_acc = show_pred_error and any([not np.isnan(e) for e in abs_perc_accs])
        has_offset_acc = show_last_year_offset_error and any([not np.isnan(e) for e in last_year_plus_offset_accs]) if 'last_year_plus_offset_accs' in locals() else False

        # 计算平均准确率
        mean_acc_text = None
        if has_pred_acc:
            valid_accs = [e for e in abs_perc_accs if not np.isnan(e)]
            if len(valid_accs) > 0:
                mean_acc = np.mean(valid_accs)
                mean_acc_text = f"平均准确率: {mean_acc:.2f}%"
            else:
                mean_acc_text = None

        lines_right = []
        labels_right = []
        if has_pred_acc or has_offset_acc:
            ax2 = ax.twinx()
            ax2.ticklabel_format(style='plain', axis='y')
            if has_pred_acc:
                line3, = ax2.plot([dt.strftime('%Y-%m-%d') for dt in pred_dates], abs_perc_accs, marker='s', linestyle='--', color=color_map['相对准确率百分比'], label='准确率(%)')
                lines_right.append(line3)
                labels_right.append(f'准确率(%)（{color_map["相对准确率百分比"]}）')
                # 在图上显示平均准确率
                if mean_acc_text is not None:
                    # 右上角显示
                    ax2.text(
                        0.98, 0.98, mean_acc_text,
                        transform=ax2.transAxes,
                        fontsize=12,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                        fontproperties=my_font if my_font is not None else None,
                        color=color_map['相对准确率百分比']
                    )
            else:
                line3 = None
            if has_offset_acc:
                line5, = ax2.plot([dt.strftime('%Y-%m-%d') for dt in pred_dates], last_year_plus_offset_accs, marker='x', linestyle='--', color=color_map['去年同期+偏移准确率'], label='去年同期+偏移准确率(%)')
                lines_right.append(line5)
                labels_right.append(f'去年同期+偏移准确率(%)（{color_map["去年同期+偏移准确率"]}）')
            else:
                line5 = None
            # 设置y轴标签
            if has_pred_acc and has_offset_acc:
                ax2.set_ylabel("准确率(%)", fontsize=12, fontproperties=my_font if my_font is not None else None, color=color_map['相对准确率百分比'])
            elif has_pred_acc:
                ax2.set_ylabel("准确率(%)", fontsize=12, fontproperties=my_font if my_font is not None else None, color=color_map['相对准确率百分比'])
            elif has_offset_acc:
                ax2.set_ylabel("去年同期+偏移准确率(%)", fontsize=12, fontproperties=my_font if my_font is not None else None, color=color_map['去年同期+偏移准确率'])
            ax2.tick_params(axis='y', labelcolor=color_map['相对准确率百分比'])
            # 合并图例，显示颜色含义
            handles = line_handles + lines_right
            labels = line_labels + labels_right
            ax2.legend(handles, labels, loc='upper left', prop=my_font if my_font is not None else None)
        else:
            # 只显示主y轴图例
            ax.legend(line_handles, line_labels, prop=my_font if my_font is not None else None)

        if pred_dates:
            ax.axvline(x=pred_dates[0].strftime('%Y-%m-%d'), color='red', linestyle='--', linewidth=1, label='预测起点')
        ax.set_title(f"{line_name} 日预测", fontsize=14, fontproperties=my_font if my_font is not None else None)
        ax.set_xlabel("日期", fontsize=12, fontproperties=my_font if my_font is not None else None)
        ax.set_ylabel("客流量", fontsize=12, fontproperties=my_font if my_font is not None else None)
        ax.set_xticks(all_labels)
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10, fontproperties=my_font if my_font is not None else None)
        ax.grid(True, alpha=0.3)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    flow_type_str = FLOW_OPTIONS.get(flow_type, str(flow_type) if flow_type is not None else "未知类型")
    metric_type_str = FLOW_OPTIONS.get(metric_type, str(metric_type) if metric_type is not None else "未知指标")

    plt.suptitle(
        f"{flow_type_str} - {metric_type_str} 日预测客流量 (预测起始日: {predict_start_date})",
        fontsize=16,
        fontproperties=my_font if my_font is not None else None
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
