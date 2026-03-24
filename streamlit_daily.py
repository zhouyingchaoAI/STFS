# streamlit_daily.py
# 日预测tab的实现（支持KNN/Prophet/Transformer等多算法）

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io
import math
from config_utils import load_yaml_config, save_yaml_config
from predict_daily import predict_and_plot_timeseries_flow_daily
import os
import matplotlib.pyplot as plt
import matplotlib
import warnings
from typing import List, Dict, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

# 导入UI组件
from ui_components import (
    render_section_header,
)

# 导入数据库连接池
from db_pool import get_db_connection

# 导入节假日预测工具
from holiday_predict_utils import predict_line_flow as expert_predict_line_flow
from predict_daily import get_holiday_fusion_alpha

# ==================== 配置常量 ====================

FLOW_METRIC_OPTIONS = [
    ("F_PKLCOUNT", "客运量"),
    ("F_ENTRANCE", "进站量"),
    ("F_EXIT", "出站量"),
    ("F_TRANSFER", "换乘量"),
    ("F_BOARD_ALIGHT", "乘降量")
]

FLOW_TYPES = {
    "xianwangxianlu": "线路线网",
    "duanmian": "断面",
    "chezhan": "车站"
}

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

# 算法名称映射
ALGO_DISPLAY_MAP = {
    "knn": "🔮 智能混合算法",
    "prophet": "🔍 智能搜索算法",
    "transformer": "🧠 深度学习算法",
    "xgboost": "📊 传统机器学习",
    "lstm": "🔄 长短时记忆网络",
    "lightgbm": "⚡ 轻量级梯度提升"
}

ALGO_REAL_MAP = {v: k for k, v in ALGO_DISPLAY_MAP.items()}
DEFAULT_HOLIDAY_FUSION_ALPHA = 0.9
HOLIDAY_FUSION_ALPHA_BY_TYPE = {
    1: 1.0,  # 元旦
    2: 1.0,  # 春节默认交给分段策略覆盖
    3: 1.0,  # 清明
    4: 0.1,  # 劳动节
    5: 0.7,  # 端午
    6: 0.9,  # 中秋，当前无专项评估，保留默认
    7: 0.5,  # 国庆
}
SPRING_FESTIVAL_SEGMENT_ALPHA = {
    "early": 1.0,  # 第1-3天
    "mid": 0.0,    # 第4-5天
    "late": 1.0,   # 第6天及以后
}


# ==================== 节假日检测函数 ====================

# 节假日类型映射（参考 script/get_clander.py）
HOLIDAY_TYPE_NAMES = {
    1: "元旦",
    2: "春节",
    3: "清明节",
    4: "劳动节",
    5: "端午节",
    6: "中秋节",
    7: "国庆节",
    8: "周末",
    9: "调休补班",
    11: "暑假",
    12: "寒假"
}


def get_holiday_dates(start_date: str, end_date: str) -> List[Dict]:
    """
    获取日期范围内的节假日列表
    
    判定规则（参考 script/get_clander.py）：
    - F_HOLIDAYWHICHDAY > 0 表示正处于节假日期间（第1天、第2天...）
    - F_HOLIDAYTYPE 1-7 为法定节假日（元旦、春节、清明、劳动、端午、中秋、国庆）
    - 排除 F_HOLIDAYTYPE = 8（周末）和 9（调休补班）
    
    参数:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        
    返回:
        节假日列表 [{'date': '20250501', 'type': '劳动节', 'type_id': 4, 'days': 5, 'which_day': 1}, ...]
    """
    # 主要通过 F_HOLIDAYWHICHDAY > 0 判断是否为节假日
    # F_HOLIDAYTYPE IN (1,2,3,4,5,6,7) 筛选法定节假日
    query = """
    SELECT F_DATE, F_HOLIDAYTYPE, F_HOLIDAYDAYS, F_HOLIDAYWHICHDAY
    FROM CalendarHistory 
    WHERE F_DATE >= %s AND F_DATE <= %s 
      AND F_HOLIDAYWHICHDAY > 0
      AND F_HOLIDAYTYPE IN (1, 2, 3, 4, 5, 6, 7)
    ORDER BY F_DATE
    """
    
    try:
        with get_db_connection() as conn:
            df = pd.read_sql(query, conn, params=(start_date, end_date))
        
        print(f"[DEBUG] 节假日检测: {start_date} - {end_date}, 查询到 {len(df)} 条记录")
        
        holidays = []
        for _, row in df.iterrows():
            holiday_type_id = int(row['F_HOLIDAYTYPE']) if row['F_HOLIDAYTYPE'] else 0
            holiday_type_name = HOLIDAY_TYPE_NAMES.get(holiday_type_id, f"节假日{holiday_type_id}")
            
            holidays.append({
                'date': str(row['F_DATE']),
                'type': holiday_type_name,
                'type_id': holiday_type_id,
                'days': int(row['F_HOLIDAYDAYS']) if row['F_HOLIDAYDAYS'] else 0,
                'which_day': int(row['F_HOLIDAYWHICHDAY']) if row['F_HOLIDAYWHICHDAY'] else 0
            })
        
        if holidays:
            print(f"[DEBUG] 检测到节假日: {[h['type'] + ' ' + h['date'] for h in holidays[:5]]}")
        
        return holidays
    except Exception as e:
        print(f"[ERROR] 节假日检测失败: {e}")
        # 发生错误时返回空列表，不影响主流程
        return []


def has_holiday_in_range(start_date: str, end_date: str) -> bool:
    """
    检查日期范围是否包含节假日
    
    参数:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        
    返回:
        是否包含节假日
    """
    holidays = get_holiday_dates(start_date, end_date)
    return len(holidays) > 0


def get_holiday_summary(holidays: List[Dict]) -> str:
    """
    获取节假日摘要信息
    
    参数:
        holidays: 节假日列表
        
    返回:
        摘要字符串，如 "劳动节 (5/1-5/5, 共5天)"
    """
    if not holidays:
        return ""
    
    # 按节假日类型分组
    holiday_types = {}
    for h in holidays:
        h_type = h['type']
        if h_type not in holiday_types:
            holiday_types[h_type] = {
                'dates': [],
                'total_days': h.get('days', 0)  # 节假日总天数
            }
        holiday_types[h_type]['dates'].append(h['date'])
    
    summaries = []
    for h_type, info in holiday_types.items():
        dates = sorted(info['dates'])
        total_days = info['total_days']
        
        if len(dates) == 1:
            date_str = dates[0]
            formatted = f"{int(date_str[4:6])}/{int(date_str[6:8])}"
            if total_days > 0:
                summaries.append(f"{h_type} ({formatted}, 共{total_days}天)")
            else:
                summaries.append(f"{h_type} ({formatted})")
        else:
            start = dates[0]
            end = dates[-1]
            formatted = f"{int(start[4:6])}/{int(start[6:8])}-{int(end[4:6])}/{int(end[6:8])}"
            day_count = len(dates)
            if total_days > 0 and total_days != day_count:
                summaries.append(f"{h_type} ({formatted}, 预测{day_count}天/共{total_days}天)")
            else:
                summaries.append(f"{h_type} ({formatted}, 共{day_count}天)")
    
    return "、".join(summaries)


def get_holiday_fusion_alpha(holiday_info: Optional[Dict]) -> float:
    """按节日类型返回融合权重，春节按假期第几天单独处理。"""
    if not holiday_info:
        return DEFAULT_HOLIDAY_FUSION_ALPHA

    holiday_type_id = int(holiday_info.get("type_id", 0) or 0)
    which_day = int(holiday_info.get("which_day", 0) or 0)

    if holiday_type_id == 2:
        if which_day <= 3:
            return SPRING_FESTIVAL_SEGMENT_ALPHA["early"]
        if which_day <= 5:
            return SPRING_FESTIVAL_SEGMENT_ALPHA["mid"]
        return SPRING_FESTIVAL_SEGMENT_ALPHA["late"]

    return HOLIDAY_FUSION_ALPHA_BY_TYPE.get(holiday_type_id, DEFAULT_HOLIDAY_FUSION_ALPHA)


def should_use_expert_fusion(holiday_info: Optional[Dict]) -> bool:
    """仅法定节假日切换专家系统，周末/调休仍沿用 KNN。"""
    if not holiday_info:
        return False
    holiday_type_id = int(holiday_info.get("type_id", 0) or 0)
    holiday_days = int(holiday_info.get("total_days", 0) or 0)
    which_day = int(holiday_info.get("which_day", 0) or 0)
    return holiday_type_id in {1, 2, 3, 4, 5, 6, 7} and holiday_days > 0 and 1 <= which_day <= holiday_days


def build_expert_switch_dates(holidays: List[Dict], pre_days: int = 1, post_days: int = 0) -> set[str]:
    """法定节假日当天及节前指定天数切换专家系统。"""
    holiday_dates = sorted({str(item.get("date", "")).strip() for item in holidays if item.get("date")})
    if not holiday_dates:
        return set()

    switch_dates = set(holiday_dates)
    blocks = []
    current_block = []
    for date_str in holiday_dates:
        current_dt = datetime.strptime(date_str, "%Y%m%d")
        if current_block:
            prev_dt = datetime.strptime(current_block[-1], "%Y%m%d")
            if current_dt == prev_dt + timedelta(days=1):
                current_block.append(date_str)
            else:
                blocks.append(current_block)
                current_block = [date_str]
        else:
            current_block = [date_str]
    if current_block:
        blocks.append(current_block)

    for block in blocks:
        start_dt = datetime.strptime(block[0], "%Y%m%d")
        end_dt = datetime.strptime(block[-1], "%Y%m%d")
        for offset in range(1, pre_days + 1):
            switch_dates.add((start_dt - timedelta(days=offset)).strftime("%Y%m%d"))
        for offset in range(1, post_days + 1):
            switch_dates.add((end_dt + timedelta(days=offset)).strftime("%Y%m%d"))
    return switch_dates


def get_model_versions(model_dir, prefix=""):
    """获取模型目录下所有模型版本（以日期为子目录）"""
    if not os.path.exists(model_dir):
        return []
    dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    versions = [d for d in dirs if len(d) == 8 and d.isdigit()]
    versions.sort(reverse=True)
    return versions


def setup_plot_style():
    """设置图表样式 - 增强中文支持"""
    plt.style.use('dark_background')
    
    # 尝试多种中文字体，按优先级顺序
    chinese_fonts = [
        'WenQuanYi Micro Hei',  # Linux 常用
        'WenQuanYi Zen Hei',    # Linux 常用
        'Noto Sans CJK SC',      # Google 开源字体
        'Noto Sans SC',          # Google 开源字体
        'Microsoft YaHei',       # Windows
        'SimHei',                # Windows
        'PingFang SC',           # macOS
        'Heiti SC',              # macOS
        'DejaVu Sans',           # 回退字体
        'sans-serif'
    ]
    
    # 检测可用字体
    import matplotlib.font_manager as fm
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    
    # 选择第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        matplotlib.rcParams['font.family'] = [selected_font]
        matplotlib.rcParams['font.sans-serif'] = [selected_font] + chinese_fonts
    else:
        # 如果没有找到中文字体，使用默认设置但保持列表
        matplotlib.rcParams['font.sans-serif'] = chinese_fonts
    
    matplotlib.rcParams['axes.unicode_minus'] = False


def plot_daily_flow(
    df_plot, 
    line_name=None, 
    SUBWAY_GREEN="#00ffd5", 
    SUBWAY_ACCENT="#bf00ff", 
    SUBWAY_CARD="#1e2439", 
    SUBWAY_BG="#0f1423", 
    SUBWAY_FONT="#ffffff", 
    return_fig=False
):
    """绘制日客流预测图 - 赛博朋克风格"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # 设置背景
    fig.patch.set_facecolor(SUBWAY_BG)
    ax.set_facecolor(SUBWAY_CARD)
    
    # 绘制数据线 - 带发光效果
    color = SUBWAY_GREEN if line_name is None else SUBWAY_ACCENT
    
    # 发光效果 - 多层叠加
    for alpha, lw in [(0.1, 8), (0.2, 5), (0.4, 3)]:
        ax.plot(df_plot["日期"], df_plot["预测客流"], 
                color=color, linewidth=lw, alpha=alpha)
    
    # 主线
    line = ax.plot(df_plot["日期"], df_plot["预测客流"], 
                   marker='o', color=color, linewidth=2.5, 
                   markersize=8, markerfacecolor=SUBWAY_BG,
                   markeredgecolor=color, markeredgewidth=2,
                   label=line_name, zorder=10)
    
    # 填充区域
    ax.fill_between(df_plot["日期"], df_plot["预测客流"], 
                    alpha=0.15, color=color)
    
    # 添加数据点标注
    for i, (x, y) in enumerate(zip(df_plot["日期"], df_plot["预测客流"])):
        if i % max(1, len(df_plot) // 8) == 0:  # 每隔几个点显示一个标签
            ax.annotate(f'{y:,.0f}', (x, y), 
                       textcoords="offset points", xytext=(0, 12),
                       ha='center', fontsize=9, color=color,
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=SUBWAY_CARD, 
                                edgecolor=color, alpha=0.8))
    
    # 图例
    if line_name is not None:
        legend = ax.legend(
            facecolor=SUBWAY_CARD, 
            edgecolor=SUBWAY_GREEN,
            fontsize=12,
            loc='upper right',
            framealpha=0.9
        )
        plt.setp(legend.get_texts(), color=SUBWAY_FONT)
    
    # 轴标签
    ax.set_xlabel("日期", color=SUBWAY_GREEN, fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel("预测客流量", color=SUBWAY_GREEN, fontsize=13, fontweight='bold', labelpad=10)
    
    # X轴刻度
    ax.set_xticks(df_plot["日期"])
    ax.set_xticklabels(df_plot["日期"], rotation=45, ha='right', color=SUBWAY_FONT, fontsize=10)
    
    # 标题
    title = "📈 日预测客流量" + (f" · {line_name}" if line_name else "")
    ax.set_title(title, color=SUBWAY_ACCENT, fontsize=18, fontweight='bold', pad=20)
    
    # 网格线
    ax.grid(True, linestyle='--', alpha=0.2, color=SUBWAY_ACCENT)
    ax.tick_params(axis='y', colors=SUBWAY_FONT, labelsize=10)
    
    # 边框样式
    for spine in ax.spines.values():
        spine.set_color(SUBWAY_GREEN)
        spine.set_alpha(0.3)
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        st.pyplot(fig)
        plt.close(fig)


def figure_to_png_bytes(fig) -> bytes:
    """将 matplotlib 图对象导出为 PNG 二进制。"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def build_holiday_comparison_image(
    knn_results: Dict,
    expert_results: Dict,
    holidays: List[Dict],
    flow_type: str,
    SUBWAY_GREEN: str = "#00ffd5",
    SUBWAY_ACCENT: str = "#bf00ff",
    SUBWAY_CARD: str = "#1e2439",
    SUBWAY_BG: str = "#0f1423",
    SUBWAY_FONT: str = "#ffffff",
) -> Optional[bytes]:
    """生成节假日三指标对比图片：机器学习、人工算法、融合、实际。"""
    expert_predictions = expert_results.get("predictions", []) if isinstance(expert_results, dict) else []
    if not expert_predictions:
        return None

    holiday_dates = [h["date"] for h in holidays]
    holiday_set = set(holiday_dates)
    if flow_type == "chezhan":
        expert_switch_dates = build_expert_switch_dates(holidays, pre_days=2, post_days=0)
    else:
        expert_switch_dates = build_expert_switch_dates(holidays, pre_days=3, post_days=0)
    if not holiday_set:
        return None

    series_by_line: Dict[str, List[Dict]] = {}
    for pred in expert_predictions:
        line_name = pred.get("线路名称", pred.get("F_LINENAME", pred.get("line_name", "未知")))
        date_str = pred.get("预测日期", pred.get("date", ""))
        if date_str not in holiday_set:
            continue

        matched_knn_key = None
        for knn_key in knn_results.keys():
            if knn_key == line_name or knn_key in line_name or line_name in knn_key:
                matched_knn_key = knn_key
                break

        knn_pred = 0
        if matched_knn_key and isinstance(knn_results.get(matched_knn_key), dict):
            knn_pred = int(knn_results[matched_knn_key].get(date_str, 0) or 0)

        expert_pred = int(pred.get("预测客流", pred.get("predicted_flow", 0)) or 0)
        holiday_type = int(pred.get("节假日类型", 0) or 0)
        holiday_day_index = int(pred.get("第几天", pred.get("节假日第几天", 0)) or 0)
        fusion_pred = expert_pred if date_str in expert_switch_dates else knn_pred

        actual_flow = pred.get("实际客流")
        actual_flow = int(actual_flow) if actual_flow is not None else None

        series_by_line.setdefault(line_name, []).append({
            "日期": date_str,
            "机器学习": knn_pred,
            "人工算法": expert_pred,
            "融合预测": fusion_pred,
            "实际客流": actual_flow,
        })

    entities = sorted(series_by_line.keys())
    if not entities:
        return None

    setup_plot_style()
    ncols = 2 if len(entities) > 1 else 1
    nrows = math.ceil(len(entities) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9 * ncols, 5 * nrows), squeeze=False)
    fig.patch.set_facecolor(SUBWAY_BG)
    axes = axes.flatten()

    color_map = {
        "机器学习": SUBWAY_GREEN,
        "人工算法": "#ffaa00",
        "融合预测": "#60a5fa",
        "实际客流": "#ff5c8a",
    }

    for idx, entity_name in enumerate(entities):
        ax = axes[idx]
        ax.set_facecolor(SUBWAY_CARD)
        entity_df = pd.DataFrame(series_by_line[entity_name]).sort_values("日期")
        x = list(range(len(entity_df)))

        for column in ["机器学习", "人工算法", "融合预测"]:
            ax.plot(
                x,
                entity_df[column].tolist(),
                marker="o",
                linewidth=2.2,
                markersize=6,
                label=column,
                color=color_map[column],
            )

        actual_mask = entity_df["实际客流"].notna()
        if actual_mask.any():
            actual_x = [i for i, ok in enumerate(actual_mask.tolist()) if ok]
            actual_y = entity_df.loc[actual_mask, "实际客流"].astype(float).tolist()
            ax.plot(
                actual_x,
                actual_y,
                marker="s",
                linestyle="--",
                linewidth=2,
                markersize=6,
                label="实际客流",
                color=color_map["实际客流"],
            )

        ax.set_title(entity_name, color=SUBWAY_ACCENT, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(entity_df["日期"].tolist(), rotation=45, ha="right", color=SUBWAY_FONT, fontsize=9)
        ax.tick_params(axis="y", colors=SUBWAY_FONT, labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.18, color=SUBWAY_GREEN)
        legend = ax.legend(facecolor=SUBWAY_CARD, edgecolor=SUBWAY_GREEN, fontsize=9)
        plt.setp(legend.get_texts(), color=SUBWAY_FONT)
        for spine in ax.spines.values():
            spine.set_color(SUBWAY_GREEN)
            spine.set_alpha(0.25)

    for idx in range(len(entities), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("节假日预测对比图：机器学习 vs 人工算法 vs 融合 vs 实际", color=SUBWAY_FONT, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    image_bytes = figure_to_png_bytes(fig)
    plt.close(fig)
    return image_bytes


# ==================== 节假日对比展示组件 ====================

def render_holiday_comparison(
    knn_results: Dict,
    expert_results: Dict,
    holidays: List[Dict],
    flow_type: str,
    SUBWAY_GREEN: str = "#00ffd5",
    SUBWAY_ACCENT: str = "#bf00ff",
    SUBWAY_CARD: str = "#1e2439",
    SUBWAY_BG: str = "#0f1423",
    SUBWAY_FONT: str = "#ffffff"
):
    """
    节假日预测对比展示
    
    参数:
        knn_results: KNN预测结果 {line_name: {date: flow, ...}, ...}
        expert_results: 人工算法预测结果 (从 predict_line_flow 返回)
        holidays: 节假日列表
        SUBWAY_*: 颜色配置
    """
    holiday_summary = get_holiday_summary(holidays)
    
    # 对比面板标题
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(255, 170, 0, 0.12), rgba(255, 100, 0, 0.08));
        border: 1px solid rgba(255, 170, 0, 0.4);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
    ">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 1.5rem;">🎯</span>
            <div>
                <h3 style="
                    margin: 0;
                    font-family: 'Rajdhani', sans-serif;
                    font-size: 1.2rem;
                    font-weight: 700;
                    color: #ffaa00;
                ">节假日预测对比（自动检测）</h3>
                <p style="margin: 0.25rem 0 0 0; color: #ffd080; font-size: 0.85rem;">
                    检测到节假日: {holiday_summary}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 提示信息
    st.markdown("""
    <div style="
        background: rgba(255, 170, 0, 0.08);
        border-left: 3px solid #ffaa00;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        border-radius: 0 8px 8px 0;
    ">
        <p style="margin: 0; color: #d0d7e8; font-size: 0.9rem;">
            💡 <strong style="color: #ffaa00;">建议</strong>：融合结果序列当前采用
            <strong>平常日使用 KNN，节前和节中使用专家系统</strong> 的切换策略。
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 检查人工算法结果
    if not expert_results or expert_results.get("error"):
        st.warning(f"⚠️ 人工算法预测失败: {expert_results.get('error', '未知错误')}")
        return
    
    # 获取人工算法预测数据
    expert_predictions = expert_results.get("predictions", [])
    if not expert_predictions:
        st.warning("⚠️ 人工算法未返回预测结果")
        return
    
    # 调试：打印第一条预测记录的字段名
    if expert_predictions:
        first_pred = expert_predictions[0]
        print(f"[DEBUG] 人工算法预测记录字段: {list(first_pred.keys())}")
        print(f"[DEBUG] 人工算法预测示例: 线路名称={first_pred.get('线路名称', 'N/A')}, 预测日期={first_pred.get('预测日期', 'N/A')}")
    
    # 调试：打印KNN结果的格式
    if knn_results:
        print(f"[DEBUG] KNN结果keys: {list(knn_results.keys())}")
        first_knn_key = list(knn_results.keys())[0]
        if isinstance(knn_results[first_knn_key], dict):
            first_dates = list(knn_results[first_knn_key].keys())[:3]
            print(f"[DEBUG] KNN日期格式示例: {first_dates}")
    
    # 按线路分组人工算法结果
    # 注意：holiday_predict_utils返回的字段是中文名：线路名称、预测日期、预测客流等
    expert_by_line = {}
    for pred in expert_predictions:
        line_name = pred.get("线路名称", pred.get("F_LINENAME", pred.get("line_name", "未知")))
        if line_name not in expert_by_line:
            expert_by_line[line_name] = {}
        date_str = pred.get("预测日期", pred.get("date", ""))
        flow = pred.get("预测客流", pred.get("predicted_flow", 0))
        expert_by_line[line_name][date_str] = flow

    if flow_type == "chezhan":
        expert_switch_dates = build_expert_switch_dates(holidays, pre_days=2, post_days=0)
    else:
        expert_switch_dates = build_expert_switch_dates(holidays, pre_days=3, post_days=0)
    fusion_by_line = {}
    all_line_names = set(knn_results.keys()) | set(expert_by_line.keys())
    for line_name in all_line_names:
        fusion_by_line[line_name] = {}
        line_knn = knn_results.get(line_name, {})
        line_expert = expert_by_line.get(line_name, {})
        all_dates = set(line_knn.keys()) | set(line_expert.keys())

        for date_str in all_dates:
            knn_flow = line_knn.get(date_str)
            expert_flow = line_expert.get(date_str)
            if knn_flow is not None and expert_flow is not None:
                fusion_flow = expert_flow if date_str in expert_switch_dates else knn_flow
            else:
                fusion_flow = knn_flow if knn_flow is not None else expert_flow

            fusion_by_line[line_name][date_str] = {
                "flow": int(fusion_flow) if fusion_flow else 0,
                "alpha": 0.0,
            }

    comparison_image = build_holiday_comparison_image(
        knn_results=knn_results,
        expert_results=expert_results,
        holidays=holidays,
        flow_type=flow_type,
        SUBWAY_GREEN=SUBWAY_GREEN,
        SUBWAY_ACCENT=SUBWAY_ACCENT,
        SUBWAY_CARD=SUBWAY_CARD,
        SUBWAY_BG=SUBWAY_BG,
        SUBWAY_FONT=SUBWAY_FONT,
    )
    if comparison_image:
        st.markdown("### 节假日预测对比图片")
        st.image(comparison_image, caption="机器学习、人工算法、融合预测与实际客流对比图", use_container_width=True)

    # 创建对比展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: rgba(0, 255, 213, 0.08);
            border: 1px solid rgba(0, 255, 213, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        ">
            <h4 style="margin: 0; color: #00ffd5; font-size: 1rem;">
                🔮 KNN机器学习预测
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # 转换KNN结果为DataFrame
        knn_data = []
        for line_name, line_data in knn_results.items():
            if isinstance(line_data, dict):
                for date_str, flow in line_data.items():
                    knn_data.append({
                        "线路": line_name,
                        "日期": date_str,
                        "预测客流": int(flow) if flow else 0
                    })
        
        if knn_data:
            df_knn = pd.DataFrame(knn_data)
            # 只显示节假日期间的数据
            holiday_dates = [h['date'] for h in holidays]
            df_knn_holiday = df_knn[df_knn['日期'].isin(holiday_dates)]
            if not df_knn_holiday.empty:
                st.dataframe(df_knn_holiday, use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_knn.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("暂无KNN预测数据")
    
    with col2:
        st.markdown("""
        <div style="
            background: rgba(255, 170, 0, 0.08);
            border: 1px solid rgba(255, 170, 0, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        ">
            <h4 style="margin: 0; color: #ffaa00; font-size: 1rem;">
                🎯 人工算法预测（历年增长率）
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # 转换人工算法结果为DataFrame（使用中文字段名：线路名称、预测日期等）
        expert_data = []
        for pred in expert_predictions:
            line_name = pred.get("线路名称", pred.get("F_LINENAME", pred.get("line_name", "未知")))
            date_str = pred.get("预测日期", pred.get("date", ""))
            flow = pred.get("预测客流", pred.get("predicted_flow", 0))
            growth_rate = pred.get("最优增长率", pred.get("growth_rate", 0))
            source_year = pred.get("最优来源年份", "")
            expert_data.append({
                "线路": line_name,
                "日期": date_str,
                "预测客流": int(flow) if flow else 0,
                "增长率%": f"{growth_rate:.1f}" if growth_rate else "0.0",
                "参考年份": str(source_year) if source_year else "-"
            })
        
        if expert_data:
            df_expert = pd.DataFrame(expert_data)
            # 只显示节假日期间的数据
            holiday_dates = [h['date'] for h in holidays]
            df_expert_holiday = df_expert[df_expert['日期'].isin(holiday_dates)]
            if not df_expert_holiday.empty:
                st.dataframe(df_expert_holiday, use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_expert.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("暂无人工算法预测数据")

    with col3:
        st.markdown("""
        <div style="
            background: rgba(96, 165, 250, 0.08);
            border: 1px solid rgba(96, 165, 250, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        ">
            <h4 style="margin: 0; color: #60a5fa; font-size: 1rem;">
                🧩 动态融合预测（按节日类型自动分配权重）
            </h4>
        </div>
        """, unsafe_allow_html=True)

        fusion_data = []
        for line_name, line_data in fusion_by_line.items():
            for date_str, fusion_info in line_data.items():
                fusion_data.append({
                    "线路": line_name,
                    "日期": date_str,
                    "预测客流": int(fusion_info.get("flow", 0)),
                    "ML权重": f"{fusion_info.get('alpha', DEFAULT_HOLIDAY_FUSION_ALPHA) * 100:.0f}%",
                })

        if fusion_data:
            df_fusion = pd.DataFrame(fusion_data)
            holiday_dates = [h['date'] for h in holidays]
            df_fusion_holiday = df_fusion[df_fusion['日期'].isin(holiday_dates)]
            if not df_fusion_holiday.empty:
                st.dataframe(df_fusion_holiday, use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_fusion.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("暂无融合预测数据")
    
    # 显示准确率对比（如果有历史数据）
    # 检查predictions中是否有准确率数据
    has_actual = expert_results.get("has_actual", False)
    
    if has_actual:
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        with st.expander("📊 准确率对比：人工算法 vs KNN机器学习 vs 融合预测", expanded=True):
            # 收集有实际数据的预测记录（仅统计KNN结果中存在的线路，即应用线路过滤）
            accuracy_data = []
            
            for pred in expert_predictions:
                actual_flow = pred.get("实际客流")
                if actual_flow is not None:
                    line_name = pred.get("线路名称", pred.get("F_LINENAME", pred.get("line_name", "未知")))
                    date_str = pred.get("预测日期", pred.get("date", ""))
                    expert_pred = pred.get("预测客流", 0)
                    expert_accuracy = pred.get("准确率", 0)
                    
                    # 获取对应的KNN预测值（需要匹配线路名称）
                    knn_pred = 0
                    knn_accuracy = 0
                    # KNN结果的key可能是"线网"、"1号线"等，需要匹配
                    matched_knn_key = None
                    for knn_key in knn_results.keys():
                        if knn_key == line_name or knn_key in line_name or line_name in knn_key:
                            matched_knn_key = knn_key
                            break
                    
                    # 只统计KNN结果中存在的线路（应用线路过滤配置）
                    if not matched_knn_key:
                        continue  # 跳过被过滤的线路
                    
                    if isinstance(knn_results[matched_knn_key], dict):
                        knn_pred = knn_results[matched_knn_key].get(date_str, 0)
                        if actual_flow and actual_flow > 0 and knn_pred:
                            knn_accuracy = (1 - abs(knn_pred - actual_flow) / actual_flow) * 100
                            knn_accuracy = max(0, min(100, knn_accuracy))

                    fusion_info = fusion_by_line.get(matched_knn_key, {}).get(date_str)
                    if not fusion_info:
                        fusion_info = fusion_by_line.get(line_name, {}).get(date_str, {})
                    fusion_pred = int(fusion_info.get("flow", 0)) if fusion_info else 0
                    fusion_alpha = fusion_info.get("alpha", DEFAULT_HOLIDAY_FUSION_ALPHA) if fusion_info else DEFAULT_HOLIDAY_FUSION_ALPHA

                    fusion_accuracy = 0
                    if actual_flow and actual_flow > 0 and fusion_pred:
                        fusion_accuracy = (1 - abs(fusion_pred - actual_flow) / actual_flow) * 100
                        fusion_accuracy = max(0, min(100, fusion_accuracy))

                    accuracy_data.append({
                        "线路": line_name,
                        "日期": date_str,
                        "实际客流": int(actual_flow),
                        "KNN预测": int(knn_pred) if knn_pred else 0,
                        "KNN准确率%": f"{knn_accuracy:.1f}",
                        "人工预测": int(expert_pred) if expert_pred else 0,
                        "人工准确率%": f"{expert_accuracy:.1f}" if expert_accuracy else "0.0",
                        "融合预测": int(fusion_pred) if fusion_pred else 0,
                        "融合准确率%": f"{fusion_accuracy:.1f}",
                        "融合ML权重": f"{fusion_alpha * 100:.0f}%"
                    })
            
            if accuracy_data:
                df_accuracy = pd.DataFrame(accuracy_data)
                # 只显示节假日期间的数据
                holiday_dates = [h['date'] for h in holidays]
                df_accuracy_holiday = df_accuracy[df_accuracy['日期'].isin(holiday_dates)]
                if not df_accuracy_holiday.empty:
                    st.dataframe(df_accuracy_holiday, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(df_accuracy, use_container_width=True, hide_index=True)
                
                # 计算平均准确率
                avg_knn_accuracy = sum(float(a["KNN准确率%"]) for a in accuracy_data) / len(accuracy_data)
                avg_expert_accuracy = sum(float(a["人工准确率%"]) for a in accuracy_data) / len(accuracy_data)
                avg_fusion_accuracy = sum(float(a["融合准确率%"]) for a in accuracy_data) / len(accuracy_data)

                avg_metrics = {
                    "KNN机器学习": avg_knn_accuracy,
                    "人工算法": avg_expert_accuracy,
                    "融合预测": avg_fusion_accuracy,
                }
                winner = max(avg_metrics, key=avg_metrics.get)
                winner_color = {
                    "KNN机器学习": "#00ffd5",
                    "人工算法": "#ffaa00",
                    "融合预测": "#60a5fa",
                }[winner]
                
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: space-around;
                    padding: 1rem;
                    background: rgba(30, 36, 57, 0.8);
                    border-radius: 12px;
                    margin-top: 1rem;
                ">
                    <div style="text-align: center;">
                        <div style="color: #a0a8c0; font-size: 0.85rem;">KNN机器学习</div>
                        <div style="color: #00ffd5; font-size: 1.5rem; font-weight: 700;">{avg_knn_accuracy:.1f}%</div>
                    </div>
                    <div style="text-align: center; border-left: 1px solid rgba(255,255,255,0.1); border-right: 1px solid rgba(255,255,255,0.1); padding: 0 1.5rem;">
                        <div style="color: #a0a8c0; font-size: 0.85rem;">融合预测</div>
                        <div style="color: #60a5fa; font-size: 1.5rem; font-weight: 700;">{avg_fusion_accuracy:.1f}%</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #a0a8c0; font-size: 0.85rem;">人工算法</div>
                        <div style="color: #ffaa00; font-size: 1.5rem; font-weight: 700;">{avg_expert_accuracy:.1f}%</div>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 0.75rem;">
                    <span style="
                        background: {winner_color}20;
                        color: {winner_color};
                        padding: 0.4rem 1rem;
                        border-radius: 20px;
                        font-weight: 600;
                        font-size: 0.9rem;
                    ">
                        🏆 {winner} 准确率更高
                    </span>
                </div>
                """, unsafe_allow_html=True)


def render_training_panel(
    SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG,
    flow_type, metric_type, config_daily
):
    """渲染训练面板"""
    daily_algos = list(ALGO_DISPLAY_MAP.keys())
    daily_algo_display_names = list(ALGO_DISPLAY_MAP.values())
    
    # 面板标题
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(0, 255, 213, 0.08), rgba(191, 0, 255, 0.05));
        border: 1px solid rgba(0, 255, 213, 0.3);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 1.5rem;">🎓</span>
            <div>
                <h3 style="
                    margin: 0;
                    font-family: 'Rajdhani', sans-serif;
                    font-size: 1.2rem;
                    font-weight: 700;
                    color: #00ffd5;
                ">模型训练</h3>
                <p style="margin: 0.25rem 0 0 0; color: #b8c4d8; font-size: 0.85rem;">配置参数并训练预测模型</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 算法选择
    train_daily_algo_display = st.selectbox(
        "🧪 选择训练算法",
        options=daily_algo_display_names,
        key="daily_train_algo",
        help="选择用于训练的机器学习/深度学习算法"
    )
    train_daily_algo = ALGO_REAL_MAP.get(train_daily_algo_display, train_daily_algo_display)
    
    # 高级参数设置
    with st.expander("⚙️ 高级训练参数", expanded=False):
        train_params = config_daily.get("train_params", {})
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            lookback_days = st.number_input(
                "回溯天数", min_value=1, max_value=30,
                value=train_params.get("lookback_days", 7) or 7,
                key="daily_lookback_days",
                help="用于预测的历史数据天数"
            )
            epochs = st.number_input(
                "训练轮次", min_value=10, max_value=500,
                value=train_params.get("epochs", 100) or 100,
                key="daily_epochs"
            )
            patience = st.number_input(
                "早停耐心值", min_value=1, max_value=50,
                value=train_params.get("patience", 10) or 10,
                key="daily_patience"
            )
        
        with col_p2:
            lr_val = train_params.get("learning_rate", 0.001) or 0.001
            learning_rate = st.number_input(
                "学习率", min_value=0.0001, max_value=0.1,
                value=float(lr_val), step=0.0001, format="%.4f",
                key="daily_lr"
            )
            batch_size = st.number_input(
                "批次大小", min_value=8, max_value=256,
                value=train_params.get("batch_size", 32) or 32,
                key="daily_batch_size"
            )
        
        # 算法特定参数
        n_neighbors = None
        d_model = None
        nhead = None
        num_layers = None
        
        if train_daily_algo == "knn":
            st.markdown("##### KNN 特定参数")
            n_neighbors = st.number_input(
                "K邻居数", min_value=1, max_value=30,
                value=train_params.get("n_neighbors", 5) or 5,
                key="daily_n_neighbors"
            )
        
        if train_daily_algo == "transformer":
            st.markdown("##### Transformer 特定参数")
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                d_model = st.number_input(
                    "隐藏维度", min_value=8, max_value=512,
                    value=train_params.get("d_model", 64) or 64,
                    key="daily_d_model"
                )
            with col_t2:
                nhead = st.number_input(
                    "注意力头数", min_value=1, max_value=16,
                    value=train_params.get("nhead", 4) or 4,
                    key="daily_nhead"
                )
            with col_t3:
                num_layers = st.number_input(
                    "编码器层数", min_value=1, max_value=8,
                    value=train_params.get("num_layers", 2) or 2,
                    key="daily_num_layers"
                )
    
    # 更新配置
    config_daily["train_params"] = {
        "lookback_days": lookback_days,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    if train_daily_algo == "knn" and n_neighbors:
        config_daily["train_params"]["n_neighbors"] = n_neighbors
    if train_daily_algo == "transformer":
        config_daily["train_params"]["d_model"] = d_model
        config_daily["train_params"]["nhead"] = nhead
        config_daily["train_params"]["num_layers"] = num_layers
    save_yaml_config(config_daily, "model_config_daily.yaml")
    
    # 训练设置
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    train_start_date = st.date_input(
        "📅 训练数据截止日期",
        value=datetime(2025, 4, 26),
        key="daily_train_date",
        help="模型将使用此日期之前的数据进行训练"
    )
    train_start_date_str = train_start_date.strftime("%Y%m%d")
    
    retrain_daily = st.checkbox(
        "🔄 强制重新训练",
        value=True,
        key="daily_retrain",
        help="勾选后将重新训练模型，否则加载已有模型"
    )
    
    # 训练按钮
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    if st.button("🚀 开始训练", key="run_daily_train"):
        with st.spinner(""):
            # 显示训练进度
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            status_placeholder.markdown("""
            <div style="
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 1rem;
                background: rgba(0, 255, 213, 0.05);
                border: 1px solid rgba(0, 255, 213, 0.2);
                border-radius: 10px;
            ">
                <div style="
                    width: 24px;
                    height: 24px;
                    border: 3px solid rgba(0, 255, 213, 0.2);
                    border-top-color: #00ffd5;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                "></div>
                <span style="color: #00ffd5; font-weight: 600;">正在训练模型...</span>
            </div>
            <style>
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            </style>
            """, unsafe_allow_html=True)
            
            model_save_dir_daily = os.path.join(
                "models", flow_type, "daily", metric_type, 
                train_start_date_str, train_daily_algo
            )
            os.makedirs(model_save_dir_daily, exist_ok=True)
            
            result = predict_and_plot_timeseries_flow_daily(
                file_path="",
                predict_start_date=train_start_date_str,
                algorithm=train_daily_algo,
                retrain=retrain_daily,
                save_path="timeseries_predict_daily.png",
                mode="train",
                days=15,
                config=config_daily,
                model_version=None,
                model_save_dir=model_save_dir_daily,
                flow_type=flow_type,
                metric_type=metric_type
            )
            
            status_placeholder.empty()
        
        if isinstance(result, dict) and "error" in result:
            st.error(f"❌ 训练失败: {result['error']}")
        else:
            st.success("✅ 模型训练完成！")
            st.info("💡 请在右侧推理模块选择模型版本进行预测")
    
    return train_daily_algo, config_daily


def render_prediction_panel(
    SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG,
    flow_type, metric_type, config_daily
):
    """渲染预测面板"""
    daily_algos = list(ALGO_DISPLAY_MAP.keys())
    
    # 面板标题
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(191, 0, 255, 0.08), rgba(255, 0, 128, 0.05));
        border: 1px solid rgba(191, 0, 255, 0.3);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 1.5rem;">🔮</span>
            <div>
                <h3 style="
                    margin: 0;
                    font-family: 'Rajdhani', sans-serif;
                    font-size: 1.2rem;
                    font-weight: 700;
                    color: #bf00ff;
                ">推理预测</h3>
                <p style="margin: 0.25rem 0 0 0; color: #b8c4d8; font-size: 0.85rem;">使用训练好的模型进行预测</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 模型版本选择
    daily_model_root = os.path.join("models", flow_type, "daily", metric_type)
    daily_versions = get_model_versions(daily_model_root)
    
    if daily_versions:
        daily_version = st.selectbox(
            "📁 模型版本",
            options=daily_versions,
            key="daily_model_version",
            help="选择要使用的模型版本（按日期排序）"
        )
        
        algo_dir_daily = os.path.join(daily_model_root, daily_version)
        available_daily_algos = [
            d for d in os.listdir(algo_dir_daily) 
            if os.path.isdir(os.path.join(algo_dir_daily, d)) and d in daily_algos
        ]
        
        if available_daily_algos:
            available_daily_algos_display = [
                ALGO_DISPLAY_MAP.get(a, a) for a in available_daily_algos
            ]
            predict_daily_algo_display = st.selectbox(
                "🧪 推理算法",
                options=available_daily_algos_display,
                key="daily_predict_model_type"
            )
            predict_daily_algo = ALGO_REAL_MAP.get(
                predict_daily_algo_display, predict_daily_algo_display
            )
            model_dir_daily = os.path.join(daily_model_root, daily_version, predict_daily_algo)
        else:
            model_dir_daily = None
            predict_daily_algo = None
            st.warning("⚠️ 该版本下未找到可用的算法模型")
    else:
        daily_version = None
        model_dir_daily = None
        predict_daily_algo = None
        st.markdown("""
        <div style="
            background: rgba(255, 170, 0, 0.1);
            border: 1px solid rgba(255, 170, 0, 0.3);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        ">
            <span style="font-size: 2rem;">📭</span>
            <p style="color: #ffaa00; margin: 0.5rem 0 0 0; font-weight: 600;">暂无可用模型</p>
            <p style="color: #b8c4d8; font-size: 0.85rem; margin: 0.25rem 0 0 0;">请先在左侧训练模型</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    # 预测参数
    if daily_version is None:
        default_date = datetime.now()
    else:
        default_date = datetime.strptime(daily_version, "%Y%m%d") + pd.Timedelta(days=1)
    
    col_date, col_days, col_history = st.columns(3)
    with col_date:
        predict_start_date = st.date_input(
            "📅 预测起始日期",
            value=default_date,
            key="daily_predict_date"
        )
    with col_days:
        days = st.number_input(
            "📊 预测天数",
            min_value=1, max_value=30, value=15,
            key="daily_days"
        )
    with col_history:
        history_years = st.number_input(
            "📆 参考历史年数",
            min_value=1, max_value=5, value=2,
            key="daily_history_years",
            help="节假日对比时，人工算法参考前几年的数据"
        )
    
    predict_start_date_str = predict_start_date.strftime("%Y%m%d")
    
    # 预测按钮
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    if st.button("🎯 开始预测", key="run_daily_predict"):
        if not daily_version or not model_dir_daily or not os.path.exists(model_dir_daily) or not predict_daily_algo:
            st.error("❌ 请先训练并选择有效的模型版本")
        else:
            with st.spinner(""):
                status_placeholder = st.empty()
                status_placeholder.markdown("""
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    padding: 1rem;
                    background: rgba(191, 0, 255, 0.05);
                    border: 1px solid rgba(191, 0, 255, 0.2);
                    border-radius: 10px;
                ">
                    <div style="
                        width: 24px;
                        height: 24px;
                        border: 3px solid rgba(191, 0, 255, 0.2);
                        border-top-color: #bf00ff;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                    "></div>
                    <span style="color: #bf00ff; font-weight: 600;">正在进行预测...</span>
                </div>
                """, unsafe_allow_html=True)
                
                output_image_path = "timeseries_predict_daily.png"
                result = predict_and_plot_timeseries_flow_daily(
                    file_path="",
                    predict_start_date=predict_start_date_str,
                    algorithm=predict_daily_algo,
                    retrain=False,
                    save_path=output_image_path,
                    mode="predict",
                    days=days,
                    config=config_daily,
                    model_version=None,
                    model_save_dir=model_dir_daily,
                    flow_type=flow_type,
                    metric_type=metric_type
                )
                
                status_placeholder.empty()
            
            if isinstance(result, dict) and "error" in result:
                st.error(f"❌ 预测失败: {result['error']}")
            else:
                st.success("✅ 预测完成！")
                st.session_state["daily_prediction_image_path"] = output_image_path if os.path.exists(output_image_path) else None
                
                # 显示结果
                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
                render_section_header("预测结果", "可视化展示", "📊")
                
                info = result if isinstance(result, dict) else {}
                if info.get("error"):
                    st.warning(f"⚠️ {info['error']}")
                else:
                    # info的格式是 {line_no: {predict_daily_flow: {date: flow}, ...}, ...}
                    # 需要重新构建daily_flow格式为 {line_name: {date: flow}}
                    # 简单的线路编号到名称映射
                    LINE_NO_NAME_MAP = {
                        "00": "线网", "01": "1号线", "02": "2号线", "03": "3号线",
                        "04": "4号线", "05": "5号线", "06": "6号线", "31": "西环线",
                        "60": "磁浮快线", "83": "互联网平台"
                    }
                    daily_flow = {}
                    for line_no, line_result in info.items():
                        if isinstance(line_result, dict) and "predict_daily_flow" in line_result:
                            line_name = LINE_NO_NAME_MAP.get(line_no, f"线路{line_no}")
                            daily_flow[line_name] = line_result.get("predict_daily_flow", {})
                    
                    plot_results = []
                    
                    print(f"[DEBUG] daily_flow重构完成, 键数量: {len(daily_flow)}, 键名: {list(daily_flow.keys())[:5]}")
                    
                    if isinstance(daily_flow, dict) and daily_flow and all(isinstance(v, dict) for v in daily_flow.values()):
                        for line_name, line_daily in daily_flow.items():
                            dates = sorted(line_daily.keys())
                            flows = [line_daily.get(date, 0) or 0 for date in dates]
                            df_plot = pd.DataFrame({
                                "日期": dates,
                                "预测客流": flows
                            })
                            plot_results.append({
                                "line_name": line_name,
                                "df_plot": df_plot,
                                "predict_start_date": info.get('predict_start_date', predict_start_date_str)
                            })
                    else:
                        dates = sorted(daily_flow.keys())
                        flows = [daily_flow.get(date, 0) or 0 for date in dates]
                        df_plot = pd.DataFrame({
                            "日期": dates,
                            "预测客流": flows
                        })
                        plot_results.append({
                            "line_name": None,
                            "df_plot": df_plot,
                            "predict_start_date": info.get('predict_start_date', predict_start_date_str)
                        })
                    
                    st.session_state["daily_plot_results"] = plot_results
                    st.session_state["daily_plot_figs"] = None
                    
                    # ========== 节假日检测和人工算法对比 ==========
                    # 计算预测日期范围
                    predict_end_date = (datetime.strptime(predict_start_date_str, "%Y%m%d") + 
                                       timedelta(days=days - 1)).strftime("%Y%m%d")
                    
                    print(f"[DEBUG] 预测完成，开始节假日检测: {predict_start_date_str} - {predict_end_date}")
                    
                    # 检测是否包含节假日
                    holidays = get_holiday_dates(predict_start_date_str, predict_end_date)
                    
                    print(f"[DEBUG] 节假日检测结果: {len(holidays)} 个节假日")
                    
                    if holidays:
                        # 将plot_results转换为knn_results格式 {line_name: {date: flow}}
                        knn_results_dict = {}
                        for pr in plot_results:
                            line_name = pr.get("line_name") or "全线路"
                            df = pr.get("df_plot")
                            if df is not None and not df.empty:
                                knn_results_dict[line_name] = {}
                                for _, row in df.iterrows():
                                    date_str = str(row["日期"])
                                    flow = row["预测客流"]
                                    knn_results_dict[line_name][date_str] = flow
                        
                        print(f"[DEBUG] knn_results_dict构建完成: {list(knn_results_dict.keys())}")
                        
                        # 保存节假日信息和KNN结果到session state
                        st.session_state["daily_holidays"] = holidays
                        st.session_state["daily_knn_results"] = knn_results_dict
                        st.session_state["daily_predict_dates"] = {
                            "start": predict_start_date_str,
                            "end": predict_end_date
                        }
                        st.session_state["daily_metric_type"] = metric_type
                        st.session_state["daily_history_years_value"] = history_years  # 保存参考历史年数（使用不同的key避免与widget冲突）
                        # 清除之前的人工算法结果，以便重新计算
                        st.session_state.pop("daily_expert_results", None)
                        
                        # 显示节假日检测提示
                        holiday_names = list(set([h['type'] for h in holidays]))
                        st.info(f"🎯 检测到预测日期包含 **{', '.join(holiday_names)}** 节假日（共{len(holidays)}天），请滚动页面查看人工算法与机器学习的对比结果！")
                    else:
                        # 清除之前的节假日相关数据
                        st.session_state.pop("daily_holidays", None)
                        st.session_state.pop("daily_knn_results", None)
                        st.session_state.pop("daily_predict_dates", None)
                        st.session_state.pop("daily_expert_results", None)
    
    return daily_version, model_dir_daily


def daily_tab(SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG, flow_type, metric_type):
    """日预测tab主函数"""
    config_daily = load_yaml_config("model_config_daily.yaml", default_daily=True)
    
    # 页面标题
    render_section_header(
        f"{FLOW_OPTIONS[flow_type]} · {FLOW_OPTIONS[metric_type]} 日预测",
        "支持 KNN / Prophet / Transformer / XGBoost / LSTM / LightGBM 算法",
        "📅"
    )
    
    # 初始化session state
    st.session_state.setdefault("daily_plot_results", None)
    st.session_state.setdefault("daily_prediction_image_path", None)
    
    # 双栏布局
    col_train, col_pred = st.columns([1, 1.1], gap="large")
    
    with col_train:
        render_training_panel(
            SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG,
            flow_type, metric_type, config_daily
        )
    
    with col_pred:
        render_prediction_panel(
            SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG,
            flow_type, metric_type, config_daily
        )
    
    print(f"[DEBUG] daily_tab: 检查 daily_holidays = {st.session_state.get('daily_holidays') is not None}")

    image_path = st.session_state.get("daily_prediction_image_path")
    if image_path and os.path.exists(image_path):
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 预测结果图片")
        st.image(image_path, caption="预测曲线图片", use_container_width=True)

    # ========== 节假日对比展示（移到外部，不依赖于daily_plot_figs） ==========
    holidays = st.session_state.get("daily_holidays")
    print(f"[DEBUG] daily_tab: 节假日数据 = {len(holidays) if holidays else 'None'}")
    if holidays:
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        
        knn_results = st.session_state.get("daily_knn_results", {})
        predict_dates = st.session_state.get("daily_predict_dates", {})
        current_metric_type = st.session_state.get("daily_metric_type", metric_type)
        current_history_years = st.session_state.get("daily_history_years_value", 2)  # 获取用户配置的参考年数
        
        print(f"[DEBUG] daily_tab: knn_results keys = {list(knn_results.keys()) if knn_results else 'None'}")
        print(f"[DEBUG] daily_tab: predict_dates = {predict_dates}, history_years = {current_history_years}")
        
        # 调用人工算法获取预测结果
        expert_results = st.session_state.get("daily_expert_results")
        
        if expert_results is None and predict_dates:
            try:
                print(f"[DEBUG] daily_tab: 开始调用人工算法, 参考{current_history_years}年数据")
                with st.spinner(f"正在调用人工算法进行节假日预测对比（参考{current_history_years}年数据）..."):
                    expert_results = expert_predict_line_flow(
                        metric_type=current_metric_type,
                        predict_start=predict_dates["start"],
                        predict_end=predict_dates["end"],
                        history_years=current_history_years,  # 使用用户配置的参考年数
                        custom_configs=None
                    )
                    st.session_state["daily_expert_results"] = expert_results
                    print(f"[DEBUG] daily_tab: 人工算法调用完成, error={expert_results.get('error')}")
            except Exception as e:
                print(f"[DEBUG] daily_tab: 人工算法调用失败: {e}")
                expert_results = {"error": str(e)}
                st.session_state["daily_expert_results"] = expert_results
        
        if expert_results:
            print(f"[DEBUG] daily_tab: 开始渲染对比界面")
            render_holiday_comparison(
                knn_results=knn_results,
                expert_results=expert_results,
                holidays=holidays,
                flow_type=flow_type,
                SUBWAY_GREEN=SUBWAY_GREEN,
                SUBWAY_ACCENT=SUBWAY_ACCENT,
                SUBWAY_CARD=SUBWAY_CARD,
                SUBWAY_BG=SUBWAY_BG,
                SUBWAY_FONT=SUBWAY_FONT
            )
        else:
            st.warning("⚠️ 人工算法结果为空，无法显示对比界面")
    
