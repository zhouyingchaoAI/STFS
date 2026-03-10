# streamlit_holiday.py
# 节假日预测tab的实现（基于历年最优增长率的人工算法）

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

# 导入UI组件
from ui_components import (
    CYBER_THEME,
    render_section_header,
    render_glass_card,
)

# 导入预测工具
from holiday_predict_utils import (
    METRIC_NAMES,
    predict_line_flow,
    predict_station_flow,
    check_data_availability,
    format_predictions_for_display,
    calculate_summary_stats,
)


# ==================== 配置常量 ====================

HISTORY_YEAR_OPTIONS = [
    (1, "参考前1年"),
    (2, "参考前2年"),
    (3, "参考前3年"),
    (5, "参考前5年"),
]

PREDICTION_TYPE_OPTIONS = [
    ("line", "🚇 线路预测"),
    ("station", "🏢 车站预测"),
]


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


def plot_predictions_chart(
    predictions: list,
    metric_name: str,
    is_station: bool = False,
    SUBWAY_GREEN="#00ffd5",
    SUBWAY_ACCENT="#bf00ff",
    SUBWAY_CARD="#1e2439",
    SUBWAY_BG="#0f1423",
    SUBWAY_FONT="#ffffff",
):
    """绘制预测结果图表"""
    setup_plot_style()
    
    if not predictions:
        return None
    
    df = pd.DataFrame(predictions)
    id_field = '车站ID' if is_station else '线路编号'
    name_field = '车站名称' if is_station else '线路名称'
    
    # 获取唯一实体
    entities = df[id_field].unique()
    n_entities = len(entities)
    
    if n_entities == 0:
        return None
    
    # 根据实体数量决定布局
    if n_entities <= 3:
        n_cols = n_entities
        n_rows = 1
    elif n_entities <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 3
        n_rows = min((n_entities + 2) // 3, 4)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.patch.set_facecolor(SUBWAY_BG)
    
    # 确保axes是二维数组
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 颜色循环
    colors = [SUBWAY_GREEN, SUBWAY_ACCENT, "#00ff88", "#ff6b35", "#00b4d8", "#ffd700"]
    
    for idx, entity_id in enumerate(entities[:n_rows * n_cols]):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        ax.set_facecolor(SUBWAY_CARD)
        
        entity_data = df[df[id_field] == entity_id].sort_values('预测日期')
        entity_name = entity_data[name_field].iloc[0]
        color = colors[idx % len(colors)]
        
        x = range(len(entity_data))
        
        # 绘制预测值
        pred_values = entity_data['预测客流'].values
        ax.plot(x, pred_values, 'o-', color=color, linewidth=2.5,
                markersize=8, markerfacecolor=SUBWAY_BG,
                markeredgecolor=color, markeredgewidth=2,
                label='预测', zorder=10)
        
        # 填充区域
        ax.fill_between(x, pred_values, alpha=0.15, color=color)
        
        # 如果有实际值，绘制对比
        if '实际客流' in entity_data.columns:
            actual_values = entity_data['实际客流'].values
            # 过滤掉None值
            mask = [v is not None for v in actual_values]
            if any(mask):
                actual_x = [i for i, m in enumerate(mask) if m]
                actual_y = [v for v, m in zip(actual_values, mask) if m]
                ax.plot(actual_x, actual_y, 's--', color="#ff0080", linewidth=2,
                        markersize=7, label='实际', alpha=0.8, zorder=9)
        
        # 设置标题和标签
        ax.set_title(f"📊 {entity_name}", color=color, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("预测天数", color=SUBWAY_GREEN, fontsize=10)
        ax.set_ylabel(metric_name, color=SUBWAY_GREEN, fontsize=10)
        
        # X轴标签
        dates = entity_data['预测日期'].values
        ax.set_xticks(x)
        ax.set_xticklabels([f"第{i+1}天" for i in x], rotation=45, ha='right', 
                          color=SUBWAY_FONT, fontsize=9)
        
        # 网格
        ax.grid(True, linestyle='--', alpha=0.2, color=SUBWAY_GREEN)
        ax.tick_params(colors=SUBWAY_FONT)
        
        # 图例
        ax.legend(loc='upper right', facecolor=SUBWAY_CARD, 
                 edgecolor=color, fontsize=9, framealpha=0.9)
        
        # 边框
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_alpha(0.3)
    
    # 隐藏多余的子图
    for idx in range(n_entities, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_accuracy_chart(
    predictions: list,
    is_station: bool = False,
    SUBWAY_GREEN="#00ffd5",
    SUBWAY_ACCENT="#bf00ff",
    SUBWAY_CARD="#1e2439",
    SUBWAY_BG="#0f1423",
    SUBWAY_FONT="#ffffff",
):
    """绘制准确率分布图"""
    setup_plot_style()
    
    if not predictions:
        return None
    
    df = pd.DataFrame(predictions)
    
    if '准确率' not in df.columns:
        return None
    
    # 过滤有效准确率
    valid_df = df[df['准确率'].notna()].copy()
    if valid_df.empty:
        return None
    
    id_field = '车站ID' if is_station else '线路编号'
    name_field = '车站名称' if is_station else '线路名称'
    
    # 按实体汇总
    entity_acc = valid_df.groupby([id_field, name_field])['准确率'].mean().reset_index()
    entity_acc = entity_acc.sort_values('准确率', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(entity_acc) * 0.5)))
    fig.patch.set_facecolor(SUBWAY_BG)
    ax.set_facecolor(SUBWAY_CARD)
    
    # 设置颜色
    colors = []
    for acc in entity_acc['准确率']:
        if acc >= 90:
            colors.append(SUBWAY_GREEN)
        elif acc >= 80:
            colors.append("#ffaa00")
        else:
            colors.append("#ff4757")
    
    # 绘制水平条形图
    bars = ax.barh(range(len(entity_acc)), entity_acc['准确率'], color=colors, alpha=0.8, height=0.6)
    
    # 添加数值标签
    for i, (idx, row) in enumerate(entity_acc.iterrows()):
        ax.text(row['准确率'] + 1, i, f"{row['准确率']:.1f}%", 
                va='center', color=SUBWAY_FONT, fontsize=10, fontweight='bold')
    
    # 设置标签
    ax.set_yticks(range(len(entity_acc)))
    ax.set_yticklabels(entity_acc[name_field], color=SUBWAY_FONT, fontsize=10)
    ax.set_xlabel("准确率 (%)", color=SUBWAY_GREEN, fontsize=12, fontweight='bold')
    ax.set_title("📊 预测准确率对比", color=SUBWAY_ACCENT, fontsize=16, fontweight='bold', pad=15)
    
    # 添加参考线
    ax.axvline(x=90, color=SUBWAY_GREEN, linestyle='--', alpha=0.5, label='优秀 (≥90%)')
    ax.axvline(x=80, color="#ffaa00", linestyle='--', alpha=0.5, label='良好 (≥80%)')
    
    ax.set_xlim(0, 105)
    ax.legend(loc='lower right', facecolor=SUBWAY_CARD, 
             edgecolor=SUBWAY_GREEN, fontsize=9, framealpha=0.9)
    
    # 网格
    ax.grid(True, axis='x', linestyle='--', alpha=0.2, color=SUBWAY_GREEN)
    ax.tick_params(colors=SUBWAY_FONT)
    
    for spine in ax.spines.values():
        spine.set_color(SUBWAY_GREEN)
        spine.set_alpha(0.3)
    
    plt.tight_layout()
    return fig


def render_prediction_card(pred: dict, is_station: bool = False, idx: int = 0):
    """渲染单条预测结果卡片"""
    id_field = '车站ID' if is_station else '线路编号'
    name_field = '车站名称' if is_station else '线路名称'
    
    # 准确率颜色
    acc = pred.get('准确率')
    if acc is not None:
        if acc >= 90:
            acc_color = "#00ff88"
            acc_icon = "✅"
        elif acc >= 80:
            acc_color = "#ffaa00"
            acc_icon = "⚠️"
        else:
            acc_color = "#ff4757"
            acc_icon = "❌"
        acc_text = f"{acc:.1f}%"
    else:
        acc_color = "#6c7893"
        acc_icon = "❓"
        acc_text = "待验证"
    
    # 增长率颜色
    growth = pred.get('最优增长率', 0)
    growth_color = "#00ff88" if growth >= 0 else "#ff4757"
    growth_icon = "📈" if growth >= 0 else "📉"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(15, 20, 35, 0.95), rgba(30, 36, 57, 0.8));
        border: 1px solid rgba(0, 255, 213, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="color: #00ffd5; font-weight: 700; font-size: 1rem;">
                {pred.get(name_field, '未知')}
            </span>
            <span style="
                background: rgba(191, 0, 255, 0.2);
                border: 1px solid rgba(191, 0, 255, 0.4);
                color: #bf00ff;
                padding: 0.2rem 0.6rem;
                border-radius: 12px;
                font-size: 0.75rem;
            ">第{pred.get('第几天', idx+1)}天</span>
        </div>
        <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
            <div>
                <span style="color: #9aa5bd; font-size: 0.8rem;">预测客流</span>
                <div style="color: #ffffff; font-size: 1.1rem; font-weight: 700;">
                    {pred.get('预测客流', 0):,}
                </div>
            </div>
            <div>
                <span style="color: #9aa5bd; font-size: 0.8rem;">基期日均</span>
                <div style="color: #d0d7e8; font-size: 1rem;">
                    {pred.get('基期日均', 0):,}
                </div>
            </div>
            <div>
                <span style="color: #9aa5bd; font-size: 0.8rem;">{growth_icon} 增长率</span>
                <div style="color: {growth_color}; font-size: 1rem; font-weight: 600;">
                    {growth:+.2f}%
                </div>
            </div>
            <div>
                <span style="color: #9aa5bd; font-size: 0.8rem;">来源年份</span>
                <div style="color: #d0d7e8; font-size: 1rem;">
                    {pred.get('最优来源年份', '-')}年
                </div>
            </div>
            <div>
                <span style="color: #9aa5bd; font-size: 0.8rem;">{acc_icon} 准确率</span>
                <div style="color: {acc_color}; font-size: 1rem; font-weight: 600;">
                    {acc_text}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def holiday_tab(
    SUBWAY_GREEN="#00ffd5",
    SUBWAY_ACCENT="#bf00ff",
    SUBWAY_CARD="#1e2439",
    SUBWAY_FONT="#ffffff",
    SUBWAY_BG="#0f1423",
    flow_type="xianwangxianlu",
    metric_type="F_PKLCOUNT"
):
    """
    节假日预测标签页主函数
    基于历年最优增长率的人工算法预测
    """
    
    # 标题和说明
    render_section_header(
        title="节假日客流预测",
        subtitle="基于历年同期最优增长率的智能预测算法，适用于节假日等特殊时期的客流预测",
        icon="🎯"
    )
    
    # 算法说明
    with st.expander("📖 算法说明", expanded=False):
        st.markdown("""
        <div style="color: #d0d7e8; line-height: 1.8;">
        
        **核心原理**：通过分析历年同期（如往年五一假期）的客流增长规律，选取每一天的最优增长率进行预测。
        
        **预测公式**：
        ```
        预测客流 = 基期日均 × (1 + 最优增长率 / 100)
        ```
        
        **计算步骤**：
        1. **计算基期日均**：取预测月份的上一个完整月作为基期，计算各线路/车站的日均客流
        2. **查询历史同期**：获取往年同期（如去年五一、前年五一）的客流数据
        3. **计算历史增长率**：对比历史同期与其各自基期，计算每日增长率
        4. **提取最优增长率**：对于每一天，选取历年中增长率最高的值作为预测参考
        5. **生成预测值**：将最优增长率应用于当前基期日均，得出预测客流
        
        **适用场景**：
        - 🎉 节假日客流预测（五一、国庆、春节等）
        - 📅 特殊活动期间预测
        - 🔄 验证机器学习模型效果
        
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # 预测配置区域
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(15, 20, 35, 0.95), rgba(30, 36, 57, 0.8));
        border: 1px solid rgba(0, 255, 213, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
            <span style="font-size: 1.3rem;">⚙️</span>
            <span style="font-family: 'Rajdhani', sans-serif; font-size: 1.1rem; font-weight: 700; color: #00ffd5;">
                预测配置
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 配置行
    col1, col2, col3, col4 = st.columns([1, 1.2, 1.2, 1])
    
    with col1:
        # 预测类型选择
        pred_type_labels = [label for _, label in PREDICTION_TYPE_OPTIONS]
        selected_pred_type_label = st.selectbox(
            "🎯 预测类型",
            options=pred_type_labels,
            index=0,
            key="holiday_pred_type",
            help="选择线路预测或车站预测"
        )
        selected_pred_type = [k for k, v in PREDICTION_TYPE_OPTIONS if v == selected_pred_type_label][0]
        is_station = selected_pred_type == 'station'
    
    with col2:
        # 预测开始日期
        default_start = datetime.now() + timedelta(days=1)
        predict_start = st.date_input(
            "📅 预测开始日期",
            value=default_start,
            key="holiday_predict_start",
            help="选择预测期的开始日期"
        )
    
    with col3:
        # 预测结束日期
        default_end = default_start + timedelta(days=4)
        predict_end = st.date_input(
            "📅 预测结束日期",
            value=default_end,
            key="holiday_predict_end",
            help="选择预测期的结束日期"
        )
    
    with col4:
        # 历史年限选择
        history_labels = [label for _, label in HISTORY_YEAR_OPTIONS]
        selected_history_label = st.selectbox(
            "📊 参考历史",
            options=history_labels,
            index=1,  # 默认参考前2年
            key="holiday_history_years",
            help="选择参考多少年的历史数据"
        )
        history_years = [y for y, label in HISTORY_YEAR_OPTIONS if label == selected_history_label][0]
    
    # 日期验证
    if predict_start > predict_end:
        st.error("⚠️ 开始日期不能晚于结束日期")
        return
    
    predict_days = (predict_end - predict_start).days + 1
    
    # 计算基期信息
    base_month = predict_start.month - 1
    base_year = predict_start.year
    if base_month < 1:
        base_month = 12
        base_year -= 1
    last_day = calendar.monthrange(base_year, base_month)[1]
    base_period_str = f"{base_year}年{base_month}月1日 - {base_year}年{base_month}月{last_day}日"
    
    # 显示配置摘要
    st.markdown(f"""
    <div style="
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1rem 0;
    ">
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(0, 255, 213, 0.1);
            border: 1px solid rgba(0, 255, 213, 0.4);
            padding: 0.5rem 1rem;
            border-radius: 8px;
        ">
            <span style="color: #00ffd5;">📅</span>
            <span style="color: #ffffff; font-weight: 600;">预测期: {predict_start.strftime('%Y-%m-%d')} 至 {predict_end.strftime('%Y-%m-%d')} ({predict_days}天)</span>
        </div>
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(191, 0, 255, 0.1);
            border: 1px solid rgba(191, 0, 255, 0.4);
            padding: 0.5rem 1rem;
            border-radius: 8px;
        ">
            <span style="color: #bf00ff;">📊</span>
            <span style="color: #ffffff; font-weight: 600;">基期: {base_period_str}</span>
        </div>
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.4);
            padding: 0.5rem 1rem;
            border-radius: 8px;
        ">
            <span style="color: #00ff88;">🔍</span>
            <span style="color: #ffffff; font-weight: 600;">参考: 前{history_years}年同期数据</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 高级设置（可折叠）
    with st.expander("🔧 高级设置 - 自定义各年参考期和基期", expanded=False):
        st.markdown("""
        <div style="color: #9aa5bd; font-size: 0.9rem; margin-bottom: 1rem;">
            默认情况下，系统会自动计算各年的参考期和基期。如需针对特殊情况（如疫情年份）进行调整，可在此自定义。
        </div>
        """, unsafe_allow_html=True)
        
        custom_configs = {}
        use_custom = st.checkbox("启用自定义配置", key="holiday_use_custom")
        
        if use_custom:
            for i in range(1, history_years + 1):
                history_year = predict_start.year - i
                st.markdown(f"**{history_year}年配置**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    # 默认参考期
                    default_ref_start = predict_start.replace(year=history_year)
                    default_ref_end = predict_end.replace(year=history_year)
                    
                    ref_start = st.date_input(
                        f"参考期开始",
                        value=default_ref_start,
                        key=f"custom_ref_start_{history_year}"
                    )
                    ref_end = st.date_input(
                        f"参考期结束",
                        value=default_ref_end,
                        key=f"custom_ref_end_{history_year}"
                    )
                
                with col_b:
                    # 默认基期
                    history_base_month = predict_start.month - 1
                    history_base_year = history_year
                    if history_base_month < 1:
                        history_base_month = 12
                        history_base_year -= 1
                    
                    default_base_start = datetime(history_base_year, history_base_month, 1)
                    history_last_day = calendar.monthrange(history_base_year, history_base_month)[1]
                    default_base_end = datetime(history_base_year, history_base_month, history_last_day)
                    
                    base_start = st.date_input(
                        f"基期开始",
                        value=default_base_start,
                        key=f"custom_base_start_{history_year}"
                    )
                    base_end = st.date_input(
                        f"基期结束",
                        value=default_base_end,
                        key=f"custom_base_end_{history_year}"
                    )
                
                custom_configs[str(history_year)] = {
                    'ref_start': ref_start.strftime('%Y-%m-%d'),
                    'ref_end': ref_end.strftime('%Y-%m-%d'),
                    'base_start': base_start.strftime('%Y-%m-%d'),
                    'base_end': base_end.strftime('%Y-%m-%d'),
                }
                
                st.markdown("<hr style='border-color: rgba(0, 255, 213, 0.2); margin: 1rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # 预测按钮
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_clicked = st.button(
            "🚀 开始预测",
            use_container_width=True,
            type="primary",
            key="holiday_predict_btn"
        )
    
    # 执行预测
    if predict_clicked:
        predict_start_str = predict_start.strftime('%Y%m%d')
        predict_end_str = predict_end.strftime('%Y%m%d')
        
        # 检查是否使用自定义配置
        configs = custom_configs if (use_custom if 'use_custom' in dir() else False) and custom_configs else None
        
        with st.spinner("🔮 正在计算预测..."):
            try:
                # 调用预测函数
                if is_station:
                    result = predict_station_flow(
                        metric_type=metric_type,
                        predict_start=predict_start_str,
                        predict_end=predict_end_str,
                        history_years=history_years,
                        custom_configs=configs
                    )
                else:
                    result = predict_line_flow(
                        metric_type=metric_type,
                        predict_start=predict_start_str,
                        predict_end=predict_end_str,
                        history_years=history_years,
                        custom_configs=configs
                    )
                
                if not result.get('success', False):
                    st.error(f"❌ 预测失败: {result.get('error', '未知错误')}")
                    return
                
                predictions = result.get('predictions', [])
                has_actual = result.get('has_actual', False)
                metric_name = result.get('metric_name', '客流量')
                
                if not predictions:
                    st.warning("⚠️ 未生成任何预测结果，请检查数据是否充足")
                    return
                
                # 保存结果到session state
                st.session_state['holiday_predictions'] = predictions
                st.session_state['holiday_result'] = result
                st.session_state['holiday_is_station'] = is_station
                
                st.success(f"✅ 预测完成！共生成 {len(predictions)} 条预测记录")
                
            except Exception as e:
                st.error(f"❌ 预测过程出错: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # 显示结果
    if 'holiday_predictions' in st.session_state and st.session_state['holiday_predictions']:
        predictions = st.session_state['holiday_predictions']
        result = st.session_state.get('holiday_result', {})
        is_station = st.session_state.get('holiday_is_station', False)
        has_actual = result.get('has_actual', False)
        metric_name = result.get('metric_name', '客流量')
        
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # 结果摘要
        render_section_header(
            title="预测结果",
            subtitle=f"基期: {result.get('base_period', '-')} | 预测期: {result.get('predict_period', '-')}",
            icon="📈"
        )
        
        # 统计摘要
        stats = calculate_summary_stats(predictions, is_station)
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(0, 255, 213, 0.1), rgba(0, 255, 213, 0.05));
                border: 1px solid rgba(0, 255, 213, 0.3);
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
            ">
                <div style="color: #9aa5bd; font-size: 0.85rem;">预测记录数</div>
                <div style="color: #00ffd5; font-size: 1.8rem; font-weight: 700;">{stats.get('total_predictions', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s2:
            entity_name = "车站数" if is_station else "线路数"
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(191, 0, 255, 0.1), rgba(191, 0, 255, 0.05));
                border: 1px solid rgba(191, 0, 255, 0.3);
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
            ">
                <div style="color: #9aa5bd; font-size: 0.85rem;">{entity_name}</div>
                <div style="color: #bf00ff; font-size: 1.8rem; font-weight: 700;">{stats.get('unique_entities', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s3:
            avg_flow = stats.get('avg_predicted_flow', 0)
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 255, 136, 0.05));
                border: 1px solid rgba(0, 255, 136, 0.3);
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
            ">
                <div style="color: #9aa5bd; font-size: 0.85rem;">平均预测客流</div>
                <div style="color: #00ff88; font-size: 1.8rem; font-weight: 700;">{avg_flow:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s4:
            avg_acc = stats.get('avg_accuracy')
            if avg_acc is not None:
                acc_color = "#00ff88" if avg_acc >= 90 else ("#ffaa00" if avg_acc >= 80 else "#ff4757")
                acc_text = f"{avg_acc:.1f}%"
            else:
                acc_color = "#6c7893"
                acc_text = "待验证"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(255, 170, 0, 0.1), rgba(255, 170, 0, 0.05));
                border: 1px solid rgba(255, 170, 0, 0.3);
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
            ">
                <div style="color: #9aa5bd; font-size: 0.85rem;">平均准确率</div>
                <div style="color: {acc_color}; font-size: 1.8rem; font-weight: 700;">{acc_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # 图表展示
        tab_chart, tab_table, tab_detail = st.tabs(["📊 图表分析", "📋 数据表格", "🔍 详细信息"])
        
        with tab_chart:
            # 预测趋势图
            st.markdown("### 预测趋势")
            fig_pred = plot_predictions_chart(
                predictions, metric_name, is_station,
                SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_BG, SUBWAY_FONT
            )
            if fig_pred:
                st.pyplot(fig_pred)
                plt.close(fig_pred)
            
            # 准确率对比图（如果有实际数据）
            if has_actual:
                st.markdown("### 准确率对比")
                fig_acc = plot_accuracy_chart(
                    predictions, is_station,
                    SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_BG, SUBWAY_FONT
                )
                if fig_acc:
                    st.pyplot(fig_acc)
                    plt.close(fig_acc)
        
        with tab_table:
            # 数据表格
            df_display = format_predictions_for_display(predictions, is_station)
            st.dataframe(df_display, use_container_width=True, height=400)
            
            # 导出按钮
            csv_data = df_display.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 导出CSV",
                data=csv_data,
                file_name=f"节假日预测_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab_detail:
            # 历史参考详情
            history_details = result.get('history_details', [])
            if history_details:
                st.markdown("### 历史数据参考详情")
                for detail in history_details:
                    with st.expander(f"📅 {detail['year']}年", expanded=False):
                        st.markdown(f"""
                        - **参考期**: {detail['ref_period']}
                        - **基期**: {detail['base_period']}
                        """)
                        
                        if detail.get('line_stats'):
                            stats_df = pd.DataFrame(detail['line_stats'])
                            stats_df.columns = ['编号', '名称', '基期日均', '平均增长率', '最大增长率']
                            st.dataframe(stats_df, use_container_width=True)
            
            # 预测卡片详情
            st.markdown("### 预测详情卡片")
            id_field = '车站ID' if is_station else '线路编号'
            name_field = '车站名称' if is_station else '线路名称'
            
            # 按实体分组显示
            df_pred = pd.DataFrame(predictions)
            for entity_id in df_pred[id_field].unique()[:10]:  # 最多显示10个
                entity_data = df_pred[df_pred[id_field] == entity_id]
                entity_name = entity_data[name_field].iloc[0]
                
                with st.expander(f"🔹 {entity_name}", expanded=False):
                    for idx, (_, pred) in enumerate(entity_data.iterrows()):
                        render_prediction_card(pred.to_dict(), is_station, idx)


# 如果直接运行此模块（测试用）
if __name__ == "__main__":
    st.set_page_config(page_title="节假日预测测试", layout="wide")
    from ui_components import inject_global_styles
    inject_global_styles()
    holiday_tab()
