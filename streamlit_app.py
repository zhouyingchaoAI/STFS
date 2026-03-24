# streamlit_app.py
# 主框架：负责页面布局、主题、tab切换，调用小时/天预测子模块

import streamlit as st
from datetime import datetime
from config_utils import load_yaml_config, save_yaml_config
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

# 导入UI组件库
from ui_components import (
    CYBER_THEME,
    inject_global_styles,
    render_hero_banner,
    render_section_header,
    render_footer,
)

# ==================== 配置选项 ====================

FLOW_TYPES = {
    "xianwangxianlu": "线路线网",
    "duanmian": "断面",
    "chezhan": "车站"
}

FLOW_METRIC_OPTIONS = [
    ("F_PKLCOUNT", "客运量"),
    ("F_ENTRANCE", "进站量"),
    ("F_EXIT", "出站量"),
    ("F_TRANSFER", "换乘量"),
    ("F_BOARD_ALIGHT", "乘降量")
]

# ==================== 自定义选择器样式 ====================

def render_mode_selector():
    """渲染预测模式选择器"""
    st.markdown("""
    <style>
    .mode-selector-container {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .mode-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(15, 20, 35, 0.9), rgba(30, 36, 57, 0.7));
        border: 2px solid rgba(0, 255, 213, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .mode-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00ffd5, #bf00ff);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .mode-card:hover {
        border-color: rgba(0, 255, 213, 0.6);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 255, 213, 0.15);
    }
    
    .mode-card:hover::before {
        transform: scaleX(1);
    }
    
    .mode-card.active {
        border-color: #00ffd5;
        background: linear-gradient(135deg, rgba(0, 255, 213, 0.1), rgba(191, 0, 255, 0.05));
        box-shadow: 0 8px 32px rgba(0, 255, 213, 0.2);
    }
    
    .mode-card.active::before {
        transform: scaleX(1);
    }
    
    .mode-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 0 15px rgba(0, 255, 213, 0.5));
    }
    
    .mode-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #00ffd5;
        margin-bottom: 0.5rem;
    }
    
    .mode-desc {
        font-size: 0.9rem;
        color: #a8b2d1;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)


def render_config_panel(flow_type_key: str, selected_flow_metric_key: str):
    """渲染配置面板"""
    st.markdown("""
    <style>
    .config-panel {
        background: linear-gradient(135deg, rgba(15, 20, 35, 0.95), rgba(30, 36, 57, 0.8));
        border: 1px solid rgba(0, 255, 213, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .config-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(0, 255, 213, 0.15);
    }
    
    .config-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #00ffd5;
        margin: 0;
    }
    
    .config-badge {
        background: rgba(191, 0, 255, 0.2);
        border: 1px solid rgba(191, 0, 255, 0.4);
        color: #bf00ff;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    
    <div class="config-panel">
        <div class="config-header">
            <span style="font-size: 1.3rem;">⚙️</span>
            <h4 class="config-title">当前配置</h4>
            <span class="config-badge">实时更新</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """
    Streamlit 应用主入口，提供小时和日客流预测界面
    """
    st.set_page_config(
        page_title="客流模型算法测试平台",
        layout="wide",
        page_icon="🚇",
        initial_sidebar_state="collapsed"
    )

    # 注入全局样式
    inject_global_styles()
    render_mode_selector()

    # Hero横幅
    render_hero_banner(
        title="客流模型算法测试平台",
        subtitle="基于机器学习与深度学习技术的智能客流预测系统，支持多算法对比分析，实现精准预测与可视化展示",
        features=[
            "🔮 多算法融合",
            "🎯 高精度预测",
            "📊 实时可视化",
            "⚡ 智能分析",
            "🔧 参数调优",
            "📈 趋势预警"
        ],
        icon="🚇"
    )

    # 配置选择区域
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(15, 20, 35, 0.9), rgba(30, 36, 57, 0.7));
        border: 1px solid rgba(0, 255, 213, 0.2);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
    ">
        <div style="
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        ">
            <span style="font-size: 1.3rem;">⚙️</span>
            <span style="
                font-family: 'Rajdhani', sans-serif;
                font-size: 1.1rem;
                font-weight: 700;
                color: #00ffd5;
            ">预测配置</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 配置选项行
    col1, col2, col3 = st.columns([1, 1, 1.2], gap="medium")

    with col1:
        flow_type_label = st.selectbox(
            "🚇 客流类型",
            options=list(FLOW_TYPES.values()),
            index=0,
            key="main_flow_type_select",
            help="选择要分析的客流数据类型"
        )
        flow_type_key = [k for k, v in FLOW_TYPES.items() if v == flow_type_label][0]

    with col2:
        flow_metric_labels = [label for _, label in FLOW_METRIC_OPTIONS]
        selected_flow_metric_label = st.selectbox(
            "📊 预测指标",
            options=flow_metric_labels,
            index=0,
            key="main_flow_metric_select",
            help="选择要预测的客流量指标"
        )
        selected_flow_metric_key = [k for k, v in FLOW_METRIC_OPTIONS if v == selected_flow_metric_label][0]

    with col3:
        tab_options = [
            f"📅 {flow_type_label} - 日预测",
            f"🕐 {flow_type_label} - 小时预测",
            f"🎯 节假日预测 (人工算法)",
            f"📚 {flow_type_label} - 预测结果查看",
        ]
        tab_choice = st.selectbox(
            "⏱️ 预测模式",
            options=tab_options,
            key="prediction_mode_select",
            help="选择日级别、小时级别或节假日预测"
        )

    # 显示当前配置状态
    if "节假日" in tab_choice:
        mode_text = "节假日预测"
    elif "预测结果查看" in tab_choice:
        mode_text = "预测结果查看"
    elif "日预测" in tab_choice:
        mode_text = "日预测"
    else:
        mode_text = "小时预测"
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
            <span style="color: #00ffd5;">🚇</span>
            <span style="color: #ffffff; font-weight: 600;">类型: {flow_type_label}</span>
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
            <span style="color: #ffffff; font-weight: 600;">指标: {selected_flow_metric_label}</span>
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
            <span style="color: #00ff88;">✓</span>
            <span style="color: #ffffff; font-weight: 600;">模式: {mode_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 根据选择加载对应模块
    if tab_choice is not None:
        if isinstance(tab_choice, str) and "节假日" in tab_choice:
            # 节假日预测模块 - 基于人工算法
            try:
                from streamlit_holiday import holiday_tab
            except ModuleNotFoundError:
                from .streamlit_holiday import holiday_tab
            holiday_tab(
                SUBWAY_GREEN=CYBER_THEME["PRIMARY"],
                SUBWAY_ACCENT=CYBER_THEME["ACCENT"],
                SUBWAY_CARD=CYBER_THEME["BG_SURFACE"],
                SUBWAY_FONT=CYBER_THEME["TEXT_PRIMARY"],
                SUBWAY_BG=CYBER_THEME["BG_BASE"],
                flow_type=flow_type_key,
                metric_type=selected_flow_metric_key
            )
        elif isinstance(tab_choice, str) and "小时预测" in tab_choice:
            try:
                from streamlit_hourly import hourly_tab
            except ModuleNotFoundError:
                from .streamlit_hourly import hourly_tab
            hourly_tab(
                SUBWAY_GREEN=CYBER_THEME["PRIMARY"],
                SUBWAY_ACCENT=CYBER_THEME["ACCENT"],
                SUBWAY_CARD=CYBER_THEME["BG_SURFACE"],
                SUBWAY_FONT=CYBER_THEME["TEXT_PRIMARY"],
                SUBWAY_BG=CYBER_THEME["BG_BASE"],
                flow_type=flow_type_key,
                metric_type=selected_flow_metric_key
            )
        elif isinstance(tab_choice, str) and "预测结果查看" in tab_choice:
            try:
                from streamlit_prediction_view import prediction_view_tab
            except ModuleNotFoundError:
                from .streamlit_prediction_view import prediction_view_tab
            prediction_view_tab(
                SUBWAY_GREEN=CYBER_THEME["PRIMARY"],
                SUBWAY_ACCENT=CYBER_THEME["ACCENT"],
                SUBWAY_CARD=CYBER_THEME["BG_SURFACE"],
                SUBWAY_FONT=CYBER_THEME["TEXT_PRIMARY"],
                SUBWAY_BG=CYBER_THEME["BG_BASE"],
                flow_type=flow_type_key,
                metric_type=selected_flow_metric_key
            )
        elif isinstance(tab_choice, str) and "日预测" in tab_choice:
            try:
                from streamlit_daily import daily_tab
            except ModuleNotFoundError:
                from .streamlit_daily import daily_tab
            daily_tab(
                SUBWAY_GREEN=CYBER_THEME["PRIMARY"],
                SUBWAY_ACCENT=CYBER_THEME["ACCENT"],
                SUBWAY_CARD=CYBER_THEME["BG_SURFACE"],
                SUBWAY_FONT=CYBER_THEME["TEXT_PRIMARY"],
                SUBWAY_BG=CYBER_THEME["BG_BASE"],
                flow_type=flow_type_key,
                metric_type=selected_flow_metric_key
            )
    else:
        st.warning("未检测到有效的预测模式选项，请重新选择。")

    # 页脚
    render_footer()


if __name__ == "__main__":
    main()
