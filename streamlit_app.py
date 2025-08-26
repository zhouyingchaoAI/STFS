# streamlit_app.py
# 主框架：负责页面布局、主题、tab切换，调用小时/天预测子模块

import streamlit as st
from datetime import datetime
from config_utils import load_yaml_config, save_yaml_config
import os

# --------- 仅深色主题色与样式设置 ---------
DARK_THEME = {
    "SUBWAY_PRIMARY": "#00e09e",
    "SUBWAY_SECONDARY": "#00bfff",
    "SUBWAY_ACCENT": "#7c4dff",
    "SUBWAY_PINK": "#ff4081",
    "SUBWAY_ORANGE": "#ff9800",
    "SUBWAY_YELLOW": "#ffc107",
    "SUBWAY_BG": "#0a0d1a",
    "SUBWAY_CARD": "#1a1f2e",
    "SUBWAY_SURFACE": "#242938",
    "SUBWAY_FONT": "#ffffff",
    "SUBWAY_FONT_SECONDARY": "#b8c5d6",
    "SUBWAY_DARK": "#121620",
    "SUBWAY_SUCCESS": "#4caf50",
    "SUBWAY_WARNING": "#ff9800",
    "SUBWAY_ERROR": "#f44336",
}

def subway_optimized_style(theme):
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        :root {{
            --primary: {theme['SUBWAY_PRIMARY']};
            --secondary: {theme['SUBWAY_SECONDARY']};
            --accent: {theme['SUBWAY_ACCENT']};
            --pink: {theme['SUBWAY_PINK']};
            --orange: {theme['SUBWAY_ORANGE']};
            --yellow: {theme['SUBWAY_YELLOW']};
            --bg: {theme['SUBWAY_BG']};
            --card: {theme['SUBWAY_CARD']};
            --surface: {theme['SUBWAY_SURFACE']};
            --font: {theme['SUBWAY_FONT']};
            --font-secondary: {theme['SUBWAY_FONT_SECONDARY']};
            --dark: {theme['SUBWAY_DARK']};
            --success: {theme['SUBWAY_SUCCESS']};
            --warning: {theme['SUBWAY_WARNING']};
            --error: {theme['SUBWAY_ERROR']};
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body, .stApp {{
            background: linear-gradient(135deg, var(--bg) 0%, var(--dark) 100%);
            color: var(--font);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            min-height: 100vh;
        }}
        
        .stApp {{
            background-attachment: fixed;
        }}
        
        .block-container {{
            padding: 2rem 1.5rem 3rem 1.5rem;
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .subway-hero-banner {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 35%, var(--accent) 70%, var(--pink) 100%);
            padding: 3rem 2.5rem 2.5rem 2.5rem;
            border-radius: 24px;
            margin-bottom: 3rem;
            position: relative;
            overflow: hidden;
            box-shadow: 
                0 20px 40px rgba(0, 224, 158, 0.15),
                0 10px 20px rgba(0, 191, 255, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .subway-hero-banner::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(124, 77, 255, 0.1) 0%, transparent 50%);
            pointer-events: none;
        }}
        
        .subway-hero-banner h1 {{
            color: var(--dark);
            font-weight: 900;
            font-size: clamp(2.2rem, 4vw, 3.2rem);
            letter-spacing: -0.02em;
            margin-bottom: 0.8rem;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            line-height: 1.1;
        }}
        
        .subway-hero-banner .hero-subtitle {{
            color: var(--dark);
            font-size: clamp(1rem, 2vw, 1.25rem);
            font-weight: 500;
            line-height: 1.6;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }}
        
        .subway-hero-banner .hero-features {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1.5rem;
            position: relative;
            z-index: 1;
        }}
        
        .hero-feature-tag {{
            background: rgba(18, 22, 32, 0.8);
            color: var(--primary);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(0, 224, 158, 0.3);
            backdrop-filter: blur(10px);
        }}
        
        div[data-testid="stRadio"] {{
            background: var(--card);
            padding: 1.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        div[data-testid="stRadio"] > label {{
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 1rem;
            display: block;
        }}
        
        div[data-testid="stRadio"] > div {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        
        /* --- 修改tab切换按钮字体颜色，增强可读性 --- */
        div[data-testid="stRadio"] label[data-baseweb="radio"] {{
            background: var(--surface);
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border: 2px solid transparent;
            cursor: pointer;
            transition: all 0.3s ease;
            flex: 1;
            min-width: 200px;
            text-align: center;
            font-weight: 700;
            color: var(--font); /* 由原来的var(--font-secondary)改为var(--font)以增强对比度 */
            font-size: 1.08rem;
            letter-spacing: 0.01em;
            text-shadow: 0 1px 2px rgba(0,0,0,0.10);
        }}
        
        div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
            border-color: var(--primary);
            background: linear-gradient(135deg, var(--primary)10, var(--secondary)10);
            color: var(--font);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 224, 158, 0.15);
        }}
        
        div[data-testid="stRadio"] input:checked + label {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: var(--dark);
            border-color: var(--primary);
            box-shadow: 0 4px 20px rgba(0, 224, 158, 0.3);
            font-weight: 900;
            text-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        /* --- 加粗并增大“线路小时客流预测”tab字体，提升可读性 --- */
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(span:contains('线路小时客流预测')) {{
            font-size: 1.25rem !important;
            font-weight: 900 !important;
            color: var(--primary) !important;
            text-shadow: 0 2px 8px rgba(0,224,158,0.18) !important;
            letter-spacing: 0.02em !important;
        }}
        /* 针对streamlit 1.32+，用属性选择器匹配tab内容 */
        div[data-testid="stRadio"] label[data-baseweb="radio"] span:contains('线路小时客流预测') {{
            font-size: 1.25rem !important;
            font-weight: 900 !important;
            color: var(--primary) !important;
            text-shadow: 0 2px 8px rgba(0,224,158,0.18) !important;
            letter-spacing: 0.02em !important;
        }}

        /* 兼容性：如果:has和:contains不生效，则用nth-child(1)高亮第一个tab */
        div[data-testid="stRadio"] > div > label[data-baseweb="radio"]:nth-child(1) {{
            font-size: 1.25rem !important;
            font-weight: 900 !important;
            color: var(--primary) !important;
            text-shadow: 0 2px 8px rgba(0,224,158,0.18) !important;
            letter-spacing: 0.02em !important;
        }}
        
        .stContainer > div, 
        div[data-testid="column"] > div,
        .element-container {{
            background: var(--card);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }}
        
        .stButton > button {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: var(--dark);
            font-weight: 700;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 224, 158, 0.3);
            position: relative;
            overflow: hidden;
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, var(--secondary) 0%, var(--accent) 100%);
            color: var(--font);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 191, 255, 0.4);
        }}
        
        .stButton > button:active {{
            transform: translateY(0);
        }}
        
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > input {{
            background: var(--surface);
            color: var(--font);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > input:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(0, 224, 158, 0.1);
        }}
        
        .stDataFrame {{
            background: var(--card) !important;
            border-radius: 12px !important;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
        }}
        
        .stDataFrame table {{
            background: transparent !important;
        }}
        
        .stDataFrame th {{
            background: linear-gradient(135deg, var(--primary)20, var(--secondary)20) !important;
            color: var(--font) !important;
            font-weight: 700 !important;
            border: none !important;
            padding: 1rem !important;
        }}
        
        .stDataFrame td {{
            color: var(--font-secondary) !important;
            border: none !important;
            padding: 0.75rem 1rem !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        }}
        
        .stPlotlyChart, .stAltairChart, .stPyplotChart {{
            background: var(--card) !important;
            border-radius: 16px !important;
            padding: 1rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }}
        
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--card) 0%, var(--surface) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
            color: var(--font);
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 1rem;
            line-height: 1.2;
        }}
        
        .stMarkdown h2 {{
            color: var(--primary);
            font-size: 1.8rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .stMarkdown h3 {{
            color: var(--secondary);
            font-size: 1.4rem;
        }}
        
        .stSuccess {{
            background: linear-gradient(135deg, var(--success)15, var(--success)05);
            border: 1px solid var(--success);
            border-radius: 12px;
            color: var(--font);
        }}
        
        .stWarning {{
            background: linear-gradient(135deg, var(--warning)15, var(--warning)05);
            border: 1px solid var(--warning);
            border-radius: 12px;
            color: var(--font);
        }}
        
        .stError {{
            background: linear-gradient(135deg, var(--error)15, var(--error)05);
            border: 1px solid var(--error);
            border-radius: 12px;
            color: var(--font);
        }}
        
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: var(--surface);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: linear-gradient(135deg, var(--secondary), var(--accent));
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .loading {{
            animation: pulse 2s infinite;
        }}
        
        @media (max-width: 768px) {{
            .block-container {{
                padding: 1rem;
            }}
            
            .subway-hero-banner {{
                padding: 2rem 1.5rem;
                margin-bottom: 2rem;
            }}
            
            .subway-hero-banner h1 {{
                font-size: 2rem;
            }}
            
            .hero-features {{
                flex-direction: column;
            }}
            
            div[data-testid="stRadio"] > div {{
                flex-direction: column;
            }}
            
            div[data-testid="stRadio"] label[data-baseweb="radio"] {{
                min-width: unset;
            }}
        }}
        
        * {{
            -webkit-transform: translateZ(0);
            transform: translateZ(0);
        }}
        
        .stApp {{
            will-change: transform;
            backface-visibility: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    # 只保留深色主题
    theme = DARK_THEME

    # 应用优化样式
    subway_optimized_style(theme)

    # Hero横幅
    st.markdown(
        f"""
        <div class="subway-hero-banner">
            <h1>🚇 客流模型算法测试平台</h1>
            <div class="hero-subtitle">
                基于机器学习与深度学习技术的智能客流预测系统，支持多算法对比分析，实现精准预测与可视化展示
            </div>
            <div class="hero-features">
                <div class="hero-feature-tag">📊 多算法支持</div>
                <div class="hero-feature-tag">🎯 高精度预测</div>
                <div class="hero-feature-tag">📈 实时可视化</div>
                <div class="hero-feature-tag">⚡ 智能分析</div>
                <div class="hero-feature-tag">🔧 参数调优</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 将两个下拉框放在同一行
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        # 定义 FLOW_TYPES 下拉选项
        FLOW_TYPES = {
            "xianwangxianlu": "线路线网",
            "duanmian": "断面",
            "chezhan": "车站"
        }
        # 选择客流类型
        flow_type_label = st.selectbox(
            "🚇 选择客流类型",
            options=list(FLOW_TYPES.values()),
            index=0,
            key="main_flow_type_select"
        )
        # 反查 key
        flow_type_key = [k for k, v in FLOW_TYPES.items() if v == flow_type_label][0]

    with col2:
        # 预测客流类型下拉框
        FLOW_METRIC_OPTIONS = [
            ("F_PKLCOUNT", "客运量"),
            ("F_ENTRANCE", "进站量"),
            ("F_EXIT", "出站量"),
            ("F_TRANSFER", "换乘量"),
            ("F_BOARD_ALIGHT", "乘降量")
        ]
        flow_metric_labels = [label for _, label in FLOW_METRIC_OPTIONS]
        selected_flow_metric_label = st.selectbox(
            "🚦 选择预测客流范围",
            options=flow_metric_labels,
            index=0,
            key="main_flow_metric_select",
            help="选择要预测的客流量类型"
        )
        selected_flow_metric_key = [k for k, v in FLOW_METRIC_OPTIONS if v == selected_flow_metric_label][0]

    # tab切换，动态显示FLOW_TYPES 这个也需要完整占一行
    # 横向铺满整个屏幕的 tab/radio，使用 st.columns 实现
    col_radio = st.columns([1], gap="medium")[0]
    with col_radio:
        tab_choice = st.radio(
            "🎯 选择预测模式",
            [
                f"📅 {flow_type_label}日客流预测",
                f"🕒 {flow_type_label}小时客流预测",
            ],
            horizontal=True,
            index=0,
            key="main_tab_radio",
            # help="选择您需要进行预测的时间粒度"
        )

    # st.markdown("---")

    # 根据选择加载对应模块，并增强字体颜色对比度
    BRIGHT_FONT_COLOR = "#FFFFFF"  # 纯白色，确保字体鲜明

    # 防止 tab_choice 为 None 时出现 AttributeError
    if tab_choice is not None:
        if isinstance(tab_choice, str) and tab_choice.startswith("🕒"):
            from streamlit_hourly import hourly_tab
            hourly_tab(
                SUBWAY_GREEN=theme["SUBWAY_PRIMARY"],
                SUBWAY_ACCENT=theme["SUBWAY_SECONDARY"],
                SUBWAY_CARD=theme["SUBWAY_CARD"],
                SUBWAY_FONT=BRIGHT_FONT_COLOR,  # 使用更鲜明的字体颜色
                SUBWAY_BG=theme["SUBWAY_BG"],
                flow_type=flow_type_key,
                metric_type=selected_flow_metric_key
            )
        elif isinstance(tab_choice, str) and tab_choice.startswith("📅"):
            from streamlit_daily import daily_tab
            daily_tab(
                SUBWAY_GREEN=theme["SUBWAY_PRIMARY"],
                SUBWAY_ACCENT=theme["SUBWAY_SECONDARY"],
                SUBWAY_CARD=theme["SUBWAY_CARD"],
                SUBWAY_FONT=BRIGHT_FONT_COLOR,  # 使用更鲜明的字体颜色
                SUBWAY_BG=theme["SUBWAY_BG"],
                flow_type=flow_type_key,
                metric_type=selected_flow_metric_key
            )
    else:
        st.warning("未检测到有效的预测模式选项，请重新选择。")

    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: var(--font-secondary); font-size: 0.9rem; padding: 2rem 0;">
            <p>🚇 客流模型算法测试平台 | 科技赋能智慧交通 | Powered by Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()