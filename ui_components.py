# ui_components.py
# 统一UI组件库：提供美观、一致的界面组件

import streamlit as st
from typing import Optional, List, Dict, Any
from datetime import datetime

# ==================== 设计系统 ====================

# 赛博朋克风格主题 - 更独特的美学
CYBER_THEME = {
    # 主色调 - 霓虹蓝绿
    "PRIMARY": "#00ffd5",
    "PRIMARY_DARK": "#00b8a9",
    "PRIMARY_GLOW": "rgba(0, 255, 213, 0.4)",
    
    # 强调色 - 电紫色
    "ACCENT": "#bf00ff",
    "ACCENT_DARK": "#8b00b8",
    "ACCENT_GLOW": "rgba(191, 0, 255, 0.4)",
    
    # 警告/热色
    "HOT_PINK": "#ff0080",
    "ORANGE": "#ff6b35",
    "YELLOW": "#ffd700",
    
    # 成功/信息
    "SUCCESS": "#00ff88",
    "INFO": "#00b4d8",
    "WARNING": "#ffaa00",
    "ERROR": "#ff4757",
    
    # 背景系统 - 深色分层
    "BG_DEEPEST": "#05080f",
    "BG_DEEP": "#0a0e1a",
    "BG_BASE": "#0f1423",
    "BG_ELEVATED": "#161b2e",
    "BG_SURFACE": "#1e2439",
    "BG_OVERLAY": "#252b44",
    
    # 文字 - 提高对比度
    "TEXT_PRIMARY": "#ffffff",
    "TEXT_SECONDARY": "#d0d7e8",  # 提亮次要文字
    "TEXT_MUTED": "#9aa5bd",      # 提亮静音文字
    
    # 边框
    "BORDER": "rgba(0, 255, 213, 0.2)",
    "BORDER_ACTIVE": "rgba(0, 255, 213, 0.6)",
}

# 字体系统
FONTS = {
    "DISPLAY": "'Orbitron', 'Rajdhani', 'Share Tech Mono', sans-serif",
    "HEADING": "'Rajdhani', 'Share Tech', 'Inter', sans-serif",
    "BODY": "'Inter', 'Segoe UI', system-ui, sans-serif",
    "MONO": "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
}


def inject_global_styles():
    """注入全局CSS样式"""
    theme = CYBER_THEME
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* ===== 全局重置 ===== */
    :root {{
        --primary: {theme['PRIMARY']};
        --primary-dark: {theme['PRIMARY_DARK']};
        --primary-glow: {theme['PRIMARY_GLOW']};
        --accent: {theme['ACCENT']};
        --accent-dark: {theme['ACCENT_DARK']};
        --accent-glow: {theme['ACCENT_GLOW']};
        --hot-pink: {theme['HOT_PINK']};
        --orange: {theme['ORANGE']};
        --yellow: {theme['YELLOW']};
        --success: {theme['SUCCESS']};
        --info: {theme['INFO']};
        --warning: {theme['WARNING']};
        --error: {theme['ERROR']};
        --bg-deepest: {theme['BG_DEEPEST']};
        --bg-deep: {theme['BG_DEEP']};
        --bg-base: {theme['BG_BASE']};
        --bg-elevated: {theme['BG_ELEVATED']};
        --bg-surface: {theme['BG_SURFACE']};
        --bg-overlay: {theme['BG_OVERLAY']};
        --text-primary: {theme['TEXT_PRIMARY']};
        --text-secondary: {theme['TEXT_SECONDARY']};
        --text-muted: {theme['TEXT_MUTED']};
        --border: {theme['BORDER']};
        --border-active: {theme['BORDER_ACTIVE']};
    }}
    
    * {{
        box-sizing: border-box;
    }}
    
    /* ===== 主背景 ===== */
    .stApp {{
        background: 
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0, 255, 213, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse 60% 40% at 100% 100%, rgba(191, 0, 255, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse 40% 60% at 0% 50%, rgba(255, 0, 128, 0.04) 0%, transparent 50%),
            linear-gradient(180deg, var(--bg-deepest) 0%, var(--bg-deep) 50%, var(--bg-base) 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }}
    
    /* ===== 网格背景动画 ===== */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            linear-gradient(rgba(0, 255, 213, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 213, 0.03) 1px, transparent 1px);
        background-size: 60px 60px;
        pointer-events: none;
        z-index: 0;
        animation: gridPulse 8s ease-in-out infinite;
    }}
    
    @keyframes gridPulse {{
        0%, 100% {{ opacity: 0.3; }}
        50% {{ opacity: 0.6; }}
    }}
    
    /* ===== 容器样式 ===== */
    .block-container {{
        padding: 2rem 2rem 4rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }}
    
    /* ===== 按钮样式 ===== */
    .stButton > button {{
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 50%, var(--accent) 100%);
        background-size: 200% 200%;
        color: var(--bg-deepest);
        font-family: {FONTS['HEADING']};
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: none;
        border-radius: 8px;
        padding: 0.85rem 2rem;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 4px 20px var(--primary-glow),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s ease;
    }}
    
    .stButton > button:hover {{
        background-position: 100% 0;
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 8px 30px var(--primary-glow),
            0 4px 15px var(--accent-glow),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) scale(0.98);
    }}
    
    /* ===== 输入框样式 ===== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background: #1e2439 !important;
        color: #ffffff !important;
        border: 2px solid rgba(0, 255, 213, 0.3) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        font-family: {FONTS['BODY']} !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: #00ffd5 !important;
        box-shadow: 0 0 0 3px rgba(0, 255, 213, 0.2) !important;
        outline: none !important;
    }}
    
    /* 数字输入框按钮 */
    .stNumberInput button {{
        background: #1e2439 !important;
        border-color: rgba(0, 255, 213, 0.3) !important;
        color: #00ffd5 !important;
    }}
    
    .stNumberInput button:hover {{
        background: rgba(0, 255, 213, 0.15) !important;
    }}
    
    /* ===== 选择框样式 ===== */
    .stSelectbox [data-baseweb="select"] {{
        background: transparent !important;
    }}
    
    .stSelectbox [data-baseweb="select"] > div {{
        background: #1e2439 !important;
        border: 2px solid rgba(0, 255, 213, 0.3) !important;
        border-radius: 10px !important;
    }}
    
    .stSelectbox [data-baseweb="select"] > div > div {{
        color: #ffffff !important;
    }}
    
    .stSelectbox [data-baseweb="select"]:focus-within > div {{
        border-color: #00ffd5 !important;
        box-shadow: 0 0 0 3px rgba(0, 255, 213, 0.2) !important;
    }}
    
    /* 下拉菜单样式 */
    [data-baseweb="popover"] {{
        background: #1e2439 !important;
        border: 1px solid rgba(0, 255, 213, 0.3) !important;
        border-radius: 10px !important;
    }}
    
    [data-baseweb="menu"] {{
        background: #1e2439 !important;
    }}
    
    [data-baseweb="menu"] li {{
        background: transparent !important;
        color: #ffffff !important;
    }}
    
    [data-baseweb="menu"] li:hover {{
        background: rgba(0, 255, 213, 0.15) !important;
    }}
    
    [data-baseweb="menu"] li[aria-selected="true"] {{
        background: rgba(0, 255, 213, 0.25) !important;
    }}
    
    /* 选择框文字颜色 */
    .stSelectbox span, .stSelectbox div {{
        color: #ffffff !important;
    }}
    
    /* 选择框箭头图标 */
    .stSelectbox svg {{
        fill: #00ffd5 !important;
    }}
    
    /* ===== 复选框样式 ===== */
    .stCheckbox > label {{
        color: #e0e5f0 !important;
        font-family: {FONTS['BODY']} !important;
    }}
    
    .stCheckbox > label > span[data-baseweb="checkbox"] {{
        background: #1e2439 !important;
        border-color: rgba(0, 255, 213, 0.3) !important;
    }}
    
    .stCheckbox > label > span[data-baseweb="checkbox"]:has(input:checked) {{
        background: #00ffd5 !important;
        border-color: #00ffd5 !important;
    }}
    
    .stCheckbox p {{
        color: #e0e5f0 !important;
    }}
    
    /* ===== 日期选择器 ===== */
    .stDateInput > div > div > input {{
        background: #1e2439 !important;
        color: #ffffff !important;
        border: 2px solid rgba(0, 255, 213, 0.3) !important;
        border-radius: 10px !important;
    }}
    
    .stDateInput > div > div > input:focus {{
        border-color: #00ffd5 !important;
        box-shadow: 0 0 0 3px rgba(0, 255, 213, 0.2) !important;
    }}
    
    /* 日期选择弹窗 */
    [data-baseweb="calendar"] {{
        background: #1e2439 !important;
        color: #ffffff !important;
    }}
    
    [data-baseweb="calendar"] div {{
        color: #ffffff !important;
    }}
    
    [data-baseweb="calendar"] button {{
        color: #ffffff !important;
    }}
    
    [data-baseweb="calendar"] button:hover {{
        background: rgba(0, 255, 213, 0.15) !important;
    }}
    
    /* ===== 展开器样式 ===== */
    .streamlit-expanderHeader {{
        background: #1e2439 !important;
        border: 1px solid rgba(0, 255, 213, 0.3) !important;
        border-radius: 10px !important;
        color: #e0e5f0 !important;
        font-family: {FONTS['HEADING']} !important;
        font-weight: 600 !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        border-color: #00ffd5 !important;
        color: #00ffd5 !important;
    }}
    
    .streamlit-expanderContent {{
        background: #161b2e !important;
        border: 1px solid rgba(0, 255, 213, 0.2) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }}
    
    /* 展开器内文字 */
    .streamlit-expanderContent p,
    .streamlit-expanderContent span,
    .streamlit-expanderContent label {{
        color: #e0e5f0 !important;
    }}
    
    /* 展开器箭头 */
    .streamlit-expanderHeader svg {{
        fill: #00ffd5 !important;
    }}
    
    /* ===== 数据表格 ===== */
    .stDataFrame {{
        background: var(--bg-surface) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid var(--border) !important;
    }}
    
    .stDataFrame [data-testid="stDataFrameResizable"] {{
        background: var(--bg-surface) !important;
    }}
    
    /* ===== 图表容器 ===== */
    .stPlotlyChart, .stPyplotChart {{
        background: var(--bg-surface) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3) !important;
    }}
    
    /* ===== Spinner/加载动画 ===== */
    .stSpinner > div {{
        border-color: var(--primary) !important;
    }}
    
    /* ===== 成功/警告/错误消息 ===== */
    .stSuccess {{
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 255, 136, 0.05) 100%) !important;
        border: 1px solid var(--success) !important;
        border-radius: 10px !important;
        color: var(--success) !important;
    }}
    
    .stWarning {{
        background: linear-gradient(135deg, rgba(255, 170, 0, 0.1) 0%, rgba(255, 170, 0, 0.05) 100%) !important;
        border: 1px solid var(--warning) !important;
        border-radius: 10px !important;
        color: var(--warning) !important;
    }}
    
    .stError {{
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.1) 0%, rgba(255, 71, 87, 0.05) 100%) !important;
        border: 1px solid var(--error) !important;
        border-radius: 10px !important;
        color: var(--error) !important;
    }}
    
    /* ===== 滚动条 ===== */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--bg-deep);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, var(--primary-dark), var(--accent));
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, var(--primary), var(--accent));
    }}
    
    /* ===== 标签样式 ===== */
    .stMarkdown label, 
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label,
    .stDateInput label {{
        color: #e8ecf4 !important;
        font-family: {FONTS['BODY']} !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }}
    
    /* ===== 所有文本提亮 ===== */
    .stMarkdown, .stMarkdown p, .stText {{
        color: #e0e5f0 !important;
    }}
    
    /* 确保help文本可见 */
    .stTooltipIcon {{
        color: #9aa5bd !important;
    }}
    
    /* 工具提示弹窗 */
    [data-baseweb="tooltip"] {{
        background: #1e2439 !important;
        color: #ffffff !important;
        border: 1px solid rgba(0, 255, 213, 0.3) !important;
    }}
    
    /* 确保所有输入组件的占位符可见 */
    input::placeholder {{
        color: #6c7893 !important;
        opacity: 1 !important;
    }}
    
    /* multiselect 样式 */
    .stMultiSelect [data-baseweb="tag"] {{
        background: rgba(0, 255, 213, 0.2) !important;
        color: #00ffd5 !important;
    }}
    
    .stMultiSelect [data-baseweb="tag"] span {{
        color: #00ffd5 !important;
    }}
    
    /* 警告/信息框内文字 */
    .stAlert p, .stAlert span {{
        color: inherit !important;
    }}
    
    /* ===== 分隔线 ===== */
    hr {{
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--primary), var(--accent), transparent) !important;
        margin: 2rem 0 !important;
    }}
    
    /* ===== 侧边栏 ===== */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--bg-deep) 0%, var(--bg-base) 100%) !important;
        border-right: 1px solid var(--border) !important;
    }}
    
    section[data-testid="stSidebar"] .block-container {{
        padding: 2rem 1rem !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_hero_banner(
    title: str,
    subtitle: str,
    features: List[str],
    icon: str = "🚇"
):
    """渲染霓虹风格的Hero横幅"""
    features_html = "".join([
        f'<span class="hero-tag">{f}</span>' for f in features
    ])
    
    st.markdown(f"""
    <style>
    .cyber-hero {{
        position: relative;
        background: linear-gradient(135deg, 
            rgba(0, 255, 213, 0.15) 0%, 
            rgba(191, 0, 255, 0.1) 50%, 
            rgba(255, 0, 128, 0.08) 100%);
        border: 1px solid rgba(0, 255, 213, 0.3);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        margin-bottom: 2.5rem;
        overflow: hidden;
    }}
    
    .cyber-hero::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(
            from 0deg at 50% 50%,
            transparent 0deg,
            rgba(0, 255, 213, 0.1) 60deg,
            transparent 120deg,
            rgba(191, 0, 255, 0.1) 180deg,
            transparent 240deg,
            rgba(255, 0, 128, 0.1) 300deg,
            transparent 360deg
        );
        animation: heroRotate 20s linear infinite;
        pointer-events: none;
    }}
    
    @keyframes heroRotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    
    .cyber-hero::after {{
        content: '';
        position: absolute;
        inset: 0;
        background: 
            radial-gradient(circle at 20% 80%, rgba(0, 255, 213, 0.2) 0%, transparent 40%),
            radial-gradient(circle at 80% 20%, rgba(191, 0, 255, 0.15) 0%, transparent 40%);
        pointer-events: none;
    }}
    
    .hero-content {{
        position: relative;
        z-index: 1;
    }}
    
    .hero-icon {{
        font-size: 3.5rem;
        margin-bottom: 1rem;
        display: inline-block;
        animation: iconFloat 3s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(0, 255, 213, 0.5));
    }}
    
    @keyframes iconFloat {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-10px); }}
    }}
    
    .hero-title {{
        font-family: 'Orbitron', 'Rajdhani', sans-serif;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 900;
        background: linear-gradient(135deg, #00ffd5 0%, #00b4d8 30%, #bf00ff 70%, #ff0080 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        letter-spacing: 0.02em;
        text-shadow: 0 0 40px rgba(0, 255, 213, 0.3);
        line-height: 1.2;
    }}
    
    .hero-subtitle {{
        font-family: 'Inter', sans-serif;
        font-size: clamp(1rem, 1.8vw, 1.2rem);
        color: #e0e5f0;
        line-height: 1.7;
        max-width: 700px;
    }}
    
    .hero-tags {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.5rem;
    }}
    
    .hero-tag {{
        background: rgba(0, 255, 213, 0.1);
        border: 1px solid rgba(0, 255, 213, 0.4);
        color: #00ffd5;
        padding: 0.5rem 1.2rem;
        border-radius: 30px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        transition: all 0.3s ease;
        cursor: default;
    }}
    
    .hero-tag:hover {{
        background: rgba(0, 255, 213, 0.2);
        border-color: #00ffd5;
        box-shadow: 0 0 20px rgba(0, 255, 213, 0.3);
        transform: translateY(-2px);
    }}
    </style>
    
    <div class="cyber-hero">
        <div class="hero-content">
            <div class="hero-icon">{icon}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
            <div class="hero-tags">{features_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_glass_card(
    title: str,
    content: str = "",
    icon: str = "",
    color: str = "primary"
):
    """渲染玻璃态卡片"""
    color_map = {
        "primary": ("#00ffd5", "rgba(0, 255, 213, 0.15)"),
        "accent": ("#bf00ff", "rgba(191, 0, 255, 0.15)"),
        "pink": ("#ff0080", "rgba(255, 0, 128, 0.15)"),
        "orange": ("#ff6b35", "rgba(255, 107, 53, 0.15)"),
        "success": ("#00ff88", "rgba(0, 255, 136, 0.15)"),
    }
    accent_color, bg_color = color_map.get(color, color_map["primary"])
    
    st.markdown(f"""
    <div style="
        background: {bg_color};
        backdrop-filter: blur(10px);
        border: 1px solid {accent_color}40;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    " onmouseover="this.style.borderColor='{accent_color}80'; this.style.boxShadow='0 8px 32px {accent_color}20';" 
       onmouseout="this.style.borderColor='{accent_color}40'; this.style.boxShadow='none';">
        <div style="position: relative; z-index: 1;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                {f'<span style="font-size: 1.5rem;">{icon}</span>' if icon else ''}
                <h3 style="
                    margin: 0;
                    color: {accent_color};
                    font-family: 'Rajdhani', sans-serif;
                    font-size: 1.2rem;
                    font-weight: 700;
                    letter-spacing: 0.02em;
                ">{title}</h3>
            </div>
            {f'<p style="margin: 0; color: #e0e5f0; font-size: 0.95rem; line-height: 1.6;">{content}</p>' if content else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = "", icon: str = ""):
    """渲染段落标题"""
    st.markdown(f"""
    <div style="margin: 2rem 0 1.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            {f'<span style="font-size: 1.8rem; filter: drop-shadow(0 0 10px rgba(0, 255, 213, 0.5));">{icon}</span>' if icon else ''}
            <div>
                <h2 style="
                    margin: 0;
                    font-family: 'Orbitron', 'Rajdhani', sans-serif;
                    font-size: 1.6rem;
                    font-weight: 700;
                    background: linear-gradient(135deg, #00ffd5, #bf00ff);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                ">{title}</h2>
                {f'<p style="margin: 0.25rem 0 0 0; color: #b8c4d8; font-size: 0.9rem;">{subtitle}</p>' if subtitle else ''}
            </div>
        </div>
        <div style="
            height: 2px;
            background: linear-gradient(90deg, #00ffd5, #bf00ff, transparent);
            margin-top: 1rem;
            border-radius: 1px;
        "></div>
    </div>
    """, unsafe_allow_html=True)


def render_stat_card(
    value: str,
    label: str,
    trend: Optional[str] = None,
    trend_up: bool = True,
    color: str = "primary"
):
    """渲染统计数据卡片"""
    color_map = {
        "primary": "#00ffd5",
        "accent": "#bf00ff",
        "pink": "#ff0080",
        "orange": "#ff6b35",
    }
    accent = color_map.get(color, color_map["primary"])
    trend_color = "#00ff88" if trend_up else "#ff4757"
    trend_icon = "↑" if trend_up else "↓"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(15, 20, 35, 0.8), rgba(30, 36, 57, 0.6));
        border: 1px solid {accent}30;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    " onmouseover="this.style.borderColor='{accent}'; this.style.transform='translateY(-4px)'; this.style.boxShadow='0 12px 40px {accent}20';"
       onmouseout="this.style.borderColor='{accent}30'; this.style.transform='translateY(0)'; this.style.boxShadow='none';">
        <div style="
            font-family: 'Orbitron', monospace;
            font-size: 2.2rem;
            font-weight: 800;
            color: {accent};
            text-shadow: 0 0 20px {accent}60;
            margin-bottom: 0.5rem;
        ">{value}</div>
        <div style="
            color: #d0d7e8;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        ">{label}</div>
        {f'''<div style="
            color: {trend_color};
            font-size: 0.85rem;
            font-weight: 600;
        ">{trend_icon} {trend}</div>''' if trend else ''}
    </div>
    """, unsafe_allow_html=True)


def render_progress_bar(progress: float, label: str = "", color: str = "primary"):
    """渲染进度条"""
    color_map = {
        "primary": ("linear-gradient(90deg, #00ffd5, #00b4d8)", "rgba(0, 255, 213, 0.2)"),
        "accent": ("linear-gradient(90deg, #bf00ff, #ff0080)", "rgba(191, 0, 255, 0.2)"),
    }
    gradient, bg = color_map.get(color, color_map["primary"])
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        {f'<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;"><span style="color: #d0d7e8; font-size: 0.9rem;">{label}</span><span style="color: #00ffd5; font-weight: 600;">{int(progress * 100)}%</span></div>' if label else ''}
        <div style="
            background: {bg};
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
        ">
            <div style="
                width: {progress * 100}%;
                height: 100%;
                background: {gradient};
                border-radius: 10px;
                transition: width 0.5s ease;
                box-shadow: 0 0 15px rgba(0, 255, 213, 0.5);
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status: str, text: str):
    """渲染状态徽章 - 使用Streamlit原生组件确保兼容性"""
    status_colors = {
        "success": "🟢",
        "warning": "🟡", 
        "error": "🔴",
        "info": "🔵",
        "processing": "🟣",
    }
    icon = status_colors.get(status, "🔵")
    st.caption(f"{icon} {text}")


def render_metric_row(metrics: List[Dict[str, Any]]):
    """渲染一行指标卡片"""
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            render_stat_card(
                value=metric.get("value", "0"),
                label=metric.get("label", ""),
                trend=metric.get("trend"),
                trend_up=metric.get("trend_up", True),
                color=metric.get("color", "primary")
            )


def render_tab_selector(options: List[str], key: str = "tab_selector") -> str:
    """渲染自定义标签选择器"""
    st.markdown("""
    <style>
    .tab-container {
        display: flex;
        gap: 0.5rem;
        background: rgba(15, 20, 35, 0.6);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 255, 213, 0.2);
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    return st.radio(
        label="",
        options=options,
        key=key,
        horizontal=True,
        label_visibility="collapsed"
    )


def render_loading_animation(text: str = "处理中..."):
    """渲染加载动画"""
    st.markdown(f"""
    <style>
    @keyframes loadingDots {{
        0%, 20% {{ content: '.'; }}
        40% {{ content: '..'; }}
        60%, 100% {{ content: '...'; }}
    }}
    
    @keyframes spinnerRotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    </style>
    
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        padding: 2rem;
        background: rgba(0, 255, 213, 0.05);
        border: 1px solid rgba(0, 255, 213, 0.2);
        border-radius: 12px;
    ">
        <div style="
            width: 40px;
            height: 40px;
            border: 3px solid rgba(0, 255, 213, 0.2);
            border-top-color: #00ffd5;
            border-radius: 50%;
            animation: spinnerRotate 1s linear infinite;
        "></div>
        <span style="
            color: #00ffd5;
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
        ">{text}</span>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    """渲染页脚"""
    current_year = datetime.now().year
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 3rem 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(0, 255, 213, 0.2);
    ">
        <div style="
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
            color: #b8c4d8;
            letter-spacing: 0.1em;
        ">
            <span style="color: #00ffd5;">🚇</span> 
            客流模型算法测试平台 
            <span style="color: #bf00ff;">|</span> 
            科技赋能智慧交通 
            <span style="color: #ff0080;">|</span> 
            © {current_year}
        </div>
        <div style="
            margin-top: 0.75rem;
            font-size: 0.8rem;
            color: #9aa5bd;
        ">
            Powered by Machine Learning & Deep Learning
        </div>
    </div>
    """, unsafe_allow_html=True)
