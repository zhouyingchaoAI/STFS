# streamlit_hourly.py
# 小时预测tab的实现（支持KNN/LSTM/Prophet/XGBoost）

import streamlit as st
import pandas as pd
from datetime import datetime
from config_utils import load_yaml_config, save_yaml_config
from predict_hourly import predict_and_plot_timeseries_flow
import os
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*missing from current font")

# 导入UI组件
from ui_components import (
    render_section_header,
)

# ==================== 配置常量 ====================

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
    "lstm": "🧠 深度学习算法",
    "prophet": "📈 时序预测算法",
    "xgboost": "📊 传统机器学习"
}

ALGO_REAL_MAP = {v: k for k, v in ALGO_DISPLAY_MAP.items()}


def get_model_versions(model_dir, prefix=""):
    """获取模型目录下所有模型版本"""
    if not os.path.exists(model_dir):
        return []
    dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    versions = [d for d in dirs if len(d) == 8 and d.isdigit()]
    versions.sort(reverse=True)
    return versions


def setup_plot_style():
    """设置图表样式"""
    plt.style.use('dark_background')
    matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False


def plot_hourly_flow(
    df_plot, 
    line_name=None, 
    predict_date=None,
    SUBWAY_GREEN="#00ffd5", 
    SUBWAY_ACCENT="#bf00ff", 
    SUBWAY_CARD="#1e2439", 
    SUBWAY_BG="#0f1423", 
    SUBWAY_FONT="#ffffff", 
    return_fig=False
):
    """绘制小时客流预测图 - 赛博朋克风格"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # 设置背景
    fig.patch.set_facecolor(SUBWAY_BG)
    ax.set_facecolor(SUBWAY_CARD)
    
    # 颜色选择
    color = SUBWAY_GREEN if line_name is None else SUBWAY_ACCENT
    
    # 发光效果 - 多层叠加
    for alpha, lw in [(0.1, 8), (0.2, 5), (0.4, 3)]:
        ax.plot(df_plot["小时"], df_plot["预测客流"], 
                color=color, linewidth=lw, alpha=alpha)
    
    # 主线
    ax.plot(df_plot["小时"], df_plot["预测客流"], 
            marker='o', color=color, linewidth=2.5,
            markersize=8, markerfacecolor=SUBWAY_BG,
            markeredgecolor=color, markeredgewidth=2,
            label=line_name, zorder=10)
    
    # 填充区域
    ax.fill_between(df_plot["小时"], df_plot["预测客流"], 
                    alpha=0.15, color=color)
    
    # 高峰时段标注
    for i, (x, y) in enumerate(zip(df_plot["小时"], df_plot["预测客流"])):
        # 早高峰 7-9 和晚高峰 17-19
        if x in [7, 8, 9, 17, 18, 19]:
            ax.annotate(f'{y:,.0f}', (x, y), 
                       textcoords="offset points", xytext=(0, 12),
                       ha='center', fontsize=9, color="#ff0080",
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=SUBWAY_CARD, 
                                edgecolor="#ff0080", alpha=0.9))
    
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
    ax.set_xlabel("小时", color=SUBWAY_GREEN, fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel("预测客流量", color=SUBWAY_GREEN, fontsize=13, fontweight='bold', labelpad=10)
    
    # X轴刻度
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], 
                       rotation=45, ha='right', color=SUBWAY_FONT, fontsize=9)
    
    # 标题
    title = "⏰ 小时预测客流量"
    if predict_date:
        title += f" · {predict_date}"
    if line_name:
        title += f" · {line_name}"
    ax.set_title(title, color=SUBWAY_ACCENT, fontsize=18, fontweight='bold', pad=20)
    
    # 网格线
    ax.grid(True, linestyle='--', alpha=0.2, color=SUBWAY_ACCENT)
    ax.tick_params(axis='y', colors=SUBWAY_FONT, labelsize=10)
    
    # 边框样式
    for spine in ax.spines.values():
        spine.set_color(SUBWAY_GREEN)
        spine.set_alpha(0.3)
    
    # 添加高峰时段背景标注
    ax.axvspan(7, 9, alpha=0.05, color="#ff0080", label='早高峰')
    ax.axvspan(17, 19, alpha=0.05, color="#ff6b35", label='晚高峰')
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        st.pyplot(fig)
        plt.close(fig)


def render_training_panel(
    SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG,
    flow_type, metric_type, config_hourly
):
    """渲染训练面板"""
    hourly_algos = list(ALGO_DISPLAY_MAP.keys())
    hourly_algo_display_names = list(ALGO_DISPLAY_MAP.values())
    
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
                <p style="margin: 0.25rem 0 0 0; color: #b8c4d8; font-size: 0.85rem;">配置参数并训练小时预测模型</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 算法选择
    train_hourly_algo_display = st.selectbox(
        "🧪 选择训练算法",
        options=hourly_algo_display_names,
        key="hourly_train_algo",
        help="选择用于训练的机器学习/深度学习算法"
    )
    train_hourly_algo = ALGO_REAL_MAP.get(train_hourly_algo_display, train_hourly_algo_display)
    
    # 高级参数设置
    with st.expander("⚙️ 高级训练参数", expanded=False):
        train_params = config_hourly.get("train_params", {})
        
        # 通用参数
        lookback_hours = st.number_input(
            "回溯小时数", min_value=24, max_value=168,
            value=train_params.get("lookback_hours", 72) or 72,
            key="hourly_lookback_hours",
            help="用于预测的历史数据小时数"
        )
        
        # 算法特定参数
        epochs = None
        patience = None
        learning_rate = None
        batch_size = None
        hidden_size = None
        num_layers = None
        max_depth = None
        n_estimators = None
        seasonality_mode = None
        yearly_seasonality = None
        weekly_seasonality = None
        daily_seasonality = None
        
        if train_hourly_algo == 'lstm':
            st.markdown("##### LSTM 参数")
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                epochs = st.number_input(
                    "训练轮次", min_value=10, max_value=500,
                    value=train_params.get("epochs", 100) or 100,
                    key="hourly_epochs"
                )
                patience = st.number_input(
                    "早停耐心值", min_value=1, max_value=50,
                    value=train_params.get("patience", 10) or 10,
                    key="hourly_patience"
                )
                lr_val = train_params.get("learning_rate", 0.001) or 0.001
                learning_rate = st.number_input(
                    "学习率", min_value=0.0001, max_value=0.1,
                    value=float(lr_val), step=0.0001, format="%.4f",
                    key="hourly_lr"
                )
            with col_l2:
                batch_size = st.number_input(
                    "批次大小", min_value=8, max_value=256,
                    value=train_params.get("batch_size", 32) or 32,
                    key="hourly_batch_size"
                )
                hidden_size = st.number_input(
                    "隐藏层大小", min_value=16, max_value=256,
                    value=train_params.get("hidden_size", 64) or 64,
                    key="hourly_hidden_size"
                )
                num_layers = st.number_input(
                    "LSTM层数", min_value=1, max_value=8,
                    value=train_params.get("num_layers", 2) or 2,
                    key="hourly_num_layers"
                )
        
        elif train_hourly_algo == 'xgboost':
            st.markdown("##### XGBoost 参数")
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                max_depth = st.number_input(
                    "最大深度", min_value=1, max_value=20,
                    value=train_params.get("max_depth", 6) or 6,
                    key="hourly_max_depth"
                )
                n_estimators = st.number_input(
                    "估计器数量", min_value=10, max_value=1000,
                    value=train_params.get("n_estimators", 100) or 100,
                    key="hourly_n_estimators"
                )
            with col_x2:
                lr_val = train_params.get("learning_rate", 0.1) or 0.1
                learning_rate = st.number_input(
                    "学习率", min_value=0.01, max_value=1.0,
                    value=float(lr_val), step=0.01, format="%.2f",
                    key="hourly_xgb_lr"
                )
        
        elif train_hourly_algo == 'prophet':
            st.markdown("##### Prophet 参数")
            seasonality_mode = st.selectbox(
                "季节性模式",
                options=['additive', 'multiplicative'],
                index=0 if train_params.get("seasonality_mode", 'additive') == 'additive' else 1,
                key="hourly_seasonality_mode"
            )
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                yearly_seasonality = st.checkbox(
                    "年季节性",
                    value=train_params.get("yearly_seasonality", True),
                    key="hourly_yearly_seasonality"
                )
            with col_p2:
                weekly_seasonality = st.checkbox(
                    "周季节性",
                    value=train_params.get("weekly_seasonality", True),
                    key="hourly_weekly_seasonality"
                )
            with col_p3:
                daily_seasonality = st.checkbox(
                    "日季节性",
                    value=train_params.get("daily_seasonality", True),
                    key="hourly_daily_seasonality"
                )
    
    # 更新配置
    config_hourly["train_params"] = {"lookback_hours": lookback_hours}
    
    if train_hourly_algo == 'lstm':
        config_hourly["train_params"].update({
            "epochs": epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers
        })
    elif train_hourly_algo == 'xgboost':
        config_hourly["train_params"].update({
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate
        })
    elif train_hourly_algo == 'prophet':
        config_hourly["train_params"].update({
            "seasonality_mode": seasonality_mode,
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality
        })
    
    save_yaml_config(config_hourly, "model_config_hourly.yaml")
    
    # 训练设置
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    train_date = st.date_input(
        "📅 训练数据截止日期",
        value=datetime(2025, 4, 26),
        key="hourly_train_date",
        help="模型将使用此日期之前的数据进行训练"
    )
    train_date_str = train_date.strftime("%Y%m%d")
    
    retrain_hourly = st.checkbox(
        "🔄 强制重新训练",
        value=True,
        key="hourly_retrain",
        help="勾选后将重新训练模型，否则加载已有模型"
    )
    
    # 训练按钮
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    if st.button("🚀 开始训练", key="run_hourly_train"):
        with st.spinner(""):
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
                <span style="color: #00ffd5; font-weight: 600;">正在训练小时模型...</span>
            </div>
            <style>
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            </style>
            """, unsafe_allow_html=True)
            
            model_save_dir_hourly = os.path.join(
                "models", flow_type, "hourly", metric_type,
                train_date_str, train_hourly_algo
            )
            os.makedirs(model_save_dir_hourly, exist_ok=True)
            
            result = predict_and_plot_timeseries_flow(
                file_path="",
                predict_date=train_date_str,
                algorithm=train_hourly_algo,
                retrain=retrain_hourly,
                save_path="timeseries_predict_hourly.png",
                mode="train",
                config=config_hourly,
                model_version=None,
                model_save_dir=model_save_dir_hourly,
                flow_type=flow_type,
                metric_type=metric_type
            )
            
            status_placeholder.empty()
        
        if isinstance(result, dict) and "error" in result:
            st.error(f"❌ 训练失败: {result['error']}")
        else:
            st.success("✅ 小时模型训练完成！")
            st.info("💡 请在右侧推理模块选择模型版本进行预测")
    
    return train_hourly_algo, config_hourly


def render_prediction_panel(
    SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG,
    flow_type, metric_type, config_hourly
):
    """渲染预测面板"""
    hourly_algos = list(ALGO_DISPLAY_MAP.keys())
    
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
                <p style="margin: 0.25rem 0 0 0; color: #b8c4d8; font-size: 0.85rem;">使用训练好的模型进行小时级预测</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 模型版本选择
    hourly_model_root = os.path.join("models", flow_type, "hourly", metric_type)
    hourly_versions = get_model_versions(hourly_model_root)
    
    if hourly_versions:
        hourly_version = st.selectbox(
            "📁 模型版本",
            options=hourly_versions,
            key="hourly_model_version",
            help="选择要使用的模型版本（按日期排序）"
        )
        
        algo_dir_hourly = os.path.join(hourly_model_root, hourly_version)
        available_hourly_algos = [
            d for d in os.listdir(algo_dir_hourly)
            if os.path.isdir(os.path.join(algo_dir_hourly, d)) and d in hourly_algos
        ]
        
        if available_hourly_algos:
            available_hourly_algos_display = [
                ALGO_DISPLAY_MAP.get(a, a) for a in available_hourly_algos
            ]
            predict_hourly_algo_display = st.selectbox(
                "🧪 推理算法",
                options=available_hourly_algos_display,
                key="hourly_predict_model_type"
            )
            predict_hourly_algo = ALGO_REAL_MAP.get(
                predict_hourly_algo_display, predict_hourly_algo_display
            )
            model_dir_hourly = os.path.join(hourly_model_root, hourly_version, predict_hourly_algo)
        else:
            model_dir_hourly = None
            predict_hourly_algo = None
            st.warning("⚠️ 该版本下未找到可用的算法模型")
    else:
        hourly_version = None
        model_dir_hourly = None
        predict_hourly_algo = None
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
    if hourly_version is None:
        default_date = datetime.now()
    else:
        default_date = datetime.strptime(hourly_version, "%Y%m%d") + pd.Timedelta(days=1)
    
    predict_date = st.date_input(
        "📅 预测日期",
        value=default_date,
        key="hourly_predict_date",
        help="选择要预测的日期"
    )
    predict_date_str = predict_date.strftime("%Y%m%d")
    
    # 预测按钮
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    if st.button("🎯 开始预测", key="run_hourly_predict"):
        if not hourly_version or not model_dir_hourly or not os.path.exists(model_dir_hourly) or not predict_hourly_algo:
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
                    <span style="color: #bf00ff; font-weight: 600;">正在进行小时预测...</span>
                </div>
                """, unsafe_allow_html=True)
                
                result = predict_and_plot_timeseries_flow(
                    file_path="",
                    predict_date=predict_date_str,
                    algorithm=predict_hourly_algo,
                    retrain=False,
                    save_path="timeseries_predict_hourly.png",
                    mode="predict",
                    config=config_hourly,
                    model_version=None,
                    model_save_dir=model_dir_hourly,
                    flow_type=flow_type,
                    metric_type=metric_type
                )
                
                status_placeholder.empty()
            
            if isinstance(result, dict) and "error" in result:
                st.error(f"❌ 预测失败: {result['error']}")
            else:
                st.success("✅ 小时预测完成！")
                
                # 显示结果
                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
                render_section_header("预测结果", f"{predict_date_str} 24小时客流预测", "📊")
                
                figs = []
                plot_results = []
                
                if isinstance(result, dict):
                    for line_no, line_result in result.items():
                        if isinstance(line_result, dict) and "predict_hourly_flow" in line_result:
                            hourly_flow = line_result.get("predict_hourly_flow", {})
                            error_msg = line_result.get("error")
                            
                            if error_msg:
                                st.warning(f"⚠️ 线路 {line_no}: {error_msg}")
                                continue
                            
                            # 准备绘图数据
                            hours = [int(h) for h in sorted(hourly_flow.keys())]
                            flows = [hourly_flow.get(f"{h:02d}", 0) for h in hours]
                            
                            df_plot = pd.DataFrame({
                                "小时": hours,
                                "预测客流": flows
                            })
                            
                            # 获取线路名称
                            line_name = f"线路{line_no}"
                            if "line_data" in line_result:
                                line_data = line_result["line_data"]
                                if not line_data.empty and "F_LINENAME" in line_data.columns:
                                    line_name = line_data["F_LINENAME"].iloc[0]
                            
                            plot_results.append({
                                "line_name": line_name,
                                "line_no": line_no,
                                "df_plot": df_plot,
                                "predict_date": predict_date_str
                            })
                            
                            fig = plot_hourly_flow(
                                df_plot,
                                line_name=line_name,
                                predict_date=predict_date_str,
                                SUBWAY_GREEN=SUBWAY_GREEN,
                                SUBWAY_ACCENT=SUBWAY_ACCENT,
                                SUBWAY_CARD=SUBWAY_CARD,
                                SUBWAY_BG=SUBWAY_BG,
                                SUBWAY_FONT=SUBWAY_FONT,
                                return_fig=True
                            )
                            figs.append((fig, line_name))
                
                st.session_state["hourly_plot_results"] = plot_results
                st.session_state["hourly_plot_figs"] = figs
    
    return hourly_version, model_dir_hourly


def hourly_tab(SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG, flow_type, metric_type):
    """小时预测tab主函数"""
    config_hourly = load_yaml_config("model_config_hourly.yaml", default_daily=False)
    
    # 页面标题
    render_section_header(
        f"{FLOW_OPTIONS[flow_type]} · {FLOW_OPTIONS[metric_type]} 小时预测",
        "支持 KNN / LSTM / Prophet / XGBoost 算法",
        "🕐"
    )
    
    # 初始化session state
    st.session_state.setdefault("hourly_plot_results", None)
    st.session_state.setdefault("hourly_plot_figs", None)
    
    # 双栏布局
    col_train, col_pred = st.columns([1, 1.1], gap="large")
    
    with col_train:
        render_training_panel(
            SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG,
            flow_type, metric_type, config_hourly
        )
    
    with col_pred:
        render_prediction_panel(
            SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG,
            flow_type, metric_type, config_hourly
        )
    
    # 显示图表结果
    if st.session_state.get("hourly_plot_figs"):
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        
        for idx, (fig, line_name) in enumerate(st.session_state["hourly_plot_figs"]):
            if line_name:
                st.markdown(f"""
                <div style="
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    background: rgba(191, 0, 255, 0.1);
                    border: 1px solid rgba(191, 0, 255, 0.3);
                    border-radius: 20px;
                    padding: 0.4rem 1rem;
                    margin-bottom: 1rem;
                ">
                    <span style="font-size: 1rem;">🚇</span>
                    <span style="color: #bf00ff; font-weight: 600;">{line_name}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.pyplot(fig)
            plt.close(fig)
        
        # 显示数据表格
        if st.session_state.get("hourly_plot_results"):
            with st.expander("📋 查看详细数据", expanded=False):
                for plot_info in st.session_state["hourly_plot_results"]:
                    df_display = plot_info["df_plot"].copy()
                    df_display["时段"] = df_display["小时"].apply(
                        lambda x: f"{x:02d}:00-{(x+1) % 24:02d}:00"
                    )
                    df_display = df_display[["时段", "预测客流"]]
                    df_display["预测客流"] = df_display["预测客流"].apply(lambda x: f"{x:,.0f}")
                    
                    st.markdown(f"**{plot_info['line_name']}** ({plot_info['predict_date']})")
                    st.dataframe(df_display, hide_index=True)
    
    # 兼容旧图片显示
    if os.path.exists("timeseries_predict_hourly.png"):
        st.image("timeseries_predict_hourly.png", caption="小时预测结果可视化")
