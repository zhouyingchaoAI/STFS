# 日预测tab的实现（支持KNN/Prophet/Transformer）

import streamlit as st
import pandas as pd
from datetime import datetime
from config_utils import load_yaml_config, save_yaml_config
from predict_daily import predict_and_plot_timeseries_flow_daily
import os
import matplotlib.pyplot as plt


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

def get_model_versions(model_dir, prefix=""):
    """获取模型目录下所有模型版本（以日期为子目录）"""
    if not os.path.exists(model_dir):
        return []
    dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    versions = [d for d in dirs if len(d) == 8 and d.isdigit()]
    versions.sort(reverse=True)
    return versions

def plot_daily_flow(df_plot, line_name=None, SUBWAY_GREEN="#00e09e", SUBWAY_ACCENT="#00bfff", SUBWAY_CARD="#181d2a", SUBWAY_BG="#10131a", SUBWAY_FONT="#e6f7ff", return_fig=False):
    fig, ax = plt.subplots(figsize=(16, 6))  # 放大图像
    color = SUBWAY_GREEN if line_name is None else SUBWAY_ACCENT
    ax.plot(df_plot["日期"], df_plot["预测客流"], marker='o', color=color, linewidth=2.5, label=line_name)
    if line_name is not None:
        ax.legend(facecolor=SUBWAY_CARD, edgecolor=SUBWAY_GREEN, fontsize=12)
    ax.set_facecolor(SUBWAY_CARD)
    fig.patch.set_facecolor(SUBWAY_BG)
    ax.set_xlabel("日期", color=SUBWAY_GREEN, fontsize=13, fontweight='bold')
    ax.set_ylabel("预测客流", color=SUBWAY_GREEN, fontsize=13, fontweight='bold')
    ax.set_xticks(df_plot["日期"])
    ax.set_xticklabels(df_plot["日期"], rotation=45, ha='right', color=SUBWAY_FONT)
    ax.set_title("日预测客流量" + (f" - {line_name}" if line_name else ""), color=SUBWAY_ACCENT, fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color=SUBWAY_ACCENT)
    ax.tick_params(axis='y', colors=SUBWAY_FONT)
    if return_fig:
        return fig
    else:
        st.pyplot(fig, use_container_width=True)

def daily_tab(SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG, flow_type, metric_type):
    config_daily = load_yaml_config("model_config_daily.yaml", default_daily=True)

    st.markdown(
        f"<h2 style='color:{SUBWAY_ACCENT};font-weight:800;'>📅 {FLOW_OPTIONS[flow_type]} - {FLOW_OPTIONS[metric_type]} 日客流预测（KNN/Prophet/Transformer/xgboost）</h2>",
        unsafe_allow_html=True
    )
    daily_algos = ["knn", "prophet", "transformer", "xgboost", "lstm", "lightgbm"]

    col_train, col_pred = st.columns([1, 1.1], gap="large")

    # 用于存储推理结果的绘图数据
    st.session_state.setdefault("daily_plot_results", None)
    st.session_state.setdefault("daily_plot_figs", None)
    st.session_state["daily_plot_results"] = None
    st.session_state["daily_plot_figs"] = None

    with col_train:
        st.markdown(
            f"<div style='background:{SUBWAY_CARD};border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;box-shadow:0 0 8px {SUBWAY_GREEN}22;'>"
            "<b>日模型训练参数设置</b> <span style='color:#888;'>(可自定义)</span>"
            "</div>",
            unsafe_allow_html=True
        )
        train_daily_algo = st.selectbox("训练日模型算法类型", options=daily_algos, key="daily_train_algo")
        # 合并训练参数到下拉隐藏
        with st.expander("🔧 高级训练参数设置", expanded=False):
            train_params = config_daily.get("train_params", {})
            # lookback_days
            lookback_days = st.number_input("lookback_days", min_value=1, max_value=30,
                                            value=train_params.get("lookback_days", 7) if train_params.get("lookback_days", 7) is not None else 7, key="daily_lookback_days")
            # epochs
            epochs = st.number_input("epochs", min_value=10, max_value=500,
                                     value=train_params.get("epochs", 100) if train_params.get("epochs", 100) is not None else 100, key="daily_epochs")
            # patience
            patience = st.number_input("patience", min_value=1, max_value=50,
                                       value=train_params.get("patience", 10) if train_params.get("patience", 10) is not None else 10, key="daily_patience")
            # learning_rate
            lr_val = train_params.get("learning_rate", 0.001)
            if lr_val is None:
                lr_val = 0.001
            try:
                lr_val = float(lr_val)
            except Exception:
                lr_val = 0.001
            learning_rate = st.number_input("learning_rate", min_value=0.0001, max_value=0.1,
                                            value=lr_val,
                                            step=0.0001, format="%.4f", key="daily_lr")
            # batch_size
            batch_size = st.number_input("batch_size", min_value=8, max_value=256,
                                         value=train_params.get("batch_size", 32) if train_params.get("batch_size", 32) is not None else 32, key="daily_batch_size")
            # 仅KNN参数
            n_neighbors = None
            if train_daily_algo == "knn":
                n_neighbors_val = train_params.get("n_neighbors", 5)
                if n_neighbors_val is None:
                    n_neighbors_val = 5
                n_neighbors = st.number_input("KNN邻居数(n_neighbors)", min_value=1, max_value=30,
                                              value=n_neighbors_val, key="daily_n_neighbors")
            # Transformer参数
            d_model = None
            nhead = None
            num_layers = None
            if train_daily_algo == "transformer":
                d_model_val = train_params.get("d_model", 64)
                if d_model_val is None:
                    d_model_val = 64
                d_model = st.number_input("Transformer隐藏维度(d_model)", min_value=8, max_value=512, value=d_model_val, key="daily_d_model")
                nhead_val = train_params.get("nhead", 4)
                if nhead_val is None:
                    nhead_val = 4
                nhead = st.number_input("Transformer头数(nhead)", min_value=1, max_value=16, value=nhead_val, key="daily_nhead")
                num_layers_val = train_params.get("num_layers", 2)
                if num_layers_val is None:
                    num_layers_val = 2
                num_layers = st.number_input("Transformer层数(num_layers)", min_value=1, max_value=8, value=num_layers_val, key="daily_num_layers")
        # 更新config
        config_daily["train_params"] = {
            "lookback_days": lookback_days,
            "epochs": epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        if train_daily_algo == "knn":
            config_daily["train_params"]["n_neighbors"] = n_neighbors
        if train_daily_algo == "transformer":
            config_daily["train_params"]["d_model"] = d_model
            config_daily["train_params"]["nhead"] = nhead
            config_daily["train_params"]["num_layers"] = num_layers
        save_yaml_config(config_daily, "model_config_daily.yaml")

        train_start_date = st.date_input("训练数据截止日期", value=datetime(2025, 4, 26), key="hourly_train_date")
        # train_start_date = st.date_input("训练数据截止日期（日）", value=datetime.now(), key="daily_train_date")
        train_start_date_str = train_start_date.strftime("%Y%m%d")
        retrain_daily = st.checkbox("强制重新训练日模型", value=True, key="daily_retrain")

        if st.button("🚆 开始日模型训练", key="run_daily_train"):
            with st.spinner("正在训练日模型..."):
                # 修改：模型保存路径加上xianwangxianlu
                model_save_dir_daily = os.path.join("models", flow_type, "daily", metric_type, train_start_date_str, train_daily_algo)
                os.makedirs(model_save_dir_daily, exist_ok=True)
                result = predict_and_plot_timeseries_flow_daily(
                    file_path=None,
                    predict_start_date=train_start_date_str,
                    algorithm=train_daily_algo,
                    retrain=retrain_daily,
                    save_path="timeseries_predict_daily.png",
                    mode="train",
                    days=15,
                    config=config_daily,
                    model_version=None,
                    model_save_dir=model_save_dir_daily,
                    flow_type = flow_type, 
                    metric_type = metric_type
                )
            if isinstance(result, dict) and "error" in result:
                st.error(result["error"])
            else:
                st.success("✅ 日模型训练完成！")
                st.info("请在右侧推理模块选择模型版本和算法类型进行预测。")

    with col_pred:
        st.markdown(
            f"<div style='background:{SUBWAY_CARD};border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;box-shadow:0 0 8px {SUBWAY_ACCENT}22;'>"
            "<b>日客流推理预测</b>"
            "</div>",
            unsafe_allow_html=True
        )
        # 修改：模型根目录加上xianwangxianlu
        daily_model_root = os.path.join("models", flow_type, "daily", metric_type)
        daily_versions = get_model_versions(daily_model_root)
        if daily_versions:
            daily_version = st.selectbox("选择日模型日期版本", options=daily_versions, key="daily_model_version")
            algo_dir_daily = os.path.join(daily_model_root, daily_version)
            available_daily_algos = [d for d in os.listdir(algo_dir_daily) if os.path.isdir(os.path.join(algo_dir_daily, d))]
            available_daily_algos = [a for a in available_daily_algos if a in daily_algos]
            if available_daily_algos:
                predict_daily_algo = st.selectbox("推理模型类型", options=available_daily_algos, key="daily_predict_model_type")
                model_dir_daily = os.path.join(daily_model_root, daily_version, predict_daily_algo)
            else:
                model_dir_daily = None
                predict_daily_algo = None
                st.warning("该日期下未找到可用的日算法模型，请先训练模型。")
        else:
            daily_version = None
            model_dir_daily = None
            predict_daily_algo = None
            st.warning("未找到可用的日模型版本，请先训练模型。")

        if daily_version is None:
            default_date = datetime.now()
        else:
            default_date = datetime.strptime(daily_version, "%Y%m%d") + pd.Timedelta(days=1)
        predict_start_date = st.date_input("预测起始日期", value=default_date, key="daily_predict_date")
        # predict_start_date = st.date_input("预测起始日期", value=datetime.now(), key="daily_predict_date")
        predict_start_date_str = predict_start_date.strftime("%Y%m%d")
        days = st.number_input("预测天数", min_value=1, max_value=30, value=15, key="daily_days")

        if st.button("🚇 开始日推理预测", key="run_daily_predict"):
            if not daily_version or not model_dir_daily or not os.path.exists(model_dir_daily) or not predict_daily_algo:
                st.error("请先训练并选择模型版本和算法类型。")
            else:
                with st.spinner("正在进行日推理预测..."):
                    result = predict_and_plot_timeseries_flow_daily(
                        file_path=None,
                        predict_start_date=predict_start_date_str,
                        algorithm=predict_daily_algo,
                        retrain=False,
                        save_path="timeseries_predict_daily.png",
                        mode="predict",
                        days=days,
                        config=config_daily,
                        model_version=None,
                        model_save_dir=model_dir_daily,
                        flow_type = flow_type, 
                        metric_type = metric_type
                    )
                if isinstance(result, dict) and "error" in result:
                    st.error(result["error"])
                else:
                    st.success("✅ 日推理预测完成！")
                    info = result if isinstance(result, dict) else {}
                    st.markdown(
                        f"<h4 style='color:{SUBWAY_ACCENT};font-weight:700;'>日预测结果</h4>",
                        unsafe_allow_html=True
                    )
                    if info.get("error"):
                        st.warning(f"预测错误: {info['error']}")
                    else:
                        daily_flow = info.get("predict_daily_flow", {})
                        figs = []
                        plot_results = []
                        if isinstance(daily_flow, dict) and all(isinstance(v, dict) for v in daily_flow.values()):
                            for line_name, line_daily in daily_flow.items():
                                dates = sorted(line_daily.keys())
                                flows = []
                                for date in dates:
                                    v = line_daily.get(date)
                                    if v is None:
                                        v = 0
                                    flows.append(v)
                                df_plot = pd.DataFrame({
                                    "日期": dates,
                                    "预测客流": flows
                                })
                                plot_results.append({
                                    "line_name": line_name,
                                    "df_plot": df_plot,
                                    "predict_start_date": info.get('predict_start_date', predict_start_date_str)
                                })
                                # 获取fig对象
                                fig = plot_daily_flow(df_plot, line_name=line_name, SUBWAY_GREEN=SUBWAY_GREEN, SUBWAY_ACCENT=SUBWAY_ACCENT, SUBWAY_CARD=SUBWAY_CARD, SUBWAY_BG=SUBWAY_BG, SUBWAY_FONT=SUBWAY_FONT, return_fig=True)
                                figs.append((fig, line_name))
                        else:
                            dates = sorted(daily_flow.keys())
                            flows = []
                            for date in dates:
                                v = daily_flow.get(date)
                                if v is None:
                                    v = 0
                                flows.append(v)
                            df_plot = pd.DataFrame({
                                "日期": dates,
                                "预测客流": flows
                            })
                            plot_results.append({
                                "line_name": None,
                                "df_plot": df_plot,
                                "predict_start_date": info.get('predict_start_date', predict_start_date_str)
                            })
                            fig = plot_daily_flow(df_plot, SUBWAY_GREEN=SUBWAY_GREEN, SUBWAY_ACCENT=SUBWAY_ACCENT, SUBWAY_CARD=SUBWAY_CARD, SUBWAY_BG=SUBWAY_BG, SUBWAY_FONT=SUBWAY_FONT, return_fig=True)
                            figs.append((fig, None))
                        # 存储到session_state，供下方全宽显示
                        st.session_state["daily_plot_results"] = plot_results
                        st.session_state["daily_plot_figs"] = figs


    if os.path.exists("timeseries_predict_daily.png"):
        st.image("timeseries_predict_daily.png", caption="日预测结果可视化", use_container_width=True)