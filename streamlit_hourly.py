# 小时预测tab的实现（支持LSTM/Prophet/XGBoost）- 参考日预测tab风格

import streamlit as st
import pandas as pd
from datetime import datetime
from config_utils import load_yaml_config, save_yaml_config
from predict_hourly import predict_and_plot_timeseries_flow  # 引用你的主流程模块
import os
import matplotlib.pyplot as plt

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

def plot_hourly_flow(df_plot, line_name=None, predict_date=None, SUBWAY_GREEN="#00e09e", SUBWAY_ACCENT="#00bfff", SUBWAY_CARD="#181d2a", SUBWAY_BG="#10131a", SUBWAY_FONT="#e6f7ff", return_fig=False):
    """绘制小时客流预测图"""
    fig, ax = plt.subplots(figsize=(16, 6))  # 放大图像
    color = SUBWAY_GREEN if line_name is None else SUBWAY_ACCENT
    ax.plot(df_plot["小时"], df_plot["预测客流"], marker='o', color=color, linewidth=2.5, label=line_name)
    if line_name is not None:
        ax.legend(facecolor=SUBWAY_CARD, edgecolor=SUBWAY_GREEN, fontsize=12)
    ax.set_facecolor(SUBWAY_CARD)
    fig.patch.set_facecolor(SUBWAY_BG)
    ax.set_xlabel("小时", color=SUBWAY_GREEN, fontsize=13, fontweight='bold')
    ax.set_ylabel("预测客流", color=SUBWAY_GREEN, fontsize=13, fontweight='bold')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha='right', color=SUBWAY_FONT)
    title = f"小时预测客流量"
    if predict_date:
        title += f" - {predict_date}"
    if line_name:
        title += f" - {line_name}"
    ax.set_title(title, color=SUBWAY_ACCENT, fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color=SUBWAY_ACCENT)
    ax.tick_params(axis='y', colors=SUBWAY_FONT)
    plt.tight_layout()
    if return_fig:
        return fig
    else:
        st.pyplot(fig, use_container_width=True)

def hourly_tab(SUBWAY_GREEN, SUBWAY_ACCENT, SUBWAY_CARD, SUBWAY_FONT, SUBWAY_BG, flow_type, metric_type):
    """小时预测tab主函数"""
    config_hourly = load_yaml_config("model_config_hourly.yaml", default_daily=False)

    st.markdown(
        f"<h2 style='color:{SUBWAY_ACCENT};font-weight:800;'>⏰ {FLOW_OPTIONS[flow_type]} - {FLOW_OPTIONS[metric_type]} 小时客流预测（LSTM/Prophet/XGBoost）</h2>",
        unsafe_allow_html=True
    )

    hourly_algos = ["knn", "lstm", "prophet", "xgboost"]

    col_train, col_pred = st.columns([1, 1.1], gap="large")

    # 用于存储推理结果的绘图数据
    st.session_state.setdefault("hourly_plot_results", None)
    st.session_state.setdefault("hourly_plot_figs", None)
    st.session_state["hourly_plot_results"] = None
    st.session_state["hourly_plot_figs"] = None

    with col_train:
        st.markdown(
            f"<div style='background:{SUBWAY_CARD};border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;box-shadow:0 0 8px {SUBWAY_GREEN}22;'>"
            "<b>小时模型训练参数设置</b> <span style='color:#888;'>(可自定义)</span>"
            "</div>",
            unsafe_allow_html=True
        )
        train_hourly_algo = st.selectbox("训练小时模型算法类型", options=hourly_algos, key="hourly_train_algo")
        
        # 合并训练参数到下拉隐藏
        with st.expander("🔧 高级训练参数设置", expanded=False):
            train_params = config_hourly.get("train_params", {})
            
            # 通用参数
            lookback_hours = st.number_input("lookback_hours", min_value=24, max_value=168,
                                            value=train_params.get("lookback_hours", 72) if train_params.get("lookback_hours", 72) is not None else 72, 
                                            key="hourly_lookback_hours",
                                            help="用于预测的历史小时数")
            
            if train_hourly_algo == 'lstm':
                epochs = st.number_input("epochs", min_value=10, max_value=500,
                                        value=train_params.get("epochs", 100) if train_params.get("epochs", 100) is not None else 100, 
                                        key="hourly_epochs")
                patience = st.number_input("patience", min_value=1, max_value=50,
                                          value=train_params.get("patience", 10) if train_params.get("patience", 10) is not None else 10, 
                                          key="hourly_patience")
                lr_val = train_params.get("learning_rate", 0.001)
                if lr_val is None:
                    lr_val = 0.001
                try:
                    lr_val = float(lr_val)
                except Exception:
                    lr_val = 0.001
                learning_rate = st.number_input("learning_rate", min_value=0.0001, max_value=0.1,
                                               value=lr_val,
                                               step=0.0001, format="%.4f", key="hourly_lr")
                batch_size = st.number_input("batch_size", min_value=8, max_value=256,
                                            value=train_params.get("batch_size", 32) if train_params.get("batch_size", 32) is not None else 32, 
                                            key="hourly_batch_size")
                hidden_size = st.number_input("hidden_size", min_value=16, max_value=256,
                                             value=train_params.get("hidden_size", 64) if train_params.get("hidden_size", 64) is not None else 64,
                                             key="hourly_hidden_size")
                num_layers = st.number_input("num_layers", min_value=1, max_value=8,
                                            value=train_params.get("num_layers", 2) if train_params.get("num_layers", 2) is not None else 2,
                                            key="hourly_num_layers")
                
            elif train_hourly_algo == 'xgboost':
                max_depth = st.number_input("max_depth", min_value=1, max_value=20,
                                           value=train_params.get("max_depth", 6) if train_params.get("max_depth", 6) is not None else 6,
                                           key="hourly_max_depth")
                n_estimators = st.number_input("n_estimators", min_value=10, max_value=1000,
                                              value=train_params.get("n_estimators", 100) if train_params.get("n_estimators", 100) is not None else 100,
                                              key="hourly_n_estimators")
                learning_rate = st.number_input("learning_rate", min_value=0.01, max_value=1.0,
                                               value=train_params.get("learning_rate", 0.1) if train_params.get("learning_rate", 0.1) is not None else 0.1,
                                               step=0.01, format="%.2f", key="hourly_xgb_lr")
                
            elif train_hourly_algo == 'prophet':
                seasonality_mode = st.selectbox("seasonality_mode", 
                                               options=['additive', 'multiplicative'],
                                               index=0 if train_params.get("seasonality_mode", 'additive') == 'additive' else 1,
                                               key="hourly_seasonality_mode")
                yearly_seasonality = st.checkbox("yearly_seasonality", 
                                                 value=train_params.get("yearly_seasonality", True),
                                                 key="hourly_yearly_seasonality")
                weekly_seasonality = st.checkbox("weekly_seasonality",
                                                value=train_params.get("weekly_seasonality", True), 
                                                key="hourly_weekly_seasonality")
                daily_seasonality = st.checkbox("daily_seasonality",
                                               value=train_params.get("daily_seasonality", True),
                                               key="hourly_daily_seasonality")
        
        # 更新config
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
        train_date = st.date_input("训练数据截止日期", value=datetime(2025, 4, 26), key="hourly_train_date")
        # train_date = st.date_input("训练数据截止日期", value=datetime.now(), key="hourly_train_date")
        train_date_str = train_date.strftime("%Y%m%d")
        retrain_hourly = st.checkbox("强制重新训练小时模型", value=True, key="hourly_retrain")

        if st.button("🚆 开始小时模型训练", key="run_hourly_train"):
            with st.spinner("正在训练小时模型..."):
                # 修改：模型保存目录加上xianwangxianlu
                model_save_dir_hourly = os.path.join("models", flow_type, "hourly", metric_type, train_date_str, train_hourly_algo)
                os.makedirs(model_save_dir_hourly, exist_ok=True)
                result = predict_and_plot_timeseries_flow(
                    file_path=None,
                    predict_date=train_date_str,
                    algorithm=train_hourly_algo,
                    retrain=retrain_hourly,
                    save_path="timeseries_predict_hourly.png",
                    mode="train",
                    config=config_hourly,
                    model_version=None,
                    model_save_dir=model_save_dir_hourly,
                    flow_type = flow_type, 
                    metric_type = metric_type
                )
            if isinstance(result, dict) and "error" in result:
                st.error(result["error"])
            else:
                st.success("✅ 小时模型训练完成！")
                st.info("请在右侧推理模块选择模型版本和算法类型进行预测。")

    with col_pred:
        st.markdown(
            f"<div style='background:{SUBWAY_CARD};border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;box-shadow:0 0 8px {SUBWAY_ACCENT}22;'>"
            "<b>小时客流推理预测</b>"
            "</div>",
            unsafe_allow_html=True
        )
        
        # 修改：模型根目录加上xianwangxianlu
        hourly_model_root = os.path.join("models", flow_type, "hourly", metric_type)
        hourly_versions = get_model_versions(hourly_model_root)
        
        if hourly_versions:
            hourly_version = st.selectbox("选择小时模型日期版本", options=hourly_versions, key="hourly_model_version")
            algo_dir_hourly = os.path.join(hourly_model_root, hourly_version)
            available_hourly_algos = [d for d in os.listdir(algo_dir_hourly) if os.path.isdir(os.path.join(algo_dir_hourly, d))]
            available_hourly_algos = [a for a in available_hourly_algos if a in hourly_algos]
            
            if available_hourly_algos:
                predict_hourly_algo = st.selectbox("推理模型类型", options=available_hourly_algos, key="hourly_predict_model_type")
                model_dir_hourly = os.path.join(hourly_model_root, hourly_version, predict_hourly_algo)
            else:
                model_dir_hourly = None
                predict_hourly_algo = None
                st.warning("该日期下未找到可用的小时算法模型，请先训练模型。")
        else:
            hourly_version = None
            model_dir_hourly = None
            predict_hourly_algo = None
            st.warning("未找到可用的小时模型版本，请先训练模型。")

        if hourly_version is None:
            default_date = datetime.now()
        else:
            default_date = datetime.strptime(hourly_version, "%Y%m%d") + pd.Timedelta(days=1)

        predict_date = st.date_input("预测日期", value=default_date, key="hourly_predict_date")
        predict_date_str = predict_date.strftime("%Y%m%d")

        if st.button("🚇 开始小时推理预测", key="run_hourly_predict"):
            if not hourly_version or not model_dir_hourly or not os.path.exists(model_dir_hourly) or not predict_hourly_algo:
                st.error("请先训练并选择模型版本和算法类型。")
            else:
                with st.spinner("正在进行小时推理预测..."):
                    result = predict_and_plot_timeseries_flow(
                        file_path=None,
                        predict_date=predict_date_str,
                        algorithm=predict_hourly_algo,
                        retrain=False,
                        save_path="timeseries_predict_hourly.png",
                        mode="predict",
                        config=config_hourly,
                        model_version=None,
                        model_save_dir=model_dir_hourly,
                        flow_type = flow_type, 
                        metric_type = metric_type
                    )
                
                if isinstance(result, dict) and "error" in result:
                    st.error(result["error"])
                else:
                    st.success("✅ 小时推理预测完成！")
                    st.markdown(
                        f"<h4 style='color:{SUBWAY_ACCENT};font-weight:700;'>小时预测结果</h4>",
                        unsafe_allow_html=True
                    )
                    
                    # 解析结果并准备绘图数据
                    figs = []
                    plot_results = []
                    
                    if isinstance(result, dict):
                        for line_no, line_result in result.items():
                            if isinstance(line_result, dict) and "predict_hourly_flow" in line_result:
                                hourly_flow = line_result.get("predict_hourly_flow", {})
                                error_msg = line_result.get("error")
                                
                                if error_msg:
                                    st.warning(f"线路 {line_no} 预测错误: {error_msg}")
                                    continue
                                    
                                # 准备绘图数据
                                hours = [int(h) for h in sorted(hourly_flow.keys())]
                                flows = [hourly_flow[f"{h:02d}"] for h in hours]
                                
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
                                
                                fig = plot_hourly_flow(df_plot, SUBWAY_GREEN=SUBWAY_GREEN, SUBWAY_ACCENT=SUBWAY_ACCENT, SUBWAY_CARD=SUBWAY_CARD, SUBWAY_BG=SUBWAY_BG, SUBWAY_FONT=SUBWAY_FONT, return_fig=True)
                                figs.append((fig, None))
                                # # 生成图表
                                # fig = plot_hourly_flow(
                                #     df_plot, 
                                #     line_name=line_name, 
                                #     predict_date=predict_date_str,
                                #     SUBWAY_GREEN=SUBWAY_GREEN, 
                                #     SUBWAY_ACCENT=SUBWAY_ACCENT, 
                                #     SUBWAY_CARD=SUBWAY_CARD, 
                                #     SUBWAY_BG=SUBWAY_BG, 
                                #     SUBWAY_FONT=SUBWAY_FONT, 
                                #     return_fig=True
                                # )
                                # figs.append((fig, line_name))
                    
                    # 存储到session_state，供下方全宽显示
                    st.session_state["hourly_plot_results"] = plot_results
                    st.session_state["hourly_plot_figs"] = figs

    # 统一全宽显示预测图和表格
    # if st.session_state.get("hourly_plot_results") is not None:
    #     st.markdown("---")
    #     st.markdown(
    #         f"<h4 style='color:{SUBWAY_ACCENT};font-weight:700;'>小时预测结果可视化</h4>",
    #         unsafe_allow_html=True
    #     )
        
    #     for idx, plot_info in enumerate(st.session_state["hourly_plot_results"]):
    #         line_name = plot_info["line_name"]
    #         line_no = plot_info["line_no"]
    #         df_plot = plot_info["df_plot"]
    #         predict_date = plot_info["predict_date"]
    #         fig, _ = st.session_state["hourly_plot_figs"][idx]
            
    #         st.markdown(
    #             f"<div style='color:{SUBWAY_GREEN};font-weight:600;font-size:1.1rem;'>线路：{line_name} ({line_no})</div>",
    #             unsafe_allow_html=True
    #         )
    #         st.pyplot(fig, use_container_width=True)
    #         st.markdown(
    #             f"<span style='color:{SUBWAY_ACCENT};'>预测日期: {predict_date}</span>",
    #             unsafe_allow_html=True
    #         )
            
    #         # 格式化表格显示
    #         df_display = df_plot.copy()
    #         df_display["小时段"] = df_display["小时"].apply(lambda x: f"{x:02d}:00-{x+1:02d}:00")
    #         df_display = df_display[["小时段", "预测客流"]]
    #         st.dataframe(df_display, use_container_width=True)
    
    # 兼容原有图片文件
    if os.path.exists("timeseries_predict_hourly.png"):
        st.image("timeseries_predict_hourly.png", caption="小时预测结果可视化", use_container_width=True)