# Streamlit 应用模块：提供 Web 界面进行客流预测
import streamlit as st
import pandas as pd
from datetime import datetime
from config_utils import load_yaml_config, save_yaml_config
from predict_hourly import predict_and_plot_timeseries_flow
from predict_daily import predict_and_plot_timeseries_flow_daily
import os
import matplotlib.pyplot as plt

def get_model_versions(model_dir, prefix=""):
    """获取模型目录下所有模型版本（以日期为子目录）"""
    if not os.path.exists(model_dir):
        return []
    # 只列出日期子目录
    dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    # 只保留8位数字的日期目录
    versions = [d for d in dirs if len(d) == 8 and d.isdigit()]
    versions.sort(reverse=True)
    return versions

def plot_hourly_flow(df_plot, line_name=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    if line_name is not None:
        ax.plot(df_plot["小时"], df_plot["预测客流"], marker='o', label=line_name)
        ax.legend()
    else:
        ax.plot(df_plot["小时"], df_plot["预测客流"], marker='o')
    ax.set_xlabel("小时")
    ax.set_ylabel("预测客流")
    ax.set_xticks(df_plot["小时"])
    ax.set_title("小时预测客流量" + (f" - {line_name}" if line_name else ""))
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

def plot_daily_flow(df_plot, line_name=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    if line_name is not None:
        ax.plot(df_plot["日期"], df_plot["预测客流"], marker='o', label=line_name)
        ax.legend()
    else:
        ax.plot(df_plot["日期"], df_plot["预测客流"], marker='o')
    ax.set_xlabel("日期")
    ax.set_ylabel("预测客流")
    ax.set_xticks(df_plot["日期"])
    ax.set_xticklabels(df_plot["日期"], rotation=45, ha='right')
    ax.set_title("日预测客流量" + (f" - {line_name}" if line_name else ""))
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

def main():
    """
    Streamlit 应用主入口，提供小时和日客流预测界面
    """
    st.set_page_config(page_title="长沙地铁客流预测算法平台", layout="wide")
    st.title("长沙地铁客流预测算法平台")
    st.markdown("本平台支持线路小时客流预测（LSTM/Prophet）与线路日客流预测（KNN/Prophet），可选择预测参数、训练模型、预测并可视化结果。")

    config = load_yaml_config()
    config_daily = load_yaml_config("model_config_daily.yaml", default_daily=True)

    tab1, tab2 = st.tabs(["线路小时客流预测", "线路日客流预测"])

    with tab1:
        st.header("线路小时客流预测（LSTM/Prophet）")

        # 选择训练模型类型
        train_model_type = st.selectbox("训练模型类型", options=["lstm", "prophet"], 
                                        index=0 if config.get("default_algorithm", "lstm") == "lstm" else 1, key="hour_train_model_type")
        today = datetime.now()
        train_date = st.date_input("训练数据截止日期", value=today, key="hour_train_date")
        if isinstance(train_date, datetime):
            train_date_str = train_date.strftime("%Y%m%d")
        else:
            train_date_str = train_date.strftime("%Y%m%d")
        retrain = st.checkbox("强制重新训练模型", value=True, key="hour_retrain")

        with st.expander("高级训练参数设置", expanded=False):
            lookback_days = st.number_input("lookback_days", min_value=1, max_value=30, 
                                           value=config.get("train_params", {}).get("lookback_days", 7), key="hour_lookback_days")
            hidden_size = st.number_input("hidden_size", min_value=10, max_value=200, 
                                         value=config.get("train_params", {}).get("hidden_size", 50), key="hour_hidden_size")
            num_layers = st.number_input("num_layers", min_value=1, max_value=5, 
                                        value=config.get("train_params", {}).get("num_layers", 2), key="hour_num_layers")
            dropout = st.slider("dropout", min_value=0.0, max_value=0.8, 
                               value=float(config.get("train_params", {}).get("dropout", 0.2)), key="hour_dropout")
            batch_size = st.number_input("batch_size", min_value=8, max_value=256, 
                                        value=config.get("train_params", {}).get("batch_size", 32), key="hour_batch_size")
            epochs = st.number_input("epochs", min_value=10, max_value=500, 
                                    value=config.get("train_params", {}).get("epochs", 100), key="hour_epochs")
            patience = st.number_input("patience", min_value=1, max_value=50, 
                                      value=config.get("train_params", {}).get("patience", 10), key="hour_patience")
            learning_rate = st.number_input("learning_rate", min_value=0.0001, max_value=0.1, 
                                          value=float(config.get("train_params", {}).get("learning_rate", 0.001)), 
                                          step=0.0001, format="%.4f", key="hour_lr")
            config["train_params"] = {
                "lookback_days": lookback_days,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "batch_size": batch_size,
                "epochs": epochs,
                "patience": patience,
                "learning_rate": learning_rate
            }
            save_yaml_config(config)

        if st.button("开始小时模型训练", key="run_hour_train"):
            with st.spinner("正在训练小时模型..."):
                # 结构: models/hour/日期/算法类型
                model_save_dir = os.path.join("models", "hour", train_date_str, train_model_type)
                os.makedirs(model_save_dir, exist_ok=True)
                result = predict_and_plot_timeseries_flow(
                    file_path=None,
                    predict_date=train_date_str,
                    algorithm=train_model_type,
                    retrain=retrain,
                    save_path="timeseries_predict_hourly.png",
                    mode="train",
                    config=config,
                    model_version=None,
                    model_save_dir=model_save_dir
                )
            if isinstance(result, dict) and "error" in result:
                st.error(result["error"])
            else:
                st.success("小时模型训练完成！")
                st.info("请在下方推理模块选择模型版本进行预测。")

        st.markdown("---")
        st.subheader("小时客流推理预测")
        # 选择推理模型日期版本
        hour_model_root = os.path.join("models", "hour")
        hour_versions = get_model_versions(hour_model_root)
        if hour_versions:
            hour_version = st.selectbox("选择模型日期版本", options=hour_versions, key="hour_model_version")
            # 选择推理算法类型（该日期下有哪些算法模型文件夹）
            algo_dir = os.path.join(hour_model_root, hour_version)
            available_algos = [d for d in os.listdir(algo_dir) if os.path.isdir(os.path.join(algo_dir, d))]
            # 只保留支持的算法
            available_algos = [a for a in available_algos if a in ["lstm", "prophet"]]
            if available_algos:
                predict_model_type = st.selectbox("推理模型类型", options=available_algos, key="hour_predict_model_type")
                model_dir = os.path.join(hour_model_root, hour_version, predict_model_type)
            else:
                model_dir = None
                predict_model_type = None
                st.warning("该日期下未找到可用的小时算法模型，请先训练模型。")
        else:
            hour_version = None
            model_dir = None
            predict_model_type = None
            st.warning("未找到可用的小时模型版本，请先训练模型。")

        predict_date = st.date_input("预测日期", value=today, key="hour_predict_date")
        if isinstance(predict_date, datetime):
            predict_date_str = predict_date.strftime("%Y%m%d")
        else:
            predict_date_str = predict_date.strftime("%Y%m%d")

        if st.button("开始小时推理预测", key="run_hour_predict"):
            if not hour_version or not model_dir or not os.path.exists(model_dir) or not predict_model_type:
                st.error("请先训练并选择模型版本和算法类型。")
            else:
                with st.spinner("正在进行小时推理预测..."):
                    result = predict_and_plot_timeseries_flow(
                        file_path=None,
                        predict_date=predict_date_str,
                        algorithm=predict_model_type,
                        retrain=False,
                        save_path="timeseries_predict_hourly.png",
                        mode="predict",
                        config=config,
                        model_version=None,  # 版本由目录结构决定
                        model_save_dir=model_dir
                    )
                if isinstance(result, dict) and "error" in result:
                    st.error(result["error"])
                else:
                    st.success("小时推理预测完成！")
                    info = result if isinstance(result, dict) else {}
                    st.subheader("小时预测结果")
                    if info.get("error"):
                        st.warning(f"预测错误: {info['error']}")
                    else:
                        # 分线路展示
                        hourly_flow = info.get("predict_hourly_flow", {})
                        if isinstance(hourly_flow, dict) and all(isinstance(v, dict) for v in hourly_flow.values()):
                            for line_name, line_hourly in hourly_flow.items():
                                # 修正：小时key可能为字符串"00"~"23"或int 0~23，统一处理
                                # 先将所有key转为字符串并补零
                                hour_keys = [str(h).zfill(2) for h in line_hourly.keys()]
                                # 统一小时排序
                                hours = sorted([int(h) for h in hour_keys])
                                flows = []
                                for h in hours:
                                    # 优先取"hh"格式
                                    v = line_hourly.get(str(h).zfill(2))
                                    if v is None:
                                        v = line_hourly.get(str(h))
                                    if v is None:
                                        v = 0
                                    flows.append(v)
                                df_plot = pd.DataFrame({
                                    "小时": hours,
                                    "预测客流": flows
                                })
                                st.markdown(f"**线路：{line_name}**")
                                plot_hourly_flow(df_plot, line_name=line_name)
                                st.write(f"预测日期: {info.get('predict_date', predict_date_str)}")
                                st.dataframe(df_plot)
                        else:
                            # 单线路或整体结构
                            hour_keys = [str(h).zfill(2) for h in hourly_flow.keys()]
                            hours = sorted([int(h) for h in hour_keys])
                            flows = []
                            for h in hours:
                                v = hourly_flow.get(str(h).zfill(2))
                                if v is None:
                                    v = hourly_flow.get(str(h))
                                if v is None:
                                    v = 0
                                flows.append(v)
                            df_plot = pd.DataFrame({
                                "小时": hours,
                                "预测客流": flows
                            })
                            plot_hourly_flow(df_plot)
                            st.write(f"预测日期: {info.get('predict_date', predict_date_str)}")
                            st.dataframe(df_plot)
                    if os.path.exists("timeseries_predict_hourly.png"):
                        st.image("timeseries_predict_hourly.png", caption="小时预测结果可视化", use_container_width=True)

    with tab2:
        st.header("线路日客流预测（KNN/Prophet）")
        # 支持KNN和Prophet两种日预测算法
        daily_algos = ["knn", "prophet"]
        train_daily_algo = st.selectbox("训练日模型算法类型", options=daily_algos, key="daily_train_algo")
        n_neighbors = None
        if train_daily_algo == "knn":
            n_neighbors = st.number_input("KNN邻居数(n_neighbors)", min_value=1, max_value=30, 
                                          value=config_daily.get("train_params", {}).get("n_neighbors", 5), key="daily_n_neighbors")
            config_daily["train_params"]["n_neighbors"] = n_neighbors
        save_yaml_config(config_daily, "model_config_daily.yaml")
        train_start_date = st.date_input("训练数据截止日期（日）", value=datetime.now(), key="daily_train_date")
        if isinstance(train_start_date, datetime):
            train_start_date_str = train_start_date.strftime("%Y%m%d")
        else:
            train_start_date_str = train_start_date.strftime("%Y%m%d")
        retrain_daily = st.checkbox("强制重新训练日模型", value=True, key="daily_retrain")

        if st.button("开始日模型训练", key="run_daily_train"):
            with st.spinner("正在训练日模型..."):
                # 结构: models/daily/日期/算法类型
                model_save_dir_daily = os.path.join("models", "daily", train_start_date_str, train_daily_algo)
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
                    model_save_dir=model_save_dir_daily
                )
            if isinstance(result, dict) and "error" in result:
                st.error(result["error"])
            else:
                st.success("日模型训练完成！")
                st.info("请在下方推理模块选择模型版本和算法类型进行预测。")

        st.markdown("---")
        st.subheader("日客流推理预测")
        # 选择推理模型日期版本
        daily_model_root = os.path.join("models", "daily")
        daily_versions = get_model_versions(daily_model_root)
        if daily_versions:
            daily_version = st.selectbox("选择日模型日期版本", options=daily_versions, key="daily_model_version")
            # 选择推理算法类型（该日期下有哪些算法模型文件夹）
            algo_dir_daily = os.path.join(daily_model_root, daily_version)
            available_daily_algos = [d for d in os.listdir(algo_dir_daily) if os.path.isdir(os.path.join(algo_dir_daily, d))]
            # 只保留支持的算法
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

        predict_start_date = st.date_input("预测起始日期", value=datetime.now(), key="daily_predict_date")
        if isinstance(predict_start_date, datetime):
            predict_start_date_str = predict_start_date.strftime("%Y%m%d")
        else:
            predict_start_date_str = predict_start_date.strftime("%Y%m%d")
        days = st.number_input("预测天数", min_value=1, max_value=30, value=15, key="daily_days")

        if st.button("开始日推理预测", key="run_daily_predict"):
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
                        model_version=None,  # 版本由目录结构决定
                        model_save_dir=model_dir_daily
                    )
                if isinstance(result, dict) and "error" in result:
                    st.error(result["error"])
                else:
                    st.success("日推理预测完成！")
                    info = result if isinstance(result, dict) else {}
                    st.subheader("日预测结果")
                    if info.get("error"):
                        st.warning(f"预测错误: {info['error']}")
                    else:
                        # 分线路展示
                        daily_flow = info.get("predict_daily_flow", {})
                        if isinstance(daily_flow, dict) and all(isinstance(v, dict) for v in daily_flow.values()):
                            for line_name, line_daily in daily_flow.items():
                                # 修正：日期key为字符串，flows需为数值型
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
                                st.markdown(f"**线路：{line_name}**")
                                plot_daily_flow(df_plot, line_name=line_name)
                                st.write(f"预测起始日期: {info.get('predict_start_date', predict_start_date_str)}")
                                st.dataframe(df_plot)
                        else:
                            # 单线路或整体结构
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
                            plot_daily_flow(df_plot)
                            st.write(f"预测起始日期: {info.get('predict_start_date', predict_start_date_str)}")
                            st.dataframe(df_plot)
                    if os.path.exists("timeseries_predict_daily.png"):
                        st.image("timeseries_predict_daily.png", caption="日预测结果可视化", use_container_width=True)