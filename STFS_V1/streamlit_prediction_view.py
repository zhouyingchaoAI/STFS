# streamlit_prediction_view.py
# 只读预测结果查看页：按预测批次读取预测表并对比实际值

import io
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from db_pool import get_db_connection
from db_utils import fix_dataframe_encoding
from ui_components import render_section_header


LINE_PRED_METRIC_FIELDS = {
    "F_PKLCOUNT": "F_PKLCOUNT",
    "F_ENTRANCE": "ENTRY_NUM",
    "F_EXIT": "EXIT_NUM",
    "F_TRANSFER": "CHANGE_NUM",
    "F_BOARD_ALIGHT": "FLOW_NUM",
}

STATION_PRED_METRIC_FIELDS = {
    "F_PKLCOUNT": "PASSENGER_NUM",
    "F_ENTRANCE": "ENTRY_NUM",
    "F_EXIT": "EXIT_NUM",
    "F_TRANSFER": "CHANGE_NUM",
    "F_BOARD_ALIGHT": "FLOW_NUM",
}

LINE_ACTUAL_METRIC_FIELDS = {
    "F_PKLCOUNT": "F_KLCOUNT",
    "F_ENTRANCE": "ENTRY_NUM",
    "F_EXIT": "EXIT_NUM",
    "F_TRANSFER": "CHANGE_NUM",
    "F_BOARD_ALIGHT": "FLOW_NUM",
}

STATION_ACTUAL_METRIC_FIELDS = {
    "F_PKLCOUNT": "PASSENGER_NUM",
    "F_ENTRANCE": "ENTRY_NUM",
    "F_EXIT": "EXIT_NUM",
    "F_TRANSFER": "CHANGE_NUM",
    "F_BOARD_ALIGHT": "FLOW_NUM",
}

CREATOR_LABELS = {
    "knn_predict": "机器学习",
    "expert_predict": "专家系统",
    "fusion_predict": "融合结果",
}

CREATOR_ORDER = ["knn_predict", "expert_predict", "fusion_predict"]


def _normalize_predict_date(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text.replace("-", "")[:8]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_available_target_dates(flow_type: str, metric_type: str) -> List[str]:
    if flow_type == "xianwangxianlu":
        query = """
        SELECT DISTINCT TOP 365 CONVERT(varchar(8), F_DATE) AS target_date
        FROM [CxFlowPredict].[dbo].[LineDailyFlowPrediction]
        WHERE F_DATE IS NOT NULL
          AND CREATOR IN ('knn_predict', 'expert_predict', 'fusion_predict')
        ORDER BY target_date DESC
        """
    elif flow_type == "chezhan":
        query = """
        SELECT DISTINCT TOP 365 REPLACE(CONVERT(varchar(10), SQUAD_DATE, 120), '-', '') AS target_date
        FROM [CxFlowPredict].[dbo].[STATION_FLOW_PREDICT]
        WHERE SQUAD_DATE IS NOT NULL
          AND CREATOR IN ('knn_predict', 'expert_predict', 'fusion_predict')
        ORDER BY target_date DESC
        """
    else:
        return []

    with get_db_connection() as conn:
        df = pd.read_sql(query, conn)
    df = fix_dataframe_encoding(df)
    dates = [_normalize_predict_date(v) for v in df.get("target_date", []).tolist()]
    return [d for d in dates if d]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prediction_rows(flow_type: str, metric_type: str, start_date: str, end_date: str) -> pd.DataFrame:
    start_date = _normalize_predict_date(start_date)
    end_date = _normalize_predict_date(end_date)

    if flow_type == "xianwangxianlu":
        metric_field = LINE_PRED_METRIC_FIELDS[metric_type]
        query = f"""
        SELECT
            CONVERT(varchar(8), F_DATE) AS target_date,
            RIGHT('00' + CAST(F_LINENO AS varchar(8)), 2) AS entity_id,
            F_LINENAME AS entity_name,
            CREATOR AS creator,
            SUM(COALESCE({metric_field}, 0)) AS predicted_value
        FROM [CxFlowPredict].[dbo].[LineDailyFlowPrediction]
        WHERE F_DATE >= %s
          AND F_DATE <= %s
          AND CREATOR IN ('knn_predict', 'expert_predict', 'fusion_predict')
        GROUP BY F_DATE, F_LINENO, F_LINENAME, CREATOR
        ORDER BY target_date, entity_id, creator
        """
    elif flow_type == "chezhan":
        metric_field = STATION_PRED_METRIC_FIELDS[metric_type]
        query = f"""
        SELECT
            REPLACE(CONVERT(varchar(10), SQUAD_DATE, 120), '-', '') AS target_date,
            STATION_NAME AS entity_id,
            STATION_NAME AS entity_name,
            CREATOR AS creator,
            SUM(COALESCE({metric_field}, 0)) AS predicted_value
        FROM [CxFlowPredict].[dbo].[STATION_FLOW_PREDICT]
        WHERE REPLACE(CONVERT(varchar(10), SQUAD_DATE, 120), '-', '') >= %s
          AND REPLACE(CONVERT(varchar(10), SQUAD_DATE, 120), '-', '') <= %s
          AND CREATOR IN ('knn_predict', 'expert_predict', 'fusion_predict')
        GROUP BY SQUAD_DATE, STATION_NAME, CREATOR
        ORDER BY target_date, entity_name, creator
        """
    else:
        return pd.DataFrame()

    with get_db_connection() as conn:
        df = pd.read_sql(query, conn, params=(start_date, end_date))
    df = fix_dataframe_encoding(df)
    if df.empty:
        return df

    df["target_date"] = df["target_date"].astype(str).str.strip()
    df["entity_id"] = df["entity_id"].astype(str).str.strip()
    df["entity_name"] = df["entity_name"].astype(str).str.strip()
    df["creator"] = df["creator"].astype(str).str.strip()
    df["predicted_value"] = pd.to_numeric(df["predicted_value"], errors="coerce").fillna(0)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_actual_rows(flow_type: str, metric_type: str, start_date: str, end_date: str) -> pd.DataFrame:
    if flow_type == "xianwangxianlu":
        metric_field = LINE_ACTUAL_METRIC_FIELDS[metric_type]
        query = f"""
        SELECT
            CONVERT(varchar(8), L.F_DATE) AS target_date,
            RIGHT('00' + CAST(L.F_LINENO AS varchar(8)), 2) AS entity_id,
            L.F_LINENAME AS entity_name,
            SUM(COALESCE(L.{metric_field}, 0)) AS actual_value
        FROM dbo.LineDailyFlowHistory AS L
        WHERE L.CREATOR = 'chency'
          AND L.F_DATE >= %s
          AND L.F_DATE <= %s
        GROUP BY L.F_DATE, L.F_LINENO, L.F_LINENAME
        ORDER BY target_date, entity_id
        """
    elif flow_type == "chezhan":
        metric_field = STATION_ACTUAL_METRIC_FIELDS[metric_type]
        query = f"""
        SELECT
            REPLACE(S.SQUAD_DATE, '-', '') AS target_date,
            S.STATION_NAME AS entity_id,
            S.STATION_NAME AS entity_name,
            SUM(COALESCE(S.{metric_field}, 0)) AS actual_value
        FROM [StationFlowPredict].[dbo].[STATION_FLOW_HISTORY] AS S
        WHERE REPLACE(S.SQUAD_DATE, '-', '') >= %s
          AND REPLACE(S.SQUAD_DATE, '-', '') <= %s
        GROUP BY REPLACE(S.SQUAD_DATE, '-', ''), S.STATION_NAME
        ORDER BY target_date, entity_name
        """
    else:
        return pd.DataFrame()

    with get_db_connection() as conn:
        df = pd.read_sql(query, conn, params=(start_date, end_date))
    df = fix_dataframe_encoding(df)
    if df.empty:
        return df

    df["target_date"] = df["target_date"].astype(str).str.strip()
    df["entity_id"] = df["entity_id"].astype(str).str.strip()
    df["entity_name"] = df["entity_name"].astype(str).str.strip()
    df["actual_value"] = pd.to_numeric(df["actual_value"], errors="coerce").fillna(0)
    return df


def build_view_dataframe(pred_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()

    pivot_df = pred_df.pivot_table(
        index=["target_date", "entity_id", "entity_name"],
        columns="creator",
        values="predicted_value",
        aggfunc="sum",
    ).reset_index()
    pivot_df.columns.name = None

    merged = pivot_df.merge(
        actual_df,
        on=["target_date", "entity_id", "entity_name"],
        how="left",
    )

    for creator in CREATOR_ORDER:
        if creator not in merged.columns:
            merged[creator] = pd.NA

    for creator in CREATOR_ORDER:
        acc_col = f"{creator}_accuracy"
        merged[acc_col] = merged.apply(
            lambda row: round((1 - abs(float(row[creator]) - float(row["actual_value"])) / float(row["actual_value"])) * 100, 2)
            if pd.notna(row.get(creator)) and pd.notna(row.get("actual_value")) and float(row.get("actual_value") or 0) > 0
            else pd.NA,
            axis=1,
        )

    merged["target_date"] = merged["target_date"].astype(str)
    return merged.sort_values(["target_date", "entity_id"]).reset_index(drop=True)


def build_accuracy_summary(view_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for creator in CREATOR_ORDER:
        acc_col = f"{creator}_accuracy"
        valid = pd.to_numeric(view_df.get(acc_col), errors="coerce").dropna()
        pred_count = pd.to_numeric(view_df.get(creator), errors="coerce").notna().sum()
        rows.append({
            "结果类型": CREATOR_LABELS[creator],
            "平均准确率%": round(float(valid.mean()), 2) if not valid.empty else None,
            "有效样本数": int(valid.shape[0]),
            "预测行数": int(pred_count),
        })
    return pd.DataFrame(rows)


def build_accuracy_curve(view_df: pd.DataFrame) -> pd.DataFrame:
    curve_rows = []
    for creator in CREATOR_ORDER:
        acc_col = f"{creator}_accuracy"
        subset = view_df[["target_date", acc_col]].copy()
        subset[acc_col] = pd.to_numeric(subset[acc_col], errors="coerce")
        subset = subset.dropna(subset=[acc_col])
        if subset.empty:
            continue
        grouped = subset.groupby("target_date", as_index=False)[acc_col].mean()
        grouped["creator"] = CREATOR_LABELS[creator]
        grouped = grouped.rename(columns={acc_col: "avg_accuracy"})
        curve_rows.append(grouped)

    if not curve_rows:
        return pd.DataFrame()
    curve_df = pd.concat(curve_rows, ignore_index=True)
    curve_df["target_date"] = pd.to_datetime(curve_df["target_date"], format="%Y%m%d")
    return curve_df.sort_values(["target_date", "creator"])


def build_flow_curve(view_df: pd.DataFrame) -> pd.DataFrame:
    flow_rows = []
    series_map = {
        "actual_value": "实际值",
        "knn_predict": "机器学习",
        "expert_predict": "专家系统",
        "fusion_predict": "融合结果",
    }

    for col, label in series_map.items():
        if col not in view_df.columns:
            continue
        subset = view_df[["target_date", col]].copy()
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
        subset = subset.dropna(subset=[col])
        if subset.empty:
            continue
        grouped = subset.groupby("target_date", as_index=False)[col].sum()
        grouped["series"] = label
        grouped = grouped.rename(columns={col: "value"})
        flow_rows.append(grouped)

    if not flow_rows:
        return pd.DataFrame()

    curve_df = pd.concat(flow_rows, ignore_index=True)
    curve_df["target_date"] = pd.to_datetime(curve_df["target_date"], format="%Y%m%d")
    return curve_df.sort_values(["target_date", "series"])


def render_accuracy_chart(curve_df: pd.DataFrame, title: str) -> bytes | None:
    if curve_df.empty:
        return None

    color_map = {
        "机器学习": "#00ffd5",
        "专家系统": "#ffaa00",
        "融合结果": "#60a5fa",
    }

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor("#0f1423")
    ax.set_facecolor("#1e2439")

    for creator, creator_df in curve_df.groupby("creator"):
        ax.plot(
            creator_df["target_date"],
            creator_df["avg_accuracy"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=creator,
            color=color_map.get(creator, "#ffffff"),
        )

    ax.set_title(title, color="#ffffff", fontsize=14, fontweight="bold")
    ax.set_ylabel("平均准确率 (%)", color="#d9e1f2")
    ax.set_xlabel("预测日期", color="#d9e1f2")
    ax.grid(alpha=0.15, color="#8aa0bf")
    ax.tick_params(colors="#d9e1f2")
    ax.legend(facecolor="#1e2439", edgecolor="#34415f", labelcolor="#ffffff")
    fig.autofmt_xdate()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def render_flow_chart(curve_df: pd.DataFrame, title: str) -> bytes | None:
    if curve_df.empty:
        return None

    color_map = {
        "实际值": "#ff5c8a",
        "机器学习": "#00ffd5",
        "专家系统": "#ffaa00",
        "融合结果": "#60a5fa",
    }

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor("#0f1423")
    ax.set_facecolor("#1e2439")

    for series, series_df in curve_df.groupby("series"):
        ax.plot(
            series_df["target_date"],
            series_df["value"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=series,
            color=color_map.get(series, "#ffffff"),
        )

    ax.set_title(title, color="#ffffff", fontsize=14, fontweight="bold")
    ax.set_ylabel("客流/人数", color="#d9e1f2")
    ax.set_xlabel("预测日期", color="#d9e1f2")
    ax.grid(alpha=0.15, color="#8aa0bf")
    ax.tick_params(colors="#d9e1f2")
    ax.legend(facecolor="#1e2439", edgecolor="#34415f", labelcolor="#ffffff")
    fig.autofmt_xdate()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def prediction_view_tab(
    SUBWAY_GREEN: str,
    SUBWAY_ACCENT: str,
    SUBWAY_CARD: str,
    SUBWAY_FONT: str,
    SUBWAY_BG: str,
    flow_type: str,
    metric_type: str,
):
    if flow_type not in {"xianwangxianlu", "chezhan"}:
        st.info("预测结果查看页当前仅支持线路线网和车站日预测结果。")
        return

    render_section_header(
        "预测结果查看",
        "只读取预测表和实际表，查看定时任务产出的结果与准确率，不触发任何推理。",
        "📚",
    )

    available_dates = fetch_available_target_dates(flow_type, metric_type)
    if not available_dates:
        st.warning("当前预测表中还没有可查看的目标日期。")
        return

    all_dates = sorted(available_dates)
    min_dt = datetime.strptime(all_dates[0], "%Y%m%d").date()
    max_dt = datetime.strptime(all_dates[-1], "%Y%m%d").date()

    if "prediction_view_start_date" not in st.session_state:
        st.session_state["prediction_view_start_date"] = min_dt
    elif st.session_state["prediction_view_start_date"] > max_dt:
        st.session_state["prediction_view_start_date"] = max_dt

    selected_start_state = st.session_state["prediction_view_start_date"]
    default_end = min(selected_start_state + pd.Timedelta(days=14), max_dt)
    previous_start = st.session_state.get("prediction_view_previous_start_date")

    if "prediction_view_end_date" not in st.session_state:
        st.session_state["prediction_view_end_date"] = default_end
    elif previous_start != selected_start_state:
        st.session_state["prediction_view_end_date"] = default_end
    elif st.session_state["prediction_view_end_date"] < selected_start_state:
        st.session_state["prediction_view_end_date"] = selected_start_state
    elif st.session_state["prediction_view_end_date"] > max_dt:
        st.session_state["prediction_view_end_date"] = max_dt

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        selected_start = st.date_input(
            "开始日期",
            max_value=max_dt,
            key="prediction_view_start_date",
        )

    if previous_start != selected_start:
        st.session_state["prediction_view_end_date"] = min(selected_start + pd.Timedelta(days=14), max_dt)
    st.session_state["prediction_view_previous_start_date"] = selected_start

    if st.session_state["prediction_view_end_date"] < selected_start:
        st.session_state["prediction_view_end_date"] = selected_start

    with date_col2:
        selected_end = st.date_input(
            "结束日期",
            min_value=selected_start,
            max_value=max_dt,
            key="prediction_view_end_date",
        )

    if selected_start > selected_end:
        st.warning("开始日期不能晚于结束日期。")
        return

    selected_start_str = selected_start.strftime("%Y%m%d")
    selected_end_str = selected_end.strftime("%Y%m%d")
    pred_df = fetch_prediction_rows(flow_type, metric_type, selected_start_str, selected_end_str)
    if pred_df.empty:
        st.info("当前时间段没有预测结果，不绘制曲线。")
        return

    actual_df = fetch_actual_rows(flow_type, metric_type, selected_start_str, selected_end_str)
    view_df = build_view_dataframe(pred_df, actual_df)

    entity_options = ["全部"] + sorted(view_df["entity_name"].dropna().astype(str).unique().tolist())
    selected_entity = st.selectbox("查看对象", options=entity_options, index=0)
    if selected_entity != "全部":
        filtered_df = view_df[view_df["entity_name"] == selected_entity].copy()
    else:
        filtered_df = view_df.copy()

    if filtered_df.empty:
        st.info("当前时间段或对象下没有预测结果，不绘制曲线。")
        return

    summary_df = build_accuracy_summary(filtered_df)
    curve_df = build_accuracy_curve(filtered_df)
    flow_curve_df = build_flow_curve(filtered_df)
    chart_bytes = render_accuracy_chart(
        curve_df,
        f"{selected_start_str}-{selected_end_str} 准确率曲线" + (f" - {selected_entity}" if selected_entity != "全部" else ""),
    )
    flow_chart_bytes = render_flow_chart(
        flow_curve_df,
        f"{selected_start_str}-{selected_end_str} 预测对比曲线" + (f" - {selected_entity}" if selected_entity != "全部" else ""),
    )

    col1, col2, col3 = st.columns(3)
    summary_map = {
        row["结果类型"]: row for _, row in summary_df.iterrows()
    }
    for column, label in zip(
        [col1, col2, col3],
        ["机器学习", "专家系统", "融合结果"],
    ):
        row = summary_map.get(label, {})
        avg_acc = row.get("平均准确率%", None)
        valid_count = row.get("有效样本数", 0)
        with column:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, rgba(15,20,35,0.95), rgba(30,36,57,0.82));
                    border: 1px solid rgba(0,255,213,0.18);
                    border-radius: 14px;
                    padding: 1rem 1.1rem;
                    min-height: 110px;
                ">
                    <div style="color:#9aa5bd;font-size:0.88rem;">{label}</div>
                    <div style="color:#ffffff;font-size:1.7rem;font-weight:700;margin-top:0.35rem;">
                        {f"{avg_acc:.2f}%" if pd.notna(avg_acc) else "--"}
                    </div>
                    <div style="color:#7f8ba3;font-size:0.82rem;margin-top:0.35rem;">
                        有效样本 {valid_count}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("结果类型数", len(CREATOR_ORDER))
    info_col2.metric("覆盖日期", f"{selected_start_str} ~ {selected_end_str}")
    info_col3.metric("查看对象数", filtered_df['entity_name'].nunique())

    if flow_chart_bytes:
        st.image(flow_chart_bytes, caption="实际值与三类预测结果对比曲线", use_container_width=True)
    else:
        st.info("当前筛选条件下没有可展示的预测曲线。")

    if chart_bytes:
        st.image(chart_bytes, caption="按预测日期聚合的平均准确率曲线", use_container_width=True)
    else:
        st.info("当前筛选条件下缺少实际值或预测值，无法绘制准确率曲线。")

    display_df = filtered_df.copy()

    for col in [
        "knn_predict",
        "knn_predict_accuracy",
        "expert_predict",
        "expert_predict_accuracy",
        "fusion_predict",
        "fusion_predict_accuracy",
    ]:
        if col not in display_df.columns:
            display_df[col] = pd.NA

    display_columns = [
        "target_date",
        "entity_id",
        "entity_name",
        "actual_value",
        "knn_predict",
        "knn_predict_accuracy",
        "expert_predict",
        "expert_predict_accuracy",
        "fusion_predict",
        "fusion_predict_accuracy",
    ]
    display_df = display_df[display_columns].rename(columns={
        "target_date": "预测日期",
        "entity_id": "对象ID",
        "entity_name": "对象名称",
        "actual_value": "实际值",
        "knn_predict": "机器学习",
        "knn_predict_accuracy": "机器学习准确率%",
        "expert_predict": "专家系统",
        "expert_predict_accuracy": "专家系统准确率%",
        "fusion_predict": "融合结果",
        "fusion_predict_accuracy": "融合结果准确率%",
    })

    st.markdown("### 预测表与实际值对比")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载当前结果 CSV",
        data=csv_bytes,
        file_name=f"prediction_view_{flow_type}_{metric_type}_{selected_start_str}_{selected_end_str}.csv",
        mime="text/csv",
    )
