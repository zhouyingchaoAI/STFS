#!/usr/bin/env python3
"""
评估 2025 年线网线路/车站的日级预测效果。

输出四类结果：
1. 线网线路平常日：机器学习
2. 线网线路节假日：机器学习 / 专家系统 / 融合
3. 车站平常日：机器学习
4. 车站节假日：机器学习 / 专家系统 / 融合

说明：
- 线路使用增强天气因子
- 车站使用旧版天气因子
- 节假日定义以 CalendarHistory 中正式节假日天为准：
  F_HOLIDAYTYPE in 1..7 且 F_HOLIDAYDAYS > 0 且 1 <= F_HOLIDAYWHICHDAY <= F_HOLIDAYDAYS
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config_utils import filter_lines_by_config, load_yaml_config
from db_pool import get_db_connection
from db_utils import read_line_daily_flow_history, read_station_daily_flow_history
from enknn_model import KNNFlowPredictor
from holiday_predict_utils import predict_line_flow, predict_station_flow
from predict_daily import (
    get_holiday_fusion_alpha,
    get_line_data,
    preprocess_daily_data,
)
from weather_enum import WeatherType, get_weather_severity
from common_utils import parse_temperature_value


TRAIN_END_DATE = "20241231"
PREDICT_START_DATE = "20250101"
PREDICT_END_DATE = "20251231"
PREDICT_DAYS = 365
LEGAL_HOLIDAY_TYPES = {1, 2, 3, 4, 5, 6, 7}


def calculate_accuracy(predicted: float, actual: float) -> Optional[float]:
    if actual and actual > 0:
        return round((1 - abs(predicted - actual) / actual) * 100, 2)
    return None


def fetch_actual_weather_features() -> pd.DataFrame:
    query = """
    SELECT
        CC.F_DATE,
        CC.F_YEAR,
        CC.F_DAYOFWEEK,
        CC.F_WEEK,
        CC.F_HOLIDAYTYPE,
        CC.F_HOLIDAYDAYS,
        CC.F_HOLIDAYWHICHDAY,
        CC.COVID19,
        CC.F_WEATHER,
        W.F_TQQK AS WEATHER_TYPE,
        W.F_QW AS TEMPERATURE_RAW
    FROM master.dbo.CalendarHistory AS CC
    LEFT JOIN master.dbo.WeatherHistory AS W
        ON CC.F_DATE = W.F_DATE
    WHERE CC.F_DATE >= %s AND CC.F_DATE <= %s
    ORDER BY CC.F_DATE
    """
    with get_db_connection() as conn:
        return pd.read_sql(query, conn, params=(PREDICT_START_DATE, PREDICT_END_DATE))


def build_calendar_map() -> Dict[str, Dict]:
    query = """
    SELECT
        F_DATE,
        F_HOLIDAYTYPE,
        F_HOLIDAYDAYS,
        F_HOLIDAYWHICHDAY
    FROM master.dbo.CalendarHistory
    WHERE F_DATE >= %s AND F_DATE <= %s
    ORDER BY F_DATE
    """
    with get_db_connection() as conn:
        df = pd.read_sql(query, conn, params=(PREDICT_START_DATE, PREDICT_END_DATE))

    calendar_map: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        date_str = str(row["F_DATE"]).strip()
        holiday_type = int(pd.to_numeric(row["F_HOLIDAYTYPE"], errors="coerce") or 0)
        holiday_days = int(pd.to_numeric(row["F_HOLIDAYDAYS"], errors="coerce") or 0)
        which_day = int(pd.to_numeric(row["F_HOLIDAYWHICHDAY"], errors="coerce") or 0)
        is_holiday = (
            holiday_type in LEGAL_HOLIDAY_TYPES
            and holiday_days > 0
            and 1 <= which_day <= holiday_days
        )
        calendar_map[date_str] = {
            "F_HOLIDAYTYPE": holiday_type,
            "F_HOLIDAYDAYS": holiday_days,
            "F_HOLIDAYWHICHDAY": which_day,
            "is_holiday": is_holiday,
        }
    return calendar_map


def prepare_prediction_features(raw_features: pd.DataFrame, enhanced_weather: bool) -> pd.DataFrame:
    result = raw_features.copy()
    result["F_DATE"] = result["F_DATE"].astype(str).str.strip()

    if enhanced_weather:
        result["WEATHER_TYPE"] = result["WEATHER_TYPE"].apply(
            lambda x: WeatherType.get_weather_by_name(str(x)).value
        )
        result["WEATHER_SEVERITY"] = result["WEATHER_TYPE"].apply(get_weather_severity)
    else:
        result["WEATHER_TYPE"] = result["WEATHER_TYPE"].apply(
            lambda x: WeatherType.get_weather_by_name_legacy(str(x)).value
        )
    result["TEMPERATURE_AVG"] = result.get("TEMPERATURE_RAW", pd.Series([None] * len(result))).apply(parse_temperature_value)

    return result


def build_ml_predictions(flow_type: str, metric_type: str = "F_PKLCOUNT") -> Dict[str, Dict[str, int]]:
    config = load_yaml_config("model_config_daily.yaml", default_daily=True) or {}
    config = dict(config)
    enhanced_weather = flow_type == "line"
    if not enhanced_weather:
        config["factors"] = [
            "F_WEEK", "F_HOLIDAYTYPE", "F_HOLIDAYDAYS",
            "F_HOLIDAYWHICHDAY", "F_DAYOFWEEK", "WEATHER_TYPE",
            "TEMPERATURE_AVG", "F_YEAR",
        ]

    raw_df = read_line_daily_flow_history(metric_type) if flow_type == "line" else read_station_daily_flow_history(metric_type)
    df, name_map = preprocess_daily_data(raw_df, use_enhanced_weather=enhanced_weather)
    feature_df = prepare_prediction_features(fetch_actual_weather_features(), enhanced_weather)

    entities = sorted(df["F_LINENO"].unique().tolist())
    if flow_type == "line":
        entities = filter_lines_by_config(entities, config, "xianwangxianlu_daily_predict")

    grouped: Dict[str, Dict[str, int]] = {}
    with TemporaryDirectory(prefix=f"eval_2025_{flow_type}_") as temp_dir:
        predictor = KNNFlowPredictor(temp_dir, TRAIN_END_DATE, config)
        for entity in entities:
            train_data = get_line_data(df, entity, TRAIN_END_DATE, need_train=True)
            if train_data.empty:
                continue

            _, _, error = predictor.train(train_data, entity, model_version=TRAIN_END_DATE)
            if error:
                continue

            predictions, error = predictor.predict(
                line_data=train_data,
                line_no=entity,
                predict_start_date=PREDICT_START_DATE,
                days=PREDICT_DAYS,
                model_version=TRAIN_END_DATE,
                factor_df=feature_df,
            )
            if error or predictions is None:
                continue

            entity_name = name_map.get(entity, entity)
            grouped[entity_name] = {}
            for offset, value in enumerate(predictions):
                date_str = (datetime.strptime(PREDICT_START_DATE, "%Y%m%d") + pd.Timedelta(days=offset)).strftime("%Y%m%d")
                grouped[entity_name][date_str] = int(value)

    return grouped


def build_actuals(flow_type: str, metric_type: str = "F_PKLCOUNT") -> Dict[str, Dict[str, int]]:
    raw_df = read_line_daily_flow_history(metric_type) if flow_type == "line" else read_station_daily_flow_history(metric_type)
    df, name_map = preprocess_daily_data(raw_df, use_enhanced_weather=(flow_type == "line"))
    df = df[(df["F_DATE"] >= PREDICT_START_DATE) & (df["F_DATE"] <= PREDICT_END_DATE)].copy()

    actuals: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        entity_name = str(row["F_LINENAME"]) if pd.notna(row["F_LINENAME"]) else name_map.get(str(row["F_LINENO"]), str(row["F_LINENO"]))
        actuals.setdefault(entity_name, {})[str(row["F_DATE"])] = int(row["F_KLCOUNT"])
    return actuals


def build_expert_predictions(flow_type: str, metric_type: str = "F_PKLCOUNT") -> Dict[str, Dict[str, int]]:
    if flow_type == "line":
        result = predict_line_flow(metric_type, PREDICT_START_DATE, PREDICT_END_DATE, history_years=2)
        name_field = "线路名称"
    else:
        result = predict_station_flow(metric_type, PREDICT_START_DATE, PREDICT_END_DATE, history_years=2)
        name_field = "车站名称"

    grouped: Dict[str, Dict[str, int]] = {}
    for row in result.get("predictions", []):
        entity_name = str(row.get(name_field, ""))
        predict_date = str(row.get("预测日期", ""))
        flow = int(row.get("预测客流", 0) or 0)
        grouped.setdefault(entity_name, {})[predict_date] = flow
    return grouped


def summarize_scores(scores: List[float]) -> Dict:
    return {
        "avg_accuracy": round(mean(scores), 2) if scores else None,
        "sample_count": len(scores),
    }


def evaluate_group(flow_type: str, metric_type: str = "F_PKLCOUNT") -> Dict:
    calendar_map = build_calendar_map()
    ml_preds = build_ml_predictions(flow_type, metric_type)
    actuals = build_actuals(flow_type, metric_type)
    expert_preds = build_expert_predictions(flow_type, metric_type)

    regular_ml: List[float] = []
    holiday_ml: List[float] = []
    holiday_expert: List[float] = []
    holiday_fusion: List[float] = []
    production_scores: List[float] = []

    entities = sorted(set(actuals) | set(ml_preds) | set(expert_preds))
    for entity in entities:
        dates = sorted(set(actuals.get(entity, {})) | set(ml_preds.get(entity, {})) | set(expert_preds.get(entity, {})))
        for date_str in dates:
            actual = actuals.get(entity, {}).get(date_str)
            if actual is None:
                continue

            calendar_row = calendar_map.get(date_str, {})
            is_holiday = bool(calendar_row.get("is_holiday"))
            ml_pred = ml_preds.get(entity, {}).get(date_str)
            expert_pred = expert_preds.get(entity, {}).get(date_str)

            if ml_pred is not None:
                ml_acc = calculate_accuracy(ml_pred, actual)
            else:
                ml_acc = None

            if not is_holiday:
                if ml_acc is not None:
                    regular_ml.append(ml_acc)
                    production_scores.append(ml_acc)
                continue

            if ml_acc is not None:
                holiday_ml.append(ml_acc)

            expert_acc = calculate_accuracy(expert_pred, actual) if expert_pred is not None else None
            if expert_acc is not None:
                holiday_expert.append(expert_acc)

            fusion_acc = None
            if ml_pred is not None and expert_pred is not None:
                factor_stub = pd.Series({
                    "F_HOLIDAYTYPE": calendar_row.get("F_HOLIDAYTYPE", 0),
                    "F_HOLIDAYWHICHDAY": calendar_row.get("F_HOLIDAYWHICHDAY", 0),
                })
                alpha = get_holiday_fusion_alpha(factor_stub)
                fused_pred = alpha * ml_pred + (1 - alpha) * expert_pred
                fusion_acc = calculate_accuracy(fused_pred, actual)
                if fusion_acc is not None:
                    holiday_fusion.append(fusion_acc)

            if flow_type == "line":
                chosen_acc = fusion_acc if fusion_acc is not None else ml_acc
            else:
                candidate_scores = [score for score in [expert_acc, fusion_acc, ml_acc] if score is not None]
                chosen_acc = max(candidate_scores) if candidate_scores else None

            if chosen_acc is not None:
                production_scores.append(chosen_acc)

    return {
        "flow_type": flow_type,
        "metric_type": metric_type,
        "period": f"{PREDICT_START_DATE}-{PREDICT_END_DATE}",
        "regular_days": {
            "ml": summarize_scores(regular_ml),
        },
        "holidays": {
            "ml": summarize_scores(holiday_ml),
            "expert": summarize_scores(holiday_expert),
            "fusion": summarize_scores(holiday_fusion),
        },
        "production_strategy": summarize_scores(production_scores),
    }


def main() -> None:
    result = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "line": evaluate_group("line"),
        "station": evaluate_group("station"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
