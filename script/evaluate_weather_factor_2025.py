#!/usr/bin/env python3
"""
评估 2025 全年天气因子优化效果。

对比两套策略：
1. legacy_weather: 旧版精确匹配天气映射，仅使用 WEATHER_TYPE
2. improved_weather: 新版归一化映射 + WEATHER_SEVERITY
"""

from __future__ import annotations

import json
import os
import sys
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Dict, List

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config_utils import load_yaml_config, filter_lines_by_config
from db_pool import get_db_connection
from db_utils import read_line_daily_flow_history
from enknn_model import KNNFlowPredictor
from predict_daily import get_line_data
from weather_enum import WeatherType, get_weather_severity


TRAIN_END_DATE = "20241231"
PREDICT_START_DATE = "20250101"
PREDICT_END_DATE = "20251231"
PREDICT_DAYS = 365


def calculate_accuracy(predicted: float, actual: float) -> float | None:
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
        W.F_TQQK AS WEATHER_TYPE
    FROM master.dbo.CalendarHistory AS CC
    LEFT JOIN master.dbo.WeatherHistory AS W
        ON CC.F_DATE = W.F_DATE
    WHERE CC.F_DATE >= %s AND CC.F_DATE <= %s
    ORDER BY CC.F_DATE
    """
    with get_db_connection() as conn:
        return pd.read_sql(query, conn, params=(PREDICT_START_DATE, PREDICT_END_DATE))


def prepare_training_data(df: pd.DataFrame, mapper_name: str) -> pd.DataFrame:
    result = df.copy()
    result = result.dropna(subset=["F_DATE", "F_KLCOUNT", "F_LINENO"])
    result["F_DATE"] = result["F_DATE"].astype(str).str.strip()
    result["F_KLCOUNT"] = pd.to_numeric(result["F_KLCOUNT"], errors="coerce").fillna(0)
    result["F_LINENO"] = result["F_LINENO"].astype(str).str.zfill(2)

    if mapper_name == "legacy":
        result["WEATHER_TYPE"] = result["WEATHER_TYPE"].apply(
            lambda x: WeatherType.get_weather_by_name_legacy(str(x)).value
        )
    else:
        result["WEATHER_TYPE"] = result["WEATHER_TYPE"].apply(
            lambda x: WeatherType.get_weather_by_name(str(x)).value
        )
        result["WEATHER_SEVERITY"] = result["WEATHER_TYPE"].apply(get_weather_severity)

    return result


def prepare_prediction_features(raw_features: pd.DataFrame, mapper_name: str) -> pd.DataFrame:
    result = raw_features.copy()
    result["F_DATE"] = result["F_DATE"].astype(str).str.strip()

    if mapper_name == "legacy":
        result["WEATHER_TYPE"] = result["WEATHER_TYPE"].apply(
            lambda x: WeatherType.get_weather_by_name_legacy(str(x)).value
        )
    else:
        result["WEATHER_TYPE"] = result["WEATHER_TYPE"].apply(
            lambda x: WeatherType.get_weather_by_name(str(x)).value
        )
        result["WEATHER_SEVERITY"] = result["WEATHER_TYPE"].apply(get_weather_severity)

    return result


def evaluate_strategy(metric_type: str, mapper_name: str, factors: List[str]) -> Dict:
    config = load_yaml_config("model_config_daily.yaml", default_daily=True) or {}
    config = dict(config)
    config["factors"] = factors

    raw_df = read_line_daily_flow_history(metric_type)
    df = prepare_training_data(raw_df, mapper_name)
    weather_features = prepare_prediction_features(fetch_actual_weather_features(), mapper_name)

    line_name_map = {
        str(row["F_LINENO"]).zfill(2): row["F_LINENAME"]
        for _, row in df[["F_LINENO", "F_LINENAME"]].drop_duplicates().iterrows()
    }
    actual_2025 = df[(df["F_DATE"] >= PREDICT_START_DATE) & (df["F_DATE"] <= PREDICT_END_DATE)].copy()

    lines = sorted(df["F_LINENO"].unique().tolist())
    lines = filter_lines_by_config(lines, config, "xianwangxianlu_daily_predict")

    overall_accuracies: List[float] = []
    line_breakdown: List[Dict] = []

    with TemporaryDirectory(prefix=f"weather_eval_{mapper_name}_") as temp_dir:
        predictor = KNNFlowPredictor(temp_dir, TRAIN_END_DATE, config)

        for line in lines:
            line_data = get_line_data(df, line, TRAIN_END_DATE, need_train=True)
            if line_data.empty:
                continue

            _, _, error = predictor.train(line_data, line, model_version=TRAIN_END_DATE)
            if error:
                continue

            predictions, error = predictor.predict(
                line_data=line_data,
                line_no=line,
                predict_start_date=PREDICT_START_DATE,
                days=PREDICT_DAYS,
                model_version=TRAIN_END_DATE,
                factor_df=weather_features,
            )
            if error or predictions is None:
                continue

            line_actual = actual_2025[actual_2025["F_LINENO"] == line].sort_values("F_DATE")
            actual_map = dict(zip(line_actual["F_DATE"], line_actual["F_KLCOUNT"]))

            line_accs: List[float] = []
            for offset, predicted in enumerate(predictions):
                date_str = (pd.Timestamp(PREDICT_START_DATE) + pd.Timedelta(days=offset)).strftime("%Y%m%d")
                actual = actual_map.get(date_str)
                acc = calculate_accuracy(predicted, actual)
                if acc is not None:
                    line_accs.append(acc)
                    overall_accuracies.append(acc)

            line_breakdown.append({
                "line_no": line,
                "line_name": line_name_map.get(line, line),
                "avg_accuracy": round(mean(line_accs), 2) if line_accs else None,
                "sample_count": len(line_accs),
            })

    line_breakdown.sort(key=lambda item: item["avg_accuracy"] if item["avg_accuracy"] is not None else -999, reverse=True)
    return {
        "strategy": mapper_name,
        "factors": factors,
        "avg_accuracy": round(mean(overall_accuracies), 2) if overall_accuracies else None,
        "sample_count": len(overall_accuracies),
        "line_breakdown": line_breakdown,
    }


def main() -> None:
    result = {
        "metric_type": "F_PKLCOUNT",
        "period": f"{PREDICT_START_DATE}-{PREDICT_END_DATE}",
        "legacy_weather": evaluate_strategy(
            metric_type="F_PKLCOUNT",
            mapper_name="legacy",
            factors=[
                "F_WEEK", "F_HOLIDAYTYPE", "F_HOLIDAYDAYS",
                "F_HOLIDAYWHICHDAY", "F_DAYOFWEEK", "WEATHER_TYPE", "F_YEAR",
            ],
        ),
        "improved_weather": evaluate_strategy(
            metric_type="F_PKLCOUNT",
            mapper_name="improved",
            factors=[
                "F_WEEK", "F_HOLIDAYTYPE", "F_HOLIDAYDAYS",
                "F_HOLIDAYWHICHDAY", "F_DAYOFWEEK", "WEATHER_TYPE",
                "WEATHER_SEVERITY", "F_YEAR",
            ],
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
