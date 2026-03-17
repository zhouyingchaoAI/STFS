#!/usr/bin/env python3
"""
评估 2025 年法定节假日的机器学习、专家系统及融合结果准确率。

默认评估:
- 客流类型: 线网线路
- 指标: F_PKLCOUNT
- 模型: KNN 日预测
- 融合: alpha * ML + (1 - alpha) * Expert
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config_utils import load_yaml_config
from db_utils import read_line_daily_flow_history, fetch_holiday_features
from enknn_model import KNNFlowPredictor
from holiday_predict_utils import predict_line_flow
from predict_daily import preprocess_daily_data, get_line_data
from script.get_clander import HolidayGenerator
from weather_enum import WeatherType


HOLIDAY_NAMES = ["元旦", "春节", "清明节", "劳动节", "端午节", "国庆节"]
HOLIDAY_TYPE_IDS = {
    "元旦": 1,
    "春节": 2,
    "清明节": 3,
    "劳动节": 4,
    "端午节": 5,
    "中秋节": 6,
    "国庆节": 7,
}
LINE_NO_NAME_MAP = {
    "00": "线网",
    "01": "1号线",
    "02": "2号线",
    "03": "3号线",
    "04": "4号线",
    "05": "5号线",
    "06": "6号线",
    "31": "西环线",
    "60": "磁浮快线",
    "83": "互联网平台",
}


def get_2025_holiday_periods() -> List[Tuple[str, str, str]]:
    generator = HolidayGenerator()
    periods = []
    for name, spans in generator.holiday_arrangements[2025].items():
        if name in HOLIDAY_NAMES:
            start, end = spans[0]
            periods.append((name, start.replace("-", ""), end.replace("-", "")))
    periods.sort(key=lambda item: item[1])
    return periods


def build_ml_predictions(metric_type: str, predict_start: str, predict_end: str) -> Dict[str, Dict[str, int]]:
    config = load_yaml_config("model_config_daily.yaml", default_daily=True)
    version = predict_start
    days = (datetime.strptime(predict_end, "%Y%m%d") - datetime.strptime(predict_start, "%Y%m%d")).days + 1

    df = read_line_daily_flow_history(metric_type)
    df, line_name_map = preprocess_daily_data(df)
    lines = sorted(df["F_LINENO"].unique().tolist())

    holiday_features = fetch_holiday_features(predict_start, days)
    holiday_features["WEATHER_TYPE"] = holiday_features["WEATHER_TYPE"].apply(
        lambda x: WeatherType.get_weather_by_name(str(x)).value
    )

    results: Dict[str, Dict[str, int]] = {}
    with TemporaryDirectory(prefix="holiday_fusion_models_") as temp_dir:
        predictor = KNNFlowPredictor(temp_dir, version, config or {})
        for line in lines:
            line_data = get_line_data(df, line, predict_start, need_train=True)
            if line_data.empty:
                continue

            _, _, error = predictor.train(line_data, line, model_version=version)
            if error:
                continue

            predictions, error = predictor.predict(
                line_data,
                line,
                predict_start,
                days,
                model_version=version,
                factor_df=holiday_features,
            )
            if error or predictions is None:
                continue

            line_name = line_name_map.get(line, LINE_NO_NAME_MAP.get(line, line))
            results[line_name] = {}
            for offset, value in enumerate(predictions):
                date_str = (datetime.strptime(predict_start, "%Y%m%d") + pd.Timedelta(days=offset)).strftime("%Y%m%d")
                results[line_name][date_str] = int(value)

    return results


def build_expert_predictions(metric_type: str, predict_start: str, predict_end: str) -> Dict[str, Dict[str, int]]:
    result = predict_line_flow(metric_type, predict_start, predict_end, history_years=2)
    grouped: Dict[str, Dict[str, int]] = {}
    for row in result.get("predictions", []):
        line_name = str(row.get("线路名称", "未知"))
        predict_date = str(row.get("预测日期"))
        flow = int(row.get("预测客流", 0))
        grouped.setdefault(line_name, {})[predict_date] = flow
    return grouped


def build_actuals(metric_type: str, predict_start: str, predict_end: str) -> Dict[str, Dict[str, int]]:
    df = read_line_daily_flow_history(metric_type)
    df, line_name_map = preprocess_daily_data(df)
    mask = (df["F_DATE"] >= predict_start) & (df["F_DATE"] <= predict_end)
    df = df.loc[mask].copy()

    actuals: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        line_name = str(row["F_LINENAME"]) if pd.notna(row["F_LINENAME"]) else line_name_map.get(str(row["F_LINENO"]), str(row["F_LINENO"]))
        actuals.setdefault(line_name, {})[str(row["F_DATE"])] = int(row["F_KLCOUNT"])
    return actuals


def calculate_accuracy(predicted: float, actual: float) -> float | None:
    if actual > 0:
        return round((1 - abs(predicted - actual) / actual) * 100, 2)
    return None


def resolve_policy_alpha(
    holiday_type_id: int,
    which_day: int,
    holiday_policy: Dict[int, float],
    spring_festival_policy: Dict[str, float] | None = None,
    default_alpha: float = 0.9,
) -> float:
    if holiday_type_id == 2 and spring_festival_policy:
        if which_day <= 3:
            return spring_festival_policy.get("early", holiday_policy.get(2, default_alpha))
        if which_day <= 5:
            return spring_festival_policy.get("mid", holiday_policy.get(2, default_alpha))
        return spring_festival_policy.get("late", holiday_policy.get(2, default_alpha))
    return holiday_policy.get(holiday_type_id, default_alpha)


def collect_holiday_prediction_dataset(metric_type: str = "F_PKLCOUNT") -> List[Dict]:
    dataset: List[Dict] = []
    periods = get_2025_holiday_periods()

    for holiday_name, start, end in periods:
        holiday_type_id = HOLIDAY_TYPE_IDS[holiday_name]
        ml_preds = build_ml_predictions(metric_type, start, end)
        expert_preds = build_expert_predictions(metric_type, start, end)
        actuals = build_actuals(metric_type, start, end)
        start_dt = datetime.strptime(start, "%Y%m%d")

        all_lines = sorted(set(actuals) | set(ml_preds) | set(expert_preds))
        for line_name in all_lines:
            dates = sorted(set(actuals.get(line_name, {})) | set(ml_preds.get(line_name, {})) | set(expert_preds.get(line_name, {})))
            for date_str in dates:
                actual = actuals.get(line_name, {}).get(date_str)
                ml_pred = ml_preds.get(line_name, {}).get(date_str)
                expert_pred = expert_preds.get(line_name, {}).get(date_str)
                which_day = (datetime.strptime(date_str, "%Y%m%d") - start_dt).days + 1
                dataset.append({
                    "holiday_name": holiday_name,
                    "holiday_type_id": holiday_type_id,
                    "start": start,
                    "end": end,
                    "line_name": line_name,
                    "date_str": date_str,
                    "which_day": which_day,
                    "actual": actual,
                    "ml_pred": ml_pred,
                    "expert_pred": expert_pred,
                })

    return dataset


def evaluate_fusion(metric_type: str = "F_PKLCOUNT", dataset: List[Dict] | None = None) -> Dict:
    periods = get_2025_holiday_periods()
    dataset = dataset or collect_holiday_prediction_dataset(metric_type)
    alpha_grid = [round(x / 10, 1) for x in range(0, 11)]
    scores = {alpha: [] for alpha in alpha_grid}
    ml_scores: List[float] = []
    expert_scores: List[float] = []
    holiday_breakdown: List[Dict] = []

    for holiday_name, start, end in periods:
        ml_accs: List[float] = []
        expert_accs: List[float] = []
        fusion_accs = {alpha: [] for alpha in alpha_grid}

        for row in dataset:
            if row["holiday_name"] != holiday_name:
                continue
            actual = row["actual"]
            ml_pred = row["ml_pred"]
            expert_pred = row["expert_pred"]

            if actual is None:
                continue

            if ml_pred is not None:
                acc = calculate_accuracy(ml_pred, actual)
                if acc is not None:
                    ml_accs.append(acc)
                    ml_scores.append(acc)

            if expert_pred is not None:
                acc = calculate_accuracy(expert_pred, actual)
                if acc is not None:
                    expert_accs.append(acc)
                    expert_scores.append(acc)

            if ml_pred is None or expert_pred is None:
                continue

            for alpha in alpha_grid:
                fused_pred = alpha * ml_pred + (1 - alpha) * expert_pred
                acc = calculate_accuracy(fused_pred, actual)
                if acc is not None:
                    fusion_accs[alpha].append(acc)
                    scores[alpha].append(acc)

        holiday_breakdown.append({
            "holiday": holiday_name,
            "start": start,
            "end": end,
            "ml_avg_accuracy": round(mean(ml_accs), 2) if ml_accs else None,
            "expert_avg_accuracy": round(mean(expert_accs), 2) if expert_accs else None,
            "fusion_avg_accuracy": {
                str(alpha): round(mean(values), 2) if values else None
                for alpha, values in fusion_accs.items()
            },
            "sample_count": len(ml_accs),
        })

    alpha_ranking = sorted(
        [
            {
                "alpha": alpha,
                "avg_accuracy": round(mean(values), 2) if values else None,
                "count": len(values),
            }
            for alpha, values in scores.items()
        ],
        key=lambda item: item["avg_accuracy"] if item["avg_accuracy"] is not None else -999,
        reverse=True,
    )

    return {
        "metric_type": metric_type,
        "ml_avg_accuracy": round(mean(ml_scores), 2) if ml_scores else None,
        "expert_avg_accuracy": round(mean(expert_scores), 2) if expert_scores else None,
        "alpha_ranking": alpha_ranking,
        "best_alpha": alpha_ranking[0]["alpha"] if alpha_ranking else None,
        "holiday_breakdown": holiday_breakdown,
    }


def evaluate_policy(
    holiday_policy: Dict[int, float],
    spring_festival_policy: Dict[str, float] | None = None,
    metric_type: str = "F_PKLCOUNT",
    dataset: List[Dict] | None = None,
) -> Dict:
    periods = get_2025_holiday_periods()
    dataset = dataset or collect_holiday_prediction_dataset(metric_type)
    accuracies: List[float] = []
    holiday_breakdown: List[Dict] = []

    for holiday_name, start, end in periods:
        holiday_type_id = HOLIDAY_TYPE_IDS[holiday_name]
        holiday_accs: List[float] = []

        for row in dataset:
            if row["holiday_name"] != holiday_name:
                continue
            actual = row["actual"]
            ml_pred = row["ml_pred"]
            expert_pred = row["expert_pred"]
            if actual is None or ml_pred is None or expert_pred is None:
                continue

            alpha = resolve_policy_alpha(holiday_type_id, row["which_day"], holiday_policy, spring_festival_policy)
            fused_pred = alpha * ml_pred + (1 - alpha) * expert_pred
            acc = calculate_accuracy(fused_pred, actual)
            if acc is not None:
                holiday_accs.append(acc)
                accuracies.append(acc)

        holiday_breakdown.append({
            "holiday": holiday_name,
            "holiday_type_id": holiday_type_id,
            "avg_accuracy": round(mean(holiday_accs), 2) if holiday_accs else None,
            "sample_count": len(holiday_accs),
        })

    return {
        "metric_type": metric_type,
        "holiday_policy": holiday_policy,
        "spring_festival_policy": spring_festival_policy,
        "avg_accuracy": round(mean(accuracies), 2) if accuracies else None,
        "sample_count": len(accuracies),
        "holiday_breakdown": holiday_breakdown,
    }


def search_segmented_policy(metric_type: str = "F_PKLCOUNT", dataset: List[Dict] | None = None) -> Dict:
    alpha_grid = [round(x / 10, 1) for x in range(0, 11)]
    dataset = dataset or collect_holiday_prediction_dataset(metric_type)
    base_policy = {
        1: 1.0,
        2: 1.0,
        3: 1.0,
        4: 0.1,
        5: 0.7,
        6: 0.9,
        7: 0.5,
    }

    best_result = None
    for early in alpha_grid:
        for mid in alpha_grid:
            for late in alpha_grid:
                result = evaluate_policy(
                    holiday_policy=base_policy,
                    spring_festival_policy={"early": early, "mid": mid, "late": late},
                    metric_type=metric_type,
                    dataset=dataset,
                )
                result["spring_festival_policy"] = {"early": early, "mid": mid, "late": late}
                if best_result is None or (result["avg_accuracy"] or -999) > (best_result["avg_accuracy"] or -999):
                    best_result = result

    return best_result


if __name__ == "__main__":
    dataset = collect_holiday_prediction_dataset()
    result = {
        "global_alpha_search": evaluate_fusion(dataset=dataset),
        "segmented_policy_search": search_segmented_policy(dataset=dataset),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
