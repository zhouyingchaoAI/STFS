# holiday_predict_utils.py
# 节假日客流预测工具模块：基于历年最优增长率的预测算法
"""
该模块实现基于历年同期增长率的节假日客流预测算法，包括：
- 线路客流预测
- 车站客流预测
- 基期计算
- 历年最优增长率提取
- 准确率对比

核心预测公式：
    预测客流 = 基期日均 × (1 + 最优增长率 / 100)
    
准确率计算：
    准确率 = (1 - |预测值 - 实际值| / 实际值) × 100%
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
from typing import Dict, List, Optional, Tuple, Any
from logger_config import get_logger
from db_pool import get_db_connection
from db_utils import fix_dataframe_encoding, QUERY_START_DATE

logger = get_logger(__name__)

# 指标类型的中文名称映射
METRIC_NAMES = {
    'F_PKLCOUNT': '客流量',
    'F_ENTRANCE': '进站人数',
    'F_EXIT': '出站人数',
    'F_TRANSFER': '换乘人数',
    'F_BOARD_ALIGHT': '乘降量'
}

# 指标字段映射
METRIC_FIELD_MAPPING_LINE = {
    "F_PKLCOUNT": "L.F_KLCOUNT",
    "F_ENTRANCE": "L.ENTRY_NUM",
    "F_EXIT": "L.EXIT_NUM",
    "F_TRANSFER": "L.CHANGE_NUM",
    "F_BOARD_ALIGHT": "L.FLOW_NUM"
}

METRIC_FIELD_MAPPING_STATION = {
    "F_PKLCOUNT": "S.PASSENGER_NUM",
    "F_ENTRANCE": "S.ENTRY_NUM",
    "F_EXIT": "S.EXIT_NUM",
    "F_TRANSFER": "S.CHANGE_NUM",
    "F_BOARD_ALIGHT": "S.FLOW_NUM"
}


def read_line_daily_flow_history_range(metric_type: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从数据库读取线路日客流历史数据（指定日期范围）
    
    参数:
        metric_type: 指标类型 (F_PKLCOUNT, F_ENTRANCE 等)
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        
    返回:
        日客流数据 DataFrame
    """
    if metric_type not in METRIC_FIELD_MAPPING_LINE:
        raise ValueError(f"无效的 metric_type: {metric_type}")
    
    selected_field = METRIC_FIELD_MAPPING_LINE[metric_type]
    
    query = f"""
    SELECT 
        L.ID, L.F_DATE, L.F_LB, L.F_LINENO, L.F_LINENAME,
        {selected_field} AS F_KLCOUNT,
        L.CREATETIME, L.CREATOR,
        CC.F_YEAR, CC.F_DAYOFWEEK, CC.F_WEEK, CC.F_HOLIDAYTYPE,
        CC.F_HOLIDAYDAYS, CC.F_HOLIDAYWHICHDAY, CC.COVID19, CC.F_WEATHER,
        W.F_TQQK AS WEATHER_TYPE
    FROM dbo.LineDailyFlowHistory AS L
    LEFT JOIN dbo.CalendarHistory AS CC ON L.F_DATE = CC.F_DATE
    LEFT JOIN dbo.WeatherHistory AS W ON L.F_DATE = W.F_DATE
    WHERE L.CREATOR = 'chency' 
        AND L.F_DATE >= '{start_date}'
        AND L.F_DATE <= '{end_date}'
    ORDER BY L.F_DATE, L.F_LINENO
    """
    
    try:
        # 不设置 charset 参数，让 pymssql 正确处理 NVARCHAR 字段的中文
        with get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        df = fix_dataframe_encoding(df)
        logger.debug(f"线路日客流数据读取完成: {start_date}-{end_date}, 共 {len(df)} 条")
        return df
    except Exception as e:
        logger.error(f"读取线路日客流数据失败: {e}")
        raise RuntimeError(f"数据库读取失败: {e}")


def read_station_daily_flow_history_range(metric_type: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从数据库读取车站日客流历史数据（指定日期范围）
    
    参数:
        metric_type: 指标类型
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        
    返回:
        车站日客流数据 DataFrame
    """
    from db_utils import STATION_FILTER_NAMES
    
    if metric_type not in METRIC_FIELD_MAPPING_STATION:
        raise ValueError(f"无效的 metric_type: {metric_type}")
    
    selected_field = METRIC_FIELD_MAPPING_STATION[metric_type]
    
    # 车站过滤条件
    station_filter = ""
    if STATION_FILTER_NAMES:
        names = ", ".join([f"N'{name}'" for name in STATION_FILTER_NAMES])
        station_filter = f"AND S.STATION_NAME IN ({names})"
    
    query = f"""
    SELECT 
        MIN(S.ID),
        REPLACE(S.SQUAD_DATE, '-', '') AS F_DATE,
        S.STATION_NAME AS F_LINENO,
        S.STATION_NAME AS F_LINENAME,
        SUM({selected_field}) AS F_KLCOUNT,
        MIN(CC.F_YEAR) AS F_YEAR,
        MIN(CC.F_DAYOFWEEK) AS F_DAYOFWEEK,
        MIN(CC.F_WEEK) AS F_WEEK,
        MIN(CC.F_HOLIDAYTYPE) AS F_HOLIDAYTYPE,
        MIN(CC.F_HOLIDAYDAYS) AS F_HOLIDAYDAYS,
        MIN(CC.F_HOLIDAYWHICHDAY) AS F_HOLIDAYWHICHDAY,
        MIN(CC.COVID19) AS COVID19,
        MIN(CC.F_WEATHER) AS F_WEATHER,
        MIN(W.F_TQQK) AS WEATHER_TYPE
    FROM [StationFlowPredict].[dbo].[STATION_FLOW_HISTORY] AS S
    LEFT JOIN master.dbo.CalendarHistory AS CC ON REPLACE(S.SQUAD_DATE, '-', '') = CC.F_DATE
    LEFT JOIN master.dbo.WeatherHistory AS W ON REPLACE(S.SQUAD_DATE, '-', '') = W.F_DATE
    WHERE REPLACE(S.SQUAD_DATE, '-', '') >= '{start_date}'
        AND REPLACE(S.SQUAD_DATE, '-', '') <= '{end_date}'
        {station_filter}
    GROUP BY S.STATION_NAME, REPLACE(S.SQUAD_DATE, '-', '')
    ORDER BY REPLACE(S.SQUAD_DATE, '-', ''), S.STATION_NAME
    """
    
    try:
        # 不设置 charset 参数，让 pymssql 正确处理 NVARCHAR 字段的中文
        with get_db_connection() as conn:
            df = pd.read_sql(query, conn)
        df = fix_dataframe_encoding(df)
        
        # 聚合处理
        sum_fields = ['F_KLCOUNT']
        first_fields = ['F_LINENO', 'F_LINENAME', 'F_WEEK', 'F_HOLIDAYTYPE', 'F_HOLIDAYDAYS',
                       'WEATHER_TYPE', 'F_YEAR', 'F_DAYOFWEEK', 'F_HOLIDAYWHICHDAY', 'COVID19', 'F_WEATHER']
        
        # 只使用存在的列
        sum_fields = [f for f in sum_fields if f in df.columns]
        first_fields = [f for f in first_fields if f in df.columns]
        
        if not df.empty and 'F_DATE' in df.columns and 'F_LINENAME' in df.columns:
            grouped = df.groupby(['F_DATE', 'F_LINENAME'], as_index=False).agg(
                {**{f: 'sum' for f in sum_fields}, **{f: 'first' for f in first_fields}}
            )
            grouped = fix_dataframe_encoding(grouped)
            logger.debug(f"车站日客流数据读取完成: {start_date}-{end_date}, 共 {len(grouped)} 条")
            return grouped
        
        return df
    except Exception as e:
        logger.error(f"读取车站日客流数据失败: {e}")
        raise RuntimeError(f"数据库读取失败: {e}")


def calculate_base_period(predict_start: str) -> Tuple[str, str, int, int]:
    """
    计算基期（上一个完整月）
    
    参数:
        predict_start: 预测开始日期 (YYYYMMDD格式)
        
    返回:
        (基期开始日期, 基期结束日期, 基期年份, 基期月份)
    """
    predict_start_date = datetime.strptime(predict_start, '%Y%m%d')
    predict_year = predict_start_date.year
    predict_month = predict_start_date.month
    
    # 基期是上一个月
    base_month = predict_month - 1
    base_year = predict_year
    if base_month < 1:
        base_month = 12
        base_year -= 1
    
    # 基期开始和结束
    base_start = f"{base_year}{str(base_month).zfill(2)}01"
    last_day = calendar.monthrange(base_year, base_month)[1]
    base_end = f"{base_year}{str(base_month).zfill(2)}{str(last_day).zfill(2)}"
    
    return base_start, base_end, base_year, base_month


def calculate_history_periods(
    predict_start: str,
    predict_end: str,
    history_year: int,
    predict_year: int,
    predict_month: int,
    custom_config: Optional[Dict] = None
) -> Optional[Dict[str, str]]:
    """
    计算历年参考期和基期
    
    参数:
        predict_start: 预测开始日期
        predict_end: 预测结束日期
        history_year: 历史年份
        predict_year: 预测年份
        predict_month: 预测月份
        custom_config: 自定义配置 (可选)
        
    返回:
        包含 history_start, history_end, history_base_start, history_base_end 的字典
        如果日期无效返回 None
    """
    predict_start_date = datetime.strptime(predict_start, '%Y%m%d')
    predict_end_date = datetime.strptime(predict_end, '%Y%m%d')
    i = predict_year - history_year
    
    if custom_config:
        # 使用自定义配置
        ref_start_str = custom_config['ref_start'].replace('-', '')
        ref_end_str = custom_config['ref_end'].replace('-', '')
        base_start_str = custom_config['base_start'].replace('-', '')
        base_end_str = custom_config['base_end'].replace('-', '')
        
        # 验证日期
        try:
            ref_start_date = datetime.strptime(ref_start_str, '%Y%m%d')
            ref_end_date = datetime.strptime(ref_end_str, '%Y%m%d')
            if ref_start_date > ref_end_date:
                logger.warning(f"{history_year}年参考期开始日期晚于结束日期")
                return None
                
            base_start_date = datetime.strptime(base_start_str, '%Y%m%d')
            base_end_date = datetime.strptime(base_end_str, '%Y%m%d')
            if base_start_date > base_end_date:
                logger.warning(f"{history_year}年基期开始日期晚于结束日期")
                return None
                
            return {
                'history_start': ref_start_str,
                'history_end': ref_end_str,
                'history_base_start': base_start_str,
                'history_base_end': base_end_str
            }
        except ValueError as e:
            logger.error(f"{history_year}年日期格式错误: {e}")
            return None
    else:
        # 使用默认计算
        history_start_date = predict_start_date - relativedelta(years=i)
        history_end_date = predict_end_date - relativedelta(years=i)
        history_start = history_start_date.strftime('%Y%m%d')
        history_end = history_end_date.strftime('%Y%m%d')
        
        # 历年基期
        history_base_month = predict_month - 1
        history_base_year = history_year
        if history_base_month < 1:
            history_base_month = 12
            history_base_year -= 1
        
        history_base_start = f"{history_base_year}{str(history_base_month).zfill(2)}01"
        history_last_day = calendar.monthrange(history_base_year, history_base_month)[1]
        history_base_end = f"{history_base_year}{str(history_base_month).zfill(2)}{str(history_last_day).zfill(2)}"
        
        return {
            'history_start': history_start,
            'history_end': history_end,
            'history_base_start': history_base_start,
            'history_base_end': history_base_end
        }


def calculate_growth_rate(current_value: float, base_value: float) -> float:
    """
    计算增长率
    
    参数:
        current_value: 当前值
        base_value: 基期值
        
    返回:
        增长率（百分比）
    """
    if base_value > 0:
        return round(((current_value - base_value) / base_value) * 100, 2)
    return 0.0


def calculate_accuracy(predicted: float, actual: float) -> Optional[float]:
    """
    计算预测准确率
    
    参数:
        predicted: 预测值
        actual: 实际值
        
    返回:
        准确率（百分比），如果实际值为0返回None
    """
    if actual > 0:
        error_rate = abs(predicted - actual) / actual
        return round((1 - error_rate) * 100, 2)
    return None


def predict_flow(
    metric_type: str,
    predict_start: str,
    predict_end: str,
    history_years: int = 2,
    custom_configs: Optional[Dict] = None,
    data_source: str = 'line',
    read_data_func=None
) -> Dict[str, Any]:
    """
    基于历年最优增长率预测客流
    
    参数:
        metric_type: 指标类型 (F_PKLCOUNT等)
        predict_start: 预测开始日期 (YYYYMMDD)
        predict_end: 预测结束日期 (YYYYMMDD)
        history_years: 参考历史年数
        custom_configs: 自定义各年配置
        data_source: 数据源类型 ('line'线路 或 'station'车站)
        read_data_func: 数据读取函数 (需接受 metric_type, start_date, end_date 参数)
        
    返回:
        预测结果字典，包含 predictions, has_actual, metric_name 等
    """
    if read_data_func is None:
        # 使用本地的带日期范围的函数
        if data_source == 'station':
            read_data_func = read_station_daily_flow_history_range
        else:
            read_data_func = read_line_daily_flow_history_range
    
    metric_name = METRIC_NAMES.get(metric_type, '客流量')
    is_station = data_source == 'station'
    id_field = '车站ID' if is_station else '线路编号'
    name_field = '车站名称' if is_station else '线路名称'
    
    logger.info(f"开始{'车站' if is_station else '线路'}预测: {predict_start} - {predict_end}, 参考{history_years}年")
    
    # 解析预测日期
    predict_start_date = datetime.strptime(predict_start, '%Y%m%d')
    predict_end_date = datetime.strptime(predict_end, '%Y%m%d')
    predict_year = predict_start_date.year
    predict_month = predict_start_date.month
    predict_days = (predict_end_date - predict_start_date).days + 1
    
    # 计算基期
    base_start, base_end, base_year, base_month = calculate_base_period(predict_start)
    
    # 查询基期数据
    try:
        df_base = read_data_func(metric_type, base_start, base_end)
        if df_base.empty:
            return {
                'success': False,
                'error': f'基期（{base_year}年{base_month}月）数据不存在'
            }
        df_base = df_base.rename(columns={'F_KLCOUNT': metric_name})
    except Exception as e:
        logger.error(f"查询基期数据失败: {e}")
        return {'success': False, 'error': f'查询基期数据失败: {str(e)}'}
    
    # 计算基期日均
    base_avg = df_base.groupby(['F_LINENO', 'F_LINENAME'])[metric_name].mean().reset_index()
    base_avg.columns = ['F_LINENO', 'F_LINENAME', '基期日均']
    
    if base_avg.empty:
        return {'success': False, 'error': '基期数据处理失败，无法计算日均客流'}
    
    logger.info(f"基期日均计算完成: {len(base_avg)}条记录")
    
    # 查询历年同期数据
    all_years_data = []
    history_details = []
    ref_start_dates = {}  # 用于计算"第几天"
    
    for i in range(1, history_years + 1):
        history_year = predict_year - i
        
        # 获取自定义配置
        custom_config = None
        if custom_configs and str(history_year) in custom_configs:
            custom_config = custom_configs[str(history_year)]
        
        # 计算历年期间
        periods = calculate_history_periods(
            predict_start, predict_end, history_year,
            predict_year, predict_month, custom_config
        )
        
        if periods is None:
            continue
        
        history_start = periods['history_start']
        history_end = periods['history_end']
        history_base_start = periods['history_base_start']
        history_base_end = periods['history_base_end']
        
        # 记录参考期开始日期
        ref_start_dates[history_year] = datetime.strptime(history_start, '%Y%m%d')
        
        try:
            # 查询历年参考期数据
            df_history = read_data_func(metric_type, history_start, history_end)
            if df_history.empty:
                logger.warning(f"{history_year}年参考期数据为空")
                continue
            df_history = df_history.rename(columns={'F_KLCOUNT': metric_name})
            
            # 查询历年基期
            df_history_base = read_data_func(metric_type, history_base_start, history_base_end)
            if df_history_base.empty:
                logger.warning(f"{history_year}年基期数据为空")
                continue
            df_history_base = df_history_base.rename(columns={'F_KLCOUNT': metric_name})
            
            # 计算历年基期日均
            history_base_avg = df_history_base.groupby(['F_LINENO', 'F_LINENAME'])[metric_name].mean().reset_index()
            history_base_avg.columns = ['F_LINENO', 'F_LINENAME', f'基期日均_{history_year}']
            
            # 标记年份
            if 'F_YEAR' in df_history.columns:
                df_history['年份'] = df_history['F_YEAR']
            else:
                df_history['年份'] = history_year
            
            # 合并历年基期
            df_history = df_history.merge(history_base_avg, on=['F_LINENO', 'F_LINENAME'], how='inner')
            
            # 计算增长率
            base_col = f'基期日均_{history_year}'
            df_history['增长率'] = df_history.apply(
                lambda row: calculate_growth_rate(row[metric_name], row[base_col]),
                axis=1
            )
            
            all_years_data.append(df_history)
            logger.info(f"{history_year}年数据: 参考期{len(df_history)}行")
            
            # 记录详情
            history_details.append({
                'year': int(history_year),
                'ref_period': f"{history_start[:4]}-{history_start[4:6]}-{history_start[6:]} 至 {history_end[:4]}-{history_end[4:6]}-{history_end[6:]}",
                'base_period': f"{history_base_start[:4]}-{history_base_start[4:6]}-{history_base_start[6:]} 至 {history_base_end[:4]}-{history_base_end[4:6]}-{history_base_end[6:]}",
                'line_stats': []
            })
            
            # 统计信息
            for line_no in df_history['F_LINENO'].unique():
                line_data = df_history[df_history['F_LINENO'] == line_no]
                history_details[-1]['line_stats'].append({
                    'line_no': str(line_no) if is_station else int(line_no),
                    'line_name': str(line_data['F_LINENAME'].iloc[0]),
                    'base_avg': int(line_data[base_col].iloc[0]),
                    'avg_growth': round(float(line_data['增长率'].mean()), 2),
                    'max_growth': round(float(line_data['增长率'].max()), 2)
                })
                
        except Exception as e:
            logger.error(f"{history_year}年数据处理失败: {e}")
            continue
    
    if not all_years_data:
        return {'success': False, 'error': '没有找到历年同期数据'}
    
    # 合并所有年份数据
    df_all_history = pd.concat(all_years_data, ignore_index=True)
    logger.info(f"合并完成: 共{len(df_all_history)}行")
    
    # 为历年数据标记"第几天"
    def calculate_day_num(row):
        try:
            date_str = str(row['F_DATE'])
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            
            for history_year, ref_start in ref_start_dates.items():
                day_diff = (date_obj - ref_start).days + 1
                if 1 <= day_diff <= predict_days:
                    return day_diff
            return None
        except:
            return None
    
    for line_no in df_all_history['F_LINENO'].unique():
        mask = df_all_history['F_LINENO'] == line_no
        df_all_history.loc[mask, '第几天'] = df_all_history[mask].apply(calculate_day_num, axis=1)
    
    # 生成预测
    predictions = []
    
    for line_no in base_avg['F_LINENO'].unique():
        line_name = base_avg[base_avg['F_LINENO'] == line_no]['F_LINENAME'].iloc[0]
        base_daily_avg = base_avg[base_avg['F_LINENO'] == line_no]['基期日均'].iloc[0]
        
        line_history = df_all_history[df_all_history['F_LINENO'] == line_no].copy()
        if line_history.empty:
            continue
        
        for day_num in range(1, predict_days + 1):
            day_data = line_history[line_history['第几天'] == day_num]
            
            if day_data.empty:
                continue
            
            # 取最大增长率
            max_growth_rate = day_data['增长率'].max()
            max_year = day_data.loc[day_data['增长率'].idxmax(), '年份']
            
            # 预测客流
            predicted_flow = base_daily_avg * (1 + max_growth_rate / 100)
            
            # 生成预测日期
            predict_date_obj = predict_start_date + timedelta(days=day_num - 1)
            predict_date_str = predict_date_obj.strftime('%Y%m%d')
            
            predictions.append({
                id_field: str(line_no) if is_station else int(line_no),
                name_field: str(line_name),
                '预测日期': predict_date_str,
                '第几天': int(day_num),
                '基期日均': int(base_daily_avg),
                '最优增长率': round(float(max_growth_rate), 2),
                '最优来源年份': int(max_year),
                '预测客流': int(predicted_flow)
            })
    
    logger.info(f"预测计算完成: 共{len(predictions)}条记录")
    
    # 查询实际数据
    has_actual = False
    matched_count = 0
    
    try:
        df_actual = read_data_func(metric_type, predict_start, predict_end)
        
        if not df_actual.empty:
            df_actual = df_actual.rename(columns={'F_KLCOUNT': metric_name})
            if is_station:
                df_actual['F_LINENO'] = df_actual['F_LINENO'].astype(str).str.strip()
            else:
                df_actual['F_LINENO'] = df_actual['F_LINENO'].astype(int)
            df_actual['F_DATE'] = df_actual['F_DATE'].astype(str).str.strip()
            
            # 匹配实际数据
            for pred in predictions:
                if is_station:
                    pred_line = str(pred[id_field]).strip()
                else:
                    pred_line = int(pred[id_field])
                pred_date = str(pred['预测日期']).strip()
                
                actual_row = df_actual[
                    (df_actual['F_LINENO'] == pred_line) &
                    (df_actual['F_DATE'] == pred_date)
                ]
                
                if not actual_row.empty:
                    actual_flow = int(actual_row[metric_name].iloc[0])
                    pred['实际客流'] = actual_flow
                    pred['准确率'] = calculate_accuracy(pred['预测客流'], actual_flow)
                    pred['误差'] = pred['预测客流'] - actual_flow
                    matched_count += 1
                else:
                    pred['实际客流'] = None
                    pred['准确率'] = None
                    pred['误差'] = None
            
            has_actual = matched_count > 0
            logger.info(f"准确率对比完成: {matched_count}/{len(predictions)}条匹配")
            
    except Exception as e:
        logger.warning(f"查询实际数据失败: {e}")
        for pred in predictions:
            pred['实际客流'] = None
            pred['准确率'] = None
            pred['误差'] = None
    
    # 格式化基期和预测期显示
    base_period_str = f"{base_start[:4]}-{base_start[4:6]}-{base_start[6:]} 至 {base_end[:4]}-{base_end[4:6]}-{base_end[6:]}"
    predict_period_str = f"{predict_start[:4]}-{predict_start[4:6]}-{predict_start[6:]} 至 {predict_end[:4]}-{predict_end[4:6]}-{predict_end[6:]}"
    
    return {
        'success': True,
        'predictions': predictions,
        'has_actual': has_actual,
        'metric_name': metric_name,
        'base_period': base_period_str,
        'predict_period': predict_period_str,
        'history_years': history_years,
        'history_details': history_details
    }


def predict_line_flow(
    metric_type: str,
    predict_start: str,
    predict_end: str,
    history_years: int = 2,
    custom_configs: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    线路客流预测
    
    参数:
        metric_type: 指标类型
        predict_start: 预测开始日期 (YYYYMMDD)
        predict_end: 预测结束日期 (YYYYMMDD)
        history_years: 参考历史年数
        custom_configs: 自定义各年配置
        
    返回:
        预测结果字典
    """
    return predict_flow(
        metric_type=metric_type,
        predict_start=predict_start,
        predict_end=predict_end,
        history_years=history_years,
        custom_configs=custom_configs,
        data_source='line',
        read_data_func=read_line_daily_flow_history_range
    )


def predict_station_flow(
    metric_type: str,
    predict_start: str,
    predict_end: str,
    history_years: int = 2,
    custom_configs: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    车站客流预测
    
    参数:
        metric_type: 指标类型
        predict_start: 预测开始日期 (YYYYMMDD)
        predict_end: 预测结束日期 (YYYYMMDD)
        history_years: 参考历史年数
        custom_configs: 自定义各年配置
        
    返回:
        预测结果字典
    """
    return predict_flow(
        metric_type=metric_type,
        predict_start=predict_start,
        predict_end=predict_end,
        history_years=history_years,
        custom_configs=custom_configs,
        data_source='station',
        read_data_func=read_station_daily_flow_history_range
    )


def check_data_availability(
    metric_type: str,
    start_date: str,
    end_date: str,
    data_source: str = 'line'
) -> Dict[str, Any]:
    """
    检查指定日期范围是否有数据
    
    参数:
        metric_type: 指标类型
        start_date: 开始日期 (YYYYMMDD或YYYY-MM-DD)
        end_date: 结束日期
        data_source: 数据源类型 ('line'或'station')
        
    返回:
        数据可用性信息字典
    """
    # 标准化日期格式
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')
    
    try:
        if data_source == 'station':
            df = read_station_daily_flow_history_range(metric_type, start_date, end_date)
            entity_name = '车站'
            count_field = 'station_count'
        else:
            df = read_line_daily_flow_history_range(metric_type, start_date, end_date)
            entity_name = '线路'
            count_field = 'line_count'
        
        if df.empty:
            return {
                'success': True,
                'has_data': False,
                'message': f'该日期范围暂无{entity_name}实际数据，可以进行预测但无法对比准确率'
            }
        else:
            entity_count = df['F_LINENO'].nunique()
            date_count = df['F_DATE'].nunique()
            total_records = len(df)
            
            return {
                'success': True,
                'has_data': True,
                'message': f'找到{entity_name}实际数据！共{entity_count}条{entity_name}，{date_count}天，{total_records}条记录。预测后可对比准确率 ✓',
                count_field: int(entity_count),
                'date_count': int(date_count),
                'total_records': int(total_records)
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def format_predictions_for_display(predictions: List[Dict], is_station: bool = False) -> pd.DataFrame:
    """
    将预测结果格式化为展示用的DataFrame
    
    参数:
        predictions: 预测结果列表
        is_station: 是否为车站预测
        
    返回:
        格式化后的DataFrame
    """
    if not predictions:
        return pd.DataFrame()
    
    df = pd.DataFrame(predictions)
    
    # 格式化日期
    if '预测日期' in df.columns:
        df['预测日期'] = pd.to_datetime(df['预测日期'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    
    # 重命名列
    id_col = '车站ID' if is_station else '线路编号'
    name_col = '车站名称' if is_station else '线路名称'
    
    column_order = [
        id_col, name_col, '预测日期', '第几天', '基期日均',
        '最优增长率', '最优来源年份', '预测客流'
    ]
    
    # 如果有实际数据
    if '实际客流' in df.columns:
        column_order.extend(['实际客流', '误差', '准确率'])
    
    # 只保留存在的列
    column_order = [col for col in column_order if col in df.columns]
    
    return df[column_order]


def calculate_summary_stats(predictions: List[Dict], is_station: bool = False) -> Dict[str, Any]:
    """
    计算预测汇总统计
    
    参数:
        predictions: 预测结果列表
        is_station: 是否为车站预测
        
    返回:
        汇总统计字典
    """
    if not predictions:
        return {}
    
    df = pd.DataFrame(predictions)
    id_field = '车站ID' if is_station else '线路编号'
    name_field = '车站名称' if is_station else '线路名称'
    
    stats = {
        'total_predictions': len(predictions),
        'unique_entities': df[id_field].nunique() if id_field in df.columns else 0,
        'avg_predicted_flow': int(df['预测客流'].mean()) if '预测客流' in df.columns else 0,
        'max_predicted_flow': int(df['预测客流'].max()) if '预测客流' in df.columns else 0,
        'avg_growth_rate': round(df['最优增长率'].mean(), 2) if '最优增长率' in df.columns else 0,
    }
    
    # 如果有准确率数据
    if '准确率' in df.columns:
        valid_accuracy = df['准确率'].dropna()
        if len(valid_accuracy) > 0:
            stats['avg_accuracy'] = round(valid_accuracy.mean(), 2)
            stats['min_accuracy'] = round(valid_accuracy.min(), 2)
            stats['max_accuracy'] = round(valid_accuracy.max(), 2)
            stats['accuracy_count'] = len(valid_accuracy)
    
    # 按线路/车站汇总
    entity_stats = []
    for entity_id in df[id_field].unique():
        entity_data = df[df[id_field] == entity_id]
        entity_stat = {
            'id': entity_id,
            'name': entity_data[name_field].iloc[0],
            'avg_predicted': int(entity_data['预测客流'].mean()),
            'avg_growth_rate': round(entity_data['最优增长率'].mean(), 2)
        }
        
        if '准确率' in entity_data.columns:
            valid_acc = entity_data['准确率'].dropna()
            if len(valid_acc) > 0:
                entity_stat['avg_accuracy'] = round(valid_acc.mean(), 2)
        
        entity_stats.append(entity_stat)
    
    stats['entity_stats'] = entity_stats
    
    return stats
