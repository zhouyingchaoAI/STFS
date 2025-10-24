from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from yunying import read_line_daily_flow_history, read_station_daily_flow_history
import io
from datetime import datetime
import json

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 确保中文正常显示

# 添加全局响应头设置
@app.after_request
def after_request(response):
    # 只对JSON响应设置正确的Content-Type
    if response.content_type == 'application/json':
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

def fix_chinese_encoding(data):
    """修复中文字符编码问题"""
    if isinstance(data, dict):
        return {key: fix_chinese_encoding(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [fix_chinese_encoding(item) for item in data]
    elif isinstance(data, str):
        # 尝试修复可能的编码问题
        try:
            # 检查是否包含乱码字符
            if '?' in data or '\ufffd' in data or any(ord(c) > 127 and ord(c) < 256 for c in data):
                # 尝试多种编码修复方法
                try:
                    # 方法1: 从latin-1解码再重新编码为utf-8
                    return data.encode('latin-1').decode('utf-8')
                except (UnicodeDecodeError, UnicodeEncodeError):
                    try:
                        # 方法2: 从cp1252解码再重新编码为utf-8
                        return data.encode('cp1252').decode('utf-8')
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        try:
                            # 方法3: 从iso-8859-1解码再重新编码为utf-8
                            return data.encode('iso-8859-1').decode('utf-8')
                        except (UnicodeDecodeError, UnicodeEncodeError):
                            # 如果所有方法都失败，返回原字符串
                            return data
            return data
        except Exception:
            return data
    else:
        return data

# 指标类型的中文名称映射
METRIC_NAMES = {
    'F_PKLCOUNT': '客流量',
    'F_ENTRANCE': '进站人数',
    'F_EXIT': '出站人数',
    'F_TRANSFER': '换乘人数',
    'F_BOARD_ALIGHT': '乘降量'
}

@app.route('/')
def index():
    response = render_template('index.html')
    return response

@app.route('/query', methods=['POST'])
def query_data():
    try:
        data = request.json
        metric_type = data.get('metric_type')
        start_date = data.get('start_date').replace('-', '')  # 转换为 yyyymmdd 格式
        end_date = data.get('end_date').replace('-', '')  # 转换为 yyyymmdd 格式
        
        # 调用查询函数
        df = read_line_daily_flow_history(metric_type, start_date, end_date)
        
        # 重命名客流量列为中文名称
        metric_name = METRIC_NAMES.get(metric_type, '客流量')
        df = df.rename(columns={'F_KLCOUNT': metric_name})
        
        # 只保留需要显示的列
        display_columns = [
            'F_DATE',
            'F_LINENO',
            'F_LINENAME',
            metric_name,
            'WEATHER_TYPE',
            'F_DATEFEATURES'
        ]
        
        # 筛选存在的列
        available_columns = [col for col in display_columns if col in df.columns]
        df_display = df[available_columns]
        
        # 计算每条线路的均值统计
        line_stats = []
        if 'F_LINENO' in df.columns and metric_name in df.columns:
            grouped = df.groupby(['F_LINENO', 'F_LINENAME'])[metric_name].agg(['mean', 'sum', 'count']).reset_index()
            grouped.columns = ['F_LINENO', 'F_LINENAME', '平均值', '总计', '天数']
            # 四舍五入到整数
            grouped['平均值'] = grouped['平均值'].round(0).astype(int)
            grouped['总计'] = grouped['总计'].astype(int)
            line_stats = grouped.to_dict('records')
        
        # 处理中文字符编码问题
        df_display_clean = df_display.copy()
        for col in df_display_clean.columns:
            if df_display_clean[col].dtype == 'object':  # 字符串列
                df_display_clean[col] = df_display_clean[col].astype(str).apply(fix_chinese_encoding)
        
        # 转换为 JSON 格式
        result = {
            'success': True,
            'data': df_display_clean.to_dict('records'),
            'columns': available_columns,
            'total_records': len(df_display_clean),
            'line_stats': fix_chinese_encoding(line_stats),  # 添加线路统计数据
            'metric_name': metric_name  # 添加指标名称
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/calculate_growth_multi_year', methods=['POST'])
def calculate_growth_multi_year():
    """支持多年自定义日期配置的增长率计算"""
    try:
        data = request.json
        metric_type = data.get('metric_type')
        year_configs = data.get('year_configs', [])  # 每年的配置列表
        
        if not year_configs:
            return jsonify({
                'success': False,
                'error': '未提供年份配置'
            }), 400
        
        metric_name = METRIC_NAMES.get(metric_type, '客流量')
        all_years_data = []
        base_period_info = None
        
        # 处理每年的数据
        for year_config in year_configs:
            year = year_config.get('year')
            base_start = year_config.get('base_start').replace('-', '')
            base_end = year_config.get('base_end').replace('-', '')
            compare_start = year_config.get('compare_start').replace('-', '')
            compare_end = year_config.get('compare_end').replace('-', '')
            
            # 查询基期数据
            df_base = read_line_daily_flow_history(metric_type, base_start, base_end)
            if df_base.empty:
                continue
            df_base = df_base.rename(columns={'F_KLCOUNT': metric_name})
            
            # 计算基期日均值
            base_stats = df_base.groupby(['F_LINENO', 'F_LINENAME'])[metric_name].mean().reset_index()
            base_stats.columns = ['F_LINENO', 'F_LINENAME', '基期日均']
            
            # 查询对比期数据
            df_compare = read_line_daily_flow_history(metric_type, compare_start, compare_end)
            if df_compare.empty:
                continue
            df_compare = df_compare.rename(columns={'F_KLCOUNT': metric_name})
            
            # 合并基期日均到对比期数据
            df_merged = df_compare.merge(base_stats, on=['F_LINENO', 'F_LINENAME'], how='inner')
            df_merged['年份'] = year
            
            # 计算增长率
            df_merged['增长率'] = df_merged.apply(
                lambda row: ((row[metric_name] - row['基期日均']) / row['基期日均'] * 100) 
                if row['基期日均'] > 0 else 0, axis=1
            ).round(2)
            df_merged['增长量'] = (df_merged[metric_name] - df_merged['基期日均']).round(0)
            
            all_years_data.append(df_merged)
            
            # 保存第一年的基期信息
            if base_period_info is None:
                base_period_info = f"{base_start[:4]}-{base_start[4:6]}-{base_start[6:]} 至 {base_end[:4]}-{base_end[4:6]}-{base_end[6:]}"
        
        if not all_years_data:
            return jsonify({
                'success': False,
                'error': '没有找到任何有效数据'
            }), 400
        
        # 合并所有年份数据
        df_all = pd.concat(all_years_data, ignore_index=True)
        
        # 准备返回数据
        result_df = df_all[['F_DATE', 'F_LINENO', 'F_LINENAME', '年份', '基期日均', 
                             metric_name, '增长率', '增长量']].copy()
        result_df.columns = ['日期', '线路编号', '线路名称', '年份', '基期日均', '当日客流', '增长率', '增长量']
        result_df = result_df.sort_values(['年份', '线路编号', '日期'], ascending=[False, True, True])
        
        # 数据清理
        result_df['基期日均'] = result_df['基期日均'].fillna(0).round(0).astype(int)
        result_df['当日客流'] = result_df['当日客流'].fillna(0).round(0).astype(int)
        result_df['增长量'] = result_df['增长量'].fillna(0).astype(int)
        result_df['增长率'] = result_df['增长率'].fillna(0).round(2)
        
        growth_list = result_df.replace([np.inf, -np.inf], 0).to_dict('records')
        
        # 清理NaN值
        for item in growth_list:
            for key, value in item.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    item[key] = 0
        
        # 计算每条线路"第几天"的最大增长率
        max_growth_by_day = []
        
        for line_no in result_df['线路编号'].unique():
            line_data = result_df[result_df['线路编号'] == line_no]
            line_name = line_data['线路名称'].iloc[0]
            
            # 为每条线路的数据添加"第几天"标记
            line_data_with_day = line_data.copy()
            
            # 按年份分组，为每年的数据标记天数
            for year in line_data_with_day['年份'].unique():
                year_mask = line_data_with_day['年份'] == year
                year_dates = line_data_with_day[year_mask]['日期'].sort_values()
                day_mapping = {date: idx + 1 for idx, date in enumerate(year_dates.unique())}
                line_data_with_day.loc[year_mask, '第几天'] = line_data_with_day.loc[year_mask, '日期'].map(day_mapping)
            
            # 按"第几天"分组，找出每天的最大增长率
            for day_num in sorted(line_data_with_day['第几天'].unique()):
                day_data = line_data_with_day[line_data_with_day['第几天'] == day_num]
                max_row = day_data.loc[day_data['增长率'].idxmax()]
                
                max_growth_by_day.append({
                    '线路编号': int(line_no.item()) if hasattr(line_no, 'item') else int(line_no),
                    '线路名称': str(line_name),
                    '第几天': int(day_num.item()) if hasattr(day_num, 'item') else int(day_num),
                    '最大增长率': float(max_row['增长率'].item()) if hasattr(max_row['增长率'], 'item') else float(max_row['增长率']),
                    '最大值年份': int(max_row['年份'].item()) if hasattr(max_row['年份'], 'item') else int(max_row['年份']),
                    '最大值日期': str(max_row['日期']),
                    '最大值客流': int(max_row['当日客流'].item()) if hasattr(max_row['当日客流'], 'item') else int(max_row['当日客流']),
                    '基期日均': int(max_row['基期日均'].item()) if hasattr(max_row['基期日均'], 'item') else int(max_row['基期日均'])
                })
        
        # 获取年份列表
        years_list = sorted(result_df['年份'].unique(), reverse=True)
        
        # 获取对比期信息（使用第一个配置的对比期）
        first_config = year_configs[0]
        compare_period_info = f"{first_config['compare_start']} 至 {first_config['compare_end']}"
        
        result = {
            'success': True,
            'growth_data': fix_chinese_encoding(growth_list),
            'max_growth_by_day': fix_chinese_encoding(max_growth_by_day),  # 每天的最大增长率
            'years_list': [int(y.item()) if hasattr(y, 'item') else int(y) for y in years_list],
            'metric_name': metric_name,
            'base_period': base_period_info,
            'compare_period': compare_period_info,
            'year_range': len(year_configs) - 1
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/calculate_growth', methods=['POST'])
def calculate_growth():
    try:
        data = request.json
        metric_type = data.get('metric_type')
        
        # 基期数据（第一步的日期范围）
        base_start = data.get('base_start_date').replace('-', '')
        base_end = data.get('base_end_date').replace('-', '')
        
        # 对比期数据（第二步的日期范围）
        compare_start = data.get('compare_start_date').replace('-', '')
        compare_end = data.get('compare_end_date').replace('-', '')
        
        # 年限（对比前几年）
        year_range = int(data.get('year_range', 0))
        
        metric_name = METRIC_NAMES.get(metric_type, '客流量')
        
        # 查询基期数据
        df_base = read_line_daily_flow_history(metric_type, base_start, base_end)
        if df_base.empty:
            return jsonify({
                'success': False,
                'error': '基期日期范围内没有找到数据'
            }), 400
        df_base = df_base.rename(columns={'F_KLCOUNT': metric_name})
        
        # 查询对比期数据
        df_compare = read_line_daily_flow_history(metric_type, compare_start, compare_end)
        if df_compare.empty:
            return jsonify({
                'success': False,
                'error': '对比期日期范围内没有找到数据'
            }), 400
        df_compare = df_compare.rename(columns={'F_KLCOUNT': metric_name})
        
        # 计算基期日均值
        base_stats = df_base.groupby(['F_LINENO', 'F_LINENAME'])[metric_name].mean().reset_index()
        base_stats.columns = ['F_LINENO', 'F_LINENAME', '基期日均']
        base_stats['基期天数'] = df_base.groupby(['F_LINENO', 'F_LINENAME'])['F_DATE'].count().values
        
        # 存储所有年份的数据
        all_years_data = []
        
        # 当年数据
        df_compare_current = df_compare.copy()
        df_compare_current['年份'] = int(compare_start[:4])
        all_years_data.append(df_compare_current)
        
        # 查询历年同期数据
        if year_range > 0:
            from datetime import datetime
            compare_start_date = datetime.strptime(compare_start, '%Y%m%d')
            compare_end_date = datetime.strptime(compare_end, '%Y%m%d')
            
            for i in range(1, year_range + 1):
                # 计算前i年的同期日期
                prev_year = int(compare_start[:4]) - i
                prev_start = f"{prev_year}{compare_start[4:]}"
                prev_end = f"{prev_year}{compare_end[4:]}"
                
                try:
                    df_prev = read_line_daily_flow_history(metric_type, prev_start, prev_end)
                    if not df_prev.empty:
                        df_prev = df_prev.rename(columns={'F_KLCOUNT': metric_name})
                        df_prev['年份'] = prev_year
                        all_years_data.append(df_prev)
                except Exception as e:
                    print(f"查询{prev_year}年同期数据失败: {e}")
                    continue
        
        # 合并所有年份的数据
        df_all_years = pd.concat(all_years_data, ignore_index=True)
        
        # 获取对比期每天的数据并添加基期日均
        df_compare_with_base = df_all_years.merge(base_stats[['F_LINENO', 'F_LINENAME', '基期日均']], 
                                                   on=['F_LINENO', 'F_LINENAME'], how='inner')
        
        if df_compare_with_base.empty:
            return jsonify({
                'success': False,
                'error': '基期和对比期没有相同的线路数据，无法计算增长率'
            }), 400
        
        # 计算每天的增长率
        df_compare_with_base['增长率'] = df_compare_with_base.apply(
            lambda row: ((row[metric_name] - row['基期日均']) / row['基期日均'] * 100) 
            if row['基期日均'] > 0 else 0, axis=1
        ).round(2)
        df_compare_with_base['增长量'] = (df_compare_with_base[metric_name] - df_compare_with_base['基期日均']).round(0)
        
        # 选择需要的列
        result_df = df_compare_with_base[['F_DATE', 'F_LINENO', 'F_LINENAME', '年份', '基期日均', 
                                           metric_name, '增长率', '增长量']].copy()
        result_df.columns = ['日期', '线路编号', '线路名称', '年份', '基期日均', '当日客流', '增长率', '增长量']
        
        # 自定义排序：线网(0)优先，然后按线路编号排序
        def custom_sort_key(x):
            if x == 0:
                return -1  # 线网排第一
            else:
                return x
        
        result_df['排序键'] = result_df['线路编号'].apply(custom_sort_key)
        result_df = result_df.sort_values(['年份', '排序键', '日期'], ascending=[False, True, True])
        result_df = result_df.drop('排序键', axis=1)
        
        # 填充NaN值并转换数据类型
        result_df['基期日均'] = result_df['基期日均'].fillna(0).round(0).astype(int)
        result_df['当日客流'] = result_df['当日客流'].fillna(0).round(0).astype(int)
        result_df['增长量'] = result_df['增长量'].fillna(0).astype(int)
        result_df['增长率'] = result_df['增长率'].fillna(0).round(2)
        
        # 转换为字典列表
        growth_list = result_df.replace([np.inf, -np.inf], 0).to_dict('records')
        
        # 确保没有NaN或inf值
        for item in growth_list:
            for key, value in item.items():
                if isinstance(value, float):
                    if np.isnan(value) or np.isinf(value):
                        item[key] = 0
        
        # 计算每条线路"第几天"的最大增长率
        max_growth_by_day = []
        
        for line_no in result_df['线路编号'].unique():
            line_data = result_df[result_df['线路编号'] == line_no]
            line_name = line_data['线路名称'].iloc[0]
            
            # 为每条线路的数据添加"第几天"标记
            line_data_with_day = line_data.copy()
            
            # 按年份分组，为每年的数据标记天数
            for year in line_data_with_day['年份'].unique():
                year_mask = line_data_with_day['年份'] == year
                year_dates = line_data_with_day[year_mask]['日期'].sort_values()
                day_mapping = {date: idx + 1 for idx, date in enumerate(year_dates.unique())}
                line_data_with_day.loc[year_mask, '第几天'] = line_data_with_day.loc[year_mask, '日期'].map(day_mapping)
            
            # 按"第几天"分组，找出每天的最大增长率
            for day_num in sorted(line_data_with_day['第几天'].unique()):
                day_data = line_data_with_day[line_data_with_day['第几天'] == day_num]
                max_row = day_data.loc[day_data['增长率'].idxmax()]
                
                max_growth_by_day.append({
                    '线路编号': int(line_no.item()) if hasattr(line_no, 'item') else int(line_no),
                    '线路名称': str(line_name),
                    '第几天': int(day_num.item()) if hasattr(day_num, 'item') else int(day_num),
                    '最大增长率': float(max_row['增长率'].item()) if hasattr(max_row['增长率'], 'item') else float(max_row['增长率']),
                    '最大值年份': int(max_row['年份'].item()) if hasattr(max_row['年份'], 'item') else int(max_row['年份']),
                    '最大值日期': str(max_row['日期']),
                    '最大值客流': int(max_row['当日客流'].item()) if hasattr(max_row['当日客流'], 'item') else int(max_row['当日客流']),
                    '基期日均': int(max_row['基期日均'].item()) if hasattr(max_row['基期日均'], 'item') else int(max_row['基期日均'])
                })
        
        # 按线路和年份分组统计信息
        line_summary = []
        years_list = sorted(result_df['年份'].unique(), reverse=True)
        
        for line_no in result_df['线路编号'].unique():
            line_data = result_df[result_df['线路编号'] == line_no]
            line_name = line_data['线路名称'].iloc[0]
            
            # 每年的平均增长率
            year_stats = {}
            for year in years_list:
                year_data = line_data[line_data['年份'] == year]
                if len(year_data) > 0:
                    avg_growth = year_data['增长率'].mean()
                    avg_flow = year_data['当日客流'].mean()
                    year_stats[str(year)] = {
                        '平均增长率': float(round(avg_growth, 2)),
                        '平均客流': int(avg_flow.item()) if hasattr(avg_flow, 'item') else int(avg_flow),
                        '天数': int(len(year_data))
                    }
            
            line_summary.append({
                '线路编号': int(line_no.item()) if hasattr(line_no, 'item') else int(line_no),
                '线路名称': str(line_name),
                '年份统计': year_stats
            })
        
        result = {
            'success': True,
            'growth_data': fix_chinese_encoding(growth_list),
            'line_summary': fix_chinese_encoding(line_summary),
            'max_growth_by_day': fix_chinese_encoding(max_growth_by_day),  # 每天的最大增长率
            'years_list': [int(y.item()) if hasattr(y, 'item') else int(y) for y in years_list],
            'metric_name': metric_name,
            'base_period': f"{base_start[:4]}-{base_start[4:6]}-{base_start[6:]} 至 {base_end[:4]}-{base_end[4:6]}-{base_end[6:]}",
            'compare_period': f"{compare_start[:4]}-{compare_start[4:6]}-{compare_start[6:]} 至 {compare_end[:4]}-{compare_end[4:6]}-{compare_end[6:]}",
            'year_range': year_range
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/export_growth', methods=['POST'])
def export_growth():
    """导出增长率分析数据"""
    try:
        data = request.json
        metric_type = data.get('metric_type')
        year_configs = data.get('year_configs')
        
        metric_name = METRIC_NAMES.get(metric_type, '客流量')
        
        # 如果有自定义配置，使用自定义配置
        if year_configs:
            all_years_data = []
            
            for year_config in year_configs:
                year = year_config.get('year')
                base_start = year_config.get('base_start').replace('-', '')
                base_end = year_config.get('base_end').replace('-', '')
                compare_start = year_config.get('compare_start').replace('-', '')
                compare_end = year_config.get('compare_end').replace('-', '')
                
                # 查询基期数据
                df_base = read_line_daily_flow_history(metric_type, base_start, base_end)
                if df_base.empty:
                    continue
                df_base = df_base.rename(columns={'F_KLCOUNT': metric_name})
                
                # 计算基期日均值
                base_stats = df_base.groupby(['F_LINENO', 'F_LINENAME'])[metric_name].mean().reset_index()
                base_stats.columns = ['F_LINENO', 'F_LINENAME', '基期日均']
                
                # 查询对比期数据
                df_compare = read_line_daily_flow_history(metric_type, compare_start, compare_end)
                if df_compare.empty:
                    continue
                df_compare = df_compare.rename(columns={'F_KLCOUNT': metric_name})
                
                # 合并基期日均到对比期数据
                df_merged = df_compare.merge(base_stats, on=['F_LINENO', 'F_LINENAME'], how='inner')
                df_merged['年份'] = year
                df_merged['基期范围'] = f"{base_start[:4]}-{base_start[4:6]}-{base_start[6:]} 至 {base_end[:4]}-{base_end[4:6]}-{base_end[6:]}"
                
                # 计算增长率
                df_merged['增长率'] = df_merged.apply(
                    lambda row: ((row[metric_name] - row['基期日均']) / row['基期日均'] * 100) 
                    if row['基期日均'] > 0 else 0, axis=1
                ).round(2)
                df_merged['增长量'] = (df_merged[metric_name] - df_merged['基期日均']).round(0)
                
                all_years_data.append(df_merged)
            
            if not all_years_data:
                return jsonify({'success': False, 'error': '没有找到任何有效数据'}), 400
            
            df_export = pd.concat(all_years_data, ignore_index=True)
        else:
            # 使用简单配置
            base_start = data.get('base_start_date').replace('-', '')
            base_end = data.get('base_end_date').replace('-', '')
            compare_start = data.get('compare_start_date').replace('-', '')
            compare_end = data.get('compare_end_date').replace('-', '')
            year_range = int(data.get('year_range', 0))
            
            # （复用原有逻辑...）
            df_base = read_line_daily_flow_history(metric_type, base_start, base_end)
            df_base = df_base.rename(columns={'F_KLCOUNT': metric_name})
            
            base_stats = df_base.groupby(['F_LINENO', 'F_LINENAME'])[metric_name].mean().reset_index()
            base_stats.columns = ['F_LINENO', 'F_LINENAME', '基期日均']
            
            all_years_data = []
            df_compare_current = read_line_daily_flow_history(metric_type, compare_start, compare_end)
            df_compare_current = df_compare_current.rename(columns={'F_KLCOUNT': metric_name})
            df_compare_current['年份'] = int(compare_start[:4])
            all_years_data.append(df_compare_current)
            
            # 查询历年数据
            if year_range > 0:
                for i in range(1, year_range + 1):
                    prev_year = int(compare_start[:4]) - i
                    prev_start = f"{prev_year}{compare_start[4:]}"
                    prev_end = f"{prev_year}{compare_end[4:]}"
                    
                    try:
                        df_prev = read_line_daily_flow_history(metric_type, prev_start, prev_end)
                        if not df_prev.empty:
                            df_prev = df_prev.rename(columns={'F_KLCOUNT': metric_name})
                            df_prev['年份'] = prev_year
                            all_years_data.append(df_prev)
                    except:
                        continue
            
            df_all_years = pd.concat(all_years_data, ignore_index=True)
            df_export = df_all_years.merge(base_stats, on=['F_LINENO', 'F_LINENAME'], how='inner')
            df_export['基期范围'] = f"{base_start[:4]}-{base_start[4:6]}-{base_start[6:]} 至 {base_end[:4]}-{base_end[4:6]}-{base_end[6:]}"
            
            df_export['增长率'] = df_export.apply(
                lambda row: ((row[metric_name] - row['基期日均']) / row['基期日均'] * 100) 
                if row['基期日均'] > 0 else 0, axis=1
            ).round(2)
            df_export['增长量'] = (df_export[metric_name] - df_export['基期日均']).round(0)
        
        # 选择导出的列
        export_columns = ['年份', 'F_DATE', 'F_LINENO', 'F_LINENAME', '基期范围', '基期日均', 
                         metric_name, '增长量', '增长率']
        df_export_final = df_export[export_columns].copy()
        df_export_final.columns = ['年份', '日期', '线路编号', '线路名称', '基期范围', '基期日均', 
                                    f'当日{metric_name}', '增长量', '增长率(%)']
        
        # 按年份、线路、日期排序
        df_export_final = df_export_final.sort_values(['年份', '线路编号', '日期'], ascending=[False, True, True])
        
        # 创建内存中的CSV文件
        output = io.BytesIO()
        df_export_final.to_csv(output, index=False, encoding='utf_8_sig')
        output.seek(0)
        
        # 生成文件名
        compare_start_display = data.get('compare_start_date', compare_start[:4] + '-' + compare_start[4:6] + '-' + compare_start[6:])
        compare_end_display = data.get('compare_end_date', compare_end[:4] + '-' + compare_end[4:6] + '-' + compare_end[6:])
        filename = f"增长率分析_{metric_name}_{compare_start_display}至{compare_end_display}.csv"
        
        return send_file(
            io.BytesIO(output.read()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/check_data_availability', methods=['POST'])
def check_data_availability():
    """检查指定日期范围是否有线路实际数据"""
    try:
        data = request.json
        metric_type = data.get('metric_type')
        start_date = data.get('start_date').replace('-', '')
        end_date = data.get('end_date').replace('-', '')
        
        # 查询线路数据
        df = read_line_daily_flow_history(metric_type, start_date, end_date)
        
        if df.empty:
            return jsonify({
                'success': True,
                'has_data': False,
                'message': '该日期范围暂无实际数据，可以进行预测但无法对比准确率'
            })
        else:
            # 统计数据情况
            line_count = df['F_LINENO'].nunique()
            date_count = df['F_DATE'].nunique()
            total_records = len(df)
            
            return jsonify({
                'success': True,
                'has_data': True,
                'message': f'找到实际数据！共{line_count}条线路，{date_count}天，{total_records}条记录。预测后可对比准确率 ✓',
                'line_count': int(line_count),
                'date_count': int(date_count),
                'total_records': int(total_records)
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/check_station_data_availability', methods=['POST'])
def check_station_data_availability():
    """检查指定日期范围是否有车站实际数据"""
    try:
        data = request.json
        metric_type = data.get('metric_type')
        start_date = data.get('start_date').replace('-', '')
        end_date = data.get('end_date').replace('-', '')
        
        # 查询车站数据
        df = read_station_daily_flow_history(metric_type, start_date, end_date)
        
        if df.empty:
            return jsonify({
                'success': True,
                'has_data': False,
                'message': '该日期范围暂无车站实际数据，可以进行预测但无法对比准确率'
            })
        else:
            # 统计数据情况
            station_count = df['F_LINENO'].nunique()
            date_count = df['F_DATE'].nunique()
            total_records = len(df)
            
            return jsonify({
                'success': True,
                'has_data': True,
                'message': f'找到车站实际数据！共{station_count}个车站，{date_count}天，{total_records}条记录。预测后可对比准确率 ✓',
                'station_count': int(station_count),
                'date_count': int(date_count),
                'total_records': int(total_records)
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict_flow', methods=['POST'])
def predict_flow():
    """基于历年最优增长率预测线路客流，并与实际值对比"""
    try:
        data = request.json
        metric_type = data.get('metric_type')
        predict_start = data.get('predict_start').replace('-', '')
        predict_end = data.get('predict_end').replace('-', '')
        history_years = int(data.get('history_years', 2))
        custom_configs = data.get('custom_configs')
        
        metric_name = METRIC_NAMES.get(metric_type, '客流量')
        
        print(f"\n{'='*60}")
        print(f"开始预测线路客流")
        print(f"指标类型: {metric_type} ({metric_name})")
        print(f"预测日期: {predict_start} - {predict_end}")
        print(f"参考历史: 前{history_years}年")
        print(f"自定义配置: {'是' if custom_configs else '否'}")
        print(f"{'='*60}\n")
        
        # 计算基期
        from datetime import datetime
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
        import calendar
        base_start = f"{base_year}{str(base_month).zfill(2)}01"
        last_day = calendar.monthrange(base_year, base_month)[1]
        base_end = f"{base_year}{str(base_month).zfill(2)}{str(last_day).zfill(2)}"
        
        # 查询基期数据
        df_base = read_line_daily_flow_history(metric_type, base_start, base_end)
        if df_base.empty:
            return jsonify({
                'success': False,
                'error': f'基期（{base_year}年{base_month}月）数据不存在'
            }), 400
        df_base = df_base.rename(columns={'F_KLCOUNT': metric_name})
        print(f"基期数据查询成功：{len(df_base)}行")
        
        # 计算基期日均
        base_avg = df_base.groupby(['F_LINENO', 'F_LINENAME'])[metric_name].mean().reset_index()
        base_avg.columns = ['F_LINENO', 'F_LINENAME', '基期日均']
        
        # 验证 base_avg 是否有效
        if base_avg.empty or len(base_avg) == 0:
            return jsonify({
                'success': False,
                'error': f'基期数据处理失败，无法计算日均客流'
            }), 400
        
        print(f"基期日均计算完成，共{len(base_avg)}条线路数据")
        print(f"base_avg 类型: {type(base_avg)}, 形状: {base_avg.shape}, 列: {base_avg.columns.tolist()}")
        if len(base_avg) > 0:
            print(f"base_avg 前几行:\n{base_avg.head()}")
        
        # 第二步：查询历年同期数据
        all_years_data = []
        history_details = []
        
        for i in range(1, history_years + 1):
            history_year = predict_year - i
            print(f"\n===== 处理第{i}年历史数据：{history_year}年 =====")
            
            # 检查是否有自定义配置
            if custom_configs and str(history_year) in custom_configs:
                config = custom_configs[str(history_year)]
                history_start = config['ref_start'].replace('-', '')
                history_end = config['ref_end'].replace('-', '')
                history_base_start = config['base_start'].replace('-', '')
                history_base_end = config['base_end'].replace('-', '')
            else:
                # 使用默认计算
                history_start = f"{history_year}{predict_start[4:]}"
                history_end = f"{history_year}{predict_end[4:]}"
                
                # 查询历年基期（该年的上月）
                history_base_month = predict_month - 1
                history_base_year = history_year
                if history_base_month < 1:
                    history_base_month = 12
                    history_base_year -= 1
                
                history_base_start = f"{history_base_year}{str(history_base_month).zfill(2)}01"
                history_last_day = calendar.monthrange(history_base_year, history_base_month)[1]
                history_base_end = f"{history_base_year}{str(history_base_month).zfill(2)}{str(history_last_day).zfill(2)}"
            
            try:
                # 查询历年参考期数据
                df_history = read_line_daily_flow_history(metric_type, history_start, history_end)
                if df_history.empty:
                    print(f"⚠️ {history_year}年参考期数据为空，跳过")
                    continue
                df_history = df_history.rename(columns={'F_KLCOUNT': metric_name})
                
                # 查询历年基期
                df_history_base = read_line_daily_flow_history(metric_type, history_base_start, history_base_end)
                if df_history_base.empty:
                    print(f"⚠️ {history_year}年基期数据为空，跳过")
                    continue
                df_history_base = df_history_base.rename(columns={'F_KLCOUNT': metric_name})
                
                # 计算历年基期日均
                history_base_avg = df_history_base.groupby(['F_LINENO', 'F_LINENAME'])[metric_name].mean().reset_index()
                history_base_avg.columns = ['F_LINENO', 'F_LINENAME', f'基期日均_{history_year}']
                
                # 添加年份标记
                df_history['年份'] = history_year
                
                # 合并历年基期
                df_history = df_history.merge(history_base_avg, on=['F_LINENO', 'F_LINENAME'], how='inner')
                
                # 计算增长率
                df_history['增长率'] = df_history.apply(
                    lambda row: ((row[metric_name] - row[f'基期日均_{history_year}']) / row[f'基期日均_{history_year}'] * 100)
                    if row[f'基期日均_{history_year}'] > 0 else 0, axis=1
                ).round(2)
                
                all_years_data.append(df_history)
                print(f"✓ {history_year}年：参考期{len(df_history)}行")
                
                # 记录详情（用于显示）
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
                        'line_no': int(line_no),
                        'line_name': str(line_data['F_LINENAME'].iloc[0]),
                        'base_avg': int(line_data[f'基期日均_{history_year}'].iloc[0]),
                        'avg_growth': round(float(line_data['增长率'].mean()), 2),
                        'max_growth': round(float(line_data['增长率'].max()), 2)
                    })
                
            except Exception as e:
                print(f"❌ {history_year}年数据处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_years_data:
            return jsonify({
                'success': False,
                'error': '没有找到历年同期数据'
            }), 400
        
        # 合并所有年份数据（参考增长率分析的方式）
        df_all_history = pd.concat(all_years_data, ignore_index=True)
        print(f"✓ 合并完成：共{len(df_all_history)}行")
        
        # 第三步：为历年数据标记"第几天"
        print("\n===== 开始计算预测 =====")
        
        # 为每年的每条线路标记"第几天"
        for year in df_all_history['年份'].unique():
            for line_no in df_all_history['F_LINENO'].unique():
                mask = (df_all_history['年份'] == year) & (df_all_history['F_LINENO'] == line_no)
                dates = df_all_history[mask]['F_DATE'].sort_values().unique()
                day_mapping = {date: idx + 1 for idx, date in enumerate(dates)}
                df_all_history.loc[mask, '第几天'] = df_all_history.loc[mask, 'F_DATE'].map(day_mapping)
        
        print(f"标记完成，开始计算每天的最优增长率...")
        
        # 第四步：计算每条线路每天的最优增长率
        predictions = []
        
        for line_no in base_avg['F_LINENO'].unique():
            line_name = base_avg[base_avg['F_LINENO'] == line_no]['F_LINENAME'].iloc[0]
            base_daily_avg = base_avg[base_avg['F_LINENO'] == line_no]['基期日均'].iloc[0]
            
            # 获取该线路的历年数据
            line_history = df_all_history[df_all_history['F_LINENO'] == line_no].copy()
            if line_history.empty:
                continue
            
            # 按"第几天"分组，取最大增长率
            for day_num in sorted(line_history['第几天'].dropna().unique()):
                day_data = line_history[line_history['第几天'] == day_num]
                max_growth_rate = day_data['增长率'].max()
                max_year = day_data.loc[day_data['增长率'].idxmax(), '年份']
                
                # 预测客流 = 基期日均 × (1 + 最优增长率/100)
                predicted_flow = base_daily_avg * (1 + max_growth_rate / 100)
                
                # 生成预测日期
                from datetime import timedelta
                predict_date_obj = predict_start_date + timedelta(days=int(day_num) - 1)
                predict_date_str = predict_date_obj.strftime('%Y%m%d')
                
                predictions.append({
                    '线路编号': int(line_no),
                    '线路名称': str(line_name),
                    '预测日期': predict_date_str,
                    '第几天': int(day_num),
                    '基期日均': int(base_daily_avg),
                    '最优增长率': round(float(max_growth_rate), 2),
                    '最优来源年份': int(max_year),
                    '预测客流': int(predicted_flow)
                })
        
        print(f"\n✓ 预测计算完成：共{len(predictions)}条预测记录")
        
        # 第五步：查询实际数据（如果存在）
        print(f"\n===== 查询实际数据 =====")
        print(f"日期范围：{predict_start} - {predict_end}")
        has_actual = False
        matched_count = 0
        
        try:
            df_actual = read_line_daily_flow_history(metric_type, predict_start, predict_end)
            print(f"查询结果：{len(df_actual)}行")
            
            if not df_actual.empty:
                df_actual = df_actual.rename(columns={'F_KLCOUNT': metric_name})
                
                # 数据类型标准化：确保匹配时类型一致
                df_actual['F_LINENO'] = df_actual['F_LINENO'].astype(int)
                df_actual['F_DATE'] = df_actual['F_DATE'].astype(str).str.strip()
                
                print(f"实际数据的线路：{df_actual['F_LINENO'].unique()}")
                print(f"实际数据的日期：{sorted(df_actual['F_DATE'].unique())}")
                
                # 调试：显示预测数据的前几条
                print(f"\n预测数据示例（前3条）：")
                for i in range(min(3, len(predictions))):
                    pred = predictions[i]
                    print(f"  预测 {i+1}: 线路={pred['线路编号']} ({type(pred['线路编号'])}), 日期={pred['预测日期']} ({type(pred['预测日期'])})")
                
                # 调试：显示实际数据的前几条
                print(f"\n实际数据示例（前3条）：")
                for i in range(min(3, len(df_actual))):
                    print(f"  实际 {i+1}: 线路={df_actual.iloc[i]['F_LINENO']} ({type(df_actual.iloc[i]['F_LINENO'])}), 日期={df_actual.iloc[i]['F_DATE']} ({type(df_actual.iloc[i]['F_DATE'])})")
                
                print(f"\n开始匹配预测值和实际值...")
                
                # 为预测数据添加实际值和准确率
                for i, pred in enumerate(predictions):
                    pred_line = int(pred['线路编号'])
                    pred_date = str(pred['预测日期']).strip()
                    
                    actual_row = df_actual[
                        (df_actual['F_LINENO'] == pred_line) & 
                        (df_actual['F_DATE'] == pred_date)
                    ]
                    
                    if i < 3:  # 调试前3条
                        print(f"  匹配 {i+1}: 线路{pred['线路编号']} 日期{pred['预测日期']} -> 找到{len(actual_row)}行")
                    
                    if not actual_row.empty:
                        actual_flow = int(actual_row[metric_name].iloc[0])
                        pred['实际客流'] = actual_flow
                        
                        # 计算准确率 = 1 - |预测值 - 实际值| / 实际值
                        error_rate = abs(pred['预测客流'] - actual_flow) / actual_flow if actual_flow > 0 else 1
                        accuracy = (1 - error_rate) * 100
                        pred['准确率'] = round(accuracy, 2)
                        pred['误差'] = pred['预测客流'] - actual_flow
                        matched_count += 1
                        
                        if i < 3:  # 只打印前3条示例
                            print(f"  匹配成功 - 线路{pred['线路编号']} {pred['预测日期']}: 预测={pred['预测客流']}, 实际={actual_flow}, 准确率={pred['准确率']}%")
                    else:
                        pred['实际客流'] = None
                        pred['准确率'] = None
                        pred['误差'] = None
                
                print(f"\n✓ 匹配完成：{matched_count}/{len(predictions)} 条记录有实际数据")
                has_actual = matched_count > 0  # 只要有一条匹配成功就算有实际数据
                
                # 统计有准确率的记录数量
                accuracy_count = sum(1 for p in predictions if p.get('准确率') is not None)
                print(f"✓ 有准确率的记录：{accuracy_count}/{len(predictions)}")
                
            else:
                print(f"⚠️ 未查询到实际数据")
                # 设置所有预测记录为无实际数据
                for pred in predictions:
                    pred['实际客流'] = None
                    pred['准确率'] = None
                    pred['误差'] = None
                    
        except Exception as e:
            print(f"❌ 查询实际数据失败：{e}")
            import traceback
            traceback.print_exc()
            # 设置所有预测记录为无实际数据
            for pred in predictions:
                pred['实际客流'] = None
                pred['准确率'] = None
                pred['误差'] = None
        
        print(f"\n{'='*60}")
        print(f"准备返回结果：has_actual = {has_actual}")
        print(f"predictions 总数 = {len(predictions)}")
        if predictions:
            # 显示第一条预测记录的结构
            print(f"第一条记录示例：{predictions[0]}")
        print(f"{'='*60}\n")
        
        result = {
            'success': True,
            'predictions': fix_chinese_encoding(predictions),
            'has_actual': has_actual,
            'metric_name': metric_name,
            'base_period': f"{base_start[:4]}-{base_start[4:6]}-{base_start[6:]} 至 {base_end[:4]}-{base_end[4:6]}-{base_end[6:]}",
            'predict_period': f"{predict_start[:4]}-{predict_start[4:6]}-{predict_start[6:]} 至 {predict_end[:4]}-{predict_end[4:6]}-{predict_end[6:]}",
            'history_years': history_years,
            'history_details': fix_chinese_encoding(history_details)
        }
        
        print(f"\n{'='*60}")
        print(f"预测完成！返回{len(predictions)}条记录")
        print(f"{'='*60}\n")
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/export', methods=['POST'])
def export_csv():
    try:
        data = request.json
        metric_type = data.get('metric_type')
        start_date = data.get('start_date').replace('-', '')
        end_date = data.get('end_date').replace('-', '')
        
        # 调用查询函数
        df = read_line_daily_flow_history(metric_type, start_date, end_date)
        
        # 重命名客流量列为中文名称
        metric_name = METRIC_NAMES.get(metric_type, '客流量')
        df = df.rename(columns={'F_KLCOUNT': metric_name})
        
        # 只保留需要显示的列
        display_columns = [
            'F_DATE',
            'F_LINENO',
            'F_LINENAME',
            metric_name,
            'WEATHER_TYPE',
            'F_DATEFEATURES'
        ]
        
        # 筛选存在的列
        available_columns = [col for col in display_columns if col in df.columns]
        df_display = df[available_columns]
        
        # 创建内存中的CSV文件
        output = io.BytesIO()
        df_display.to_csv(output, index=False, encoding='utf_8_sig')
        output.seek(0)
        
        # 生成文件名
        filename = f"客流数据_{metric_name}_{start_date}_{end_date}.csv"
        
        return send_file(
            io.BytesIO(output.read()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict_station_flow', methods=['POST'])
def predict_station_flow():
    """车站预测 - 完全复用线路预测逻辑，只替换数据源"""
    try:
        data = request.json
        metric_type = data.get('metric_type')
        predict_start = data.get('predict_start').replace('-', '')
        predict_end = data.get('predict_end').replace('-', '')
        history_years = int(data.get('history_years', 2))
        custom_configs = data.get('custom_configs')
        
        metric_name = METRIC_NAMES.get(metric_type, '客流量')
        
        print(f"\n{'='*60}")
        print(f"开始预测车站客流")
        print(f"{'='*60}\n")
        
        from datetime import datetime, timedelta
        import calendar
        
        predict_start_date = datetime.strptime(predict_start, '%Y%m%d')
        predict_year = predict_start_date.year
        predict_month = predict_start_date.month
        
        base_month = predict_month - 1
        base_year = predict_year
        if base_month < 1:
            base_month = 12
            base_year -= 1
        
        base_start = f"{base_year}{str(base_month).zfill(2)}01"
        last_day = calendar.monthrange(base_year, base_month)[1]
        base_end = f"{base_year}{str(base_month).zfill(2)}{str(last_day).zfill(2)}"
        
        # 查询基期 - 用车站数据源
        df_base = read_station_daily_flow_history(metric_type, base_start, base_end)
        if df_base.empty:
            return jsonify({'success': False, 'error': f'基期车站数据不存在'}), 400
        df_base = df_base.rename(columns={'F_KLCOUNT': metric_name})
        
        base_avg = df_base.groupby(['F_LINENO', 'F_LINENAME'])[metric_name].mean().reset_index()
        base_avg.columns = ['F_LINENO', 'F_LINENAME', '基期日均']
        print(f"✓ 基期：{len(base_avg)}个车站")
        
        # 查询历年同期数据
        all_years_data = []
        history_details = []
        
        print(f"\n🔍 开始查询历年同期数据（共{history_years}年）...")
        
        for i in range(1, history_years + 1):
            history_year = predict_year - i
            print(f"\n📅 处理第{i}年：{history_year}年")
            
            if custom_configs and str(history_year) in custom_configs:
                config = custom_configs[str(history_year)]
                history_start = config['ref_start'].replace('-', '')
                history_end = config['ref_end'].replace('-', '')
                history_base_start = config['base_start'].replace('-', '')
                history_base_end = config['base_end'].replace('-', '')
            else:
                history_start = f"{history_year}{predict_start[4:]}"
                history_end = f"{history_year}{predict_end[4:]}"
                
                history_base_month = predict_month - 1
                history_base_year = history_year
                if history_base_month < 1:
                    history_base_month = 12
                    history_base_year -= 1
                
                history_base_start = f"{history_base_year}{str(history_base_month).zfill(2)}01"
                history_last_day = calendar.monthrange(history_base_year, history_base_month)[1]
                history_base_end = f"{history_base_year}{str(history_base_month).zfill(2)}{str(history_last_day).zfill(2)}"
            
            try:
                print(f"  🔸 查询参考期：{history_start} - {history_end}")
                df_history = read_station_daily_flow_history(metric_type, history_start, history_end)
                if df_history.empty:
                    print(f"  ⚠️  {history_year}年参考期数据为空，跳过")
                    continue
                df_history = df_history.rename(columns={'F_KLCOUNT': metric_name})
                print(f"  ✓ 参考期：{len(df_history)}行，{df_history['F_LINENO'].nunique()}个车站")
                
                print(f"  🔸 查询基期：{history_base_start} - {history_base_end}")
                df_history_base = read_station_daily_flow_history(metric_type, history_base_start, history_base_end)
                if df_history_base.empty:
                    print(f"  ⚠️  {history_year}年基期数据为空，跳过")
                    continue
                df_history_base = df_history_base.rename(columns={'F_KLCOUNT': metric_name})
                print(f"  ✓ 基期：{len(df_history_base)}行，{df_history_base['F_LINENO'].nunique()}个车站")
                
                print(f"  🔸 计算{history_year}年基期日均...")
                history_base_avg = df_history_base.groupby(['F_LINENO', 'F_LINENAME'])[metric_name].mean().reset_index()
                history_base_avg.columns = ['F_LINENO', 'F_LINENAME', f'基期日均_{history_year}']
                print(f"  ✓ 基期日均：{len(history_base_avg)}个车站")
                
                print(f"  🔸 合并数据并计算增长率...")
                df_history['年份'] = history_year
                before_merge = len(df_history)
                df_history = df_history.merge(history_base_avg, on=['F_LINENO', 'F_LINENAME'], how='inner')
                after_merge = len(df_history)
                print(f"  ✓ 合并：{before_merge}行 -> {after_merge}行")
                
                df_history['增长率'] = df_history.apply(
                    lambda row: ((row[metric_name] - row[f'基期日均_{history_year}']) / row[f'基期日均_{history_year}'] * 100)
                    if row[f'基期日均_{history_year}'] > 0 else 0, axis=1
                ).round(2)
                print(f"  ✓ 增长率：平均{df_history['增长率'].mean():.2f}%，最大{df_history['增长率'].max():.2f}%")
                
                all_years_data.append(df_history)
                
                history_details.append({
                    'year': int(history_year),
                    'ref_period': f"{history_start[:4]}-{history_start[4:6]}-{history_start[6:]} 至 {history_end[:4]}-{history_end[4:6]}-{history_end[6:]}",
                    'base_period': f"{history_base_start[:4]}-{history_base_start[4:6]}-{history_base_start[6:]} 至 {history_base_end[:4]}-{history_base_end[4:6]}-{history_base_end[6:]}",
                    'line_stats': []
                })
                
                station_count = 0
                for line_no in df_history['F_LINENO'].unique():
                    line_data = df_history[df_history['F_LINENO'] == line_no]
                    history_details[-1]['line_stats'].append({
                        'line_no': str(line_no),
                        'line_name': str(line_data['F_LINENAME'].iloc[0]),
                        'base_avg': int(line_data[f'基期日均_{history_year}'].iloc[0]),
                        'avg_growth': round(float(line_data['增长率'].mean()), 2),
                        'max_growth': round(float(line_data['增长率'].max()), 2)
                    })
                    station_count += 1
                print(f"  ✓ 统计完成：{station_count}个车站")
                
            except Exception as e:
                print(f"  ❌ {history_year}年数据处理异常：{type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_years_data:
            print(f"❌ 没有找到任何历年同期数据")
            return jsonify({'success': False, 'error': '没有找到历年同期车站数据'}), 400
        
        print(f"\n🔄 合并所有历年数据...")
        df_all_history = pd.concat(all_years_data, ignore_index=True)
        print(f"✓ 合并完成：{len(df_all_history)}行")
        
        print(f"🔸 排序并标记天数...")
        df_all_history = df_all_history.sort_values(['年份', 'F_LINENO', 'F_DATE'])
        df_all_history['第几天'] = df_all_history.groupby(['年份', 'F_LINENO']).cumcount() + 1
        print(f"✓ 标记完成：{df_all_history['第几天'].max()}天")
        
        # 计算预测
        predictions = []
        
        print(f"\n🔮 开始预测计算...")
        print(f"📊 待预测车站数：{len(base_avg)}个")
        
        station_processed = 0
        for line_no in base_avg['F_LINENO'].unique():
            station_processed += 1
            line_name = base_avg[base_avg['F_LINENO'] == line_no]['F_LINENAME'].iloc[0]
            base_daily_avg = base_avg[base_avg['F_LINENO'] == line_no]['基期日均'].iloc[0]
            
            if station_processed <= 3 or station_processed % 20 == 0:
                print(f"  🔸 [{station_processed}/{len(base_avg)}] {line_name}（ID:{line_no}，基期日均:{int(base_daily_avg)}）")
            
            line_history = df_all_history[df_all_history['F_LINENO'] == line_no].copy()
            if line_history.empty:
                if station_processed <= 3:
                    print(f"    ⚠️  该车站无历史数据，跳过")
                continue
            
            day_count = 0
            for day_num in sorted(line_history['第几天'].dropna().unique()):
                day_data = line_history[line_history['第几天'] == day_num]
                max_growth_rate = day_data['增长率'].max()
                max_year = day_data.loc[day_data['增长率'].idxmax(), '年份']
                
                predicted_flow = base_daily_avg * (1 + max_growth_rate / 100)
                predict_date_obj = predict_start_date + timedelta(days=int(day_num) - 1)
                predict_date_str = predict_date_obj.strftime('%Y%m%d')
                
                predictions.append({
                    '车站ID': str(line_no),
                    '车站名称': str(line_name),
                    '预测日期': predict_date_str,
                    '第几天': int(day_num),
                    '基期日均': int(base_daily_avg),
                    '最优增长率': round(float(max_growth_rate), 2),
                    '最优来源年份': int(max_year),
                    '预测客流': int(predicted_flow)
                })
                day_count += 1
            
            if station_processed <= 3:
                print(f"    ✓ 预测{day_count}天")
        
        print(f"\n✅ 预测完成：共{len(predictions)}条记录")
        
        # 查询实际数据
        has_actual = False
        print(f"\n📊 查询实际数据进行准确率对比...")
        print(f"🔸 查询日期范围：{predict_start} - {predict_end}")
        try:
            df_actual = read_station_daily_flow_history(metric_type, predict_start, predict_end)
            if not df_actual.empty:
                print(f"✓ 查到实际数据：{len(df_actual)}行，{df_actual['F_LINENO'].nunique()}个车站")
                df_actual = df_actual.rename(columns={'F_KLCOUNT': metric_name})
                df_actual['F_LINENO'] = df_actual['F_LINENO'].astype(str).str.strip()
                df_actual['F_DATE'] = df_actual['F_DATE'].astype(str).str.strip()
                
                print(f"🔸 开始匹配预测数据与实际数据...")
                matched_count = 0
                match_sample_shown = 0
                for pred in predictions:
                    actual_row = df_actual[
                        (df_actual['F_LINENO'] == str(pred['车站ID']).strip()) & 
                        (df_actual['F_DATE'] == str(pred['预测日期']).strip())
                    ]
                    
                    if not actual_row.empty:
                        actual_flow = int(actual_row[metric_name].iloc[0])
                        pred['实际客流'] = actual_flow
                        error_rate = abs(pred['预测客流'] - actual_flow) / actual_flow if actual_flow > 0 else 1
                        pred['准确率'] = round((1 - error_rate) * 100, 2)
                        pred['误差'] = pred['预测客流'] - actual_flow
                        matched_count += 1
                        
                        # 显示前3个匹配示例
                        if match_sample_shown < 3:
                            print(f"    ✓ [{pred['车站名称']}] {pred['预测日期']}: 预测{pred['预测客流']} vs 实际{actual_flow}，准确率{pred['准确率']}%")
                            match_sample_shown += 1
                    else:
                        pred['实际客流'] = None
                        pred['准确率'] = None
                        pred['误差'] = None
                
                has_actual = matched_count > 0
                print(f"✅ 准确率对比完成：{matched_count}/{len(predictions)}条匹配成功")
                
                if matched_count > 0:
                    avg_accuracy = sum([p['准确率'] for p in predictions if p['准确率'] is not None]) / matched_count
                    print(f"📈 平均准确率：{avg_accuracy:.2f}%")
            else:
                print(f"⚠️  未查到实际数据，无法进行准确率对比")
        except Exception as e:
            print(f"⚠️  查询实际数据异常：{type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"🎉 车站预测完成！")
        print(f"  📊 预测记录数：{len(predictions)}条")
        print(f"  🎯 准确率对比：{'有' if has_actual else '无'}")
        print(f"  📅 预测期：{predict_start[:4]}-{predict_start[4:6]}-{predict_start[6:]} 至 {predict_end[:4]}-{predict_end[4:6]}-{predict_end[6:]}")
        print(f"  📆 基期：{base_start[:4]}-{base_start[4:6]}-{base_start[6:]} 至 {base_end[:4]}-{base_end[4:6]}-{base_end[6:]}")
        print(f"  🔄 历史年限：{history_years}年")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'predictions': fix_chinese_encoding(predictions),
            'has_actual': has_actual,
            'metric_name': metric_name,
            'base_period': f"{base_start[:4]}-{base_start[4:6]}-{base_start[6:]} 至 {base_end[:4]}-{base_end[4:6]}-{base_end[6:]}",
            'predict_period': f"{predict_start[:4]}-{predict_start[4:6]}-{predict_start[6:]} 至 {predict_end[:4]}-{predict_end[4:6]}-{predict_end[6:]}",
            'history_years': history_years,
            'history_details': fix_chinese_encoding(history_details)
        })
    
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ 车站预测失败！")
        print(f"错误类型：{type(e).__name__}")
        print(f"错误信息：{str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # 关闭debug模式，避免文件监控导致服务重启
    app.run(debug=False, host='0.0.0.0', port=4566)


