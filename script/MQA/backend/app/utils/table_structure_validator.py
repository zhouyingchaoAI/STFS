"""
表结构验证器
从 table_structures.txt 解析表结构，验证和修正 SQL 语句中的字段名、类型和格式
"""
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class TableStructureValidator:
    """表结构验证器"""
    
    def __init__(self, file_path: str = None):
        """
        初始化验证器
        
        Args:
            file_path: table_structures.txt 文件路径
        """
        if file_path is None:
            # 从项目根目录查找 table_structures.txt
            current_dir = Path(__file__).parent.parent.parent.parent
            file_path = current_dir / "table_structures.txt"
            
            # 如果不存在，尝试从 backend 目录的父目录查找
            if not file_path.exists():
                file_path = current_dir.parent / "table_structures.txt"
        
        self.file_path = file_path
        self.table_structures = {}  # {table_name: {field_name: {type, format, ...}}}
        self.table_databases = {}  # {table_name: database_name}
        self.date_fields = {}  # {table_name: {field_name: {type: int/date, format: YYYYMMDD/'YYYY-MM-DD'}}}
        self._load_table_structures()
    
    def _load_table_structures(self):
        """从 table_structures.txt 加载表结构"""
        if not Path(self.file_path).exists():
            logger.warning(f"Table structures file not found: {self.file_path}")
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self._parse_table_structures(content)
            logger.info(f"Loaded {len(self.table_structures)} tables from {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to load table structures: {e}")
    
    def _parse_table_structures(self, content: str):
        """解析表结构内容"""
        lines = content.split('\n')
        current_db = None
        current_table = None
        current_fields = {}
        field_names = []
        example_values = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 检测数据库
            if line.startswith('# 数据库:'):
                # 保存上一个表（在切换数据库前）
                if current_table and current_fields:
                    self.table_structures[current_table] = current_fields
                    self.table_databases[current_table] = current_db
                
                current_db = line.replace('# 数据库:', '').strip()
                current_table = None
                current_fields = {}
                field_names = []
                example_values = []
                i += 1
                continue
            
            # 检测表
            if line.startswith('表：'):
                # 保存上一个表
                if current_table and current_fields:
                    self.table_structures[current_table] = current_fields
                    self.table_databases[current_table] = current_db
                
                current_table = line.replace('表：', '').strip()
                # 提取表名（去掉 dbo. 前缀）
                table_name = current_table.split('.')[-1] if '.' in current_table else current_table
                # 统一转换为大写，便于匹配（但保留原始大小写用于显示）
                current_table = table_name.upper()
                current_fields = {}
                field_names = []
                example_values = []
                i += 1
                continue
            
            # 检测字段表头
            if line.startswith('字段名'):
                field_names = []
                i += 1
                continue
            
            # 检测示例数据
            if line.startswith('示例数据'):
                parts = line.split('\t')
                if len(parts) > 1:
                    example_values = [p.strip() for p in parts[1:] if p.strip()]
                    # 使用示例数据来验证字段格式
                    self._process_example_data(current_table, field_names, example_values)
                i += 1
                continue
            
            # 处理字段信息行
            if current_table and line and '\t' in line and not line.startswith('示例数据') and not line.startswith('字段名'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    field_name = parts[0].strip()
                    field_type = parts[1].strip() if len(parts) > 1 else ''
                    
                    # 保存字段信息
                    current_fields[field_name] = {
                        'type': field_type,
                        'name': field_name
                    }
                    
                    # 记录日期字段
                    if 'date' in field_name.lower() or 'DATE' in field_name:
                        if current_table not in self.date_fields:
                            self.date_fields[current_table] = {}
                        
                        # 根据类型确定格式
                        if field_type.lower() == 'int':
                            self.date_fields[current_table][field_name] = {
                                'type': 'int',
                                'format': 'YYYYMMDD'
                            }
                        elif field_type.lower() == 'date':
                            self.date_fields[current_table][field_name] = {
                                'type': 'date',
                                'format': "'YYYY-MM-DD'"
                            }
                    
                    # 保存字段名到列表
                    if field_name and field_name not in field_names:
                        field_names.append(field_name)
            
            i += 1
        
        # 保存最后一个表
        if current_table and current_fields:
            self.table_structures[current_table] = current_fields
            self.table_databases[current_table] = current_db
    
    def _process_example_data(self, table_name: str, field_names: List[str], example_values: List[str]):
        """处理示例数据，用于验证字段格式"""
        if not field_names or not example_values:
            return
        
        # 更新日期字段的格式信息（基于示例数据）
        if table_name in self.date_fields:
            for field_name, date_info in self.date_fields[table_name].items():
                if field_name in field_names:
                    idx = field_names.index(field_name)
                    if idx < len(example_values):
                        example_value = example_values[idx]
                        # 根据示例值判断格式
                        if date_info['type'] == 'int':
                            # 整数格式：20220101
                            if '-' not in str(example_value) and len(str(example_value)) == 8:
                                date_info['format'] = 'YYYYMMDD'
                        elif date_info['type'] == 'date':
                            # 日期格式：'2025-11-20'
                            if '-' in str(example_value):
                                date_info['format'] = "'YYYY-MM-DD'"
    
    def get_table_fields(self, table_name: str) -> Dict[str, Dict]:
        """
        获取表的字段信息
        
        Args:
            table_name: 表名（可以带或不带 dbo. 前缀）
            
        Returns:
            字段信息字典 {field_name: {type, name, ...}}
        """
        # 提取纯表名（去掉 dbo. 前缀和数据库前缀）
        clean_table = table_name.split('.')[-1] if '.' in table_name else table_name
        # 转换为大写进行匹配
        clean_table_upper = clean_table.upper()
        
        return self.table_structures.get(clean_table_upper, {})
    
    def get_table_database(self, table_name: str) -> Optional[str]:
        """获取表所属的数据库"""
        clean_table = table_name.split('.')[-1] if '.' in table_name else table_name
        clean_table_upper = clean_table.upper()
        return self.table_databases.get(clean_table_upper)
    
    def get_date_field_info(self, table_name: str, field_name: str) -> Optional[Dict]:
        """
        获取日期字段的格式信息
        
        Args:
            table_name: 表名
            field_name: 字段名
            
        Returns:
            日期字段信息 {type: int/date, format: YYYYMMDD/'YYYY-MM-DD'}
        """
        clean_table = table_name.split('.')[-1] if '.' in table_name else table_name
        clean_table_upper = clean_table.upper()
        if clean_table_upper in self.date_fields:
            return self.date_fields[clean_table_upper].get(field_name)
        return None
    
    def validate_and_fix_sql(self, sql: str) -> Tuple[str, List[str]]:
        """
        验证并修正 SQL 语句
        
        Args:
            sql: 原始 SQL 语句
            
        Returns:
            (修正后的 SQL, 修正日志列表)
        """
        fixed_sql = sql
        fixes = []
        
        # 提取 SQL 中的所有表名
        tables = self._extract_tables_from_sql(sql)
        
        for table_name in tables:
            # 转换为大写进行匹配
            table_name_upper = table_name.upper()
            table_fields = self.get_table_fields(table_name_upper)
            if not table_fields:
                # 表不存在于结构中，记录警告但继续处理
                logger.warning(f"Table {table_name} not found in table structures")
                continue
            
            # 验证和修正字段名（使用原始表名，但内部用大写匹配）
            fixed_sql, field_fixes = self._fix_field_names(fixed_sql, table_name_upper, table_fields)
            fixes.extend(field_fixes)
            
            # 验证和修正日期格式
            fixed_sql, date_fixes = self._fix_date_formats(fixed_sql, table_name_upper)
            fixes.extend(date_fixes)
            
            # 验证和修正 GROUP BY 聚合
            fixed_sql, aggregate_fixes = self._fix_group_by_aggregation(fixed_sql, table_name_upper, table_fields)
            fixes.extend(aggregate_fixes)
        
        # 检查并添加默认的时间排序（在所有表处理完后统一处理）
        fixed_sql, order_fixes = self._add_default_time_order(fixed_sql, tables)
        fixes.extend(order_fixes)
        
        return fixed_sql, fixes
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """从 SQL 中提取表名"""
        tables = []
        
        # 匹配 FROM 子句中的表名
        # 支持格式：FROM table, FROM dbo.table, FROM database.dbo.table
        patterns = [
            r'FROM\s+([\w\.]+)',
            r'JOIN\s+([\w\.]+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, sql, re.IGNORECASE)
            for match in matches:
                table = match.group(1).strip()
                # 提取纯表名（最后一个部分）
                clean_table = table.split('.')[-1]
                # 转换为大写进行去重
                clean_table_upper = clean_table.upper()
                if clean_table_upper not in [t.upper() for t in tables]:
                    tables.append(clean_table)
        
        return tables
    
    def _fix_field_names(self, sql: str, table_name: str, table_fields: Dict) -> Tuple[str, List[str]]:
        """
        修正字段名
        
        Args:
            sql: SQL 语句
            table_name: 表名
            table_fields: 表的字段信息
            
        Returns:
            (修正后的 SQL, 修正日志)
        """
        fixed_sql = sql
        fixes = []
        
        # 获取所有正确的字段名（不区分大小写）
        correct_fields = {field.lower(): field for field in table_fields.keys()}
        
        # 查找 SQL 中可能错误的字段名
        # 匹配 SELECT, WHERE, GROUP BY, ORDER BY 中的字段名
        field_pattern = r'\b([A-Z_][A-Z0-9_]*)\b'
        
        def replace_field(match):
            field = match.group(1)
            field_lower = field.lower()
            
            # 如果字段名不在正确字段列表中，尝试查找相似的字段
            if field_lower not in correct_fields:
                # 常见错误修正规则
                corrections = {
                    'f_klcount': 'F_PKLCOUNT' if 'prediction' in table_name.lower() else 'f_klcount',
                    'f_klcount': 'F_PKLCOUNT' if 'prediction' in table_name.lower() else 'f_klcount',
                    'f_kclcount': 'F_PKLCOUNT' if 'prediction' in table_name.lower() else 'f_klcount',
                    'f_date': 'SQUAD_DATE' if 'station' in table_name.lower() else 'f_date',
                    'f_date': 'F_DATE' if 'prediction' in table_name.lower() else 'f_date',
                }
                
                # 检查是否有已知的修正规则
                for wrong, correct in corrections.items():
                    if wrong.lower() == field_lower:
                        # 验证修正后的字段是否存在于表中
                        if correct.lower() in correct_fields:
                            fixes.append(f"修正字段名: {field} -> {correct} (表: {table_name})")
                            return correct
                
                # 尝试模糊匹配（大小写不敏感）
                for correct_field_lower, correct_field in correct_fields.items():
                    if correct_field_lower == field_lower:
                        fixes.append(f"修正字段名大小写: {field} -> {correct_field} (表: {table_name})")
                        return correct_field
            
            return field
        
        # 在 SELECT, WHERE, GROUP BY, ORDER BY 子句中修正字段名
        # 注意：这里只修正明显的错误，避免误修正
        
        # 1. 预测表的特殊修正（F_KLCCOUNT -> F_PKLCOUNT）
        if 'prediction' in table_name.lower():
            if re.search(r'\bF_KLCCOUNT\b', fixed_sql, re.IGNORECASE):
                fixed_sql = re.sub(r'\bF_KLCCOUNT\b', 'F_PKLCOUNT', fixed_sql, flags=re.IGNORECASE)
                fixes.append(f"修正字段名: F_KLCCOUNT -> F_PKLCOUNT (表: {table_name})")
        
        # 2. 车站表的特殊修正（f_date -> SQUAD_DATE）
        if 'station' in table_name.lower():
            if re.search(r'\bf_date\b', fixed_sql, re.IGNORECASE) and re.search(table_name, fixed_sql, re.IGNORECASE):
                fixed_sql = re.sub(r'\bf_date\b', 'SQUAD_DATE', fixed_sql, flags=re.IGNORECASE)
                fixes.append(f"修正日期字段名: f_date -> SQUAD_DATE (表: {table_name})")
        
        # 3. WeatherHistory 表的字段名修正
        if table_name == 'WEATHERHISTORY':
            # DATE -> F_DATE
            if re.search(r'\bDATE\b(?=\s+(as|,|FROM|WHERE|=))', fixed_sql, re.IGNORECASE):
                fixed_sql = re.sub(r'\bDATE\b(?=\s+(as|,|FROM|WHERE|=))', 'F_DATE', fixed_sql, flags=re.IGNORECASE)
                fixes.append(f"修正字段名: DATE -> F_DATE (表: {table_name})")
            # WEATHER_CONDITION -> F_TQQK
            if re.search(r'\bWEATHER_CONDITION\b', fixed_sql, re.IGNORECASE):
                fixed_sql = re.sub(r'\bWEATHER_CONDITION\b', 'F_TQQK', fixed_sql, flags=re.IGNORECASE)
                fixes.append(f"修正字段名: WEATHER_CONDITION -> F_TQQK (表: {table_name})")
            # RECORD_DATE -> F_DATE
            if re.search(r'\bRECORD_DATE\b', fixed_sql, re.IGNORECASE):
                fixed_sql = re.sub(r'\bRECORD_DATE\b', 'F_DATE', fixed_sql, flags=re.IGNORECASE)
                fixes.append(f"修正字段名: RECORD_DATE -> F_DATE (表: {table_name})")
        
        return fixed_sql, fixes
    
    def _fix_date_formats(self, sql: str, table_name: str) -> Tuple[str, List[str]]:
        """
        修正日期格式
        
        Args:
            sql: SQL 语句
            table_name: 表名
            
        Returns:
            (修正后的 SQL, 修正日志)
        """
        fixed_sql = sql
        fixes = []
        
        clean_table = table_name.split('.')[-1] if '.' in table_name else table_name
        clean_table_upper = clean_table.upper()
        
        # 获取该表的所有日期字段信息
        if clean_table_upper not in self.date_fields:
            return fixed_sql, fixes
        
        date_fields_info = self.date_fields[clean_table_upper]
        
        for field_name, date_info in date_fields_info.items():
            field_type = date_info['type']
            expected_format = date_info['format']
            
            # 根据字段类型修正格式
            if field_type == 'int':
                # int 类型：应该是 YYYYMMDD 整数格式
                # 修正 'YYYY-MM-DD' 格式为 YYYYMMDD
                def fix_int_date(match):
                    date_str = match.group(1)
                    if '-' in date_str:
                        date_str = date_str.replace("'", "").replace('"', '')
                        parts = date_str.split('-')
                        if len(parts) == 3:
                            fixed_date = f"{parts[0]}{parts[1]}{parts[2]}"
                            fixes.append(f"修正日期格式: {field_name} = '{date_str}' -> {field_name} = {fixed_date} (表: {table_name})")
                            return f"{field_name} = {fixed_date}"
                    return match.group(0)
                
                # 匹配 field_name = 'YYYY-MM-DD' 或 field_name='YYYY-MM-DD'
                pattern = rf'{re.escape(field_name)}\s*=\s*(["\']?\d{{4}}-\d{{2}}-\d{{2}}["\']?)'
                fixed_sql = re.sub(pattern, fix_int_date, fixed_sql, flags=re.IGNORECASE)
                
                # 修正日期范围
                def fix_int_date_range(match):
                    start_str = match.group(1).replace("'", "").replace('"', '')
                    end_str = match.group(2).replace("'", "").replace('"', '')
                    start_parts = start_str.split('-')
                    end_parts = end_str.split('-')
                    if len(start_parts) == 3 and len(end_parts) == 3:
                        start_int = f"{start_parts[0]}{start_parts[1]}{start_parts[2]}"
                        end_int = f"{end_parts[0]}{end_parts[1]}{end_parts[2]}"
                        fixes.append(f"修正日期范围格式: {field_name} (表: {table_name})")
                        return f"{field_name} >= {start_int} AND {field_name} <= {end_int}"
                    return match.group(0)
                
                # 匹配 field_name >= 'YYYY-MM-DD' AND field_name <= 'YYYY-MM-DD'
                pattern = rf'{re.escape(field_name)}\s*>=\s*(["\']?\d{{4}}-\d{{2}}-\d{{2}}["\']?)\s+AND\s+{re.escape(field_name)}\s*<=\s*(["\']?\d{{4}}-\d{{2}}-\d{{2}}["\']?)'
                fixed_sql = re.sub(pattern, fix_int_date_range, fixed_sql, flags=re.IGNORECASE)
                
                pattern = rf'{re.escape(field_name)}\s*BETWEEN\s*(["\']?\d{{4}}-\d{{2}}-\d{{2}}["\']?)\s+AND\s+(["\']?\d{{4}}-\d{{2}}-\d{{2}}["\']?)'
                fixed_sql = re.sub(pattern, fix_int_date_range, fixed_sql, flags=re.IGNORECASE)
            
            elif field_type == 'date':
                # date 类型：应该是 'YYYY-MM-DD' 字符串格式
                # 修正 YYYYMMDD 整数格式为 'YYYY-MM-DD'
                def fix_date_format(match):
                    date_int = match.group(1)
                    if len(date_int) == 8 and date_int.isdigit():
                        year = date_int[:4]
                        month = date_int[4:6]
                        day = date_int[6:8]
                        fixed_date = f"'{year}-{month}-{day}'"
                        fixes.append(f"修正日期格式: {field_name} = {date_int} -> {field_name} = {fixed_date} (表: {table_name})")
                        return f"{field_name} = {fixed_date}"
                    return match.group(0)
                
                # 匹配 field_name = YYYYMMDD
                pattern = rf'{re.escape(field_name)}\s*=\s*(\d{{8}})'
                fixed_sql = re.sub(pattern, fix_date_format, fixed_sql, flags=re.IGNORECASE)
        
        return fixed_sql, fixes
    
    def _fix_group_by_aggregation(self, sql: str, table_name: str, table_fields: Dict) -> Tuple[str, List[str]]:
        """
        修正 GROUP BY 聚合问题
        如果使用了 GROUP BY，但 SELECT 中的数值字段没有使用聚合函数，自动添加 SUM()
        
        Args:
            sql: SQL 语句
            table_name: 表名
            table_fields: 表的字段信息
            
        Returns:
            (修正后的 SQL, 修正日志)
        """
        fixed_sql = sql
        fixes = []
        
        # 检查是否使用了 GROUP BY
        group_by_match = re.search(r'\bGROUP\s+BY\s+(.+?)(?:\s+ORDER\s+BY|\s*$)', fixed_sql, re.IGNORECASE | re.DOTALL)
        if not group_by_match:
            return fixed_sql, fixes
        
        group_by_clause = group_by_match.group(1).strip()
        group_by_fields = [f.strip().upper() for f in re.split(r',', group_by_clause)]
        # 清理 GROUP BY 字段名（去掉表前缀）
        group_by_fields_clean = []
        for gb_field in group_by_fields:
            clean_field = re.sub(r'^[\w\.]+\.', '', gb_field, flags=re.IGNORECASE)
            group_by_fields_clean.append(clean_field.upper())
        
        # 提取 SELECT 子句
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', fixed_sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return fixed_sql, fixes
        
        select_clause = select_match.group(1).strip()
        original_select = select_clause
        
        # 解析 SELECT 中的所有字段表达式（考虑括号内的逗号）
        field_expressions = []
        current_expr = ""
        paren_depth = 0
        for char in select_clause:
            if char == '(':
                paren_depth += 1
                current_expr += char
            elif char == ')':
                paren_depth -= 1
                current_expr += char
            elif char == ',' and paren_depth == 0:
                if current_expr.strip():
                    field_expressions.append(current_expr.strip())
                current_expr = ""
            else:
                current_expr += char
        if current_expr.strip():
            field_expressions.append(current_expr.strip())
        
        # 分析每个字段表达式
        modified_expressions = []
        fields_to_add_to_group_by = []
        
        for expr in field_expressions:
            expr_original = expr
            expr_upper = expr.upper()
            
            # 检查是否包含聚合函数
            has_aggregate = bool(re.search(r'\b(SUM|COUNT|AVG|MAX|MIN)\s*\(', expr_upper))
            
            if has_aggregate:
                # 已经聚合，直接保留
                modified_expressions.append(expr)
                continue
            
            # 提取字段名（去掉表前缀和别名）
            field_name_match = re.search(r'(?:[\w\.]+\.)?([A-Z_][A-Z0-9_]*)', expr_upper)
            if not field_name_match:
                # 无法识别字段名，保留原样
                modified_expressions.append(expr)
                continue
            
            field_name = field_name_match.group(1)
            
            # 检查字段是否在 GROUP BY 中
            field_in_group_by = field_name in group_by_fields_clean
            
            if field_in_group_by:
                # 字段在 GROUP BY 中，保留原样
                modified_expressions.append(expr)
            else:
                # 字段不在 GROUP BY 中，需要处理
                # 判断字段类型
                is_numeric = False
                if table_fields and field_name in table_fields:
                    field_type = table_fields[field_name].get('type', '').upper()
                    is_numeric = any(t in field_type for t in ['NUMERIC', 'FLOAT', 'INT', 'DECIMAL', 'REAL', 'BIGINT', 'SMALLINT', 'TINYINT'])
                
                if is_numeric:
                    # 数值字段：判断是否应该使用聚合函数
                    # 某些字段（如 F_BEFPKLCOUNT、F_PRUTE）在同一分组中通常相同，使用 MIN() 或 MAX()
                    # 但对于需要求和的字段（如客流指标），应该使用 SUM()
                    # 根据字段名判断：如果是预测率、前一日等单值字段，使用 MIN()；否则使用 SUM()
                    use_min = False
                    if any(keyword in field_name.upper() for keyword in ['PRUTE', 'RATE', 'BEF', 'BEFORE', 'PREV', 'PREVIOUS']):
                        # 预测率、前一日等字段通常在同一分组中相同，使用 MIN()
                        use_min = True
                    
                    # 保留原有的别名
                    alias_match = re.search(r'\s+AS\s+(\w+)', expr, re.IGNORECASE)
                    if alias_match:
                        alias = alias_match.group(1)
                        if use_min:
                            new_expr = f"MIN({field_name}) AS {alias}"
                            fixes.append(f"非聚合数值字段 {field_name} 已添加聚合函数 MIN() (表: {table_name})")
                        else:
                            new_expr = f"SUM({field_name}) AS {alias}"
                            fixes.append(f"非聚合数值字段 {field_name} 已添加聚合函数 SUM() (表: {table_name})")
                    else:
                        if use_min:
                            new_expr = f"MIN({field_name})"
                            fixes.append(f"非聚合数值字段 {field_name} 已添加聚合函数 MIN() (表: {table_name})")
                        else:
                            new_expr = f"SUM({field_name})"
                            fixes.append(f"非聚合数值字段 {field_name} 已添加聚合函数 SUM() (表: {table_name})")
                    modified_expressions.append(new_expr)
                else:
                    # 非数值字段：添加到 GROUP BY
                    modified_expressions.append(expr)
                    if field_name not in fields_to_add_to_group_by:
                        fields_to_add_to_group_by.append(field_name)
                    fixes.append(f"非聚合字段 {field_name} 需要添加到 GROUP BY 子句 (表: {table_name})")
        
        # 更新 SELECT 子句
        if modified_expressions != field_expressions:
            new_select_clause = ', '.join(modified_expressions)
            fixed_sql = fixed_sql.replace(original_select, new_select_clause)
        
        # 更新 GROUP BY 子句（添加缺失的字段）
        if fields_to_add_to_group_by:
            new_group_by = f"{group_by_clause}, {', '.join(fields_to_add_to_group_by)}"
            fixed_sql = re.sub(
                r'\bGROUP\s+BY\s+' + re.escape(group_by_clause),
                f'GROUP BY {new_group_by}',
                fixed_sql,
                flags=re.IGNORECASE
            )
            fixes.append(f"已添加字段到 GROUP BY: {', '.join(fields_to_add_to_group_by)} (表: {table_name})")
        
        # 第二步：检查预定义的数值字段是否需要聚合（补充逻辑）
        # 定义需要聚合的数值字段（根据表类型）
        numeric_fields = []
        if 'STATION' in table_name:
            # 车站表的数值字段
            numeric_fields = ['PASSENGER_NUM', 'ENTRY_NUM', 'EXIT_NUM', 'CHANGE_NUM', 'FLOW_NUM']
        elif 'LINE' in table_name and 'PREDICTION' in table_name:
            # 线路预测表的数值字段
            numeric_fields = ['F_PKLCOUNT', 'entry_num', 'exit_num', 'change_num', 'flow_num']
        elif 'LINE' in table_name and 'HISTORY' in table_name:
            # 线路历史表的数值字段
            numeric_fields = ['f_klcount', 'entry_num', 'exit_num', 'change_num', 'flow_num']
        
        if not numeric_fields:
            return fixed_sql, fixes
        
        # 提取 SELECT 子句
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', fixed_sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return fixed_sql, fixes
        
        select_clause = select_match.group(1)
        original_select = select_clause
        
        # 检查每个数值字段是否已经使用了聚合函数
        for field in numeric_fields:
            # 匹配字段名（可能带别名，如 PASSENGER_NUM AS 客流量）
            # 使用单词边界确保精确匹配
            field_pattern = rf'\b{re.escape(field)}\b'
            
            # 检查字段是否已经在聚合函数中
            if re.search(rf'\b(SUM|AVG|MAX|MIN|COUNT)\s*\(\s*{re.escape(field)}\s*\)', select_clause, re.IGNORECASE):
                continue  # 已经聚合，跳过
            
            # 检查字段是否在 SELECT 中（可能带别名）
            if re.search(field_pattern, select_clause, re.IGNORECASE):
                # 字段存在但没有聚合，需要添加 SUM()
                # 匹配字段及其别名，如：PASSENGER_NUM AS 客流量 或 PASSENGER_NUM, 或 PASSENGER_NUM,
                def add_sum_aggregate(match):
                    full_match = match.group(0)
                    # 如果已经包含聚合函数，不处理
                    if re.search(r'\b(SUM|AVG|MAX|MIN|COUNT)\s*\(', full_match, re.IGNORECASE):
                        return full_match
                    
                    # 添加 SUM() 聚合
                    # 处理不同的格式：
                    # 1. PASSENGER_NUM AS 客流量 -> SUM(PASSENGER_NUM) AS 客流量
                    # 2. PASSENGER_NUM, -> SUM(PASSENGER_NUM),
                    # 3. PASSENGER_NUM -> SUM(PASSENGER_NUM)
                    
                    # 检查是否有 AS 别名
                    as_match = re.search(rf'\b{re.escape(field)}\b\s+AS\s+(\w+)', full_match, re.IGNORECASE)
                    if as_match:
                        alias = as_match.group(1)
                        fixed = f"SUM({field}) AS {alias}"
                        fixes.append(f"添加聚合函数: {field} -> SUM({field}) (表: {table_name})")
                        return fixed
                    else:
                        # 没有别名，直接添加 SUM()
                        fixed = f"SUM({field})"
                        fixes.append(f"添加聚合函数: {field} -> SUM({field}) (表: {table_name})")
                        return fixed
                
                # 替换字段（考虑各种情况）
                # 匹配：字段名 AS 别名 或 字段名, 或 字段名
                pattern = rf'\b{re.escape(field)}\b(?:\s+AS\s+\w+)?(?=\s*,|\s+FROM|\s+WHERE|\s+GROUP|\s+ORDER|$)'
                select_clause = re.sub(pattern, add_sum_aggregate, select_clause, flags=re.IGNORECASE)
        
        # 如果 SELECT 子句被修改，替换原 SQL
        if select_clause != original_select:
            fixed_sql = fixed_sql.replace(original_select, select_clause)
            logger.info(f"已修正 GROUP BY 聚合问题 (表: {table_name})")
        
        # 第三步：修正错误的 SUM() 聚合（如 F_BEFPKLCOUNT、F_PRUTE 不应该用 SUM）
        # 这些字段在同一分组中通常相同，应该使用 MIN() 而不是 SUM()
        fields_should_use_min = ['F_BEFPKLCOUNT', 'F_PRUTE', 'BEFPKLCOUNT', 'PRUTE']
        for field in fields_should_use_min:
            # 查找 SUM(field) 并替换为 MIN(field)
            pattern = rf'\bSUM\s*\(\s*{re.escape(field)}\s*\)'
            if re.search(pattern, fixed_sql, re.IGNORECASE):
                def replace_with_min(match):
                    return f"MIN({field})"
                fixed_sql = re.sub(pattern, replace_with_min, fixed_sql, flags=re.IGNORECASE)
                fixes.append(f"修正聚合函数: SUM({field}) -> MIN({field}) (表: {table_name})")
        
        return fixed_sql, fixes
    
    def _add_default_time_order(self, sql: str, tables: List[str]) -> Tuple[str, List[str]]:
        """
        添加默认的时间排序
        如果SQL中没有ORDER BY子句，自动添加按时间字段的升序排序
        如果ORDER BY中没有时间字段，添加时间字段作为主要或次要排序
        
        Args:
            sql: SQL 语句
            tables: SQL中使用的表列表
            
        Returns:
            (修正后的 SQL, 修正日志)
        """
        fixed_sql = sql
        fixes = []
        
        # 确定时间字段（根据表类型）
        date_field = None
        for table_name in tables:
            table_upper = table_name.upper()
            
            # 车站表使用 SQUAD_DATE
            if 'STATION' in table_upper:
                date_field = 'SQUAD_DATE'
                break
            # 线路预测表使用 F_DATE
            elif 'PREDICTION' in table_upper and 'LINE' in table_upper:
                date_field = 'F_DATE'
                break
            # 线路历史表使用 f_date
            elif 'HISTORY' in table_upper and 'LINE' in table_upper:
                date_field = 'f_date'
                break
            # 其他表（如CalendarHistory, WeatherHistory等）使用 f_date 或 F_DATE
            elif 'HISTORY' in table_upper or 'FUTURE' in table_upper:
                # 检查表结构中的日期字段
                table_fields = self.get_table_fields(table_upper)
                for field_name in table_fields.keys():
                    if 'date' in field_name.lower() or 'DATE' in field_name:
                        date_field = field_name
                        break
                if date_field:
                    break
        
        # 如果没有找到时间字段，尝试从SQL中查找日期字段
        if not date_field:
            # 尝试从SELECT子句中查找日期字段
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', fixed_sql, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                # 查找常见的日期字段名
                date_patterns = [
                    r'\bSQUAD_DATE\b',
                    r'\bF_DATE\b',
                    r'\bf_date\b',
                    r'\bPREDICT_DATE\b',
                ]
                for pattern in date_patterns:
                    if re.search(pattern, select_clause, re.IGNORECASE):
                        # 提取字段名（去掉AS别名）
                        match = re.search(pattern, select_clause, re.IGNORECASE)
                        if match:
                            date_field = match.group(0)
                            break
        
        if not date_field:
            # 如果没有找到时间字段，记录警告但不强制添加
            logger.warning(f"无法确定时间字段，未添加默认排序。SQL: {sql[:200]}")
            return fixed_sql, fixes
        
        # 检查是否已经有ORDER BY
        order_by_match = re.search(r'\bORDER\s+BY\s+(.+?)(?=\s*(?:;|$))', fixed_sql, re.IGNORECASE | re.DOTALL)
        if order_by_match:
            # 已经有ORDER BY，检查是否包含时间字段
            order_by_clause = order_by_match.group(1)
            # 检查ORDER BY中是否已经包含时间字段（不区分大小写）
            if re.search(rf'\b{re.escape(date_field)}\b', order_by_clause, re.IGNORECASE):
                # 已经包含时间字段，不需要添加
                return fixed_sql, fixes
            else:
                # ORDER BY中没有时间字段，在开头添加时间字段作为主要排序
                # 替换ORDER BY子句，在开头添加时间字段
                new_order_by = f"{date_field} ASC, {order_by_clause}"
                fixed_sql = fixed_sql[:order_by_match.start()] + f"ORDER BY {new_order_by}" + fixed_sql[order_by_match.end():]
                fixes.append(f"在ORDER BY中添加时间排序: {date_field} ASC（作为主要排序）")
                logger.info(f"已在ORDER BY中添加时间排序: {date_field} ASC")
                return fixed_sql, fixes
        
        # 确定时间字段（根据表类型）
        date_field = None
        for table_name in tables:
            table_upper = table_name.upper()
            
            # 车站表使用 SQUAD_DATE
            if 'STATION' in table_upper:
                date_field = 'SQUAD_DATE'
                break
            # 线路预测表使用 F_DATE
            elif 'PREDICTION' in table_upper and 'LINE' in table_upper:
                date_field = 'F_DATE'
                break
            # 线路历史表使用 f_date
            elif 'HISTORY' in table_upper and 'LINE' in table_upper:
                date_field = 'f_date'
                break
            # 其他表（如CalendarHistory, WeatherHistory等）使用 f_date 或 F_DATE
            elif 'HISTORY' in table_upper or 'FUTURE' in table_upper:
                # 检查表结构中的日期字段
                table_fields = self.get_table_fields(table_upper)
                for field_name in table_fields.keys():
                    if 'date' in field_name.lower() or 'DATE' in field_name:
                        date_field = field_name
                        break
                if date_field:
                    break
        
        # 如果没有找到时间字段，尝试从SQL中查找日期字段
        if not date_field:
            # 尝试从SELECT子句中查找日期字段
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', fixed_sql, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                # 查找常见的日期字段名
                date_patterns = [
                    r'\bSQUAD_DATE\b',
                    r'\bF_DATE\b',
                    r'\bf_date\b',
                    r'\bPREDICT_DATE\b',
                ]
                for pattern in date_patterns:
                    if re.search(pattern, select_clause, re.IGNORECASE):
                        # 提取字段名（去掉AS别名）
                        match = re.search(pattern, select_clause, re.IGNORECASE)
                        if match:
                            date_field = match.group(0)
                            break
        
        # 如果没有ORDER BY，添加ORDER BY
        # 在SQL末尾添加ORDER BY（在分号之前，如果没有分号就在末尾）
        if fixed_sql.rstrip().endswith(';'):
            fixed_sql = fixed_sql.rstrip()[:-1] + f" ORDER BY {date_field} ASC;"
        else:
            fixed_sql = fixed_sql.rstrip() + f" ORDER BY {date_field} ASC"
        fixes.append(f"添加默认时间排序: ORDER BY {date_field} ASC")
        logger.info(f"已添加默认时间排序: ORDER BY {date_field} ASC")
        
        return fixed_sql, fixes


# 全局验证器实例（单例模式）
_validator_instance = None

def get_validator() -> TableStructureValidator:
    """获取全局验证器实例"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = TableStructureValidator()
    return _validator_instance
