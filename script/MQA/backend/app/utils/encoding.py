"""
编码处理工具模块（参考 get_data_struct.py 和 db_utils.py）
统一处理数据库查询结果的中文编码问题
"""
import re
from typing import Any, Union, List, Dict


def has_chinese(s: str) -> bool:
    """检测字符串是否包含中文字符"""
    if not isinstance(s, str):
        return False
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)


def smart_encode_fix(x: Any) -> str:
    """
    智能编码修复函数（参考 db_utils.py 和 get_data_struct.py）
    处理乱码字符（latin-1编码的中文字符），尝试修复为正确的中文
    
    Args:
        x: 需要修复的值（可能是 bytes、str 或其他类型）
        
    Returns:
        修复后的字符串
    """
    if x is None:
        return ''
    
    if isinstance(x, bytes):
        # 如果是 bytes，先尝试解码
        try:
            x = x.decode('utf-8')
        except:
            try:
                x = x.decode('gbk')
            except:
                return x.decode('utf-8', errors='replace')
    
    if not isinstance(x, str):
        return str(x) if x is not None else ''
    
    # 检查是否包含乱码字符（latin-1编码的中文字符）
    # 乱码特征：字符的 ord 值在 127-255 之间，但不是有效的中文字符
    if any(ord(c) > 127 and ord(c) < 256 for c in x):
        try:
            # 尝试从latin-1重新编码为gbk（SQL Server中文数据通常是GBK）
            fixed = x.encode('latin-1').decode('gbk')
            # 验证修复结果是否包含中文
            if has_chinese(fixed):
                return fixed
        except:
            try:
                # 尝试从latin-1重新编码为utf-8
                fixed = x.encode('latin-1').decode('utf-8')
                if has_chinese(fixed):
                    return fixed
            except:
                pass
    
    return x


def fix_encoding_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    修复字典中所有值的编码问题
    
    Args:
        data: 需要修复的字典
        
    Returns:
        修复后的字典
    """
    if not isinstance(data, dict):
        return data
    
    fixed = {}
    for key, value in data.items():
        # 修复 key
        fixed_key = smart_encode_fix(key)
        
        # 修复 value
        if isinstance(value, str):
            fixed_value = smart_encode_fix(value)
        elif isinstance(value, dict):
            fixed_value = fix_encoding_in_dict(value)
        elif isinstance(value, list):
            fixed_value = fix_encoding_in_list(value)
        else:
            fixed_value = value
        
        fixed[fixed_key] = fixed_value
    
    return fixed


def fix_encoding_in_list(data: List[Any]) -> List[Any]:
    """
    修复列表中所有元素的编码问题
    
    Args:
        data: 需要修复的列表
        
    Returns:
        修复后的列表
    """
    if not isinstance(data, list):
        return data
    
    fixed = []
    for item in data:
        if isinstance(item, str):
            fixed.append(smart_encode_fix(item))
        elif isinstance(item, dict):
            fixed.append(fix_encoding_in_dict(item))
        elif isinstance(item, list):
            fixed.append(fix_encoding_in_list(item))
        else:
            fixed.append(item)
    
    return fixed


def fix_query_results(results: Union[List[Dict], List, Dict]) -> Union[List[Dict], List, Dict]:
    """
    修复查询结果的编码问题
    
    Args:
        results: 查询结果（可能是字典列表、列表或字典）
        
    Returns:
        修复后的查询结果
    """
    if isinstance(results, dict):
        return fix_encoding_in_dict(results)
    elif isinstance(results, list):
        if len(results) > 0 and isinstance(results[0], dict):
            # 字典列表
            return [fix_encoding_in_dict(item) for item in results]
        else:
            # 普通列表
            return fix_encoding_in_list(results)
    else:
        return results
