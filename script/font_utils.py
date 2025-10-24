# 字体配置模块：处理 matplotlib 的中文字体支持
import os
from matplotlib import font_manager
import matplotlib.pyplot as plt

def get_chinese_font():
    """
    获取可用的中文字体 FontProperties 对象，若无则返回 None
    """
    # 优先检测常见的本地字体文件
    font_candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/arphic/ukai.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        '/System/Library/Fonts/STHeiti Medium.ttc',
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                return font_manager.FontProperties(fname=font_path)
            except Exception:
                continue
    # 若本地文件未找到，尝试常见字体名
    fallback_fonts = [
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "STHeiti",
        "Arial Unicode MS"
    ]
    for font_name in fallback_fonts:
        try:
            return font_manager.FontProperties(family=font_name)
        except Exception:
            continue
    return None

def configure_fonts():
    """
    配置 matplotlib 使用可用的中文字体，避免找不到字体的警告和中文乱码
    并移除 'Noto Sans CJK JP'，彻底去除相关告警
    """
    my_font = get_chinese_font()
    if my_font is not None:
        # 优先使用检测到的中文字体
        font_name = my_font.get_name()
        # 这里必须用 font.sans-serif，否则 Linux 下部分 matplotlib 版本不生效
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['font.family'] = 'sans-serif'
    else:
        # 兜底：常见的可用中文字体，避免Noto Sans CJK JP等不存在字体
        plt.rcParams['font.sans-serif'] = [
            "WenQuanYi Zen Hei",
            "WenQuanYi Micro Hei",
            "SimHei",
            "Microsoft YaHei",
            "STHeiti",
            "Arial Unicode MS"
        ]
        plt.rcParams['font.family'] = 'sans-serif'
    # 处理负号显示
    plt.rcParams['axes.unicode_minus'] = False

    # 彻底移除 'Noto Sans CJK JP'，避免 findfont 警告
    try:
        from matplotlib import rcParams
        # 移除 font.sans-serif 中的 'Noto Sans CJK JP'
        rcParams['font.sans-serif'] = [f for f in rcParams['font.sans-serif'] if f != 'Noto Sans CJK JP']
        # 移除 font.family 中的 'Noto Sans CJK JP'（如果有）
        if isinstance(rcParams['font.family'], list):
            rcParams['font.family'] = [f for f in rcParams['font.family'] if f != 'Noto Sans CJK JP']
        elif isinstance(rcParams['font.family'], str) and rcParams['font.family'] == 'Noto Sans CJK JP':
            rcParams['font.family'] = 'sans-serif'
    except Exception:
        pass
