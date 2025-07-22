# 字体配置模块：处理 matplotlib 的中文字体支持
import os
from matplotlib import font_manager
import matplotlib.pyplot as plt

def get_chinese_font():
    """
    获取中文字体
    
    返回:
        FontProperties 对象或 None（若无合适字体）
    """
    # Linux下常见的中文字体路径
    font_candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/arphic/ukai.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK.ttc',
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
    # 兜底：尝试直接用字体名
    fallback_fonts = [
        "Noto Sans CJK SC",
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
    配置 matplotlib 使用中文字体并处理负号显示
    """
    my_font = get_chinese_font()
    if my_font is not None:
        # 这里必须用 font.sans-serif，否则 Linux 下部分 matplotlib 版本不生效
        plt.rcParams['font.sans-serif'] = [my_font.get_name()]
        plt.rcParams['font.family'] = 'sans-serif'
    else:
        # 兜底：常见字体名
        plt.rcParams['font.sans-serif'] = [
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "WenQuanYi Micro Hei",
            "SimHei",
            "Microsoft YaHei",
            "STHeiti",
            "Arial Unicode MS"
        ]
        plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
