# 字体配置模块：处理 matplotlib 的中文字体支持
import os
import sys
import warnings
import logging
from matplotlib import font_manager
import matplotlib.pyplot as plt

# 方案1: 最简单有效的方法 - 直接屏蔽所有UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# 方案2: 设置 matplotlib 的日志级别到ERROR
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# 方案3: 如果需要更精确的控制，使用具体的警告消息匹配
specific_warnings = [
    "findfont: Font family 'Noto Sans CJK JP' not found.",
    "findfont: Font family 'Noto Sans CJK SC' not found.",
]

for warning_msg in specific_warnings:
    warnings.filterwarnings("ignore", message=warning_msg)
    warnings.filterwarnings("ignore", category=UserWarning, message=warning_msg)

def get_chinese_font():
    """
    获取中文字体 
    
    返回:
        FontProperties 对象或 None（若无合适字体）
    """
    # 暂时禁用所有警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
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
    配置 matplotlib 使用中文字体并处理负号显示，彻底消除字体相关警告
    """
    # 方案4: 重定向 stderr 来彻底阻止字体警告（最激进的方法）
    original_stderr = sys.stderr
    
    try:
        # 暂时重定向 stderr
        from io import StringIO
        sys.stderr = StringIO()
        
        # 在警告被抑制的环境中配置字体
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
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
            
    finally:
        # 恢复 stderr
        sys.stderr = original_stderr

def configure_fonts_alternative():
    """
    替代方案：更保守的字体配置，只设置已知存在的字体
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 首先检测系统中实际存在的中文字体
        available_fonts = []
        system_fonts = font_manager.findSystemFonts()
        
        # 检查一些常见的中文字体文件
        chinese_font_files = [
            'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 
            'Noto Sans CJK SC', 'STHeiti'
        ]
        
        for font_file in system_fonts:
            try:
                font_prop = font_manager.FontProperties(fname=font_file)
                font_name = font_prop.get_name()
                if any(chinese in font_name for chinese in chinese_font_files):
                    available_fonts.append(font_name)
                    break  # 找到一个就够了
            except:
                continue
        
        if available_fonts:
            plt.rcParams['font.sans-serif'] = available_fonts + ['DejaVu Sans']
        else:
            # 如果没找到，使用默认配置但不会产生警告
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False

# 模块加载时自动配置
configure_fonts()

# 如果上面的方法仍然有警告，可以尝试这个更激进的方法
def suppress_all_font_warnings():
    """
    最后的大招：彻底屏蔽所有matplotlib相关警告
    """
    import matplotlib
    matplotlib.set_loglevel("ERROR")  # 设置matplotlib日志级别为ERROR
    
    # 屏蔽所有UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 如果还有问题，可以重新配置font manager的缓存
    font_manager._rebuild()

# 使用示例
if __name__ == "__main__":
    # 测试中文显示
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
    plt.title('中文标题测试')
    plt.xlabel('X轴标签')
    plt.ylabel('Y轴标签')
    plt.show()