def weather_to_label(weather_str):
    """
    输入天气字符串，输出压缩后的类别数字编码。
    类别编码：
        0: 晴天
        1: 阴天
        2: 雾霾/雾
        3: 小雨
        4: 中到大雨
        5: 雪
        6: 其他
    """
    weather_str = weather_str.strip()
    
    def compress_for_metro(t):
        if "转" in t:
            parts = t.split("转")
            # 取转后天气为主
            main = parts[-1]
            return compress_for_metro(main)
        if " / " in t:
            parts = t.split(" / ")
            def weight(w):
                if w in ["晴", "多云"]:
                    return 1
                if w in ["阴"]:
                    return 2
                if w in ["雾"]:
                    return 3
                if w in ["小雨", "阵雨"]:
                    return 4
                if w in ["中雨", "大雨", "暴雨"]:
                    return 5
                if w in ["小雪", "中雪", "大雪", "冻雨", "雨夹雪"]:
                    return 6
                return 0
            w1, w2 = parts
            return compress_for_metro(w1) if weight(w1) >= weight(w2) else compress_for_metro(w2)
        # 单天气归类
        if t in ["晴", "多云"]:
            return "晴天"
        elif t == "阴":
            return "阴天"
        elif t == "雾":
            return "雾霾/雾"
        elif t in ["小雨", "阵雨"]:
            return "小雨"
        elif t in ["中雨", "大雨", "暴雨"]:
            return "中到大雨"
        elif t in ["小雪", "中雪", "大雪", "冻雨", "雨夹雪"]:
            return "雪"
        else:
            return "其他"
    
    label_map = {
        "晴天": 0,
        "阴天": 1,
        "雾霾/雾": 2,
        "小雨": 3,
        "中到大雨": 4,
        "雪": 5,
        "其他": 6
    }
    
    compressed = compress_for_metro(weather_str)
    return label_map.get(compressed, 6)  # 默认返回“其他”类别编码6

# 测试示例
for w in [
    "晴 / 多云",
    "阴转雾",
    "小雨转阴",
    "中雨 / 大雨",
    "雨夹雪 / 小雪",
    "暴雨转大雨",
    "未知天气"
]:
    print(f"{w} => 类别编号: {weather_to_label(w)}")
