# 系统启动入口：运行 Streamlit 应用
try:
    from streamlit_app import main
except ModuleNotFoundError:
    from STFS_V1.streamlit_app import main
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

if __name__ == "__main__":
    main()
    # fetch_holiday_features('2025-07-26', 15)