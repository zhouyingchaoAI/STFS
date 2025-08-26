# 系统启动入口：运行 Streamlit 应用
from streamlit_app import main
from db_utils import read_line_daily_flow_history, upload_prediction_sample, fetch_holiday_features

if __name__ == "__main__":
    main()
    # fetch_holiday_features('2025-07-26', 15)