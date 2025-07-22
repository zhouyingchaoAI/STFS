长沙地铁客流预测系统文档
概述
本系统为长沙地铁客流预测算法平台，支持线路小时客流预测（使用 LSTM 或 Prophet 模型）和线路日客流预测（使用 KNN 模型）。系统通过模块化设计，实现了代码的高可维护性、可扩展性和可调试性，能够从数据库读取历史数据，训练模型，生成预测结果，并将结果可视化及存入数据库。
功能

小时客流预测：

使用 LSTM 或 Prophet 模型预测指定日期的每小时客流量。
支持模型训练、预测和结果可视化。
预测结果存储到数据库（LineHourlyFlowPrediction 表）。


日客流预测：

使用 KNN 模型预测未来多天的日客流量（默认 15 天）。
支持模型训练、预测和结果可视化。
预测结果存储到数据库（LineDailyFlowPrediction 表）。


交互式界面：

通过 Streamlit 提供用户友好的 Web 界面，允许用户选择算法、预测日期、操作模式（训练、预测或两者兼顾）等。
支持实时调整训练参数（如 LSTM 的隐藏层大小、KNN 的邻居数等）。


模块化设计：

代码分为多个模块，分别处理配置管理、数据库操作、模型实现、数据预处理、可视化和应用逻辑。
便于单独调试、维护和扩展。



模块结构
项目目录结构如下：
.
├── config_utils.py         # 配置管理（加载和保存 YAML 配置文件）
├── font_utils.py           # 字体配置（支持中文显示）
├── db_utils.py             # 数据库操作（读取历史数据、存储预测结果）
├── lstm_model.py           # LSTM 模型及数据集处理
├── prophet_model.py        # Prophet 模型
├── knn_model.py            # KNN 模型
├── plot_utils.py           # 可视化工具（生成小时和日预测图表）
├── predict_hourly.py       # 小时预测主流程
├── predict_daily.py        # 日预测主流程
├── streamlit_app.py        # Streamlit Web 应用入口
└── main.py                 # 系统启动入口

模块功能描述

config_utils.py：

功能：加载和保存 YAML 配置文件，管理模型版本和训练参数。
主要函数：
load_yaml_config：加载 YAML 配置，若文件不存在则创建默认配置。
save_yaml_config：保存配置到 YAML 文件。
get_version_dir：获取模型存储目录。
get_current_version：获取当前版本号（默认使用当天日期）。


配置文件：
model_config.yaml：小时预测配置（LSTM/Prophet）。
model_config_daily.yaml：日预测配置（KNN）。




font_utils.py：

功能：配置 matplotlib 的中文字体支持，确保图表正确显示中文。
主要函数：
get_chinese_font：查找系统中的中文字体。
configure_fonts：配置 matplotlib 的字体设置。




db_utils.py：

功能：处理数据库连接、数据读取和预测结果存储。
主要函数：
read_line_hourly_flow_history：读取小时客流历史数据。
read_line_daily_flow_history：读取日客流历史数据。
insert_hourly_prediction_to_db：将小时预测结果插入数据库。
upload_prediction_sample：将日预测结果插入数据库。




lstm_model.py：

功能：实现 LSTM 模型，包括数据预处理、模型训练和预测。
主要类：
FlowDataset：PyTorch 数据集类，用于处理 LSTM 输入。
LSTMModel：LSTM 神经网络模型定义。
LSTMFlowPredictor：LSTM 预测器，包含训练和预测逻辑。


方法：
prepare_data：预处理数据，生成 LSTM 输入序列。
train：训练 LSTM 模型并保存。
predict：进行小时客流预测。
save_model_info：保存模型元数据（MSE、MAE 等）。




prophet_model.py：

功能：实现 Prophet 模型，用于小时客流预测。
主要类：
ProphetFlowPredictor：Prophet 预测器。


方法：
prepare_data：将数据转换为 Prophet 所需的格式。
train：训练 Prophet 模型。
predict：进行小时客流预测。
save_model_info：保存模型元数据。




knn_model.py：

功能：实现 KNN 模型，用于日客流预测。
主要类：
KNNFlowPredictor：KNN 预测器。


方法：
prepare_data：基于工作日特征预处理数据。
train：训练 KNN 模型。
predict：进行日客流预测。
save_model_info：保存模型元数据。




plot_utils.py：

功能：生成小时和日预测结果的折线图。
主要函数：
plot_hourly_predictions：绘制小时客流预测图，支持前一日数据的对比。
plot_daily_predictions：绘制日客流预测图，支持历史数据的对比。




predict_hourly.py：

功能：协调小时预测流程，包括数据读取、模型训练/预测和结果可视化。
主要函数：
predict_and_plot_timeseries_flow：执行小时预测主流程，调用 LSTM 或 Prophet 模型。




predict_daily.py：

功能：协调日预测流程，包括数据读取、模型训练/预测和结果可视化。
主要函数：
predict_and_plot_timeseries_flow_daily：执行日预测主流程，调用 KNN 模型。




streamlit_app.py：

功能：提供 Web 界面，允许用户配置预测参数、运行预测并查看结果。
主要函数：
main：Streamlit 应用主入口，包含小时和日预测的选项卡。




main.py：

功能：系统启动入口，调用 Streamlit 应用。
主要函数：
main：运行 Streamlit 应用。





环境要求

Python 版本：3.8 或以上
依赖库：pandas numpy torch pymssql scikit-learn prophet joblib matplotlib streamlit pyyaml


数据库：SQL Server（需配置正确的连接参数，如服务器地址、用户名、密码等）。
字体：支持中文的字体（如 Noto Sans CJK、Microsoft YaHei 等），用于图表显示。

安装依赖：
pip install pandas numpy torch pymssql scikit-learn prophet joblib matplotlib streamlit pyyaml

使用方法

配置数据库：

修改 db_utils.py 中的数据库连接参数（server, user, password, database, port），确保能连接到 SQL Server 数据库。
确保数据库包含以下表：
LineHourlyFlowHistory：小时客流历史数据。
LineDailyFlowHistory：日客流历史数据。
LineHourlyFlowPrediction：小时预测结果存储表。
LineDailyFlowPrediction：日预测结果存储表。




运行系统：
python -m streamlit run main.py


这将启动 Streamlit 应用，浏览器会自动打开 Web 界面（默认地址：http://localhost:8501）。


小时预测：

在“线路小时客流预测”选项卡中：
选择算法（LSTM 或 Prophet）。
选择预测日期。
选择操作模式（训练、预测或两者）。
可选：调整高级训练参数（如 LSTM 的隐藏层大小、学习率等）。
点击“开始小时预测”按钮运行预测。


结果将显示为折线图、表格和总客流量统计，并存储到数据库。


日预测：

在“线路日客流预测”选项卡中：
配置 KNN 的邻居数（n_neighbors）。
选择预测起始日期和预测天数（默认 15 天）。
选择操作模式（训练、预测或两者）。
点击“开始日预测”按钮运行预测。


结果将显示为折线图（包含历史数据对比）、表格和总客流量统计，并存储到数据库。


查看结果：

预测结果图保存在 timeseries_predict_hourly.png（小时预测）或 timeseries_predict_daily.png（日预测）。
模型文件和元数据保存在配置指定的目录（默认：models/hour 或 models/daily）。
预测数据插入到数据库的相应表中。



扩展方式
添加新模型

创建新模型文件（如 xgboost_model.py）：

定义一个新类（如 XGBoostFlowPredictor），实现以下方法：
__init__(model_dir, version, config)：初始化模型。
prepare_data：预处理数据。
train(line_data, line_no)：训练模型，返回 (mse, mae, error)。
predict(line_data, line_no, predict_date)：进行预测，返回 (predictions, error)。
save_model_info：保存模型元数据。


示例：from xgboost import XGBRegressor
class XGBoostFlowPredictor:
    def __init__(self, model_dir: str, version: str, config: Dict):
        self.model_dir = model_dir
        self.version = version
        self.config = config
        self.models = {}
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    # 实现其他方法




修改 predict_hourly.py 或 predict_daily.py：

在 predict_and_plot_timeseries_flow 或 predict_and_plot_timeseries_flow_daily 中添加新模型的支持：if algorithm == 'xgboost':
    predictor = XGBoostFlowPredictor(model_dir, version, config)




更新 Streamlit 界面：

在 streamlit_app.py 的 selectbox 中添加新模型选项：algorithm = st.selectbox("选择算法", options=["lstm", "prophet", "xgboost"], ...)





添加新预测场景

创建新配置文件（如 model_config_weekly.yaml）：

定义新场景的参数（如预测周期、模型参数等）。


创建新预测模块（如 predict_weekly.py）：

参考 predict_hourly.py 或 predict_daily.py 的结构，编写新预测流程。
示例：from .xgboost_model import XGBoostFlowPredictor
def predict_and_plot_timeseries_flow_weekly(file_path, predict_start_date, algorithm='xgboost', ...):
    predictor = XGBoostFlowPredictor(...)
    # 实现预测逻辑




更新 Streamlit 界面：

在 streamlit_app.py 中添加新选项卡：tab1, tab2, tab3 = st.tabs(["线路小时客流预测", "线路日客流预测", "线路周客流预测"])
with tab3:
    st.header("线路周客流预测")
    # 添加界面元素





更改数据库

修改 db_utils.py：

更新数据库连接逻辑（如使用 psycopg2 连接 PostgreSQL）。
修改 SQL 查询以适应新数据库的表结构。
示例：import psycopg2
def read_line_hourly_flow_history():
    conn = psycopg2.connect(
        host="localhost",
        database="metro_db",
        user="user",
        password="password"
    )
    df = pd.read_sql("SELECT * FROM line_hourly_flow_history", conn)
    conn.close()
    return df




更新数据库表结构：

确保新数据库包含必要的表和字段。
调整 insert_hourly_prediction_to_db 和 upload_prediction_sample 中的 SQL 语句。



注意事项

数据完整性：确保数据库中的历史数据包含必要的字段（F_DATE, F_HOUR, F_KLCOUNT, F_LINENO, F_LINENAME），否则预测将失败。
模型存储：模型文件和元数据存储在配置指定的目录中，确保有足够的磁盘空间。
字体支持：如果系统中没有中文字体，图表可能无法正确显示中文标题和标签。
性能：LSTM 模型训练可能需要较长时间，尤其是数据量较大时，建议在 GPU 上运行以加速训练。
错误处理：系统会捕获并显示数据库连接、数据预处理和模型训练/预测中的错误，请检查日志以排查问题。

常见问题

数据库连接失败：

检查 db_utils.py 中的连接参数是否正确。
确保 SQL Server 服务正在运行且网络可达。


图表中文显示乱码：

确保系统中安装了支持中文的字体（如 Noto Sans CJK）。
检查 font_utils.py 中的字体路径是否正确。


模型预测结果为零或错误：

检查输入数据是否为空或缺失关键字段。
确保模型文件存在（可能需要重新训练）。
查看 predict_result 中的 error 字段以获取详细错误信息。


Streamlit 界面无法加载：

确保已安装 Streamlit（pip install streamlit）。
检查端口 8501 是否被占用。



未来改进

多模型集成：支持模型融合（如加权平均 LSTM 和 Prophet 的预测结果）。
实时数据更新：集成实时数据流，动态更新预测模型。
更多特征：在 KNN 或其他模型中加入天气、节假日等外部特征。
自动化调度：实现定时任务，自动运行预测并更新数据库。
