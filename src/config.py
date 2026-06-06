"""Cấu hình dùng chung cho project dự đoán nhiệt độ ngày tiếp theo bằng LSTM.

File này chỉ khai báo đường dẫn, tên cột và tham số cơ bản. Các module khác
import cấu hình từ đây để tránh hard-code đường dẫn cục bộ.
"""

from pathlib import Path


# Thư mục gốc và các thư mục dữ liệu chính của project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATA_PROCESSED_DIR = PROCESSED_DIR

# Thư mục lưu model, output huấn luyện/đánh giá và báo cáo.
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
LOG_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DATA_PROFILE_DIR = REPORT_DIR / "01_data_profile"

# Đường dẫn dữ liệu gốc và dữ liệu đã xử lý.
RAW_CSV_PATH = RAW_DIR / "temperature.csv"
DATA_RAW_PATH = RAW_CSV_PATH
CLEAN_CSV_PATH = PROCESSED_DIR / "temperature_clean.csv"
CLEAN_DATA_PATH = CLEAN_CSV_PATH
TRAIN_CSV_PATH = PROCESSED_DIR / "train_series.csv"
VAL_CSV_PATH = PROCESSED_DIR / "val_series.csv"
TEST_CSV_PATH = PROCESSED_DIR / "test_series.csv"

# Tên file cũ vẫn được giữ để tương thích với các artifact hiện có.
TRAIN_SCALED_PATH = PROCESSED_DIR / "temperature_train_scaled.csv"
VAL_SCALED_PATH = PROCESSED_DIR / "temperature_val_scaled.csv"
TEST_SCALED_PATH = PROCESSED_DIR / "temperature_test_scaled.csv"
FULL_SCALED_PATH = PROCESSED_DIR / "temperature_scaled.csv"
SPLIT_INFO_PATH = PROCESSED_DIR / "split_info.json"

# Đường dẫn model, scaler và các output cuối.
MODEL_PATH = MODEL_DIR / "temperature_lstm.keras"
SCALER_PATH = MODEL_DIR / "scaler_params.json"
SCALER_PARAMS_PATH = SCALER_PATH
HISTORY_CSV_PATH = LOG_DIR / "history.csv"
TRAINING_FIGURE_PATH = FIGURE_DIR / "training_loss_mae.png"
TEMPERATURE_SERIES_FIGURE_PATH = FIGURE_DIR / "temperature_series.png"
ACTUAL_VS_PRED_PATH = FIGURE_DIR / "actual_vs_predicted.png"
RESIDUAL_PLOT_PATH = FIGURE_DIR / "residual_plot.png"
REGRESSION_METRICS_PATH = METRIC_DIR / "regression_metrics.csv"
PREDICTIONS_TEST_PATH = METRIC_DIR / "predictions_test.csv"
NEXT_DAY_PREDICTION_PATH = METRIC_DIR / "next_day_prediction.txt"
RAW_PROFILE_PATH = REPORT_DATA_PROFILE_DIR / "raw_data_profile.json"
CLEAN_PROFILE_PATH = REPORT_DATA_PROFILE_DIR / "clean_data_profile.json"

# Tên cột chuẩn sau khi đọc CSV.
DATE_COL = "date"
TEMP_COL = "temperature"
SCALED_TEMP_COL = f"{TEMP_COL}_scaled"

# Tham số chuỗi thời gian và huấn luyện LSTM.
WINDOW_SIZE = 7
HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
BATCH_SIZE = 32
EPOCHS = 10
LSTM_UNITS = 32
SEED = 42
LEARNING_RATE = 0.001
