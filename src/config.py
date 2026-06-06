from pathlib import Path

"""Cấu hình dùng chung cho project dự đoán nhiệt độ bằng LSTM.

File này chỉ khai báo đường dẫn, tên cột và tham số cơ bản.
Các module khác import cấu hình từ đây để tránh hard-code đường dẫn cục bộ.
"""


# Thư mục gốc của project, dùng để tạo các đường dẫn tương đối ổn định.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Đường dẫn dữ liệu gốc và các output sau bước tiền xử lý.
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "temperature.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DATA_PATH = DATA_PROCESSED_DIR / "temperature_clean.csv"
SPLIT_INFO_PATH = DATA_PROCESSED_DIR / "split_info.json"
TRAIN_SCALED_PATH = DATA_PROCESSED_DIR / "temperature_train_scaled.csv"
VAL_SCALED_PATH = DATA_PROCESSED_DIR / "temperature_val_scaled.csv"
TEST_SCALED_PATH = DATA_PROCESSED_DIR / "temperature_test_scaled.csv"
FULL_SCALED_PATH = DATA_PROCESSED_DIR / "temperature_scaled.csv"

# File model và tham số chuẩn hóa được dùng lại khi evaluate/predict.
MODEL_PATH = PROJECT_ROOT / "models" / "temp_lstm.keras"
SCALER_PARAMS_PATH = PROJECT_ROOT / "models" / "scaler_params.json"

# Hồ sơ dữ liệu phục vụ báo cáo và nghiệm thu.
REPORT_DATA_PROFILE_DIR = PROJECT_ROOT / "reports" / "01_data_profile"
RAW_PROFILE_PATH = REPORT_DATA_PROFILE_DIR / "raw_data_profile.json"
CLEAN_PROFILE_PATH = REPORT_DATA_PROFILE_DIR / "clean_data_profile.json"

# Tên cột chuẩn sau khi đọc dữ liệu CSV.
DATE_COL = "date"
TEMP_COL = "temperature"

# Tham số chuỗi thời gian và huấn luyện; không shuffle dữ liệu khi chia tập.
WINDOW_SIZE = 7
HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
LEARNING_RATE = 0.001
