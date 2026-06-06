from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "temperature.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DATA_PATH = DATA_PROCESSED_DIR / "temperature_clean.csv"
SPLIT_INFO_PATH = DATA_PROCESSED_DIR / "split_info.json"
TRAIN_SCALED_PATH = DATA_PROCESSED_DIR / "temperature_train_scaled.csv"
VAL_SCALED_PATH = DATA_PROCESSED_DIR / "temperature_val_scaled.csv"
TEST_SCALED_PATH = DATA_PROCESSED_DIR / "temperature_test_scaled.csv"
FULL_SCALED_PATH = DATA_PROCESSED_DIR / "temperature_scaled.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "temp_lstm.keras"
SCALER_PARAMS_PATH = PROJECT_ROOT / "models" / "scaler_params.json"
REPORT_DATA_PROFILE_DIR = PROJECT_ROOT / "reports" / "01_data_profile"
RAW_PROFILE_PATH = REPORT_DATA_PROFILE_DIR / "raw_data_profile.json"
CLEAN_PROFILE_PATH = REPORT_DATA_PROFILE_DIR / "clean_data_profile.json"

DATE_COL = "date"
TEMP_COL = "temperature"
WINDOW_SIZE = 7
HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
LEARNING_RATE = 0.001
