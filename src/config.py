from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "temperature.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "temp_lstm.keras"
SCALER_PARAMS_PATH = PROJECT_ROOT / "models" / "scaler_params.json"

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
