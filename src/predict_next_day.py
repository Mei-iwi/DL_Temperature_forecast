from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np

try:
    from .config import CLEAN_CSV_PATH, DATE_COL, MODEL_PATH, NEXT_DAY_PREDICTION_PATH, SCALER_PARAMS_PATH, TEMP_COL, WINDOW_SIZE
    from .evaluate_lstm import inverse_scale_temperature, load_lstm_model, load_scaler_params
except ImportError:  # pragma: no cover - hỗ trợ chạy trực tiếp file
    from config import CLEAN_CSV_PATH, DATE_COL, MODEL_PATH, NEXT_DAY_PREDICTION_PATH, SCALER_PARAMS_PATH, TEMP_COL, WINDOW_SIZE
    from evaluate_lstm import inverse_scale_temperature, load_lstm_model, load_scaler_params


def build_next_day_input(values: np.ndarray, scaler_params: dict[str, float], window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Tạo input shape `(1, WINDOW_SIZE, 1)` cho dự đoán ngày tiếp theo.

    Hàm chỉ lấy `WINDOW_SIZE` ngày cuối cùng, chuẩn hóa bằng scaler đã fit từ train,
    rồi reshape đúng định dạng LSTM. Không dùng bất kỳ dữ liệu tương lai nào.
    """
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(series) < window_size:
        raise ValueError(f"Cần ít nhất {window_size} giá trị nhiệt độ để dự đoán.")
    latest = series[-window_size:]
    scaled = (latest - float(scaler_params["mean"])) / float(scaler_params["std"])
    return scaled.reshape(1, window_size, 1).astype(np.float32)


def calculate_next_date(latest_date: str):
    """Tính ngày tiếp theo từ ngày cuối cùng trong chuỗi dữ liệu."""
    from datetime import datetime

    if hasattr(latest_date, "to_pydatetime"):
        base_date = latest_date.to_pydatetime()
    else:
        base_date = datetime.fromisoformat(str(latest_date).split(" ")[0])
    return base_date + timedelta(days=1)


def predict_next_day() -> float:
    """Dự đoán nhiệt độ ngày tiếp theo bằng model LSTM đã huấn luyện.

    Hàm nạp dữ liệu sạch, lấy WINDOW_SIZE ngày cuối cùng, chuẩn hóa bằng scaler train,
    gọi `model.predict`, đảo chuẩn hóa và lưu kết quả vào `next_day_prediction.txt`.
    """
    import pandas as pd

    if not Path(CLEAN_CSV_PATH).exists():
        raise FileNotFoundError(f"Thiếu {CLEAN_CSV_PATH}. Hãy chạy: python main.py preprocess")

    model = load_lstm_model(MODEL_PATH)
    scaler_params = load_scaler_params(SCALER_PARAMS_PATH)
    clean_df = pd.read_csv(CLEAN_CSV_PATH)
    values = pd.to_numeric(clean_df[TEMP_COL], errors="coerce").dropna().to_numpy(dtype=np.float32)
    X_next = build_next_day_input(values, scaler_params, WINDOW_SIZE)
    pred_scaled = model.predict(X_next, verbose=0)
    pred_temp = float(inverse_scale_temperature(pred_scaled, scaler_params)[0])
    next_date = calculate_next_date(clean_df[DATE_COL].iloc[-1])

    NEXT_DAY_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    message = f"Next prediction date: {next_date.date().isoformat()}\nPredicted temperature: {pred_temp:.4f}\n"
    NEXT_DAY_PREDICTION_PATH.write_text(message, encoding="utf-8")
    print(f"[OK] predicted next day: {next_date.date().isoformat()} -> {pred_temp:.4f}")
    return pred_temp


def run_prediction() -> float:
    """Alias để CLI có thể gọi pipeline dự đoán."""
    return predict_next_day()


def main_test_predict_next_day() -> None:
    """Kiểm tra nhanh tính ngày và reshape input bằng dữ liệu giả, không cần model thật."""
    scaler = {"mean": 10.0, "std": 2.0}
    X = build_next_day_input(np.arange(10, dtype=np.float32), scaler, window_size=7)
    assert X.shape == (1, 7, 1)
    assert calculate_next_date("2024-01-01").date().isoformat() == "2024-01-02"
    print("[OK] predict_next_day test", "X shape:", X.shape)


if __name__ == "__main__":
    main_test_predict_next_day()
