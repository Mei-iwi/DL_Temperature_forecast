from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .config import (
        DATE_COL,
        MODEL_PATH,
        PREDICTIONS_TEST_PATH,
        REGRESSION_METRICS_PATH,
        RESIDUAL_PLOT_PATH,
        SCALER_PARAMS_PATH,
        TEST_CSV_PATH,
        TEST_SCALED_PATH,
    )
    from .visualize_results import plot_actual_vs_predicted, plot_residuals
    from .windowing import create_sequences, load_series_values
except ImportError:  # pragma: no cover - hỗ trợ chạy trực tiếp file
    from config import DATE_COL, MODEL_PATH, PREDICTIONS_TEST_PATH, REGRESSION_METRICS_PATH, RESIDUAL_PLOT_PATH, SCALER_PARAMS_PATH, TEST_CSV_PATH, TEST_SCALED_PATH
    from visualize_results import plot_actual_vs_predicted, plot_residuals
    from windowing import create_sequences, load_series_values


def load_lstm_model(model_path: str | Path = MODEL_PATH):
    """Nạp mô hình LSTM đã huấn luyện từ file `.keras`."""
    from tensorflow.keras.models import load_model

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Thiếu model {path}. Hãy chạy: python main.py train")
    return load_model(path)


def load_scaler_params(path: str | Path = SCALER_PARAMS_PATH) -> dict[str, Any]:
    """Đọc mean/std đã fit từ tập train trong `scaler_params.json`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inverse_scale_temperature(values: Any, scaler_params: dict[str, Any]) -> np.ndarray:
    """Đảo chuẩn hóa để đưa nhiệt độ về đơn vị ban đầu.

    Cần đảo chuẩn hóa vì model học trên dữ liệu scaled, còn báo cáo cần nhiệt độ thật.
    """
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return arr * float(scaler_params["std"]) + float(scaler_params["mean"])


def compute_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Tính MAE, MSE, RMSE cho bài toán hồi quy nhiệt độ."""
    true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    errors = true - pred
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors**2))
    rmse = float(math.sqrt(mse))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def build_prediction_dataframe(dates: Any, y_true: Any, y_pred: Any) -> pd.DataFrame:
    """Tạo bảng dự đoán test gồm ngày, giá trị thật, giá trị dự đoán và sai số."""
    import pandas as pd

    true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates).astype(str),
            "y_true": true,
            "y_pred": pred,
            "error": true - pred,
        }
    )


def evaluate_lstm() -> dict[str, float]:
    """Đánh giá mô hình LSTM trên tập test.

    Hàm chỉ dùng test set để đánh giá, dự đoán trên dữ liệu đã scaled, sau đó đảo chuẩn hóa
    y_true/y_pred về nhiệt độ thật trước khi tính MAE, MSE, RMSE.
    """
    import pandas as pd

    test_path = TEST_CSV_PATH if Path(TEST_CSV_PATH).exists() else TEST_SCALED_PATH
    if not Path(test_path).exists():
        raise FileNotFoundError(f"Thiếu {TEST_CSV_PATH}. Hãy chạy: python main.py preprocess")
    model = load_lstm_model(MODEL_PATH)
    scaler_params = load_scaler_params(SCALER_PARAMS_PATH)

    test_df = pd.read_csv(test_path)
    test_values = load_series_values(str(test_path))
    X_test, y_test_scaled = create_sequences(test_values)
    y_pred_scaled = model.predict(X_test, verbose=0)

    y_true = inverse_scale_temperature(y_test_scaled, scaler_params)
    y_pred = inverse_scale_temperature(y_pred_scaled, scaler_params)
    metrics = compute_regression_metrics(y_true, y_pred)

    target_dates = pd.to_datetime(test_df[DATE_COL]).iloc[-len(y_true) :]
    pred_df = build_prediction_dataframe(target_dates, y_true, y_pred)
    REGRESSION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(REGRESSION_METRICS_PATH, index=False)
    pred_df.to_csv(PREDICTIONS_TEST_PATH, index=False)

    plot_actual_vs_predicted(pred_df)
    plot_residuals(pred_df, RESIDUAL_PLOT_PATH)

    print("[OK] Đã đánh giá LSTM")
    print(metrics_df.to_string(index=False))
    return metrics


def evaluate_lstm_pipeline() -> dict[str, float]:
    """Alias để `main.py evaluate` gọi đúng pipeline đánh giá."""
    return evaluate_lstm()


def main_test_evaluate_lstm() -> None:
    """Kiểm tra nhanh metric và đảo chuẩn hóa bằng dữ liệu giả, không cần model thật."""
    scaler = {"mean": 20.0, "std": 2.0}
    restored = inverse_scale_temperature([0.0, 1.0], scaler)
    assert np.allclose(restored, [20.0, 22.0])
    metrics = compute_regression_metrics([1.0, 2.0], [1.5, 1.5])
    assert set(metrics) == {"MAE", "MSE", "RMSE"}
    print("[OK] evaluate_lstm test", metrics)


def main_test_temperature_evaluate_predict() -> None:
    """Alias kiểm tra cũ để tương thích."""
    main_test_evaluate_lstm()


if __name__ == "__main__":
    main_test_evaluate_lstm()
