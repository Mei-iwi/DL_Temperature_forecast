from __future__ import annotations

import numpy as np

try:
    from .config import HORIZON, SCALED_TEMP_COL, TEMP_COL, WINDOW_SIZE
except ImportError:  # pragma: no cover - hỗ trợ chạy trực tiếp file
    from config import HORIZON, SCALED_TEMP_COL, TEMP_COL, WINDOW_SIZE


def create_sequences(
    values: np.ndarray,
    window_size: int = WINDOW_SIZE,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Tạo cửa sổ dữ liệu đầu vào cho mô hình LSTM.

    Tham số `values` là mảng 1 chiều chứa nhiệt độ đã chuẩn hóa theo thứ tự thời gian.
    Hàm dùng `window_size` ngày trước đó để dự đoán giá trị sau `horizon` ngày.
    Trả về `X` có shape `(samples, window_size, 1)` và `y` có shape `(samples, 1)`.

    Lưu ý: dữ liệu đầu vào phải được sắp xếp theo thời gian, không shuffle.
    """
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if len(series) < window_size + horizon:
        raise ValueError("Not enough values to create LSTM sequences.")

    X, y = [], []
    last_start = len(series) - window_size - horizon + 1
    for start in range(last_start):
        end = start + window_size
        target_index = end + horizon - 1
        # Mỗi sample gồm WINDOW_SIZE ngày liên tiếp và nhãn là ngày kế tiếp.
        X.append(series[start:end])
        y.append(series[target_index])

    return np.asarray(X, dtype=np.float32)[..., np.newaxis], np.asarray(y, dtype=np.float32).reshape(-1, 1)


def load_series_values(csv_path: str, temp_col: str = SCALED_TEMP_COL) -> np.ndarray:
    """Đọc một file CSV đã xử lý và trả về chuỗi nhiệt độ dạng float32.

    `csv_path` là đường dẫn file train/validation/test.
    `temp_col` mặc định là `temperature_scaled`; nếu thiếu cột này, hàm thử dùng `temperature`.
    Trả về mảng NumPy 1 chiều để đưa vào `create_sequences()`.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    column = temp_col if temp_col in df.columns else TEMP_COL
    if column not in df.columns:
        raise ValueError(f"CSV must contain {temp_col!r} or {TEMP_COL!r}.")
    return pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=np.float32)


def main_test_windowing() -> None:
    """Kiểm tra nhanh chức năng tạo cửa sổ dữ liệu.

    Hàm dùng mảng giả nhỏ, không cần dữ liệu thật và không huấn luyện model.
    """
    sample = np.arange(20, dtype=np.float32)
    X, y = create_sequences(sample, window_size=7)
    assert X.ndim == 3
    assert X.shape[1] == 7
    assert y.shape[0] == X.shape[0]
    print("[OK] windowing", "X shape:", X.shape, "y shape:", y.shape)


if __name__ == "__main__":
    main_test_windowing()
