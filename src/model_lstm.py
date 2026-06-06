from __future__ import annotations

try:
    from .config import LEARNING_RATE, LSTM_UNITS
except ImportError:  # pragma: no cover - hỗ trợ chạy trực tiếp file
    from config import LEARNING_RATE, LSTM_UNITS


def build_lstm_model(
    input_shape: tuple[int, int],
    units: int = LSTM_UNITS,
    learning_rate: float = LEARNING_RATE,
):
    """Xây dựng mô hình LSTM đơn giản cho bài toán hồi quy nhiệt độ.

    `input_shape` thường là `(WINDOW_SIZE, 1)`: số ngày trong cửa sổ và 1 đặc trưng nhiệt độ.
    Hàm trả về model TensorFlow/Keras đã compile với loss MSE và metric MAE.
    """
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            Input(shape=input_shape),
            # LSTM học quan hệ theo thời gian trong chuỗi nhiệt độ.
            LSTM(units),
            Dropout(0.2),
            Dense(32, activation="relu"),
            # Dense(1) xuất ra một giá trị nhiệt độ dự đoán.
            Dense(1),
        ]
    )
    # MSE phù hợp để tối ưu hồi quy, MAE dễ diễn giải sai số trung bình.
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def main_test_lstm_model() -> None:
    """Kiểm tra nhanh việc build model LSTM bằng dữ liệu giả.

    Hàm không huấn luyện model; chỉ kiểm tra output shape nếu TensorFlow đã được cài.
    """
    import numpy as np

    model = build_lstm_model(input_shape=(7, 1))
    dummy = np.zeros((2, 7, 1), dtype=np.float32)
    output = model(dummy, training=False)
    assert tuple(output.shape) == (2, 1)
    model.summary()
    print("[OK] model_lstm output shape:", tuple(output.shape))


def main_test_temperature_lstm_build() -> None:
    """Alias kiểm tra cũ để tương thích với self_test."""
    main_test_lstm_model()


if __name__ == "__main__":
    main_test_lstm_model()
