from __future__ import annotations

from pathlib import Path

try:
    from .config import BATCH_SIZE, EPOCHS, HISTORY_CSV_PATH, MODEL_PATH, TRAIN_CSV_PATH, TRAINING_FIGURE_PATH, VAL_CSV_PATH
    from .config import TRAIN_SCALED_PATH, VAL_SCALED_PATH
    from .model_lstm import build_lstm_model
    from .visualize_results import plot_training_history
    from .windowing import create_sequences, load_series_values
except ImportError:  # pragma: no cover - hỗ trợ chạy trực tiếp file
    from config import BATCH_SIZE, EPOCHS, HISTORY_CSV_PATH, MODEL_PATH, TRAIN_CSV_PATH, TRAINING_FIGURE_PATH, VAL_CSV_PATH
    from config import TRAIN_SCALED_PATH, VAL_SCALED_PATH
    from model_lstm import build_lstm_model
    from visualize_results import plot_training_history
    from windowing import create_sequences, load_series_values


def _require_file(path: str | Path, instruction: str) -> Path:
    """Kiểm tra file đầu vào trước khi huấn luyện.

    Nếu file thiếu, hàm raise FileNotFoundError với hướng dẫn tiếng Việt rõ ràng.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Thiếu file {file_path}. {instruction}")
    return file_path


def _first_existing(preferred: Path, fallback: Path) -> Path:
    """Chọn file mới nếu có, nếu chưa có thì dùng artifact tên cũ."""
    return preferred if Path(preferred).exists() else fallback


def train_lstm() -> None:
    """Huấn luyện mô hình LSTM từ dữ liệu train/validation đã chuẩn hóa.

    Hàm đọc `train_series.csv` và `val_series.csv`, tạo cửa sổ dữ liệu LSTM,
    build model, gọi `model.fit`, sau đó lưu model, history.csv và biểu đồ huấn luyện.
    Không shuffle thủ công dữ liệu chuỗi thời gian trước khi tạo window.
    """
    import pandas as pd

    train_path = _first_existing(TRAIN_CSV_PATH, TRAIN_SCALED_PATH)
    val_path = _first_existing(VAL_CSV_PATH, VAL_SCALED_PATH)
    _require_file(train_path, "Hãy chạy: python main.py preprocess")
    _require_file(val_path, "Hãy chạy: python main.py preprocess")

    # Đọc chuỗi nhiệt độ đã chuẩn hóa rồi tạo cửa sổ đầu vào cho LSTM.
    train_values = load_series_values(str(train_path))
    val_values = load_series_values(str(val_path))
    X_train, y_train = create_sequences(train_values)
    X_val, y_val = create_sequences(val_values)

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]
    # Huấn luyện model thật; validation_data giữ nguyên thứ tự window đã tạo.
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(MODEL_PATH)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(HISTORY_CSV_PATH, index=False)
    plot_training_history(HISTORY_CSV_PATH, TRAINING_FIGURE_PATH)

    print("[OK] Đã huấn luyện LSTM")
    print("Model:", MODEL_PATH)
    print("History:", HISTORY_CSV_PATH)
    print("Figure:", TRAINING_FIGURE_PATH)


def train_lstm_pipeline() -> None:
    """Alias để `main.py train` gọi đúng pipeline huấn luyện."""
    train_lstm()


def main_test_train_lstm() -> None:
    """Kiểm tra nhanh module train mà không huấn luyện thật.

    Hàm chỉ kiểm tra import, đường dẫn cấu hình và create_sequences bằng dữ liệu giả.
    """
    import numpy as np

    X, y = create_sequences(np.arange(20, dtype="float32"), window_size=7)
    assert X.shape[1:] == (7, 1)
    assert len(y) == len(X)
    print("[OK] train_lstm test", "model path:", MODEL_PATH)


if __name__ == "__main__":
    main_test_train_lstm()
