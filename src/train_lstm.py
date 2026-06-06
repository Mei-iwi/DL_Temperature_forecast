from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .config import (
        BATCH_SIZE,
        EPOCHS,
        HORIZON,
        LEARNING_RATE,
        LSTM_HISTORY_CSV_PATH,
        LSTM_LOSS_CURVE_PATH,
        LSTM_MODEL_SUMMARY_PATH,
        LSTM_UNITS,
        MODEL_PATH,
        TRAIN_CSV_PATH,
        TRAIN_SCALED_PATH,
        VAL_CSV_PATH,
        VAL_SCALED_PATH,
        WINDOW_CONFIG_PATH,
        WINDOW_SIZE,
    )
    from .model_lstm import build_lstm_model, compile_lstm_model
    from .visualize_results import plot_training_history
    from .windowing import create_sequences, load_series_values
except ImportError:  # pragma: no cover - supports direct script execution
    from config import (
        BATCH_SIZE,
        EPOCHS,
        HORIZON,
        LEARNING_RATE,
        LSTM_HISTORY_CSV_PATH,
        LSTM_LOSS_CURVE_PATH,
        LSTM_MODEL_SUMMARY_PATH,
        LSTM_UNITS,
        MODEL_PATH,
        TRAIN_CSV_PATH,
        TRAIN_SCALED_PATH,
        VAL_CSV_PATH,
        VAL_SCALED_PATH,
        WINDOW_CONFIG_PATH,
        WINDOW_SIZE,
    )
    from model_lstm import build_lstm_model, compile_lstm_model
    from visualize_results import plot_training_history
    from windowing import create_sequences, load_series_values


def _require_file(path: str | Path, instruction: str) -> Path:
    """Return path if it exists, otherwise raise a clear setup error."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file {file_path}. {instruction}")
    return file_path


def _first_existing(preferred: Path, fallback: Path) -> Path:
    """Use the current processed split name, falling back to the older artifact name."""
    return preferred if Path(preferred).exists() else fallback


def create_training_callbacks(model_path: str | Path = MODEL_PATH, patience: int = 8) -> list[Any]:
    """Create callbacks that reduce overfitting during training."""
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=1e-4,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-5,
        ),
        ModelCheckpoint(
            filepath=str(path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def train_lstm_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    model_path: str | Path = MODEL_PATH,
):
    """Train an LSTM model on prepared windows and validation windows."""
    if getattr(model, "optimizer", None) is None:
        compile_lstm_model(model, learning_rate=LEARNING_RATE)

    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=create_training_callbacks(model_path),
        shuffle=False,
        verbose=1,
    )


def save_lstm_model(model, model_path: str | Path = MODEL_PATH) -> Path:
    """Save a trained LSTM model in Keras format."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def save_lstm_history(history, csv_path: str | Path = LSTM_HISTORY_CSV_PATH) -> Path:
    """Save Keras training history to CSV."""
    import pandas as pd

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history_dict = history.history if hasattr(history, "history") else dict(history)
    pd.DataFrame(history_dict).to_csv(path, index=False)
    return path


def plot_lstm_training(
    history_csv: str | Path = LSTM_HISTORY_CSV_PATH,
    out_path: str | Path = LSTM_LOSS_CURVE_PATH,
) -> Path:
    """Plot train/validation loss and MAE from the saved history CSV."""
    return plot_training_history(history_csv, out_path)


def save_window_config(
    output_path: str | Path,
    X_train,
    X_val,
    train_path: str | Path,
    val_path: str | Path,
) -> Path:
    """Save the window setup used for this training run."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "window_size": int(WINDOW_SIZE),
        "horizon": int(HORIZON),
        "features": int(X_train.shape[2]),
        "input_shape": [int(X_train.shape[1]), int(X_train.shape[2])],
        "target": "next_day_temperature_scaled",
        "train_sequences": int(X_train.shape[0]),
        "validation_sequences": int(X_val.shape[0]),
        "train_source": str(Path(train_path)),
        "validation_source": str(Path(val_path)),
    }
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_model_summary(model, output_path: str | Path = LSTM_MODEL_SUMMARY_PATH) -> Path:
    """Save model.summary() to a text file for the report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def train_lstm() -> dict[str, Path]:
    """Run the complete LSTM training pipeline for member 4."""
    train_path = _first_existing(TRAIN_CSV_PATH, TRAIN_SCALED_PATH)
    val_path = _first_existing(VAL_CSV_PATH, VAL_SCALED_PATH)
    _require_file(train_path, "Run: python main.py preprocess")
    _require_file(val_path, "Run: python main.py preprocess")

    train_values = load_series_values(str(train_path))
    val_values = load_series_values(str(val_path))
    X_train, y_train = create_sequences(train_values, WINDOW_SIZE, HORIZON)
    X_val, y_val = create_sequences(val_values, WINDOW_SIZE, HORIZON)

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), units=LSTM_UNITS)
    compile_lstm_model(model, learning_rate=LEARNING_RATE)

    window_config_path = save_window_config(WINDOW_CONFIG_PATH, X_train, X_val, train_path, val_path)
    summary_path = save_model_summary(model, LSTM_MODEL_SUMMARY_PATH)
    history = train_lstm_model(model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE, MODEL_PATH)
    history_path = save_lstm_history(history, LSTM_HISTORY_CSV_PATH)
    curve_path = plot_lstm_training(history_path, LSTM_LOSS_CURVE_PATH)

    if not Path(MODEL_PATH).exists():
        save_lstm_model(model, MODEL_PATH)

    print("[OK] trained LSTM")
    print("Model:", MODEL_PATH)
    print("Window config:", window_config_path)
    print("Summary:", summary_path)
    print("History:", history_path)
    print("Loss curve:", curve_path)

    return {
        "model": Path(MODEL_PATH),
        "window_config": window_config_path,
        "summary": summary_path,
        "history": history_path,
        "loss_curve": curve_path,
    }


def train_lstm_pipeline() -> dict[str, Path]:
    """Alias for main.py train."""
    return train_lstm()


def main_test_train_lstm() -> None:
    """Smoke test for training helpers without fitting a real model."""
    import numpy as np

    X, y = create_sequences(np.arange(20, dtype=np.float32), window_size=7)
    assert X.shape == (13, 7, 1)
    assert y.shape == (13,)
    print("[OK] train_lstm test", "model path:", MODEL_PATH)


if __name__ == "__main__":
    main_test_train_lstm()
