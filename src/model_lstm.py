from __future__ import annotations

try:
    from .config import LEARNING_RATE, LSTM_UNITS
    from .windowing import create_sequences
except ImportError:  # pragma: no cover - supports direct script execution
    from config import LEARNING_RATE, LSTM_UNITS
    from windowing import create_sequences


def build_lstm_model(input_shape: tuple[int, int], units: int = LSTM_UNITS):
    """Build a small LSTM regression model for next-day temperature.

    input_shape is usually (WINDOW_SIZE, 1). The final Dense(1) layer predicts
    one scaled temperature value for the next day.
    """
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    return Sequential(
        [
            Input(shape=input_shape),
            LSTM(units),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )


def compile_lstm_model(model, learning_rate: float = LEARNING_RATE):
    """Compile the LSTM model for a regression task."""
    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main_test_temperature_lstm_build() -> None:
    """Smoke test for sequence creation and LSTM output shape."""
    import numpy as np

    values = np.arange(20, dtype=np.float32)
    X, y = create_sequences(values, window_size=7)
    assert X.shape == (13, 7, 1)
    assert y.shape == (13,)

    model = build_lstm_model(input_shape=(7, 1), units=8)
    compile_lstm_model(model)
    output = model(np.zeros((2, 7, 1), dtype=np.float32), training=False)
    assert tuple(output.shape) == (2, 1)
    model.summary()
    print("[OK] model_lstm", "X shape:", X.shape, "y shape:", y.shape, "output shape:", tuple(output.shape))


def main_test_lstm_model() -> None:
    """Backward-compatible alias used by older self-tests."""
    main_test_temperature_lstm_build()


if __name__ == "__main__":
    main_test_temperature_lstm_build()
