from __future__ import annotations

from pathlib import Path

try:
    from .config import ACTUAL_VS_PRED_PATH, RESIDUAL_PLOT_PATH, TEMPERATURE_SERIES_FIGURE_PATH, TRAINING_FIGURE_PATH
except ImportError:  # pragma: no cover - hỗ trợ chạy trực tiếp file
    from config import ACTUAL_VS_PRED_PATH, RESIDUAL_PLOT_PATH, TEMPERATURE_SERIES_FIGURE_PATH, TRAINING_FIGURE_PATH


def _prepare_output_path(out_path: str | Path) -> Path:
    """Tạo thư mục cha trước khi lưu hình."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_history(history_csv: str | Path, out_path: str | Path = TRAINING_FIGURE_PATH) -> Path:
    """Vẽ biểu đồ loss/MAE trong quá trình huấn luyện.

    Đầu vào là `history.csv` do bước train tạo ra.
    Output là file ảnh `training_loss_mae.png`.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    history = pd.read_csv(history_csv)
    path = _prepare_output_path(out_path)
    plt.figure(figsize=(8, 4))
    if "loss" in history:
        plt.plot(history["loss"], label="train loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val loss")
    if "mae" in history:
        plt.plot(history["mae"], label="train MAE")
    if "val_mae" in history:
        plt.plot(history["val_mae"], label="val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Giá trị")
    plt.title("Lịch sử huấn luyện LSTM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_actual_vs_predicted(pred_df, out_path: str | Path = ACTUAL_VS_PRED_PATH) -> Path:
    """Vẽ biểu đồ nhiệt độ thật so với nhiệt độ dự đoán trên tập test."""
    import matplotlib.pyplot as plt

    path = _prepare_output_path(out_path)
    plt.figure(figsize=(9, 4))
    x = pred_df["date"] if "date" in pred_df.columns else pred_df.index
    plt.plot(x, pred_df["y_true"], label="Thực tế")
    plt.plot(x, pred_df["y_pred"], label="Dự đoán")
    plt.xlabel("Ngày")
    plt.ylabel("Nhiệt độ")
    plt.title("So sánh nhiệt độ thực tế và dự đoán")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_residuals(pred_df, out_path: str | Path = RESIDUAL_PLOT_PATH) -> Path:
    """Vẽ biểu đồ phần dư để xem sai số dự đoán trên tập test."""
    import matplotlib.pyplot as plt

    path = _prepare_output_path(out_path)
    residuals = pred_df["y_true"] - pred_df["y_pred"]
    plt.figure(figsize=(8, 4))
    plt.plot(residuals.to_numpy(), label="Residual")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Mẫu test")
    plt.ylabel("Sai số")
    plt.title("Residual plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_temperature_series(df, out_path: str | Path = TEMPERATURE_SERIES_FIGURE_PATH) -> Path:
    """Vẽ chuỗi nhiệt độ theo thời gian sau khi làm sạch dữ liệu."""
    import matplotlib.pyplot as plt
    import pandas as pd

    path = _prepare_output_path(out_path)
    plt.figure(figsize=(9, 4))
    plt.plot(pd.to_datetime(df["date"]), df["temperature"], label="temperature")
    plt.xlabel("Ngày")
    plt.ylabel("Nhiệt độ")
    plt.title("Chuỗi nhiệt độ theo thời gian")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def main_test_visualize_results() -> None:
    """Kiểm tra nhanh module vẽ biểu đồ bằng dữ liệu giả nhỏ."""
    import pandas as pd

    dummy = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "y_true": [1, 2, 3, 4, 5],
            "y_pred": [1.1, 1.9, 3.2, 3.8, 5.1],
            "temperature": [20, 21, 22, 21, 23],
        }
    )
    out_path = Path("outputs") / "figures" / "_test_actual_vs_predicted.png"
    plot_actual_vs_predicted(dummy, out_path)
    print("[OK] visualize_results", out_path)


if __name__ == "__main__":
    main_test_visualize_results()
