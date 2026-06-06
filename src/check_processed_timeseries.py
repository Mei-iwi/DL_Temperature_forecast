from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from .config import (
        CLEAN_DATA_PATH,
        FULL_SCALED_PATH,
        SCALER_PARAMS_PATH,
        SPLIT_INFO_PATH,
        TEMP_COL,
        TEST_SCALED_PATH,
        TRAIN_SCALED_PATH,
        VAL_SCALED_PATH,
    )
    from .preprocess_timeseries import SCALED_TEMP_COL
except ImportError:  # pragma: no cover - allows direct script execution
    from config import (
        CLEAN_DATA_PATH,
        FULL_SCALED_PATH,
        SCALER_PARAMS_PATH,
        SPLIT_INFO_PATH,
        TEMP_COL,
        TEST_SCALED_PATH,
        TRAIN_SCALED_PATH,
        VAL_SCALED_PATH,
    )
    from preprocess_timeseries import SCALED_TEMP_COL


REQUIRED_FILES = (
    CLEAN_DATA_PATH,
    TRAIN_SCALED_PATH,
    VAL_SCALED_PATH,
    TEST_SCALED_PATH,
    FULL_SCALED_PATH,
    SPLIT_INFO_PATH,
    SCALER_PARAMS_PATH,
)


def _require_file(path: Path) -> None:
    """Kiểm tra một file bắt buộc có tồn tại hay không.

    Tham số `path` là đường dẫn file cần kiểm tra.
    Hàm không trả về dữ liệu; nếu thiếu file sẽ raise FileNotFoundError.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def _check_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    """Đọc CSV và kiểm tra các cột bắt buộc.

    Đầu vào là đường dẫn CSV và danh sách cột cần có.
    Trả về DataFrame nếu file hợp lệ, ngược lại raise ValueError.
    """
    df = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")
    if df.empty:
        raise ValueError(f"{path} is empty.")
    return df


def check_processed_timeseries() -> None:
    """Kiểm tra các artifact sau bước tiền xử lý chuỗi thời gian.

    Hàm xác nhận dữ liệu sạch, dữ liệu đã chuẩn hóa, split_info và scaler_params tồn tại.
    Lưu ý quan trọng: `split_info.json` phải ghi nhận quy tắc `chronological_no_shuffle`.
    """
    for path in REQUIRED_FILES:
        _require_file(Path(path))

    # Kiểm tra dữ liệu sạch có ngày hợp lệ, tăng dần và không thiếu nhiệt độ.
    clean_df = _check_csv(CLEAN_DATA_PATH, ("date", TEMP_COL))
    clean_dates = pd.to_datetime(clean_df["date"], errors="coerce")
    if clean_dates.isna().any():
        raise ValueError("temperature_clean.csv contains invalid dates.")
    if not clean_dates.is_monotonic_increasing:
        raise ValueError("temperature_clean.csv must be sorted by date.")
    if pd.to_numeric(clean_df[TEMP_COL], errors="coerce").isna().any():
        raise ValueError("temperature_clean.csv contains missing temperatures.")

    # Kiểm tra các split đã chuẩn hóa đều có cột temperature_scaled hợp lệ.
    for path in (TRAIN_SCALED_PATH, VAL_SCALED_PATH, TEST_SCALED_PATH, FULL_SCALED_PATH):
        scaled_df = _check_csv(Path(path), ("date", TEMP_COL, SCALED_TEMP_COL))
        if pd.to_numeric(scaled_df[SCALED_TEMP_COL], errors="coerce").isna().any():
            raise ValueError(f"{path} contains missing scaled temperatures.")

    # Xác nhận dữ liệu gộp vẫn giữ nhãn train/validation/test.
    full_scaled = pd.read_csv(FULL_SCALED_PATH)
    if "split" not in full_scaled.columns:
        raise ValueError("temperature_scaled.csv must contain a split column.")
    if set(full_scaled["split"]) != {"train", "validation", "test"}:
        raise ValueError("temperature_scaled.csv split values must be train, validation, and test.")

    # Kiểm tra metadata chứng minh dữ liệu không bị shuffle.
    split_info = json.loads(Path(SPLIT_INFO_PATH).read_text(encoding="utf-8"))
    if split_info.get("split_rule") != "chronological_no_shuffle":
        raise ValueError("split_info.json must use chronological_no_shuffle.")

    # Kiểm tra tham số chuẩn hóa để các bước predict/evaluate có thể đảo chuẩn hóa.
    scaler_params = json.loads(Path(SCALER_PARAMS_PATH).read_text(encoding="utf-8"))
    if "mean" not in scaler_params or "std" not in scaler_params:
        raise ValueError("scaler_params.json must contain mean and std.")

    print("[OK] check_processed_timeseries")
    print("clean rows:", len(clean_df))
    print("splits:", split_info["train"]["rows"], split_info["validation"]["rows"], split_info["test"]["rows"])
    print("scaler mean/std:", scaler_params["mean"], scaler_params["std"])


if __name__ == "__main__":
    check_processed_timeseries()
