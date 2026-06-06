from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .config import (
        CLEAN_DATA_PATH,
        CLEAN_PROFILE_PATH,
        DATA_RAW_PATH,
        DATE_COL,
        FULL_SCALED_PATH,
        RAW_PROFILE_PATH,
        SCALER_PARAMS_PATH,
        SPLIT_INFO_PATH,
        TEMP_COL,
        TEST_SCALED_PATH,
        TRAIN_RATIO,
        TRAIN_SCALED_PATH,
        VAL_RATIO,
        VAL_SCALED_PATH,
    )
    from .data_utils import load_temperature_data, profile_temperature_data
except ImportError:  # pragma: no cover - allows direct script execution
    from config import (
        CLEAN_DATA_PATH,
        CLEAN_PROFILE_PATH,
        DATA_RAW_PATH,
        DATE_COL,
        FULL_SCALED_PATH,
        RAW_PROFILE_PATH,
        SCALER_PARAMS_PATH,
        SPLIT_INFO_PATH,
        TEMP_COL,
        TEST_SCALED_PATH,
        TRAIN_RATIO,
        TRAIN_SCALED_PATH,
        VAL_RATIO,
        VAL_SCALED_PATH,
    )
    from data_utils import load_temperature_data, profile_temperature_data


SCALED_TEMP_COL = f"{TEMP_COL}_scaled"


def clean_temperature_timeseries(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    temp_col: str = TEMP_COL,
    freq: str = "D",
    fill_method: str = "interpolate",
) -> pd.DataFrame:
    """Clean a daily temperature time series and fill missing days."""
    work = df[[date_col, temp_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[temp_col] = pd.to_numeric(work[temp_col], errors="coerce")
    work.loc[work[temp_col] == -999, temp_col] = pd.NA
    work = work.dropna(subset=[date_col])

    if work.empty:
        raise ValueError("No valid dates remain after cleaning.")

    work[date_col] = work[date_col].dt.normalize()
    work = (
        work.groupby(date_col, as_index=False)[temp_col]
        .mean()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    if freq:
        full_index = pd.date_range(work[date_col].min(), work[date_col].max(), freq=freq)
        work = (
            work.set_index(date_col)
            .reindex(full_index)
            .rename_axis(date_col)
            .reset_index()
        )

    if work[temp_col].isna().all():
        raise ValueError("No valid numeric temperatures remain after cleaning.")

    if fill_method == "interpolate":
        work[temp_col] = work[temp_col].interpolate(method="linear", limit_direction="both")
    elif fill_method == "ffill":
        work[temp_col] = work[temp_col].ffill().bfill()
    elif fill_method == "bfill":
        work[temp_col] = work[temp_col].bfill().ffill()
    elif fill_method == "drop":
        work = work.dropna(subset=[temp_col])
    else:
        raise ValueError("fill_method must be one of: interpolate, ffill, bfill, drop.")

    remaining_missing = int(work[temp_col].isna().sum())
    if remaining_missing:
        raise ValueError(f"Temperature column still has {remaining_missing} missing values.")

    return work.reset_index(drop=True)


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series in chronological order without shuffling."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    work = df.sort_values(DATE_COL).reset_index(drop=True)
    row_count = len(work)
    train_end = int(row_count * train_ratio)
    val_end = train_end + int(row_count * val_ratio)

    if train_end < 1 or val_end <= train_end or val_end >= row_count:
        raise ValueError("Not enough rows to create non-empty train, validation, and test splits.")

    train_df = work.iloc[:train_end].copy().reset_index(drop=True)
    val_df = work.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = work.iloc[val_end:].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def _date_range_info(df: pd.DataFrame, date_col: str = DATE_COL) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "start_date": None, "end_date": None}
    dates = pd.to_datetime(df[date_col], errors="coerce")
    return {
        "rows": int(len(df)),
        "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(),
    }


def build_split_info(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> dict[str, Any]:
    return {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(1 - train_ratio - val_ratio),
        "split_rule": "chronological_no_shuffle",
        "train": _date_range_info(train_df),
        "validation": _date_range_info(val_df),
        "test": _date_range_info(test_df),
    }


def fit_temperature_scaler(
    train_df: pd.DataFrame,
    temp_col: str = TEMP_COL,
) -> dict[str, Any]:
    """Fit standardization parameters on train only."""
    temperatures = pd.to_numeric(train_df[temp_col], errors="coerce").dropna()
    if temperatures.empty:
        raise ValueError("Cannot fit scaler because train temperature data is empty.")

    mean = float(temperatures.mean())
    std = float(temperatures.std(ddof=0))
    if not math.isfinite(std) or std == 0:
        std = 1.0

    return {"method": "standard", "mean": mean, "std": std}


def transform_temperature(
    df: pd.DataFrame,
    scaler_params: dict[str, Any],
    temp_col: str = TEMP_COL,
    output_col: str = SCALED_TEMP_COL,
) -> pd.DataFrame:
    """Apply train-fitted standardization to a dataframe."""
    mean = float(scaler_params["mean"])
    std = float(scaler_params["std"])
    work = df.copy()
    work[output_col] = (pd.to_numeric(work[temp_col], errors="coerce") - mean) / std
    return work


def inverse_transform_temperature(
    values: Any,
    scaler_params: dict[str, Any],
) -> Any:
    """Convert standardized temperatures back to original units."""
    mean = float(scaler_params["mean"])
    std = float(scaler_params["std"])
    try:
        return values * std + mean
    except TypeError:
        return [float(value) * std + mean for value in values]


def scale_time_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    temp_col: str = TEMP_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scaler_params = fit_temperature_scaler(train_df, temp_col=temp_col)
    train_scaled = transform_temperature(train_df, scaler_params, temp_col=temp_col)
    val_scaled = transform_temperature(val_df, scaler_params, temp_col=temp_col)
    test_scaled = transform_temperature(test_df, scaler_params, temp_col=temp_col)
    return train_scaled, val_scaled, test_scaled, scaler_params


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_scaler_params(input_path: str | Path = SCALER_PARAMS_PATH) -> dict[str, Any]:
    return json.loads(Path(input_path).read_text(encoding="utf-8"))


def preprocess_temperature_pipeline(
    csv_path: str | Path = DATA_RAW_PATH,
    save_outputs: bool = True,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> dict[str, Any]:
    """Run the data-cleaning and normalization pipeline for temperature forecasting."""
    raw_df = load_temperature_data(csv_path)
    raw_profile = profile_temperature_data(raw_df)
    clean_df = clean_temperature_timeseries(raw_df)
    clean_profile = profile_temperature_data(clean_df)
    train_df, val_df, test_df = split_by_time(clean_df, train_ratio=train_ratio, val_ratio=val_ratio)
    train_scaled, val_scaled, test_scaled, scaler_params = scale_time_splits(train_df, val_df, test_df)
    split_info = build_split_info(train_scaled, val_scaled, test_scaled, train_ratio=train_ratio, val_ratio=val_ratio)

    full_scaled = pd.concat(
        [
            train_scaled.assign(split="train"),
            val_scaled.assign(split="validation"),
            test_scaled.assign(split="test"),
        ],
        ignore_index=True,
    )

    if save_outputs:
        CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCALER_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(CLEAN_DATA_PATH, index=False)
        train_scaled.to_csv(TRAIN_SCALED_PATH, index=False)
        val_scaled.to_csv(VAL_SCALED_PATH, index=False)
        test_scaled.to_csv(TEST_SCALED_PATH, index=False)
        full_scaled.to_csv(FULL_SCALED_PATH, index=False)
        save_json(split_info, SPLIT_INFO_PATH)
        save_json(scaler_params, SCALER_PARAMS_PATH)
        save_json(raw_profile, RAW_PROFILE_PATH)
        save_json(clean_profile, CLEAN_PROFILE_PATH)

    return {
        "raw_profile": raw_profile,
        "clean_profile": clean_profile,
        "split_info": split_info,
        "scaler_params": scaler_params,
        "clean": clean_df,
        "train": train_scaled,
        "validation": val_scaled,
        "test": test_scaled,
        "full_scaled": full_scaled,
    }


def main() -> None:
    result = preprocess_temperature_pipeline()
    print("[OK] preprocess_timeseries")
    print("clean rows:", result["clean_profile"]["rows"])
    print("date range:", result["clean_profile"]["start_date"], "->", result["clean_profile"]["end_date"])
    print("split:", result["split_info"])
    print("saved:", CLEAN_DATA_PATH)
    print("saved:", SPLIT_INFO_PATH)
    print("saved:", SCALER_PARAMS_PATH)


if __name__ == "__main__":
    main()
