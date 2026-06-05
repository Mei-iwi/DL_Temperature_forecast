from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .config import DATA_RAW_PATH, DATE_COL, TEMP_COL
except ImportError:  # pragma: no cover - allows direct script execution
    from config import DATA_RAW_PATH, DATE_COL, TEMP_COL


NASA_DATE_COLUMNS = ("YEAR", "MO", "DY", "DOY")
NASA_TEMP_COLUMNS = ("T2M", "T2M_MAX", "T2M_MIN")


def _detect_skiprows_for_nasa_csv(csv_path: Path) -> int:
    """Return rows to skip when NASA POWER metadata is present."""
    with csv_path.open("r", encoding="utf-8-sig", errors="replace") as file:
        for line_number, line in enumerate(file):
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("-END HEADER-"):
                return line_number + 1
            if upper.startswith("YEAR,") or upper.startswith("DATE,"):
                return line_number
            if line_number >= 200:
                break
    return 0


def read_temperature_csv(csv_path: str | Path = DATA_RAW_PATH) -> pd.DataFrame:
    """Read a raw temperature CSV, including NASA POWER CSV with metadata."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Temperature CSV not found: {path}")

    skiprows = _detect_skiprows_for_nasa_csv(path)
    return pd.read_csv(path, skiprows=skiprows)


def _find_column_case_insensitive(df: pd.DataFrame, candidates: list[str] | tuple[str, ...]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def standardize_temperature_columns(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    temp_col: str = TEMP_COL,
) -> pd.DataFrame:
    """Return a dataframe with canonical date and temperature columns."""
    work = df.copy()
    work.columns = [str(column).strip() for column in work.columns]

    year_col = _find_column_case_insensitive(work, ("YEAR",))
    month_col = _find_column_case_insensitive(work, ("MO", "MONTH"))
    day_col = _find_column_case_insensitive(work, ("DY", "DAY"))
    doy_col = _find_column_case_insensitive(work, ("DOY",))
    nasa_temp_col = _find_column_case_insensitive(work, NASA_TEMP_COLUMNS)

    if year_col and month_col and day_col and nasa_temp_col:
        standardized = pd.DataFrame()
        standardized[date_col] = pd.to_datetime(
            {
                "year": work[year_col],
                "month": work[month_col],
                "day": work[day_col],
            },
            errors="coerce",
        )
        standardized[temp_col] = work[nasa_temp_col]
        return standardized

    if year_col and doy_col and nasa_temp_col:
        standardized = pd.DataFrame()
        years = pd.to_numeric(work[year_col], errors="coerce")
        day_of_year = pd.to_numeric(work[doy_col], errors="coerce")
        standardized[date_col] = pd.to_datetime(years.astype("Int64").astype(str), format="%Y", errors="coerce")
        standardized[date_col] = standardized[date_col] + pd.to_timedelta(day_of_year - 1, unit="D")
        standardized[temp_col] = work[nasa_temp_col]
        return standardized

    source_date_col = _find_column_case_insensitive(work, (date_col, "datetime", "time", "day"))
    source_temp_col = _find_column_case_insensitive(work, (temp_col, "temp", "t2m", "temperature_c"))

    if source_date_col is None or source_temp_col is None:
        raise ValueError(
            "CSV must contain either date/temperature columns or NASA POWER YEAR, DOY, T2M / YEAR, MO, DY, T2M columns."
        )

    return work[[source_date_col, source_temp_col]].rename(
        columns={source_date_col: date_col, source_temp_col: temp_col}
    )


def validate_temperature_frame(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    temp_col: str = TEMP_COL,
) -> None:
    """Validate the minimum schema needed for the time-series pipeline."""
    missing_columns = [column for column in (date_col, temp_col) if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Temperature dataframe is empty.")

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    parsed_temps = pd.to_numeric(df[temp_col], errors="coerce")
    parsed_temps = parsed_temps.mask(parsed_temps == -999)
    if parsed_dates.notna().sum() == 0:
        raise ValueError(f"Column {date_col!r} does not contain valid dates.")
    if parsed_temps.notna().sum() == 0:
        raise ValueError(f"Column {temp_col!r} does not contain valid numeric temperatures.")


def load_temperature_data(
    csv_path: str | Path = DATA_RAW_PATH,
    date_col: str = DATE_COL,
    temp_col: str = TEMP_COL,
) -> pd.DataFrame:
    """Read, standardize, and validate raw temperature data."""
    raw_df = read_temperature_csv(csv_path)
    standardized = standardize_temperature_columns(raw_df, date_col=date_col, temp_col=temp_col)
    validate_temperature_frame(standardized, date_col=date_col, temp_col=temp_col)
    return standardized


def profile_temperature_data(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    temp_col: str = TEMP_COL,
) -> dict[str, Any]:
    """Build a compact data profile for reports and acceptance checks."""
    validate_temperature_frame(df, date_col=date_col, temp_col=temp_col)

    dates = pd.to_datetime(df[date_col], errors="coerce")
    temperatures = pd.to_numeric(df[temp_col], errors="coerce")
    temperatures = temperatures.mask(temperatures == -999)
    valid_dates = dates.dropna()
    valid_temperatures = temperatures.dropna()

    return {
        "rows": int(len(df)),
        "missing_dates": int(dates.isna().sum()),
        "missing_temperatures": int(temperatures.isna().sum()),
        "duplicate_dates": int(dates.duplicated().sum()),
        "start_date": None if valid_dates.empty else valid_dates.min().date().isoformat(),
        "end_date": None if valid_dates.empty else valid_dates.max().date().isoformat(),
        "temperature_min": None if valid_temperatures.empty else float(valid_temperatures.min()),
        "temperature_max": None if valid_temperatures.empty else float(valid_temperatures.max()),
        "temperature_mean": None if valid_temperatures.empty else float(valid_temperatures.mean()),
    }


def save_data_profile(profile: dict[str, Any], output_path: str | Path) -> None:
    """Save a data profile as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


def main_test_temperature_data_pipeline() -> None:
    sample_path = Path(__file__).resolve().parents[1] / "tests" / "sample_temperature.csv"
    df = load_temperature_data(sample_path)
    profile = profile_temperature_data(df)
    print("[OK] data_utils", profile)


if __name__ == "__main__":
    main_test_temperature_data_pipeline()
