import pandas as pd
from src.config.config import TARGET_COLS, TEXT_COL


def validate_raw_merged_frame(df: pd.DataFrame) -> None:
    required = {"Date", *TARGET_COLS, TEXT_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    if df["Date"].isna().any():
        raise ValueError("Date column contains NaN.")

    for col in TARGET_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} must be numeric.")