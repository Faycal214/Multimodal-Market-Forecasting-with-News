from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.config import TARGET_COLS, LAG_DAYS, TEXT_COL


def _calendar_features(dt) -> dict:
    ts = pd.Timestamp(dt)
    dow = ts.dayofweek
    month = ts.month
    day = ts.day

    return {
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "month_sin": np.sin(2 * np.pi * month / 12.0),
        "month_cos": np.cos(2 * np.pi * month / 12.0),
        "day": float(day),
        "is_month_start": float(ts.is_month_start),
        "is_month_end": float(ts.is_month_end),
        "is_quarter_end": float(ts.is_quarter_end),
    }


def _text_features(text: str) -> dict:
    text = "" if pd.isna(text) else str(text)
    return {
        "news_char_len": float(len(text)),
        "news_word_count": float(len(text.split())),
        "news_count": float(text.count("[PREMARKET]") + text.count("[POSTMARKET]")),
        "premarket_count": float(text.count("[PREMARKET]")),
        "postmarket_count": float(text.count("[POSTMARKET]")),
    }


def _rolling_features(history_prices: pd.DataFrame, price_cols: list[str], windows=(3, 7)) -> dict:
    feats = {}
    hist = history_prices.copy()

    for col in price_cols:
        series = hist[col].astype(float).values
        for w in windows:
            if len(series) >= w:
                past = series[-w:]
                feats[f"{col}_roll_mean_{w}"] = float(np.mean(past))
                feats[f"{col}_roll_std_{w}"] = float(np.std(past))
            else:
                feats[f"{col}_roll_mean_{w}"] = float(np.mean(series)) if len(series) else 0.0
                feats[f"{col}_roll_std_{w}"] = float(np.std(series)) if len(series) else 0.0

    return feats


def build_feature_names(price_cols: list[str] = TARGET_COLS, lag_days: int = LAG_DAYS):
    lag_cols = [f"{col}_lag{lag}" for lag in range(1, lag_days + 1) for col in price_cols]

    extra_cols = [
        "dow_sin", "dow_cos", "month_sin", "month_cos", "day",
        "is_month_start", "is_month_end", "is_quarter_end",
        "news_char_len", "news_word_count", "news_count",
        "premarket_count", "postmarket_count",
    ]

    for col in price_cols:
        for w in (3, 7):
            extra_cols.append(f"{col}_roll_mean_{w}")
            extra_cols.append(f"{col}_roll_std_{w}")

    return lag_cols, extra_cols


def make_feature_row(
    history_prices: pd.DataFrame,
    current_date,
    current_text: str,
    price_cols: list[str] = TARGET_COLS,
    lag_days: int = LAG_DAYS,
) -> dict | None:
    """
    Build one sample using ONLY past history.
    history_prices must contain columns: Date + price1/price2/price3
    and must be sorted ascending.
    """
    if len(history_prices) < lag_days:
        return None

    history_prices = history_prices.sort_values("Date").reset_index(drop=True)

    row = {"Date": current_date, TEXT_COL: "" if pd.isna(current_text) else str(current_text)}

    # lag features
    past = history_prices.tail(lag_days).reset_index(drop=True)
    for lag in range(1, lag_days + 1):
        past_row = past.iloc[-lag]
        for col in price_cols:
            row[f"{col}_lag{lag}"] = float(past_row[col])

    # calendar + text + rolling stats
    row.update(_calendar_features(current_date))
    row.update(_text_features(current_text))
    row.update(_rolling_features(history_prices, price_cols=price_cols, windows=(3, 7)))

    return row


def build_supervised_frame(
    merged_df: pd.DataFrame,
    price_cols: list[str] = TARGET_COLS,
    lag_days: int = LAG_DAYS,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Sequentially constructs samples without leakage.
    For each date t:
      - features use only history up to t-1
      - target is price(t)
    """
    df = merged_df.copy().sort_values("Date").reset_index(drop=True)
    lag_cols, extra_cols = build_feature_names(price_cols=price_cols, lag_days=lag_days)

    rows = []
    history = []

    for _, r in df.iterrows():
        current_date = r["Date"]
        current_text = r.get(TEXT_COL, "")

        if len(history) >= lag_days:
            feat = make_feature_row(
                history_prices=pd.DataFrame(history),
                current_date=current_date,
                current_text=current_text,
                price_cols=price_cols,
                lag_days=lag_days,
            )
            if feat is not None:
                for col in price_cols:
                    feat[col] = float(r[col])
                rows.append(feat)

        # update history with the actual row prices after building features
        history.append(
            {
                "Date": current_date,
                price_cols[0]: float(r[price_cols[0]]),
                price_cols[1]: float(r[price_cols[1]]),
                price_cols[2]: float(r[price_cols[2]]),
            }
        )

    feature_df = pd.DataFrame(rows).reset_index(drop=True)
    return feature_df, lag_cols, extra_cols