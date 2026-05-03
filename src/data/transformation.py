import re
import numpy as np
import pandas as pd


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tag_news_row(dt: pd.Timestamp, news: str) -> str:
    """
    Keep the same idea as the notebook:
    premarket if hour < 9, else postmarket.
    """
    news = clean_text(news)
    if pd.isna(dt):
        return f"[UNKNOWN] {news}"

    hour = pd.to_datetime(dt).hour
    tag = "[PREMARKET]" if hour < 9 else "[POSTMARKET]"
    return f"{tag} {news}"


def prepare_news_frame(news_df: pd.DataFrame) -> pd.DataFrame:
    df = news_df.copy()

    df["Datetime"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).copy()

    df["Date"] = df["Datetime"].dt.date
    df["Hour"] = df["Datetime"].dt.hour
    df["PreMarket"] = (df["Hour"] < 9).astype(int)
    df["News"] = df["News"].fillna("").astype(str)
    df["TaggedNews"] = df.apply(lambda r: tag_news_row(r["Datetime"], r["News"]), axis=1)

    return df[["Date", "Datetime", "Hour", "PreMarket", "News", "TaggedNews"]]


def group_news_by_date(news_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        news_df.groupby("Date")["TaggedNews"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )
    grouped["TaggedNews"] = grouped["TaggedNews"].fillna("").astype(str)
    return grouped


def prepare_prices_frame(prices_df: pd.DataFrame) -> pd.DataFrame:
    df = prices_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def merge_prices_and_news(prices_df: pd.DataFrame, news_by_date_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(prices_df, news_by_date_df, on="Date", how="left")
    merged["TaggedNews"] = merged["TaggedNews"].fillna("")
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged