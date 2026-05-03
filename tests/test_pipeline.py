import pandas as pd

from src.data.transformation import prepare_news_frame, group_news_by_date, prepare_prices_frame, merge_prices_and_news
from src.features.feature_engineering import build_supervised_frame, make_feature_row


def test_prepare_news_frame_and_grouping():
    news = pd.DataFrame(
        {
            "Date": ["2024-01-01 08:30:00", "2024-01-01 10:15:00", "2024-01-02 07:00:00"],
            "News": ["a", "b", "c"],
        }
    )

    prepared = prepare_news_frame(news)
    assert "TaggedNews" in prepared.columns
    assert prepared.shape[0] == 3

    grouped = group_news_by_date(prepared)
    assert grouped.shape[0] == 2
    assert "TaggedNews" in grouped.columns


def test_prepare_prices_and_merge():
    prices = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "price1": [1.0, 2.0],
            "price2": [3.0, 4.0],
            "price3": [5.0, 6.0],
        }
    )

    news = pd.DataFrame(
        {
            "Date": ["2024-01-01 08:00:00"],
            "News": ["hello"],
        }
    )

    prices_clean = prepare_prices_frame(prices)
    news_clean = prepare_news_frame(news)
    daily_news = group_news_by_date(news_clean)
    merged = merge_prices_and_news(prices_clean, daily_news)

    assert merged.shape[0] == 2
    assert "TaggedNews" in merged.columns


def test_feature_building_and_single_row():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "price1": [float(i) for i in range(1, 13)],
            "price2": [float(i + 10) for i in range(1, 13)],
            "price3": [float(i + 20) for i in range(1, 13)],
            "TaggedNews": ["[PREMARKET] sample news"] * 12,
        }
    )

    feature_df, lag_cols, extra_cols = build_supervised_frame(df, lag_days=7)
    assert not feature_df.empty
    assert len(lag_cols) == 21
    assert len(extra_cols) > 0

    history = df.iloc[:7][["Date", "price1", "price2", "price3"]]
    feat = make_feature_row(
        history_prices=history,
        current_date=df.iloc[7]["Date"],
        current_text="[PREMARKET] sample news",
        lag_days=7,
    )
    assert feat is not None
    assert "price1_lag1" in feat
    assert "news_word_count" in feat