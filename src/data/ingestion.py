import pandas as pd

from src.config.config import (
    RAW_TRAIN_PRICES,
    RAW_TRAIN_NEWS,
    RAW_TEST_PRICES,
    RAW_TEST_NEWS,
)
from src.data.transformation import (
    prepare_news_frame,
    group_news_by_date,
    prepare_prices_frame,
    merge_prices_and_news,
)


def load_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_train_raw(train_prices_path=None, train_news_path=None) -> pd.DataFrame:
    prices_path = train_prices_path or RAW_TRAIN_PRICES
    news_path = train_news_path or RAW_TRAIN_NEWS

    prices = load_csv(prices_path)
    news = load_csv(news_path)

    prices = prepare_prices_frame(prices)
    news = prepare_news_frame(news)
    news_by_date = group_news_by_date(news)

    merged = merge_prices_and_news(prices, news_by_date)
    return merged


def load_test_raw(test_prices_path=None, test_news_path=None) -> pd.DataFrame:
    prices_path = test_prices_path or RAW_TEST_PRICES
    news_path = test_news_path or RAW_TEST_NEWS

    prices = load_csv(prices_path)
    news = load_csv(news_path)

    prices = prepare_prices_frame(prices)
    news = prepare_news_frame(news)
    news_by_date = group_news_by_date(news)

    merged = merge_prices_and_news(prices, news_by_date)
    return merged