import pandas as pd

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features to a historical OHLCV DataFrame:
    - Daily return
    - 5-day moving average of Close
    - 5-day rolling standard deviation
    - 1-day lag of Close
    """
    df = df.copy()

    # Daily return (percentage change)
    df["Daily_Return"] = df["Close"].pct_change()

    # 5-day moving average of closing price
    df["MA_5"] = df["Close"].rolling(window=5).mean()

    # 5-day rolling standard deviation (volatility)
    df["STD_5"] = df["Close"].rolling(window=5).std()

    # 1-day lag of closing price
    df["Lag_1"] = df["Close"].shift(1)

    return df
