# data/fetch_prices.py

import os
import json
import yfinance as yf
import pandas as pd
from typing import List


def fetch_data(ticker: str, period: str = "5y", save: bool = True, save_dir: str = "data/saved_data/") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker using yfinance.
    
    Args:
        ticker (str): Stock or index symbol.
        period (str): Time range, e.g., "5y", "1mo", "1d".
        save (bool): Whether to save the resulting DataFrame as a CSV.
        save_dir (str): Directory to save CSV file.
        
    Returns:
        pd.DataFrame: DataFrame with historical data or empty if unavailable.
    """
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            print(f"[WARN] No data returned for {ticker}")
            return pd.DataFrame()

        data.reset_index(inplace=True)  # Ensure 'Date' is a column
        if save:
            os.makedirs(save_dir, exist_ok=True)
            safe_ticker = ticker.replace("^", "")  # Handle index symbols like ^GSPC
            path = os.path.join(save_dir, f"{safe_ticker}.csv")
            data.to_csv(path, index=False)
            print(f"[INFO] Saved data for {ticker} to {path}")

        return data

    except Exception as e:
        print(f"[ERROR] Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def fetch_all_from_json(json_path: str = "data/symbols.json", period: str = "5y", save_dir: str = "data/saved_data/"):
    """
    Fetch and optionally save data for multiple tickers from a JSON list.
    
    JSON format:
    {
        "stocks": ["AAPL", "TSLA"],
        "indices": ["^GSPC", "^DJI"]
    }
    """
    try:
        with open(json_path, "r") as f:
            symbols = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] JSON file not found: {json_path}")
        return

    all_tickers = symbols.get("stocks", []) + symbols.get("indices", [])
    
    for ticker in all_tickers:
        fetch_data(ticker, period=period, save=True, save_dir=save_dir)
