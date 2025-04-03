import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from prediction.predict import predict_tomorrow, visualize_prediction
from data.fetch_prices import fetch_data
from features.feature_engineering import generate_features
from data.add_event_flags import add_event_flags

st.set_page_config(page_title="Stocksasa AI", layout="wide")

st.title("📊 AI Stock Market Trend Predictor")

# === Popular Tickers Dropdown ===
popular_tickers = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN",
    "S&P 500 (^GSPC)": "^GSPC",
    "NASDAQ (^IXIC)": "^IXIC",
    "Google (GOOGL)": "GOOGL"
}
ticker_label = st.selectbox("Choose a stock or index:", list(popular_tickers.keys()))
ticker = popular_tickers[ticker_label]

# === Period Selector ===
period = st.selectbox("Select period for analysis:", ["1mo", "3mo", "6mo", "1y", "5y"])

# === Predict Button ===
if st.button("Predict"):
    with st.spinner("Fetching and analyzing data..."):
        df = fetch_data(ticker, period=period, save=False)

        if df.empty:
            st.error("No data found.")
        else:
            df = generate_features(df)
            df = add_event_flags(df)

            prediction = predict_tomorrow(ticker)
            st.success(f"Prediction: **{prediction}** tomorrow")

            # === Chart ===
            st.subheader("📈 Visualized Trend and Prediction")
            visualize_prediction(df, prediction, ticker)

            # === Data Preview ===
            st.subheader("🧾 Last 10 Data Rows")
            last_10 = df.tail(10)

            event_flags = [col for col in last_10.columns if col.startswith("is_")]
            event_flags_to_drop = [col for col in event_flags if last_10[col].sum() == 0]
            last_10.drop(columns=event_flags_to_drop, inplace=True)
            last_10.drop(columns=["Dividends", "Stock Splits"], errors="ignore", inplace=True)
            last_10 = last_10.loc[:, ~last_10.columns.str.match(r"^\d+$")]

            rename_cols = {
                "MA_5": "5-Day MA",
                "STD_5": "5-Day Volatility",
                "Lag_1": "Previous Close",
                "Daily_Return": "Daily % Change"
            }
            last_10.rename(columns=rename_cols, inplace=True)
            st.dataframe(last_10)

            # === Export Button ===
            st.download_button(
                label="Download last 10 rows as CSV",
                data=last_10.to_csv(index=False).encode("utf-8"),
                file_name=f"{ticker}_recent_data.csv",
                mime="text/csv"
            )

            # === Explanation of Prediction Logic ===
            st.subheader("📌 How this prediction is made")
            st.markdown("""
This prediction is generated using a machine learning model (Random Forest) trained on:
- **Technical indicators** like daily returns, moving averages, and volatility
- **Historical patterns** of stock behavior
- **Real-world events** such as tariffs or interest rate changes

The model analyzes recent conditions and compares them to past data to estimate whether the stock is likely to **go up or down tomorrow**.
""")
