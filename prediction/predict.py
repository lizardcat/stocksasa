# prediction/predict.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from data.fetch_prices import fetch_data
from features.feature_engineering import generate_features
from data.add_event_flags import add_event_flags

def predict_tomorrow(ticker: str, model_path: str = "models/random_forest_model.pkl") -> str:
    """
    Load the trained model and predict whether the stock will go up or down tomorrow.
    Returns "Up" or "Down".
    """
    print(f"[INFO] Fetching latest data for {ticker}...")
    df = fetch_data(ticker, period="6mo", save=False)
    if df.empty:
        print("[ERROR] No data available.")
        return "Unknown"

    print("[INFO] Preparing features...")
    df = generate_features(df)
    df = add_event_flags(df)

    # Load trained feature names
    feature_columns = joblib.load("models/feature_columns.pkl")

    # Keep only required columns for prediction
    df = df.dropna(subset=feature_columns)
    latest = df[feature_columns].iloc[-1:].copy()


    print("[INFO] Loading model...")
    model = joblib.load(model_path)

    print("[INFO] Making prediction...")
    prediction = model.predict(latest)[0]
    probas = model.predict_proba(latest)[0]

    print(f"[RESULT] {ticker} is predicted to go {'📈 UP' if prediction == 1 else '📉 DOWN'} tomorrow.")
    print(f"[Confidence] UP: {probas[1]*100:.2f}% | DOWN: {probas[0]*100:.2f}%")

    return "Up" if prediction == 1 else "Down"

def visualize_prediction(df, prediction, ticker):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], label="Close Price", color='blue')

    # Highlight the last point
    last_date = df["Date"].iloc[-1]
    last_close = df["Close"].iloc[-1]
    plt.scatter(
        last_date, last_close,
        color='green' if prediction == "Up" else 'red',
        s=100, label=f"Predicted {prediction}"
    )

    # Optional: show event markers
    event_cols = [col for col in df.columns if col.startswith("is_") and df[col].sum() > 0]
    for col in event_cols:
        highlight = df[df[col] == 1]
        if not highlight.empty:
            plt.scatter(
                highlight["Date"],
                highlight["Close"],
                s=40,
                label=col.replace("is_", "").replace("_", " ").title()
            )

    plt.title(f"{ticker} - Closing Price and Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    st.pyplot(plt.gcf())
    plt.clf() 