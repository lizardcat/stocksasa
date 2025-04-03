from data.fetch_prices import fetch_data
from features.feature_engineering import generate_features
from data.add_event_flags import add_event_flags
from models.train_model import train_model
from prediction.predict import predict_tomorrow

import pandas as pd

def main():
    # === Step 1: Fetch historical stock data ===
    print("[INFO] Fetching historical data...")
    df = fetch_data("AAPL", period="6mo", save=False)

    if df.empty:
        print("[ERROR] No data fetched. Exiting.")
        return

    # === Step 2: Generate technical features ===
    print("[INFO] Generating features...")
    df = generate_features(df)

    # === Step 3: Add real-world event flags ===
    print("[INFO] Adding event flags...")
    df = add_event_flags(df, events_path="data/events.csv")

    # === Step 4: Preview results ===
    print("[INFO] Final data preview:")
    print(df.tail(10))  # Print last 10 rows with all features

    # === Step 5: Train Model ===
    print("[INFO] Training predictive model...")
    train_model(df)

    # === Step 6: Make a prediction for tomorrow ===
    print("[INFO] Predicting tomorrow's trend...")
    predict_tomorrow("AAPL")

if __name__ == "__main__":
    main()
