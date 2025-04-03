# models/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_model(df: pd.DataFrame, save_model: bool = True, model_path: str = "models/random_forest_model.pkl"):
    """
    Train a Random Forest classifier to predict stock trend (up/down).
    
    Features used:
    - Daily_Return, MA_5, STD_5, Lag_1, is_tariff_day
    
    Target:
    - 1 if next day's Close > today's Close, else 0
    """
    df = df.copy()

    # === Step 1: Define the target ===
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    # === Step 2: Select features ===
    features = [
    "Daily_Return", "MA_5", "STD_5", "Lag_1",
    "is_tariff_day", "is_earnings_day", "is_monetary_policy_day"
    ]
    X = df[features]
    y = df["Target"]

    # === Step 3: Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # === Step 4: Train Random Forest ===
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # === Step 5: Evaluate ===
    y_pred = model.predict(X_test)
    print("[INFO] Model Evaluation Report:\n")
    print(classification_report(y_test, y_pred))

    # === Step 6: Save model ===
    if save_model:
        joblib.dump(model, model_path)
        # Save features list
        joblib.dump(features, "models/feature_columns.pkl")
        print(f"[INFO] Model saved to: {model_path}")

    return model
