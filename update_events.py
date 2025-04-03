# update_events.py

import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("NEWSAPI_KEY")
ENDPOINT = "https://newsapi.org/v2/top-headlines"
KEYWORDS = ["tariff", "inflation", "fed", "interest rate", "earnings", "GDP"]
EVENTS_CSV_PATH = "data/events.csv"

def fetch_events():
    if not API_KEY:
        raise ValueError("API key not found. Make sure NEWSAPI_KEY is set in your environment.")

    events = []
    for keyword in KEYWORDS:
        params = {
            "q": keyword,
            "language": "en",
            "apiKey": API_KEY,
            "pageSize": 5,
        }
        res = requests.get(ENDPOINT, params=params).json()
        for article in res.get("articles", []):
            date = article["publishedAt"][:10]
            title = article["title"]
            events.append({
                "date": date,
                "event": title,
                "event_type": keyword.title(),
                "country": "Global"
            })
    return pd.DataFrame(events)

def update_events_csv(path=EVENTS_CSV_PATH):
    new_events = fetch_events()
    try:
        existing = pd.read_csv(path)
    except FileNotFoundError:
        existing = pd.DataFrame()

    combined = pd.concat([existing, new_events]).drop_duplicates()
    combined.to_csv(path, index=False)
    print(f"[INFO] Events CSV updated with {len(new_events)} new rows.")

if __name__ == "__main__":
    update_events_csv()