# data/add_event_flags.py

import pandas as pd

def add_event_flags(df: pd.DataFrame, events_path: str = "data/events.csv") -> pd.DataFrame:
    """
    Adds one-hot encoded event type flags for each event that occurred on the same date.
    For example: is_tariff_day, is_earnings_day, etc.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    events = pd.read_csv(events_path)
    events["date"] = pd.to_datetime(events["date"])

    # Get unique event types from the CSV
    event_types = events["event_type"].unique()

    # Initialize flag columns in the main DataFrame
    for event_type in event_types:
        col_name = f"is_{event_type.lower().replace(' ', '_')}_day"
        df[col_name] = 0

    # Apply flags row by row
    for _, row in events.iterrows():
        match_date = row["date"]
        event_type = row["event_type"]
        flag_col = f"is_{event_type.lower().replace(' ', '_')}_day"

        df.loc[df["Date"] == match_date, flag_col] = 1

    return df
