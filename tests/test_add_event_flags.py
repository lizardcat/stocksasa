# tests/test_add_event_flags.py

import pandas as pd
import pytest
from data.add_event_flags import add_event_flags

@pytest.fixture
def stock_data():
    return pd.DataFrame({
        "Date": pd.date_range(start="2025-04-01", periods=10, freq='D'),
        "Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })

@pytest.fixture
def temp_events_csv(tmp_path):
    content = """date,event,event_type,country
2025-04-05,Universal 10% U.S. tariff on all imports,Tariff,USA
2025-04-09,Reciprocal tariffs on China,Tariff,USA
"""
    path = tmp_path / "events.csv"
    path.write_text(content)
    return str(path)

def test_add_event_flags_applies_flags(stock_data, temp_events_csv):
    df = add_event_flags(stock_data, events_path=temp_events_csv)
    assert df.loc[df["Date"] == pd.Timestamp("2025-04-05"), "is_tariff_day"].iloc[0] == 1
    assert df.loc[df["Date"] == pd.Timestamp("2025-04-09"), "is_tariff_day"].iloc[0] == 1
    assert df["is_tariff_day"].sum() == 2
