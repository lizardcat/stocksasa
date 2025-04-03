# tests/test_fetch_prices.py

import os
import pytest
import pandas as pd
from data.fetch_prices import fetch_data, fetch_all_from_json

def test_fetch_data_returns_dataframe():
    df = fetch_data("AAPL", period="1mo", save=False)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Close" in df.columns

def test_fetch_data_saves_to_csv(tmp_path):
    # Save manually using temp directory
    test_path = tmp_path / "AAPL.csv"
    df = fetch_data("AAPL", period="1mo", save=False)
    df.to_csv(test_path)
    assert os.path.exists(test_path)

def test_fetch_data_invalid_ticker_returns_empty():
    df = fetch_data("FAKE123", period="1mo", save=False)
    assert df.empty

def test_fetch_all_from_json(tmp_path):
    # Create mock symbols.json file
    json_path = tmp_path / "symbols.json"
    symbols = {
        "stocks": ["AAPL"],
        "indices": ["^GSPC"]
    }

    import json
    with open(json_path, "w") as f:
        json.dump(symbols, f)

    save_dir = tmp_path / "saved"
    fetch_all_from_json(str(json_path), save_dir=str(save_dir))

    # Check saved files
    assert (save_dir / "AAPL.csv").exists()
    assert (save_dir / "GSPC.csv").exists()  # ^GSPC saved as GSPC.csv
