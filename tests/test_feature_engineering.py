import pytest
import pandas as pd
from features.feature_engineering import generate_features

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Close": [100, 102, 101, 105, 107, 106],
        "Volume": [1000, 1100, 1050, 1200, 1300, 1250]
    })

def test_generate_features_adds_columns(sample_data):
    df = generate_features(sample_data)
    expected = {"Daily_Return", "MA_5", "STD_5", "Lag_1"}
    assert expected.issubset(set(df.columns))

def test_feature_values_are_not_nan(sample_data):
    df = generate_features(sample_data)
    df = df.dropna()
    assert not df[["Daily_Return", "MA_5", "STD_5", "Lag_1"]].isnull().any().any()
