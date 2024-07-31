import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="Stock Prediction App", page_icon="📈")

@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

def engineer_features(stock_data):
    df = stock_data.copy()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['Middle_BB'] = df['Close'].rolling(window=20).mean()
    df['Upper_BB'] = df['Middle_BB'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_BB'] = df['Middle_BB'] - 2 * df['Close'].rolling(window=20).std()
    
    df['Pct_Change'] = df['Close'].pct_change()
    
    for i in [1, 2, 3, 5]:
        df[f'Lag_{i}'] = df['Close'].shift(i)
    
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    
    return df.dropna()

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def create_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def predict_stock_price(data, model, scaler):
    last_sequence = data[-60:].reshape(1, 60, -1)
    predicted_price = model.predict(last_sequence)
    return scaler.inverse_transform(predicted_price)[0, 0]

def implement_strategy(data, predictions):
    buy_signals = predictions > data['Close'].shift(1)
    sell_signals = predictions < data['Close'].shift(1)
    return pd.Series(np.where(buy_signals, 1, np.where(sell_signals, -1, 0)), index=predictions.index)

def backtest_strategy(data, signals, initial_capital=10000):
    position = signals.cumsum()
    holdings = position * data['Close']
    cash = initial_capital - (signals.diff().fillna(0) * data['Close']).cumsum()
    total_holdings = holdings + cash
    returns = total_holdings.pct_change()
    return total_holdings, returns

def calculate_metrics(returns, risk_free_rate=0.02):
    total_return = (returns + 1).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

st.title('Stock Price Prediction and Trading Strategy')

st.markdown("""
## About Stock Trading

Stock trading involves buying and selling shares of publicly traded companies. Here are some key concepts:

- **Open**: The price at which a stock starts trading when the market opens.
- **Close**: The final price of a stock when the market closes.
- **High**: The highest price a stock reaches during a trading session.
- **Low**: The lowest price a stock reaches during a trading session.
- **Volume**: The number of shares traded during a given time period.
- **Market Cap**: The total value of a company's outstanding shares.

**Disclaimer**: Stock price predictions are based on historical data and machine learning models. 
They should not be considered as financial advice. The stock market is inherently unpredictable 
and involves risk. Always do your own research and consider consulting with a financial advisor 
before making investment decisions.
""")

ticker = st.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('today'))

if st.button('Analyze'):
    data = fetch_stock_data(ticker, start_date, end_date)
    
    st.subheader('Stock Price History')
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    engineered_data = engineer_features(data)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(engineered_data)
    
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    
    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    
    predictions = []
    for i in range(len(engineered_data) - seq_length):
        seq = scaled_data[i:i+seq_length]
        prediction = model.predict(seq.reshape(1, seq_length, -1))
        predictions.append(scaler.inverse_transform(prediction)[0, 0])
    
    predictions = pd.Series(predictions, index=engineered_data.index[seq_length:])
    
    st.subheader('Price Predictions')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name='Predicted'))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    signals = implement_strategy(engineered_data, predictions)
    total_holdings, returns = backtest_strategy(engineered_data, signals)
    
    st.subheader('Trading Strategy Performance')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=total_holdings.index, y=total_holdings, mode='lines', name='Strategy'))
    fig.add_trace(go.Scatter(x=data.index, y=(data['Close'] / data['Close'].iloc[0]) * 10000, mode='lines', name='Buy and Hold'))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    metrics = calculate_metrics(returns)
    st.subheader('Performance Metrics')
    for metric, value in metrics.items():
        st.write(f"{metric}: {value:.4f}")
    
    st.markdown("""
    **Note**: These predictions and strategy results are for educational purposes only. 
    Do not use them as the sole basis for investment decisions. Past performance does not guarantee future results.
    """)
