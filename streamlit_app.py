import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf

# Your existing functions here (data preparation, model creation, backtesting, etc.)

def main():
    st.title('Stock Price Prediction App')

    # Add a stock ticker input
    ticker = st.text_input('Enter a stock ticker (e.g., AAPL, GOOGL):', 'AAPL')

    # Add a date range selector
    start_date = st.date_input('Start date', pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End date', pd.to_datetime('today'))

    if st.button('Analyze'):
        # Fetch data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Your data preparation code here

def prepare_stock_data(df, ticker):
    # Select data for the given stock
    stock_data = df[df['company_name'] == ticker].copy()
    
    if stock_data.empty:
        print(f"No data found for ticker {ticker}")
        return None, None
    
    # Sort by date
    stock_data = stock_data.sort_values('Date')
    
    # Select 'Date' and 'Close' columns
    stock_data = stock_data[['Date', 'Close']]
    
    # Set 'Date' as index
    stock_data.set_index('Date', inplace=True)
    
    # Handle missing values (if any)
    stock_data = stock_data.dropna()
    
    if len(stock_data) < 5:
        print(f"Not enough data points for ticker {ticker}. Need at least 5, but got {len(stock_data)}.")
        return None, None
    
    # Create a 5-day moving average
    stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
    
    # Drop rows with NaN values created by the moving average
    stock_data = stock_data.dropna()
    
    if stock_data.empty:
        print(f"No data left after preprocessing for ticker {ticker}")
        return None, None
    
    # Normalize the data
    scaler = MinMaxScaler()
    stock_data_scaled = pd.DataFrame(scaler.fit_transform(stock_data), 
                                     columns=stock_data.columns, 
                                     index=stock_data.index)
    
    return stock_data_scaled, scaler

# Print unique company names in the DataFrame
print("Unique company names in the DataFrame:")
print(df['company_name'].unique())

# Print the number of rows for each company
print("\nNumber of rows for each company:")
print(df['company_name'].value_counts())

# Example usage:
for ticker in df['company_name'].unique():
    print(f"\nProcessing {ticker}")
    prepared_data, scaler = prepare_stock_data(df, ticker)
    
    if prepared_data is not None:
        print(prepared_data.head())
        print("\nShape of prepared data:", prepared_data.shape)
    else:
        print("Could not prepare data for this ticker.")
        # Your model training code here


        def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Predict only the closing price
    return np.array(X), np.array(y)

def create_and_train_model(data, ticker):
    # Separate features and target
    features = data.drop('Close', axis=1)
    target = data['Close']

    # Normalize the features
    scaler_features = MinMaxScaler()
    scaled_features = scaler_features.fit_transform(features)

    # Normalize the target
    scaler_target = MinMaxScaler()
    scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

    # Combine scaled features and target
    scaled_data = np.hstack((scaled_target, scaled_features))

    # Create sequences
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define the model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler_target.inverse_transform(train_predict)
    test_predict = scaler_target.inverse_transform(test_predict)
    y_train_actual = scaler_target.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))

    return model, history, train_predict, test_predict, y_train_actual, y_test_actual

# Train model for each stock
for ticker, data in engineered_data.items():
    print(f"\nTraining model for {ticker}")
    model, history, train_predict, test_predict, y_train_actual, y_test_actual = create_and_train_model(data, ticker)
    
    # Print some results
    print(f"Train MSE: {np.mean((y_train_actual - train_predict)**2):.4f}")
    print(f"Test MSE: {np.mean((y_test_actual - test_predict)**2):.4f}")

        # Your backtesting code here

    def implement_strategy(data, predictions):
    """Implement a simple trading strategy based on predictions."""
    # Ensure predictions and data are aligned
    predictions = pd.Series(predictions.flatten(), index=data.index[-len(predictions):])
    
    buy_signals = predictions > data['Close'].shift(1)
    sell_signals = predictions < data['Close'].shift(1)
    return pd.Series(np.where(buy_signals, 1, np.where(sell_signals, -1, 0)), index=predictions.index)

def backtest_strategy(data, signals, initial_capital=10000):
    """Backtest the trading strategy."""
    position = signals.cumsum()
    holdings = position * data['Close']
    cash = initial_capital - (signals.diff().fillna(0) * data['Close']).cumsum()
    total_holdings = holdings + cash
    returns = total_holdings.pct_change()
    return total_holdings, returns

def calculate_metrics(returns, risk_free_rate=0.02):
    """Calculate performance metrics."""
    total_return = (returns + 1).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio
    }

# Implement strategy and backtest for each stock
for ticker, data in engineered_data.items():
    print(f"\nBacktesting strategy for {ticker}")
    
    # Train the model and get predictions
    model, _, _, test_predict, _, y_test_actual = create_and_train_model(data, ticker)
    
    # Implement strategy
    test_data = data.iloc[-len(test_predict):]
    signals = implement_strategy(test_data, test_predict)
    
    # Backtest strategy
    total_holdings, returns = backtest_strategy(test_data, signals)
    
    # Calculate metrics
    metrics = calculate_metrics(returns)
    
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 7))
    plt.plot(total_holdings, label='Strategy')
    plt.plot((test_data['Close'] / test_data['Close'].iloc[0]) * 10000, label='Buy and Hold')
    plt.title(f'{ticker} - Strategy vs Buy and Hold', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Portfolio Value ($)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

        # Display results
        st.subheader('Stock Price Prediction Results')
        st.line_chart(data['Close'])

        st.subheader('Model Performance Metrics')
        # Display your metrics here

        st.subheader('Trading Strategy Performance')
        # Display your trading strategy results here

if __name__ == '__main__':
    main()
