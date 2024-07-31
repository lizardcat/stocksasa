import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Title and introductory text
st.title('Stock Market Prediction Application')
st.write('This application predicts stock market trends using historical data and machine learning.')

# User input for stock selection
ticker = st.text_input('Enter the stock ticker symbol, e.g., AAPL, GOOGL, MSFT:', 'AAPL')
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Displaying the data as a dataframe
st.write('Displaying stock data:')
st.dataframe(data.tail())
