# App.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from model import fetch_stock_info, fetch_stock_history, generate_stock_prediction

st.set_page_config(page_title="Stock Prediction App", page_icon="📈")

##### Sidebar Start #####

st.sidebar.markdown("## **User Input Features**")

stock_ticker = "^GSPC"  # S&P 500 Ticker

st.sidebar.markdown("### **Stock ticker**")
st.sidebar.text_input(label="Stock ticker code", placeholder=stock_ticker, disabled=True)

periods = {
    "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
    "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
    "1mo": ["30m", "60m", "90m", "1d"],
    "3mo": ["1d", "5d", "1wk", "1mo"],
    "6mo": ["1d", "5d", "1wk", "1mo"],
    "1y": ["1d", "5d", "1wk", "1mo"],
    "2y": ["1d", "5d", "1wk", "1mo"],
    "5y": ["1d", "5d", "1wk", "1mo"],
    "10y": ["1d", "5d", "1wk", "1mo"],
    "max": ["1d", "5d", "1wk", "1mo"],
}

st.sidebar.markdown("### **Select period**")
period = st.sidebar.selectbox("Choose a period", list(periods.keys()))

st.sidebar.markdown("### **Select interval**")
interval = st.sidebar.selectbox("Choose an interval", periods[period])

##### Sidebar End #####

##### Title #####

st.markdown("# **Stock Price Prediction**")
st.markdown("##### **Enhance Investment Decisions through Data-Driven Forecasting**")

##### Title End #####

# Fetch the stock historical data
stock_data = fetch_stock_history(stock_ticker, period, interval)

##### Historical Data Graph #####

st.markdown("## **Historical Data**")

fig = go.Figure(data=[go.Candlestick(
    x=stock_data.index,
    open=stock_data['Open'],
    high=stock_data['High'],
    low=stock_data['Low'],
    close=stock_data['Close'],
)])

fig.update_xaxes(type="category")
fig.update_layout(height=600)

st.plotly_chart(fig, use_container_width=True)

##### Historical Data Graph End #####

##### Stock Information #####

st.markdown("## **Stock Information**")

stock_info = fetch_stock_info(stock_ticker)
for key, value in stock_info.items():
    st.markdown(f"### **{key}**")
    st.table
