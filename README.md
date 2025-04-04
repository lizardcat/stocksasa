# Stocksasa - A Simple AI Market Trend Prediction App

**Stocksasa** is a stock trend prediction system powered by AI. It uses historical market data, technical indicators, and macroeconomic event analysis (like tariffs and interest rate changes) to predict whether a stock or index is likely to go up or down the next day.

Built with:
- Python 
- Streamlit for interactive UI
- Random Forest Classifier for AI predictions
- yFinance for market data
- NewsAPI for real-time economic events (optional)

## Features

- Predicts up/down movement for any stock or index
- Visualizes recent trends and AI predictions
- Learns from technical indicators + global event flags
- Export historical data as CSV
- Automatically update macroeconomic events with `update_events.py`

## Installation

Clone the repo and install requirements:

```bash
git clone https://github.com/lizardcat/stocksasa.git

cd stocksasa

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## API Key Setup (for NewsAPI)

To use live news updates, you’ll need a free API key from [NewsAPI.org](https://newsapi.org/).

- Create a file named .env in the root of your project.

- Inside that file, add your own NewsAPI key like this:

```
NEWSAPI_KEY=your_actual_api_key_here
```

Replace `your_newsapi_key_here` with your personal API key from https://newsapi.org.
This file is ignored by Git and will stay private.

## How to Use

### 1. Train the model:
```bash
python main.py
```

### 2. Run the Streamlit app:
```bash
streamlit run app.py
```

### 3. Update events from live news:
```bash
python update_events.py
```