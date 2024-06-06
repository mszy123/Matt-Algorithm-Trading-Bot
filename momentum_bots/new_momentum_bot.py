import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv('API_KEY')
secret_key = os.getenv('API_SECRET')

if api_key is None or secret_key is None:
    raise ValueError("API key not found. Please set the API_KEY and API_SECRET environment variables.")
else:
    logger.info("API key loaded successfully.")

API_KEY = api_key
API_SECRET = secret_key
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading URL for testing

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Define the symbol and parameters
symbol = 'MSFT'
timeframe = '1Min'  # 1 minute timeframe for trading

# Fetch historical market data for AAPL
def fetch_historical_data():
    end = datetime.now()
    start = end - timedelta(days=1)
    start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ')
    logger.info(f"Fetching historical data for {symbol} from {start_str} to {end_str}")
    bars = api.get_bars(symbol, timeframe, start=start_str, end=end_str).df
    return bars

# Calculate RSI
def calculate_rsi(data, window):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Calculate momentum indicators (e.g., Moving Averages and RSI)
def calculate_indicators(data, sma_20, sma_50, rsi_window):
    data['SMA_20'] = data['close'].rolling(window=sma_20).mean()
    data['SMA_50'] = data['close'].rolling(window=sma_50).mean()
    data = calculate_rsi(data, rsi_window)
    return data

# Define the trading logic
def trading_signal(data):
    if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1] and data['RSI'].iloc[-1] < 30:
        return 'buy'
    elif data['SMA_20'].iloc[-1] < data['SMA_50'].iloc[-1] and data['RSI'].iloc[-1] > 70:
        return 'sell'
    else:
        return 'hold'

# Execute trade
def execute_trade(signal):
    if signal == 'buy':
        logger.info(f"Executing trade: BUY {symbol}")
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
    elif signal == 'sell':
        logger.info(f"Executing trade: SELL {symbol}")
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
    else:
        logger.info(f"No trade executed: HOLD {symbol}")

# Grid search for parameter optimization
def grid_search():
    best_params = None
    best_performance = -np.inf
    for sma_20 in range(10, 31, 5):
        for sma_50 in range(30, 61, 5):
            for rsi_window in range(10, 21, 5):
                data = fetch_historical_data()
                data = calculate_indicators(data, sma_20, sma_50, rsi_window)
                performance = backtest(data)
                if performance > best_performance:
                    best_performance = performance
                    best_params = (sma_20, sma_50, rsi_window)
    return best_params

# Backtest the strategy
def backtest(data):
    initial_balance = 100
    balance = initial_balance
    position = 0
    for i in range(len(data) - 1):
        signal = trading_signal(data.iloc[:i+1])
        if signal == 'buy' and balance > data['close'].iloc[i]:
            balance -= data['close'].iloc[i]
            position += 1
        elif signal == 'sell' and position > 0:
            balance += data['close'].iloc[i]
            position -= 1
    final_balance = balance + position * data['close'].iloc[-1]
    return final_balance - initial_balance

# Main loop to run the bot
if __name__ == "__main__":
    logger.info("Starting grid search for best parameters...")
    sma_20, sma_50, rsi_window = grid_search()
    logger.info(f"Best parameters found: SMA_20={sma_20}, SMA_50={sma_50}, RSI_Window={rsi_window}")

    while True:
        logger.info("Fetching data and calculating indicators...")
        data = fetch_historical_data()
        data = calculate_indicators(data, sma_20, sma_50, rsi_window)
        signal = trading_signal(data)
        logger.info(f"Trading signal: {signal}")
        execute_trade(signal)
        logger.info("Sleeping for 60 seconds...")
        for _ in tqdm(range(60), desc="Waiting"):
            time.sleep(1)
