import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from itertools import product
import requests


# Load environment variables from .env file
load_dotenv()

api_key = os.getenv('API_KEY')
secret_key = os.getenv('API_SECRET')

if api_key is None or secret_key is None:
    raise ValueError("API key not found. Please set the API_KEY environment variable.")
else:
    print("API key loaded successfully.")

API_KEY = api_key
API_SECRET = secret_key
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Fetch historical data
def get_historical_data(symbol, start_date, end_date, timeframe='1Min'):
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    bars = api.get_bars(symbol, timeframe, start=start_date_str, end=end_date_str, adjustment='raw', feed='iex').df
    bars = bars.tz_convert('America/New_York')  # Convert to your preferred timezone if necessary
    return bars

# Develop a trading strategy (Moving Average Crossover with RSI filter)
def moving_average_strategy(data, short_window=15, long_window=50, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
    data['short_mavg'] = data['close'].rolling(window=short_window, min_periods=1).mean()
    data['long_mavg'] = data['close'].rolling(window=long_window, min_periods=1).mean()
    
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    data['signal'] = 0
    data.iloc[short_window:, data.columns.get_loc('signal')] = np.where(
        (data['short_mavg'].iloc[short_window:] > data['long_mavg'].iloc[short_window:]) & (data['rsi'].iloc[short_window:] < rsi_oversold), 1, 0
    )
    data.iloc[short_window:, data.columns.get_loc('signal')] = np.where(
        (data['short_mavg'].iloc[short_window:] < data['long_mavg'].iloc[short_window:]) & (data['rsi'].iloc[short_window:] > rsi_overbought), -1, data['signal'].iloc[short_window:]
    )
    data['positions'] = data['signal'].diff()
    return data

# Backtest the strategy
def backtest_strategy(data, symbol, initial_capital=100.0):
    positions = pd.DataFrame(index=data.index).fillna(0.0)
    cash = initial_capital
    portfolio_value = []
    holdings = 0.0

    for i in range(1, len(data)):
        if data['positions'].iloc[i] == 1.0:
            shares_to_buy = cash / data['close'].iloc[i]
            holdings += shares_to_buy
            cash -= shares_to_buy * data['close'].iloc[i]
        elif data['positions'].iloc[i] == -1.0:
            cash += holdings * data['close'].iloc[i]
            holdings = 0.0
        portfolio_value.append(cash + holdings * data['close'].iloc[i])

    portfolio = pd.DataFrame(index=data.index[1:], data={'total': portfolio_value})
    portfolio['returns'] = portfolio['total'].pct_change()
    
    total_profit = portfolio['total'].iloc[-1] - initial_capital
    return portfolio, total_profit

# Grid search to find the best strategy parameters
def grid_search(data, symbol, param_grid, initial_capital=100.0):
    best_profit = -np.inf
    best_params = None

    for params in product(*param_grid.values()):
        short_window, long_window, rsi_period, rsi_overbought, rsi_oversold = params
        strategy_data = moving_average_strategy(data.copy(), short_window, long_window, rsi_period, rsi_overbought, rsi_oversold)
        _, total_profit = backtest_strategy(strategy_data, symbol, initial_capital)
        if total_profit > best_profit:
            best_profit = total_profit
            best_params = params

    return best_params, best_profit

# Main function to run the backtest and grid search
def main():
    symbol = 'GOOG'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)

    # Fetch historical data
    data = get_historical_data(symbol, start_date, end_date, '1Min')

    # Define parameter grid
    param_grid = {
        'short_window': [10, 15, 20],
        'long_window': [50, 60, 70],
        'rsi_period': [10, 14, 20],
        'rsi_overbought': [70, 75, 80],
        'rsi_oversold': [20, 30, 40]
    }

    # Perform grid search
    best_params, best_profit = grid_search(data, symbol, param_grid)
    print(f"Best Parameters: {best_params}")
    print(f"Best Profit: ${best_profit:.2f}")

    # Apply the best strategy
    strategy_data = moving_average_strategy(data, *best_params)

    # Backtest the strategy with the best parameters
    portfolio, total_profit = backtest_strategy(strategy_data, symbol)
    print(f"Total Profit with Best Parameters: ${total_profit:.2f}")
    print(portfolio.tail())

    # Plot portfolio performance
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio['total'], label='Total Portfolio Value')

    # Plot buy and sell signals
    buy_signals = strategy_data.loc[strategy_data['positions'] == 1.0].index
    sell_signals = strategy_data.loc[strategy_data['positions'] == -1.0].index
    plt.plot(buy_signals, portfolio.loc[buy_signals, 'total'], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(sell_signals, portfolio.loc[sell_signals, 'total'], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

    plt.title('Portfolio Performance Over the Past Week')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
