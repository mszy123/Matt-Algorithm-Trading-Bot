import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from itertools import product
import time
from tqdm import tqdm
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
def backtest_strategy(data, symbol, initial_capital=500.0):
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
def grid_search(data, symbol, param_grid, initial_capital=500.0):
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

# Execute a buy order
def execute_buy(symbol, cash):
    price = api.get_latest_trade(symbol).price
    shares_to_buy = cash / price
    api.submit_order(
        symbol=symbol,
        qty=shares_to_buy,
        side='buy',
        type='market',
        time_in_force='day'  # Changed from 'gtc' to 'day'
    )
    print(f"Executed BUY order for {shares_to_buy} shares of {symbol} at ${price}")

# Execute a sell order
def execute_sell(symbol, shares):
    price = api.get_latest_trade(symbol).price
    api.submit_order(
        symbol=symbol,
        qty=shares,
        side='sell',
        type='market',
        time_in_force='day'  # Changed from 'gtc' to 'day'
    )
    print(f"Executed SELL order for {shares} shares of {symbol} at ${price}")

# Monitor the market and execute trades
def monitor_and_trade(symbol, short_window, long_window, rsi_period, rsi_overbought, rsi_oversold):
    # Get current cash and holdings
    account = api.get_account()
    cash = float(account.cash)
    positions = api.list_positions()
    holdings = 0.0
    for position in positions:
        if position.symbol == symbol:
            holdings = float(position.qty)
            break

    print(f"Starting with cash: ${cash} and holdings: {holdings} shares of {symbol}")

    while True:
        end_date = datetime.now()
        start_date = end_date - timedelta(minutes=30)
        data = get_historical_data(symbol, start_date, end_date, '1Min')
        strategy_data = moving_average_strategy(data, short_window, long_window, rsi_period, rsi_overbought, rsi_oversold)
        
        latest_signal = strategy_data['positions'].iloc[-1]
        
        if latest_signal == 1.0:
            if cash > 0:
                execute_buy(symbol, cash)
                holdings = cash / api.get_latest_trade(symbol).price
                cash = 0.0
                print(f"Decision: BUY {holdings} shares of {symbol} at ${api.get_latest_trade(symbol).price}")
                send_discord_notification(f"Decision: BUY {holdings} shares of {symbol} at ${api.get_latest_trade(symbol).price}")
        elif latest_signal == -1.0:
            if holdings > 0:
                execute_sell(symbol, holdings)
                cash = holdings * api.get_latest_trade(symbol).price
                holdings = 0.0
                print(f"Decision: SELL {holdings} shares of {symbol} at ${api.get_latest_trade(symbol).price}")
                send_discord_notification(f"Decision: SELL {holdings} shares of {symbol} at ${api.get_latest_trade(symbol).price}")
        else:
            print("No decision")

        # Progress bar for 60 seconds
        for _ in tqdm(range(60), desc="Waiting for next check", leave=False):
            time.sleep(1)


def send_discord_notification(message):
    data = {
        "content": message,
        "username": "Notification Bot"
    }
    response = requests.post("https://discord.com/api/webhooks/1243289392496509110/gWOLYRQDzrk-r4hT_B6FrxjdSxVXPnSSHcKLxDTE8RuU2PdzjvsoHy49GlwPoD4a3zKy", json=data)
    if response.status_code == 204:
        print("Notification sent successfully!")
    else:
        print(f"Failed to send notification. Status code: {response.status_code}")




# Main function to run the backtest, grid search, and real-time trading
def main():
    symbol = 'INTC'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

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

    # Apply the best strategy for real-time trading
    monitor_and_trade(symbol, *best_params)

# Run the main function
if __name__ == "__main__":
    main()
