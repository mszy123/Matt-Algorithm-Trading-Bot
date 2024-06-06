import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv('API_KEY')
secret_key = os.getenv('API_SECRET')

if api_key is None or secret_key is None:
    raise ValueError("API key not found. Please set the API_KEY environment variable.")
else:
    print("API key loaded successfully.")

API_KEY = api_key
API_SECRET = secret_key
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading URL for testing

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# List of stock symbols to backtest
symbols = ['MSFT']

timeframe = '15Min'  # Correct timeframe for daily data
start = '2023-01-01T00:00:00Z'  # ISO 8601 format with timezone
end = '2023-12-31T00:00:00Z'    # ISO 8601 format with timezone

def compute_rsi(data, window):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest_strategy(symbol, short_window, long_window, rsi_window, rsi_buy_threshold, rsi_sell_threshold, initial_capital):
    # Fetch historical market data for the symbol
    bars = api.get_bars(symbol, timeframe, start=start, end=end).df
    
    # Calculate moving averages
    bars['short_mavg'] = bars['close'].rolling(window=short_window).mean()
    bars['long_mavg'] = bars['close'].rolling(window=long_window).mean()
    bars['RSI'] = compute_rsi(bars, rsi_window)

    # Ensure there's no NaN in moving averages and RSI
    bars = bars.dropna()

    # Generate signals
    bars['signal'] = 0.0
    bars['signal'] = np.where(
        (bars['short_mavg'] > bars['long_mavg']) & (bars['RSI'] < rsi_buy_threshold), 
        1.0, 
        np.where(
            (bars['short_mavg'] < bars['long_mavg']) & (bars['RSI'] > rsi_sell_threshold), 
            -1.0, 
            0.0
        )
    )

    # Generate trading orders
    bars['positions'] = bars['signal'].diff()

    # Initialize portfolio
    bars['cash'] = initial_capital
    bars['holdings'] = 0.0  # Change to float for fractional shares
    bars['total'] = initial_capital
    shares_bought = []
    shares_sold = []

    for i in range(1, len(bars)):
        if bars['positions'].iloc[i] == 1.0:  # Buy signal
            shares_to_buy = bars['cash'].iloc[i - 1] / bars['close'].iloc[i]
            cost = shares_to_buy * bars['close'].iloc[i]
            if cost <= bars['cash'].iloc[i - 1]:
                bars.loc[bars.index[i], 'cash'] = bars['cash'].iloc[i - 1] - cost
                bars.loc[bars.index[i], 'holdings'] = bars['holdings'].iloc[i - 1] + shares_to_buy
                shares_bought.append((bars.index[i], shares_to_buy))
        elif bars['positions'].iloc[i] == -1.0:  # Sell signal
            shares_to_sell = bars['holdings'].iloc[i - 1]
            revenue = shares_to_sell * bars['close'].iloc[i]
            bars.loc[bars.index[i], 'cash'] = bars['cash'].iloc[i - 1] + revenue
            bars.loc[bars.index[i], 'holdings'] = 0.0
            shares_sold.append((bars.index[i], shares_to_sell))
        else:  # Hold
            bars.loc[bars.index[i], 'cash'] = bars['cash'].iloc[i - 1]
            bars.loc[bars.index[i], 'holdings'] = bars['holdings'].iloc[i - 1]
        
        bars.loc[bars.index[i], 'total'] = bars['cash'].iloc[i] + (bars['holdings'].iloc[i] * bars['close'].iloc[i])

    # Calculate total profit
    final_portfolio_value = bars['total'].iloc[-1]
    total_profit = final_portfolio_value - initial_capital
    total_return = (final_portfolio_value / initial_capital) - 1

    return symbol, total_profit, total_return, shares_bought, shares_sold, bars

# Define parameters for grid search
short_windows = [10, 20, 30]
long_windows = [40, 50, 60]
rsi_window = 14
rsi_buy_threshold = 30
rsi_sell_threshold = 70
initial_capital = 100.0  # Example with $100 initial capital

# Dictionary to store results for each stock
results = {}

for short_window in short_windows:
    for long_window in long_windows:
        for symbol in symbols:
            symbol, profit, total_return, shares_bought, shares_sold, bars = backtest_strategy(symbol, short_window, long_window, rsi_window, rsi_buy_threshold, rsi_sell_threshold, initial_capital)
            results[(symbol, short_window, long_window)] = {
                'profit': profit,
                'total_return': total_return,
                'shares_bought': shares_bought,
                'shares_sold': shares_sold,
                'bars': bars
            }

# Print and plot results for the best-performing parameters
best_result = max(results.items(), key=lambda x: x[1]['total_return'])
symbol, short_window, long_window = best_result[0]
profit, total_return, shares_bought, shares_sold, bars = best_result[1].values()

print(f"Best results for {symbol} with short_window={short_window}, long_window={long_window}:")
print(f"Total profit: {profit}")
print(f"Total return: {total_return}\n")

# Plot the portfolio value over time with buy and sell signals for each stock
buy_signals = bars[bars['positions'] == 1.0]
sell_signals = bars[bars['positions'] == -1.0]

plt.figure(figsize=(12, 6))
plt.plot(bars['total'], label=f'{symbol} Portfolio Total Value')
plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='Buy Signal', alpha=1)
plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='Sell Signal', alpha=1)
plt.title(f'{symbol} Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.show()

# Optionally, combine results to get a total portfolio view across all stocks
total_portfolio_value = sum([result['bars']['total'].iloc[-1] for result in results.values()])
total_initial_capital = initial_capital * len(symbols)
total_profit = total_portfolio_value - total_initial_capital
total_return = (total_portfolio_value / total_initial_capital) - 1

print(f"Combined portfolio value: {total_portfolio_value}")
print(f"Combined total profit: {total_profit}")
print(f"Combined total return: {total_return}")
