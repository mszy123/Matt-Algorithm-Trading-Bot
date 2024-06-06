import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

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

# Define trading parameters
PROFIT_TARGET = 0.1  # Example profit target in dollars
STOP_LOSS = 0.1  # Example stop loss in dollars
TRADE_SIZE = 1  # Trade size in dollars
SYMBOL = 'GOOG'  # Example symbol
TIMEFRAME = '1Min'  # Timeframe for scalping
TRANSACTION_COST = 0.02  # Increased transaction cost per share
SLIPPAGE = 0.01  # Example slippage per trade

# Fetch historical data
def get_historical_data(symbol, start_date, end_date, timeframe='1Min'):
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    bars = api.get_bars(symbol, timeframe, start=start_date_str, end=end_date_str, adjustment='raw', feed='iex').df
    bars = bars.tz_convert('America/New_York')
    return bars

# Identify trade signals
def identify_signals(data):
    data['ma_short'] = data['close'].rolling(window=5).mean()
    data['ma_long'] = data['close'].rolling(window=20).mean()
    
    data['signal'] = 0
    data.loc[data['ma_short'] > data['ma_long'], 'signal'] = 1
    data.loc[data['ma_short'] < data['ma_long'], 'signal'] = -1
    
    return data

# Backtest the scalping algorithm
def backtest_scalping(symbol, start_date, end_date, profit_target, stop_loss, trade_size, transaction_cost, slippage, initial_capital=1.0):
    data = get_historical_data(symbol, start_date, end_date, TIMEFRAME)
    data = identify_signals(data)
    
    cash = initial_capital
    holdings = 0.0  # Allow fractional holdings
    portfolio_value = []
    trades = []
    
    open_position = None
    entry_price = None
    
    for i in range(len(data)):
        current_price = data['close'].iloc[i]
        signal = data['signal'].iloc[i]
        
        if open_position is None:
            # Enter new trade
            if signal != 0:
                entry_price = current_price
                open_position = signal
                shares_to_trade = trade_size / current_price  # Calculate fractional shares
                holdings = shares_to_trade if signal == 1 else -shares_to_trade
                # Adjust for transaction cost and slippage
                cash -= holdings * (current_price + slippage) + abs(holdings) * transaction_cost
                trades.append(('BUY' if signal == 1 else 'SELL', current_price, data.index[i]))
        else:
            # Check for exit conditions
            if open_position == 1:
                # Long position
                if current_price >= entry_price + profit_target or current_price <= entry_price - stop_loss:
                    cash += holdings * (current_price - slippage) - abs(holdings) * transaction_cost
                    holdings = 0.0
                    open_position = None
                    trades.append(('SELL', current_price, data.index[i]))
            elif open_position == -1:
                # Short position
                if current_price <= entry_price - profit_target or current_price >= entry_price + stop_loss:
                    cash += -holdings * (current_price + slippage) - abs(holdings) * transaction_cost
                    holdings = 0.0
                    open_position = None
                    trades.append(('BUY', current_price, data.index[i]))
        
        # Debugging: Print the cash and holdings after each iteration
        #print(f"Time: {data.index[i]}, Cash: {cash}, Holdings: {holdings}, Portfolio Value: {cash + holdings * current_price}")

        portfolio_value.append(cash + holdings * current_price)
    
    portfolio = pd.DataFrame(index=data.index, data={'total': portfolio_value})
    portfolio['returns'] = portfolio['total'].pct_change()
    
    total_profit = portfolio['total'].iloc[-1] - initial_capital
    num_trades = len(trades)
    win_rate = len([t for t in trades if t[0] == 'SELL' and t[1] > entry_price]) / num_trades if num_trades > 0 else 0
    
    return portfolio, total_profit, num_trades, win_rate, trades

# Main function to run the backtest
def main():
    symbol = SYMBOL
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    portfolio, total_profit, num_trades, win_rate, trades = backtest_scalping(symbol, start_date, end_date, PROFIT_TARGET, STOP_LOSS, TRADE_SIZE, TRANSACTION_COST, SLIPPAGE)
    
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Number of Trades: {num_trades}")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio.index, portfolio['total'], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.show()
    
    return portfolio, trades

# Run the main function
if __name__ == "__main__":
    portfolio, trades = main()