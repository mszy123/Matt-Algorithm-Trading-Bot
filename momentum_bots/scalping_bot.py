import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
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

# Define trading parameters
PROFIT_TARGET = 0.1  # Profit target in dollars
STOP_LOSS = 3.0  # Stop loss in dollars
SYMBOL = 'INTC'  # Example symbol
TIMEFRAME = '1Min'  # Timeframe for scalping
TRANSACTION_COST = 0.02  # Increased transaction cost per share
SLIPPAGE = 0.01  # Example slippage per trade
TRADE_SIZE_PERCENT = 0.75  # Trade size as a percentage of buying power

# Function to fetch historical data
def get_historical_data(symbol, start_date, end_date, timeframe='1Min'):
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    bars = api.get_bars(symbol, timeframe, start=start_date_str, end=end_date_str).df
    bars = bars.tz_convert('America/New_York')
    return bars

# Function to fetch the latest market data
def get_latest_data(symbol, timeframe='1Min'):
    now = datetime.now()
    start_date = now - timedelta(minutes=120)  # Get data from the last 120 minutes to ensure enough data points
    return get_historical_data(symbol, start_date, now, timeframe)

# Identify trade signals
def identify_signals(data):
    if 'close' not in data.columns:
        print("Error: 'close' column not found in data")
        print(data.head())  # Print the first few rows of the data for debugging
        return data
    
    data['ma_short'] = data['close'].rolling(window=5).mean()
    data['ma_long'] = data['close'].rolling(window=20).mean()
    
    data['signal'] = 0
    data.loc[data['ma_short'] > data['ma_long'], 'signal'] = 1
    data.loc[data['ma_short'] < data['ma_long'], 'signal'] = -1
    
    return data

# Function to execute trades based on signals
def execute_trades():
    data = get_latest_data(SYMBOL, TIMEFRAME)
    
    if data.empty:
        print("No data available.")
        return
    
    data = identify_signals(data)
    
    if data.empty or 'close' not in data.columns:
        print("Data is empty or 'close' column is missing after identifying signals.")
        return
    
    signal = data['signal'].iloc[-1]
    current_price = data['close'].iloc[-1]
    
    print(f"Short moving average: {data['ma_short'].iloc[-1]}")
    print(f"Long moving average: {data['ma_long'].iloc[-1]}")
    print(f"Signal: {signal}")

    account = api.get_account()
    buying_power = float(account.buying_power)
    positions = api.list_positions()

    # Calculate the trade size based on a percentage of available buying power
    trade_size = buying_power * TRADE_SIZE_PERCENT
    print(f"Buying power: ${buying_power:.2f}, Trade size: ${trade_size:.2f}")

    # Check if we have an open position in the symbol
    current_position = next((pos for pos in positions if pos.symbol == SYMBOL), None)
    if current_position:
        qty = int(float(current_position.qty))  # Ensure qty is an integer
        avg_entry_price = float(current_position.avg_entry_price)
    else:
        qty = 0
        avg_entry_price = 0

    print(f"Current position: {qty} shares at average entry price {avg_entry_price}")

    # Sell conditions: either profit target or stop loss
    if qty > 0:
        profit_loss = (current_price - avg_entry_price) * qty
        if profit_loss >= PROFIT_TARGET * qty:
            try:
                api.submit_order(
                    symbol=SYMBOL,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Profit target hit: Selling {qty} shares of {SYMBOL} at {current_price}")
                send_discord_notification(f"Profit target hit: Selling {qty} shares of {SYMBOL} at {current_price}")
            except Exception as e:
                print(f"Error executing profit target sell order: {e}")
        elif profit_loss <= -STOP_LOSS:
            try:
                api.submit_order(
                    symbol=SYMBOL,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Stop loss hit: Selling {qty} shares of {SYMBOL} at {current_price}")
                send_discord_notification(f"Stop loss hit: Selling {qty} shares of {SYMBOL} at {current_price}")
            except Exception as e:
                print(f"Error executing stop loss sell order: {e}")

    if signal == 1 and qty == 0:  # Buy signal
        shares_to_buy = int(trade_size // current_price)
        if shares_to_buy <= 0:
            print(f"Insufficient buying power to buy shares at {current_price}")
        else:
            try:
                api.submit_order(
                    symbol=SYMBOL,
                    qty=shares_to_buy,
                    side='buy',
                    type='market',
                    time_in_force='gtc'  # Good 'til canceled to avoid day trade classification
                )
                print(f"Buy signal: Buying {shares_to_buy} shares of {SYMBOL} at {current_price}")
                send_discord_notification(f"Buy signal: Buying {shares_to_buy} shares of {SYMBOL} at {current_price}")
            except Exception as e:
                print(f"Error executing buy order: {e}")

    elif signal == -1 and qty > 0:  # Sell signal
        try:
            api.submit_order(
                symbol=SYMBOL,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'  # Good 'til canceled to avoid day trade classification
            )
            print(f"Sell signal: Selling {qty} shares of {SYMBOL} at {current_price}")
            send_discord_notification(f"Sell signal: Selling {qty} shares of {SYMBOL} at {current_price}")
        except Exception as e:
            print(f"Error executing sell order: {e}")

    else:
        print("No buy or sell signal generated.")
        print(f"Current price: {current_price}")
        print(f"Short moving average: {data['ma_short'].iloc[-1]}")
        print(f"Long moving average: {data['ma_long'].iloc[-1]}")
        print(f"Signal: {signal}")


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

# Main function to continuously execute trades
def main():
    while True:
        execute_trades()
        print("Waiting for the next iteration...")
        for _ in tqdm(range(60), desc="Time remaining"):
            time.sleep(1)

# Run the main function
if __name__ == "__main__":
    main()
