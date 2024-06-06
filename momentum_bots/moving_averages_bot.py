import alpaca_trade_api as tradeapi
import numpy as np  # Import numpy
import pandas as pd
import os
from dotenv import load_dotenv

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




# Fetch historical market data for AAPL
symbol = 'AAPL'
timeframe = '1Day'  # Correct timeframe for daily data
start = '2023-01-01T00:00:00Z'  # ISO 8601 format with timezone
end = '2023-12-31T00:00:00Z'    # ISO 8601 format with timezone

bars = api.get_bars(symbol, timeframe, start=start, end=end).df

# Calculate moving averages
bars['short_mavg'] = bars['close'].rolling(window=40).mean()
bars['long_mavg'] = bars['close'].rolling(window=100).mean()

# Generate signals
bars['signal'] = 0.0
bars.loc[bars.index[40:], 'signal'] = np.where(
    bars.loc[bars.index[40:], 'short_mavg'] > bars.loc[bars.index[40:], 'long_mavg'], 
    1.0, 
    0.0
)

# Generate trading orders
bars['positions'] = bars['signal'].diff()

# Function to execute trades
def execute_trades(data):
    for i in range(len(data)):
        if data['positions'].iloc[i] == 1.0:
            print(f"Buy signal on {data.index[i]}: Buy {symbol} at {data['close'].iloc[i]}")
            # Uncomment the following lines to place real orders
            # api.submit_order(
            #     symbol=symbol,
            #     qty=1,
            #     side='buy',
            #     type='market',
            #     time_in_force='gtc'
            # )
        elif data['positions'].iloc[i] == -1.0:
            print(f"Sell signal on {data.index[i]}: Sell {symbol} at {data['close'].iloc[i]}")
            # Uncomment the following lines to place real orders
            # api.submit_order(
            #     symbol=symbol,
            #     qty=1,
            #     side='sell',
            #     type='market',
            #     time_in_force='gtc'
            # )

# Execute trades based on the generated signals
execute_trades(bars)

# Print the signals and positions
print(bars[['close', 'short_mavg', 'long_mavg', 'signal', 'positions']])