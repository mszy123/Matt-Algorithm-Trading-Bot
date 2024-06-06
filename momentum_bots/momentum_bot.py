import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

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

try:
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    logger.info("Alpaca API connection established successfully.")
except Exception as e:
    logger.error("Failed to connect to Alpaca API: %s", e)
    raise

# Fetch historical market data for AAPL using IEX data
symbol = 'AAPL'
timeframe = '1D'  # Correct timeframe for daily data
start = '2023-01-01'
end = datetime.today().strftime('%Y-%m-%d')  # End date is today

try:
    bars = api.get_bars(symbol, timeframe, start=start, end=end, feed='iex').df
    logger.info("Historical market data fetched successfully.")
except Exception as e:
    logger.error("Failed to fetch historical market data: %s", e)
    raise

# Calculate moving averages
bars['short_mavg'] = bars['close'].rolling(window=15).mean()
bars['long_mavg'] = bars['close'].rolling(window=50).mean()

# Generate signals
bars['signal'] = 0.0
bars.loc[bars.index[50:], 'signal'] = np.where(
    bars.loc[bars.index[50:], 'short_mavg'] > bars.loc[bars.index[50:], 'long_mavg'], 
    1.0, 
    0.0
)

# Generate trading orders
bars['positions'] = bars['signal'].diff()

# Function to check position
def get_current_position(symbol):
    try:
        position = api.get_position(symbol)
        return int(position.qty)
    except tradeapi.rest.APIError as e:
        if 'position does not exist' in str(e):
            return 0
        else:
            logger.error(f"Failed to retrieve position for {symbol}: {e}")
            raise

# Function to execute trades
def execute_trades(data):
    for i in range(len(data)):
        try:
            current_position = get_current_position(symbol)
        except Exception as e:
            logger.error(f"Error retrieving current position: {e}")
            continue
        
        if data['positions'].iloc[i] == 1.0 and current_position == 0:
            logger.info(f"Buy signal on {data.index[i]}: Buy {symbol} at {data['close'].iloc[i]}")
            try:
                api.submit_order(
                    symbol=symbol,
                    qty=1,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            except Exception as e:
                logger.error(f"Failed to execute buy order: {e}")
        elif data['positions'].iloc[i] == -1.0 and current_position > 0:
            logger.info(f"Sell signal on {data.index[i]}: Sell {symbol} at {data['close'].iloc[i]}")
            try:
                api.submit_order(
                    symbol=symbol,
                    qty=current_position,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
            except Exception as e:
                logger.error(f"Failed to execute sell order: {e}")

# Execute trades based on the generated signals
execute_trades(bars)

# Print the signals and positions
print(bars[['close', 'short_mavg', 'long_mavg', 'signal', 'positions']])
