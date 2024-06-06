import alpaca_trade_api as tradeapi

API_KEY = 'PKLQJE3ND8EH0JUSQT2B'
API_SECRET = 'qcratCxIsXu09nxVUgRdwRSGXcObEgxD0ndjdSBy'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading URL for testing

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')


# Fetch historical market data for AAPL
symbol = 'AAPL'
timeframe = '1Day'  # Correct timeframe for daily data
start = '2023-01-01T00:00:00Z'  # ISO 8601 format with timezone
end = '2023-12-31T00:00:00Z'    # ISO 8601 format with timezone

bars = api.get_bars(symbol, timeframe, start=start, end=end).df

# Print the DataFrame
print(bars)