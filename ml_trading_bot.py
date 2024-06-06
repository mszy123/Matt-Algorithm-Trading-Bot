import yfinance as yf
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time

#load environment variables from .env file
load_dotenv()

api_key = os.getenv('API_KEY')
secret_key = os.getenv('API_SECRET')

if api_key is None or secret_key is None:
    raise ValueError("API key not found. Please set the API_KEY and API_SECRET environment variables.")
else:
    print("API key loaded successfully.")

#load the saved model
model = load_model('GOOG - best_stock_price_model.h5')

#alpaca API credentials
API_KEY = api_key
API_SECRET = secret_key
BASE_URL = 'https://paper-api.alpaca.markets'

#initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

#define the stock to trade
ticker = 'GOOG'

#function to fetch and preprocess data
def fetch_and_preprocess_data(ticker):
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    #fetch historical data
    df = yf.download(ticker, start=start_date, end=end_date)

    #calculate features
    df['Return'] = df['Close'].pct_change()
    df['Moving_Avg_20'] = df['Close'].rolling(window=20).mean()
    df['Moving_Avg_50'] = df['Close'].rolling(window=50).mean()
    df['Volume'] = df['Volume']
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df = df.dropna().copy()  # Ensure we are working with a copy to avoid the warning

    #create features DataFrame
    features = df[['Return', 'Moving_Avg_20', 'Moving_Avg_50', 'Volume', 'Volatility']]

    #standardize the data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return df, features_scaled

#function to execute trading logic
def execute_trading_logic():
    df, features_scaled = fetch_and_preprocess_data(ticker)

    #make predictions
    predictions = (model.predict(features_scaled) > 0.5).astype(int)
    df.loc[:, 'Prediction'] = predictions

    #get the latest prediction
    latest_prediction = df['Prediction'].iloc[-1]

    #get the current account information
    account = api.get_account()
    cash_available = float(account.cash)

    #get the current position
    positions = api.list_positions()
    position_size = 0
    for position in positions:
        if position.symbol == ticker:
            position_size = int(position.qty)

    #determine trade action
    if latest_prediction == 1 and position_size == 0 and cash_available >= df['Close'].iloc[-1]:
        #buy signal, no current position, and enough cash available
        print(f"Buying 1 share of {ticker}")
        api.submit_order(
            symbol=ticker,
            qty=1,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
    elif latest_prediction == 0 and position_size > 0:
        #sell signal and currently holding position
        print(f"Selling all shares of {ticker}")
        api.submit_order(
            symbol=ticker,
            qty=position_size,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
    else:
        if latest_prediction == 1 and position_size > 0:
            print("No trade action needed: Buy signal, but already holding position.")
        elif latest_prediction == 0 and position_size == 0:
            print("No trade action needed: Sell signal, but no position held.")
        elif latest_prediction == 1 and position_size == 0 and cash_available < df['Close'].iloc[-1]:
            print("No trade action needed: Buy signal, but not enough cash available.")
        else:
            print("No trade action needed for unknown reasons.")

    #check order status
    orders = api.list_orders(status='all')
    for order in orders:
        if order.symbol == ticker:
            print(f"Order {order.id} for {order.symbol}: {order.qty} shares {order.side} at {order.filled_avg_price}")

#main loop to check for trading actions every minute
while True:
    execute_trading_logic()
    print("Waiting for 60 seconds...")
    time.sleep(60)
