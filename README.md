# Algorithmic Trading Bot

This project contains an algorithmic trading bot that uses machine learning to predict stock prices and make trading decisions. The bot fetches historical stock data, preprocesses it, makes predictions using a pre-trained model, and executes trades using the Alpaca API.

## Project Structure

```
AlgorithmTrading/
├── keras_tuner_dir/
├── momentum_bots/
├── steps/
├── .env
├── GOOG - best_stock_price_model.h5
├── INTC - best_stock_price_model.h5
├── ml_stock_training.py
└── ml_trading_bot.py
```

- `keras_tuner_dir/`: Directory for Keras Tuner files.
- `momentum_bots/`: Directory for momentum-based trading bots.
- `steps/`: Directory for step-by-step guides or additional scripts.
- `.env`: Environment file containing API keys (not included in the repository).
- `GOOG - best_stock_price_model.h5`: Pre-trained model for trading GOOG stock.
- `INTC - best_stock_price_model.h5`: Pre-trained model for trading INTC stock.
- `ml_stock_training.py`: Script to train the machine learning model.
- `ml_trading_bot.py`: Script to run the trading bot.

## Setup

### Prerequisites

- Python 3.7+
- Alpaca API account (for paper trading or live trading)
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mszy123/Matt-Algorithm-Trading-Bot.git
   cd Matt-Algorithm-Trading-Bot
   ```

2. Install the required packages:
   

3. Create a `.env` file in the root directory and add your Alpaca API credentials:
   ```
   API_KEY=your_alpaca_api_key
   API_SECRET=your_alpaca_api_secret
   ```

## Usage

### Training the Model

Use the `ml_stock_training.py` script to train the machine learning model:

```bash
python ml_stock_training.py
```

### Running the Trading Bot

Use the `ml_trading_bot.py` script to run the trading bot:

```bash
python ml_trading_bot.py
```

The bot will fetch historical data, make predictions, and execute trades based on the predictions every minute.

## Customization

### Change the Stock Ticker

To change the stock ticker that the bot trades, modify the `ticker` variable in `ml_trading_bot.py`:

```python
ticker = 'GOOG'
```

### Change the Model

To use a different pre-trained model, replace the model file and update the `model = load_model('path_to_model.h5')` line in `ml_trading_bot.py`.

## API

- [Alpaca](https://alpaca.markets/) for providing a robust trading API.
- [Yahoo Finance](https://finance.yahoo.com/) for providing historical stock data.


