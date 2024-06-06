import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from kerastuner.tuners import RandomSearch

#collect 5 years of historical data for stock
ticker_symbol = 'GOOG'
ticker_data = yf.Ticker(ticker_symbol)
df = ticker_data.history(period='5y')

#calculate percent change in closing price and moving averages
df = df.dropna()
df['Return'] = df['Close'].pct_change()
df['Moving_Avg_20'] = df['Close'].rolling(window=20).mean()
df['Moving_Avg_50'] = df['Close'].rolling(window=50).mean()
df['Volume'] = df['Volume']
df['Volatility'] = df['Close'].rolling(window=20).std()
df = df.dropna()

#create features and target
features = df[['Return', 'Moving_Avg_20', 'Moving_Avg_50', 'Volume', 'Volatility']]
target = (df['Close'].shift(-1) > df['Close']).astype(int)

#ensure same length
features = features.iloc[:-1]
target = target.iloc[:-1]

#split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#convert targets to numpy arrays since TensorFlow wants inputs to be in numpy arr
y_train = np.array(y_train)
y_test = np.array(y_test)

#define a function to build the model (for KerasTuner)
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', min_value=32, max_value=128, step=32), activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(hp.Float('dropout1', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units2', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout2', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Use KerasTuner to find the best hyperparameters
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='keras_tuner_dir',
    project_name='stock_price_prediction'
)

# Perform hyperparameter search
tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the model
y_pred = (best_model.predict(X_test) > 0.5).astype(int)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#save the best model
best_model.save(f'{ticker_symbol} - best_stock_price_model.h5')
print("Model saved to best_stock_price_model.h5")

# Backtest
features_with_prediction = scaler.transform(features)
predictions = (best_model.predict(features_with_prediction) > 0.5).astype(int)
features_with_prediction = features.copy()  # Make a copy to avoid modifying the original features
features_with_prediction['Prediction'] = predictions

# Align the prediction DataFrame with the original DataFrame
df = df.iloc[:-1]
df['Prediction'] = features_with_prediction['Prediction']
df['Strategy_Return'] = df['Return'] * df['Prediction'].shift(1)
df['Cumulative_Strategy_Return'] = (df['Strategy_Return'] + 1).cumprod()

print(df[['Return', 'Strategy_Return', 'Cumulative_Strategy_Return']].tail())
