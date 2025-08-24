import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class StockPredictor:
    def __init__(self, symbol, sequence_length=60):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self, period="1y"):
        # Fetch stock data
        data = yf.download(self.symbol, period=period, auto_adjust=True)
        data = data[['Close']]

        # Error handling for empty data
        if data.empty:
            raise ValueError(f"No data found for symbol '{self.symbol}'. Check if symbol is correct.")
        
        return data

    def prepare_data(self, data):
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def train_model(self):
        data = self.fetch_data()
        X, y = self.prepare_data(data)

        # Build LSTM model
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    def predict(self, days_ahead=1):
        try:
            # Fetch historical data
            data = self.fetch_data()
            if data.empty:
                raise ValueError(f"No data found for symbol '{self.symbol}'.")

            # Get the latest stock price
            current_price = float(data['Close'].iloc[-1])

            # Train model if not already trained
            if self.model is None:
                self.train_model()

            # Prepare last sequence for prediction
            last_sequence = self.scaler.transform(data[-self.sequence_length:])
            X_test = np.array([last_sequence[:, 0]])
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            predictions = []
            for _ in range(days_ahead):
                pred = self.model.predict(X_test, verbose=0)
                pred_value = float(self.scaler.inverse_transform(pred.reshape(-1, 1))[0, 0])
                predictions.append(round(pred_value, 2))

                # Update sequence with the new prediction for next step
                pred_scaled = self.scaler.transform(pred.reshape(-1, 1))
                X_test = np.concatenate((X_test[:, 1:, :], pred_scaled.reshape(1, 1, 1)), axis=1)

            return {
                "symbol": self.symbol,
                "current_price": round(current_price, 2),
                "predicted_prices": predictions,
                "days_ahead": days_ahead
            }

        except ValueError as ve:
            return {"symbol": self.symbol, "error": str(ve)}
        except Exception as e:
            return {"symbol": self.symbol, "error": "Unexpected error occurred: " + str(e)}
