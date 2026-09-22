"""Train and export the LSTM described in the supplied notebook brief."""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

TICKER = "AAPL"
LOOKBACK = 60
EPOCHS = 20
BATCH_SIZE = 32
MODEL_DIR = Path(__file__).parent / "model"


def make_sequences(values, lookback=LOOKBACK):
    x_values, y_values = [], []
    for index in range(lookback, len(values)):
        x_values.append(values[index - lookback:index, 0])
        y_values.append(values[index, 0])
    return np.array(x_values).reshape(-1, lookback, 1), np.array(y_values)


def build_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train():
    raw = yf.download(TICKER, start="2014-01-01", end="2025-01-01", progress=False)
    if raw.empty:
        raise RuntimeError("No training data was returned for AAPL.")
    if hasattr(raw.columns, "levels"):
        raw.columns = raw.columns.get_level_values(0)
    close = raw["Close"].astype(float).values.reshape(-1, 1)
    if len(close) <= LOOKBACK + 10:
        raise RuntimeError("Not enough historical data to train the model.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close)
    scaled = scaler.transform(close)
    x_all, y_all = make_sequences(scaled)
    train_end = int(len(x_all) * 0.8)
    x_train, y_train = x_all[:train_end], y_all[:train_end]
    x_test, y_test = x_all[train_end:], y_all[train_end:]

    model = build_model()
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    predictions = scaler.inverse_transform(model.predict(x_test, verbose=0))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    metrics = {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "r2": float(r2_score(actual, predictions)),
        "ticker": TICKER, "lookback": LOOKBACK, "target": "Next-Day Closing Price",
        "model": "LSTM",
        "actual": actual.ravel().tolist(), "predicted": predictions.ravel().tolist(),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "stock_lstm.keras")
    with open(MODEL_DIR / "scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        import json
        json.dump(metrics, file, indent=2)
    print(f"Saved model and scaler to {MODEL_DIR}")
    print(f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    train()
