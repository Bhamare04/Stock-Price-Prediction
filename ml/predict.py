"""Reusable LSTM inference helpers."""
from pathlib import Path
import pickle

import numpy as np

LOOKBACK = 60
MODEL_PATH = Path(__file__).parent / "model" / "stock_lstm.keras"
SCALER_PATH = Path(__file__).parent / "model" / "scaler.pkl"
MODEL_DIR = Path(__file__).parent / "model"


def create_sequence(close_prices, scaler, lookback=LOOKBACK):
    """Scale close prices and return the latest lookback observations."""
    values = np.asarray(close_prices, dtype=float).reshape(-1, 1)
    if len(values) < lookback:
        raise ValueError(f"At least {lookback} historical closing prices are required.")
    scaled = scaler.transform(values)
    return scaled[-lookback:].reshape(1, lookback, 1)


def load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    if not Path(model_path).exists() or not Path(scaler_path).exists():
        raise FileNotFoundError("Model artifacts are missing. Run python ml/train_model.py first.")
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)
    return model, scaler


def load_artifacts_for_ticker(ticker):
    """Load artifacts trained specifically for one ticker."""
    symbol = ticker.strip().upper()
    if symbol == "AAPL" and MODEL_PATH.exists() and SCALER_PATH.exists():
        return load_artifacts(MODEL_PATH, SCALER_PATH)
    return load_artifacts(
        MODEL_DIR / f"{symbol}_stock_lstm.keras",
        MODEL_DIR / f"{symbol}_scaler.pkl",
    )


def predict_next_close(close_prices, model, scaler, lookback=LOOKBACK):
    sequence = create_sequence(close_prices, scaler, lookback)
    scaled_prediction = model.predict(sequence, verbose=0)
    return float(scaler.inverse_transform(np.asarray(scaled_prediction).reshape(-1, 1))[0, 0])
