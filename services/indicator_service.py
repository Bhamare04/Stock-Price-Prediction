"""Technical indicators used by the dashboard."""
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator

from services.market_data_service import get_stock_data


def calculate_indicators(data):
    close = data["Close"].astype(float)
    returns = close.pct_change()
    result = pd.DataFrame({
        "date": data["Date"].dt.strftime("%Y-%m-%d"),
        "SMA_20": SMAIndicator(close, window=20).sma_indicator(),
        "SMA_50": SMAIndicator(close, window=50).sma_indicator(),
        "EMA_20": EMAIndicator(close, window=20).ema_indicator(),
        "MACD": MACD(close).macd(),
        "MACD_signal": MACD(close).macd_signal(),
        "RSI": RSIIndicator(close, window=14).rsi(),
        "Volatility": returns.rolling(20).std(),
    }).dropna()
    return [{"date": row["date"], **{key: float(row[key]) for key in result.columns if key != "date"}} for _, row in result.iterrows()]

def get_indicators(ticker, period="1y"):
    return calculate_indicators(get_stock_data(ticker, period))
