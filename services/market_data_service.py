"""Yahoo Finance access with small in-memory caching."""
from time import monotonic

import yfinance as yf

_CACHE = {}
CACHE_TTL_SECONDS = 300


def _ticker_symbol(ticker):
    symbol = ticker.strip().upper()
    if not symbol or len(symbol) > 12 or any(not (char.isalnum() or char in ".-^") for char in symbol):
        raise ValueError("Ticker must be a valid stock symbol.")
    return symbol


def _download(symbol, period):
    key = (symbol, period)

    cached = _CACHE.get(key)
    if cached and monotonic() - cached[0] < CACHE_TTL_SECONDS:
        return cached[1].copy()

    try:
        data = yf.download(
            symbol,
            period=period,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError("Market data is temporarily unavailable.") from exc

    if data.empty:
        raise LookupError(f"No market data found for ticker {symbol}.")

    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Remove rows containing missing market values
    data = data.dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]
    )

    if data.empty:
        raise LookupError(f"No valid market data found for ticker {symbol}.")

    _CACHE[key] = (monotonic(), data)

    return data.copy()

def get_stock_data(ticker, period="1y"):
    return _download(_ticker_symbol(ticker), period)


def get_historical_data(ticker, period="1y"):
    data = get_stock_data(ticker, period)
    records = []
    for _, row in data.iterrows():
        date = row["Date"]
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": _number(row["Open"]), "high": _number(row["High"]),
            "low": _number(row["Low"]), "close": _number(row["Close"]),
            "volume": int(row["Volume"]),
        })
    return records


def get_latest_price(ticker):
    symbol = _ticker_symbol(ticker)
    data = _download(symbol, "3mo")
    row = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else row
    current = _number(row["Close"])
    previous_close = _number(previous["Close"])
    return {
        "ticker": symbol, "current_price": current, "previous_close": previous_close,
        "open": _number(row["Open"]), "high": _number(row["High"]), "low": _number(row["Low"]),
        "volume": int(row["Volume"]),
        "daily_change_percent": ((current - previous_close) / previous_close * 100) if previous_close else 0,
    }


def _number(value):
    return float(value)
