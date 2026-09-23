"""Simple JSON-backed watchlist storage."""

import json
from pathlib import Path

from services.market_data_service import _ticker_symbol

WATCHLIST_PATH = Path(__file__).parent.parent / "data" / "watchlist.json"


def get_watchlist():
    # Create the data directory and watchlist file if they don't exist
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not WATCHLIST_PATH.exists():
        WATCHLIST_PATH.write_text("[]", encoding="utf-8")

    try:
        with open(WATCHLIST_PATH, encoding="utf-8") as file:
            data = json.load(file)

        # Make sure the stored data is a list
        if not isinstance(data, list):
            return []

        return data

    except (json.JSONDecodeError, OSError):
        return []


def add_stock(ticker):
    symbol = _ticker_symbol(ticker)
    stocks = get_watchlist()

    if symbol not in stocks:
        stocks.append(symbol)
        _save(stocks)

    return stocks


def remove_stock(ticker):
    symbol = _ticker_symbol(ticker)
    stocks = [stock for stock in get_watchlist() if stock != symbol]

    _save(stocks)

    return stocks


def _save(stocks):
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(WATCHLIST_PATH, "w", encoding="utf-8") as file:
        json.dump(stocks, file, indent=2)