"""Simple JSON-backed watchlist storage."""
import json
from pathlib import Path

from services.market_data_service import _ticker_symbol

WATCHLIST_PATH = Path(__file__).parent.parent / "data" / "watchlist.json"


def get_watchlist():
    with open(WATCHLIST_PATH, encoding="utf-8") as file:
        return json.load(file)


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
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as file:
        json.dump(stocks, file, indent=2)
