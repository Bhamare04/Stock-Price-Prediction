"""Rule-based market analysis built from model errors and technical signals."""
from __future__ import annotations

from math import isfinite

import numpy as np


LOW_UNCERTAINTY = 0.03
MODERATE_UNCERTAINTY = 0.07
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0


def predicted_change_percent(current_price: float, predicted_price: float) -> float:
    """Return the predicted price change relative to the current price."""
    if not isfinite(current_price) or not isfinite(predicted_price) or current_price <= 0:
        raise ValueError("Current and predicted prices must be finite positive numbers.")
    return (predicted_price - current_price) / current_price * 100


def prediction_interval(predicted_price: float, errors: list[float] | None) -> dict[str, float]:
    """Build an actual-price prediction interval from historical residuals."""
    if not isfinite(predicted_price):
        raise ValueError("Predicted price must be finite.")
    valid_errors = [float(error) for error in (errors or []) if isfinite(float(error))]
    if not valid_errors:
        raise ValueError("Historical prediction errors are unavailable.")
    lower_error, upper_error = np.percentile(valid_errors, [5, 95])
    lower = max(0.0, predicted_price + float(lower_error))
    upper = max(lower, predicted_price + float(upper_error))
    return {"lower": float(lower), "upper": float(upper)}


def confidence_category(prediction_range: dict[str, float], current_price: float) -> str:
    """Classify interval width relative to price; this is not a probability."""
    if not isfinite(current_price) or current_price <= 0:
        raise ValueError("Current price must be a finite positive number.")
    width = prediction_range["upper"] - prediction_range["lower"]
    relative_uncertainty = width / current_price
    if relative_uncertainty < LOW_UNCERTAINTY:
        return "Low"
    if relative_uncertainty < MODERATE_UNCERTAINTY:
        return "Moderate"
    return "High"


def interpret_rsi(rsi: float | None) -> str:
    """Interpret RSI using conventional oversold and overbought thresholds."""
    if rsi is None or not isfinite(float(rsi)):
        return "Unavailable"
    if rsi < RSI_OVERSOLD:
        return "Oversold"
    if rsi > RSI_OVERBOUGHT:
        return "Overbought"
    return "Neutral"


def interpret_macd(macd: float | None, signal: float | None) -> str:
    """Interpret MACD relative to its signal line."""
    if macd is None or signal is None or not isfinite(float(macd)) or not isfinite(float(signal)):
        return "Unavailable"
    if macd > signal:
        return "Positive"
    if macd < signal:
        return "Negative"
    return "Neutral"


def interpret_moving_average(price: float, moving_average: float | None) -> str:
    """Interpret price relative to the most useful available moving average."""
    if moving_average is None or not isfinite(float(moving_average)):
        return "Unavailable"
    if price > moving_average:
        return "Positive"
    if price < moving_average:
        return "Negative"
    return "Neutral"


def generate_market_analysis(predicted_change: float, signals: dict[str, str], confidence: str) -> str:
    """Generate a concise neutral explanation from the model and indicators."""
    direction = "increase" if predicted_change > 0.05 else "decrease" if predicted_change < -0.05 else "remain broadly stable"
    magnitude = abs(predicted_change)
    size = "modest" if magnitude < 2 else "meaningful"
    available = [value for value in signals.values() if value != "Unavailable"]
    positive = sum(value == "Positive" for value in available)
    negative = sum(value in {"Negative", "Overbought"} for value in available)
    if positive and negative:
        momentum = "technical indicators are mixed"
    elif positive:
        momentum = "technical indicators suggest positive momentum"
    elif negative:
        momentum = "technical indicators suggest negative momentum"
    else:
        momentum = "technical indicators do not provide a clear directional signal"
    uncertainty = {"Low": "lower", "Moderate": "moderate", "High": "higher"}.get(confidence, "material")
    return f"The model predicts a {size} {direction}. {momentum.capitalize()}, with {uncertainty} uncertainty around the estimate."


def build_market_analysis(current_price: float, predicted_price: float, indicators: dict[str, float | None], errors: list[float] | None) -> dict:
    """Combine prediction, residuals, and latest indicators into API-ready data."""
    change = predicted_change_percent(current_price, predicted_price)
    prediction_range = prediction_interval(predicted_price, errors)
    confidence = confidence_category(prediction_range, current_price)
    signals = {
        "rsi": interpret_rsi(indicators.get("RSI")),
        "macd": interpret_macd(indicators.get("MACD"), indicators.get("MACD_signal")),
        "moving_average": interpret_moving_average(current_price, indicators.get("SMA_20") or indicators.get("EMA_20")),
    }
    trend = "Positive" if change > 0.05 else "Negative" if change < -0.05 else "Neutral"
    return {"predicted_change_percent": float(change), "trend": trend, "confidence": confidence, "prediction_range": prediction_range, "signals": signals, "analysis": generate_market_analysis(change, signals, confidence)}