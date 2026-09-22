import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from ml.predict import create_sequence
from ml.train_model import make_sequences
from services.market_analysis_service import confidence_category, generate_market_analysis, interpret_macd, interpret_rsi, prediction_interval, predicted_change_percent


def test_sequence_shapes():
    values = np.arange(100, dtype=float).reshape(-1, 1)
    x, y = make_sequences(values, lookback=60)
    assert x.shape == (40, 60, 1)
    assert y.shape == (40,)


def test_latest_inference_sequence_uses_lookback():
    scaler = MinMaxScaler().fit(np.arange(100, dtype=float).reshape(-1, 1))
    sequence = create_sequence(np.arange(100, dtype=float), scaler)
    assert sequence.shape == (1, 60, 1)
    np.testing.assert_allclose(sequence[0, 0, 0], scaler.transform([[40]])[0, 0])


def test_predicted_change_percent():
    assert predicted_change_percent(100, 102) == 2


def test_prediction_interval_and_confidence():
    interval = prediction_interval(100, [-2, -1, 0, 1, 2])
    assert interval["lower"] <= 100 <= interval["upper"]
    assert confidence_category({"lower": 99, "upper": 101}, 100) == "Low"


def test_prediction_interval_requires_historical_errors():
    with pytest.raises(ValueError, match="Historical prediction errors"):
        prediction_interval(100, None)


def test_indicator_interpretation():
    assert interpret_rsi(25) == "Oversold"
    assert interpret_rsi(50) == "Neutral"
    assert interpret_macd(2, 1) == "Positive"
    assert interpret_macd(1, 2) == "Negative"


def test_market_analysis_is_neutral_and_explanatory():
    text = generate_market_analysis(1.2, {"rsi": "Neutral", "macd": "Positive", "moving_average": "Positive"}, "Moderate")
    assert "predicts" in text
    assert "BUY" not in text and "SELL" not in text
