"""Read the persisted evaluation metrics produced during training."""
import json
from pathlib import Path

METRICS_PATH = Path(__file__).parent / "model" / "metrics.json"
MODEL_DIR = Path(__file__).parent / "model"


def load_metrics(ticker=None):
    """Load aggregate metrics and actual-unit test predictions for a ticker."""
    path = MODEL_DIR / f"{ticker.strip().upper()}_metrics.json" if ticker else METRICS_PATH
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_prediction_errors(ticker=None):
    """Return persisted actual-minus-predicted errors in stock-price units."""
    metrics = load_metrics(ticker)
    if not metrics or not metrics.get("actual") or not metrics.get("predicted"):
        return None
    errors = [float(actual) - float(predicted) for actual, predicted in zip(metrics["actual"], metrics["predicted"])]
    return errors or None


if __name__ == "__main__":
    metrics = load_metrics()
    if metrics is None:
        raise SystemExit("No metrics found. Run python ml/train_model.py first.")
    print(json.dumps({key: metrics[key] for key in ("mae", "rmse", "r2")}, indent=2))
