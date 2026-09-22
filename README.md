# AI-Based Stock Price Prediction and Market Analysis System

A moderate academic web application with three modules: stock market analysis, AI next-day price prediction, and watchlist/model analytics. It uses FastAPI, TensorFlow/Keras, scikit-learn, pandas, NumPy, yfinance, plain HTML/CSS/JavaScript, and Chart.js.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The exporter follows `Stock_Price_Prediction (1).ipynb`: its AAPL date range, Close-only scaler/sequences, indicators, chronological split, architecture, and training settings are documented in `docs/MODEL.md`.

## Train and run

```powershell
python ml/train_model.py
python -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/` or serve `frontend/` with VS Code Live Server. Swagger is available at `/docs`.

## API

- `GET /health`
- `GET /stocks/{ticker}`
- `GET /stocks/{ticker}/history?period=1y`
- `GET /stocks/{ticker}/indicators`
- `POST /predict` with `{"ticker":"AAPL"}`
- `GET /watchlist`, `POST /watchlist`, `DELETE /watchlist/{ticker}`
- `GET /analytics`

## Testing

```powershell
pytest
```

Tests cover sequence construction and inference shape. Live Yahoo Finance, TensorFlow training, and browser checks require network/dependencies and are intentionally not claimed here.

## Structure

`api.py` owns routes; `services/` owns market data, indicators, and watchlist storage; `ml/` owns training, evaluation, and inference; `frontend/` contains the five pages; `docs/` records architecture, model assumptions, and limitations. Run the notebook's multi-stock export cell to create one model and scaler per ticker. The API never applies AAPL artifacts to another stock.
