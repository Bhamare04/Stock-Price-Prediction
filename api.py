"""FastAPI application for stock analysis and LSTM inference."""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ml.evaluate_model import load_metrics, load_prediction_errors
from ml.predict import load_artifacts, load_artifacts_for_ticker, predict_next_close
from services.indicator_service import get_indicators
from services.market_analysis_service import build_market_analysis
from services.market_data_service import get_historical_data, get_latest_price, get_stock_data
from services.watchlist_service import add_stock, get_watchlist, remove_stock

app = FastAPI(title="AI Stock Price Prediction and Market Analysis System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend"), name="static")
MODEL = None
SCALER = None
try:
    MODEL, SCALER = load_artifacts()
except (FileNotFoundError, ImportError, OSError):
    pass


class PredictionRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)


class WatchlistRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"success": False, "error": str(exc)})


@app.get("/")
def root():
    return FileResponse(BASE_DIR / "frontend" / "index.html")


@app.get("/index.html")
def index_page():
    return FileResponse(BASE_DIR / "frontend" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None and SCALER is not None}


@app.get("/stocks/{ticker}")
def stock(ticker: str):
    try:
        return get_latest_price(ticker)
    except (ValueError, LookupError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/stocks/{ticker}/history")
def history(ticker: str, period: str = "1y"):
    try:
        return {"ticker": ticker.upper(), "data": get_historical_data(ticker, period)}
    except (ValueError, LookupError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/stocks/{ticker}/indicators")
def indicators(ticker: str):
    try:
        return {"ticker": ticker.upper(), "data": get_indicators(ticker)}
    except (ValueError, LookupError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictionRequest):
    ticker = request.ticker.strip().upper()
    try:
        ticker_model, ticker_scaler = load_artifacts_for_ticker(ticker)
        quote = get_latest_price(ticker)
        data = get_stock_data(ticker, "1y")
        predicted = predict_next_close(data["Close"].astype(float).values, ticker_model, ticker_scaler)
        current = quote["current_price"]
        latest_indicators = (get_indicators(ticker) or [])[-1]
        analysis = build_market_analysis(current, predicted, latest_indicators, load_prediction_errors(ticker))
        change = analysis["predicted_change_percent"]
        direction = "UP" if change > 0.05 else "DOWN" if change < -0.05 else "NEUTRAL"
        return {"success": True, "ticker": quote["ticker"], "current_price": current,
                "predicted_price": predicted, "predicted_change_percent": change,
            "direction": direction, "model_signal": direction, **analysis}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Prediction unavailable for {ticker}: train this ticker from the notebook multi-stock export cell first.") from exc
    except (ValueError, LookupError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/watchlist")
def watchlist():
    return {"stocks": get_watchlist()}


@app.post("/watchlist")
def watchlist_add(request: WatchlistRequest):
    try:
        return {"stocks": add_stock(request.ticker)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/watchlist/{ticker}")
def watchlist_remove(ticker: str):
    try:
        return {"stocks": remove_stock(ticker)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/analytics")
def analytics():
    metrics = load_metrics()
    if metrics is None:
        return {"available": False, "message": "Train the model to generate evaluation metrics."}
    return {"available": True, **metrics}


@app.get("/prediction.html")
def prediction_page():
    return FileResponse(BASE_DIR / "frontend" / "prediction.html")


@app.get("/watchlist.html")
def watchlist_page():
    return FileResponse(BASE_DIR / "frontend" / "watchlist.html")


@app.get("/analytics.html")
def analytics_page():
    return FileResponse(BASE_DIR / "frontend" / "analytics.html")


@app.get("/about.html")
def about_page():
    return FileResponse(BASE_DIR / "frontend" / "about.html")


@app.get("/style.css")
def stylesheet():
    return FileResponse(BASE_DIR / "frontend" / "style.css", media_type="text/css")


@app.get("/script.js")
def javascript():
    return FileResponse(BASE_DIR / "frontend" / "script.js", media_type="text/javascript")


app.mount("/frontend", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="frontend")
