# Architecture

The browser frontend uses `fetch()` to call the FastAPI REST API. FastAPI delegates quote and history work to the Yahoo Finance service, indicator calculations to the indicator service, and next-day inference to the persisted TensorFlow model and scaler.

`User -> HTML/JavaScript -> FastAPI -> services -> yfinance / LSTM -> JSON response -> dashboard`

The API does not train at startup or during requests. Training is a separate command that writes model artifacts and chronological evaluation metrics.
