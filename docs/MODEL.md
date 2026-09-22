# Model

The deployment contract follows `Stock_Price_Prediction (1).ipynb`. It downloads AAPL from `2014-01-01` through `2025-01-01`, uses historical Close prices, a `MinMaxScaler(feature_range=(0, 1))`, 60 previous trading days, and the next closing price as the target.

The network is `LSTM(50, return_sequences=True) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> Dense(1)`, compiled with Adam and mean squared error, trained for 20 epochs with batch size 32. Training uses a chronological 80/20 split; the scaler is fitted on the training portion only.

The training export stores `stock_lstm.keras`, `scaler.pkl`, and `metrics.json`. The notebook fits its scaler before creating sequences, so the exporter preserves that behavior. The notebook explicitly reports LSTM MAE and RMSE; this exporter also computes R² as a supplementary regression metric.
