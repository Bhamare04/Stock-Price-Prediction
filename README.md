# 📈 Stock Price Prediction using Machine Learning & LSTM

## 📌 Project Overview
This project focuses on predicting stock prices using historical market data.
It uses both traditional machine learning models and a deep learning LSTM model
to understand time-series patterns in stock prices.

The objective of this project is learning and correct implementation,
not guaranteed or perfect stock market prediction.

---

## 🎯 Objectives
- Analyze historical stock market data
- Perform exploratory data analysis (EDA)
- Create meaningful technical indicators
- Build baseline machine learning models
- Implement LSTM for time-series prediction
- Generate simple buy signals
- Evaluate models realistically

---

## 🗂️ Dataset
- Source: Yahoo Finance
- Stock: Apple Inc. (AAPL)
- Frequency: Daily
- Features:
  - Open, High, Low, Close, Volume
  - SMA, EMA, RSI, Volatility

---

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- yfinance

---

## 📅 Project Workflow

### Day 1 – Data Collection
- Downloaded historical stock data
- Data cleaning and basic visualization

### Day 2 – EDA & Feature Engineering
- Price and volume analysis
- Daily returns calculation
- Technical indicator creation

### Day 3 – Machine Learning Models
- Time-series train-test split
- Linear Regression (baseline)
- Random Forest model
- Evaluation using MAE and RMSE

### Day 4 – Deep Learning (LSTM)
- Data scaling using MinMaxScaler
- Sequence creation (60-day window)
- LSTM model training
- Prediction and evaluation

### Day 5 – Finalization
- Buy signal generation
- Visualization of predictions and signals
- Model comparison
- Limitations and conclusions

---

## 📊 Model Evaluation
- Metrics used:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

LSTM captures overall trends but produces smoother predictions due to
market randomness.

---

## 📈 Buy Signal Logic
- If predicted price > current price → Buy signal
- Otherwise → Hold

Signals are visualized on price charts.

---

## ⚠️ Limitations
- Stock prices are affected by news and global events
- Sudden market crashes cannot be predicted
- Past performance does not guarantee future results
- No sentiment or fundamental analysis included

---

## ✅ Conclusion
This project demonstrates an end-to-end stock price prediction pipeline
using machine learning and deep learning techniques with proper
time-series handling and realistic evaluation.

---

## 🚀 Future Improvements
- Add sell signals and backtesting
- Include news or sentiment analysis
- Try other models like GRU or Transformers
- Deploy using a web dashboard

