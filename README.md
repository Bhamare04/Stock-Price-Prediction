# 📈 AI Stock Price Prediction and Market Analysis System

An AI-powered web application that predicts the next stock closing price using an LSTM neural network and combines the prediction with technical indicators to provide market analysis, prediction confidence, and directional signals.

## 🚀 Live Demo

🌐 **Live Application:**  
https://stock-price-prediction-1-3u3u.onrender.com

📚 **API Documentation:**  
https://stock-price-prediction-1-3u3u.onrender.com/docs

---

## 📌 Project Overview

The **AI Stock Price Prediction and Market Analysis System** is a machine-learning-based application designed to analyze historical stock market data and predict the next closing price.

The system uses an **LSTM (Long Short-Term Memory)** neural network to learn patterns from historical stock prices.

It also combines the prediction with technical indicators such as:

- RSI
- MACD
- Moving Averages
- Historical price trends

The system generates:

- Current stock price
- Predicted next closing price
- Expected percentage change
- Market direction
- Trend
- Prediction confidence
- Prediction range
- Technical indicator signals
- AI-generated market analysis

> ⚠️ This project is intended for educational and research purposes. Stock predictions are estimates and should not be considered financial advice.

---

## ✨ Key Features

### 🤖 AI Stock Prediction
Uses an LSTM neural network to predict the next stock closing price.

### 📊 Technical Analysis
Analyzes technical indicators including:

- RSI
- MACD
- Moving Average

### 📈 Market Direction

The system classifies the predicted movement as:

- `UP`
- `DOWN`
- `NEUTRAL`

### 🎯 Prediction Confidence

Provides an estimated confidence level based on historical model prediction errors.

### 📉 Prediction Range

Displays an estimated range around the predicted price to represent uncertainty.

### 🧠 Market Analysis

Combines:

- Model prediction
- Price movement
- Technical indicators
- Historical prediction errors

to generate a concise market analysis.

### ⭐ Watchlist

Users can add and remove stocks from their watchlist.

### 📊 Historical Data

The application retrieves historical stock market data and displays it through the web interface.

### 🌐 REST API

FastAPI provides endpoints for:

- Stock prices
- Historical data
- Technical indicators
- Predictions
- Watchlist management
- Health monitoring

### 💻 Web Dashboard

A browser-based interface allows users to interact with the prediction system without directly using the API.

---

## 🏗️ System Architecture

```text
                 ┌──────────────────────┐
                 │      User / Web UI   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │       FastAPI        │
                 │       Backend        │
                 └──────────┬───────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
       ┌───────────┐  ┌────────────┐  ┌──────────────┐
       │ Market    │  │ Technical  │  │ Watchlist    │
       │ Data      │  │ Indicators │  │ Service      │
       └─────┬─────┘  └─────┬──────┘  └──────────────┘
             │              │
             └───────┬──────┘
                     ▼
              ┌──────────────┐
              │ LSTM Model   │
              │ TensorFlow   │
              └──────┬───────┘
                     │
                     ▼
             ┌─────────────────┐
             │ Market Analysis │
             └────────┬────────┘
                      │
                      ▼
                Prediction &
                Analysis Result
