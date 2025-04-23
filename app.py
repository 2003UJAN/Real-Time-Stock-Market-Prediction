import streamlit as st
import numpy as np
import pandas as pd
from data_loader import load_stock_data, preprocess_data
from model_lstm import build_lstm_model
from arima_model import predict_arima
from utils import plot_predictions
from sklearn.metrics import mean_squared_error
from datetime import date

st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Stock Price Predictor")

# --- Ticker List (50 popular stocks) ---
popular_tickers = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Google (GOOG)": "GOOG",
    "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "Intel (INTC)": "INTC",
    "AMD (AMD)": "AMD",
    "IBM (IBM)": "IBM",
    "Zoom (ZM)": "ZM",
    "Salesforce (CRM)": "CRM",
    "Adobe (ADBE)": "ADBE",
    "PayPal (PYPL)": "PYPL",
    "Spotify (SPOT)": "SPOT",
    "Alibaba (BABA)": "BABA",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "Walmart (WMT)": "WMT",
    "Procter & Gamble (PG)": "PG",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Coca-Cola (KO)": "KO",
    "PepsiCo (PEP)": "PEP",
    "McDonald's (MCD)": "MCD",
    "Visa (V)": "V",
    "Mastercard (MA)": "MA",
    "American Express (AXP)": "AXP",
    "Boeing (BA)": "BA",
    "Disney (DIS)": "DIS",
    "Chevron (CVX)": "CVX",
    "ExxonMobil (XOM)": "XOM",
    "Ford (F)": "F",
    "General Motors (GM)": "GM",
    "Uber (UBER)": "UBER",
    "Lyft (LYFT)": "LYFT",
    "Snapchat (SNAP)": "SNAP",
    "Twitter (TWTR)": "TWTR",
    "Roku (ROKU)": "ROKU",
    "Domino’s (DPZ)": "DPZ",
    "Nike (NKE)": "NKE",
    "Starbucks (SBUX)": "SBUX",
    "Target (TGT)": "TGT",
    "AT&T (T)": "T",
    "Verizon (VZ)": "VZ",
    "Pfizer (PFE)": "PFE",
    "Moderna (MRNA)": "MRNA",
    "Moderna (MRNA)": "MRNA",
    "Lucid (LCID)": "LCID",
    "Rivian (RIVN)": "RIVN"
}

# --- Sidebar Inputs ---
ticker_label = st.sidebar.selectbox("Select a Stock", list(popular_tickers.keys()))
ticker = popular_tickers[ticker_label]

start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", date(2024, 12, 31))
model_type = st.sidebar.selectbox("Model Type", ["LSTM", "ARIMA"])
seq_length = st.sidebar.slider("Sequence Length (LSTM)", 30, 100, 60)

# --- Load Data ---
df = load_stock_data(ticker, start_date, end_date)

if df.empty:
    st.warning("No data found for this ticker and date range.")
    st.stop()

st.subheader(f"{ticker} Closing Price")
st.line_chart(df["Close"])

# --- Predict ---
if model_type == "LSTM":
    X, y, scaler = preprocess_data(df, seq_length)
    X_train, y_train = X[:-30], y[:-30]
    X_test, y_test = X[-30:], y[-30:]

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    with st.spinner("Training LSTM Model..."):
        model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predicted.shape[0], 1)), predicted), axis=1))[:, 1]
    actual_prices = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 1)), y_test.reshape(-1, 1)), axis=1))[:, 1]

elif model_type == "ARIMA":
    actual_prices, predicted_prices = predict_arima(df)

# --- Plot Predictions ---
fig = plot_predictions(actual_prices, predicted_prices)
st.pyplot(fig)

# --- Metrics ---
mse = mean_squared_error(actual_prices, predicted_prices)
st.metric("Mean Squared Error", f"{mse:.2f}")
