import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from model_lstm import build_lstm_model
from arima_model import predict_arima, plot_arima_results, arima_rmse
from data_loader import load_stock_data, preprocess_data
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Ticker symbol dictionary with full company names
TICKERS = {
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', 'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.', 'TSLA': 'Tesla Inc.', 'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corp.', 'NFLX': 'Netflix Inc.', 'BABA': 'Alibaba Group',
    'V': 'Visa Inc.', 'JNJ': 'Johnson & Johnson', 'WMT': 'Walmart Inc.',
    'UNH': 'UnitedHealth Group', 'JPM': 'JPMorgan Chase & Co.', 'PG': 'Procter & Gamble',
    'MA': 'Mastercard Inc.', 'DIS': 'Walt Disney Co.', 'HD': 'Home Depot',
    'BAC': 'Bank of America', 'PFE': 'Pfizer Inc.', 'AMAT': 'Applied Materials',
    'IBM': 'IBM Corp.', 'AVGO': 'Broadcom Inc.', 'QCOM': 'Qualcomm Inc.',
    'TMO': 'Thermo Fisher Scientific', 'GE': 'General Electric', 'UPS': 'United Parcel Service',
    'PM': 'Philip Morris', 'CAT': 'Caterpillar Inc.', 'HON': 'Honeywell International',
    'ORCL': 'Oracle Corp.', 'INTC': 'Intel Corp.', 'CSCO': 'Cisco Systems',
    'BA': 'Boeing Co.', 'CRM': 'Salesforce Inc.', 'PEP': 'PepsiCo Inc.',
    'KO': 'Coca-Cola Co.', 'COST': 'Costco Wholesale', 'MDT': 'Medtronic PLC',
    'TXN': 'Texas Instruments', 'MCD': 'McDonald\'s Corp.', 'GS': 'Goldman Sachs',
    'SBUX': 'Starbucks Corp.', 'DE': 'Deere & Co.', 'MMM': '3M Co.', 'BLK': 'BlackRock Inc.',
    'FDX': 'FedEx Corp.', 'ADBE': 'Adobe Inc.', 'PYPL': 'PayPal Holdings'
}

# Streamlit UI setup
st.set_page_config(page_title="Real-Time Stock Predictor", layout="wide")
st.title("📈 Real-Time Stock Price Prediction App")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
selected_ticker = st.sidebar.selectbox("Select Company", list(TICKERS.keys()), format_func=lambda x: f"{x} - {TICKERS[x]}")
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
model_choice = st.sidebar.radio("Select Model", ["LSTM", "ARIMA"])
seq_length = st.sidebar.slider("Sequence Length (for LSTM)", 20, 100, 60)

# Validation
if start_date >= end_date:
    st.error("❌ Start Date must be earlier than End Date.")
    st.stop()

try:
    # Load and validate stock data
    df = load_stock_data(selected_ticker, str(start_date), str(end_date))
    if df.empty or len(df) < seq_length:
        st.warning("⚠️ Not enough data available for the selected ticker or date range.")
        st.stop()

    # Plot raw closing price
    st.write(f"### 📊 {TICKERS[selected_ticker]} ({selected_ticker}) Stock Closing Price")
    st.line_chart(df['Close'])

    # Model selection
    if model_choice == "LSTM":
        X, y, scaler = preprocess_data(df, seq_length)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = build_lstm_model((X.shape[1], 1))
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1,
                  verbose=0, callbacks=[EarlyStopping(patience=3)])

        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        result_df = pd.DataFrame({
            "Actual": y_test_inv.flatten(),
            "Predicted": y_pred_inv.flatten()
        })
        st.write("### 🤖 LSTM Prediction Results")
        st.line_chart(result_df)

    elif model_choice == "ARIMA":
        actual, predicted = predict_arima(df)

        st.write("### 🔍 ARIMA Prediction Results")
        fig = plot_arima_results(actual, predicted)
        st.pyplot(fig)

        rmse = arima_rmse(actual, predicted)
        st.success(f"📉 RMSE (ARIMA): {rmse:.4f}")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
