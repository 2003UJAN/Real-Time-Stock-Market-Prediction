import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from model_lstm import build_lstm_model
from arima_model import predict_arima, plot_arima_results, arima_rmse
from data_loader import load_stock_data, preprocess_data
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from utils import plot_predictions  # Import the new function

# Tickers without company names
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'BABA', 'V',
    'JNJ', 'WMT', 'UNH', 'JPM', 'PG', 'MA', 'DIS', 'HD', 'BAC', 'PFE', 'AMAT', 'IBM',
    'AVGO', 'QCOM', 'TMO', 'GE', 'UPS', 'PM', 'CAT', 'HON', 'ORCL', 'INTC', 'CSCO',
    'BA', 'CRM', 'PEP', 'KO', 'COST', 'MDT', 'TXN', 'MCD', 'GS', 'SBUX', 'DE', 'MMM',
    'BLK', 'FDX', 'ADBE', 'PYPL'
]

st.set_page_config(page_title="Real-Time Stock Predictor", layout="wide")
st.title("📈 Real-Time Stock Price Prediction App")

st.sidebar.header("⚙️ Configuration")
selected_ticker = st.sidebar.selectbox("Select Ticker", TICKERS)
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
model_choice = st.sidebar.radio("Select Model", ["LSTM", "ARIMA"])
seq_length = st.sidebar.slider("Sequence Length (for LSTM)", 20, 100, 60)

try:
    df = load_stock_data(selected_ticker, str(start_date), str(end_date))

    if df.empty:
        raise ValueError("No data found for the selected ticker or date range.")

    st.write(f"### 📊 {selected_ticker} Stock Closing Price")
    st.line_chart(df['Close'])

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
        plot_predictions(y_test_inv.flatten(), y_pred_inv.flatten())  # Using the function here
        st.pyplot(plt)  # Displaying the plot

    elif model_choice == "ARIMA":
        actual, predicted = predict_arima(df)

        st.write("### 🔍 ARIMA Prediction Results")
        fig = plot_arima_results(actual, predicted)
        st.pyplot(fig)

        rmse = arima_rmse(actual, predicted)
        st.success(f"📉 RMSE (ARIMA): {rmse:.4f}")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
