import streamlit as st
import pandas as pd
from datetime import datetime
from model_lstm import build_lstm_model
from arima_model import predict_arima
from data_loader import load_stock_data, preprocess_data
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

# List of 50 stock tickers
TICKERS = {
    'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Alphabet (Google)': 'GOOGL', 'Amazon': 'AMZN', 'Tesla': 'TSLA',
    'Meta (Facebook)': 'META', 'NVIDIA': 'NVDA', 'Netflix': 'NFLX', 'Alibaba': 'BABA', 'Visa': 'V',
    'Johnson & Johnson': 'JNJ', 'Walmart': 'WMT', 'UnitedHealth': 'UNH', 'JPMorgan Chase': 'JPM', 'Procter & Gamble': 'PG',
    'Mastercard': 'MA', 'Disney': 'DIS', 'Home Depot': 'HD', 'Bank of America': 'BAC', 'Pfizer': 'PFE',
    'Adobe': 'ADBE', 'Intel': 'INTC', 'Coca-Cola': 'KO', 'PepsiCo': 'PEP', 'ExxonMobil': 'XOM',
    'Merck': 'MRK', 'Cisco': 'CSCO', 'AT&T': 'T', 'Chevron': 'CVX', 'Verizon': 'VZ',
    'Nike': 'NKE', 'Oracle': 'ORCL', 'Abbott': 'ABT', 'Salesforce': 'CRM', 'Eli Lilly': 'LLY',
    'McDonald\'s': 'MCD', 'Accenture': 'ACN', 'Costco': 'COST', 'Medtronic': 'MDT', 'Texas Instruments': 'TXN',
    'Applied Materials': 'AMAT', 'IBM': 'IBM', 'Broadcom': 'AVGO', 'Qualcomm': 'QCOM', 'Thermo Fisher': 'TMO',
    'General Electric': 'GE', 'UPS': 'UPS', 'Philip Morris': 'PM', 'Caterpillar': 'CAT', 'Honeywell': 'HON'
}

st.set_page_config(page_title="Real-Time Stock Predictor", layout="wide")
st.title("📈 Real-Time Stock Price Prediction App")

st.sidebar.header("⚙️ Configuration")

ticker = st.sidebar.selectbox("Select Ticker", TICKERS, index=0)
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
model_choice = st.sidebar.radio("Select Model", ["LSTM", "ARIMA"])
seq_length = st.sidebar.slider("Sequence Length (for LSTM)", 20, 100, 60)

try:
    df = load_stock_data(ticker, str(start_date), str(end_date))
    st.write(f"### {ticker} Stock Closing Price")
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
        st.write("### 📊 LSTM Prediction Results")
        st.line_chart(result_df)

    elif model_choice == "ARIMA":
        y_test, y_pred = predict_arima(df)
        result_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        })
        st.write("### 📊 ARIMA Prediction Results")
        st.line_chart(result_df)

except Exception as e:
    st.error(f"❌ Error: {e}")
