import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']].dropna()
    if df.empty:
        raise ValueError("No data found for the selected ticker or date range.")
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def preprocess_data(df, seq_length):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = create_sequences(scaled, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # reshape for LSTM input
    return X, y, scaler
