import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads stock data for the given ticker and date range.
    Returns a DataFrame with only the 'Close' column.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty or 'Close' not in df:
        raise ValueError("No data found for the selected ticker or date range.")
    
    df = df[['Close']].dropna()
    return df

def create_sequences(data: np.ndarray, seq_length: int) -> tuple:
    """
    Splits data into sequences for LSTM input.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def preprocess_data(df: pd.DataFrame, seq_length: int) -> tuple:
    """
    Scales the data and returns LSTM-ready sequences.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = create_sequences(scaled, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM input shape: (samples, time steps, features)

    return X, y, scaler
