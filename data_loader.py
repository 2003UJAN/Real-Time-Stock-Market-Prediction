import yfinance as yf
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import MinMaxScaler

def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

def load_stock_data():
    cfg = load_config()
    df = yf.download(cfg["ticker"], start=cfg["start_date"], end=cfg["end_date"])
    df = df[['Close']].dropna()

    if df.empty:
        raise ValueError("Stock data could not be fetched. Check ticker or date range.")
    
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

    # Reshape X to be [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler
