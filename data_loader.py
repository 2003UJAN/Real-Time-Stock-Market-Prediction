import yfinance as yf
import pandas as pd
import numpy as np
import yaml

def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

def load_stock_data():
    cfg = load_config()
    df = yf.download(cfg["ticker"], start=cfg["start_date"], end=cfg["end_date"])
    df = df[['Close']].dropna()
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def preprocess_data(df, seq_length):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = create_sequences(scaled, seq_length)
    return X, y, scaler
