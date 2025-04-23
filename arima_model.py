import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def train_arima(data, order=(5, 1, 0)):
    model = ARIMA(data, order=order)
    return model.fit()

def predict_arima(df):
    train = df["Close"][:-30]
    test = df["Close"][-30:]

    model = train_arima(train)
    forecast = model.forecast(steps=len(test))
    return test.values, forecast.values
