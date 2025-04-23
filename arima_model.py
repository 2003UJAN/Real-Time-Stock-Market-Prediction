import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def train_arima(data, order=(5, 1, 0)):
    # Ensure data is 1D
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = np.ravel(data.values)  # Flatten to 1D array
    return ARIMA(data, order=order).fit()

def predict_arima(df):
    train = df["Close"][:-30]
    test = df["Close"][-30:]

    model = train_arima(train)
    forecast = model.forecast(steps=len(test))
    return test.values, forecast
