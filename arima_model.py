import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math

def train_arima(data, order=(5, 1, 0)):
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    data = np.asarray(data).flatten()
    return ARIMA(data, order=order).fit()

def predict_arima(df):
    train = df["Close"][:-30]
    test = df["Close"][-30:]

    model = train_arima(train)
    forecast = model.forecast(steps=len(test))
    return test.values, forecast

def plot_arima_results(actual, predicted):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual, label='Actual', marker='o')
    ax.plot(predicted, label='ARIMA Forecast', marker='x')
    ax.set_title("ARIMA Forecast vs Actual")
    ax.legend()
    return fig

def arima_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))
