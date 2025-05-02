import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math

def train_arima(data, order=(5, 1, 0)):
    """
    Trains an ARIMA model on the input time series data.
    """
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    data = np.asarray(data).flatten()
    model = ARIMA(data, order=order)
    return model.fit()

def predict_arima(df, forecast_steps=30):
    """
    Splits the closing price into train/test, trains ARIMA, and forecasts.
    Returns actual test values and predicted values.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    data = df['Close'].dropna()
    if len(data) <= forecast_steps:
        raise ValueError("Not enough data for forecasting.")

    train = data[:-forecast_steps]
    test = data[-forecast_steps:]

    model = train_arima(train)
    forecast = model.forecast(steps=forecast_steps)
    return test.values, forecast

def plot_arima_results(actual, predicted):
    """
    Plots actual vs predicted values.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual, label='Actual', marker='o')
    ax.plot(predicted, label='ARIMA Forecast', marker='x')
    ax.set_title("ARIMA Forecast vs Actual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    return fig

def arima_rmse(actual, predicted):
    """
    Calculates RMSE between actual and predicted values.
    """
    return math.sqrt(mean_squared_error(actual, predicted))
