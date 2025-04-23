import streamlit as st
import numpy as np
from data_loader import load_stock_data, preprocess_data, load_config
from model_lstm import build_lstm_model
from model_transformer import build_transformer_model
from utils import plot_predictions

st.title("📈 Stock Price Prediction App")

cfg = load_config()
st.sidebar.write("Model Config")
st.sidebar.write(cfg)

df = load_stock_data()
seq_length = cfg["sequence_length"]
X, y, scaler = preprocess_data(df, seq_length)

X_train, y_train = X[:-30], y[:-30]
X_test, y_test = X[-30:], y[-30:]

model_type = cfg["model_type"]
input_shape = (X_train.shape[1], X_train.shape[2])

if model_type == "lstm":
    model = build_lstm_model(input_shape)
else:
    model = build_transformer_model(input_shape)

with st.spinner("Training model..."):
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predicted.shape[0], 1)), predicted), axis=1))[:, 1]
actual_prices = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 1)), y_test.reshape(-1,1)), axis=1))[:, 1]

fig = plot_predictions(actual_prices, predicted_prices)
st.pyplot(fig)
