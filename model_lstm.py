from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """
    Builds and returns a compiled LSTM model for time series prediction.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        units (int): Number of LSTM units per layer.
        dropout_rate (float): Dropout rate to prevent overfitting.

    Returns:
        model (Sequential): Compiled LSTM model.
    """
    model = Sequential()
    
    # First LSTM layer (return sequences)
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(units))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
