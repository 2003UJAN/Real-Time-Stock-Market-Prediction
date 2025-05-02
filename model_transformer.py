import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

def build_transformer_model(input_shape, embed_dim=32, num_heads=2, ff_dim=32, dropout_rate=0.1):
    """
    Builds and compiles a simple Transformer model for time series prediction.

    Args:
        input_shape (tuple): Shape of the input time series data (timesteps, features).
        embed_dim (int): Dimensionality of the embedding space.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward layer.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.keras.Model: Compiled Transformer model.
    """
    inputs = layers.Input(shape=input_shape)
    x = TransformerBlock(embed_dim, num_heads, ff_dim, rate=dropout_rate)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
