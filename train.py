import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from music21 import stream, note, chord

# Load the prepared data
with open('data/prepared_data.pkl', 'rb') as f:
    prepared_data = pickle.load(f)

network_input = prepared_data['network_input']
instrument_input = prepared_data['instrument_input']
network_output = prepared_data['network_output']
note_to_int = prepared_data['note_to_int']
int_to_note = prepared_data['int_to_note']
instrument_to_int = prepared_data['instrument_to_int']
int_to_instrument = prepared_data['int_to_instrument']
n_vocab = prepared_data['n_vocab']
n_instruments = prepared_data['n_instruments']
sequence_length = prepared_data['sequence_length']

# ---------------------------
# Transformer Model Components
# ---------------------------


def get_positional_encoding(max_len, d_model):
    """Generates a positional encoding matrix."""
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    # apply sin to even indices and cos to odd indices
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# Define the Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Ensure embed_dim and ff_dim are consistent
embed_dim = 512  # Adjust this to match the expected dimension
num_heads = 8
ff_dim = 512  # Adjust this to match the expected dimension
dropout_rate = 0.1
num_transformer_blocks = 4

input_seq = layers.Input(shape=(sequence_length,), dtype=tf.int32)
instrument_seq = layers.Input(shape=(sequence_length,), dtype=tf.int32)

# Token embedding.
embedding_layer = layers.Embedding(input_dim=n_vocab, output_dim=embed_dim)
instrument_embedding_layer = layers.Embedding(
    input_dim=n_instruments, output_dim=embed_dim)

x = embedding_layer(input_seq)
instrument_x = instrument_embedding_layer(instrument_seq)

# Combine note and instrument embeddings
x = layers.Add()([x, instrument_x])

# Add positional encoding.
pos_encoding = get_positional_encoding(sequence_length, embed_dim)
x = x + pos_encoding

# Create a causal mask for the self-attention layers.
causal_mask = tf.linalg.band_part(
    tf.ones((sequence_length, sequence_length)), -1, 0)
causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]

# Add Transformer blocks with dense layers in between.
for _ in range(num_transformer_blocks):
    transformer_block = TransformerBlock(
        embed_dim, num_heads, ff_dim, dropout_rate)
    x = transformer_block(x, training=True, mask=causal_mask)
    x = layers.Dense(ff_dim)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)

# Global average pooling and final dense layers.
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(512)(x)  # Ensure this matches the embed_dim
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(dropout_rate)(x)
output = layers.Dense(n_vocab, activation="softmax")(x)

model = models.Model(inputs=[input_seq, instrument_seq], outputs=output)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss="sparse_categorical_crossentropy")
model.summary()

# ---------------------------
# Train the Model
# ---------------------------

# Define the ModelCheckpoint callback
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath='saved_model/best_model.keras',
    monitor='loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train the model with the checkpoint callback
model.fit([network_input, instrument_input], network_output,
          epochs=100, batch_size=64, callbacks=[checkpoint_callback])

print("Training complete. Best model saved to 'saved_model/best_model.h5'")
