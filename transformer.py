from re import X
import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
import tensorflow as tf
import tensorflow_datasets as tfds
import preprocessing
import numpy as np
from pydub import AudioSegment
import scipy.io.wavfile as wav
#import model
from tensorflow.keras.callbacks import ProgbarLogger
import tensorflow_hub as hub
# import tfm.nlp.layers

# class PositionalEmbedding(tf.keras.layers.Layer):
#   def __init__(self, vocab_size, d_model):
#     super().__init__()
#     self.d_model = d_model
#     self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
#     self.pos_encoding = positional_encoding(length=2048, depth=d_model)

#   def compute_mask(self, *args, **kwargs):
#     return self.embedding.compute_mask(*args, **kwargs)

#   def call(self, x):
#     length = tf.shape(x)[1]
#     x = self.embedding(x)
#     # This factor sets the relative scale of the embedding and positonal_encoding.
#     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
#     x = x + self.pos_encoding[tf.newaxis, :length, :]
#     return x

# ENCODING SIDE - CHINESE - SHAPE SHOULD BE SAME AS INPUT
# 1. Positional Encoding for Chinese 
# 2. Call Multi-Head Attention directly on this
# 3. Add & Normalization to avoid vanishing gradient using Add() and LayerNormalization()
# 4. Feed Forward
# 5. Add & Normalization using Add() and LayerNormalization()
  
class EncoderLayer(tf.keras.layers.Layer):
#   def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
  def __init__(self): #will probably have to add batch_size param to take a bunch of Chinese examples to train instead of just one
    super().__init__()

    # self.self_attention = GlobalSelfAttention(
    #     num_heads=num_heads,
    #     key_dim=d_model,
    #     dropout=dropout_rate)

    # what to figure out:
    # 1) input size & output size is same - We believe this should be axis=1 ("second dim") of Chinese. This is one single sample, and the one and only dimension represents the number of timesteps
    # 2) What is key_dim
    

    # self.pos_encoding = tf.keras.layers.Embedding(input_dim=9322, output_dim=9322)
    # self.pos_encoding = tf.keras.layers.Embedding(input_dim=100, output_dim=1, input_length=9322, mask_zero=True)

    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=64) # We are kind of guessing with this 64, can do 512 for more complex

    self.add1 = tf.keras.layers.Add()
    self.norm1 = tf.keras.layers.LayerNormalization()

    self.flatten = tf.keras.layers.Flatten(input_shape=(9322, 1))

    self.dense1 = tf.keras.layers.Dense(units=9322, activation='relu')
    # might need more dense and possibly a dropout- just make a sequential

    self.reshape = tf.keras.layers.Reshape((9322, 1))

    self.add2 = tf.keras.layers.Add()
    self.norm2 = tf.keras.layers.LayerNormalization()


  def call(self, x):
    # encoded = self.pos_encoding(x)
    attentioned = self.mha(x, x)
    added1 = self.add1([attentioned, x])
    normed1 = self.norm1(added1)
    print("normed1", normed1.shape)
    flattened = self.flatten(normed1)
    print("flattened", flattened.shape)
    densed = self.dense1(flattened)
    print("densed", densed.shape)
    reshaped = self.reshape(densed)
    print("reshaped", reshaped.shape)
    added2 = self.add1([reshaped, x])
    print(added2.shape)
    normed2 = self.norm1(added2)
    print(normed2.shape)
    return normed2

if __name__ == "__main__":
  encoder_layer = EncoderLayer()
  random_array = np.random.rand(1, 9322, 1) #(batch_size=how many samples we're dealing with at once, sequence_length=9322 in this example, hidden_size=(for RGB for example, it would be 3))- MultiHeadAttention expects this
  print(encoder_layer(random_array))

  
# class Encoder(tf.keras.layers.Layer):
#   def __init__(self, *, num_layers, d_model, num_heads,
#                dff, vocab_size, dropout_rate=0.1):
#     super().__init__()

#     self.d_model = d_model
#     self.num_layers = num_layers

#     # self.pos_embedding = tfm.nlp.layers.PositionEmbedding(max_length=1864516)
#     # inputs = tf.keras.Input((1864516, 32), dtype=tf.float32)
#     # outputs = self.pos_embedding(inputs)

#     self.enc_layers = [
#         EncoderLayer(d_model=d_model,
#                      num_heads=num_heads,
#                      dff=dff,
#                      dropout_rate=dropout_rate)
#         for _ in range(num_layers)]
#     self.dropout = tf.keras.layers.Dropout(dropout_rate)

#   def call(self, x):
#     # `x` is token-IDs shape: (batch, seq_len)
#     x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

#     # Add dropout.
#     x = self.dropout(x)

#     for i in range(self.num_layers):
#       x = self.enc_layers[i](x)

#     return x  # Shape `(batch_size, seq_len, d_model)`.

