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
    self.norm1 = tf.keras.layers.LayerNormalization(axis=1)

    self.flatten = tf.keras.layers.Flatten(input_shape=(9322, 1))

    self.dense1 = tf.keras.layers.Dense(units=9322, activation='relu')
    # might need more dense and possibly a dropout- just make a sequential

    self.reshape = tf.keras.layers.Reshape((9322, 1))

    self.add2 = tf.keras.layers.Add()
    self.norm2 = tf.keras.layers.LayerNormalization(axis=1)


  def call(self, x):
    # encoded = self.pos_encoding(x) # *** NOTE: We left out the positional encoding *** might need to add ***
    attentioned = self.mha(x, x)
    added1 = self.add1([attentioned, x])
    normed1 = self.norm1(added1)
    flattened = self.flatten(normed1)
    densed = self.dense1(flattened)
    reshaped = self.reshape(densed)
    added2 = self.add2([reshaped, x])
    normed2 = self.norm2(added2)
    # print(normed2)
    # print(normed2.shape)
    # print("N2segment", normed2[0][2000:2100])
    # print("N2full", normed2)
    return normed2


class DecoderLayer(tf.keras.layers.Layer):
#   def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
  def __init__(self):
    super().__init__()

    # DECODING SIDE - ITALIAN
    # 1. Positional encoding of Italian shifted right one step
    # 2. Masked Multi-Head Attention
    # 3. Add() and LayerNormalization()
    # 4. Cross Attention
    # 5. Add() and LayerNormalization()
    # 6. Feed Forward Layer
    # 7. Add() and LayerNormalization()
    # 8. Linear
    # 9. Softmax

    # *** NOTE: We are not going to shift the input for now since the time steps are very small ***
    # NOTE: We also left out the positional encoding here
    self.mask = tf.keras.layers.Masking(mask_value=0.0)
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=64)

    self.add1 = tf.keras.layers.Add()
    self.norm1 = tf.keras.layers.LayerNormalization(axis=1)

    #Here we are calling the encoder to get its outputs for the cross attention:
    self.encoder = EncoderLayer()

    self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=64)

    self.flatten = tf.keras.layers.Flatten(input_shape=(11114, 1)) # len=11114 is # of timesteps in one sample of Italian

    self.dense1 = tf.keras.layers.Dense(units=11114, activation='relu')
    # might need more dense and possibly a dropout- just make a sequential

    self.reshape = tf.keras.layers.Reshape((11114, 1))

    self.add2 = tf.keras.layers.Add()
    self.norm2 = tf.keras.layers.LayerNormalization(axis=1)
    
    self.flatten2 = tf.keras.layers.Flatten(input_shape=(11114, 1))
    self.dense2 = tf.keras.layers.Dense(units=11114, activation='softmax')
    self.reshape2 = tf.keras.layers.Reshape((11114, 1))


  def call(self, x, context): #need context here from encoder - more specifically, context is one Chinese sample with context taken into account retrieved from encoder
    # encoded = self.pos_encoding(x)
    masked = self.mask(x)
    masked_attentioned = self.mha(masked, masked)
    added1 = self.add1([masked_attentioned, masked])
    normed1 = self.norm1(added1)
    cross_attentioned = self.cross_attention(normed1, context)
    # print("normed1", normed1.shape)
    flattened1= self.flatten(cross_attentioned)
    # print("flattened", flattened.shape)
    densed1 = self.dense1(flattened1)
    # print("densed", densed.shape)
    reshaped = self.reshape(densed1)
    # print("reshaped", reshaped.shape)
    added2 = self.add2([reshaped, masked]) # Not sure if it is masked or x here
    # print(added2.shape)
    normed2 = self.norm2(added2)
    # print(normed2.shape)
    flattened2 = self.flatten2(normed2)
    densed2 = self.dense2(flattened2)
    reshaped2 = self.reshape2(densed2)
    # print(reshaped2[2200:2300])

    return reshaped2
  

# NEXT STEPS:
# 1) Create a class which stacks up encoder and decoder blocks into a full transformer
# 2) Adjust the model so that you can specify batch size, and adjust the layer shapes accordingly
# 3) Train a simple model
# 4) Improve the model



if __name__ == "__main__":
  encoder_layer = EncoderLayer()
  random_chinese = np.random.rand(1, 9322, 1) #(batch_size=how many samples we're dealing with at once, sequence_length=9322 in this example, hidden_size=(for RGB for example, it would be 3))- MultiHeadAttention expects this
  # print("RANDCHINSEG", random_chinese[0][2000:2100])
  # print("RANDCHINFULLL", random_chinese)
  context = encoder_layer(random_chinese)
  # print(encoder_layer(random_chinese))

  decoder_layer = DecoderLayer()
  random_italian = np.random.rand(1, 11114, 1)
  print(decoder_layer(random_italian, context))

  
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

