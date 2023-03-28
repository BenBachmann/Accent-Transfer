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
import os
from nturl2path import url2pathname
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment
import numpy as np
import pydub.exceptions
import pickle
# import tfm.nlp.layers

# SHAPES
# Inputs to encoder (therefore also to the whole model): (batch_size, source_seq_len, 1)
    # source_seq_len is the length (no. timesteps) of a single WAV file of the source language
    # 1 is no. values at one timestep in the WAV file. (If dealing with RGB values, would be 3)
# Output of encoder: same shape as input to encoder
# Input to decoder: (batch_size, target_seq_len, 1)
    # target_seq_len, here, is NOT necessarily the same as enc_seq_len, but the no. timesteps of a single WAV file of the target language
# Output of decoder: (batch_size, target_seq_len, 1)
# Output of the final model: depends on the output size of the dense layer. Most sensible will be to keep it the same as the decoder output

# model.fit takes in these parameters: (x, y, batch_size, epochs, validation_split,*kwargs)
  # shape of x: (num_samples, input_dim)
  # shape of y: (num_samples, output_dim)
  # batch_size MUST BE DIVISIBLE by the total number of batches passed into fit

# Hyperparameters
#BATCH_SIZE = 6

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
  def __init__(self, sequence_length): #will probably have to add batch_size param to take a bunch of Chinese examples to train instead of just one
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

    self.flatten = tf.keras.layers.Flatten(input_shape=(2, sequence_length, 1))

    self.dense1 = tf.keras.layers.Dense(units=sequence_length, activation='relu')
    # might need more dense and possibly a dropout- just make a sequential

    self.reshape = tf.keras.layers.Reshape((sequence_length, 1))

    self.add2 = tf.keras.layers.Add()
    self.norm2 = tf.keras.layers.LayerNormalization(axis=1)


  def call(self, x):
    # encoded = self.pos_encoding(x) # *** NOTE: We left out the positional encoding *** might need to add ***
    print("x shape on entering model: ", x.shape)

    attentioned = self.mha(x, x) # takes in (batch_size, sequence_length, embedding_dim)
    added1 = self.add1([attentioned, x])
    normed1 = self.norm1(added1)
    flattened = self.flatten(normed1)
    densed = self.dense1(flattened)
    reshaped = self.reshape(densed)
    added2 = self.add2([reshaped, x])
    normed2 = self.norm2(added2)
    return normed2


class DecoderLayer(tf.keras.layers.Layer):
#   def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
  def __init__(self, sequence_length):
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

    self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=64)

    self.flatten = tf.keras.layers.Flatten(input_shape=(2, sequence_length, 1)) # len=11114 is # of timesteps in one sample of Italian

    self.dense1 = tf.keras.layers.Dense(units=sequence_length, activation='relu')
    # might need more dense and possibly a dropout- just make a sequential

    self.reshape = tf.keras.layers.Reshape((sequence_length, 1))

    self.add2 = tf.keras.layers.Add()
    self.norm2 = tf.keras.layers.LayerNormalization(axis=1)


  def call(self, x, context): # context is one Chinese sample with context taken into account retrieved from encoder
    # encoded = self.pos_encoding(x)
    print(x.shape)
    print(context.shape)
    masked = self.mask(x)
    masked_attentioned = self.mha(masked, masked)
    added1 = self.add1([masked_attentioned, masked])
    normed1 = self.norm1(added1)
    cross_attentioned = self.cross_attention(normed1, context)
    flattened1= self.flatten(cross_attentioned)
    densed1 = self.dense1(flattened1)
    reshaped = self.reshape(densed1)
    print("reshaped", reshaped.shape)
    print("masked", masked.shape)
    added2 = self.add2([reshaped, masked]) # Not sure if it is masked or x here
    normed2 = self.norm2(added2)

    return normed2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers):
    super().__init__()

    self.num_layers = num_layers

    self.enc_layers = [
        EncoderLayer(sequence_length=9322) # this is HARD CODED- CHANGE THIS
        for _ in range(self.num_layers)]

  def call(self, x):
    # `x` is shape: (batch_size, seq_len, 1)
    # seq_len is the length (no. timesteps) of a single WAV file
    # 1 is no. values at one timestep in the WAV file. (If dealing with RGB values, would be 3)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers):
    super().__init__()

    self.num_layers = num_layers

    self.dec_layers = [
        DecoderLayer(sequence_length=11114) # CHANGE THIS HARD CODED
        for _ in range(self.num_layers)]

  def call(self, x, context):
    # `x` is shape: (batch, seq_len, 1)

    for i in range(self.num_layers):
      x = self.dec_layers[i](x, context)

    return x
  

# NEXT STEPS:
# 1) Create a class which stacks up encoder and decoder blocks into a full transformer - check
# 2) Adjust the model so that you can specify batch size and num_layers, and adjust the layer shapes accordingly (create Encoder and Decoder classes) - check
# 3) Train a simple model
# 4) Improve the model- add in positional encoding


class Transformer(tf.keras.Model):
  def __init__(self, num_layers):
    super().__init__()

    self.num_layers = num_layers

    self.encoder = Encoder(num_layers=self.num_layers)

    self.decoder = Decoder(num_layers=self.num_layers)

    self.final_flatten = tf.keras.layers.Flatten(input_shape=(2, 11114, 1))
    self.final_dense = tf.keras.layers.Dense(units=11114, activation='softmax') # (units = sequence_length * batch_size)
    self.final_reshape = tf.keras.layers.Reshape((11114, 1))

    

  def call(self, inputs):

    print("INP", inputs)
    # !!!TO DO!!! We need to pass in inputs (Chinese) and tar (Italian). Then get context by calling self.encoder(inputs) and then decoded info by calling self.decoder(tar, context).
    # Overall, need to change logic here because right now it is just using Chinese when training and basically learning nothing.
    # How do we generate from our weights (add flag to model?)? 

    context = self.encoder(inputs)  # (batch_size, context_len, d_model)

    decoded = self.decoder(inputs, context)  # (batch_size, target_len, d_model)

    final_flattened = self.final_flatten(decoded)
    final_densed = self.final_dense(final_flattened)
    final_reshaped = self.final_reshape(final_densed)
    
    # Return the final output and the attention weights?
    return final_reshaped



# HYPERPARAMETERS
NUM_LAYERS = 3
NUM_EPOCHS = 2
BATCH_SIZE = 1 # note: may need to change this to be a divisor of num_samples
# LEARNING_RATE = 0.01 add this and more necessary hyperparameters

def pipeline(source_language, target_language, num_samples, divisor, TRAIN, wav_outfile, weights_file='transformer_weights_v1.h5'):

  # generates an array of batch_size samples
  # Trains if TRAIN
  # calls model.predict + generates WAV

  # NOTES
  # Function only makes 1 prediction. Therefore, if TRAIN, model trains num_samples - 1
  # Need to be careful with source & target language- this depends on what the folders are named.
  # num_samples must be less than or equal to the number of files in the folder
  # the default is currently a random file. This is very messy, and it could help to clean it up.
  source_data, max_source_len = preprocessing.mp3_to_numpy(source_language, divisor, num_samples)
  target_data, max_target_len = preprocessing.mp3_to_numpy(target_language, divisor, num_samples)
  print("MAXSOURCELEN", max_source_len)
  print("MAXTARLEN", max_target_len)
  X = tf.keras.preprocessing.sequence.pad_sequences(source_data, maxlen=max_source_len, padding='post', value=0, dtype='float32')
  Y = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=max_target_len, padding='post', value=0, dtype='float32')
  X = np.expand_dims(X, axis=-1)
  Y = np.expand_dims(Y, axis=-1)
  X_TRAIN, X_TEST = X[:-1], X[-1]
  Y_TRAIN, Y_TEST = Y[:-1], Y[-1]
  X_TEST = np.expand_dims(X_TEST, axis=0)
  print("X_TRAIN", X_TRAIN)
  print("XTRAINSHAPE", X_TRAIN.shape)
  print("X_TEST", X_TEST)
  print("XTESTSHAPE", X_TEST.shape)
  print("Y_TRAIN", Y_TRAIN)
  print("YSHAPE", Y_TRAIN.shape)

  print("A")
  # this is the training loop.
  if TRAIN:
    model = Transformer(num_layers=NUM_LAYERS)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X_TRAIN, Y_TRAIN, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
    model.save_weights('./transformer_weights_v2.h5')

  print("B")
  # sets the weights file. If Train, the file used in save_weights is used. Otherwise, the user should specify a weights file or rely on the default.
  if TRAIN:
    weights_file = './transformer_weights_v2.h5'
  else:
    weights_file = weights_file
    print(weights_file)

  print("C")
  test_model = Transformer(num_layers=NUM_LAYERS)
  print("C1")
  # test_model.build(input_shape=(2, max_source_len, 1)) # unsure what first dimension should be
  # print("max len", max_source_len)
  # print("C2")
  # test_model.load_weights(weights_file)
  # weights = test_model.get_weights()
  # print("C3")
  # test_model.get_weights()
  # # X_TEST = np.expand_dims(X_TEST, axis=-1)
  # # X_TEST = np.expand_dims(X_TEST, axis=0)

  # print("D")

  output_data = test_model(X_TEST, Y_TEST)
  # dummy_input = tf.zeros((1, max_source_len, 1))
  # _ = test_model(dummy_input)

  output_data = test_model.predict(X_TEST, batch_size=1)
  output_data = np.squeeze(output_data)

  print("E")

  rate = 44100
  scaled = np.int16(output_data / np.max(np.abs(output_data)) * 32767)
  wav.write(wav_outfile, rate, scaled)
  
  

if __name__ == "__main__":
  # X = preprocessing.unpickle_file("./chinese.pickle")
  # Y = preprocessing.unpickle_file("./italian.pickle")
  # X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=11114, padding='post', value=0, dtype='float32')
  # X = np.expand_dims(X, axis=-1)
  # Y = np.expand_dims(Y, axis=-1)
  # # print("X", X)

  # model = Transformer(num_layers=3)
  # model.compile(optimizer='adam', loss='categorical_crossentropy')

  # # model.fit takes in these parameters: (x, y, batch_size, epochs, validation_split,*kwargs)
  # # shape of x: (num_samples, source_sequence_length) = (34, 9322)
  # ### input_dim is the DIMENSIONALITY of the data, NOT the number of timesteps in the WAV file.
  # # shape of y: (num_samples, target_sequence_length) = (34, 11114)
  
  # # model(X, Y)

  # model.fit(X, Y, batch_size=2, epochs=2)
  # model.save_weights('./transformer_weights_v1.h5')

  pipeline("chinese", "italian", 16, 200, False, 'transformer_outputs_1.wav')



