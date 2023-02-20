import numpy as np
import tensorflow as tf
from tensorflow import keras
import preprocessing
import numpy as np
from pydub import AudioSegment
import scipy.io.wavfile as wav
from tensorflow.keras.callbacks import ProgbarLogger

# Define the encoder layer
def encoder(input_shape):
  inputs = keras.Input(shape=input_shape)
  x = keras.layers.Dense(128, activation='relu')(inputs)
  x = keras.layers.Dense(64, activation='relu')(x)
  return keras.Model(inputs, x)

# Define the decoder layer
def decoder(latent_dim):
  inputs = keras.Input(shape=(latent_dim,))
  x = keras.layers.Dense(128, activation='relu')(inputs)
  x = keras.layers.Dense(input_shape[1], activation='sigmoid')(x)
  return keras.Model(inputs, x)

# Load the audio data
X_train = preprocessing.mp3_to_numpy("chinese")[:-1]
#print("XTRAINSHAPE", X_train.shape)
y_train = preprocessing.mp3_to_numpy("italian")[:-1]
X_test = preprocessing.mp3_to_numpy("chinese")[-1]

# input shape is num_samples x num_features
input_shape = (X_train.shape[0], X_train.shape[1])
# audio_data = np.load('audio.npy') #add a numpy array file here with chinese numpy array or should this just be X_train?
audio_data = X_train

# Split the data into train and test sets
train_data = audio_data[:int(0.8 * len(audio_data))]
test_data = audio_data[int(0.8 * len(audio_data)):]

# Instantiate the encoder and decoder models
encoder = encoder(input_shape)
decoder = decoder(64)

# Combine the encoder and decoder into a single autoencoder model
inputs = keras.Input(shape=input_shape)
latent = encoder(inputs)
outputs = decoder(latent)
autoencoder = keras.Model(inputs, outputs)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder on the train data
autoencoder.fit(train_data, train_data, epochs=100, batch_size=32, validation_data=(test_data, test_data), callbacks=[ProgbarLogger(count_mode='steps')])

# Use the autoencoder to generate new audio samples
latent_samples = np.random.randn(64, 64)
generated_samples = decoder.predict(latent_samples)

# Use the model to generate new audio

data = generated_samples[0]

rate = 44100
scaled = np.int16(data / np.max(np.abs(data)) * 32767)
wav.write('test4.wav', rate, scaled)