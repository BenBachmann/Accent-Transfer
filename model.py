# import numpy as np
# import tensorflow as tf
# import tensorflow_datasets as tfds

# # Load the dataset
# train_data = tfds.load("accent_dataset", split="train[:80%]")
# val_data = tfds.load("accent_dataset", split="train[80%:90%]")
# test_data = tfds.load("accent_dataset", split="train[90%:100%]")

# # Load pre-trained model and tokenizer
# transformer_model = tf.keras.models.load_model("transformer_model")
# tokenizer = tfds.features.text.Tokenizer()

# # Fine-tune the model
# transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
# transformer_model.fit(train_data, epochs=5)

# Define the transformer model architecture
# transformer = tf.keras.models.Transformer(
#     num_layers=6,
#     d_model=256,
#     num_heads=8,
#     dff=1024
# )

# # Define the optimizer and loss function
# optimizer = tf.keras.optimizers.Adam()
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# # Define the training loop
# @tf.function
# def train_step(inputs, targets):
#     with tf.GradientTape() as tape:
#         logits = transformer(inputs, training=True)
#         loss_value = loss_object(targets, logits)

#     grads = tape.gradient(loss_value, transformer.trainable_variables)
#     optimizer.apply_gradients(zip(grads, transformer.trainable_variables))

#     return loss_value

# # Train the model on your dataset
# for epoch in range(num_epochs):
#     for inputs, targets in dataset:
#         loss_value = train_step(inputs, targets)

# # Save the trained model
# transformer.save("accent_style_transfer.h5")

# # Load the trained model
# transformer = tf.keras.models.load_model("accent_style_transfer.h5")

# # Input audio sample
# input_audio = your_input_audio_sample

# # Generate output with target accent
# output_audio = transformer(input_audio)

from re import X
import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')

import tensorflow as tf
import tensorflow_datasets as tfds
import preprocessing

# def transformer():
#     # Load the dataset
#     train_data = tfds.load("accentdb", split="train[:1%]")
#     val_data = tfds.load("accentdb", split="train[1%:1.1%]")
#     test_data = tfds.load("accentdb", split="train[1.1%:1.2%]")

#     # Load pre-trained model and tokenizer
#     transformer_model = tf.keras.models.load_model("transformer_model")
#     tokenizer = tfds.features.text.Tokenizer()

#     # Fine-tune the model
#     transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
#     transformer_model.fit(train_data, epochs=5)

#     input_audio = preprocessing.mp3_to_numpy()[0]

#     output_audio = transformer_model(input_audio)
#     return output_audio

X_train = preprocessing.mp3_to_numpy("chinese")[:-2]
#print("XTRAINSHAPE", X_train.shape)
y_train = preprocessing.mp3_to_numpy("italian")[:-1]
X_test = preprocessing.mp3_to_numpy("chinese")[-1]
print("XTEST", X_test.shape)
X_test = X_test.reshape(1, X_test.shape[0])


#reshape
# X_train = X_train.reshape(1, X_train.shape[1])
# y_train = y_train.reshape(1, y_train.shape[1])
# X_test = X_test.reshape(1, X_test.shape[1])
# X_train = X_train.flatten()
# y_train = y_train.flatten()
# X_test = X_test.flatten()

# for GRU:
X_train = tf.expand_dims(X_train, 2)
y_train = tf.expand_dims(y_train, 2)
X_test = tf.expand_dims(X_test, 2)

# Define the RNN model
model = tf.keras.Sequential()
# model.add(tf.keras.layers.GRU(8, input_shape=(None, X_train.shape[1]), return_sequences=True))
model.add(tf.keras.layers.GRU(y_train.shape[1], return_sequences=True))
model.add(tf.keras.layers.Dense(y_train.shape[1], activation="leaky_relu", use_bias=True))
model.add(tf.keras.layers.Dropout(0.2))


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, verbose=2)

# Use the model to generate new images
generated_image = model.predict(X_test)

import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
import numpy as np
from pydub import AudioSegment
import preprocessing
import scipy.io.wavfile as wav
#import model

data = generated_image[0]

rate = 44100
scaled = np.int16(data / np.max(np.abs(data)) * 32767)
wav.write('test.wav', rate, scaled)