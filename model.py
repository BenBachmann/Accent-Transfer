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
import numpy as np
from pydub import AudioSegment
import scipy.io.wavfile as wav
#import model
from tensorflow.keras.callbacks import ProgbarLogger
import tensorflow_hub as hub



def transformer_train_generate():
    # Load the dataset
    # train_data = tfds.load("accentdb", split="train[:1%]")
    # val_data = tfds.load("accentdb", split="train[1%:1.1%]")
    # test_data = tfds.load("accentdb", split="train[1.1%:1.2%]")

    X_train = preprocessing.mp3_to_numpy("chinese")[:-1]
    y_train = preprocessing.mp3_to_numpy("italian")[:-1]
    X_test = preprocessing.mp3_to_numpy("chinese")[-1]

    # Load pre-trained model and tokenizer
    # transformer_model = tf.keras.models.load_model("transformer_model")

    # Fine-tune the model
    # transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # transformer_model.fit(train_data, epochs=5)

    # Load the pre-trained model
    model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                   input_shape=(224,224,3))])

    # Compile the model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()

    print(X_test.shape)
    output_audio = model.predict(X_test)
    print(output_audio)

    # generated_audio = model.predict(X_test)
    # print(generated_audio)

    # data = generated_audio[0]

    rate = 44100
    scaled = np.int16(output_audio / np.max(np.abs(output_audio)) * 32767)
    print(scaled)
    wav.write('test5.wav', rate, scaled)

    # return output_audio

def base_train_generate():

    X_train = preprocessing.mp3_to_numpy("chinese")[:-1]
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
    # X_train = tf.expand_dims(X_train, 2)
    # y_train = tf.expand_dims(y_train, 2)
    # X_test = tf.expand_dims(X_test, 2)

    # for CNN:
    # X_train = tf.expand_dims(X_train, 2)
    # y_train = tf.expand_dims(y_train, 2)
    # X_test = tf.expand_dims(X_test, 2)
    # X_train = tf.expand_dims(X_train, 2)
    # y_train = tf.expand_dims(y_train, 2)
    # X_test = tf.expand_dims(X_test, 2)


    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.GRU(8, input_shape=(None, X_train.shape[1]), return_sequences=True))
    #model.add(tf.keras.layers.GRU(y_train.shape[1], return_sequences=True))
    #model.add(tf.keras.layers.Conv2D(5, 1000))
    model.add(tf.keras.layers.Dense(1000, use_bias=True)) ##**comment in for Ben's base model
    #model.add(tf.keras.layers.Dense(100, use_bias=True))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation="leaky_relu", use_bias=True)) ##**comment in for Ben's base model
    model.add(tf.keras.layers.Dropout(0.3)) ##**comment in for Ben's base model

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    model.fit(X_train, y_train, epochs=3, verbose=2)
    model.save('base_weights.h5')

    generated_audio = model.predict(X_test)

    data = generated_audio[0]

    rate = 44100
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    wav.write('test3.wav', rate, scaled)


def LSTM_train_generate():

    # Input dimensions: (batch_size, time_steps, num_features) --> (4, 1000, X_train.shape[1])
    # Output dimensions: (num_features,)

    X_train = preprocessing.mp3_to_numpy("chinese")[:-1]
    X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    #print("XTRAINSHAPE", X_train.shape)
    y_train = preprocessing.mp3_to_numpy("italian")[:-1]
    y_train_reshaped = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
    # y_train_reshaped = y_train.reshape(1000)
    X_test = preprocessing.mp3_to_numpy("chinese")[-1]
    X_test_reshaped = X_test.reshape(1, X_test.shape[0])
    # print("XTEST", X_test.shape)
    # X_test = X_test.reshape(1, X_test.shape[0])

    model = tf.keras.Sequential()
    # RNN using LSTM:
    model.add(tf.keras.layers.LSTM(units=128, input_shape=(1, X_train.shape[1])))
    model.add(tf.keras.layers.Dense(1, activation='linear')) #also try with sigmoid activation here
    # model.add(tf.keras.layers.Dropout(0.4))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.summary()

    # Reshape the audio data for LSTM input
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    print(X_train.shape)
    print(X_train_reshaped.shape)
    print(y_train.shape)
    print(y_train_reshaped.shape)
    

    # Train the model - should it be y_train as second parameter?
    model.fit(X_train_reshaped, y_train_reshaped, epochs=10, shuffle=False, verbose=1, callbacks=[ProgbarLogger(count_mode='steps')])

    # Save weights
    model.save('LSTM_weights.h5')

    print(X_test.shape)
    print(X_test_reshaped.shape)

    # Use the model to generate new audio
    generated_audio = model.predict(X_test_reshaped)
    print(generated_audio)

    data = generated_audio[0]
    print(data)

    rate = 44100
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    print(scaled)
    wav.write('test3.wav', rate, scaled)

if __name__ == "__main__":
    # base_train_generate()
    LSTM_train_generate()
    # transformer_train_generate()